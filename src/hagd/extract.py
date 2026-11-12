"""End-to-end circuit extraction for one (model, task): the HAGD method and its ablations.

    attribution graph -> hierarchical decomposition -> GNN-guided traversal
                      -> causal validation (size selection, necessity pruning, edge verification)

``finalize_circuit`` produces the result record shared by HAGD and every baseline, so all
methods are scored by exactly the same held-out protocol.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .attribution import AttributionGraph, attribute_edges, sparsify
from .data.tasks import Task, TaskBatch
from .gnn import CircuitGNN, heuristic_scores, make_sample
from .hierarchy import build_hierarchy
from .metrics import concept_purity, description_length, modularity, sparsity_metrics
from .replacement import FeatureModel, FeatureStats, nodes_to_dict
from .search import FlatRanking, Traversal, select_by_score, select_size
from .utils import peak_memory_gb
from .validation import necessity_prune, sufficiency_report

VARIANTS = ("full", "no_hierarchy", "no_gnn", "no_cross_layer", "no_causal_validation")


@dataclass
class TaskData:
    task: Task
    discover: list[TaskBatch]
    val: list[TaskBatch]
    test: list[TaskBatch]
    full_val: dict = field(default_factory=dict)  # full-model reference runs (filled lazily)
    full_test: dict = field(default_factory=dict)

    def ensure_refs(self, fm: FeatureModel) -> None:
        if not self.full_val:
            self.full_val = fm.reference(self.val)
        if not self.full_test:
            self.full_test = fm.reference(self.test)


def make_evaluator(fm: FeatureModel, stats: FeatureStats, graph: AttributionGraph, batches: list[TaskBatch], vcfg):
    """evaluate(vertices) -> {acc, loss} of the circuit-restricted model M(x; C) on ``batches``."""
    cache: dict[tuple, dict] = {}

    def evaluate(nodes: np.ndarray) -> dict:
        key = tuple(sorted(int(v) for v in nodes))
        if key not in cache:
            iv = fm.intervention(stats, nodes_to_dict(graph.node_layer, graph.node_feat, np.asarray(nodes, dtype=np.int64)),
                                 mode="restrict", error=vcfg.error_mode, ablation=vcfg.ablation)
            r = fm.evaluate(batches, iv)
            cache[key] = {"acc": r["acc"], "loss": r["loss"]}
            evaluate.calls += 1
        return cache[key]

    evaluate.calls = 0
    return evaluate


def verify_edges(fm: FeatureModel, graph: AttributionGraph, nodes: np.ndarray, batches: list[TaskBatch], acfg) -> dict:
    """Recompute Eq. 5 between the circuit's own features on validation data ("edge verification")."""
    nodes = np.asarray(nodes, dtype=np.int64)
    dev = fm.lm.device
    per_layer = [nodes[graph.node_layer[nodes] == li] for li in range(fm.n_layers)]
    cands = [torch.as_tensor(graph.node_feat[ids], dtype=torch.long, device=dev) for ids in per_layer]
    w_abs, w_signed, _, info = attribute_edges(fm, batches, cands, int(acfg.jac_chunk), bool(acfg.batched_grads))
    threshold = float(graph.info.get("edge_threshold", 0.0))
    src, dst, wt = [], [], []
    for li, wa in enumerate(w_abs):
        keep = sparsify(wa, int(acfg.edge_topk), threshold)
        i, j = keep.nonzero(as_tuple=True)
        src.append(per_layer[li][i.cpu().numpy()])
        dst.append(per_layer[li + 1][j.cpu().numpy()])
        wt.append(wa[i, j].cpu().numpy())
    cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dtype=dt)  # noqa: E731
    return {"src": cat(src, np.int64), "dst": cat(dst, np.int64), "w": cat(wt, np.float32),
            "pairs_checked": int(info["pairs_evaluated"]), "seconds": float(info["edge_seconds"])}


def induced_edges(graph: AttributionGraph, nodes: np.ndarray) -> dict:
    e = graph.induced_edges(nodes)
    return {"src": graph.edge_src[e], "dst": graph.edge_dst[e], "w": graph.edge_w[e], "pairs_checked": 0, "seconds": 0.0}


def finalize_circuit(fm: FeatureModel, graph: AttributionGraph, stats: FeatureStats, nodes: np.ndarray, edges: dict | None,
                     data: TaskData, cfg, method: str, seconds: dict, extra: dict | None = None, seed: int = 0) -> dict:
    """Score a circuit on the held-out test split and assemble its result record."""
    nodes = np.unique(np.asarray(nodes, dtype=np.int64))
    data.ensure_refs(fm)
    if edges is None:
        edges = induced_edges(graph, nodes)
    local = {int(v): i for i, v in enumerate(nodes)}
    e_src = np.array([local[int(s)] for s in edges["src"]], dtype=np.int64)
    e_dst = np.array([local[int(d)] for d in edges["dst"]], dtype=np.int64)
    n_edges = int(len(e_src))
    metrics = sparsity_metrics(graph, nodes, n_edges)
    metrics.update(sufficiency_report(fm, stats, graph, nodes, data.test, data.full_test, cfg.validation, seed))
    metrics.update(modularity(len(nodes), e_src, e_dst, edges["w"], seed))
    metrics["description_length_bits"] = description_length(graph, nodes, n_edges)
    metrics.update(concept_purity(fm, graph, nodes, data.test))
    total = float(sum(seconds.values()))
    return {
        "model": graph.model, "task": graph.task, "method": method,
        "metrics": metrics,
        "seconds": {**{k: float(v) for k, v in seconds.items()}, "total": total},
        "peak_memory_gb": peak_memory_gb(),
        "nodes": [{"layer_index": int(graph.node_layer[v]), "block": int(graph.layers[graph.node_layer[v]]),
                   "feature": int(graph.node_feat[v]), "vertex": int(v)} for v in nodes],
        "edges": [[int(s), int(d), float(w)] for s, d, w in zip(e_src, e_dst, edges["w"])],
        "chance": float(data.task.chance),
        **(extra or {}),
    }


def extract_circuit(fm: FeatureModel, graph: AttributionGraph, stats: FeatureStats, data: TaskData,
                    gnn: CircuitGNN | None, cfg, variant: str = "full", seed: int = 0) -> dict:
    """Run HAGD (or one of its Table V ablations) and return the scored circuit."""
    assert variant in VARIANTS
    data.ensure_refs(fm)
    scfg, hcfg, vcfg = cfg.search, cfg.hierarchy, cfg.validation
    t0 = time.time()
    flat = variant == "no_hierarchy"
    hier = build_hierarchy(graph.n, graph.edge_src, graph.edge_dst, graph.edge_w,
                           branching=max(graph.n, 2) if flat else int(hcfg.branching),
                           embed_dim=int(hcfg.embed_dim), max_cluster_factor=int(hcfg.max_cluster_factor),
                           seed=seed, order_hint=graph.node_layer)
    sample = make_sample(graph, hier)
    use_gnn = gnn is not None and variant != "no_gnn"
    scores = gnn.score(sample) if use_gnn else heuristic_scores(sample)
    t_hier = time.time() - t0

    evaluate = make_evaluator(fm, stats, graph, data.val, vcfg)
    acc_full = data.full_val["acc"]
    ranking = FlatRanking(scores[0]) if flat else Traversal(hier, scores, float(scfg.gamma))
    t1 = time.time()
    info: dict = {"variant": variant, "gnn_guided": use_gnn, "gnn": getattr(gnn, "name", None) if use_gnn else None,
                  "gnn_saw_target_model": bool(getattr(gnn, "saw_target_model", False)) if use_gnn else False,
                  "hierarchy_depth": hier.depth, "hierarchy_sizes": [lvl.n for lvl in hier.levels]}
    if variant == "no_causal_validation":
        nodes = select_by_score(scores[0], 0.5, int(scfg.max_nodes), int(scfg.min_nodes))
        edges = induced_edges(graph, nodes)
        info.update({"reached_theta": None, "trace": [], "pruned": 0})
        t_search, t_valid = time.time() - t1, 0.0
    else:
        nodes, trace, reached = select_size(ranking, evaluate, acc_full, float(scfg.theta), int(scfg.min_nodes),
                                            int(scfg.max_nodes), float(scfg.growth), graph.n)
        t_search = time.time() - t1
        t2 = time.time()
        nodes, prune = necessity_prune(nodes, evaluate, acc_full, float(vcfg.epsilon), float(scfg.theta), int(vcfg.rounds))
        edges = verify_edges(fm, graph, nodes, data.val, cfg.attribution)
        t_valid = time.time() - t2
        info.update({"reached_theta": bool(reached), "trace": trace, "pruned": int(prune["removed"]),
                     "necessity_deltas": prune["deltas"]})
    n_c = int(len(nodes))
    info["ops"] = {**ranking.ops, "model_evaluations": int(evaluate.calls), "edge_pairs_verified": int(edges["pairs_checked"]),
                   "n_vertices": int(graph.n), "n_circuit": n_c}
    seconds = {"attribution_nodes": graph.info.get("node_seconds", 0.0), "attribution_edges": graph.info.get("edge_seconds", 0.0),
               "hierarchy_gnn": t_hier, "search": t_search, "causal_validation": t_valid}
    return finalize_circuit(fm, graph, stats, nodes, edges, data, cfg, "hierarchical", seconds, info, seed)
