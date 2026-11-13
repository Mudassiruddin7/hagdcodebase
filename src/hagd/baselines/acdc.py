"""ACDC baseline [4]: iterative edge pruning by activation patching.

Starting from the complete graph over a pool of candidate features (every feature of
layer l connected to every feature of layer l+1, last-layer features connected to the
output), ACDC visits edges from the output backwards.  Each edge is removed by an actual
intervention and stays removed if the KL divergence to the full model's output grows by
less than a threshold tau.  One model run per edge makes the cost grow with the number of
edges, which is what limits ACDC at scale.

Exact edge interventions in a feature graph.  Removing u -> v must change only what v
receives from u.  The feature v is therefore recomputed from a counterfactual copy of the
upstream hidden state in which the features R_v (sources of v's removed edges) are
mean-ablated.  All counterfactual copies ride along as extra batch "replicas" in a single
forward pass: replica k of a layer carries the hidden state that feature k of the next
layer is allowed to see.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..data.tasks import TaskBatch, score_batch
from ..extract import finalize_circuit, make_criterion, make_evaluator
from ..utils import is_oom, log


class _EdgeState:
    def __init__(self, sizes: list[int]) -> None:
        # removed[l][v, u]: edge (u in layer l) -> (v in layer l+1) has been pruned
        self.removed = [torch.zeros(sizes[i + 1], sizes[i], dtype=torch.bool) for i in range(len(sizes) - 1)]
        self.out_removed = torch.zeros(sizes[-1], dtype=torch.bool)

    def alive(self) -> list[torch.Tensor]:
        """Features that still have a path to the output."""
        live = [None] * (len(self.removed) + 1)
        live[-1] = ~self.out_removed
        for li in range(len(self.removed) - 1, -1, -1):
            live[li] = ((~self.removed[li]) & live[li + 1].unsqueeze(1)).any(0)
        return live


class _ReplicaModel:
    def __init__(self, fm, stats, pools: list[torch.Tensor], error: str = "mean") -> None:
        self.fm, self.stats, self.error = fm, stats, error
        self.pools = [p.to(fm.lm.device) for p in pools]
        self.k_max = max(len(p) for p in pools)

    @torch.no_grad()
    def kl(self, state: _EdgeState, batches: list[TaskBatch], refs: list[torch.Tensor]) -> float:
        fm, tc, st = self.fm, self.fm.tc, self.stats
        dev = fm.lm.device
        n_layers = fm.n_layers
        reps = 1 + self.k_max
        total, count = 0.0, 0
        for b, ref in zip(batches, refs):
            b = b.to(dev)
            n, s = b.input_ids.shape
            tm = fm.sites(b).unsqueeze(-1)

            def make(li: int):
                def fn(h: torch.Tensor) -> torch.Tensor:
                    x = h.float().view(reps, n, s, -1)
                    x0 = x[0]
                    f0 = tc.encode(x0, li)
                    pool = self.pools[li]
                    k = len(pool)
                    base = st.mean[li]
                    if li == 0:
                        fv = f0[..., pool]
                    else:  # feature k is read from the replica that carries its counterfactual input
                        enc = tc.encode(x[1:1 + k].reshape(k * n, s, -1), li).view(k, n, s, -1)
                        fv = enc[torch.arange(k, device=dev), :, :, pool].permute(1, 2, 0)
                    f_new = base.expand_as(f0).clone()  # features outside the pool are ablated throughout
                    f_new[..., pool] = fv
                    if li == n_layers - 1:
                        gone = pool[state.out_removed.to(dev)]
                        f_new[..., gone] = base[gone]
                    if self.error == "clean":  # same treatment of the transcoder residual as M_C
                        x_main = x0 + tc.decode_delta(f_new - f0, li)
                    elif self.error == "mean":
                        x_main = tc.decode(f_new, li) + st.err_mean[li]
                    else:
                        x_main = tc.decode(f_new, li)
                    x0_new = torch.where(tm, x_main, x0)
                    if li == n_layers - 1:
                        return x0_new  # only the main replica continues to the unembedding
                    out = x0_new.unsqueeze(0).repeat(reps, 1, 1, 1)
                    k_next = len(self.pools[li + 1])
                    mask = state.removed[li].to(dev).float()  # (k_next, k)
                    delta = base[pool] - f_new[..., pool]  # effect of ablating each pool feature
                    per = ((mask[:, None, None, :] * delta.unsqueeze(0)) @ tc.W_dec[li][:, pool].t()) * tc.scale[li]
                    out[1:1 + k_next] += torch.where(tm, per, torch.zeros_like(per))
                    return out.view(reps * n, s, -1)

                return fn

            with fm.lm.adapter_on(fm.adapter_on):
                logits = fm.lm.run(b.input_ids.repeat(reps, 1), b.attention_mask.repeat(reps, 1),
                                   {layer: make(li) for li, layer in enumerate(fm.layers)})
            dist = score_batch(logits, b)["dist"]
            p = ref.to(dev)
            total += float((p.exp() * (p - dist)).sum(-1).clamp_min(0).sum())
            count += p.shape[0]
        return total / max(count, 1)


def _prune(model: _ReplicaModel, sizes: list[int], batches, refs, tau: float, deadline: float) -> tuple[_EdgeState, dict]:
    state = _EdgeState(sizes)
    kl = model.kl(state, batches, refs)
    kl_start = kl
    tested, removed, status = 0, 0, "ok"
    last = len(sizes) - 1

    def try_remove(set_removed, unset) -> None:
        nonlocal kl, tested, removed
        set_removed()
        new = model.kl(state, batches, refs)
        tested += 1
        if new - kl < tau:
            kl = new
            removed += 1
        else:
            unset()

    for li in range(last, -1, -1):
        if li == last:
            for v in range(sizes[li]):
                if time.time() > deadline:
                    status = "timeout"
                    break
                try_remove(lambda v=v: state.out_removed.__setitem__(v, True),
                           lambda v=v: state.out_removed.__setitem__(v, False))
        if li == 0 or status != "ok":
            continue
        live = state.alive()[li]
        for v in range(sizes[li]):
            if not bool(live[v]):  # no path to the output: its incoming edges are irrelevant
                state.removed[li - 1][v, :] = True
                continue
            for u in range(sizes[li - 1]):
                if time.time() > deadline:
                    status = "timeout"
                    break
                try_remove(lambda v=v, u=u: state.removed[li - 1].__setitem__((v, u), True),
                           lambda v=v, u=u: state.removed[li - 1].__setitem__((v, u), False))
            if status != "ok":
                break
    return state, {"edges_tested": tested, "edges_removed": removed, "kl_start": kl_start, "kl_final": kl, "status": status}


def run_acdc(fm, graph, stats, data, cfg, seed: int = 0) -> dict:
    data.ensure_refs(fm)
    a = cfg.baselines.acdc
    theta = float(cfg.search.theta)
    t0 = time.time()
    deadline = t0 + float(a.max_minutes) * 60
    per_layer = int(a.nodes_per_layer)
    # pool: strongest features of every layer (graph vertex ids, grouped by layer)
    pools_v = []
    for li in range(fm.n_layers):
        ids = np.nonzero(graph.node_layer == li)[0]
        ids = ids[np.argsort(-np.abs(graph.node_ie[ids]), kind="stable")][:per_layer]
        pools_v.append(ids)
    if any(len(p) == 0 for p in pools_v):
        raise RuntimeError("ACDC needs at least one candidate feature in every layer")
    sizes = [len(p) for p in pools_v]
    pools_f = [torch.as_tensor(graph.node_feat[p], dtype=torch.long) for p in pools_v]

    batches, n = [], 0
    for b in data.discover:
        batches.append(b)
        n += b.n_examples
        if n >= int(a.n_examples):
            break
    refs = fm.evaluate(batches, None, keep_dist=True)["dist"]
    model = _ReplicaModel(fm, stats, pools_f, cfg.validation.error_mode)
    evaluate = make_evaluator(fm, stats, graph, data.val, cfg.validation)
    crit = make_criterion(data, evaluate, cfg)
    weight = {(int(s), int(d)): float(w) for s, d, w in zip(graph.edge_src, graph.edge_dst, graph.edge_w)}

    best, runs, status = None, [], "ok"
    for tau in a.taus:
        try:
            state, info = _prune(model, sizes, batches, refs, float(tau), deadline)
        except RuntimeError as err:
            if not is_oom(err):
                raise
            status = "OOM"
            log.warning("ACDC %s/%s ran out of memory", graph.model, graph.task)
            break
        live = state.alive()
        nodes = np.concatenate([pools_v[li][live[li].numpy()] for li in range(fm.n_layers)]).astype(np.int64)
        src, dst, wt = [], [], []
        for li in range(fm.n_layers - 1):
            keep = (~state.removed[li]) & live[li + 1].unsqueeze(1) & live[li].unsqueeze(0)
            for v, u in keep.nonzero().tolist():
                s_id, d_id = int(pools_v[li][u]), int(pools_v[li + 1][v])
                src.append(s_id)
                dst.append(d_id)
                wt.append(weight.get((s_id, d_id), 0.0))
        phi = float(crit(evaluate(nodes))) if len(nodes) else 0.0
        run = {"tau": float(tau), "nodes": int(len(nodes)), "edges": len(src), "val_preservation": phi, **info}
        runs.append(run)
        cand = (phi >= theta, -len(nodes) if phi >= theta else phi)
        if best is None or cand > best[0]:
            best = (cand, nodes, {"src": np.asarray(src, dtype=np.int64), "dst": np.asarray(dst, dtype=np.int64),
                                  "w": np.asarray(wt, dtype=np.float32)})
        if info["status"] != "ok":
            status = info["status"]
            break

    seconds = {"attribution_nodes": graph.info.get("node_seconds", 0.0), "search": time.time() - t0}
    full_edges = sizes[-1] + sum(sizes[i] * sizes[i + 1] for i in range(len(sizes) - 1))
    extra = {"status": status, "runs": runs, "pool_per_layer": per_layer, "initial_edges": int(full_edges),
             "reached_theta": bool(best[0][0]) if best else False,
             "ops": {"model_evaluations": int(sum(r["edges_tested"] for r in runs)), "replicas": 1 + model.k_max}}
    if best is None:
        return {"model": graph.model, "task": graph.task, "method": "acdc", "metrics": {}, "status": status,
                "seconds": {**seconds, "total": float(sum(seconds.values()))}, **extra}
    log.info("ACDC %s/%s: %d -> %d edges, %d vertices (%s, %.0fs)", graph.model, graph.task, full_edges,
             len(best[2]["src"]), len(best[1]), status, time.time() - t0)
    return finalize_circuit(fm, graph, stats, best[1], best[2], data, cfg, "acdc", seconds, extra, seed)
