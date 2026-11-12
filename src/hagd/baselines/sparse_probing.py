"""Sparse probing baseline [5]: L1-regularised linear probes on dictionary features.

A multinomial logistic probe predicts the task label from pooled feature activations.
Features are ranked by the magnitude of their probe weights.  Probing yields a feature
set but no edges ("lacks the edge structure necessary for complete circuit
characterization", Section VI-B), so the edge count is reported as undefined.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..extract import finalize_circuit, make_evaluator
from ..search import FlatRanking, select_size


def sequence_labels(batches) -> np.ndarray:
    """One class per sequence: the answer class (generative) or correct/incorrect choice."""
    out = []
    for b in batches:
        for g, meta in enumerate(b.meta):
            if b.kind == "generative":
                out.append(int(meta.get("cls", 0)))
            else:
                out.extend(int(c == int(b.labels[g])) for c in range(b.n_choices))
    return np.asarray(out, dtype=np.int64)


def l1_logistic(x: np.ndarray, y: np.ndarray, c: float = 0.1, max_iter: int = 300, seed: int = 0) -> np.ndarray:
    """Multinomial logistic regression with an L1 penalty, solved by proximal gradient (ISTA).

    Returns the (classes, features) weight matrix.  ``c`` is the inverse penalty strength.
    """
    torch.manual_seed(seed)
    xt = torch.tensor(x, dtype=torch.float32)
    xt = xt / xt.abs().amax(0, keepdim=True).clamp_min(1e-8)
    classes, yi = np.unique(y, return_inverse=True)
    yt = torch.tensor(yi, dtype=torch.long)
    n, d = xt.shape
    w = torch.zeros(len(classes), d, requires_grad=True)
    bias = torch.zeros(len(classes), requires_grad=True)
    lam = 1.0 / (c * n)
    step = 1.0 / (0.25 * float((xt**2).sum(1).max()) + 1e-8)  # 1 / Lipschitz bound of the logistic loss
    for _ in range(max_iter):
        loss = torch.nn.functional.cross_entropy(xt @ w.t() + bias, yt)
        gw, gb = torch.autograd.grad(loss, [w, bias])
        with torch.no_grad():
            w -= step * gw
            bias -= step * gb
            w.copy_(torch.sign(w) * (w.abs() - step * lam).clamp_min(0))  # soft threshold
    return w.detach().numpy()


def run_sparse_probing(fm, graph, stats, data, cfg, seed: int = 0) -> dict:
    data.ensure_refs(fm)
    s, p = cfg.search, cfg.baselines.sparse_probing
    t0 = time.time()
    y = sequence_labels(data.discover)
    x = graph.anchors  # (n_seq, n_vertices): max activation of every candidate feature per sequence
    importance = np.zeros(graph.n)
    if len(np.unique(y)) > 1 and x.shape[0] == len(y):
        importance = np.abs(l1_logistic(x, y, float(p.C), int(p.max_iter), seed)).sum(0)
    evaluate = make_evaluator(fm, stats, graph, data.val, cfg.validation)
    ranking = FlatRanking(importance + 1e-9 * np.abs(graph.node_ie))  # attribution only breaks ties among zero weights
    nodes, trace, reached = select_size(ranking, evaluate, data.full_val["acc"], float(s.theta), int(s.min_nodes),
                                        int(s.max_nodes), float(s.growth), graph.n)
    seconds = {"activations": graph.info.get("node_seconds", 0.0), "probe_and_search": time.time() - t0}
    empty = {"src": np.zeros(0, dtype=np.int64), "dst": np.zeros(0, dtype=np.int64), "w": np.zeros(0, dtype=np.float32)}
    extra = {"reached_theta": bool(reached), "trace": trace, "status": "ok", "edges_defined": False,
             "probe_nonzero": int((importance > 0).sum()), "ops": {**ranking.ops, "model_evaluations": int(evaluate.calls)}}
    return finalize_circuit(fm, graph, stats, nodes, empty, data, cfg, "sparse_probing", seconds, extra, seed)
