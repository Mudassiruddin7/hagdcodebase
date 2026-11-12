"""Cross-architecture circuit transfer (paper Section V-C, Eq. 11).

    tau(C1, C2) = |E(C1) ∩ E(C2)| / |E(C1) ∪ E(C2)|

"Edge correspondence is established through feature alignment based on activation
correlations": both models are run on the same task examples, every circuit feature is
summarised by its per-example peak activation, and features are matched one-to-one by
maximum-weight assignment on the Pearson correlation matrix (matches below ``min_corr``
are discarded).

Models of different depth do not share a notion of "adjacent layer".  An edge a -> b of
one circuit therefore counts as present in the other when the images of a and b are
connected by a directed path of at most ceil(L_other / L_this) edges.  The intersection
is the mean of the two directed counts.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment


def _corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(0, keepdims=True)
    b = b - b.mean(0, keepdims=True)
    na = np.linalg.norm(a, axis=0, keepdims=True)
    nb = np.linalg.norm(b, axis=0, keepdims=True)
    c = (a.T @ b) / np.maximum(na.T @ nb, 1e-12)
    c[(na.T == 0).ravel()] = 0
    c[:, (nb == 0).ravel()] = 0
    return np.nan_to_num(c)


def _reach(n: int, edges: np.ndarray, hops: int) -> np.ndarray:
    adj = np.zeros((n, n), dtype=bool)
    if len(edges):
        adj[edges[:, 0], edges[:, 1]] = True
    reach, step = adj.copy(), adj.copy()
    for _ in range(hops - 1):
        step = (step.astype(np.int32) @ adj.astype(np.int32)) > 0
        reach |= step
    return reach


def transfer_coefficient(c1: dict, anchors1: np.ndarray, n_layers1: int, c2: dict, anchors2: np.ndarray,
                         n_layers2: int, min_corr: float = 0.3) -> dict:
    """Eq. 11 between two circuit records (as produced by ``finalize_circuit``).

    ``anchors*`` are the attribution graphs' (n_examples, n_vertices) activation summaries.
    """
    v1 = np.array([n["vertex"] for n in c1["nodes"]], dtype=np.int64)
    v2 = np.array([n["vertex"] for n in c2["nodes"]], dtype=np.int64)
    e1 = np.array([[e[0], e[1]] for e in c1["edges"]], dtype=np.int64).reshape(-1, 2)
    e2 = np.array([[e[0], e[1]] for e in c2["edges"]], dtype=np.int64).reshape(-1, 2)
    out = {"tau": float("nan"), "aligned": 0, "mean_corr": float("nan"), "edges_1": int(len(e1)), "edges_2": int(len(e2)),
           "nodes_1": int(len(v1)), "nodes_2": int(len(v2))}
    rows = min(anchors1.shape[0], anchors2.shape[0])
    if len(v1) == 0 or len(v2) == 0 or rows < 3 or (len(e1) + len(e2)) == 0:
        return out
    corr = _corr(anchors1[:rows][:, v1].astype(np.float64), anchors2[:rows][:, v2].astype(np.float64))
    r, c = linear_sum_assignment(-corr)
    ok = corr[r, c] >= min_corr
    fwd = {int(i): int(j) for i, j in zip(r[ok], c[ok])}
    bwd = {j: i for i, j in fwd.items()}
    h2 = max(1, math.ceil(n_layers2 / n_layers1))
    h1 = max(1, math.ceil(n_layers1 / n_layers2))
    reach1, reach2 = _reach(len(v1), e1, h1), _reach(len(v2), e2, h2)
    shared1 = sum(1 for a, b in e1 if int(a) in fwd and int(b) in fwd and reach2[fwd[int(a)], fwd[int(b)]])
    shared2 = sum(1 for a, b in e2 if int(a) in bwd and int(b) in bwd and reach1[bwd[int(a)], bwd[int(b)]])
    inter = (shared1 + shared2) / 2.0
    union = len(e1) + len(e2) - inter
    out.update({"tau": float(inter / union) if union > 0 else float("nan"), "aligned": int(ok.sum()),
                "mean_corr": float(corr[r, c][ok].mean()) if ok.any() else float("nan"),
                "shared_1": int(shared1), "shared_2": int(shared2)})
    return out
