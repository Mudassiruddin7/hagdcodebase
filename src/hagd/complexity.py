"""Empirical check of Theorem 1 (O(n^2 log n) hierarchical search vs. O(2^n) enumeration).

Synthetic layered attribution graphs of growing size n get a planted circuit.  The real
hierarchy builder and traversal are run on them with operation counters switched on, in
the worst case (the traversal is forced to rank every vertex and every adjacent-layer
pair of the circuit is verified).  The counts are compared with the bound of the theorem
and with the costs of exhaustive enumeration and ACDC-style edge pruning.
"""

from __future__ import annotations

import math

import numpy as np

from .hierarchy import build_hierarchy
from .search import Traversal


def synthetic_graph(n: int, n_layers: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Layered graph with a planted, strongly connected circuit of about sqrt(n) vertices."""
    layer = np.sort(rng.integers(0, n_layers, size=n))
    n_c = max(2, int(round(math.sqrt(n))))
    circuit = np.sort(rng.choice(n, size=n_c, replace=False))
    in_c = np.zeros(n, dtype=bool)
    in_c[circuit] = True
    src, dst, w = [], [], []
    by_layer = [np.nonzero(layer == li)[0] for li in range(n_layers)]
    for li in range(n_layers - 1):
        up, down = by_layer[li], by_layer[li + 1]
        if len(up) == 0 or len(down) == 0:
            continue
        for u in up:
            for v in rng.choice(down, size=min(len(down), 6), replace=False):
                strong = in_c[u] and in_c[v]
                src.append(u)
                dst.append(v)
                w.append(rng.uniform(0.6, 1.0) if strong else rng.uniform(0.0, 0.15))
    return np.array(src), np.array(dst), np.array(w), layer, circuit


def measure(n: int, branching: int, embed_dim: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n_layers = max(3, int(round(math.log2(n))))
    src, dst, w, layer, circuit = synthetic_graph(n, n_layers, rng)
    hier = build_hierarchy(n, src, dst, w, branching=branching, embed_dim=embed_dim, seed=seed, order_hint=layer)
    in_c = np.zeros(n, dtype=bool)
    in_c[circuit] = True
    # noisy membership scores at every level (stand-in for the GNN)
    scores = [np.clip(np.array([in_c[m].mean() for m in lvl.members]) * 0.8 + rng.uniform(0, 0.2, lvl.n), 1e-6, 1)
              for lvl in hier.levels]
    trav = Traversal(hier, scores, 0.3)
    trav.take(n)  # worst case: every vertex is ranked
    counts = np.bincount(layer, minlength=n_layers)
    edge_checks = int(sum(int(counts[i]) * int(counts[i + 1]) for i in range(n_layers - 1)))
    heap_cost = trav.ops["pq_ops"] * math.log2(max(n, 2))
    total = trav.ops["decisions"] + heap_cost + edge_checks * math.log2(max(n, 2))
    return {
        "n": n, "depth": hier.depth, "log_b_n": math.log(n, branching), "decisions": trav.ops["decisions"],
        "pq_ops": trav.ops["pq_ops"], "edge_checks": edge_checks, "hierarchical_ops": total,
        "bound_n2_log_n": n * n * math.log2(max(n, 2)), "exhaustive_log2_ops": float(n),  # 2^n subsets
        "acdc_ops": float(edge_checks) * n,  # one model run per edge, each O(n) feature updates
    }


def complexity_study(sizes: list[int], trials: int, branching: int, embed_dim: int, seed: int = 0) -> dict:
    rows = []
    for n in sizes:
        runs = [measure(int(n), branching, embed_dim, seed + 31 * t) for t in range(trials)]
        rows.append({k: float(np.mean([r[k] for r in runs])) for k in runs[0]})
    ns = np.log([r["n"] for r in rows])
    fit = lambda key: float(np.polyfit(ns, np.log([max(r[key], 1.0) for r in rows]), 1)[0]) if len(rows) > 1 else float("nan")  # noqa: E731
    return {
        "rows": rows,
        "exponent_hierarchical": fit("hierarchical_ops"),  # Theorem 1 predicts <= 2 (plus the log factor)
        "exponent_decisions": fit("decisions"),  # predicted 1
        "exponent_acdc": fit("acdc_ops"),
        "max_ratio_to_bound": float(max(r["hierarchical_ops"] / r["bound_n2_log_n"] for r in rows)),
        "depth_over_log_b_n": float(np.mean([r["depth"] / max(r["log_b_n"], 1e-9) for r in rows])),
    }
