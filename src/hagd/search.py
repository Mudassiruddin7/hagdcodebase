"""Circuit search through the hierarchy (paper Section III-D/E, Theorem 1).

A best-first traversal keeps a priority queue of (super)nodes ordered by the GNN's
circuit-membership score.  Popping a supernode pushes its O(b) children; popping a leaf
emits a feature.  The emitted sequence is a ranking of features, and the circuit is the
shortest prefix whose behavioural preservation reaches the fidelity threshold theta
(Eq. 2: minimise |C| subject to phi(C) >= theta).

The traversal counts its own operations so the complexity claim of Theorem 1 can be
checked empirically (``hagd.complexity``).
"""

from __future__ import annotations

import heapq
import math
from typing import Callable

import numpy as np

from .hierarchy import Hierarchy


class Traversal:
    """Lazy best-first descent through the hierarchy."""

    def __init__(self, hier: Hierarchy, scores: list[np.ndarray], gamma: float = 0.3) -> None:
        self.hier, self.scores, self.gamma = hier, scores, float(gamma)
        self.children: list[list[np.ndarray]] = [[]]
        for r in range(1, len(hier.levels)):
            order = np.argsort(hier.parent[r - 1], kind="stable")
            bounds = np.searchsorted(hier.parent[r - 1][order], np.arange(hier.levels[r].n + 1))
            self.children.append([order[bounds[v]:bounds[v + 1]] for v in range(hier.levels[r].n)])
        self.heap: list[tuple[float, int, int, int]] = []
        self.order: list[int] = []
        self.ops = {"decisions": 0, "pq_ops": 0, "expansions": 0}
        self._tick = 0
        top = hier.depth
        for v in range(hier.levels[top].n):
            self._push(float(scores[top][v]), top, v)

    def _push(self, priority: float, level: int, v: int) -> None:
        heapq.heappush(self.heap, (-priority, self._tick, level, v))
        self._tick += 1
        self.ops["decisions"] += 1
        self.ops["pq_ops"] += 1

    def take(self, n: int) -> list[int]:
        """The first ``n`` features of the ranking (expanding the hierarchy as far as needed)."""
        while len(self.order) < n and self.heap:
            neg, _, level, v = heapq.heappop(self.heap)
            self.ops["pq_ops"] += 1
            if level == 0:
                self.order.append(int(v))
                continue
            self.ops["expansions"] += 1
            parent_priority = max(-neg, 1e-12)
            for c in self.children[level][v]:
                own = max(float(self.scores[level - 1][c]), 1e-12)
                self._push(own ** (1 - self.gamma) * parent_priority ** self.gamma, level - 1, int(c))
        return self.order[:n]


class FlatRanking:
    """'No hierarchy' ablation: score every feature directly and sort."""

    def __init__(self, leaf_scores: np.ndarray) -> None:
        self.order = [int(i) for i in np.argsort(-leaf_scores, kind="stable")]
        n = len(self.order)
        self.ops = {"decisions": n, "pq_ops": int(n * max(1.0, math.log2(max(n, 2)))), "expansions": 0}

    def take(self, n: int) -> list[int]:
        return self.order[:n]


def select_size(ranking, evaluate: Callable[[np.ndarray], dict], score: Callable[[dict], float], theta: float,
                min_nodes: int, max_nodes: int, growth: float, n_total: int) -> tuple[np.ndarray, list[dict], bool]:
    """Smallest ranked prefix with score >= theta (geometric growth, then bisection).

    Returns (selected vertices, evaluation trace, whether theta was reached).
    """
    cap = int(min(max_nodes, n_total))
    trace: list[dict] = []

    def phi_at(k: int) -> float:
        nodes = np.asarray(ranking.take(k), dtype=np.int64)
        res = evaluate(nodes)
        phi = float(score(res))
        trace.append({"nodes": int(len(nodes)), "phi": phi, "acc": res["acc"], "loss": res["loss"]})
        return phi

    k = int(min(max(1, min_nodes), cap))
    lo = 0
    phi = phi_at(k)
    while phi < theta and k < cap:
        lo = k
        k = int(min(cap, max(k + 1, math.ceil(k * growth))))
        phi = phi_at(k)
    reached = phi >= theta
    if reached:
        hi = k
        while hi - lo > max(1, int(0.05 * hi)):
            mid = (lo + hi) // 2
            if phi_at(mid) >= theta:
                hi = mid
            else:
                lo = mid
        k = hi
    else:  # theta is out of reach within the budget: smallest evaluated prefix within 5% of the best phi seen
        best = max(t["phi"] for t in trace)
        k = min(t["nodes"] for t in trace if t["phi"] >= 0.95 * best)
    return np.asarray(ranking.take(k), dtype=np.int64), trace, reached


def select_by_score(leaf_scores: np.ndarray, threshold: float, max_nodes: int, min_nodes: int) -> np.ndarray:
    """'No causal validation' ablation: take every feature the GNN scores above ``threshold``."""
    order = np.argsort(-leaf_scores, kind="stable")
    k = int(np.clip(int((leaf_scores >= threshold).sum()), min_nodes, max_nodes))
    return order[:k].astype(np.int64)
