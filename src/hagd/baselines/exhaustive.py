"""Exhaustive search baseline ("brute-force enumeration where computationally feasible").

Enumerating all 2^n vertex subsets is only possible for a handful of vertices, so the
search is exhaustive in three stages:

1. Pool.  Start from the ``pool`` most strongly attributed features and double the pool
   until it satisfies the fidelity criterion (``hagd.validation.Criterion``).  If even the
   whole attribution graph does not, the task is reported as ``unreachable`` and the whole
   graph is returned unchanged - a search that cannot reach theta must not shrink to a
   trivially small "circuit".
2. Elimination.  Every round measures, by intervention, the effect of removing *each*
   remaining feature and drops the least harmful ones (the largest number that keeps the
   criterion, found by bisection, at most ``drop_frac`` of the circuit per round).
   One model run per feature per round.
3. Enumeration.  Once at most ``max_enum`` features remain, every subset is enumerated
   (smallest first) and the smallest one that still meets the criterion is returned - the
   exact optimum of Eq. 2 inside that set.

Its output doubles as the ground-truth circuit labels the GNN is trained on
("obtained through exhaustive search on small models", Section III-E).
"""

from __future__ import annotations

import itertools
import time
from typing import Callable

import numpy as np

from ..extract import finalize_circuit, induced_edges, make_criterion, make_evaluator
from ..utils import log


def search(order: np.ndarray, score: Callable[[np.ndarray], float], theta: float, pool: int, max_enum: int,
           drop_frac: float, budget_seconds: float) -> tuple[np.ndarray, dict]:
    """The three-stage search over vertices ranked by ``order``; ``score(vertices)`` is the fidelity criterion."""
    t0 = time.time()
    n = len(order)
    # stage 1: smallest pool (doubling) that meets the criterion, then bisect back to the smallest passing prefix
    size, lo = int(min(max(1, pool), n)), 0
    while True:
        pool_score = score(order[:size])
        if pool_score >= theta or size >= n:
            break
        lo, size = size, int(min(n, size * 2))
    if pool_score >= theta:
        while size - lo > max(1, int(0.05 * size)):
            mid = (lo + size) // 2
            if score(order[:mid]) >= theta:
                size = mid
            else:
                lo = mid
        pool_score = score(order[:size])
    circuit = np.asarray(order[:size], dtype=np.int64)
    status = "ok" if pool_score >= theta else "unreachable"

    # stage 2: elimination by measured leave-one-out effect
    rounds = 0
    while status == "ok" and len(circuit) > max_enum:
        if time.time() - t0 > budget_seconds:
            status = "timeout"
            break
        kept = np.array([score(np.delete(circuit, i)) for i in range(len(circuit))])  # score without each feature
        cand = np.argsort(-kept, kind="stable")  # least harmful removals first
        cap = max(1, int(drop_frac * len(circuit))) if drop_frac > 0 else 1
        cand = cand[:cap]
        lo_, hi = 0, len(cand)
        while lo_ < hi:
            mid = (lo_ + hi + 1) // 2
            if score(np.delete(circuit, cand[:mid])) >= theta:
                lo_ = mid
            else:
                hi = mid - 1
        if lo_ == 0:
            break
        circuit = np.delete(circuit, cand[:lo_])
        rounds += 1

    # stage 3: exact enumeration of the survivors (smallest subset first)
    enumerated = False
    if status == "ok" and 1 < len(circuit) <= max_enum:
        enumerated = True
        for k in range(1, len(circuit)):
            best = None
            for subset in itertools.combinations(circuit.tolist(), k):
                s = score(np.asarray(subset, dtype=np.int64))
                if s >= theta and (best is None or s > best[0]):
                    best = (s, subset)
            if time.time() - t0 > budget_seconds:
                status = "timeout"
                break
            if best is not None:
                circuit = np.asarray(best[1], dtype=np.int64)
                break
    return circuit, {"status": status, "pool": size, "pool_score": float(pool_score), "rounds": rounds,
                     "enumerated_exactly": enumerated}


def run_exhaustive(fm, graph, stats, data, cfg, seed: int = 0, pool: int | None = None) -> dict:
    data.ensure_refs(fm)
    e = cfg.baselines.exhaustive
    theta = float(cfg.search.theta)
    t0 = time.time()
    evaluate = make_evaluator(fm, stats, graph, data.val, cfg.validation)
    crit = make_criterion(data, evaluate, cfg)
    score = lambda nodes: float(crit(evaluate(nodes)))  # noqa: E731
    order = np.argsort(-np.abs(graph.node_ie), kind="stable").astype(np.int64)
    circuit, info = search(order, score, theta, int(pool or e.pool), int(e.max_enum), float(e.drop_frac),
                           float(e.max_minutes) * 60)
    log.info("exhaustive %s/%s: pool %d (score %.3f, %s) -> %d vertices after %d rounds, %d model evaluations",
             graph.model, graph.task, info["pool"], info["pool_score"], info["status"], len(circuit), info["rounds"],
             evaluate.calls)
    seconds = {"attribution_nodes": graph.info.get("node_seconds", 0.0),
               "attribution_edges": graph.info.get("edge_seconds", 0.0), "search": time.time() - t0}
    extra = {**info, "reached_theta": bool(info["pool_score"] >= theta), "criterion": crit.kind,
             "val_score": score(circuit), "ops": {"model_evaluations": int(evaluate.calls)}}
    return finalize_circuit(fm, graph, stats, circuit, induced_edges(graph, circuit), data, cfg, "exhaustive",
                            seconds, extra, seed)
