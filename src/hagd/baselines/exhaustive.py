"""Exhaustive search baseline ("brute-force enumeration where computationally feasible").

Enumerating all 2^n vertex subsets is only possible for a handful of vertices, so the
search is exhaustive in two stages:

1. Backward elimination over a pool of the ``pool`` most strongly attributed features.
   Every round measures, by intervention, the effect of removing *each* remaining feature
   and discards the one that hurts least, while phi(C) >= theta.  O(pool^2) model runs.
2. Once at most ``max_enum`` features remain, every subset is enumerated (smallest first)
   and the smallest one that still satisfies phi >= theta is returned - the exact optimum
   of Eq. 2 inside that set.

Its output doubles as the ground-truth circuit labels the GNN is trained on
("obtained through exhaustive search on small models", Section III-E).
"""

from __future__ import annotations

import itertools
import time

import numpy as np

from ..extract import finalize_circuit, induced_edges, make_evaluator
from ..utils import log


def run_exhaustive(fm, graph, stats, data, cfg, seed: int = 0, pool: int | None = None) -> dict:
    data.ensure_refs(fm)
    e = cfg.baselines.exhaustive
    theta = float(cfg.search.theta)
    acc_full = data.full_val["acc"]
    budget = float(e.max_minutes) * 60
    t0 = time.time()
    evaluate = make_evaluator(fm, stats, graph, data.val, cfg.validation)
    phi = lambda nodes: (evaluate(nodes)["acc"] / acc_full) if acc_full > 0 else 0.0  # noqa: E731

    size = int(min(pool or e.pool, graph.n))
    circuit = np.argsort(-np.abs(graph.node_ie), kind="stable")[:size].astype(np.int64)
    phi_pool = phi(circuit)
    floor = theta if phi_pool >= theta else phi_pool - 0.01
    status = "ok"

    # stage 1: exhaustive leave-one-out elimination
    while len(circuit) > 1:
        if time.time() - t0 > budget:
            status = "timeout"
            break
        trials = [np.delete(circuit, i) for i in range(len(circuit))]
        res = [evaluate(t) for t in trials]
        feasible = [i for i, r in enumerate(res) if (r["acc"] / acc_full if acc_full > 0 else 0.0) >= floor]
        if not feasible:
            break
        feasible.sort(key=lambda i: (-res[i]["acc"], res[i]["loss"]))
        n_drop = max(1, int(float(e.drop_frac) * len(circuit)))
        drop = feasible[:n_drop]
        if len(drop) > 1 and phi(np.delete(circuit, drop)) < floor:
            drop = drop[:1]
        circuit = np.delete(circuit, drop)

    # stage 2: exact enumeration of the survivors
    enumerated = False
    if status == "ok" and 1 < len(circuit) <= int(e.max_enum):
        enumerated = True
        for k in range(1, len(circuit)):
            best = None
            for subset in itertools.combinations(circuit.tolist(), k):
                r = evaluate(np.asarray(subset, dtype=np.int64))
                if (r["acc"] / acc_full if acc_full > 0 else 0.0) >= floor:
                    key = (r["acc"], -r["loss"])
                    if best is None or key > best[0]:
                        best = (key, subset)
            if time.time() - t0 > budget:
                status = "timeout"
                break
            if best is not None:
                circuit = np.asarray(best[1], dtype=np.int64)
                break

    log.info("exhaustive %s/%s: pool %d -> %d vertices, %d model evaluations (%s)",
             graph.model, graph.task, size, len(circuit), evaluate.calls, status)
    seconds = {"attribution_nodes": graph.info.get("node_seconds", 0.0),
               "attribution_edges": graph.info.get("edge_seconds", 0.0), "search": time.time() - t0}
    extra = {"status": status, "pool": size, "pool_preservation": phi_pool, "reached_theta": bool(phi_pool >= theta),
             "enumerated_exactly": enumerated, "ops": {"model_evaluations": int(evaluate.calls)}}
    return finalize_circuit(fm, graph, stats, circuit, induced_edges(graph, circuit), data, cfg, "exhaustive",
                            seconds, extra, seed)
