"""Attribution patching baseline [13]: gradient-based approximation of activation patching.

A feature's effect is estimated in one backward pass as dy/df * (f - f_mean) instead of
being measured by an intervention.  Features are ranked by that estimate and the smallest
prefix reaching the fidelity threshold is returned.  There is no hierarchy, no learned
guidance and no necessity pruning.
"""

from __future__ import annotations

import time

import numpy as np

from ..extract import finalize_circuit, make_criterion, induced_edges, make_evaluator
from ..search import FlatRanking, select_size


def run_attribution_patching(fm, graph, stats, data, cfg, seed: int = 0) -> dict:
    data.ensure_refs(fm)
    s = cfg.search
    t0 = time.time()
    evaluate = make_evaluator(fm, stats, graph, data.val, cfg.validation)
    ranking = FlatRanking(np.abs(graph.node_ie))
    nodes, trace, reached = select_size(ranking, evaluate, make_criterion(data, evaluate, cfg), float(s.theta), int(s.min_nodes),
                                        int(s.max_nodes), float(s.growth), graph.n)
    seconds = {"attribution_nodes": graph.info.get("node_seconds", 0.0),
               "attribution_edges": graph.info.get("edge_seconds", 0.0), "search": time.time() - t0}
    extra = {"reached_theta": bool(reached), "trace": trace, "status": "ok",
             "ops": {**ranking.ops, "model_evaluations": int(evaluate.calls)}}
    return finalize_circuit(fm, graph, stats, nodes, induced_edges(graph, nodes), data, cfg, "attribution_patching",
                            seconds, extra, seed)
