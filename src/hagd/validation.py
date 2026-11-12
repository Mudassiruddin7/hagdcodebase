"""Causal validation protocol (paper Section III-F).

Necessity (Eq. 9):  Delta_i = E_x[ L(M(x; C \\ {f_i})) - L(M(x; C)) ];  features with
Delta_i < epsilon are pruned.

Sufficiency (Eq. 10):  phi(C) = Acc(M_C) / Acc(M_full);  phi > 0.9 counts as sufficient.

To limit the validation circularity the paper warns about (Section VI-C), discovery and
pruning use the ``discover`` / ``val`` splits and every reported number comes from a
``test`` split that the search never sees.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .utils import ratio_bootstrap_ci


def necessity_prune(nodes: np.ndarray, evaluate: Callable[[np.ndarray], dict], acc_full: float, epsilon: float,
                    theta: float, rounds: int = 2) -> tuple[np.ndarray, dict]:
    """Remove features whose individual ablation does not raise the task loss by epsilon."""
    nodes = np.asarray(nodes, dtype=np.int64)
    n_evals, removed_total, deltas_last = 0, 0, {}
    for _ in range(max(1, rounds)):
        if len(nodes) <= 1:
            break
        base = evaluate(nodes)
        n_evals += 1
        base_phi = base["acc"] / acc_full if acc_full > 0 else 0.0
        floor = theta if base_phi >= theta else base_phi - 0.01
        deltas = np.empty(len(nodes))
        for i in range(len(nodes)):
            deltas[i] = evaluate(np.delete(nodes, i))["loss"] - base["loss"]  # Eq. 9
        n_evals += len(nodes)
        deltas_last = {int(v): float(d) for v, d in zip(nodes, deltas)}
        cand = np.argsort(deltas, kind="stable")
        cand = cand[deltas[cand] < epsilon]
        if len(cand) == 0:
            break
        # drop as many unnecessary features as possible without losing sufficiency
        lo, hi = 0, len(cand)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            trial = np.delete(nodes, cand[:mid])
            n_evals += 1
            ok = len(trial) > 0 and (evaluate(trial)["acc"] / acc_full if acc_full > 0 else 0.0) >= floor
            if ok:
                lo = mid
            else:
                hi = mid - 1
        if lo == 0:
            break
        nodes = np.delete(nodes, cand[:lo])
        removed_total += lo
    return nodes, {"evaluations": n_evals, "removed": removed_total, "deltas": deltas_last}


def sufficiency_report(fm, stats, graph, nodes: np.ndarray, batches, full: dict, vcfg, seed: int = 0) -> dict:
    """All faithfulness numbers of Section IV-C for circuit ``nodes`` on held-out ``batches``.

    ``full`` is the result of ``fm.reference(batches)`` (full-model run).
    """
    from .replacement import nodes_to_dict

    def run(node_ids, mode="restrict", error=None):
        iv = fm.intervention(stats, nodes_to_dict(graph.node_layer, graph.node_feat, node_ids), mode=mode,
                             error=error or vcfg.error_mode, ablation=vcfg.ablation)
        return fm.evaluate(batches, iv, ref=full["dist"])

    circ = run(nodes)
    empty = run(np.zeros(0, dtype=np.int64))
    strict = circ if vcfg.error_mode == "mean" else run(nodes, error="mean")
    lenient = circ if vcfg.error_mode == "clean" else run(nodes, error="clean")
    lenient_empty = empty if vcfg.error_mode == "clean" else run(np.zeros(0, dtype=np.int64), error="clean")
    knock = run(nodes, mode="knockout")
    acc_full, loss_full = full["acc"], full["loss"]
    phi = circ["acc"] / acc_full if acc_full > 0 else float("nan")
    lo, hi = ratio_bootstrap_ci(circ["correct"], full["correct"], int(vcfg.bootstrap), seed)
    gap_acc = acc_full - empty["acc"]
    gap_loss = empty["loss"] - loss_full
    return {
        "acc_full": acc_full, "acc_circuit": circ["acc"], "loss_full": loss_full, "loss_circuit": circ["loss"],
        "preservation": phi,  # Eq. 10
        "preservation_ci95": [lo, hi],
        "sufficient": bool(phi > 0.9),
        "error_mode": vcfg.error_mode,
        # the same circuit under both treatments of the transcoder residual
        "preservation_strict": strict["acc"] / acc_full if acc_full > 0 else float("nan"),  # residual mean-ablated
        "preservation_lenient": lenient["acc"] / acc_full if acc_full > 0 else float("nan"),  # residual kept
        # a circuit with no features at all: how much "preservation" comes for free
        "preservation_empty": empty["acc"] / acc_full if acc_full > 0 else float("nan"),
        "preservation_empty_lenient": lenient_empty["acc"] / acc_full if acc_full > 0 else float("nan"),
        "faithfulness_acc": (circ["acc"] - empty["acc"]) / gap_acc if abs(gap_acc) > 1e-9 else float("nan"),
        "faithfulness_loss": (empty["loss"] - circ["loss"]) / gap_loss if abs(gap_loss) > 1e-9 else float("nan"),
        "intervention_effect": acc_full - knock["acc"],  # accuracy change when the circuit is ablated
        "knockout_loss_increase": knock["loss"] - loss_full,
        "reconstruction_error_kl": circ["kl"],  # KL(full || circuit) of the output distribution
        "n_test": int(full["n"]),
    }
