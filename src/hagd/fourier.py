"""Fourier analysis of modular-arithmetic circuit features (paper Section V-A).

The paper reports that recovered circuits "exhibit interpretable Fourier-basis structure"
with "sinusoidal activation patterns".  For every circuit feature this module takes its
peak activation on each prompt "a + b mod p =", averages it as a function of a, of b and
of the answer c, and measures how concentrated the discrete Fourier spectrum of that
profile is.  A permutation null (shuffled residue labels) says how much concentration
arises by chance.
"""

from __future__ import annotations

import numpy as np
import torch

from .data.tasks import encode_examples


def _spectrum_stats(profile: np.ndarray) -> tuple[float, int, float]:
    """(share of non-DC power in the strongest frequency, that frequency, share in the top two)."""
    power = np.abs(np.fft.rfft(profile - profile.mean())) ** 2
    power = power[1:]
    total = power.sum()
    if total <= 1e-18:
        return 0.0, 0, 0.0
    order = np.argsort(-power)
    return float(power[order[0]] / total), int(order[0] + 1), float(power[order[:2]].sum() / total)


@torch.no_grad()
def fourier_analysis(fm, graph, nodes: np.ndarray, task, data_cfg, n_examples: int = 2048, n_null: int = 50,
                     seed: int = 0) -> dict:
    p = int(data_cfg.mod_p)
    nodes = np.asarray(nodes, dtype=np.int64)
    if len(nodes) == 0:
        return {"n_features": 0}
    examples = task.make("train", n_examples, seed) + task.make("test", n_examples // 4, seed)
    batches = encode_examples(fm.lm, examples, task.kind, int(data_cfg.batch_size), int(data_cfg.max_len))
    dev = fm.lm.device
    per_layer = {int(li): np.nonzero(graph.node_layer[nodes] == li)[0] for li in np.unique(graph.node_layer[nodes])}
    acts = np.zeros((len(examples), len(nodes)), dtype=np.float32)
    row = 0
    for b in batches:
        b = b.to(dev)
        feats = fm.features(b)
        tm = b.token_mask().unsqueeze(-1)
        for li, local in per_layer.items():
            ids = torch.as_tensor(graph.node_feat[nodes[local]], device=dev)
            a = torch.where(tm, feats[li][..., ids], torch.zeros((), device=dev)).amax(1)
            acts[row:row + a.shape[0], local] = a.cpu().numpy()
        row += b.input_ids.shape[0]
    abc = np.array([[ex.meta["a"], ex.meta["b"], ex.meta["c"]] for ex in examples])
    rng = np.random.default_rng(seed)
    features = []
    for j, v in enumerate(nodes):
        best = (0.0, 0, 0.0, "a")
        null = []
        for col, name in enumerate("abc"):
            sums = np.bincount(abc[:, col], weights=acts[:, j], minlength=p)
            cnt = np.maximum(np.bincount(abc[:, col], minlength=p), 1)
            profile = sums / cnt
            s = _spectrum_stats(profile)
            if s[0] > best[0]:
                best = (*s, name)
            for _ in range(n_null):
                null.append(_spectrum_stats(profile[rng.permutation(p)])[0])
        thresh = float(np.quantile(null, 0.95)) if null else 1.0
        features.append({"vertex": int(v), "layer_index": int(graph.node_layer[v]), "feature": int(graph.node_feat[v]),
                         "variable": best[3], "top_frequency": best[1], "top_share": best[0], "top2_share": best[2],
                         "null_p95": thresh, "fourier_structured": bool(best[0] > thresh and best[0] > 0)})
    shares = np.array([f["top_share"] for f in features])
    structured = np.array([f["fourier_structured"] for f in features])
    freqs = [f["top_frequency"] for f in features if f["fourier_structured"]]
    return {
        "p": p, "n_features": len(features), "n_examples": len(examples),
        "fraction_fourier_structured": float(structured.mean()),
        "median_top_share": float(np.median(shares)), "mean_top2_share": float(np.mean([f["top2_share"] for f in features])),
        "mean_null_p95": float(np.mean([f["null_p95"] for f in features])),
        "key_frequencies": {int(k): int(c) for k, c in zip(*np.unique(freqs, return_counts=True))} if freqs else {},
        "features": features,
    }
