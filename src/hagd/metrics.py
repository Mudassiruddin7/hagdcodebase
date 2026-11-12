"""Circuit quality metrics (paper Section IV-C).

Sparsity:          node count, edge count, compression ratio.
Faithfulness:      see ``hagd.validation.sufficiency_report``.
Interpretability:  concept purity, circuit modularity, description length.

Two of the interpretability metrics are defined only informally in the paper, so the
operationalisations used here are stated explicitly:

* Concept purity – "fraction of features with single interpretable meanings".  Proxy: a
  feature is pure when at least half of its activation mass falls on one token identity,
  either the token it fires on or the token that follows.
* Modularity – "ratio of intra-module to inter-module edge weights", reported as the share
  intra / (intra + inter) so it lies in [0, 1] like the paper's values.  Modules are the
  spectral partition of the circuit that maximises Newman's Q.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
import torch

from .attribution import AttributionGraph
from .hierarchy import spectral_partition


def _log2_binom(n: int, k: int) -> float:
    if k < 0 or k > n or n <= 0:
        return 0.0
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2)


def sparsity_metrics(g: AttributionGraph, nodes: np.ndarray, n_edges: int) -> dict:
    return {
        "nodes": int(len(nodes)),
        "edges": int(n_edges),
        "compression_nodes": len(nodes) / max(g.n, 1),  # fraction of the attribution graph retained
        "compression_edges": n_edges / max(g.n_edges, 1),
        "dictionary_fraction": len(nodes) / max(g.n_dictionary, 1),
    }


def description_length(g: AttributionGraph, nodes: np.ndarray, n_edges: int, weight_bits: int = 8) -> float:
    """Bits to specify the circuit: which features, which adjacent-layer pairs, and edge weights."""
    layers = g.node_layer[np.asarray(nodes, dtype=np.int64)] if len(nodes) else np.zeros(0, dtype=np.int64)
    counts = np.bincount(layers, minlength=len(g.layers))
    possible = int(sum(int(counts[i]) * int(counts[i + 1]) for i in range(len(counts) - 1)))
    return _log2_binom(g.n_dictionary, len(nodes)) + _log2_binom(possible, min(n_edges, possible)) + weight_bits * n_edges


def modularity(n: int, src: np.ndarray, dst: np.ndarray, w: np.ndarray, seed: int = 0) -> dict:
    """Intra-module weight share and Newman's Q for the best spectral partition into 2..8 modules."""
    if n < 4 or len(w) == 0 or float(np.sum(w)) <= 0:
        return {"modularity": float("nan"), "newman_q": float("nan"), "modules": 0}
    a = sp.coo_matrix((np.abs(w).astype(np.float64), (src, dst)), shape=(n, n)).tocsr()
    a = a + a.T
    total = a.sum() / 2.0
    deg = np.asarray(a.sum(1)).ravel()
    best = None
    coo = a.tocoo()
    for k in range(2, min(8, n // 2) + 1):
        labels = spectral_partition(a, k, embed_dim=k, max_size=n, seed=seed)
        same = labels[coo.row] == labels[coo.col]
        intra = float(coo.data[same].sum() / 2.0)
        q = intra / total - float(sum((deg[labels == c].sum() / (2 * total)) ** 2 for c in np.unique(labels)))
        if best is None or q > best[0]:
            best = (q, intra / total, int(labels.max()) + 1)
    return {"modularity": best[1], "newman_q": best[0], "modules": best[2]}


@torch.no_grad()
def concept_purity(fm, graph: AttributionGraph, nodes: np.ndarray, batches, threshold: float = 0.5) -> dict:
    """Token-level purity proxy for the circuit's features (see module docstring)."""
    nodes = np.asarray(nodes, dtype=np.int64)
    if len(nodes) == 0:
        return {"concept_purity": float("nan"), "mean_purity": float("nan")}
    dev = fm.lm.device
    vocab = int(fm.lm.model.config.vocab_size)
    per_layer = {int(li): torch.as_tensor(graph.node_feat[nodes[graph.node_layer[nodes] == li]], device=dev)
                 for li in np.unique(graph.node_layer[nodes])}
    cur = {li: torch.zeros(len(ids), vocab, device=dev) for li, ids in per_layer.items()}
    nxt = {li: torch.zeros(len(ids), vocab, device=dev) for li, ids in per_layer.items()}
    for b in batches:
        b = b.to(dev)
        feats = fm.features(b)
        tm = b.token_mask()
        rows, cols = tm.nonzero(as_tuple=True)
        tok = b.input_ids[rows, cols]
        has_next = cols + 1 < b.input_ids.shape[1]
        nxt_tok = b.input_ids[rows, (cols + 1).clamp_max(b.input_ids.shape[1] - 1)]
        valid_next = has_next & b.attention_mask[rows, (cols + 1).clamp_max(b.input_ids.shape[1] - 1)].bool()
        for li, ids in per_layer.items():
            acts = feats[li][rows, cols][:, ids]  # (T, k)
            cur[li].index_add_(1, tok, acts.t())
            nxt[li].index_add_(1, nxt_tok[valid_next], acts[valid_next].t())
    purities = []
    for li in per_layer:
        for mass in (cur[li], nxt[li]):
            mass.div_(mass.sum(1, keepdim=True).clamp_min(1e-12))
        purities.append(torch.maximum(cur[li].amax(1), nxt[li].amax(1)).cpu().numpy())
    p = np.concatenate(purities)
    return {"concept_purity": float((p >= threshold).mean()), "mean_purity": float(p.mean())}
