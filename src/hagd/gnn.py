"""GNN meta-model that guides hierarchical circuit search (paper Section III-E).

Message passing follows Eq. 7-8 exactly:

    z_v^(t+1) = MLP( z_v^(t) + sum_{u in N(v)} alpha_uv z_u^(t) )
    alpha_uv  = softmax_u( a^T [ z_u^(t) || z_v^(t) ] )

and the output layer gives a circuit-membership probability per vertex.  The same network
scores every level of the hierarchy (supernodes carry aggregated vertex features).

Training needs ground-truth circuits; they come from exhaustive search on small models
(``hagd.baselines.exhaustive``).  "The meta-learning formulation trains on a distribution
of circuit discovery tasks": each (model, task) graph is one meta-learning task and the
weights are meta-trained with Reptile, so the model transfers to graphs of unseen models.
Vertex features are normalised per graph and contain no model-specific identifiers.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .attribution import AttributionGraph
from .hierarchy import Hierarchy
from .utils import load_json, log, save_json

N_FEATURES = 27


@dataclass
class LevelData:
    x: torch.Tensor  # (n, N_FEATURES)
    edge_index: torch.Tensor  # (2, E) both directions
    logw: torch.Tensor  # (E,) log edge weight, normalised by the level maximum
    y: torch.Tensor | None  # (n,) 1 = contains a ground-truth circuit vertex


@dataclass
class GraphSample:
    name: str
    model: str
    levels: list[LevelData]


def _rank01(v: np.ndarray) -> np.ndarray:
    if len(v) <= 1:
        return np.ones(len(v), dtype=np.float32)
    order = np.argsort(np.argsort(v, kind="stable"), kind="stable")
    return (order / (len(v) - 1)).astype(np.float32)


def leaf_features(g: AttributionGraph) -> tuple[np.ndarray, np.ndarray]:
    """Extensive (summed over members) and intensive (averaged) raw vertex features."""
    n = g.n
    in_s, out_s = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
    np.add.at(in_s, g.edge_dst, g.edge_w)
    np.add.at(out_s, g.edge_src, g.edge_w)
    last = g.node_layer == (len(g.layers) - 1)
    ext = np.stack([g.node_attr_abs, g.node_attr, np.abs(g.node_ie), g.node_ie, in_s, out_s, g.node_xl_in,
                    g.node_xl_out, np.where(last, g.node_attr_abs, 0.0)], axis=1).astype(np.float32)
    depth = g.node_layer / max(1, len(g.layers) - 1)
    inten = np.stack([g.node_act, g.node_freq, depth], axis=1).astype(np.float32)
    return ext, inten


def make_sample(g: AttributionGraph, hier: Hierarchy, circuit: np.ndarray | None = None, name: str | None = None) -> GraphSample:
    """Vertex features, edges and (optional) labels for every level of the hierarchy."""
    ext, inten = leaf_features(g)
    leaf_scale = np.abs(ext).max(0, keepdims=True) + 1e-12
    in_circuit = np.zeros(g.n, dtype=bool)
    if circuit is not None:
        in_circuit[np.asarray(circuit, dtype=np.int64)] = True
    total_w = float(g.edge_w.sum()) + 1e-12
    levels = []
    for r, lvl in enumerate(hier.levels):
        e_sum = np.stack([ext[m].sum(0) for m in lvl.members])
        e_max = np.stack([np.abs(ext[m]).max(0) for m in lvl.members])
        i_mean = np.stack([inten[m].mean(0) for m in lvl.members])
        depth_std = np.array([inten[m][:, 2].std() for m in lvl.members], dtype=np.float32)
        size = np.array([len(m) for m in lvl.members], dtype=np.float32)
        feats = np.concatenate([
            e_sum / (np.abs(e_sum).max(0, keepdims=True) + 1e-12),  # 9: share of the level's strongest vertex
            e_max / leaf_scale,  # 9: strongest member relative to the strongest feature of the graph
            i_mean / (np.abs(i_mean).max(0, keepdims=True) + 1e-12),  # 3
            depth_std[:, None],
            (np.log1p(size) / np.log1p(g.n))[:, None],
            np.full((lvl.n, 1), r / max(1, hier.depth), dtype=np.float32),
            _rank01(e_sum[:, 0])[:, None],
            _rank01(e_sum[:, 2])[:, None],
            (lvl.intra / total_w)[:, None],
        ], axis=1).astype(np.float32)
        assert feats.shape[1] == N_FEATURES
        coo = lvl.adj.tocoo()
        w = coo.data.astype(np.float32)
        logw = np.log(w / (w.max() + 1e-12) + 1e-6) if len(w) else w
        y = None
        if circuit is not None:
            y = torch.tensor([bool(in_circuit[m].any()) for m in lvl.members], dtype=torch.float32)
        levels.append(LevelData(torch.tensor(feats), torch.tensor(np.stack([coo.row, coo.col]), dtype=torch.long),
                                torch.tensor(logw, dtype=torch.float32), y))
    return GraphSample(name or f"{g.model}/{g.task}", g.model, levels)


class GATLayer(nn.Module):
    def __init__(self, dim: int, use_edge_weight: bool) -> None:
        super().__init__()
        self.a_src = nn.Parameter(torch.randn(dim) * dim**-0.5)  # a = [a_src || a_dst] in Eq. 8
        self.a_dst = nn.Parameter(torch.randn(dim) * dim**-0.5)
        self.beta = nn.Parameter(torch.zeros(())) if use_edge_weight else None
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor, logw: torch.Tensor) -> torch.Tensor:
        agg = torch.zeros_like(z)
        if edge_index.shape[1] > 0:
            src, dst = edge_index  # message u = src -> v = dst
            e = (z @ self.a_src)[src] + (z @ self.a_dst)[dst]
            if self.beta is not None:
                e = e + self.beta * logw
            e_max = torch.full((z.shape[0],), -torch.inf, device=z.device).scatter_reduce(0, dst, e, "amax")
            ex = (e - e_max[dst]).exp()
            denom = torch.zeros(z.shape[0], device=z.device).index_add_(0, dst, ex)
            alpha = ex / denom[dst].clamp_min(1e-12)  # Eq. 8
            agg = agg.index_add(0, dst, alpha.unsqueeze(-1) * z[src])
        return self.mlp(z + agg)  # Eq. 7


class CircuitGNN(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 3, use_edge_weight: bool = False) -> None:
        super().__init__()
        self.cfg = {"hidden": hidden, "layers": layers, "use_edge_weight": use_edge_weight}
        self.inp = nn.Linear(N_FEATURES, hidden)
        self.gat = nn.ModuleList([GATLayer(hidden, use_edge_weight) for _ in range(layers)])
        self.out = nn.Linear(hidden, 1)

    def forward(self, lvl: LevelData) -> torch.Tensor:
        z = self.inp(lvl.x)
        for layer in self.gat:
            z = F.relu(layer(z, lvl.edge_index, lvl.logw))
        return self.out(z).squeeze(-1)  # logits of circuit membership

    @torch.no_grad()
    def score(self, sample: GraphSample) -> list[np.ndarray]:
        """Membership probability of every vertex at every hierarchy level."""
        self.eval()
        return [torch.sigmoid(self(lvl)).cpu().numpy() for lvl in sample.levels]

    def save(self, directory: Path, meta: dict | None = None) -> None:
        from safetensors.torch import save_file

        directory.mkdir(parents=True, exist_ok=True)
        save_file({k: v.cpu() for k, v in self.state_dict().items()}, str(directory / "gnn.safetensors"))
        save_json(directory / "gnn.json", {**self.cfg, **(meta or {})})

    @classmethod
    def load(cls, directory: Path) -> "CircuitGNN":
        from safetensors.torch import load_file

        meta = load_json(directory / "gnn.json")
        model = cls(meta["hidden"], meta["layers"], meta["use_edge_weight"])
        model.load_state_dict(load_file(str(directory / "gnn.safetensors")))
        model.meta = meta
        return model


def sample_loss(model: CircuitGNN, sample: GraphSample) -> torch.Tensor:
    losses = []
    for lvl in sample.levels:
        if lvl.y is None or lvl.x.shape[0] < 2:
            continue
        pos = lvl.y.sum()
        pw = ((len(lvl.y) - pos) / pos.clamp_min(1)).clamp(1.0, 20.0)
        losses.append(F.binary_cross_entropy_with_logits(model(lvl), lvl.y, pos_weight=pw))
    return torch.stack(losses).mean() if losses else torch.zeros((), requires_grad=True)


def leaf_auc(model: CircuitGNN, samples: list[GraphSample]) -> float:
    from sklearn.metrics import roc_auc_score

    aucs = []
    for s in samples:
        y = s.levels[0].y
        if y is None or y.min() == y.max():
            continue
        aucs.append(roc_auc_score(y.numpy(), model.score(s)[0]))
    return float(np.mean(aucs)) if aucs else float("nan")


def train_gnn(samples: list[GraphSample], gcfg, seed: int = 0) -> tuple[CircuitGNN, dict]:
    """Reptile meta-training over (model, task) circuit-discovery graphs."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = CircuitGNN(int(gcfg.hidden), int(gcfg.layers), bool(gcfg.use_edge_weight))
    samples = [s for s in samples if any(lvl.y is not None and lvl.y.sum() > 0 for lvl in s.levels)]
    history = []
    if not samples:
        return model, {"trained": False, "reason": "no labelled graphs"}
    iters = int(gcfg.meta_iters)
    for it in range(iters):
        task = rng.choice(samples)
        fast = copy.deepcopy(model)
        fast.train()
        opt = torch.optim.Adam(fast.parameters(), lr=float(gcfg.inner_lr))
        for _ in range(int(gcfg.inner_steps)):
            loss = sample_loss(fast, task)
            opt.zero_grad()
            loss.backward()
            opt.step()
        eps = float(gcfg.meta_lr) * (1 - it / iters)
        with torch.no_grad():
            for p, q in zip(model.parameters(), fast.parameters()):
                p.add_(eps * (q - p))
        if (it + 1) % max(1, iters // 10) == 0:
            with torch.no_grad():
                mean_loss = float(np.mean([float(sample_loss(model, s)) for s in samples]))
            history.append({"iter": it + 1, "loss": mean_loss})
    info = {"trained": True, "n_graphs": len(samples), "graphs": [s.name for s in samples],
            "train_auc": leaf_auc(model, samples), "history": history}
    log.info("gnn: meta-trained on %d graphs, train leaf AUROC %.3f", len(samples), info["train_auc"])
    return model, info


def heuristic_scores(sample: GraphSample) -> list[np.ndarray]:
    """Guidance without a GNN: rank of the summed attribution magnitude inside each level."""
    return [lvl.x[:, 24].numpy().copy() for lvl in sample.levels]
