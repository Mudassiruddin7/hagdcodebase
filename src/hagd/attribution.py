"""Attribution graph construction (paper Section III-C).

Vertices are transcoder features, edges connect features of adjacent instrumented layers
and carry the gradient-times-activation attribution of Eq. 5

        A_{i->j} = (d f_j / d f_i) * f_i ,

aggregated over positions and examples.  The Jacobian is taken through the real model
(attention and MLP of the blocks between the two layers), and the graph is built one
layer pair at a time so no more than one pair's gradient graph is ever alive
("computing attributions layer-by-layer rather than storing the complete gradient graph").

Vertex selection: |V| << |A| (Section II-A) is enforced by keeping, per layer, the features
with the largest attribution to the task objective y (the log-likelihood of the correct
answer), optionally extended by features the cross-layer prediction heads link to them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .data.tasks import TaskBatch, score_batch
from .models import StopForward
from .replacement import FeatureModel, FeatureStats
from .utils import load_json, log, save_json


@dataclass
class NodeAttribution:
    stats: FeatureStats
    attr: list[torch.Tensor]  # (m,) mean over examples of sum_pos dy/df * f      (signed)
    attr_abs: list[torch.Tensor]  # (m,) mean over examples of |sum_pos dy/df * f|
    ie: list[torch.Tensor]  # (m,) attribution-patching estimate with a mean-ablation baseline
    n_examples: int
    seconds: float


@dataclass
class AttributionGraph:
    model: str
    task: str
    layers: list[int]  # block index of each transcoder layer
    m: int  # dictionary size per layer
    node_layer: np.ndarray  # (n,) transcoder-layer index
    node_feat: np.ndarray  # (n,) feature id inside its dictionary
    node_attr: np.ndarray  # (n,) signed attribution to y
    node_attr_abs: np.ndarray
    node_ie: np.ndarray
    node_act: np.ndarray  # (n,) mean activation
    node_freq: np.ndarray
    node_xl_in: np.ndarray  # (n,) prediction-head weight mass from upstream candidates
    node_xl_out: np.ndarray  # (n,) prediction-head weight mass to downstream candidates
    edge_src: np.ndarray  # (E,) node index (layer l)
    edge_dst: np.ndarray  # (E,) node index (layer l + 1)
    edge_w: np.ndarray  # (E,) mean |A_{i->j}|   (attribution magnitude)
    edge_signed: np.ndarray  # (E,) mean A_{i->j}
    anchors: np.ndarray  # (n_seq, n) per-sequence max activation of every node (feature alignment, Eq. 11)
    info: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(len(self.node_layer))

    @property
    def n_edges(self) -> int:
        return int(len(self.edge_src))

    @property
    def n_dictionary(self) -> int:
        return int(len(self.layers) * self.m)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {k: v for k, v in self.__dict__.items() if isinstance(v, np.ndarray)}
        arrays["anchors"] = arrays["anchors"].astype(np.float16)
        np.savez_compressed(directory / "graph.npz", **arrays)
        save_json(directory / "graph.json", {"model": self.model, "task": self.task, "layers": self.layers,
                                             "m": self.m, "n": self.n, "n_edges": self.n_edges, "info": self.info})

    @classmethod
    def load(cls, directory: Path) -> "AttributionGraph":
        meta = load_json(directory / "graph.json")
        z = np.load(directory / "graph.npz")
        arrays = {k: z[k] for k in z.files}
        arrays["anchors"] = arrays["anchors"].astype(np.float32)
        return cls(model=meta["model"], task=meta["task"], layers=meta["layers"], m=meta["m"], info=meta["info"], **arrays)

    def induced_edges(self, nodes: np.ndarray) -> np.ndarray:
        """Indices of the graph's edges whose two endpoints are both in ``nodes``."""
        inside = np.zeros(self.n, dtype=bool)
        inside[np.asarray(nodes, dtype=np.int64)] = True
        return np.nonzero(inside[self.edge_src] & inside[self.edge_dst])[0]


# --------------------------------------------------------------------------------------
# vertices
# --------------------------------------------------------------------------------------


def attribute_nodes(fm: FeatureModel, batches: list[TaskBatch]) -> NodeAttribution:
    """One forward/backward pass per batch: attribution of every feature of every layer to y."""
    lm, tc = fm.lm, fm.tc
    n_l, dev = fm.n_layers, lm.device
    zeros = lambda size: [torch.zeros(size, device=dev) for _ in range(n_l)]  # noqa: E731
    a_sum, a_abs, g_sum, f_sum, fired = zeros(tc.m), zeros(tc.m), zeros(tc.m), zeros(tc.m), zeros(tc.m)
    e_sum = zeros(tc.d)
    n_tok, n_ex = 0, 0
    t0 = time.time()
    for b in batches:
        b = b.to(dev)
        store: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        def make(li: int):
            def fn(h: torch.Tensor) -> torch.Tensor:
                z = torch.zeros_like(h, requires_grad=True)
                store[li] = (z, h.detach())
                return h + z

            return fn

        with torch.enable_grad(), lm.adapter_on(fm.adapter_on):
            logits = lm.run(b.input_ids, b.attention_mask, {layer: make(li) for li, layer in enumerate(fm.layers)})
            y = -score_batch(logits, b)["loss"].sum()
            grads = torch.autograd.grad(y, [store[li][0] for li in range(n_l)])
        tm = fm.sites(b)
        tmf = tm.unsqueeze(-1).float()
        with torch.no_grad():
            for li in range(n_l):
                h = store[li][1].float()
                f = tc.encode(h, li)
                gd = (grads[li].float() @ tc.W_dec[li]) * tc.scale[li] * tmf  # dy/df at every position
                per_seq = (gd * f).sum(1)
                per_ex = per_seq.view(b.n_examples, b.n_choices, -1).sum(1)
                a_sum[li] += per_ex.sum(0)
                a_abs[li] += per_ex.abs().sum(0)
                g_sum[li] += gd.sum((0, 1))
                f_sum[li] += (f * tmf).sum((0, 1))
                fired[li] += ((f > 0).float() * tmf).sum((0, 1))
                e_sum[li] += ((h - tc.decode(f, li)) * tmf).sum((0, 1))
        n_tok += int(tm.sum())
        n_ex += b.n_examples
        del store, grads, logits
    mean = [x / max(n_tok, 1) for x in f_sum]
    stats = FeatureStats(mean, [x / max(n_tok, 1) for x in fired], [x / max(n_tok, 1) for x in e_sum], n_tok)
    ie = [(a_sum[li] - mean[li] * g_sum[li]) / max(n_ex, 1) for li in range(n_l)]
    return NodeAttribution(stats, [x / max(n_ex, 1) for x in a_sum], [x / max(n_ex, 1) for x in a_abs], ie,
                           n_ex, time.time() - t0)


def select_candidates(fm: FeatureModel, na: NodeAttribution, nodes_per_layer: int, max_nodes: int,
                      cross_layer_expand: int = 0) -> list[torch.Tensor]:
    """Per-layer candidate vertices: strongest attributions, plus prediction-head neighbours."""
    tc = fm.tc
    per_layer = max(1, min(int(nodes_per_layer), int(max_nodes) // fm.n_layers))
    cands = []
    for li in range(fm.n_layers):
        score = na.attr_abs[li]
        k = int(min(per_layer, int((score > 0).sum())))
        cands.append(score.topk(k).indices if k > 0 else score.new_zeros(0, dtype=torch.long))
    if cross_layer_expand > 0 and tc.cross_layer:
        with torch.no_grad():
            for li in range(fm.n_layers - 1):
                down = cands[li + 1]
                if len(down) == 0:
                    continue
                w = tc.pred_weight_rows(li, down).abs()  # (Nd, m)
                score = (w * na.stats.mean[li]).sum(0)
                score[cands[li]] = -1
                score[na.stats.freq[li] <= 0] = -1
                extra = score.topk(min(int(cross_layer_expand), tc.m)).indices
                extra = extra[score[extra] > 0]
                cands[li] = torch.cat([cands[li], extra])
    return cands


# --------------------------------------------------------------------------------------
# edges (Eq. 5)
# --------------------------------------------------------------------------------------


def _jacobian_times_act(out: torch.Tensor, leaf: torch.Tensor, cols: torch.Tensor, chunk: int,
                        site: torch.Tensor) -> tuple[torch.Tensor, int]:
    """For each downstream feature j in ``cols``: sum over the sites of d out[j] / d leaf * leaf.

    ``leaf`` is (B, S, Nu) and ``site`` a (B, S, 1) 0/1 weight; the result is (len(cols), B, Nu).
    Positions are summed inside the chunk loop, so memory does not grow with sequence length.
    ``chunk`` backward passes are vectorised at a time; it is halved on out-of-memory and
    drops to 0 (plain sequential backward passes) if an operation has no vmap rule.
    Returns the result and the chunk size that worked, for the caller to reuse.
    """
    from .utils import free_memory, is_oom

    act = leaf.detach() * site
    while chunk > 0:
        rows = []
        try:
            for start in range(0, len(cols), chunk):
                sel = cols[start:start + chunk]
                basis = torch.zeros(len(sel), out.shape[0], device=out.device, dtype=out.dtype)
                basis[torch.arange(len(sel)), sel] = 1.0
                (g,) = torch.autograd.grad(out, leaf, grad_outputs=basis, is_grads_batched=True, retain_graph=True)
                rows.append((g * act).sum(-2))
                del g
            return torch.cat(rows), chunk
        except Exception as e:  # noqa: BLE001
            del rows
            if is_oom(e):
                free_memory()
                chunk //= 2
                log.info("edge attribution: out of memory, vectorising %d backward passes at a time", chunk)
            else:
                log.info("edge attribution: batched Jacobian unavailable (%s); using sequential backward passes", str(e)[:120])
                chunk = 0
    rows = []
    for j in cols.tolist():
        (g,) = torch.autograd.grad(out[j], leaf, retain_graph=True, allow_unused=True)
        g = torch.zeros_like(leaf) if g is None else g
        rows.append((g * act).sum(-2).unsqueeze(0))
    return torch.cat(rows), 0


def attribute_edges(fm: FeatureModel, batches: list[TaskBatch], cands: list[torch.Tensor], jac_chunk: int = 32,
                    batched: bool = True) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], dict]:
    """Eq. 5 between the candidates of every adjacent layer pair.

    Returns per pair ``W_abs[i, j] = E_x |A_{i->j}|`` and ``W_signed[i, j] = E_x A_{i->j}``,
    the per-sequence max activation of every candidate (one (n_seq, N_l) tensor per layer),
    and timing information.
    """
    lm, tc = fm.lm, fm.tc
    dev = lm.device
    w_abs, w_signed = [], []
    anchors: list[list[torch.Tensor]] = [[] for _ in range(fm.n_layers)]
    n_backward, pairs_evaluated = 0, 0
    t0 = time.time()
    chunk = int(jac_chunk) if batched else 0
    for li in range(fm.n_layers - 1):
        up_ids, down_ids = cands[li], cands[li + 1]
        wa = torch.zeros(len(up_ids), len(down_ids), device=dev)
        ws = torch.zeros_like(wa)
        n_ex = 0
        if len(up_ids) == 0 or len(down_ids) == 0:
            w_abs.append(wa)
            w_signed.append(ws)
            continue
        d_up = tc.W_dec[li][:, up_ids]  # (d, Nu)
        for b in batches:
            b = b.to(dev)
            tm = fm.sites(b)
            tmf = tm.unsqueeze(-1).float()
            hold: dict[str, torch.Tensor] = {}

            def up(h: torch.Tensor) -> torch.Tensor:
                x = h.float()
                f = tc.encode(x, li)[..., up_ids].detach()
                leaf = f.clone().requires_grad_(True)
                hold["leaf"] = leaf
                # numerically the identity, but makes h depend on the candidate features
                return x + ((leaf - f) @ d_up.t()) * tc.scale[li]

            def down(h: torch.Tensor) -> None:
                hold["h"] = h
                raise StopForward

            with torch.enable_grad(), lm.adapter_on(fm.adapter_on):
                lm.run(b.input_ids, b.attention_mask, {fm.layers[li]: up, fm.layers[li + 1]: down})
                leaf = hold["leaf"]
                fd = tc.encode(hold["h"].float(), li + 1)[..., down_ids] * tmf  # (B, S, Nd)
                out = fd.sum((0, 1))
                active = (out.detach() > 0).nonzero(as_tuple=True)[0]
                if len(active) > 0:
                    a, chunk = _jacobian_times_act(out, leaf, active, chunk, tmf)  # (n_active, N_seq, Nu)
                    n_backward += len(active)
                    a = a.view(len(active), b.n_examples, b.n_choices, -1).sum(2)  # per example
                    ws[:, active] += a.sum(1).t()
                    wa[:, active] += a.abs().sum(1).t()
            with torch.no_grad():
                neg = torch.finfo(leaf.dtype).min
                anchors[li].append(torch.where(tm.unsqueeze(-1), leaf.detach(), torch.full_like(leaf, neg)).amax(1).clamp_min(0).cpu())
                if li == fm.n_layers - 2:
                    anchors[li + 1].append(fd.detach().amax(1).cpu())
            n_ex += b.n_examples
            pairs_evaluated += len(up_ids) * len(down_ids)
            del hold, out, fd
        w_abs.append(wa / max(n_ex, 1))
        w_signed.append(ws / max(n_ex, 1))
    info = {"edge_seconds": time.time() - t0, "backward_passes": n_backward, "pairs_evaluated": pairs_evaluated,
            "jacobian_chunk": chunk}
    n_seq = max((sum(x.shape[0] for x in a) for a in anchors), default=0)
    pooled = [torch.cat(a).float() if a else torch.zeros(n_seq, len(cands[i])) for i, a in enumerate(anchors)]
    return w_abs, w_signed, pooled, info


def sparsify(w_abs: torch.Tensor, topk: int, min_weight: float) -> torch.Tensor:
    """Boolean (Nu, Nd) mask: strongest ``topk`` incoming edges per target and outgoing per source."""
    if w_abs.numel() == 0:
        return torch.zeros_like(w_abs, dtype=torch.bool)
    keep = torch.zeros_like(w_abs, dtype=torch.bool)
    k_in = min(topk, w_abs.shape[0])
    keep.scatter_(0, w_abs.topk(k_in, dim=0).indices, True)
    k_out = min(topk, w_abs.shape[1])
    keep.scatter_(1, w_abs.topk(k_out, dim=1).indices, True)
    return keep & (w_abs > min_weight)


def build_graph(fm: FeatureModel, batches: list[TaskBatch], acfg, model: str, task: str,
                na: NodeAttribution | None = None) -> tuple[AttributionGraph, NodeAttribution]:
    """Full attribution-graph construction for one (model, task)."""
    tc = fm.tc
    na = na or attribute_nodes(fm, batches)
    cands = select_candidates(fm, na, int(acfg.nodes_per_layer), int(acfg.max_nodes),
                              int(acfg.cross_layer_expand) if tc.cross_layer else 0)
    w_abs, w_signed, anchors, info = attribute_edges(fm, batches, cands, int(acfg.jac_chunk), bool(acfg.batched_grads))

    offsets = np.cumsum([0] + [len(c) for c in cands])
    n = int(offsets[-1])
    node_layer = np.concatenate([np.full(len(c), li, dtype=np.int64) for li, c in enumerate(cands)])
    node_feat = np.concatenate([c.cpu().numpy().astype(np.int64) for c in cands])
    pick = lambda per_layer: np.concatenate([per_layer[li][c].cpu().numpy() for li, c in enumerate(cands)]).astype(np.float32)  # noqa: E731

    xl_in, xl_out = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
    if tc.cross_layer:
        with torch.no_grad():
            for li in range(fm.n_layers - 1):
                if len(cands[li]) and len(cands[li + 1]):
                    w = tc.pred_weight_rows(li, cands[li + 1])[:, cands[li]].abs()  # (Nd, Nu)
                    xl_in[offsets[li + 1]:offsets[li + 2]] = w.sum(1).cpu().numpy()
                    xl_out[offsets[li]:offsets[li + 1]] = w.sum(0).cpu().numpy()

    global_max = max((float(w.max()) for w in w_abs if w.numel()), default=0.0)
    src, dst, wt, sg = [], [], [], []
    for li, (wa, ws) in enumerate(zip(w_abs, w_signed)):
        keep = sparsify(wa, int(acfg.edge_topk), float(acfg.edge_min_frac) * global_max)
        i, j = keep.nonzero(as_tuple=True)
        src.append((i + int(offsets[li])).cpu().numpy())
        dst.append((j + int(offsets[li + 1])).cpu().numpy())
        wt.append(wa[i, j].cpu().numpy())
        sg.append(ws[i, j].cpu().numpy())
    cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dtype=dt)  # noqa: E731
    anchor_mat = torch.cat([a.float() for a in anchors], dim=1).numpy() if n else np.zeros((0, 0), dtype=np.float32)

    info.update({"edge_threshold": float(acfg.edge_min_frac) * global_max,
                 "node_seconds": na.seconds, "n_examples": na.n_examples, "n_dictionary": fm.n_layers * tc.m,
                 "n_candidates": n, "tokens": na.stats.n_tokens,
                 "features_per_second": fm.n_layers * tc.m * na.n_examples / max(na.seconds, 1e-9),
                 "edges_per_second": info["pairs_evaluated"] / max(info["edge_seconds"], 1e-9)})
    graph = AttributionGraph(
        model=model, task=task, layers=list(fm.layers), m=tc.m, node_layer=node_layer, node_feat=node_feat,
        node_attr=pick(na.attr), node_attr_abs=pick(na.attr_abs), node_ie=pick(na.ie), node_act=pick(na.stats.mean),
        node_freq=pick(na.stats.freq), node_xl_in=xl_in, node_xl_out=xl_out,
        edge_src=cat(src, np.int64), edge_dst=cat(dst, np.int64), edge_w=cat(wt, np.float32),
        edge_signed=cat(sg, np.float32), anchors=anchor_mat, info=info,
    )
    log.info("graph %s/%s: %d vertices (of %d features), %d edges | nodes %.1fs edges %.1fs",
             model, task, graph.n, graph.n_dictionary, graph.n_edges, na.seconds, info["edge_seconds"])
    return graph, na
