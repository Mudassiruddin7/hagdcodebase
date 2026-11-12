"""Feature-level view of a language model, and causal interventions on it.

``FeatureModel`` couples a language model with its transcoders.  ``Intervention``
implements the two model variants the validation protocol needs (paper Section III-F):

* ``restrict``  – M(x; C): only the circuit's features keep their values, every other
  feature is ablated.  Used for sufficiency (Eq. 10) and necessity (Eq. 9).
* ``knockout``  – the circuit's features are ablated and everything else is left alone.
  Used for the "intervention effect" metric (Section IV-C).

Ablation replaces a feature by its mean activation on the task (or by zero).  The part of
a hidden state that the transcoder does not reconstruct (the "dark matter" of Section VI-C)
is handled by ``error``: ``clean`` keeps it as computed, ``mean`` replaces it by its
average, ``zero`` drops it.  BOS and padding positions are never modified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data.tasks import TaskBatch, score_batch
from .models import LM
from .transcoder import CrossLayerTranscoder


@dataclass
class FeatureStats:
    """Per-layer statistics of feature activations on one task (content tokens only)."""

    mean: list[torch.Tensor]  # (m,) mean activation
    freq: list[torch.Tensor]  # (m,) firing frequency
    err_mean: list[torch.Tensor]  # (d,) mean reconstruction residual
    n_tokens: int = 0

    def to(self, device) -> "FeatureStats":
        return FeatureStats([t.to(device) for t in self.mean], [t.to(device) for t in self.freq],
                            [t.to(device) for t in self.err_mean], self.n_tokens)

    def save(self, path) -> None:
        np.savez_compressed(path, mean=torch.stack(self.mean).cpu().numpy(), freq=torch.stack(self.freq).cpu().numpy(),
                            err_mean=torch.stack(self.err_mean).cpu().numpy(), n_tokens=self.n_tokens)

    @classmethod
    def load(cls, path, device="cpu") -> "FeatureStats":
        z = np.load(path)
        as_list = lambda a: [torch.tensor(r, device=device) for r in a]  # noqa: E731
        return cls(as_list(z["mean"]), as_list(z["freq"]), as_list(z["err_mean"]), int(z["n_tokens"]))


class Intervention:
    def __init__(self, fm: "FeatureModel", stats: FeatureStats, nodes: dict[int, torch.Tensor] | None,
                 mode: str = "restrict", error: str = "clean", ablation: str = "mean") -> None:
        assert mode in ("restrict", "knockout") and error in ("clean", "mean", "zero") and ablation in ("mean", "zero")
        self.fm, self.stats, self.mode, self.error, self.ablation = fm, stats, mode, error, ablation
        dev = fm.lm.device
        self.masks = []
        for li in range(fm.n_layers):
            mask = torch.zeros(fm.tc.m, dtype=torch.bool, device=dev)
            ids = (nodes or {}).get(li)
            if ids is not None and len(ids) > 0:
                mask[torch.as_tensor(ids, device=dev, dtype=torch.long)] = True
            self.masks.append(mask)

    def apply(self, h: torch.Tensor, li: int, token_mask: torch.Tensor) -> torch.Tensor:
        tc, st = self.fm.tc, self.stats
        x = h.float()
        f = tc.encode(x, li)
        base = st.mean[li] if self.ablation == "mean" else torch.zeros_like(st.mean[li])
        in_circuit = self.masks[li]
        if self.mode == "restrict":
            f_new = torch.where(in_circuit, f, base)
        else:
            f_new = torch.where(in_circuit, base.expand_as(f), f)
        if self.mode == "knockout" or self.error == "clean":
            x_new = x + tc.decode_delta(f_new - f, li)
        elif self.error == "mean":
            x_new = tc.decode(f_new, li) + st.err_mean[li]
        else:
            x_new = tc.decode(f_new, li)
        return torch.where(token_mask.unsqueeze(-1), x_new, x)


class FeatureModel:
    def __init__(self, lm: LM, tc: CrossLayerTranscoder, adapter_on: bool = False) -> None:
        self.lm, self.tc = lm, tc
        self.layers = tc.layers  # block index of each transcoder layer
        self.n_layers = len(tc.layers)
        self.adapter_on = adapter_on

    # ------------------------------------------------------------------ forward
    def forward(self, batch: TaskBatch, iv: Intervention | None = None) -> torch.Tensor:
        fns = {}
        if iv is not None:
            tm = batch.token_mask()
            for li, layer in enumerate(self.layers):
                fns[layer] = (lambda h, li=li: iv.apply(h, li, tm))
        with self.lm.adapter_on(self.adapter_on):
            return self.lm.run(batch.input_ids, batch.attention_mask, fns)

    @torch.no_grad()
    def features(self, batch: TaskBatch) -> list[torch.Tensor]:
        """Clean feature activations (B, S, m) at every transcoder layer."""
        with self.lm.adapter_on(self.adapter_on):
            hs = self.lm.hidden_states(batch.input_ids, batch.attention_mask, self.layers)
        return [self.tc.encode(h, li) for li, h in enumerate(hs)]

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate(self, batches: list[TaskBatch], iv: Intervention | None = None,
                 ref: list[torch.Tensor] | None = None, keep_dist: bool = False) -> dict:
        """Accuracy / loss over ``batches``; with ``ref`` also KL(full || intervened) of the output."""
        correct, losses, kls, dists = [], [], [], []
        for i, b in enumerate(batches):
            b = b.to(self.lm.device)
            s = score_batch(self.forward(b, iv), b)
            correct.append(s["correct"].float().cpu())
            losses.append(s["loss"].float().cpu())
            if keep_dist:
                dists.append(s["dist"].cpu())
            if ref is not None:
                p = ref[i].to(s["dist"].device)
                kls.append((p.exp() * (p - s["dist"])).sum(-1).clamp_min(0).cpu())
        correct_all, loss_all = torch.cat(correct).numpy(), torch.cat(losses).numpy()
        out = {"acc": float(correct_all.mean()), "loss": float(loss_all.mean()),
               "correct": correct_all, "losses": loss_all, "n": int(len(correct_all))}
        if ref is not None:
            out["kl"] = float(torch.cat(kls).mean())
        if keep_dist:
            out["dist"] = dists
        return out

    def reference(self, batches: list[TaskBatch]) -> dict:
        """Full-model run whose output distributions serve as the KL reference."""
        return self.evaluate(batches, None, keep_dist=True)

    def intervention(self, stats: FeatureStats, nodes: dict[int, torch.Tensor] | None, **kw) -> Intervention:
        return Intervention(self, stats, nodes, **kw)


def nodes_to_dict(layer_idx: np.ndarray, feat_idx: np.ndarray, select: np.ndarray | list[int] | None = None) -> dict[int, torch.Tensor]:
    """Group graph nodes (given as parallel layer / feature arrays) by transcoder layer."""
    if select is not None:
        select = np.asarray(select, dtype=np.int64)
        layer_idx, feat_idx = layer_idx[select], feat_idx[select]
    out: dict[int, torch.Tensor] = {}
    for li in np.unique(layer_idx):
        out[int(li)] = torch.as_tensor(feat_idx[layer_idx == li], dtype=torch.long)
    return out
