"""Cross-layer transcoder (paper Section III-A/B, Eq. 1, 3, 4).

For every instrumented layer ``l`` there is

* an encoder      f_l = TopK(ReLU(W^E_l h_l + b^E_l), k)                    (Eq. 1)
* a decoder       h_hat_l = D_l f_l
* a prediction head  f_hat_{l+1} = sigmoid(W^P_{l->l+1} f_l + b^P_{l->l+1})  (Eq. 3)

trained jointly with
    L = sum_l ||h_l - D_l f_l||^2 + lambda_1 ||f_{l+1} - f_hat_{l+1}||^2 + lambda_2 ||f_l||_1   (Eq. 4)

Implementation notes (not specified in the paper):

* Activations are standardised with fixed, non-learned statistics, h~ = (h - mu_l) / s_l with
  s_l chosen so that E||h~||^2 = 1.  All equations hold in that space; the reconstruction
  term then *is* the fraction of unexplained variance, and feature activations fall inside
  the (0, 1) range the sigmoid head of Eq. 3 can represent.
* Decoder columns are kept at unit norm (otherwise the L1 term is trivially minimised).
* W^P is factorised as P_out P_in with rank ``pred_rank``; a dense m x m matrix per layer
  is infeasible for the dictionary sizes of billion-parameter models.
* An optional AuxK term (TopK-SAE reference [7]) revives dead features.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .utils import load_json, save_json


class CrossLayerTranscoder(nn.Module):
    def __init__(self, layers: list[int], d_model: int, expansion: int = 8, k: int = 32,
                 pred_rank: int | None = 512, cross_layer: bool = True, init_std: float = 0.1) -> None:
        super().__init__()
        self.layers = [int(x) for x in layers]
        self.d = int(d_model)
        self.m = int(expansion * d_model)
        self.k = min(int(k), self.m)
        self.expansion = int(expansion)
        self.cross_layer = bool(cross_layer) and len(self.layers) > 1
        self.pred_rank = None if (pred_rank is None or pred_rank >= self.m) else int(pred_rank)
        n = len(self.layers)

        self.W_enc = nn.ParameterList()
        self.b_enc = nn.ParameterList()
        self.W_dec = nn.ParameterList()
        for _ in range(n):
            # Algorithm 1, steps 1-2: W^E ~ N(0, 0.01); decoder = transpose of the encoder.
            w = torch.randn(self.m, self.d) * init_std
            w = w / w.norm(dim=1, keepdim=True)
            self.W_enc.append(nn.Parameter(w))
            self.b_enc.append(nn.Parameter(torch.zeros(self.m)))
            self.W_dec.append(nn.Parameter(w.t().clone()))

        self.P_in = nn.ParameterList()
        self.P_out = nn.ParameterList()
        self.b_pred = nn.ParameterList()
        if self.cross_layer:
            for _ in range(n - 1):  # Algorithm 1, step 3: random prediction heads
                if self.pred_rank:
                    self.P_in.append(nn.Parameter(torch.randn(self.pred_rank, self.m) / self.m**0.5))
                    self.P_out.append(nn.Parameter(torch.randn(self.m, self.pred_rank) * 0.01))
                else:
                    self.P_in.append(nn.Parameter(torch.randn(self.m, self.m) * 0.01 / self.m**0.5))
                    self.P_out.append(nn.Parameter(torch.empty(0)))
                # sigmoid(-6) ~ 0: the head starts by predicting "inactive" for every feature
                self.b_pred.append(nn.Parameter(torch.full((self.m,), -6.0)))

        self.register_buffer("mu", torch.zeros(n, self.d))
        self.register_buffer("scale", torch.ones(n))
        self.register_buffer("tokens_since_fired", torch.zeros(n, self.m))

    # ------------------------------------------------------------------ core maps
    def standardize(self, h: torch.Tensor, li: int) -> torch.Tensor:
        return (h - self.mu[li]) / self.scale[li]

    def pre_acts(self, h: torch.Tensor, li: int) -> torch.Tensor:
        return self.standardize(h, li) @ self.W_enc[li].t() + self.b_enc[li]

    def topk(self, pre: torch.Tensor, k: int | None = None) -> torch.Tensor:
        act = F.relu(pre)
        vals, idx = act.topk(k or self.k, dim=-1)
        return torch.zeros_like(act).scatter_(-1, idx, vals)

    def encode(self, h: torch.Tensor, li: int) -> torch.Tensor:
        """Eq. 1.  ``h`` is a raw hidden state (..., d); returns dense sparse-coded features (..., m)."""
        return self.topk(self.pre_acts(h, li))

    def decode(self, f: torch.Tensor, li: int) -> torch.Tensor:
        """h_hat = D f, mapped back to the model's activation space."""
        return (f @ self.W_dec[li].t()) * self.scale[li] + self.mu[li]

    def decode_delta(self, df: torch.Tensor, li: int) -> torch.Tensor:
        """Change of the hidden state caused by a change ``df`` of the feature vector."""
        return (df @ self.W_dec[li].t()) * self.scale[li]

    def predict(self, f: torch.Tensor, li: int) -> torch.Tensor:
        """Eq. 3: predicted features of layer index ``li + 1`` from those of ``li``."""
        if self.pred_rank:
            z = (f @ self.P_in[li].t()) @ self.P_out[li].t()
        else:
            z = f @ self.P_in[li].t()
        return torch.sigmoid(z + self.b_pred[li])

    def pred_weight_rows(self, li: int, rows: torch.Tensor) -> torch.Tensor:
        """Rows of W^P_{li -> li+1} (downstream feature x all upstream features)."""
        if self.pred_rank:
            return self.P_out[li][rows] @ self.P_in[li]
        return self.P_in[li][rows]

    # ------------------------------------------------------------------ training
    @torch.no_grad()
    def fit_normalization(self, hs: list[torch.Tensor]) -> None:
        for li, h in enumerate(hs):
            mu = h.mean(0)
            self.mu[li] = mu
            self.scale[li] = (h - mu).pow(2).sum(-1).mean().sqrt().clamp_min(1e-6)

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        for w in self.W_dec:
            w.div_(w.norm(dim=0, keepdim=True).clamp_min(1e-8))

    def loss(self, hs: list[torch.Tensor], lambda_pred: float, lambda_l1: float, aux_k_coef: float = 0.0,
             dead_tokens: int = 1_000_000, detach_pred_target: bool = False) -> tuple[torch.Tensor, dict]:
        """Eq. 4 on a list of (N, d) hidden states, one tensor per instrumented layer."""
        feats, stats = [], {"fvu": [], "l1": [], "pred": [], "dead": []}
        total = hs[0].new_zeros(())
        for li, h in enumerate(hs):
            x = self.standardize(h, li)
            pre = x @ self.W_enc[li].t() + self.b_enc[li]
            f = self.topk(pre)
            recon = f @ self.W_dec[li].t()
            err = x - recon
            rec = err.pow(2).sum(-1).mean()
            l1 = f.abs().sum(-1).mean()
            total = total + rec + lambda_l1 * l1
            with torch.no_grad():
                fired = (f > 0).any(0)
                self.tokens_since_fired[li] += h.shape[0]
                self.tokens_since_fired[li][fired] = 0
                dead = self.tokens_since_fired[li] > dead_tokens
                stats["fvu"].append(float(rec / x.pow(2).sum(-1).mean().clamp_min(1e-8)))
                stats["l1"].append(float(l1))
                stats["dead"].append(float(dead.float().mean()))
            if aux_k_coef > 0 and dead.any():
                k_aux = int(min(max(self.d // 2, 1), int(dead.sum())))
                pre_dead = torch.where(dead, F.relu(pre), torch.zeros_like(pre))
                vals, idx = pre_dead.topk(k_aux, dim=-1)
                f_aux = torch.zeros_like(pre).scatter_(-1, idx, vals)
                e = err.detach()
                aux = (e - f_aux @ self.W_dec[li].t()).pow(2).sum(-1).mean() / e.pow(2).sum(-1).mean().clamp_min(1e-8)
                total = total + aux_k_coef * aux
            feats.append(f)
        if self.cross_layer:
            for li in range(len(hs) - 1):
                target = feats[li + 1].detach() if detach_pred_target else feats[li + 1]
                pred = (target - self.predict(feats[li], li)).pow(2).sum(-1).mean()
                total = total + lambda_pred * pred
                stats["pred"].append(float(pred.detach()))
        return total, stats

    @torch.no_grad()
    def reconstruction_stats(self, hs: list[torch.Tensor]) -> dict:
        """Per-layer fraction of variance unexplained and prediction-head quality."""
        out = {"fvu": [], "pred_r2": [], "l0": []}
        feats = []
        for li, h in enumerate(hs):
            f = self.encode(h, li)
            feats.append(f)
            resid = (h - self.decode(f, li)).pow(2).sum()
            total = (h - h.mean(0)).pow(2).sum().clamp_min(1e-8)
            out["fvu"].append(float(resid / total))
            out["l0"].append(float((f > 0).float().sum(-1).mean()))
        if self.cross_layer:
            for li in range(len(hs) - 1):
                t = feats[li + 1]
                sse = (t - self.predict(feats[li], li)).pow(2).sum()
                sst = (t - t.mean(0)).pow(2).sum().clamp_min(1e-8)
                out["pred_r2"].append(float(1 - sse / sst))
        return out

    # ------------------------------------------------------------------ IO
    def config(self) -> dict:
        return {"layers": self.layers, "d_model": self.d, "expansion": self.expansion, "k": self.k,
                "pred_rank": self.pred_rank, "cross_layer": self.cross_layer}

    def save(self, directory: str | Path, extra: dict | None = None) -> None:
        from safetensors.torch import save_file

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()},
                  str(directory / "transcoder.safetensors"))
        save_json(directory / "transcoder.json", {**self.config(), **(extra or {})})

    @classmethod
    def load(cls, directory: str | Path, device: torch.device | str = "cpu") -> "CrossLayerTranscoder":
        from safetensors.torch import load_file

        directory = Path(directory)
        meta = load_json(directory / "transcoder.json")
        tc = cls(meta["layers"], meta["d_model"], meta["expansion"], meta["k"], meta["pred_rank"], meta["cross_layer"])
        tc.load_state_dict(load_file(str(directory / "transcoder.safetensors")))
        tc.meta = meta
        return tc.to(device)
