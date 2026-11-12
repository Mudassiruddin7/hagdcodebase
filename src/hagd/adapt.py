"""Task adaptation with LoRA.

The paper evaluates behavioural preservation as an accuracy ratio (Eq. 10) on modular
arithmetic, parity and sorting.  Off-the-shelf GPT-2 / Pythia / Llama checkpoints score at
or near chance on those tasks, which leaves the ratio undefined, and the paper does not say
how the models acquire the behaviour.  This module teaches the algorithmic tasks with a
low-rank adapter that is switched on only while those tasks run; language tasks always use
the untouched pretrained weights.  Set ``adapt.mode: none`` to study the raw checkpoints.

LoRA is implemented here (no ``peft`` dependency) so it behaves identically across
architectures and with quantised base layers.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path

import torch
from torch import nn

from .data.tasks import SPLITS, Task, encode_examples, score_batch
from .utils import load_json, log, save_json


def _dims(base: nn.Module) -> tuple[int, int] | None:
    if isinstance(base, nn.Linear):
        return base.in_features, base.out_features
    if type(base).__name__ == "Conv1D" and hasattr(base, "weight"):  # GPT-2 style linear
        return base.weight.shape[0], base.weight.shape[1]
    return None


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Module, rank: int, alpha: float, adapter: "LoRAAdapter") -> None:
        super().__init__()
        in_f, out_f = _dims(base)
        self.base = base
        self.A = nn.Parameter(torch.empty(rank, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scaling = alpha / rank
        self._adapter = [adapter]  # list: keep the adapter out of the module tree

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        out = self.base(x, *args, **kwargs)
        if self._adapter[0].enabled:
            delta = (x.to(self.A.dtype) @ self.A.t()) @ self.B.t()
            out = out + (delta * self.scaling).to(out.dtype)
        return out


class LoRAAdapter:
    """All LoRA modules of one model plus a global on/off switch."""

    def __init__(self) -> None:
        self.enabled = False
        self.modules: dict[str, LoRALinear] = {}

    def parameters(self) -> list[nn.Parameter]:
        return [p for m in self.modules.values() for p in (m.A, m.B)]

    def state_dict(self) -> dict[str, torch.Tensor]:
        out = {}
        for name, m in self.modules.items():
            out[f"{name}.A"] = m.A.detach().cpu()
            out[f"{name}.B"] = m.B.detach().cpu()
        return out

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, m in self.modules.items():
                m.A.copy_(state[f"{name}.A"])
                m.B.copy_(state[f"{name}.B"])

    def save(self, directory: Path, meta: dict) -> None:
        from safetensors.torch import save_file

        directory.mkdir(parents=True, exist_ok=True)
        save_file(self.state_dict(), str(directory / "adapter.safetensors"))
        save_json(directory / "adapter.json", meta)

    def load(self, directory: Path) -> dict:
        from safetensors.torch import load_file

        self.load_state_dict(load_file(str(directory / "adapter.safetensors")))
        return load_json(directory / "adapter.json")


def attach_lora(lm, rank: int, alpha: float) -> LoRAAdapter:
    """Wrap every linear map inside the transformer blocks with a LoRA branch."""
    if lm.adapter is not None:
        return lm.adapter
    adapter = LoRAAdapter()
    targets = []
    for bi, block in enumerate(lm.blocks):
        for name, module in block.named_modules():
            if _dims(module) is not None:
                targets.append((bi, name, module))
    for bi, name, module in targets:
        parent = lm.blocks[bi]
        *path, leaf = name.split(".")
        for part in path:
            parent = getattr(parent, part)
        wrapped = LoRALinear(module, rank, alpha, adapter).to(lm.device)
        wrapped.base.requires_grad_(False)
        setattr(parent, leaf, wrapped)
        adapter.modules[f"block{bi}.{name}"] = wrapped
    lm.adapter = adapter
    log.info("LoRA: wrapped %d linear maps (rank %d, %.2fM trainable parameters)",
             len(targets), rank, sum(p.numel() for p in adapter.parameters()) / 1e6)
    return adapter


@torch.no_grad()
def plain_accuracy(lm, batches) -> tuple[float, float]:
    """Accuracy and mean loss of the unmodified model on task batches."""
    correct, loss, n = 0.0, 0.0, 0
    for b in batches:
        b = b.to(lm.device)
        s = score_batch(lm.logits(b.input_ids, b.attention_mask), b)
        correct += float(s["correct"].float().sum())
        loss += float(s["loss"].sum())
        n += b.n_examples
    return correct / max(n, 1), loss / max(n, 1)


def finetune_adapter(lm, tasks: list[Task], cfg, seed: int = 0) -> dict:
    """Multi-task LoRA fine-tuning on the algorithmic tasks.  Returns training metrics."""
    a, d = cfg.adapt, cfg.data
    adapter = attach_lora(lm, int(a.rank), float(a.alpha))
    adapter.enabled = True
    train, val = [], {}
    full_domain = []
    for t in tasks:
        if t.finite_domain and bool(a.get("full_domain", True)):
            # A function with an enumerable domain is learned over all of it (as in the grokking setting
            # the paper cites), so the behaviour exists on every input circuits are later evaluated on.
            # Circuit discovery, validation and reporting still use mutually disjoint examples.
            examples = [e for split in SPLITS for e in t.make(split, 10**9, seed)]
            full_domain.append(t.name)
        else:
            examples = t.make("train", int(d.n_train), seed)
        train += encode_examples(lm, examples, t.kind, int(a.batch_size), int(d.max_len))
        val[t.name] = encode_examples(lm, t.make("val", min(int(d.n_val), 200), seed), t.kind,
                                      int(d.batch_size), int(d.max_len))
    params = adapter.parameters()
    opt = torch.optim.AdamW(params, lr=float(a.lr), weight_decay=float(a.weight_decay))
    steps = int(a.steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))))
    rng = random.Random(seed)
    t0 = time.time()
    history = []
    order: list = []
    best = {"min_acc": -1.0, "state": None, "step": 0}
    step = 0
    for step in range(1, steps + 1):
        if not order:
            order = list(train)
            rng.shuffle(order)
        b = order.pop().to(lm.device)
        loss = score_batch(lm.logits(b.input_ids, b.attention_mask), b)["loss"].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % int(a.eval_every) == 0 or step == steps:
            accs = {name: plain_accuracy(lm, bs)[0] for name, bs in val.items()}
            history.append({"step": step, "loss": float(loss), **accs})
            log.info("adapt step %d/%d loss %.4f | %s", step, steps, float(loss),
                     " ".join(f"{k}={v:.3f}" for k, v in accs.items()))
            if min(accs.values()) > best["min_acc"]:
                best = {"min_acc": min(accs.values()), "state": adapter.state_dict(), "step": step}
            if min(accs.values()) >= float(a.target_acc):
                break
            if (time.time() - t0) / 60 > float(a.max_minutes):
                log.warning("adapt: time budget of %s min reached at step %d", a.max_minutes, step)
                break
    if best["state"] is not None:
        adapter.load_state_dict(best["state"])
    adapter.enabled = False
    return {"steps": step, "best_step": best["step"], "seconds": time.time() - t0, "history": history,
            "val_acc": history[-1] if history else {}, "rank": int(a.rank), "tasks": [t.name for t in tasks],
            "full_domain_tasks": full_domain, "train_batches": len(train)}
