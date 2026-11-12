"""Algorithm 1 — cross-layer transcoder training.

Hidden states are extracted on the fly from the frozen language model (step 6), all layers'
transcoders are updated jointly on Eq. 4 with AdamW (steps 7-11).  Training is resumable:
a checkpoint is written (and pushed to the Hub) every ``ckpt_minutes``.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Iterator

import torch

from .data.corpus import documents, token_batches
from .data.tasks import Task, encode_examples
from .hub import Stages
from .models import LM, transcoder_layers
from .transcoder import CrossLayerTranscoder
from .utils import load_json, log, save_json


def _task_stream(lm: LM, tasks: list[Task], cfg, seed: int) -> Iterator[tuple[torch.Tensor, torch.Tensor, bool]]:
    """Endless stream of (input_ids, attention_mask, adapter_on) from the tasks' train splits."""
    pool = []
    for t in tasks:
        try:
            exs = t.make("train", int(cfg.data.n_train), seed)
        except Exception as e:  # noqa: BLE001
            log.warning("task mix-in: skipping %s (%s)", t.name, str(e)[:120])
            continue
        use_adapter = bool(t.uses_adapter and lm.adapter is not None)
        for b in encode_examples(lm, exs, t.kind, int(cfg.transcoder.batch_seqs), int(cfg.data.max_len)):
            pool.append((b.input_ids, b.attention_mask, use_adapter))
    rng = random.Random(seed)
    while pool:
        yield rng.choice(pool)


def _hidden_tokens(lm: LM, ids: torch.Tensor, attn: torch.Tensor | None, layers: list[int],
                   max_tokens: int, gen: torch.Generator) -> list[torch.Tensor]:
    """Hidden states of the content tokens (no BOS, no padding), the same tokens for every layer."""
    ids = ids.to(lm.device)
    attn = attn.to(lm.device) if attn is not None else None
    hs = lm.hidden_states(ids, attn, layers)
    keep = torch.ones_like(ids, dtype=torch.bool) if attn is None else attn.bool().clone()
    keep[:, 0] = False
    idx = keep.flatten().nonzero(as_tuple=True)[0]
    if len(idx) > max_tokens:
        idx = idx[torch.randperm(len(idx), generator=gen)[:max_tokens].to(idx.device)]
    return [h.reshape(-1, h.shape[-1])[idx] for h in hs]


def train_transcoders(lm: LM, cfg, stages: Stages, rel: str, tasks: list[Task], cross_layer: bool = True,
                      seed: int = 0) -> CrossLayerTranscoder:
    """Train (or resume) the transcoders of one model; artifacts go to ``<run>/<rel>``."""
    tcfg = cfg.transcoder
    out_dir = stages.dir(rel)
    layers = transcoder_layers(lm.n_layers, int(tcfg.layer_stride))
    k = int(tcfg.k or (lm.spec.k if lm.spec else 32))
    tc = CrossLayerTranscoder(layers, lm.d_model, int(tcfg.expansion), k, tcfg.pred_rank, cross_layer,
                              float(tcfg.init_std)).to(lm.device)
    target_tokens = int(tcfg.num_tokens) * int(tcfg.epochs)
    state = {"tokens_seen": 0, "step": 0, "resumes": 0, "seconds": 0.0, "normalized": False}
    stages.restore(rel)
    if (out_dir / "transcoder.safetensors").exists() and (out_dir / "train_state.json").exists():
        tc = CrossLayerTranscoder.load(out_dir, lm.device)
        state = load_json(out_dir / "train_state.json")
        state["resumes"] += 1
        log.info("transcoder: resuming %s at %d tokens (resume #%d)", rel, state["tokens_seen"], state["resumes"])

    gen = torch.Generator().manual_seed(seed + state["resumes"])
    rng = random.Random(seed + 17 * state["resumes"])
    c = cfg.corpus
    corpus_seed = seed + 1000 * state["resumes"]
    corpus = token_batches(lm, c.source, c.subsets, int(c.ctx_len), int(tcfg.batch_seqs), corpus_seed, bool(c.english_only))
    mix = _task_stream(lm, tasks, cfg, seed) if float(tcfg.task_mix) > 0 and tasks else None
    tokens_per_step = int(tcfg.tokens_per_step)

    def next_hidden() -> list[torch.Tensor]:
        if mix is not None and rng.random() < float(tcfg.task_mix):
            # task prompts are short: pool several batches so a task step carries a comparable number of tokens
            parts: list[list[torch.Tensor]] = []
            n_tok = 0
            while n_tok < tokens_per_step and len(parts) < 16:
                ids, attn, adapter_on = next(mix)
                with lm.adapter_on(adapter_on):
                    parts.append(_hidden_tokens(lm, ids, attn, layers, tokens_per_step, gen))
                n_tok += parts[-1][0].shape[0]
            return [torch.cat([p[i] for p in parts])[:tokens_per_step] for i in range(len(layers))]
        with lm.adapter_on(False):
            return _hidden_tokens(lm, next(corpus), None, layers, tokens_per_step, gen)

    if not state["normalized"]:
        # fixed standardisation statistics from the first batches (corpus and task prompts alike)
        samples = [next_hidden() for _ in range(8)]
        tc.fit_normalization([torch.cat([s[i] for s in samples]) for i in range(len(layers))])
        state["normalized"] = True

    decay, no_decay = [], []
    for name, p in tc.named_parameters():
        (decay if name.startswith(("P_in", "P_out")) else no_decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": float(tcfg.weight_decay)},
                             {"params": no_decay, "weight_decay": 0.0}], lr=float(tcfg.lr), betas=(0.9, 0.999))
    total_steps = max(1, target_tokens // tokens_per_step)
    warm = int(tcfg.warmup_steps)

    def lr_at(step: int) -> float:
        ramp = min(1.0, (step - start_step + 1) / max(1, warm))  # re-warm after a resume (fresh optimiser state)
        return ramp * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step, total_steps) / total_steps)))

    start_step = state["step"]
    dead_tokens = max(50_000, target_tokens // 20)
    t_start = time.time()
    last_ckpt = time.time()
    budget_hit = False
    recent: dict = {}

    def checkpoint(final: bool = False) -> None:
        state["seconds"] += time.time() - checkpoint.t_mark  # type: ignore[attr-defined]
        checkpoint.t_mark = time.time()  # type: ignore[attr-defined]
        tc.save(out_dir, {"model": lm.key, "hf_id": getattr(lm, "hf_id", None), "corpus": getattr(documents, "source_used", None)})
        save_json(out_dir / "train_state.json", {**state, "final": final, "recent": recent})
        stages.hub.push(rel, f"transcoder checkpoint {rel} @ {state['tokens_seen']} tokens", checkpoints=True)

    checkpoint.t_mark = time.time()  # type: ignore[attr-defined]

    while state["tokens_seen"] < target_tokens:
        hs = next_hidden()
        for g in opt.param_groups:
            g["lr"] = float(tcfg.lr) * lr_at(state["step"])
        loss, stats = tc.loss(hs, float(tcfg.lambda_pred) if cross_layer else 0.0, float(tcfg.lambda_l1),
                              float(tcfg.aux_k_coef), dead_tokens, bool(tcfg.detach_pred_target))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tc.parameters(), 1.0)
        opt.step()
        tc.normalize_decoder()
        state["step"] += 1
        state["tokens_seen"] += int(hs[0].shape[0])
        recent = {"loss": float(loss), "fvu": sum(stats["fvu"]) / len(stats["fvu"]),
                  "dead": sum(stats["dead"]) / len(stats["dead"]),
                  "pred": (sum(stats["pred"]) / len(stats["pred"])) if stats["pred"] else None}
        if state["step"] % 50 == 0 or state["step"] == start_step + 1:
            rate = (state["step"] - start_step) * tokens_per_step / max(time.time() - t_start, 1e-6)
            log.info("transcoder %s step %d | %.2fM/%.2fM tokens | loss %.4f fvu %.3f dead %.3f | %.0f tok/s",
                     lm.key, state["step"], state["tokens_seen"] / 1e6, target_tokens / 1e6, recent["loss"],
                     recent["fvu"], recent["dead"], rate)
        if (time.time() - last_ckpt) / 60 >= float(tcfg.ckpt_minutes):
            checkpoint()
            last_ckpt = time.time()
        # the budget counts training time across sessions, not just this one
        if (state["seconds"] + time.time() - checkpoint.t_mark) / 60 >= float(tcfg.max_minutes):  # type: ignore[attr-defined]
            budget_hit = True
            log.warning("transcoder: time budget (%s min) reached at %d of %d tokens",
                        tcfg.max_minutes, state["tokens_seen"], target_tokens)
            break

    state["budget_hit"] = budget_hit
    checkpoint(final=True)

    # held-out reconstruction quality (fresh corpus seed)
    eval_stream = token_batches(lm, c.source, c.subsets, int(c.ctx_len), int(tcfg.batch_seqs), seed + 999_983,
                                bool(c.english_only))
    acc: list[list[torch.Tensor]] = []
    n = 0
    with lm.adapter_on(False):
        while n < int(tcfg.eval_tokens):
            hs = _hidden_tokens(lm, next(eval_stream), None, layers, tokens_per_step, gen)
            acc.append(hs)
            n += hs[0].shape[0]
    per_batch = [tc.reconstruction_stats(hs) for hs in acc]
    eval_stats = {key: [sum(pb[key][i] for pb in per_batch) / len(per_batch) for i in range(len(per_batch[0][key]))]
                  for key in per_batch[0]}
    eval_stats["dead_fraction"] = [float((tc.tokens_since_fired[i] > dead_tokens).float().mean()) for i in range(len(layers))]
    result = {
        "model": lm.key, "layers": layers, "m": tc.m, "k": tc.k, "cross_layer": cross_layer,
        "tokens_seen": state["tokens_seen"], "target_tokens": target_tokens, "budget_hit": budget_hit,
        "train_seconds": state["seconds"], "corpus": getattr(documents, "source_used", None),
        "eval": eval_stats, "mean_fvu": sum(eval_stats["fvu"]) / len(eval_stats["fvu"]),
    }
    save_json(Path(out_dir) / "eval.json", result)
    log.info("transcoder %s done: mean held-out FVU %.3f over %d layers (%d tokens)",
             lm.key, result["mean_fvu"], len(layers), state["tokens_seen"])
    return tc
