"""Language-model wrapper.

All interventions are done with forward hooks on the transformer blocks, so the same code
runs GPT-2, GPT-NeoX (Pythia) and Llama on any transformers version.  ``h_l`` in the paper
is the residual stream at the output of block ``l``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator

import torch
from torch import nn

from .config import ModelSpec
from .utils import log

HookFn = Callable[[torch.Tensor], "torch.Tensor | None"]

_BLOCK_PATHS = ["transformer.h", "gpt_neox.layers", "model.layers", "model.decoder.layers", "transformer.blocks"]
_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


class StopForward(Exception):
    """Raised inside a hook to abandon the rest of a forward pass."""


def find_blocks(model: nn.Module) -> nn.ModuleList:
    for path in _BLOCK_PATHS:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList) and len(obj) > 0:
            return obj
    raise ValueError(f"could not locate transformer blocks in {type(model).__name__}")


class LM:
    """A causal LM with block-output hooks, a BOS-aware tokenizer and optional LoRA adapters."""

    def __init__(self, model: nn.Module, tokenizer, spec: ModelSpec | None = None, key: str = "custom") -> None:
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.tokenizer = tokenizer
        self.spec = spec
        self.key = spec.key if spec else key
        self.blocks = find_blocks(model)
        self.n_layers = len(self.blocks)
        self.d_model = int(model.config.hidden_size)
        self.device = next(model.parameters()).device
        self.dtype = next(p for p in model.parameters() if p.is_floating_point()).dtype
        self.adapter = None  # set by hagd.adapt.attach_lora
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, spec: ModelSpec, device: torch.device, quantization: str | None = None) -> "LM":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = _DTYPES[spec.dtype] if device.type == "cuda" else torch.float32
        last_err: Exception | None = None
        for hf_id in (spec.hf_id, *spec.fallback_ids):
            try:
                kwargs: dict = {"low_cpu_mem_usage": True}
                if quantization:
                    from transformers import BitsAndBytesConfig

                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=quantization == "4bit", load_in_8bit=quantization == "8bit",
                        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4",
                    )
                    kwargs["device_map"] = {"": 0}
                try:
                    model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=dtype, **kwargs)
                except TypeError:
                    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype, **kwargs)
                if not quantization:
                    model = model.to(device)
                tok = AutoTokenizer.from_pretrained(hf_id)
                if hf_id != spec.hf_id:
                    log.warning("%s: loaded fallback repo %s (primary %s unavailable: %s)",
                                spec.key, hf_id, spec.hf_id, str(last_err)[:120])
                lm = cls(model, tok, spec)
                lm.hf_id = hf_id
                log.info("loaded %s (%s): %d layers, d=%d, dtype=%s, device=%s",
                         spec.key, hf_id, lm.n_layers, lm.d_model, lm.dtype, lm.device)
                return lm
            except Exception as e:  # noqa: BLE001
                last_err = e
                log.warning("could not load %s: %s", hf_id, str(e)[:200])
        raise RuntimeError(f"failed to load {spec.key}: {last_err}")

    # ------------------------------------------------------------------ tokenization
    def encode(self, text: str) -> list[int]:
        """Token ids with exactly one BOS in front."""
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return [self.bos_id] + list(ids)

    def encode_pair(self, context: str, continuation: str) -> tuple[list[int], int]:
        """Ids of ``context + continuation`` and the number of leading context tokens.

        Tokenising the concatenation keeps SentencePiece/BPE boundary behaviour faithful; if the
        context is not a token-prefix of the whole we fall back to separate tokenisation.
        """
        ctx = self.encode(context)
        full = self.encode(context + continuation)
        if len(full) > len(ctx) and full[: len(ctx)] == ctx:
            return full, len(ctx)
        cont = self.tokenizer(continuation, add_special_tokens=False)["input_ids"]
        return ctx + list(cont), len(ctx)

    # ------------------------------------------------------------------ forward
    def logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return out.logits

    @contextmanager
    def hooks(self, fns: dict[int, HookFn]) -> Iterator[None]:
        """Install ``fn(h) -> h'`` on the output of block ``layer`` for each ``{layer: fn}``."""
        handles = []

        def make(fn: HookFn):
            def hook(_mod, _args, out):
                h = out[0] if isinstance(out, tuple) else out
                new = fn(h)
                if new is None:
                    return None
                new = new.to(h.dtype)
                return (new, *out[1:]) if isinstance(out, tuple) else new

            return hook

        try:
            for layer, fn in fns.items():
                handles.append(self.blocks[layer].register_forward_hook(make(fn)))
            yield
        finally:
            for h in handles:
                h.remove()

    def run(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
            fns: dict[int, HookFn] | None = None) -> torch.Tensor | None:
        """Forward pass with hooks.  Returns logits, or None if a hook raised StopForward."""
        with self.hooks(fns or {}):
            try:
                return self.logits(input_ids, attention_mask)
            except StopForward:
                return None

    @torch.no_grad()
    def hidden_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None,
                      layers: list[int]) -> list[torch.Tensor]:
        """Residual stream at the output of each requested block (float32)."""
        store: dict[int, torch.Tensor] = {}
        last = max(layers)

        def make(layer: int) -> HookFn:
            def fn(h: torch.Tensor) -> None:
                store[layer] = h.detach().float()
                if layer == last:
                    raise StopForward
                return None

            return fn

        self.run(input_ids, attention_mask, {layer: make(layer) for layer in layers})
        return [store[layer] for layer in layers]

    # ------------------------------------------------------------------ adapters
    def set_adapter(self, on: bool) -> None:
        if self.adapter is not None:
            self.adapter.enabled = bool(on)

    @contextmanager
    def adapter_on(self, on: bool) -> Iterator[None]:
        prev = self.adapter.enabled if self.adapter is not None else False
        self.set_adapter(on)
        try:
            yield
        finally:
            self.set_adapter(prev)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


def transcoder_layers(n_layers: int, stride: int = 1) -> list[int]:
    """Block indices that get a transcoder.  Always includes the last block."""
    stride = max(1, int(stride))
    layers = list(range(stride - 1, n_layers, stride))
    if not layers or layers[-1] != n_layers - 1:
        layers.append(n_layers - 1)
    return layers
