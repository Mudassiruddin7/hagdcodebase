"""Shared fixtures: a tiny randomly initialised GPT-2 so tests need no model download."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hagd.config import load_config  # noqa: E402
from hagd.models import LM, transcoder_layers  # noqa: E402
from hagd.transcoder import CrossLayerTranscoder  # noqa: E402


class _CharTokenizer:
    """Minimal tokenizer: one token per character (ids 3..), with BOS/EOS/PAD."""

    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0
    eos_token = "<eos>"

    def __init__(self) -> None:
        self.pad_token = "<pad>"

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": [3 + (ord(c) % 120) for c in text]}


@pytest.fixture(scope="session")
def cfg():
    return load_config("smoke", {"hub": {"enabled": False}})


@pytest.fixture(scope="session")
def lm() -> LM:
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(n_layer=3, n_embd=32, n_head=4, vocab_size=128, n_positions=96,
                                       bos_token_id=1, eos_token_id=2))
    return LM(model, _CharTokenizer(), key="tiny")


@pytest.fixture(scope="session")
def tc(lm: LM) -> CrossLayerTranscoder:
    torch.manual_seed(0)
    tc = CrossLayerTranscoder(transcoder_layers(lm.n_layers, 1), lm.d_model, expansion=4, k=8, pred_rank=16)
    ids = torch.randint(3, 120, (16, 24))
    ids[:, 0] = 1
    hs = [h.reshape(-1, lm.d_model) for h in lm.hidden_states(ids, None, tc.layers)]
    tc.fit_normalization(hs)
    opt = torch.optim.Adam(tc.parameters(), lr=3e-3)
    for _ in range(60):
        loss, _ = tc.loss(hs, 0.1, 1e-3)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tc.normalize_decoder()
    tc.eval()
    tc.requires_grad_(False)
    return tc
