"""Benchmark tasks (paper Section IV-B) in one scoring framework.

Every example is a list of ``(context, continuation)`` choices plus the index of the
correct one.  Two task kinds exist:

* ``generative``  – one choice; correct iff every continuation token is the arg-max
  prediction (exact match under teacher forcing); loss = mean token cross-entropy.
* ``multiple_choice`` – several choices; the prediction is the choice whose continuation
  has the highest summed log-likelihood; loss = cross-entropy over choices.

Algorithmic tasks: modular arithmetic, parity, sorting.  Language tasks: WinoGrande,
HellaSwag, factual recall (CounterFact).  Auxiliary tasks (IOI, greater-than, induction)
only widen the task distribution the GNN is meta-trained on.

Splits are disjoint: ``train`` (adapter fine-tuning, transcoder mix-in), ``discover``
(attribution + search), ``val`` (causal validation), ``test`` (reporting only).
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Iterator

import torch
import torch.nn.functional as F

SPLITS = ("train", "discover", "val", "test")


@dataclass
class Example:
    choices: list[tuple[str, str]]
    label: int = 0
    meta: dict = field(default_factory=dict)


@dataclass
class TaskBatch:
    input_ids: torch.Tensor  # (N, S)
    attention_mask: torch.Tensor  # (N, S)
    target_mask: torch.Tensor  # (N, S) True where the token at that position is scored
    labels: torch.Tensor  # (G,) index of the correct choice inside each example
    n_choices: int
    kind: str
    meta: list[dict]

    @property
    def n_examples(self) -> int:
        return int(self.labels.shape[0])

    def to(self, device: torch.device) -> "TaskBatch":
        return TaskBatch(self.input_ids.to(device), self.attention_mask.to(device), self.target_mask.to(device),
                         self.labels.to(device), self.n_choices, self.kind, self.meta)

    def token_mask(self) -> torch.Tensor:
        """Positions that carry real content: not padding and not the BOS token."""
        m = self.attention_mask.bool().clone()
        m[:, 0] = False
        return m


def _frac(key: str, salt: str = "") -> float:
    return int(hashlib.md5(f"{salt}|{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _split_of(key: str, bounds=(0.7, 0.8, 0.9)) -> str:
    x = _frac(key, "split")
    for name, b in zip(SPLITS, bounds):
        if x < b:
            return name
    return SPLITS[-1]


class Task:
    name = "task"
    kind = "generative"
    uses_adapter = False  # True: evaluated with the LoRA adapter switched on
    finite_domain = False  # True: the task's whole input space can be enumerated
    chance = 0.0  # accuracy of a constant predictor (used for chance-corrected reporting)

    def __init__(self, data_cfg) -> None:
        self.cfg = data_cfg

    def pool(self, split: str, seed: int) -> Iterator[Example]:
        raise NotImplementedError

    def make(self, split: str, n: int, seed: int = 0) -> list[Example]:
        out = []
        for ex in self.pool(split, seed):
            out.append(ex)
            if len(out) >= n:
                break
        return out


# --------------------------------------------------------------------------------------
# Algorithmic tasks
# --------------------------------------------------------------------------------------


class _Algorithmic(Task):
    uses_adapter = True

    def _with_shots(self, context: str, seed_key: str) -> str:
        k = int(self.cfg.get("icl_shots", 0))
        if k <= 0:
            return context
        rng = random.Random(f"shots-{seed_key}")
        shots = []
        for ex in self.pool("train", rng.randrange(10**6), _shots=False):
            c, t = ex.choices[0]
            shots.append(c + t)
            if len(shots) >= k:
                break
        return "\n".join(shots) + "\n" + context


class ModArith(_Algorithmic):
    """(a + b) mod p for a, b < p  (p = 113 in the paper)."""

    name = "mod_arith"
    finite_domain = True  # p * p input pairs

    def pool(self, split: str, seed: int, _shots: bool = True) -> Iterator[Example]:
        p = int(self.cfg.mod_p)
        pairs = [(a, b) for a in range(p) for b in range(p)]
        random.Random(f"{seed}-{split}-mod").shuffle(pairs)
        for a, b in pairs:
            if _split_of(f"mod-{p}-{a}-{b}") != split:
                continue
            c = (a + b) % p
            ctx = f"{a} + {b} mod {p} ="
            if _shots:
                ctx = self._with_shots(ctx, f"{a}-{b}")
            yield Example([(ctx, f" {c}")], 0, {"a": a, "b": b, "c": c, "cls": c})

    @property
    def chance(self) -> float:  # type: ignore[override]
        return 1.0 / int(self.cfg.mod_p)


class Parity(_Algorithmic):
    """Parity of a binary string of length 1..20."""

    name = "parity"
    chance = 0.5

    def pool(self, split: str, seed: int, _shots: bool = True) -> Iterator[Example]:
        rng = random.Random(f"{seed}-{split}-parity")
        max_len = int(self.cfg.parity_max_len)
        seen: set[str] = set()
        misses = 0
        while misses < 20000:
            n = rng.randint(1, max_len)
            bits = "".join(rng.choice("01") for _ in range(n))
            if bits in seen or _split_of(f"parity-{bits}") != split:
                misses += 1
                continue
            seen.add(bits)
            misses = 0
            par = bits.count("1") % 2
            ctx = f"Bits: {' '.join(bits)}\nParity:"
            if _shots:
                ctx = self._with_shots(ctx, bits)
            yield Example([(ctx, " odd" if par else " even")], 0, {"bits": bits, "cls": par, "len": n})


class Sorting(_Algorithmic):
    """Sort 5 distinct integers into ascending order."""

    name = "sorting"

    def pool(self, split: str, seed: int, _shots: bool = True) -> Iterator[Example]:
        rng = random.Random(f"{seed}-{split}-sort")
        n, hi = int(self.cfg.sort_len), int(self.cfg.sort_max_int)
        seen: set[tuple] = set()
        misses = 0
        while misses < 20000:
            nums = tuple(rng.sample(range(hi), n))
            if nums in seen or _split_of(f"sort-{nums}") != split:
                misses += 1
                continue
            seen.add(nums)
            misses = 0
            ctx = f"Sort: {' '.join(map(str, nums))}\nSorted:"
            if _shots:
                ctx = self._with_shots(ctx, str(nums))
            yield Example([(ctx, " " + " ".join(map(str, sorted(nums))))], 0, {"nums": nums, "cls": min(nums)})


# --------------------------------------------------------------------------------------
# Language tasks
# --------------------------------------------------------------------------------------


def _hf_rows(repo: str, name: str | None, split: str) -> list[dict]:
    import datasets

    return list(datasets.load_dataset(repo, name, split=split))


class _HFTask(Task):
    kind = "multiple_choice"
    _cache: dict[str, list[dict]] = {}
    eval_bounds = (0.0, 0.35, 0.55)  # train comes from the dataset's own train split

    def _rows(self, split: str) -> list[dict]:
        raise NotImplementedError

    def _example(self, row: dict, rng: random.Random) -> Example | None:
        raise NotImplementedError

    def pool(self, split: str, seed: int) -> Iterator[Example]:
        rows = self._rows(split)
        order = list(range(len(rows)))
        random.Random(f"{seed}-{split}-{self.name}").shuffle(order)
        for i in order:
            if self._row_split(i, split) != split:
                continue
            ex = self._example(rows[i], random.Random(f"{self.name}-{i}"))
            if ex is not None:
                ex.meta["row"] = i
                yield ex

    def _row_split(self, i: int, requested: str) -> str:
        if requested == "train":
            return "train"
        return _split_of(f"{self.name}-{i}", self.eval_bounds)


class WinoGrande(_HFTask):
    """Coreference resolution, partial scoring: P(text after the blank | text with option filled in)."""

    name = "winogrande"
    chance = 0.5

    def _rows(self, split: str) -> list[dict]:
        key = "wg-train" if split == "train" else "wg-val"
        if key not in self._cache:
            self._cache[key] = _hf_rows("allenai/winogrande", "winogrande_xl", "train[:20000]" if split == "train" else "validation")
        return self._cache[key]

    def _example(self, row: dict, rng: random.Random) -> Example | None:
        sent = row["sentence"]
        i = sent.find("_")
        if i < 0 or row["answer"] not in ("1", "2"):
            return None
        cont = sent[i + 1:]
        if len(cont.strip(" .,!?;:")) < 2:
            return None
        pre = sent[:i]
        return Example([(pre + row["option1"], cont), (pre + row["option2"], cont)], int(row["answer"]) - 1,
                       {"cls": int(row["answer"]) - 1})


class HellaSwag(_HFTask):
    """Commonsense sentence completion with four candidate endings."""

    name = "hellaswag"
    chance = 0.25

    def _rows(self, split: str) -> list[dict]:
        key = "hs-train" if split == "train" else "hs-val"
        if key not in self._cache:
            self._cache[key] = _hf_rows("Rowan/hellaswag", None, "train[:20000]" if split == "train" else "validation")
        return self._cache[key]

    @staticmethod
    def _clean(text: str) -> str:
        text = text.strip().replace(" [title]", ". ")
        text = re.sub(r"\[.*?\]", "", text)
        return text.replace("  ", " ")

    def _example(self, row: dict, rng: random.Random) -> Example | None:
        if row["label"] not in ("0", "1", "2", "3"):
            return None
        ctx = self._clean(f"{row['activity_label']}: {row['ctx_a']} {row['ctx_b'].capitalize()}")
        return Example([(ctx, " " + self._clean(e)) for e in row["endings"]], int(row["label"]),
                       {"cls": int(row["label"])})


class FactualQA(_HFTask):
    """Factual recall on CounterFact [11]: true object vs. the counterfactual object."""

    name = "factual_qa"
    chance = 0.5
    eval_bounds = (0.7, 0.8, 0.9)

    def _rows(self, split: str) -> list[dict]:
        if "cf" not in self._cache:
            self._cache["cf"] = _hf_rows("NeelNanda/counterfact-tracing", None, "train")
        return self._cache["cf"]

    def _row_split(self, i: int, requested: str) -> str:
        return _split_of(f"{self.name}-{i}", self.eval_bounds)

    def _example(self, row: dict, rng: random.Random) -> Example | None:
        true, false = row["target_true"], row["target_false"]
        if not true.strip() or not false.strip() or true == false:
            return None
        label = rng.randrange(2)
        choices = [(row["prompt"], false), (row["prompt"], false)]
        choices[label] = (row["prompt"], true)
        return Example(choices, label, {"relation": row["relation_id"], "cls": label})


# --------------------------------------------------------------------------------------
# Auxiliary tasks (GNN meta-training distribution)
# --------------------------------------------------------------------------------------

_NAMES = ["Mary", "John", "Alice", "Tom", "Sarah", "James", "Emma", "David", "Laura", "Mark", "Anna", "Paul",
          "Kate", "Peter", "Lucy", "Daniel", "Grace", "Henry", "Julia", "Robert"]
_PLACES = ["store", "park", "school", "office", "garden", "station", "library", "market", "beach", "museum"]
_OBJECTS = ["book", "drink", "ring", "letter", "key", "ball", "gift", "phone", "map", "coin"]
_NOUNS = ["war", "trial", "drought", "famine", "siege", "voyage", "strike", "tour", "reign", "project"]
_WORDS = ("apple river stone candle forest window silver garden pencil bridge yellow winter doctor market "
          "ocean ladder button engine pillow rocket violet farmer tunnel basket mirror planet hammer copper "
          "island jacket ribbon saddle thunder walnut zipper anchor barrel castle dragon feather goblet harbor "
          "jungle kettle lantern magnet needle orchard parrot quiver rabbit sailor temple umbrella velvet "
          "whistle blanket chimney dolphin eagle fountain glacier helmet insect jewel kitchen lemon marble").split()


class IOI(Task):
    name = "ioi"
    kind = "multiple_choice"
    chance = 0.5

    def pool(self, split: str, seed: int) -> Iterator[Example]:
        rng = random.Random(f"{seed}-{split}-ioi")
        while True:
            a, b = rng.sample(_NAMES, 2)
            place, obj = rng.choice(_PLACES), rng.choice(_OBJECTS)
            if _split_of(f"ioi-{a}-{b}-{place}-{obj}") != split:
                continue
            ctx = f"When {a} and {b} went to the {place}, {b} gave a {obj} to"
            label = rng.randrange(2)
            choices = [(ctx, f" {b}"), (ctx, f" {b}")]
            choices[label] = (ctx, f" {a}")
            yield Example(choices, label, {"cls": label})


class GreaterThan(Task):
    name = "greater_than"
    kind = "multiple_choice"
    chance = 0.5

    def pool(self, split: str, seed: int) -> Iterator[Example]:
        rng = random.Random(f"{seed}-{split}-gt")
        while True:
            cent, yy, noun = rng.randint(11, 19), rng.randint(5, 94), rng.choice(_NOUNS)
            if _split_of(f"gt-{cent}-{yy}-{noun}") != split:
                continue
            ctx = f"The {noun} lasted from the year {cent}{yy:02d} to the year {cent}"
            hi, lo = rng.randint(yy + 1, 99), rng.randint(0, yy - 1)
            label = rng.randrange(2)
            choices = [(ctx, f"{lo:02d}"), (ctx, f"{lo:02d}")]
            choices[label] = (ctx, f"{hi:02d}")
            yield Example(choices, label, {"cls": label})


class Induction(Task):
    name = "induction"
    kind = "generative"

    def pool(self, split: str, seed: int) -> Iterator[Example]:
        rng = random.Random(f"{seed}-{split}-ind")
        while True:
            seq = rng.sample(_WORDS, rng.randint(6, 10))
            if _split_of("ind-" + "-".join(seq)) != split:
                continue
            ctx = " ".join(seq) + " " + " ".join(seq[:-1])
            yield Example([(ctx, " " + seq[-1])], 0, {"cls": _WORDS.index(seq[-1])})


TASKS: dict[str, type[Task]] = {
    t.name: t for t in [ModArith, Parity, Sorting, WinoGrande, HellaSwag, FactualQA, IOI, GreaterThan, Induction]
}


def get_task(name: str, data_cfg) -> Task:
    return TASKS[name](data_cfg)


# --------------------------------------------------------------------------------------
# Tokenisation and scoring
# --------------------------------------------------------------------------------------


def encode_examples(lm, examples: list[Example], kind: str, batch_size: int, max_len: int) -> list[TaskBatch]:
    """Tokenise examples into right-padded batches (padding is causal-safe)."""
    if not examples:
        return []
    n_choices = len(examples[0].choices)
    batches = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start:start + batch_size]
        seqs, n_ctx = [], []
        for ex in chunk:
            assert len(ex.choices) == n_choices, "all examples of a task must have the same number of choices"
            for ctx, cont in ex.choices:
                ids, k = lm.encode_pair(ctx, cont)
                if len(ids) > max_len:  # keep BOS, drop the oldest context tokens
                    cut = len(ids) - max_len
                    cut = min(cut, k - 1)
                    ids = [ids[0]] + ids[1 + cut:]
                    k -= cut
                    ids = ids[:max_len]
                seqs.append(ids)
                n_ctx.append(k)
        width = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), width), lm.pad_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), width), dtype=torch.long)
        tmask = torch.zeros((len(seqs), width), dtype=torch.bool)
        for i, (ids, k) in enumerate(zip(seqs, n_ctx)):
            input_ids[i, :len(ids)] = torch.tensor(ids)
            attn[i, :len(ids)] = 1
            tmask[i, k:len(ids)] = True
        labels = torch.tensor([ex.label for ex in chunk], dtype=torch.long)
        batches.append(TaskBatch(input_ids, attn, tmask, labels, n_choices, kind, [ex.meta for ex in chunk]))
    return batches


def task_batches(lm, task: Task, split: str, n: int, batch_size: int, max_len: int, seed: int = 0) -> list[TaskBatch]:
    return encode_examples(lm, task.make(split, n, seed), task.kind, batch_size, max_len)


def score_batch(logits: torch.Tensor, batch: TaskBatch) -> dict[str, torch.Tensor]:
    """Per-example loss / correctness plus the output distribution used for KL comparisons.

    Returns ``loss`` (G,), ``correct`` (G,) bool, and ``dist`` – log-probabilities over the
    vocabulary at every scored position (generative) or over the choices (multiple choice).
    """
    tgt = batch.input_ids[:, 1:]
    mask = batch.target_mask[:, 1:]
    rows, cols = mask.nonzero(as_tuple=True)
    logp = F.log_softmax(logits[:, :-1][rows, cols].float(), dim=-1)  # (T, V)
    tok_lp = logp.gather(-1, tgt[rows, cols].unsqueeze(-1)).squeeze(-1)
    seq_lp = torch.zeros(batch.input_ids.shape[0], device=logits.device).index_add_(0, rows, tok_lp)
    if batch.kind == "generative":
        n_tok = mask.sum(1).clamp_min(1)
        hit = (logp.argmax(-1) == tgt[rows, cols]).float()
        n_hit = torch.zeros_like(seq_lp).index_add_(0, rows, hit)
        return {"loss": -seq_lp / n_tok, "correct": n_hit == mask.sum(1), "dist": logp}
    scores = seq_lp.view(-1, batch.n_choices)
    dist = F.log_softmax(scores, dim=-1)
    return {"loss": F.cross_entropy(scores, batch.labels, reduction="none"),
            "correct": scores.argmax(-1) == batch.labels, "dist": dist}
