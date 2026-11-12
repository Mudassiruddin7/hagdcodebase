"""Streaming text corpus for transcoder training.

The paper trains transcoders on RedPajama.  The Hub repo only ships URL lists (its loading
script is not supported by current ``datasets``), so shards are streamed straight from the
official ``data.together.xyz`` URLs with HTTP range requests at seeded random offsets.
Fallbacks, in order: the Hub's Parquet export of RedPajama, then WikiText-103.
"""

from __future__ import annotations

import json
import random
from typing import Iterator

import torch

from ..utils import log

RP_REPO = "togethercomputer/RedPajama-Data-1T"


def _looks_english(text: str) -> bool:
    sample = text[:400]
    if not sample:
        return False
    ascii_n = sum(1 for ch in sample if ch.isascii())
    return ascii_n / len(sample) > 0.9


def _shard_stream(url: str, rng: random.Random, max_docs: int = 2000) -> Iterator[str]:
    """Yield documents from one JSONL shard, starting at a random byte offset."""
    import requests

    size = 0
    try:
        size = int(requests.head(url, timeout=30, allow_redirects=True).headers.get("Content-Length", 0))
    except Exception:  # noqa: BLE001
        size = 0
    headers = {}
    start = 0
    if size > 64 * 2**20 and not url.endswith(".zst"):
        start = rng.randrange(0, size - 32 * 2**20)
        headers["Range"] = f"bytes={start}-"
    with requests.get(url, stream=True, timeout=60, headers=headers) as r:
        r.raise_for_status()
        if url.endswith(".zst"):
            import io

            import zstandard

            reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(r.raw), encoding="utf-8", errors="ignore")
            lines: Iterator = (ln for ln in reader)
        else:
            lines = r.iter_lines(chunk_size=1 << 16, decode_unicode=False)
        first = start > 0
        n = 0
        for raw in lines:
            if first:  # partial line after a range seek
                first = False
                continue
            if not raw:
                continue
            try:
                doc = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                continue
            text = doc.get("text") if isinstance(doc, dict) else None
            if text:
                yield text
                n += 1
                if n >= max_docs:
                    return


def _subset_stream(subset: str, seed: int) -> Iterator[str]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(RP_REPO, f"urls/{subset}.txt", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        urls = [u.strip() for u in f if u.strip()]
    if subset == "common_crawl":
        try:
            import zstandard  # noqa: F401
        except ImportError as e:
            raise RuntimeError("common_crawl shards need the `zstandard` package") from e
    rng = random.Random(f"{seed}-{subset}")
    failures = 0
    while True:
        url = rng.choice(urls)
        try:
            for text in _shard_stream(url, rng):
                failures = 0
                yield text
        except Exception as e:  # noqa: BLE001
            failures += 1
            log.debug("redpajama %s shard error: %s", subset, str(e)[:160])
            if failures >= 5:
                raise RuntimeError(f"RedPajama subset {subset} unreachable: {e}") from e


def _redpajama_urls(subsets: dict[str, float], seed: int, english_only: bool) -> Iterator[str]:
    names = [s for s, w in subsets.items() if w > 0]
    weights = [float(subsets[s]) for s in names]
    streams = {s: _subset_stream(s, seed) for s in names}
    rng = random.Random(seed)
    rejected = 0
    while names:
        s = rng.choices(names, weights)[0]
        try:
            text = next(streams[s])
        except Exception as e:  # noqa: BLE001
            log.warning("dropping RedPajama subset %s: %s", s, str(e)[:160])
            i = names.index(s)
            names.pop(i)
            weights.pop(i)
            continue
        if english_only and not _looks_english(text):
            rejected += 1
            if rejected % 300 == 0:  # stuck in a non-English region of a big shard: re-seek
                streams[s] = _subset_stream(s, seed + rejected)
            continue
        yield text
    raise RuntimeError("no RedPajama subset reachable")


def _redpajama_parquet(seed: int, english_only: bool) -> Iterator[str]:
    import datasets

    parts = []
    for subset in ("stackexchange", "wikipedia"):
        parts.append(datasets.load_dataset(
            "parquet",
            data_files=f"hf://datasets/{RP_REPO}@refs%2Fconvert%2Fparquet/{subset}/partial-train/*.parquet",
            split="train", streaming=True,
        ).shuffle(seed=seed, buffer_size=2000))
    its = [iter(p) for p in parts]
    while its:
        for it in list(its):
            try:
                text = next(it)["text"]
            except StopIteration:
                its.remove(it)
                continue
            if text and (not english_only or _looks_english(text)):
                yield text


def _wikitext(seed: int) -> Iterator[str]:
    import datasets

    ds = datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    buf: list[str] = []
    for row in ds.shuffle(seed=seed, buffer_size=5000):
        line = row["text"].strip()
        if not line:
            continue
        buf.append(line)
        if sum(len(b) for b in buf) > 2000:
            yield " ".join(buf)
            buf = []


def documents(source: str, subsets: dict[str, float], seed: int, english_only: bool = True) -> Iterator[str]:
    """Infinite document stream.  Logs (and records in ``documents.source_used``) what was used."""
    chain = {"redpajama": ["redpajama", "redpajama_parquet", "wikitext"], "wikitext": ["wikitext"]}[source]
    for name in chain:
        try:
            if name == "redpajama":
                gen = _redpajama_urls(dict(subsets), seed, english_only)
            elif name == "redpajama_parquet":
                gen = _redpajama_parquet(seed, english_only)
            else:
                gen = _wikitext(seed)
            first = next(gen)
        except Exception as e:  # noqa: BLE001
            log.warning("corpus source %s unavailable (%s)", name, str(e)[:200])
            continue
        if name != source:
            log.warning("corpus: requested %s but using fallback %s", source, name)
        documents.source_used = name  # type: ignore[attr-defined]
        yield first
        yield from gen
        return
    raise RuntimeError("no text corpus available")


def token_batches(lm, source: str, subsets: dict[str, float], ctx_len: int, batch_seqs: int, seed: int,
                  english_only: bool = True, max_chunks_per_doc: int = 8) -> Iterator[torch.Tensor]:
    """Infinite stream of (batch_seqs, ctx_len) token batches, every row starting with BOS."""
    rows: list[list[int]] = []
    body = ctx_len - 1
    for text in documents(source, subsets, seed, english_only):
        ids = lm.tokenizer(text[:20000], add_special_tokens=False)["input_ids"]
        n_chunks = min(len(ids) // body, max_chunks_per_doc)
        for c in range(n_chunks):
            rows.append([lm.bos_id] + ids[c * body:(c + 1) * body])
            if len(rows) == batch_seqs:
                yield torch.tensor(rows, dtype=torch.long)
                rows = []
