"""Small shared helpers: logging, seeding, timing, JSON IO, device and memory tracking."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

log = logging.getLogger("hagd")


def setup_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> None:
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    root = logging.getLogger("hagd")
    root.setLevel(level)
    root.handlers.clear()
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    root.propagate = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class _Enc(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, cls=_Enc)
    os.replace(tmp, path)


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class Timer:
    """Wall-clock stopwatch that can be paused, so shared prerequisites can be accounted separately."""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0


@contextmanager
def timed(store: dict, key: str) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[key] = store.get(key, 0.0) + time.perf_counter() - t0


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gb() -> float:
    """Peak GPU memory allocated since the last reset (0 on CPU)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 2**30
    return 0.0


def free_memory() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_oom(err: BaseException) -> bool:
    return isinstance(err, torch.cuda.OutOfMemoryError) or "out of memory" in str(err).lower()


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of ``values``."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0 or n_boot <= 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def ratio_bootstrap_ci(num: np.ndarray, den: np.ndarray, n_boot: int = 1000, seed: int = 0,
                       alpha: float = 0.05) -> tuple[float, float]:
    """Paired bootstrap CI for mean(num) / mean(den) (used for the accuracy ratio of Eq. 10)."""
    num, den = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
    if len(num) == 0 or n_boot <= 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    d = den[idx].mean(axis=1)
    r = np.where(d > 0, num[idx].mean(axis=1) / np.maximum(d, 1e-12), np.nan)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return (float("nan"), float("nan"))
    return (float(np.quantile(r, alpha / 2)), float(np.quantile(r, 1 - alpha / 2)))
