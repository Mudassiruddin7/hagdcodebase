"""Hugging Face Hub storage for run artifacts.

Everything a run produces lives under ``runs/<run_name>/`` both locally and in the dataset
repo, so a session that is lost (molab sandboxes are not persistent) resumes from the Hub.
A stage is finished when its directory contains ``DONE.json``.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from .utils import load_json, log, save_json

DONE = "DONE.json"
# Version of the circuit-extraction protocol.  Stages produced under an older protocol are redone;
# model-level prerequisites (task adapters, transcoders) do not depend on it and are kept.
#   1: interventions at every token, residual mean-ablated, accuracy-ratio criterion
#   2: interventions at answer positions, residual kept, combined criterion, growing exhaustive pool
PROTOCOL = 2
UNVERSIONED = ("adapter", "transcoder", "transcoder_nocross")
CKPT_PATTERNS = ["*.safetensors", "*.pt"]


class Hub:
    def __init__(self, repo_id: str, run_dir: Path, run_name: str, enabled: bool = True,
                 upload_checkpoints: bool = True) -> None:
        self.repo_id = repo_id
        self.run_dir = Path(run_dir)
        self.prefix = f"runs/{run_name}"
        self.upload_checkpoints = upload_checkpoints
        self.token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self.enabled = bool(enabled)
        self.can_write = False
        self._api = None
        self._remote: set[str] | None = None
        self._pending: dict[str, bool] = {}  # rel -> include checkpoints; pushes that failed and await a retry
        self._executor = None
        self._bg = None  # Future of the running background upload
        self._last_log_push = 0.0
        if self.enabled:
            try:
                from huggingface_hub import HfApi

                self._api = HfApi(token=self.token)
                try:
                    self._api.whoami()
                    self.can_write = True
                except Exception:  # noqa: BLE001
                    self.can_write = False
            except Exception as e:  # noqa: BLE001
                log.warning("Hub disabled (%s)", e)
                self.enabled = False
        if self.enabled and not self.can_write:
            log.warning("No valid Hugging Face token: results stay local and are NOT uploaded to %s. "
                        "Set HF_TOKEN (write access) to enable uploads and cross-session resume.", repo_id)

    # ---------------------------------------------------------------- repo
    def ensure_repo(self) -> None:
        if not (self.enabled and self.can_write):
            return
        try:
            self._api.create_repo(self.repo_id, repo_type="dataset", exist_ok=True)
        except Exception as e:  # noqa: BLE001
            log.warning("could not create/verify dataset repo %s: %s", self.repo_id, e)

    def remote_files(self) -> set[str]:
        """Paths (relative to the run prefix) that exist in the dataset repo.  Listed once per process."""
        if self._remote is None:
            self._remote = set()
            if self.enabled:
                try:
                    cut = len(self.prefix) + 1
                    self._remote = {f[cut:] for f in self._api.list_repo_files(self.repo_id, repo_type="dataset")
                                    if f.startswith(self.prefix + "/")}
                    if self._remote:
                        log.info("hub: %d artifact file(s) of this run already on %s", len(self._remote), self.repo_id)
                except Exception as e:  # noqa: BLE001
                    log.warning("hub: could not list %s (%s)", self.repo_id, str(e)[:200])
        return self._remote

    # ---------------------------------------------------------------- pull
    def pull(self, rel: str) -> bool:
        """Download every remote file under ``<rel>/`` that is missing locally."""
        if not self.enabled:
            return False
        wanted = [f for f in self.remote_files() if f.startswith(rel.rstrip("/") + "/") and not (self.run_dir / f).exists()]
        if not wanted:
            return False
        from huggingface_hub import hf_hub_download

        mirror = self.run_dir / ".hub_mirror"
        n = 0
        for f in wanted:
            try:
                got = hf_hub_download(self.repo_id, f"{self.prefix}/{f}", repo_type="dataset", local_dir=str(mirror),
                                      token=self.token)
                target = self.run_dir / f
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(got, target)
                n += 1
            except Exception as e:  # noqa: BLE001
                log.warning("hub pull %s failed: %s", f, str(e)[:200])
        if n:
            log.info("hub: restored %d file(s) for %s", n, rel)
        return n > 0

    def pull_matching(self, suffixes: tuple[str, ...] = (".json",)) -> int:
        """Download every remote file of this run with one of ``suffixes`` that is missing locally.

        Used before building the report: a resumed session only holds the stages it touched,
        but the report must cover every model finished in earlier sessions.
        """
        if not self.enabled:
            return 0
        wanted = [f for f in self.remote_files() if f.endswith(suffixes) and not (self.run_dir / f).exists()]
        if not wanted:
            return 0
        from huggingface_hub import snapshot_download

        mirror = self.run_dir / ".hub_mirror"
        try:
            snapshot_download(self.repo_id, repo_type="dataset", allow_patterns=[f"{self.prefix}/{f}" for f in wanted],
                              local_dir=str(mirror), token=self.token)
        except Exception as e:  # noqa: BLE001
            log.warning("hub: bulk pull failed (%s)", str(e)[:200])
            return 0
        n = 0
        for f in wanted:
            src = mirror / self.prefix / f
            if src.exists() and not (self.run_dir / f).exists():
                (self.run_dir / f).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), self.run_dir / f)
                n += 1
        log.info("hub: restored %d result file(s) from earlier sessions", n)
        return n

    # ---------------------------------------------------------------- push
    def _upload(self, rel: str, message: str | None, checkpoints: bool) -> bool:
        local = self.run_dir / rel
        if not local.exists():
            return True  # nothing to upload (e.g. removed): not a failure
        ignore = [] if (checkpoints and self.upload_checkpoints) else list(CKPT_PATTERNS)
        for attempt in range(4):
            try:
                if local.is_file():
                    self._api.upload_file(
                        path_or_fileobj=str(local), path_in_repo=f"{self.prefix}/{rel}",
                        repo_id=self.repo_id, repo_type="dataset",
                        commit_message=message or f"{self.prefix}: {rel}",
                    )
                else:
                    self._api.upload_folder(
                        folder_path=str(local), path_in_repo=f"{self.prefix}/{rel}",
                        repo_id=self.repo_id, repo_type="dataset", ignore_patterns=ignore + ["*.tmp", ".hub_mirror/**"],
                        commit_message=message or f"{self.prefix}: {rel}",
                    )
                return True
            except Exception as e:  # noqa: BLE001
                wait = 15 * (attempt + 1) * (4 if "429" in str(e) or "rate" in str(e).lower() else 1)
                log.warning("hub push %s failed (attempt %d, retry in %ds): %s", rel, attempt + 1, wait, str(e)[:300])
                time.sleep(wait)
        return False

    def push(self, rel: str, message: str | None = None, checkpoints: bool = False, background: bool = False) -> bool:
        """Upload the local ``<run_dir>/<rel>`` directory (or file).

        A push that still fails after retries is queued and retried before every later push,
        so a temporary Hub outage does not lose results.  ``background=True`` uploads in a
        worker thread (used for large training checkpoints, so training is not stalled); it
        returns False without doing anything while an earlier background upload is running.
        """
        if not (self.enabled and self.can_write):
            return False
        if background:
            if self.background_busy():
                return False
            self._bg = self._pool().submit(self._upload, rel, message, checkpoints)
            return True
        self.flush_pending()
        ok = self._upload(rel, message, checkpoints)
        if not ok:
            self._pending[rel] = checkpoints or self._pending.get(rel, False)
            log.warning("hub push %s queued for retry (%d pending)", rel, len(self._pending))
        else:
            self._pending.pop(rel, None)
        return ok

    def flush_pending(self) -> None:
        if not (self.enabled and self.can_write):
            return
        for rel, ckpt in list(self._pending.items()):
            if self._upload(rel, f"{self.prefix}: {rel} (retry)", ckpt):
                self._pending.pop(rel, None)
                log.info("hub: queued push %s succeeded", rel)

    def push_logs(self, every_minutes: float = 10.0) -> None:
        """Upload the pipeline log at most every ``every_minutes`` (progress is visible on the Hub)."""
        if not (self.enabled and self.can_write):
            return
        if time.time() - self._last_log_push >= every_minutes * 60:
            self._last_log_push = time.time()
            self._upload("logs", f"{self.prefix}: logs", False)

    def _pool(self):
        if self._executor is None:
            from concurrent.futures import ThreadPoolExecutor

            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hub-upload")
        return self._executor

    def background_busy(self) -> bool:
        return self._bg is not None and not self._bg.done()

    def wait_background(self) -> None:
        """Block until a running background upload has finished."""
        if self._bg is not None:
            try:
                if not self._bg.result():
                    log.warning("hub: background checkpoint upload failed (the next checkpoint will retry)")
            except Exception as e:  # noqa: BLE001
                log.warning("hub: background upload error: %s", str(e)[:200])
            self._bg = None

    def push_file_to_root(self, local: Path, path_in_repo: str, message: str) -> bool:
        if not (self.enabled and self.can_write):
            return False
        try:
            self._api.upload_file(path_or_fileobj=str(local), path_in_repo=path_in_repo,
                                  repo_id=self.repo_id, repo_type="dataset", commit_message=message)
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("hub push %s failed: %s", path_in_repo, str(e)[:300])
            return False


class Stages:
    """Stage bookkeeping on top of the run directory + Hub."""

    def __init__(self, run_dir: Path, hub: Hub) -> None:
        self.run_dir = Path(run_dir)
        self.hub = hub

    def dir(self, rel: str) -> Path:
        d = self.run_dir / rel
        d.mkdir(parents=True, exist_ok=True)
        return d

    def done(self, rel: str) -> bool:
        """True if the stage is complete (locally, or restorable from the Hub) under the current protocol."""
        path = self.run_dir / rel / DONE
        if not path.exists() and f"{rel}/{DONE}" in self.hub.remote_files():
            self.hub.pull(rel)
        if not path.exists():
            return False
        if rel.rstrip("/").split("/")[-1] in UNVERSIONED:
            return True
        return load_json(path).get("protocol") == PROTOCOL

    def restore(self, rel: str) -> None:
        """Fetch a partially finished stage (e.g. a training checkpoint) from the Hub."""
        self.hub.pull(rel)

    def result(self, rel: str) -> dict:
        return load_json(self.run_dir / rel / DONE)

    def finish(self, rel: str, meta: dict | None = None, checkpoints: bool = False) -> None:
        if checkpoints:  # never upload a folder while a background upload may still be reading it
            self.hub.wait_background()
        save_json(self.run_dir / rel / DONE, {"finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), "protocol": PROTOCOL,
                                              **(meta or {})})
        self.hub.push(rel, checkpoints=checkpoints)
        self.hub.push_logs()

    def all_done(self) -> list[str]:
        return sorted(str(p.parent.relative_to(self.run_dir)).replace("\\", "/") for p in self.run_dir.rglob(DONE))
