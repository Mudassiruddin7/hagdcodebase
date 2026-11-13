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

    # ---------------------------------------------------------------- push
    def push(self, rel: str, message: str | None = None, checkpoints: bool = False) -> bool:
        """Upload the local ``<run_dir>/<rel>`` directory (or file)."""
        if not (self.enabled and self.can_write):
            return False
        local = self.run_dir / rel
        if not local.exists():
            return False
        ignore = [] if (checkpoints and self.upload_checkpoints) else list(CKPT_PATTERNS)
        for attempt in range(3):
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
                log.warning("hub push %s failed (attempt %d): %s", rel, attempt + 1, str(e)[:300])
                time.sleep(5 * (attempt + 1))
        return False

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
        save_json(self.run_dir / rel / DONE, {"finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), "protocol": PROTOCOL,
                                              **(meta or {})})
        self.hub.push(rel, checkpoints=checkpoints)

    def all_done(self) -> list[str]:
        return sorted(str(p.parent.relative_to(self.run_dir)).replace("\\", "/") for p in self.run_dir.rglob(DONE))
