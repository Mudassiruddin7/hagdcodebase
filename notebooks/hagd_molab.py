# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "torch",
#     "transformers",
#     "datasets",
#     "huggingface_hub",
#     "safetensors",
#     "numpy",
#     "scipy",
#     "scikit-learn",
#     "pandas",
#     "matplotlib",
#     "pyyaml",
#     "requests",
#     "python-dotenv",
# ]
# ///
"""HAGD on molab.

Open on https://molab.marimo.io (new notebook -> paste this file's GitHub URL), attach the
GPU from the notebook specs button, add an HF_TOKEN secret, then press "Start / resume".
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="HAGD circuit extraction")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # Hierarchical Attribution Graph Decomposition

        Replication of *Hierarchical Sparse Circuit Extraction from Billion-Parameter Language Models
        through Scalable Attribution Graph Decomposition* (arXiv:2601.12879).

        This notebook installs the code from GitHub, launches the experiment pipeline as a background
        process on the GPU, and shows progress and results. Every finished stage is uploaded to the
        Hugging Face dataset repo, so if the session ends, **press Start again and the run resumes**.

        1. Attach the GPU (notebook specs button in the header).
        2. Add a secret `HF_TOKEN` with **write** access (developer panel -> secrets), or paste it below.
           It is needed to upload results, and to download the gated Llama-2 weights.
        3. Pick a profile and press **Start / resume**. Keep this tab open while it runs.
        """
    )
    return


@app.cell
def _(mo):
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    REPO_URL = "https://github.com/Mudassiruddin7/hagdcodebase"

    def _sh(cmd: list[str], cwd=None) -> tuple[int, str]:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        return p.returncode, (p.stdout + p.stderr)[-3000:]

    # Use the repository this notebook lives in when it was opened from GitHub; otherwise clone it.
    try:
        _here = Path(mo.notebook_dir() or Path.cwd())
    except Exception:  # noqa: BLE001  (older marimo)
        _here = Path.cwd()
    _local = [p for p in (_here.parent, _here, Path.cwd()) if (p / "pyproject.toml").exists() and (p / "src" / "hagd").exists()]
    setup_log = []
    if _local:
        repo_dir = _local[0]
        setup_log.append(f"using repository at {repo_dir}")
    else:
        repo_dir = Path.home() / "hagdcodebase"
        if (repo_dir / ".git").exists():
            _code, _out = _sh(["git", "pull", "--ff-only"], cwd=repo_dir)
            setup_log.append(f"git pull: {_out.strip()[-200:]}")
        else:
            _code, _out = _sh(["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)])
            setup_log.append(f"git clone ({_code}): {_out.strip()[-200:]}")

    # Install ONLY what is missing. molab ships a matched torch / torchvision / transformers stack;
    # letting pip "upgrade" any of them breaks transformers' lazy imports
    # ("Could not import module 'GPT2LMHeadModel'"). So: the package itself with --no-deps,
    # then each dependency only if it cannot be imported.
    import importlib.util as _iu

    def _pip(args: list[str]) -> tuple[int, str]:
        if shutil.which("uv"):
            c, o = _sh(["uv", "pip", "install", "--python", sys.executable, *args])
            if c == 0:
                return c, o
        return _sh([sys.executable, "-m", "pip", "install", *args])

    _code, _out = _pip(["--no-deps", "-e", str(repo_dir)])
    setup_log.append("hagd installed (no-deps)" if _code == 0 else f"INSTALL FAILED:\n{_out}")
    _wanted = {"torch": "torch", "transformers": "transformers", "datasets": "datasets", "huggingface_hub": "huggingface_hub",
               "safetensors": "safetensors", "scipy": "scipy", "sklearn": "scikit-learn", "pandas": "pandas",
               "matplotlib": "matplotlib", "yaml": "pyyaml", "requests": "requests", "zstandard": "zstandard",
               "bitsandbytes": "bitsandbytes", "accelerate": "accelerate", "dotenv": "python-dotenv", "psutil": "psutil"}
    _missing = [pkg for mod, pkg in _wanted.items() if _iu.find_spec(mod) is None]
    if _missing:
        # pin the torch that is already there, so e.g. accelerate cannot upgrade it under torchvision
        _code, _tv = _sh([sys.executable, "-c", "import importlib.metadata as m; print(m.version('torch'))"])
        _pin = [f"torch=={_tv.strip()}"] if _code == 0 and _tv.strip() else []
        _code, _out = _pip(_missing + _pin)
        setup_log.append(f"installed missing: {', '.join(_missing)}" if _code == 0 else f"could not install {_missing}:\n{_out}")
    # verify the model stack really imports, and show the true error if it does not
    _check = [sys.executable, "-c", "import torch, transformers; from transformers.models.gpt2 import modeling_gpt2; "
              "from transformers.models.llama import modeling_llama; from transformers.models.gpt_neox import modeling_gpt_neox; "
              "print('torch', torch.__version__, '| transformers', transformers.__version__)"]
    _code, _out = _sh(_check)
    if _code != 0 and "torchvision" in _out:
        # torchvision built for a different torch. HAGD never uses it; transformers only imports it when present.
        _r, _o = _sh(["uv", "pip", "uninstall", "--python", sys.executable, "torchvision"]) if shutil.which("uv") else (1, "")
        if _r != 0:
            _sh([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision"])
        setup_log.append("removed torchvision (version mismatch with torch; not needed)")
        _code, _out = _sh(_check)
    stack_ok = _code == 0
    setup_log.append(("model stack OK: " if stack_ok else "MODEL STACK BROKEN — restart the kernel after fixing:\n") + _out.strip()[-1500:])
    if str(repo_dir / "src") not in sys.path:
        sys.path.insert(0, str(repo_dir / "src"))

    try:
        from dotenv import load_dotenv

        load_dotenv()
        load_dotenv(repo_dir / ".env")
    except ImportError:
        pass

    runs_root = repo_dir / "runs"
    mo.md("**Setup**\n\n" + "\n".join(f"- {line}" for line in setup_log))
    return Path, os, repo_dir, runs_root, stack_ok, subprocess, sys


@app.cell
def _(mo, os):
    import torch

    if torch.cuda.is_available():
        _p = torch.cuda.get_device_properties(0)
        _gpu = f"**GPU:** {_p.name} — {_p.total_memory / 2**30:.0f} GB"
    else:
        _gpu = "**GPU:** none attached — only the `smoke` profile is practical on CPU."
    _tok = "found" if (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")) else "**missing**"
    token_box = mo.ui.text(kind="password", label="HF token (only if no HF_TOKEN secret is set)", full_width=True)
    mo.vstack([mo.md(f"{_gpu}\n\n**HF_TOKEN secret:** {_tok}"), token_box])
    return (token_box,)


@app.cell
def _(mo):
    from hagd.config import CONFIG_DIR, MODEL_REGISTRY, load_config
    from hagd.pipeline import PHASES

    profile = mo.ui.dropdown(sorted(p.stem for p in CONFIG_DIR.glob("*.yaml")), value="molab", label="Profile")
    mo.md(
        f"""
        ## Run configuration

        {profile}

        `smoke` – minutes, checks that everything works · `laptop` – GPT-2 Small only ·
        `molab` – eight models sized for one 96 GB GPU · `paper` – the paper's full protocol (multi-GPU, weeks)
        """
    )
    return MODEL_REGISTRY, PHASES, load_config, profile


@app.cell
def _(MODEL_REGISTRY, PHASES, load_config, mo, profile):
    _cfg = load_config(profile.value)
    models = mo.ui.multiselect(list(MODEL_REGISTRY), value=list(_cfg.models), label="Benchmark models")
    phases = mo.ui.multiselect(PHASES, value=PHASES, label="Phases")
    run_name = mo.ui.text(value=_cfg.run_name, label="Run name (folder on the Hub)")
    repo_id = mo.ui.text(value=_cfg.hub.repo_id, label="HF dataset repo")
    extra = mo.ui.text(value="", label="Extra overrides, e.g. transcoder.num_tokens=5e6 search.seeds=3", full_width=True)
    start = mo.ui.run_button(label="Start / resume", kind="success")
    stop = mo.ui.run_button(label="Stop", kind="danger")
    mo.vstack([mo.hstack([models, phases]), mo.hstack([run_name, repo_id]), extra, mo.hstack([start, stop], justify="start"),
               mo.md(f"GNN label models for this profile: `{', '.join(_cfg.label_models)}` · tasks: `{', '.join(_cfg.tasks)}`")])
    return extra, models, phases, repo_id, run_name, start, stop


@app.cell
def _(Path, runs_root, run_name):
    run_dir = Path(runs_root) / run_name.value
    pid_file = run_dir / "pipeline.pid"
    console_log = run_dir / "logs" / "console.log"

    def running_pid() -> int | None:
        """PID of the background pipeline if it is still alive."""
        import psutil

        try:
            pid = int(pid_file.read_text().strip())
            proc = psutil.Process(pid)
            return pid if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE else None
        except (FileNotFoundError, ValueError, psutil.Error):
            return None

    def stop_pipeline() -> int | None:
        """Terminate the pipeline process and anything it spawned."""
        import psutil

        pid = running_pid()
        if pid is not None:
            proc = psutil.Process(pid)
            for p in [*proc.children(recursive=True), proc]:
                try:
                    p.terminate()
                except psutil.Error:
                    pass
        pid_file.unlink(missing_ok=True)
        return pid

    return console_log, pid_file, run_dir, running_pid, stop_pipeline


@app.cell
def _(console_log, extra, mo, models, os, phases, pid_file, profile, repo_dir, repo_id, run_dir, run_name, running_pid,
      runs_root, stack_ok, start, subprocess, sys, token_box):
    mo.stop(not start.value)
    if not stack_ok:
        launch_msg = mo.md("**Not started:** the model stack failed its import check (see Setup above). "
                           "Restart the kernel / session so setup runs again.")
    elif running_pid() is not None:
        launch_msg = mo.md(f"A pipeline is already running (PID {running_pid()}). Stop it first to change settings.")
    else:
        _env = dict(os.environ)
        if token_box.value:
            _env["HF_TOKEN"] = token_box.value
        _env["PYTHONPATH"] = str(repo_dir / "src") + os.pathsep + _env.get("PYTHONPATH", "")
        _env["PYTHONUNBUFFERED"] = "1"
        _env["TOKENIZERS_PARALLELISM"] = "false"
        _cmd = [sys.executable, "-m", "hagd.cli", "run", "--profile", profile.value, "--run-name", run_name.value,
                "--output-dir", str(runs_root), "--repo-id", repo_id.value]
        if models.value:
            _cmd += ["--models", *models.value]
        if phases.value:
            _cmd += ["--phases", *phases.value]
        if extra.value.strip():
            _cmd += ["--set", *extra.value.split()]
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        _log = open(console_log, "a", encoding="utf-8")
        _proc = subprocess.Popen(_cmd, stdout=_log, stderr=subprocess.STDOUT, cwd=str(repo_dir), env=_env,
                                 start_new_session=True)
        pid_file.write_text(str(_proc.pid))
        launch_msg = mo.md(f"Started PID {_proc.pid}:\n\n`{' '.join(_cmd)}`")
    launch_msg
    return


@app.cell
def _(mo, stop, stop_pipeline):
    mo.stop(not stop.value)
    _pid = stop_pipeline()
    stop_msg = mo.md("No pipeline is running." if _pid is None else
                     f"Stopped PID {_pid}. Finished stages are kept; Start resumes from them.")
    stop_msg
    return


@app.cell
def _(mo):
    refresh = mo.ui.refresh(options=["15s", "30s", "1m", "5m"], default_interval="30s", label="Refresh status every")
    mo.md(f"## Progress\n\n{refresh}")
    return (refresh,)


@app.cell
def _(console_log, mo, refresh, run_dir, running_pid, subprocess):
    refresh
    import json as _json
    import time as _time

    _pid = running_pid()
    _state = f"🟢 running (PID {_pid})" if _pid else "⚪ not running"
    try:
        _gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "n/a"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        _gpu = "n/a"
    _rows = []
    for _p in sorted(run_dir.rglob("DONE.json")):
        try:
            _meta = _json.loads(_p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        _rows.append({"stage": str(_p.parent.relative_to(run_dir)).replace("\\", "/"), "finished": _meta.get("finished_at", ""),
                      "nodes": _meta.get("nodes", ""), "edges": _meta.get("edges", ""),
                      "preservation": round(_meta["preservation"], 3) if isinstance(_meta.get("preservation"), float) else "",
                      "status": _meta.get("status", "")})
    _errors = [str(_p.parent.relative_to(run_dir)).replace("\\", "/") for _p in run_dir.rglob("ERROR.json")
               if not (_p.parent / "DONE.json").exists()]
    _tail = ""
    for _f in (run_dir / "logs" / "pipeline.log", console_log):
        if _f.exists():
            _tail = "\n".join(_f.read_text(encoding="utf-8", errors="replace").splitlines()[-25:])
            break
    mo.vstack([
        mo.md(f"**Pipeline:** {_state} · **GPU (util, used, total):** {_gpu} · **finished stages:** {len(_rows)} · "
              f"**failed stages:** {len(_errors)} · updated {_time.strftime('%H:%M:%S')}"),
        mo.md("Failed: " + ", ".join(f"`{e}`" for e in _errors)) if _errors else mo.md(""),
        mo.ui.table(_rows[::-1], page_size=12, selection=None) if _rows else mo.md("_No finished stages yet._"),
        mo.md(f"```\n{_tail}\n```") if _tail else mo.md("_No log output yet._"),
    ])
    return


@app.cell
def _(mo):
    rebuild = mo.ui.run_button(label="Rebuild report from finished stages")
    mo.md(f"## Results\n\nTables put this run's measured values next to the paper's. {rebuild}")
    return (rebuild,)


@app.cell
def _(load_config, mo, profile, rebuild, refresh, repo_id, run_dir, run_name, runs_root):
    refresh
    if rebuild.value:
        from hagd.report import build_report

        build_report(run_dir, load_config(profile.value, {"run_name": run_name.value, "output_dir": str(runs_root),
                                                           "hub": {"repo_id": repo_id.value}}))
    _report = run_dir / "report" / "README.md"
    if _report.exists():
        _text = _report.read_text(encoding="utf-8")
        _text = _text.split("---", 2)[-1] if _text.startswith("---") else _text  # drop the dataset-card header
        _body = "\n".join(line for line in _text.splitlines() if not line.startswith("!["))
        _figs = [mo.image(src=str(f)) for f in sorted((run_dir / "report").glob("fig*.png"))]
        results_view = mo.vstack([mo.md(_body), *_figs])
    else:
        results_view = mo.md("_No report yet. It is written by the `report` phase at the end of a run, "
                             "or press the button above to build it from whatever has finished._")
    results_view
    return


if __name__ == "__main__":
    app.run()
