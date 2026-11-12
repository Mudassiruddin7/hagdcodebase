"""Command line entry point.

    hagd run --profile smoke                 # full pipeline for a profile
    hagd run --profile molab --models gpt2-small gpt2-medium --phases circuits report
    hagd report --profile molab              # rebuild tables / figures from stored results
    hagd status --profile molab              # list finished stages
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import CONFIG_DIR, load_config
from .pipeline import PHASES, Pipeline


def _overrides(args) -> dict:
    over: dict = {}
    if args.run_name:
        over["run_name"] = args.run_name
    if args.output_dir:
        over["output_dir"] = args.output_dir
    if args.no_hub:
        over["hub"] = {"enabled": False}
    if args.repo_id:
        over.setdefault("hub", {})["repo_id"] = args.repo_id
    for item in args.set or []:
        key, _, value = item.partition("=")
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        node = over
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return over


def main(argv: list[str] | None = None) -> int:
    profiles = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))
    ap = argparse.ArgumentParser(prog="hagd", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "report", "status"):
        p = sub.add_parser(name)
        p.add_argument("--profile", default="smoke", help=f"profile name ({', '.join(profiles)}) or path to a YAML file")
        p.add_argument("--run-name", default=None)
        p.add_argument("--output-dir", default=None)
        p.add_argument("--repo-id", default=None, help="Hugging Face dataset repo for artifacts")
        p.add_argument("--no-hub", action="store_true", help="keep everything local")
        p.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="override config values, e.g. transcoder.num_tokens=5e6")
        if name == "run":
            p.add_argument("--models", nargs="*", default=None, help="restrict to these model keys")
            p.add_argument("--phases", nargs="*", default=None, choices=PHASES)
    args = ap.parse_args(argv)
    cfg = load_config(args.profile, _overrides(args))
    if args.cmd == "run" and args.models:
        cfg["models"] = [m for m in cfg.models if m in args.models] or list(args.models)
    pipe = Pipeline(cfg)
    if args.cmd == "run":
        pipe.run(args.phases, None)
    elif args.cmd == "report":
        pipe.phase_report()
    else:
        for rel in pipe.stages.all_done():
            print(rel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
