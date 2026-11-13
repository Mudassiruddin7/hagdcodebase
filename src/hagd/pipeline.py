"""Resumable experiment pipeline.

Phases (each is a set of stages; a finished stage has a ``DONE.json`` and is skipped on
re-runs, locally or after restoring from the Hub):

    labels    small "label" models: adapter, transcoders, attribution graphs and
              exhaustive-search circuits that serve as GNN training labels
    gnn       meta-train the GNN on those labelled graphs
    circuits  benchmark models: adapter, transcoders, graphs, HAGD circuits, baselines,
              Table V ablations, Fourier analysis, scalability measurements
    transfer  cross-architecture transfer coefficients (Eq. 11)
    complexity  operation-count study for Theorem 1
    report    tables, figures and the dataset card

A failing stage is logged and recorded in ``ERROR.json``; the pipeline moves on.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .adapt import attach_lora, finetune_adapter, plain_accuracy
from .attribution import AttributionGraph, build_graph
from .baselines import RUNNERS
from .complexity import complexity_study
from .config import Cfg, deep_merge, for_model, model_spec
from .data.tasks import get_task, task_batches
from .extract import TaskData, extract_circuit
from .fourier import fourier_analysis
from .gnn import CircuitGNN, leaf_auc, make_sample, train_gnn
from .hierarchy import build_hierarchy
from .hub import Hub, Stages
from .models import LM
from .replacement import FeatureModel, FeatureStats
from .train_transcoder import train_transcoders
from .transcoder import CrossLayerTranscoder
from .transfer import transfer_coefficient
from .utils import (free_memory, is_oom, load_json, log, peak_memory_gb, pick_device, reset_peak_memory, save_json,
                    set_seed, setup_logging)

PHASES = ["labels", "gnn", "circuits", "transfer", "complexity", "report"]


class ModelContext:
    """A loaded model with its adapter, transcoders and tokenised task data."""

    def __init__(self, pipe: "Pipeline", key: str, task_names: list[str]) -> None:
        self.pipe, self.key = pipe, key
        self.cfg = for_model(pipe.cfg, key)
        self.stages = pipe.stages
        self.task_names = task_names
        self.tasks = {n: get_task(n, self.cfg.data) for n in task_names}
        self.lm = LM.load(model_spec(pipe.cfg, key), pipe.device, self.cfg.get("quantization"))
        self.data: dict[str, TaskData] = {}
        self._tc: dict[bool, CrossLayerTranscoder] = {}
        self._prepare_adapter()

    # ---------------------------------------------------------------- adapter
    def _prepare_adapter(self) -> None:
        a = self.cfg.adapt
        names = [n for n in a.tasks if n in self.tasks and self.tasks[n].uses_adapter]
        if a.mode != "lora" or not names:
            return
        rel = f"{self.key}/adapter"
        adapter = attach_lora(self.lm, int(a.rank), float(a.alpha))
        if self.stages.done(rel) and (self.stages.run_dir / rel / "adapter.safetensors").exists():
            meta = adapter.load(self.stages.run_dir / rel)
            log.info("%s: adapter restored (val acc %s)", self.key, meta.get("val_acc"))
            return
        log.info("%s: fine-tuning LoRA adapter on %s", self.key, names)
        meta = finetune_adapter(self.lm, [self.tasks[n] for n in names], self.cfg, int(self.cfg.seed))
        adapter.save(self.stages.dir(rel), meta)
        self.stages.finish(rel, {"val_acc": meta["val_acc"], "steps": meta["steps"]}, checkpoints=True)

    # ---------------------------------------------------------------- transcoders
    def transcoder(self, cross_layer: bool = True) -> CrossLayerTranscoder:
        if cross_layer not in self._tc:
            rel = f"{self.key}/transcoder" if cross_layer else f"{self.key}/transcoder_nocross"
            d = self.stages.run_dir / rel
            if self.stages.done(rel) and (d / "transcoder.safetensors").exists():
                tc = CrossLayerTranscoder.load(d, self.lm.device)
            else:
                tc = train_transcoders(self.lm, self.cfg, self.stages, rel, list(self.tasks.values()), cross_layer,
                                       int(self.cfg.seed))
                self.stages.finish(rel, load_json(d / "eval.json"), checkpoints=True)
            tc.eval()
            tc.requires_grad_(False)
            self._tc[cross_layer] = tc
        return self._tc[cross_layer]

    def fm(self, task_name: str, cross_layer: bool = True) -> FeatureModel:
        on = bool(self.tasks[task_name].uses_adapter and self.lm.adapter is not None)
        return FeatureModel(self.lm, self.transcoder(cross_layer), adapter_on=on,
                            positions=str(self.cfg.validation.get("positions", "answer")))

    # ---------------------------------------------------------------- data
    def task_data(self, name: str) -> TaskData:
        if name not in self.data:
            d, t = self.cfg.data, self.tasks[name]
            mk = lambda split, n: task_batches(self.lm, t, split, int(n), int(d.batch_size), int(d.max_len), int(self.cfg.seed))  # noqa: E731
            self.data[name] = TaskData(t, mk("discover", d.n_discover), mk("val", d.n_val), mk("test", d.n_test))
        return self.data[name]

    def close(self) -> None:
        self.lm.model.to("cpu") if not self.cfg.get("quantization") else None
        del self.lm, self._tc
        free_memory()


class Pipeline:
    def __init__(self, cfg: Cfg) -> None:
        self.cfg = cfg
        self.run_dir = Path(cfg.output_dir) / cfg.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(self.run_dir / "logs" / "pipeline.log")
        self.device = pick_device(cfg.device)
        self.hub = Hub(cfg.hub.repo_id, self.run_dir, cfg.run_name, bool(cfg.hub.enabled), bool(cfg.hub.upload_checkpoints))
        self.hub.ensure_repo()
        self.stages = Stages(self.run_dir, self.hub)
        save_json(self.run_dir / "config.json", cfg.to_dict())
        self.hub.push("config.json")
        set_seed(int(cfg.seed))
        log.info("run %s | device %s | models %s | label models %s", cfg.run_name, self.device, list(cfg.models),
                 list(cfg.label_models))

    # ================================================================ helpers
    def _guard(self, rel: str, fn) -> dict | None:
        """Run a stage body, recording (instead of propagating) failures."""
        try:
            return fn()
        except KeyboardInterrupt:
            raise
        except Exception as err:  # noqa: BLE001
            status = "OOM" if is_oom(err) else "error"
            log.error("stage %s failed (%s): %s\n%s", rel, status, err, traceback.format_exc()[-2000:])
            save_json(self.stages.dir(rel) / "ERROR.json", {"status": status, "error": str(err)[:2000],
                                                             "time": time.strftime("%Y-%m-%d %H:%M:%S")})
            self.hub.push(rel)
            free_memory()
            return None

    def _task_fvu(self, fm: FeatureModel, batches) -> list[float]:
        """Fraction of variance the transcoders leave unexplained on this task's tokens, per layer."""
        num = [0.0] * fm.n_layers
        sq = [0.0] * fm.n_layers
        s1 = [None] * fm.n_layers
        n = 0
        with torch.no_grad(), fm.lm.adapter_on(fm.adapter_on):
            for b in batches:
                b = b.to(fm.lm.device)
                hs = fm.lm.hidden_states(b.input_ids, b.attention_mask, fm.layers)
                tm = b.token_mask()
                for li, h in enumerate(hs):
                    x = h[tm]
                    num[li] += float((x - fm.tc.decode(fm.tc.encode(x, li), li)).pow(2).sum())
                    sq[li] += float(x.pow(2).sum())
                    s1[li] = x.sum(0) if s1[li] is None else s1[li] + x.sum(0)
                n += int(tm.sum())
        return [num[li] / max(sq[li] - float(s1[li].pow(2).sum()) / max(n, 1), 1e-9) for li in range(fm.n_layers)]

    def graph(self, ctx: ModelContext, task: str, cross_layer: bool = True) -> tuple[AttributionGraph, FeatureStats] | None:
        rel = f"{ctx.key}/{'graphs' if cross_layer else 'graphs_nocross'}/{task}"
        d = self.stages.run_dir / rel
        if self.stages.done(rel):
            return AttributionGraph.load(d), FeatureStats.load(d / "stats.npz", ctx.lm.device)

        def body():
            fm = ctx.fm(task, cross_layer)
            data = ctx.task_data(task)
            graph, na = build_graph(fm, data.discover, ctx.cfg.attribution, ctx.key, task)
            graph.info["task_fvu"] = self._task_fvu(fm, data.discover)
            graph.info["adapter_on"] = fm.adapter_on
            graph.save(self.stages.dir(rel))
            na.stats.save(d / "stats.npz")
            self.stages.finish(rel, {k: v for k, v in graph.info.items() if not isinstance(v, list)})
            return graph, na.stats

        return self._guard(rel, body)

    def circuit(self, ctx: ModelContext, task: str, name: str, runner, cross_layer: bool = True) -> dict | None:
        """Run (or load) one circuit-discovery method.  ``runner(fm, graph, stats, data)`` returns the record."""
        rel = f"{ctx.key}/circuits/{task}/{name}"
        d = self.stages.run_dir / rel
        if self.stages.done(rel):
            return load_json(d / "circuit.json")
        g = self.graph(ctx, task, cross_layer)
        if g is None:
            return None
        graph, stats = g

        def body():
            reset_peak_memory()
            res = runner(ctx.fm(task, cross_layer), graph, stats, ctx.task_data(task))
            res["name"] = name
            save_json(self.stages.dir(rel) / "circuit.json", res)
            m = res.get("metrics", {})
            self.stages.finish(rel, {"nodes": m.get("nodes"), "edges": m.get("edges"), "preservation": m.get("preservation"),
                                     "seconds": res.get("seconds", {}).get("total"), "status": res.get("status", "ok")})
            log.info("circuit %s/%s/%s: %s nodes, %s edges, phi=%s (full acc %s), %.0fs", ctx.key, task, name,
                     m.get("nodes"), m.get("edges"), _fmt(m.get("preservation")), _fmt(m.get("acc_full")),
                     res.get("seconds", {}).get("total", 0.0))
            return res

        return self._guard(rel, body)

    # ================================================================ phases
    def phase_labels(self, only: list[str] | None = None) -> None:
        tasks = list(self.cfg.tasks) + list(self.cfg.aux_tasks)
        for key in self.cfg.label_models:
            if only and key not in only:
                continue
            todo = [t for t in tasks if not self.stages.done(f"{key}/labels/{t}")]
            if not todo:
                continue
            ctx = self._open(key, tasks)
            if ctx is None:
                continue
            for task in todo:
                rel = f"{key}/labels/{task}"
                g = self.graph(ctx, task)
                if g is None:
                    continue
                graph, stats = g

                def body(graph=graph, stats=stats, task=task, rel=rel):
                    lcfg = Cfg.wrap(deep_merge(ctx.cfg.to_dict(), {"baselines": {"exhaustive": {
                        "max_minutes": float(ctx.cfg.gnn.get("label_minutes", 30))}}}))
                    res = RUNNERS["exhaustive"](ctx.fm(task), graph, stats, ctx.task_data(task), lcfg,
                                                int(ctx.cfg.seed), pool=int(ctx.cfg.gnn.label_pool))
                    save_json(self.stages.dir(rel) / "circuit.json", res)
                    self.stages.finish(rel, {"nodes": res["metrics"]["nodes"], "acc_full": res["metrics"]["acc_full"],
                                             "preservation": res["metrics"]["preservation"], "status": res["status"],
                                             "val_score": res["val_score"]})

                self._guard(rel, body)
            ctx.close()

    def _label_samples(self, exclude_model: str | None = None) -> list:
        samples = []
        tasks = list(self.cfg.tasks) + list(self.cfg.aux_tasks)
        h = self.cfg.hierarchy
        for key in self.cfg.label_models:
            if key == exclude_model:
                continue
            for task in tasks:
                if not (self.stages.done(f"{key}/labels/{task}") and self.stages.done(f"{key}/graphs/{task}")):
                    continue
                lab = load_json(self.run_dir / key / "labels" / task / "circuit.json")
                acc, chance = lab["metrics"]["acc_full"], lab.get("chance", 0.0)
                if acc < chance + 0.1 * (1 - chance):
                    log.info("gnn: skipping labels %s/%s (model accuracy %.3f is at chance)", key, task, acc)
                    continue
                if lab.get("status") == "unreachable":
                    log.info("gnn: skipping labels %s/%s (no circuit reached theta)", key, task)
                    continue
                graph = AttributionGraph.load(self.run_dir / key / "graphs" / task)
                hier = build_hierarchy(graph.n, graph.edge_src, graph.edge_dst, graph.edge_w, int(h.branching),
                                       int(h.embed_dim), int(h.max_cluster_factor), int(self.cfg.seed), graph.node_layer)
                samples.append(make_sample(graph, hier, np.array([n["vertex"] for n in lab["nodes"]], dtype=np.int64)))
        return samples

    def phase_gnn(self) -> None:
        variants = {"all": None}
        for key in self.cfg.models:
            if key in self.cfg.label_models:
                variants[f"holdout_{key}"] = key
        for name, exclude in variants.items():
            rel = f"gnn/{name}"
            if self.stages.done(rel) and self.stages.result(rel).get("trained"):
                continue  # an untrained GNN (no labels were available yet) is retried

            def body(name=name, exclude=exclude, rel=rel):
                samples = self._label_samples(exclude)
                if not samples:
                    log.warning("gnn/%s: no labelled graphs yet; not training (will retry on the next run)", name)
                    return
                model, info = train_gnn(samples, self.cfg.gnn, int(self.cfg.seed))
                if exclude:  # generalisation to the held-out model's graphs
                    held = [s for s in self._label_samples(None) if s.model == exclude]
                    info["heldout_auc"] = leaf_auc(model, held)
                    info["heldout_model"] = exclude
                model.save(self.stages.dir(rel), info)
                self.stages.finish(rel, {k: v for k, v in info.items() if k != "history"}, checkpoints=True)

            self._guard(rel, body)

    def gnn_for(self, model_key: str) -> CircuitGNN | None:
        for name in (f"holdout_{model_key}", "all"):
            d = self.run_dir / "gnn" / name
            if self.stages.done(f"gnn/{name}") and (d / "gnn.safetensors").exists():
                gnn = CircuitGNN.load(d)
                if gnn.meta.get("trained"):
                    gnn.name = name
                    gnn.saw_target_model = any(g.startswith(f"{model_key}/") for g in gnn.meta.get("graphs", []))
                    if gnn.saw_target_model:
                        log.warning("GNN for %s was trained on this model's own exhaustive-search labels "
                                    "(no other label model is available)", model_key)
                    return gnn
        log.warning("no trained GNN available for %s: falling back to attribution-magnitude guidance", model_key)
        return None

    def phase_circuits(self, only: list[str] | None = None) -> None:
        cfg = self.cfg
        abl_model = cfg.ablations.model or (cfg.models[0] if cfg.models else None)
        for key in cfg.models:
            if only and key not in only:
                continue
            ctx = self._open(key, list(cfg.tasks))
            if ctx is None:
                continue
            reset_peak_memory()
            t_model = time.time()
            gnn = self.gnn_for(key)
            mcfg = ctx.cfg
            for task in cfg.tasks:
                # ---- HAGD
                for seed in range(int(mcfg.search.seeds)):
                    name = "hierarchical" if seed == 0 else f"hierarchical_seed{seed}"
                    self.circuit(ctx, task, name, lambda fm, g, s, d, seed=seed: extract_circuit(
                        fm, g, s, d, gnn, mcfg, "full", int(mcfg.seed) + seed))
                # ---- baselines
                if task in mcfg.baselines.tasks:
                    for method in mcfg.baselines.methods:
                        allowed = (mcfg.baselines.get(method) or {}).get("models")
                        if allowed is not None and key not in allowed:
                            continue
                        self.circuit(ctx, task, method, lambda fm, g, s, d, method=method: RUNNERS[method](
                            fm, g, s, d, mcfg, int(mcfg.seed)))
                # ---- Table V ablations
                if key == abl_model and task in mcfg.ablations.tasks:
                    for variant in mcfg.ablations.configs:
                        if variant == "full":
                            continue
                        cross = variant != "no_cross_layer"
                        self.circuit(ctx, task, f"ablation_{variant}", lambda fm, g, s, d, variant=variant: extract_circuit(
                            fm, g, s, d, gnn, mcfg, variant, int(mcfg.seed)), cross_layer=cross)
            # ---- Fourier structure of the modular-arithmetic circuit
            if bool(mcfg.fourier.enabled) and "mod_arith" in cfg.tasks:
                self._fourier(ctx)
            self._scalability(ctx, time.time() - t_model)
            ctx.close()

    def _fourier(self, ctx: ModelContext) -> None:
        rel = f"{ctx.key}/analysis/fourier"
        circ_dir = self.run_dir / ctx.key / "circuits" / "mod_arith" / "hierarchical"
        if self.stages.done(rel) or not (circ_dir / "circuit.json").exists():
            return

        def body():
            circ = load_json(circ_dir / "circuit.json")
            graph = AttributionGraph.load(self.run_dir / ctx.key / "graphs" / "mod_arith")
            res = fourier_analysis(ctx.fm("mod_arith"), graph, np.array([n["vertex"] for n in circ["nodes"]], dtype=np.int64),
                                   ctx.tasks["mod_arith"], ctx.cfg.data, int(ctx.cfg.fourier.n_examples), seed=int(ctx.cfg.seed))
            save_json(self.stages.dir(rel) / "fourier.json", res)
            self.stages.finish(rel, {k: v for k, v in res.items() if k != "features"})

        self._guard(rel, body)

    def _scalability(self, ctx: ModelContext, session_seconds: float) -> None:
        """Table VI inputs: memory, extraction time and attribution throughput for this model."""
        rel = f"{ctx.key}/scalability"
        rows = []
        for task in self.cfg.tasks:
            c = self.run_dir / ctx.key / "circuits" / task / "hierarchical" / "circuit.json"
            g = self.run_dir / ctx.key / "graphs" / task / "graph.json"
            if c.exists() and g.exists():
                circ, info = load_json(c), load_json(g)["info"]
                rows.append({"task": task, "seconds": circ["seconds"]["total"], "peak_memory_gb": circ.get("peak_memory_gb", 0.0),
                             "features_per_second": info["features_per_second"], "edges_per_second": info["edges_per_second"]})
        if not rows:
            return
        prev = load_json(self.run_dir / rel / "scalability.json") if (self.run_dir / rel / "scalability.json").exists() else {}
        spec = model_spec(self.cfg, ctx.key)
        out = {
            "model": ctx.key, "params": spec.params, "n_params_loaded": ctx.lm.n_params(),
            "layers_instrumented": len(ctx.transcoder().layers), "dictionary_size": ctx.transcoder().m,
            # peak over every session that touched this model (a resumed session only sees part of the work)
            "peak_memory_gb": max(peak_memory_gb(), prev.get("peak_memory_gb", 0.0), max(r["peak_memory_gb"] for r in rows)),
            "extraction_hours_per_task": float(np.mean([r["seconds"] for r in rows])) / 3600,
            "extraction_hours_all_tasks": float(np.sum([r["seconds"] for r in rows])) / 3600,
            "features_per_second": float(np.mean([r["features_per_second"] for r in rows])),
            "edges_per_second": float(np.mean([r["edges_per_second"] for r in rows])),
            "device": str(self.device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "tasks": rows,
        }
        save_json(self.stages.dir(rel) / "scalability.json", out)
        self.stages.finish(rel, {k: v for k, v in out.items() if k != "tasks"})

    def phase_transfer(self) -> None:
        t = self.cfg.transfer
        pairs = [tuple(p) for p in t.pairs]
        if not pairs:
            return
        rel = "analysis/transfer"
        rows = []
        for src, dst in pairs:
            for task in t.tasks:
                paths = [self.run_dir / m / "circuits" / task / "hierarchical" for m in (src, dst)]
                if not all(self.stages.done(f"{m}/circuits/{task}/hierarchical") and self.stages.done(f"{m}/graphs/{task}")
                           for m in (src, dst)):
                    log.info("transfer %s -> %s on %s: circuits not available yet", src, dst, task)
                    continue
                c1, c2 = (load_json(p / "circuit.json") for p in paths)
                if not c1.get("nodes") or not c2.get("nodes"):
                    continue
                g1 = AttributionGraph.load(self.run_dir / src / "graphs" / task)
                g2 = AttributionGraph.load(self.run_dir / dst / "graphs" / task)
                res = transfer_coefficient(c1, g1.anchors, len(g1.layers), c2, g2.anchors, len(g2.layers), float(t.min_corr))
                rows.append({"source": src, "target": dst, "task": task, **res})
                log.info("transfer %s -> %s on %s: tau = %s (%d aligned features)", src, dst, task, _fmt(res["tau"]), res["aligned"])
        save_json(self.stages.dir(rel) / "transfer.json", rows)
        self.stages.finish(rel, {"pairs": len(rows)})

    def phase_complexity(self) -> None:
        rel = "analysis/complexity"
        if self.stages.done(rel):
            return

        def body():
            c, h = self.cfg.complexity, self.cfg.hierarchy
            res = complexity_study([int(s) for s in c.sizes], int(c.trials), int(h.branching), int(h.embed_dim), int(self.cfg.seed))
            save_json(self.stages.dir(rel) / "complexity.json", res)
            self.stages.finish(rel, {k: v for k, v in res.items() if k != "rows"})

        self._guard(rel, body)

    def phase_report(self) -> None:
        from .report import build_report

        build_report(self.run_dir, self.cfg)
        self.hub.push("report")
        self.hub.push("logs")
        card = self.run_dir / "report" / "README.md"
        if card.exists():
            self.hub.push_file_to_root(card, "README.md", f"results summary ({self.cfg.run_name})")

    # ================================================================ driver
    def _open(self, key: str, tasks: list[str]) -> ModelContext | None:
        try:
            ctx = ModelContext(self, key, tasks)
            ctx.transcoder(True)
            return ctx
        except KeyboardInterrupt:
            raise
        except Exception as err:  # noqa: BLE001
            log.error("could not prepare model %s: %s\n%s", key, err, traceback.format_exc()[-2000:])
            save_json(self.stages.dir(f"{key}") / "ERROR.json", {"status": "OOM" if is_oom(err) else "error", "error": str(err)[:2000]})
            free_memory()
            return None

    def run(self, phases: list[str] | None = None, models: list[str] | None = None) -> None:
        phases = phases or PHASES
        t0 = time.time()
        for phase in PHASES:
            if phase not in phases:
                continue
            log.info("=== phase: %s ===", phase)
            if phase == "labels":
                self.phase_labels(models)
            elif phase == "gnn":
                self.phase_gnn()
            elif phase == "circuits":
                self.phase_circuits(models)
            elif phase == "transfer":
                self.phase_transfer()
            elif phase == "complexity":
                self.phase_complexity()
            elif phase == "report":
                self.phase_report()
        log.info("pipeline finished in %.1f min; artifacts in %s", (time.time() - t0) / 60, self.run_dir)


def _fmt(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.3f}"
