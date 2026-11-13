"""Tables, figures and the dataset card.

Every table mirrors one of the paper's and puts the value this run measured next to the
value the paper reports.  Measured numbers come exclusively from the circuit records
produced by the pipeline; paper numbers come from ``hagd.paper_reference``.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from . import paper_reference as paper
from .config import ALGORITHMIC_TASKS, LANGUAGE_TASKS, MODEL_REGISTRY
from .utils import load_json, log, save_json

TASK_LABEL = {"mod_arith": "Mod. Arith.", "parity": "Parity", "sorting": "Sorting", "winogrande": "WinoGrande",
              "hellaswag": "HellaSwag", "factual_qa": "Factual QA"}
METHOD_LABEL = {"exhaustive": "Exhaustive", "acdc": "ACDC", "hierarchical": "Hierarchical",
                "attribution_patching": "Attr. patching", "sparse_probing": "Sparse probing"}
ABLATION_LABEL = {"full": "Full Framework", "no_hierarchy": "No Hierarchy", "no_gnn": "No GNN Guidance",
                  "no_cross_layer": "No Cross-Layer", "no_causal_validation": "No Causal Valid."}


def _name(key: str) -> str:
    return MODEL_REGISTRY[key].paper_name if key in MODEL_REGISTRY else key


def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _f(x, nd: int = 2) -> str:
    """Format a real number (or pass a string through)."""
    if _missing(x):
        return "—"
    if isinstance(x, str):
        return x
    return f"{float(x):,.{nd}f}"


def _i(x) -> str:
    """Format a count."""
    if _missing(x):
        return "—"
    if isinstance(x, str):
        return x
    return f"{int(round(float(x))):,}"


def collect_circuits(run_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(run_dir.glob("*/circuits/*/*/circuit.json")):
        r = load_json(path)
        m = r.get("metrics", {})
        rows.append({
            "model": r.get("model"), "task": r.get("task"), "name": r.get("name", path.parent.name),
            "method": r.get("method"), "status": r.get("status", "ok"), "variant": r.get("variant"),
            "nodes": m.get("nodes"), "edges": m.get("edges") if r.get("edges_defined", True) else None,
            "preservation": m.get("preservation"),
            "ci_lo": (m.get("preservation_ci95") or [None, None])[0], "ci_hi": (m.get("preservation_ci95") or [None, None])[1],
            "preservation_strict": m.get("preservation_strict"), "preservation_empty": m.get("preservation_empty"),
            "preservation_lenient": m.get("preservation_lenient"), "preservation_empty_lenient": m.get("preservation_empty_lenient"),
            "error_mode": m.get("error_mode"),
            "faithfulness": m.get("faithfulness_loss"), "acc_full": m.get("acc_full"), "acc_circuit": m.get("acc_circuit"),
            "chance": r.get("chance"), "modularity": m.get("modularity"), "newman_q": m.get("newman_q"),
            "intervention_effect": m.get("intervention_effect"), "kl": m.get("reconstruction_error_kl"),
            "concept_purity": m.get("concept_purity"), "description_length_bits": m.get("description_length_bits"),
            "compression": m.get("compression_nodes"), "seconds": r.get("seconds", {}).get("total"),
            "reached_theta": r.get("reached_theta"), "gnn_guided": r.get("gnn_guided"),
            "peak_memory_gb": r.get("peak_memory_gb"),
        })
    return pd.DataFrame(rows)


def _md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No results yet._\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _order(keys) -> list[str]:
    reg = list(MODEL_REGISTRY)
    return sorted(set(keys), key=lambda k: reg.index(k) if k in reg else 99)


# ---------------------------------------------------------------------------- tables


def table2(df: pd.DataFrame, task: str = "mod_arith") -> pd.DataFrame:
    out = []
    d = df[(df.task == task) & df.name.isin(METHOD_LABEL)] if not df.empty else df
    for model in _order(d.model) if not d.empty else []:
        for method in METHOD_LABEL:
            r = d[(d.model == model) & (d.name == method)]
            if r.empty:
                continue
            r = r.iloc[0]
            ref = paper.TABLE2.get((model, method)) if task == "mod_arith" else None
            ok = r.status == "ok"
            out.append({
                "Model": _name(model), "Method": METHOD_LABEL[method],
                "Nodes": _i(r.nodes), "Edges": _i(r.edges),
                "Pres.": _f(r.preservation), "95% CI": f"[{_f(r.ci_lo)}, {_f(r.ci_hi)}]",
                "Empty-circuit pres.": _f(r.preservation_empty), "Loss gap recovered": _f(r.faithfulness), "Pres. (strict)": _f(r.preservation_strict),
                "Full acc.": _f(r.acc_full),
                "Time (s)": _i(r.seconds) if ok else f"{r.status} ({_i(r.seconds)} s)",
                "Paper nodes": _i(ref[0]) if ref else "—", "Paper edges": _i(ref[1]) if ref else "—",
                "Paper pres.": _f(ref[2]) if ref else "—", "Paper time (s)": _i(ref[3]) if ref else "—",
            })
    return pd.DataFrame(out)


def table3(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    d = df[df.task.isin(LANGUAGE_TASKS) & (df.name == "hierarchical")] if not df.empty else df
    for task in LANGUAGE_TASKS:
        for model in _order(d.model) if not d.empty else []:
            r = d[(d.model == model) & (d.task == task)]
            if r.empty:
                continue
            r = r.iloc[0]
            ref = paper.TABLE3.get((model, task))
            out.append({
                "Model": _name(model), "Task": TASK_LABEL[task], "Nodes": _i(r.nodes), "Edges": _i(r.edges),
                "Pres.": _f(r.preservation), "95% CI": f"[{_f(r.ci_lo)}, {_f(r.ci_hi)}]",
                "Empty-circuit pres.": _f(r.preservation_empty), "Loss gap recovered": _f(r.faithfulness), "Pres. (strict)": _f(r.preservation_strict),
                "Mod.": _f(r.modularity), "Full acc.": _f(r.acc_full), "Chance": _f(r.chance),
                "Paper nodes": _i(ref[0]) if ref else "—", "Paper edges": _i(ref[1]) if ref else "—",
                "Paper pres.": _f(ref[2]) if ref else "—", "Paper mod.": _f(ref[3]) if ref else "—",
            })
    return pd.DataFrame(out)


def _family_key(src: str, dst: str) -> tuple[str, str]:
    fs, fd = MODEL_REGISTRY[src].family, MODEL_REGISTRY[dst].family
    return (src, dst) if fs == fd else (fs, fd)


def table4(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "analysis" / "transfer" / "transfer.json"
    if not path.exists():
        return pd.DataFrame()
    rows = load_json(path)
    out = {}
    for r in rows:
        key = (r["source"], r["target"])
        rec = out.setdefault(key, {"Source": _name(r["source"]), "Target": _name(r["target"])})
        rec[TASK_LABEL.get(r["task"], r["task"])] = _f(r["tau"])
        ref = paper.TABLE4.get(_family_key(*key), {}).get(r["task"])
        rec[f"Paper {TASK_LABEL.get(r['task'], r['task'])}"] = _f(ref)
        rec.setdefault("Aligned features", "")
        rec["Aligned features"] = (rec["Aligned features"] + " " + str(r["aligned"])).strip()
    return pd.DataFrame(list(out.values()))


def table5(df: pd.DataFrame, cfg) -> pd.DataFrame:
    if df.empty:
        return df
    model = cfg.ablations.model or (cfg.models[0] if cfg.models else None)
    tasks = [t for t in ("mod_arith", "factual_qa") if t in cfg.ablations.tasks] or list(cfg.ablations.tasks)
    d = df[df.model == model]
    base_time = {t: d[(d.task == t) & (d.name == "hierarchical")].seconds.mean() for t in tasks}
    out = []
    for variant in cfg.ablations.configs:
        name = "hierarchical" if variant == "full" else f"ablation_{variant}"
        rec = {"Configuration": ABLATION_LABEL.get(variant, variant)}
        ratios = []
        for t in tasks:
            r = d[(d.task == t) & (d.name == name)]
            rec[f"{TASK_LABEL[t]} pres."] = _f(r.preservation.iloc[0]) if not r.empty else "—"
            rec[f"{TASK_LABEL[t]} nodes"] = _i(r.nodes.iloc[0]) if not r.empty else "—"
            if not r.empty and base_time.get(t):
                ratios.append(r.seconds.iloc[0] / base_time[t])
        rec["Time"] = f"{sum(ratios) / len(ratios):.2f}×" if ratios else "—"
        ref = paper.TABLE5.get(variant)
        rec["Paper mod. pres."] = _f(ref[0]) if ref else "—"
        rec["Paper QA pres."] = _f(ref[1]) if ref else "—"
        rec["Paper time"] = f"{ref[2]}×" if ref else "—"
        out.append(rec)
    res = pd.DataFrame(out)
    res.attrs["model"] = model
    return res


def table6(run_dir: Path) -> pd.DataFrame:
    out = []
    files = {p.parent.parent.name: p for p in run_dir.glob("*/scalability/scalability.json")}
    for model in _order(files):
        s = load_json(files[model])
        ref = paper.TABLE6[paper.scale_bucket(s["params"])]
        out.append({
            "Model": _name(model), "Scale": paper.scale_bucket(s["params"]), "Mem (GB)": _f(s["peak_memory_gb"], 1),
            "Time (hr)": _f(s["extraction_hours_per_task"], 3), "Feat/s": _i(s["features_per_second"]),
            "Edge/s": _i(s["edges_per_second"]), "Layers": s["layers_instrumented"], "Dict. size": _i(s["dictionary_size"]),
            "Paper mem (GB)": _i(ref[0]), "Paper time (hr)": _f(ref[1]), "Paper feat/s": _i(ref[2]), "Paper edge/s": _i(ref[3]),
        })
    return pd.DataFrame(out)


def table_transcoders(run_dir: Path) -> pd.DataFrame:
    out = []
    for path in sorted(run_dir.glob("*/transcoder*/eval.json")):
        e = load_json(path)
        task_fvu = {}
        for g in (path.parent.parent / ("graphs" if path.parent.name == "transcoder" else "graphs_nocross")).glob("*/graph.json"):
            fv = load_json(g)["info"].get("task_fvu")
            if fv:
                task_fvu[g.parent.name] = sum(fv) / len(fv)
        out.append({
            "Model": _name(e["model"]), "Variant": "cross-layer" if e["cross_layer"] else "no cross-layer",
            "Layers": len(e["layers"]), "Dict. size m": _i(e["m"]), "k": e["k"],
            "Tokens": f"{e['tokens_seen'] / 1e6:.1f}M", "Corpus": e.get("corpus") or "—",
            "Unexplained var. (corpus)": _f(e["mean_fvu"], 3),
            "Unexplained var. (tasks)": _f(sum(task_fvu.values()) / len(task_fvu), 3) if task_fvu else "—",
            "Pred. head R²": _f(sum(e["eval"]["pred_r2"]) / len(e["eval"]["pred_r2"]), 3) if e["eval"].get("pred_r2") else "—",
            "Dead features": _f(sum(e["eval"]["dead_fraction"]) / len(e["eval"]["dead_fraction"]), 3),
            "Budget hit": "yes" if e.get("budget_hit") else "no",
        })
    return pd.DataFrame(out)


def table_methods(df: pd.DataFrame) -> pd.DataFrame:
    """Every method on every (model, task) it ran on - the unabridged version of Table II."""
    if df.empty:
        return df
    d = df[df.name.isin(METHOD_LABEL)].copy()
    out = []
    for _, r in d.sort_values(["task", "model", "name"]).iterrows():
        out.append({"Model": _name(r.model), "Task": TASK_LABEL.get(r.task, r.task), "Method": METHOD_LABEL[r["name"]],
                    "Nodes": _i(r.nodes), "Edges": _i(r.edges), "Pres.": _f(r.preservation),
                    "Empty-circuit pres.": _f(r.preservation_empty), "Loss gap recovered": _f(r.faithfulness), "Pres. (strict)": _f(r.preservation_strict),
                    "Faithfulness": _f(r.faithfulness), "Interv. effect": _f(r.intervention_effect), "KL": _f(r.kl, 3),
                    "Purity": _f(r.concept_purity), "Mod.": _f(r.modularity), "DL (bits)": _i(r.description_length_bits),
                    "Full acc.": _f(r.acc_full), "Time (s)": _i(r.seconds), "Status": r.status})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------- figures


def _figures(run_dir: Path, out: Path, df: pd.DataFrame) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    colors = {"mod_arith": "#1f77b4", "parity": "#17becf", "sorting": "#9467bd", "factual_qa": "#2ca02c",
              "winogrande": "#ff7f0e", "hellaswag": "#d62728"}
    h = df[df.name == "hierarchical"] if not df.empty else df
    if not h.empty:  # Fig. 2
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for task, grp in h.groupby("task"):
            g = grp.dropna(subset=["nodes", "preservation"])
            if not g.empty:
                ax.scatter(g.nodes, g.preservation, label=TASK_LABEL.get(task, task), color=colors.get(task), s=36)
        ax.set_xlabel("Circuit Size (Nodes)")
        ax.set_ylabel("Preservation φ")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title("Fig. 2 — preservation vs. circuit size (measured)", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "fig2_preservation_vs_size.png", dpi=160)
        plt.close(fig)
        made.append("fig2_preservation_vs_size.png")

    tpath = run_dir / "analysis" / "transfer" / "transfer.json"
    if tpath.exists() and load_json(tpath):  # Fig. 3
        t = pd.DataFrame(load_json(tpath)).dropna(subset=["tau"])
        if not t.empty:
            mean = t.groupby(["source", "target"]).tau.mean()
            models = _order(set(t.source) | set(t.target))
            mat = pd.DataFrame(float("nan"), index=models, columns=models)
            for (s, d), v in mean.items():
                mat.loc[s, d] = v
                mat.loc[d, s] = v
            for m in models:
                mat.loc[m, m] = 1.0
            fig, ax = plt.subplots(figsize=(4.6, 3.8))
            im = ax.imshow(mat.values, vmin=0, vmax=1, cmap="viridis")
            ax.set_xticks(range(len(models)), [_name(m) for m in models], rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(models)), [_name(m) for m in models], fontsize=8)
            for i in range(len(models)):
                for j in range(len(models)):
                    if mat.values[i, j] == mat.values[i, j]:
                        ax.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
            ax.set_xlabel("Target")
            ax.set_ylabel("Source")
            fig.colorbar(im, label="Transfer coeff.")
            ax.set_title("Fig. 3 — transfer coefficients, task mean (measured)", fontsize=9)
            fig.tight_layout()
            fig.savefig(out / "fig3_transfer_matrix.png", dpi=160)
            plt.close(fig)
            made.append("fig3_transfer_matrix.png")

    cpath = run_dir / "analysis" / "complexity" / "complexity.json"
    d2 = df[(df.task == "mod_arith") & df.name.isin(["exhaustive", "acdc", "hierarchical"])] if not df.empty else df
    if cpath.exists() or not d2.empty:  # Fig. 4
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
        ax = axes[0]
        if not d2.empty:
            d2 = d2.dropna(subset=["seconds"]).copy()
            d2["params"] = d2.model.map(lambda k: MODEL_REGISTRY[k].params)
            base = d2[d2.name == "hierarchical"].sort_values("params").seconds
            base = float(base.iloc[0]) if len(base) else 1.0
            for method, grp in d2.groupby("name"):
                grp = grp.sort_values("params")
                done = grp[grp.status == "ok"]
                ax.plot(done.params, done.seconds / base, marker="o", label=METHOD_LABEL[method])
                cut = grp[grp.status != "ok"]
                if not cut.empty:
                    ax.scatter(cut.params, cut.seconds / base, marker="x", color="k", zorder=3)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend(fontsize=8)
        ax.set_xlabel("Model Parameters")
        ax.set_ylabel("Relative Time (measured)")
        ax.set_title("wall-clock, modular arithmetic (× = budget/OOM)", fontsize=9)
        ax.grid(alpha=0.3, which="both")
        ax = axes[1]
        if cpath.exists():
            c = load_json(cpath)
            rows = pd.DataFrame(c["rows"])
            ax.plot(rows.n, rows.hierarchical_ops, marker="o", label=f"Hierarchical (slope {c['exponent_hierarchical']:.2f})")
            ax.plot(rows.n, rows.bound_n2_log_n, "--", label="n² log n bound")
            ax.plot(rows.n, rows.acdc_ops, marker="s", label=f"ACDC (slope {c['exponent_acdc']:.2f})")
            small = rows[rows.n <= 64]
            ax.plot(small.n, 2.0 ** small.n, ":", label="Exhaustive 2ⁿ")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(top=float(rows.bound_n2_log_n.max()) * 100)
            ax.legend(fontsize=7)
        ax.set_xlabel("Graph vertices n")
        ax.set_ylabel("Operations (counted)")
        ax.set_title("operation counts on synthetic graphs", fontsize=9)
        ax.grid(alpha=0.3, which="both")
        fig.suptitle("Fig. 4 — complexity comparison", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "fig4_complexity.png", dpi=160)
        plt.close(fig)
        made.append("fig4_complexity.png")
    return made


# ---------------------------------------------------------------------------- report


def build_report(run_dir: Path, cfg) -> Path:
    run_dir = Path(run_dir)
    out = run_dir / "report"
    out.mkdir(parents=True, exist_ok=True)
    df = collect_circuits(run_dir)
    df.to_csv(out / "all_circuits.csv", index=False)
    tables = {
        "table2_modular_arithmetic": table2(df, "mod_arith"),
        "table2b_parity": table2(df, "parity"),
        "table2c_sorting": table2(df, "sorting"),
        "table3_language_tasks": table3(df),
        "table4_transfer": table4(run_dir),
        "table5_ablations": table5(df, cfg),
        "table6_scalability": table6(run_dir),
        "transcoder_quality": table_transcoders(run_dir),
        "all_methods": table_methods(df),
    }
    for name, t in tables.items():
        t.to_csv(out / f"{name}.csv", index=False)
    figs = _figures(run_dir, out, df)

    extras: dict = {}
    for key, pattern in (("fourier", "*/analysis/fourier/fourier.json"), ("adapters", "*/adapter/adapter.json")):
        extras[key] = {p.parents[2 if key == "fourier" else 1].name: load_json(p) for p in run_dir.glob(pattern)}
    gnn = {p.parent.name: load_json(p) for p in run_dir.glob("gnn/*/gnn.json")}
    cpath = run_dir / "analysis" / "complexity" / "complexity.json"
    complexity = load_json(cpath) if cpath.exists() else None
    errors = [{"stage": str(p.parent.relative_to(run_dir)).replace("\\", "/"), **load_json(p)} for p in run_dir.rglob("ERROR.json")
              if not (p.parent / "DONE.json").exists()]

    h = df[df.name == "hierarchical"] if not df.empty else df
    ap = df[df.name == "attribution_patching"] if not df.empty else df
    summary = {"run": cfg.run_name, "models": list(cfg.models), "label_models": list(cfg.label_models),
               "n_circuits": int(len(df)), "errors": errors}
    if not h.empty:
        alg, lang = h[h.task.isin(ALGORITHMIC_TASKS)], h[h.task.isin(LANGUAGE_TASKS)]
        rng = lambda s: [float(s.min()), float(s.max())] if len(s.dropna()) else None  # noqa: E731
        summary.update({"preservation_algorithmic": rng(alg.preservation), "preservation_language": rng(lang.preservation),
                        "circuit_nodes": rng(h.nodes), "sufficient_fraction": float((h.preservation > 0.9).mean())})
        both = h.merge(ap, on=["model", "task"], suffixes=("_h", "_ap"))
        if not both.empty:
            summary["attribution_patching_gap"] = float((both.preservation_h - both.preservation_ap).mean())
    save_json(out / "summary.json", {**summary, "gnn": gnn, "complexity": complexity,
                                     "fourier": {k: {kk: vv for kk, vv in v.items() if kk != "features"} for k, v in extras["fourier"].items()}})

    md = [_card_header(cfg), f"# HAGD replication — run `{cfg.run_name}`\n",
          "Measured results of the open implementation of *Hierarchical Sparse Circuit Extraction from Billion-Parameter "
          "Language Models through Scalable Attribution Graph Decomposition* (arXiv:2601.12879). "
          "Code: https://github.com/Mudassiruddin7/hagdcodebase\n",
          "Columns named **Paper …** are the values printed in the paper; every other number was measured by this run. "
          "How to read the faithfulness columns:\n",
          "- **Pres.** — Eq. 10, `Acc(M_C) / Acc(M_full)`, on a held-out test split. Inside `M_C` every feature outside "
          "the circuit is mean-ablated at the answer positions; the part of each hidden state the transcoders do not "
          "reconstruct is kept (the always-on \"error nodes\" of sparse feature circuits).",
          "- **Empty-circuit pres.** — the same ratio for a circuit with no features. Eq. 10 is an accuracy ratio, so a "
          "model that is near chance on a task scores high here for free; compare `Pres.` against this, not against 0.",
          "- **Loss gap recovered** — `(L(empty) − L(C)) / (L(empty) − L(full))`. The search requires both this and "
          "`Pres.` to reach θ, so an empty circuit cannot pass on a near-chance task.",
          "- **Pres. (strict)** — `Pres.` with the unreconstructed part also mean-ablated. With 15–30% unexplained "
          "variance per layer this is near 0 even for the full dictionary; it measures the transcoders' dark matter, "
          "not the circuit.\n",
          _scope(cfg, tables["transcoder_quality"]),
          "## Table II — circuit discovery on modular arithmetic\n", _md(tables["table2_modular_arithmetic"])]
    for key, title in (("table2b_parity", "Parity (not tabulated in the paper)"), ("table2c_sorting", "Sorting (not tabulated in the paper)")):
        if not tables[key].empty:
            md += [f"### {title}\n", _md(tables[key])]
    md += ["## Table III — circuit discovery on natural language tasks\n", _md(tables["table3_language_tasks"]),
           "## Table IV — cross-architecture transfer coefficients (Eq. 11)\n", _md(tables["table4_transfer"]),
           f"## Table V — ablation study ({_name(tables['table5_ablations'].attrs.get('model', '')) if not tables['table5_ablations'].empty else ''})\n",
           _md(tables["table5_ablations"]),
           "## Table VI — computational scalability\n", _md(tables["table6_scalability"]),
           "Feat/s = dictionary features attributed per second (all layers, per example); Edge/s = feature pairs whose "
           "Eq. 5 attribution is evaluated per second.\n",
           "## Transcoder reconstruction (\"dark matter\")\n",
           f"The paper reports {paper.HEADLINE['reconstruction_unexplained_variance']} unexplained variance.\n",
           _md(tables["transcoder_quality"])]
    if figs:
        md.append("## Figures\n")
        md += [f"![{f}](runs/{cfg.run_name}/report/{f})\n" for f in figs]
    if extras["fourier"]:
        md.append("## Fourier structure of modular-arithmetic circuits\n")
        md.append(_md(pd.DataFrame([{
            "Model": _name(k), "Circuit features": v["n_features"],
            "Fourier-structured (vs. permutation null)": _f(v.get("fraction_fourier_structured")),
            "Median top-frequency share": _f(v.get("median_top_share")), "Null 95th pct.": _f(v.get("mean_null_p95")),
            "Key frequencies": ", ".join(f"{a}×{b}" for a, b in sorted(v.get("key_frequencies", {}).items(), key=lambda kv: -kv[1])[:6]) or "—",
        } for k, v in extras["fourier"].items() if v.get("n_features")])))
    if complexity:
        md += ["## Theorem 1 — operation counts\n",
               f"Log-log slope of counted hierarchical-search operations vs. graph size n: **{complexity['exponent_hierarchical']:.2f}** "
               f"(Theorem 1 bound: 2 plus a log factor); supernode decisions: {complexity['exponent_decisions']:.2f} (predicted 1); "
               f"ACDC-style edge pruning: {complexity['exponent_acdc']:.2f}. Largest ratio to the n² log n bound: "
               f"{complexity['max_ratio_to_bound']:.3f}. Hierarchy depth / log_b n: {complexity['depth_over_log_b_n']:.2f}.\n"]
    if gnn:
        md.append("## GNN guidance\n")
        md.append(_md(pd.DataFrame([{"GNN": k, "Trained": v.get("trained"), "Label graphs": v.get("n_graphs", 0),
                                     "Train AUROC": _f(v.get("train_auc")), "Held-out model": v.get("heldout_model", "—"),
                                     "Held-out AUROC": _f(v.get("heldout_auc"))} for k, v in gnn.items()])))
    if extras["adapters"]:
        md.append("## Task adapters (LoRA) — validation accuracy of the adapted models\n")
        md.append(_md(pd.DataFrame([{"Model": _name(k), "Steps": v.get("steps"),
                                     **{TASK_LABEL.get(t, t): _f(a) for t, a in v.get("val_acc", {}).items() if t not in ("step", "loss")}}
                                    for k, v in extras["adapters"].items()])))
    md += ["## All methods, all metrics\n", _md(tables["all_methods"])]
    if errors:
        md += ["## Stages that failed\n", _md(pd.DataFrame(errors)[["stage", "status", "error"]])]
    (out / "README.md").write_text("\n".join(md), encoding="utf-8")
    log.info("report written to %s (%d circuits, %d figures)", out, len(df), len(figs))
    return out


def _card_header(cfg) -> str:
    return ("---\nlicense: mit\ntags:\n- mechanistic-interpretability\n- circuit-discovery\n- sparse-autoencoders\n"
            "pretty_name: HAGD circuit extraction results\n---\n")


def _scope(cfg, tq: pd.DataFrame) -> str:
    t = cfg.transcoder
    lines = ["## What this run covers\n",
             f"- Models: {', '.join(_name(m) for m in cfg.models)}; GNN label models: {', '.join(_name(m) for m in cfg.label_models)}.",
             f"- Transcoders: expansion m/d = {t.expansion}, token budget {int(t.num_tokens) / 1e6:.1f}M per model "
             f"(paper: 100M), layer stride {t.layer_stride} unless overridden per model.",
             f"- Tasks: {', '.join(TASK_LABEL.get(x, x) for x in cfg.tasks)}; discovery on {cfg.data.n_discover} examples, "
             f"validation on {cfg.data.n_val}, reporting on {cfg.data.n_test} held-out examples.",
             f"- Algorithmic tasks are taught with a LoRA adapter (mode: `{cfg.adapt.mode}`), because the pretrained "
             "checkpoints cannot perform them and Eq. 10 is otherwise undefined.",
             f"- Sufficiency threshold θ = {cfg.search.theta} on the `{cfg.search.get('criterion', 'combined')}` criterion; "
             f"interventions at `{cfg.validation.get('positions', 'answer')}` positions; ablation = {cfg.validation.ablation}; "
             f"transcoder residual = {cfg.validation.error_mode}.\n"]
    return "\n".join(lines)
