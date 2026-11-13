"""Configuration: model registry (paper Table I), defaults, and YAML profiles.

A config is a nested ``Cfg`` (dict with attribute access).  ``DEFAULTS`` documents every
knob; a profile YAML (``src/hagd/configs/<name>.yaml``) overrides it, and
``model_overrides.<model-key>`` overrides it again for a single model.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).parent / "configs"


class Cfg(dict):
    """dict with attribute access, recursively."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @staticmethod
    def wrap(obj: Any) -> Any:
        if isinstance(obj, dict):
            return Cfg({k: Cfg.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [Cfg.wrap(v) for v in obj]
        return obj

    def to_dict(self) -> dict:
        def unwrap(o: Any) -> Any:
            if isinstance(o, dict):
                return {k: unwrap(v) for k, v in o.items()}
            if isinstance(o, list):
                return [unwrap(v) for v in o]
            return o

        return unwrap(self)


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# --------------------------------------------------------------------------------------
# Model registry — paper Table I, plus two tiny models used as GNN label sources.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    family: str  # gpt2 | pythia | llama
    paper_name: str
    params: float  # parameter count as listed in Table I
    layers: int
    hidden: int
    heads: int
    train_tokens: str
    k: int = 32  # TopK sparsity; the paper reports optimal values in [32, 128] by scale
    dtype: str = "float32"  # dtype used on CUDA; CPU always runs float32
    gated: bool = False
    fallback_ids: tuple[str, ...] = field(default_factory=tuple)
    in_paper: bool = True


MODEL_REGISTRY: dict[str, ModelSpec] = {
    s.key: s
    for s in [
        ModelSpec("gpt2-small", "openai-community/gpt2", "gpt2", "GPT-2 Small", 117e6, 12, 768, 12, "40B", k=32),
        ModelSpec("gpt2-medium", "openai-community/gpt2-medium", "gpt2", "GPT-2 Medium", 345e6, 24, 1024, 16, "40B", k=48),
        ModelSpec("gpt2-large", "openai-community/gpt2-large", "gpt2", "GPT-2 Large", 774e6, 36, 1280, 20, "40B", k=64),
        ModelSpec("pythia-1.4b", "EleutherAI/pythia-1.4b", "pythia", "Pythia-1.4B", 1.4e9, 24, 2048, 16, "300B", k=64, dtype="bfloat16"),
        ModelSpec("pythia-2.8b", "EleutherAI/pythia-2.8b", "pythia", "Pythia-2.8B", 2.8e9, 32, 2560, 32, "300B", k=80, dtype="bfloat16"),
        ModelSpec("pythia-6.9b", "EleutherAI/pythia-6.9b", "pythia", "Pythia-6.9B", 6.9e9, 32, 4096, 32, "300B", k=96, dtype="bfloat16"),
        ModelSpec("llama-7b", "meta-llama/Llama-2-7b-hf", "llama", "Llama-7B", 6.7e9, 32, 4096, 32, "1T", k=96,
                  dtype="bfloat16", gated=True, fallback_ids=("NousResearch/Llama-2-7b-hf",)),
        ModelSpec("llama-13b", "meta-llama/Llama-2-13b-hf", "llama", "Llama-13B", 13e9, 40, 5120, 40, "1T", k=112,
                  dtype="bfloat16", gated=True, fallback_ids=("NousResearch/Llama-2-13b-hf",)),
        ModelSpec("llama-70b", "meta-llama/Llama-2-70b-hf", "llama", "Llama-70B", 70e9, 80, 8192, 64, "1.4T", k=128,
                  dtype="bfloat16", gated=True, fallback_ids=("TheBloke/Llama-2-70B-fp16",)),
        # Not in the paper: small models used to produce exhaustive-search labels for the GNN
        # ("ground-truth circuit labels, obtained through exhaustive search on small models").
        ModelSpec("pythia-70m", "EleutherAI/pythia-70m", "pythia", "Pythia-70M", 70e6, 6, 512, 8, "300B", k=32, in_paper=False),
        ModelSpec("pythia-160m", "EleutherAI/pythia-160m", "pythia", "Pythia-160M", 160e6, 12, 768, 12, "300B", k=32, in_paper=False),
    ]
}

ALGORITHMIC_TASKS = ["mod_arith", "parity", "sorting"]
LANGUAGE_TASKS = ["winogrande", "hellaswag", "factual_qa"]
BENCHMARK_TASKS = ALGORITHMIC_TASKS + LANGUAGE_TASKS
AUX_TASKS = ["ioi", "greater_than", "induction"]


# --------------------------------------------------------------------------------------
# Defaults.  Values marked [paper] are stated in the paper; everything else is a choice
# this implementation had to make because the paper leaves it unspecified.
# --------------------------------------------------------------------------------------

DEFAULTS: dict[str, Any] = {
    "run_name": "default",
    "seed": 0,
    "output_dir": "runs",
    "device": "auto",
    "hub": {
        "enabled": True,
        "repo_id": "mudassiruddin/hagd",
        "upload_checkpoints": True,  # transcoders / adapters / GNN weights (needed to resume a lost session)
    },
    # Models the paper's tables are computed for, and small models that provide GNN labels.
    "models": ["gpt2-small"],
    "label_models": ["pythia-70m"],
    "tasks": BENCHMARK_TASKS,
    "aux_tasks": AUX_TASKS,  # extra circuit-discovery tasks for GNN meta-training (label models only)
    "data": {
        "n_train": 4000,  # per task: LoRA adaptation + task mix-in for transcoder training
        "n_discover": 192,  # attribution graph construction + circuit search
        "n_val": 96,  # causal validation (necessity pruning, size selection)
        "n_test": 256,  # held-out reporting split (never used during discovery)
        "batch_size": 32,
        "max_len": 128,
        "icl_shots": 0,  # few-shot prefix for algorithmic tasks when adapt.mode == none
        "mod_p": 113,  # [paper] (a + b) mod 113
        "parity_max_len": 20,  # [paper] binary strings up to length 20
        "sort_len": 5,  # [paper] 5 distinct integers
        "sort_max_int": 100,
    },
    "corpus": {
        "source": "redpajama",  # [paper] RedPajama; "wikitext" is an offline-friendly fallback
        "subsets": {"c4": 0.5, "wikipedia": 0.15, "arxiv": 0.1, "github": 0.1, "stackexchange": 0.15},
        "ctx_len": 128,
        "english_only": True,
    },
    # Pretrained LMs cannot do (a+b) mod 113, 20-bit parity or sorting zero-shot, which makes
    # Eq. 10 (an accuracy ratio) undefined.  The algorithmic tasks are therefore taught with a
    # LoRA adapter that is switched on only for those tasks.  mode: none disables this.
    "adapt": {
        "mode": "lora",
        "tasks": ALGORITHMIC_TASKS,
        "full_domain": True,  # modular arithmetic: train on all p*p pairs (False = held-out pairs must generalise)
        "rank": 32,
        "alpha": 32,
        "lr": 1e-3,
        "weight_decay": 0.05,
        "steps": 3000,
        "batch_size": 64,
        "eval_every": 250,
        "target_acc": 0.99,
        "max_minutes": 30,
    },
    "transcoder": {
        "expansion": 8,  # [paper] m / d = 8
        "k": None,  # TopK; None -> ModelSpec.k ([paper] 32..128 by scale)
        "layer_stride": 1,  # 1 = every layer [paper]; >1 thins layers so big models fit in memory
        "num_tokens": 2_000_000,  # [paper] 100M
        "epochs": 1,
        "tokens_per_step": 8192,
        "batch_seqs": 64,
        "lr": 4e-4,
        "weight_decay": 0.0,
        "warmup_steps": 200,
        "lambda_pred": 0.1,  # lambda_1 in Eq. 4
        "lambda_l1": 1e-3,  # lambda_2 in Eq. 4
        "init_std": 0.1,  # [paper] N(0, 0.01) read as variance 0.01
        "pred_rank": 512,  # rank of W^P (None = dense m x m; infeasible for large m)
        "detach_pred_target": False,
        "aux_k_coef": 0.03125,  # AuxK dead-feature loss from the TopK-SAE reference [7]; 0 disables
        "task_mix": 0.1,  # fraction of training batches drawn from task prompts instead of the corpus
        "ckpt_minutes": 20,
        "max_minutes": 240,
        "eval_tokens": 200_000,
    },
    "attribution": {
        "nodes_per_layer": 512,  # candidate features kept per layer (top |grad x act|); Pythia-70M mod-arith needs ~400
        "max_nodes": 8192,
        "cross_layer_expand": 8,  # extra candidates per layer proposed by the prediction heads
        "edge_topk": 16,  # strongest in/out edges kept per node
        "edge_min_frac": 0.01,  # drop edges below this fraction of the strongest edge
        "jac_chunk": 32,  # downstream features per batched backward pass
        "batched_grads": True,
    },
    "hierarchy": {
        "branching": 4,  # b in Theorem 1
        "embed_dim": 16,  # spectral embedding dimension (cap)
        "max_cluster_factor": 4,  # clusters larger than factor * b are split again
    },
    "gnn": {
        "hidden": 64,
        "layers": 3,  # message-passing iterations t
        "meta_iters": 400,
        "inner_steps": 5,
        "inner_lr": 5e-3,
        "meta_lr": 0.5,
        "use_edge_weight": False,  # False = Eq. 8 exactly; True adds log-weight bias to attention
        "label_pool": 96,  # exhaustive-search pool on label models
        "label_minutes": 30,  # time budget of one exhaustive label search
    },
    "search": {
        "theta": 0.9,  # [paper] sufficiency threshold of Eq. 10 / fidelity threshold of Eq. 2
        "criterion": "combined",  # ratio (Eq. 10 alone) | normalized (loss gap) | combined (min of both); see validation.Criterion
        "min_nodes": 2,
        "max_nodes": 6000,
        "growth": 1.5,
        "gamma": 0.3,  # weight of the parent supernode score in a child's priority
        "seeds": 1,
    },
    "validation": {
        "epsilon": 0.002,  # Eq. 9: features with Delta_i < epsilon are pruned
        "max_tests": 256,  # Eq. 9 is evaluated for the lowest-ranked features of the circuit only (cost: one run each)
        # where interventions act: "answer" = the positions whose prediction is scored (attention, which the
        # paper does not model, carries earlier positions in as context); "all" = every content token
        "positions": "answer",
        "rounds": 2,
        "ablation": "mean",  # mean | zero
        # What happens inside M_C to the part of a hidden state the transcoder cannot reconstruct:
        #   clean - kept as computed, like the always-on "error nodes" of sparse feature circuits [10] (default)
        #   mean  - replaced by its average (strict).  Measured on molab: with 15-30% unexplained variance per
        #           layer this destroys the behaviour even when *every* feature is kept, so theta is unreachable.
        #   zero  - dropped
        # Both the `clean` and `mean` preservation are always reported, with the empty-circuit baseline.
        "error_mode": "clean",
        "bootstrap": 1000,
    },
    "baselines": {
        "methods": ["attribution_patching", "sparse_probing", "acdc", "exhaustive"],
        "tasks": ["mod_arith"],  # Table II is reported on modular arithmetic
        "acdc": {"nodes_per_layer": 12, "taus": [0.01], "n_examples": 32, "max_minutes": 60,
                 "models": None},  # None = every model
        "exhaustive": {"pool": 96, "max_enum": 12, "drop_frac": 0.1, "max_minutes": 240,
                       "models": ["gpt2-small"]},  # [paper] exhaustive only on GPT-2 Small
        "sparse_probing": {"C": 0.1, "max_iter": 300},
    },
    "ablations": {  # Table V
        "model": None,  # None = first entry of models
        "tasks": ["mod_arith", "factual_qa"],
        "configs": ["full", "no_hierarchy", "no_gnn", "no_cross_layer", "no_causal_validation"],
    },
    "transfer": {  # Table IV / Fig. 3
        "pairs": [],  # [[source, target], ...]
        "tasks": ["mod_arith", "factual_qa", "winogrande"],
        "min_corr": 0.3,
    },
    "fourier": {"enabled": True, "n_examples": 2048},
    "complexity": {"sizes": [32, 64, 128, 256, 512, 1024, 2048], "trials": 3},
    "model_overrides": {},
}


def load_config(profile: str | Path | None = None, overrides: dict | None = None) -> Cfg:
    """Load DEFAULTS <- profile YAML <- overrides.

    ``profile`` is a profile name (``smoke``, ``laptop``, ``molab``, ``paper``) or a YAML path.
    """
    cfg = copy.deepcopy(DEFAULTS)
    if profile:
        path = Path(profile)
        if not path.suffix:
            path = CONFIG_DIR / f"{profile}.yaml"
        with open(path, encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
        cfg.setdefault("run_name", path.stem)
        if cfg["run_name"] == "default":
            cfg["run_name"] = path.stem
    cfg = deep_merge(cfg, overrides or {})
    for key in list(cfg["models"]) + list(cfg["label_models"]):
        if key not in MODEL_REGISTRY:
            raise KeyError(f"unknown model key {key!r}; known: {sorted(MODEL_REGISTRY)}")
    return Cfg.wrap(cfg)


def for_model(cfg: Cfg, model_key: str) -> Cfg:
    """Config with ``model_overrides[model_key]`` applied."""
    over = cfg.get("model_overrides", {}).get(model_key) or {}
    merged = deep_merge(cfg.to_dict(), over.to_dict() if isinstance(over, Cfg) else over)
    return Cfg.wrap(merged)


def model_spec(cfg: Cfg, model_key: str) -> ModelSpec:
    spec = MODEL_REGISTRY[model_key]
    hf_id = (cfg.get("model_overrides", {}).get(model_key) or {}).get("hf_id")
    if hf_id:
        spec = ModelSpec(**{**spec.__dict__, "hf_id": hf_id})
    return spec
