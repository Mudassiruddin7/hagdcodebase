# HAGD — Hierarchical Attribution Graph Decomposition

Open implementation of **"Hierarchical Sparse Circuit Extraction from Billion-Parameter Language Models
through Scalable Attribution Graph Decomposition"** ([arXiv:2601.12879](https://arxiv.org/abs/2601.12879)).

The code implements every stage, task, metric, baseline and analysis described in the paper and
regenerates each of its tables and figures from measured results. Models and datasets come from the
Hugging Face Hub; run artifacts and results are stored in the dataset repo
[`mudassiruddin/hagd`](https://huggingface.co/datasets/mudassiruddin/hagd).

> **Reading the results.** Nothing in this repository hard-codes an outcome. The numbers the paper
> reports live in `src/hagd/paper_reference.py` and are only printed in columns labelled **Paper …**
> next to the values a run actually measures. Where a run differs from the paper, the report shows it.

## Paper → code

| Paper | What it is | Code |
|---|---|---|
| §III-A/B, Eq. 1, 3, 4, Algorithm 1 | Cross-layer transcoders: TopK encoder, decoder, sigmoid prediction head, joint loss, AdamW | `transcoder.py`, `train_transcoder.py` |
| §III-C, Eq. 5 | Attribution graph, `A_{i→j} = ∂f_j/∂f_i · f_i`, built layer pair by layer pair | `attribution.py` |
| §III-D, Def. 1, Eq. 6, Thm. 1 | Multi-resolution hierarchy by normalised-Laplacian spectral clustering | `hierarchy.py`, `complexity.py` |
| §III-E, Eq. 7, 8 | Attention GNN that scores circuit membership at every level; meta-trained on exhaustive-search labels | `gnn.py` |
| §III-D/E | Priority-queue traversal of the hierarchy; smallest circuit with φ ≥ θ (Eq. 2) | `search.py`, `extract.py` |
| §III-F, Eq. 9, 10 | Causal validation: necessity pruning, sufficiency, edge verification | `validation.py`, `replacement.py` |
| §IV-A, Table I | GPT-2 S/M/L, Pythia 1.4B/2.8B/6.9B, Llama 7B/13B/70B | `config.py` |
| §IV-B | Modular arithmetic, parity, sorting, WinoGrande, HellaSwag, factual recall | `data/tasks.py` |
| §IV-C | Sparsity, faithfulness and interpretability metrics | `metrics.py`, `validation.py` |
| §IV-D | Baselines: ACDC, attribution patching, sparse probing, exhaustive search | `baselines/` |
| §V-A, Table II, Fourier structure | Algorithmic-task circuits; Fourier analysis of features | `pipeline.py`, `fourier.py` |
| §V-B, Table III, Fig. 2 | Language-task circuits | `pipeline.py`, `report.py` |
| §V-C, Eq. 11, Table IV, Fig. 3 | Cross-architecture transfer coefficient | `transfer.py` |
| §V-D, Table V | Ablations: no hierarchy / GNN / cross-layer / causal validation | `extract.py` |
| §V-E, Table VI, Fig. 4 | Memory, time, throughput; complexity comparison | `pipeline.py`, `complexity.py` |
| §VI-C | Reconstruction "dark matter" per layer and task | `pipeline.py`, `report.py` |

## Running it

### On molab (RTX PRO 6000, 96 GB)

1. Open <https://molab.marimo.io>, choose *new notebook → from GitHub URL* and paste
   `https://github.com/Mudassiruddin7/hagdcodebase/blob/main/notebooks/hagd_molab.py`.
2. Attach the GPU from the notebook-specs button.
3. Add a secret `HF_TOKEN` with **write** access to `mudassiruddin/hagd` (and accepted access to
   `meta-llama/Llama-2-*`; ungated mirrors are used as a fallback).
4. Pick the `molab` profile and press **Start / resume**.

The notebook starts the pipeline as a background process and shows live progress and results.
molab sandboxes are not persistent and can end before the 12-hour limit, so every finished stage —
and a transcoder checkpoint every 15 minutes — is pushed to the Hub. Pressing **Start** in a new
session continues from there.

### From a terminal

```bash
pip install -e ".[quant,zstd]"          # quant: 4-bit loading for Llama-70B, zstd: common_crawl shards
export HF_TOKEN=hf_...

hagd run --profile smoke --no-hub        # ~10 min on a CPU; checks every stage, numbers are not meaningful
hagd run --profile laptop                # GPT-2 Small end to end
hagd run --profile molab                 # eight models, sized for one 96 GB GPU
hagd run --profile molab --models llama-70b --phases circuits transfer report
hagd report --profile molab              # rebuild tables and figures from stored results
pytest                                   # unit tests of every equation-level component
```

Any config value can be overridden: `--set transcoder.num_tokens=5e7 search.seeds=3`.

| Profile | Models | Transcoder tokens | Purpose |
|---|---|---|---|
| `smoke` | Pythia-70M | 0.15M | Pipeline check on a CPU |
| `laptop` | GPT-2 Small | 3M | One model on a consumer GPU |
| `molab` | GPT-2 S/M/L, Pythia 1.4B/2.8B/6.9B, Llama 7B/13B (70B opt-in) | 4M–20M | Single 96 GB GPU, resumable |
| `paper` | all nine | 100M, every layer, m/d = 8 | The paper's stated protocol; needs a multi-GPU machine and weeks |

## Where results go

```
mudassiruddin/hagd
├── README.md                      latest report (tables with measured and paper values)
└── runs/<run_name>/
    ├── config.json
    ├── <model>/adapter/           LoRA weights and accuracy of the adapted model
    ├── <model>/transcoder/        transcoder weights, held-out reconstruction quality
    ├── <model>/graphs/<task>/     attribution graph (vertices, edges, attributions)
    ├── <model>/circuits/<task>/<method>/circuit.json   nodes, edges, every metric, timings
    ├── <model>/labels/<task>/     exhaustive-search circuits used as GNN labels
    ├── gnn/                       GNN weights and held-out AUROC
    ├── analysis/                  transfer coefficients, complexity study
    └── report/                    Table II–VI as CSV, Fig. 2–4 as PNG, summary.json
```

## Choices the paper leaves open

The paper does not specify the points below. Each is a config option; the default is what the
implementation needed to be runnable and well defined.

- **How models perform the algorithmic tasks.** Pretrained GPT-2, Pythia and Llama score at or near
  chance on `(a+b) mod 113`, 20-bit parity and sorting, which leaves the accuracy ratio of Eq. 10
  undefined. The three tasks are taught with a LoRA adapter that is active only while those tasks run
  (`adapt.mode: lora`); language tasks always use the untouched weights. Transcoder training mixes in
  10% task prompts so the dictionaries cover the adapted activations. `adapt.mode: none` runs the raw
  checkpoints (optionally few-shot via `data.icl_shots`).
- **Where transcoders read.** `h_ℓ` is the residual stream after block ℓ, as in §II-A. Activations are
  standardised with fixed statistics so the sigmoid head of Eq. 3 can represent feature values.
- **Prediction head size.** `W^P` is factorised with rank 512. A dense `m × m` matrix per layer is
  4.3B parameters at Llama-70B's dictionary size.
- **The circuit-restricted model `M_C`** (protocol 2; each choice was measured on molab with
  Pythia-70M before it was made):
  - *Where to intervene.* Features outside the circuit are mean-ablated at the **answer positions**
    (`validation.positions: answer`), the tokens whose next-token prediction is scored. The paper
    models MLP features only and leaves attention intact (§VI-C), so earlier tokens stay as context
    that attention carries in. Ablating at every token removes the identity of the prompt's
    numbers: modular arithmetic then needed ~2,000 features per layer to reach θ.
  - *The unreconstructed residual.* It is kept (`validation.error_mode: clean`), like the always-on
    error nodes of sparse feature circuits [10]. Mean-ablating it too (`mean`) kept only 2–28% of
    accuracy **with every feature in the circuit**, so θ was unreachable for any circuit. Reports
    still show it as `Pres. (strict)`.
  - *The criterion.* Eq. 10 is an accuracy ratio, so an empty circuit scores ~1.0 on tasks where the
    model is near chance (Pythia-70M on factual QA: 0.91). The search therefore requires both
    `Pres.` ≥ θ and that the circuit recovers θ of the loss gap between the empty circuit and the
    full model (`search.criterion: combined`). `ratio` gives Eq. 10 alone.
- **Task scoring.** WinoGrande uses partial scoring, HellaSwag picks the highest-likelihood ending,
  factual recall is CounterFact's true object against its counterfactual object.
- **Exhaustive search.** All 2ⁿ subsets cannot be enumerated beyond ~20 vertices. The baseline
  grows a pool of top-attributed features until it meets the criterion, eliminates features by
  measured leave-one-out effect, and enumerates every subset of the last few. If no pool reaches
  θ the task is reported as unreachable rather than shrunk to a trivial circuit.
- **Necessity testing (Eq. 9)** costs one model run per feature, so it is applied to the 256
  lowest-ranked features of a circuit (`validation.max_tests`).
- **Finished stages are versioned.** Results produced under an older protocol are recomputed on the
  next run; task adapters and transcoders are reused.
- **ACDC on features.** ACDC prunes edges of the feature graph with exact per-edge interventions and
  the KL criterion of the original algorithm, over a pool of candidate features per layer.
- **GNN training tasks.** Labels come from exhaustive search on Pythia-70M/160M and GPT-2 Small over
  the six benchmarks plus IOI, greater-than and induction. The GNN used for GPT-2 Small is trained
  without GPT-2 Small's graphs, so no benchmark model is guided by its own labels. (The `smoke`
  profile has a single label model and cannot hold it out; circuit records flag this as
  `gnn_saw_target_model`.)
- **Concept purity and modularity** are defined informally in the paper; `metrics.py` states the
  operational definitions used.
- **Transfer across depths.** Models with different layer counts have no shared notion of adjacent
  layers; `transfer.py` documents the path-based edge correspondence used for Eq. 11.
- **Validation circularity** (§VI-C). Discovery, pruning and reporting use three disjoint splits, and
  preservation is reported with a bootstrap 95% confidence interval.

## Repository layout

```
src/hagd/            the package (see the table above)
src/hagd/configs/    smoke / laptop / molab / paper profiles
notebooks/           marimo notebook for molab
tests/               unit tests (tiny random model, no downloads)
```
