# Agent-Tool-Use-MI

Mechanistic interpretability study of tool-call decisions in LLMs via Sparse Autoencoders (SAE).

## Research Question

> Which internal representations in an LLM determine whether to invoke a tool?

## Hypotheses

| | Hypothesis | Evidence |
|---|---|---|
| **H1** | A sparse set of latent features in the residual stream acts as a gating signal for tool-call decisions at action boundaries | Feature discovery (mean diff + AUROC) + linear probe → Fig 1 |
| **H2** | The gate accumulates continuously as the model perceives a growing information gap, rather than firing at a single point | Per-step feature intensity trajectories across multi-turn episodes → Fig 3 |
| **H3** | Targeted activation steering of gate features reliably flips tool-call behavior without significantly degrading language quality | Steering / ablation flip rate + Δperplexity → Fig 2 |

## Method Overview

- **Model**: Qwen3.5-4B (primary), Qwen3.5-9B (scale comparison)
- **Representation**: TopK SAE on residual stream at ~3/4 and ~5/6 model depth (layers 24 & 26)
- **SAE training**: Two-stage — Stage 1 on OpenWebText2 (~50M tokens), Stage 2 fine-tuned on When2Call (~25K samples)
- **Feature discovery**: `E[f|CALL] − E[f|NO_CALL]` + per-feature AUROC, top-K gate features selected
- **Causal evidence**: Activation steering / ablation on top gate features, measuring decision flip rate

## Repository Structure

```
Agent-Tool-Use-MI/
├── sae/                     # SAE model + two-stage training
│   ├── sae_model.py         # TopK SAE implementation
│   ├── train_sae.py         # Stage 1 (OpenWebText2) + Stage 2 (When2Call)
│   ├── pretrain_data.py     # OpenWebText2 streaming loader
│   ├── interp_sae.py        # Activation maximisation / interpretability utils
│   └── calc_cos_sim.py      # Feature cosine similarity
│
├── run/                     # Data loading + activation extraction + rollout generation
│   ├── when2call_adapter.py # Parses When2Call pref/sft/mcq splits → (text, label)
│   ├── cache_activations.py # Extracts & caches residual-stream activations
│   ├── generate_rollouts.py # H2: multi-step sandbox episodes (tau2-bench seeds)
│   └── rollout_logger.py    # Serialises per-step activations to disk
│
├── controller/              # H2 sandbox agent loop
│   ├── sandbox_tools.py     # search / calculator / lookup with noise injection
│   └── agent_loop.py        # Minimal agent loop; collects per-step activations
│
├── analysis/                # Hypothesis-testing scripts
│   ├── feature_discovery.py # H1: mean_diff + AUROC → top-K features (Fig 1a)
│   ├── linear_probe.py      # H1: logistic regression 5-fold CV (Fig 1b)
│   ├── steering.py          # H3: SteeringHook, flip rate + Δperplexity (Fig 2)
│   └── trajectory_analysis.py # H2: feature intensity vs step (Fig 3)
│
├── scripts/
│   ├── run_pipeline.sh      # End-to-end pipeline (Steps 1–7)
│   └── swanlab_sync.py      # SwanLab experiment sync
│
├── data/
│   └── raw/
│       ├── When2Call/       # When2Call dataset (train/test splits)
│       └── tau2-bench-main/ # tau2-bench source (H2 prompt seeds)
│
├── outputs/                 # Generated outputs (gitignored)
│   ├── sae_checkpoints/     # Stage 1 + Stage 2 SAE .pt files
│   ├── activations/         # Cached residual-stream activations
│   ├── rollouts/            # H2 episode trajectories + per-step activations
│   └── analysis/            # Feature scores JSON + figures
│
├── utils.py                 # Shared utilities (hook helpers, dtype coercion, …)
└── requirements.txt
```

## Data

| Split | Size | Label | Use |
|---|---|---|---|
| When2Call Pref | 9K | CALL (3K) / NO_CALL (6K) | Stage 2 training + H1 feature discovery |
| When2Call SFT | 15K | NO_CALL | Stage 2 training |
| When2Call MCQ | 2,590 | tool_call / cannot_answer | H1 linear probe + H3 steering (request_for_info filtered out) |
| tau2-bench | — | — | H2: `reason_for_call` field as multi-turn prompt seeds |

## Setup

```bash
pip install -r requirements.txt
```

Optionally set environment variables before running the pipeline:

```bash
export MODEL_PATH="Qwen/Qwen3.5-4B"   # default
export DATA_BASE="./data/raw"                    # default
export OUTPUT_BASE="./outputs"                   # default
export DEVICE="cuda"
export DTYPE="bfloat16"
```

## Running the Pipeline

```bash
# Full pipeline (H1 + H3; H2 skipped by default)
bash scripts/run_pipeline.sh

# Include H2 rollout generation
RUN_H2=1 bash scripts/run_pipeline.sh

# Tune rollout parameters
RUN_H2=1 H2_DOMAIN=retail H2_N_EPISODES=200 bash scripts/run_pipeline.sh
```

### Step-by-step

| Step | Script | Description |
|---|---|---|
| 1 | `sae.train_sae stage1` | SAE pretraining on OpenWebText2 (50M tokens) |
| 2 | `sae.train_sae stage2` | SAE fine-tuning on When2Call pref+sft |
| 3 | `run.cache_activations` | Extract activations — When2Call MCQ (H1/H3) |
| 3b | `run.cache_activations` | Extract activations — When2Call Pref (feature discovery) |
| 4 | `analysis.feature_discovery` | H1: top-K gate features per layer |
| 5 | `analysis.linear_probe` | H1: logistic regression probe, AUC vs K |
| 6 | `analysis.steering` | H3: steering flip rate + Δperplexity |
| 7 | `run.generate_rollouts` + `analysis.trajectory_analysis` | H2: episode rollout + trajectory plot (opt-in) |

## Experiment Priority

**H1 → H3 → H2**

H1 and H3 only depend on static labelled data (When2Call). H2 requires the full agent loop to be validated before generating sandbox rollouts.

## Status

- [x] SAE two-stage training
- [x] H1 feature discovery + linear probe scripts
- [x] H3 steering / ablation scripts
- [x] H2 agent loop + rollout infrastructure
- [ ] End-to-end pipeline tested (pending SAE training run)
- [ ] README complete ← you are here
