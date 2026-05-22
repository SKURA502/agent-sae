# To Call or Not to Call: Diagnosing Intrinsic Over-Calling Bias in LLM Agents

Official implementation of the paper **"To Call or Not to Call: Diagnosing Intrinsic Over-Calling Bias in LLM Agents"**.

This repository provides a mechanistic interpretability framework for understanding and correcting tool-calling decision bias in LLM agents, using Sparse Autoencoders (SAE) to identify and steer the internal representations that drive over-calling behavior.

---

## Overview

LLM agents frequently exhibit an *intrinsic over-calling bias* — they invoke tools even when a direct answer or a request for clarification would be more appropriate. This project:

1. **(H1) Feature discovery**: discovers SAE features that encode the "tool-call" concept at the action boundary and verifies they form a linearly separable subspace.
2. **(H2) Bias quantification**: fits a logistic regression on TC vs. RFI feature activations to measure the intrinsic TC bias `β₀` and its decision-boundary shift `δ = -β₀/β`.
3. **(H3) Causal steering**: demonstrates that suppressing tool-call features or promoting request-for-info features causally shifts model decisions.
4. **Correction**: applies **AMCS** (Adaptive Margin-Calibrated Steering) at inference time to neutralize the bias using `δ` as the sole data-driven parameter.

---

## Repository Structure

```
Agent-Tool-Use-MI/
├── sae/                        # SAE architecture and training
│   ├── sae_model.py            # TopK SAE (encoder + decoder)
│   ├── train_sae.py            # Two-stage SAE training pipeline
│   ├── pretrain_data.py        # Streaming activation extractor
│   └── interp_sae.py           # Feature interpretation utilities
│
├── run/                        # Data preparation and evaluation runners
│   ├── cache_activations.py    # Extract and label activations from the test set
│   ├── eval_amcs_accuracy.py   # AMCS bias correction evaluation
│   ├── eval_suppress_accuracy.py  # Steering (suppress / promote) evaluation
│   ├── eval_baseline_rfi.py    # RFI baseline evaluation
│   └── sweep_amcs_topn.py      # AMCS hyper-parameter sweep
│
├── analysis/                   # Analysis and visualization scripts
│   ├── feature_discovery.py    # H1: compute mean_diff and AUROC per SAE feature
│   ├── linear_probe.py         # H1: linear separability verification
│   ├── steering.py             # H3: causal steering experiments
│   ├── case_study_visualize.py # Per-sample case study visualizations
│   └── plot_linear_probe_combined.py
│
├── utils_validation/           # Validation and plotting utilities
│   ├── bias/
│   │   ├── tc_bias_logistic.py          # H2: logistic regression to quantify TC bias β₀
│   │   ├── plot_pred_tc_combined.py     # H2: predicted-TC scatter plots (multi-model)
│   │   └── plot_pred_tc_nc_combined.py  # H2: TC vs. RFI margin scatter plots
│   └── feature_discovery/
│       ├── plot_combined_bar.py         # H1: combined feature bar chart
│       ├── plot_multi_model_bar.py      # H1: multi-model feature bar chart
│       └── plot_umap_tc_nc_combined.py  # H1: UMAP visualization across models
│
├── utils/
│   ├── templates.py            # Chat templates for Qwen / Gemma / Ministral
│   ├── when2call_adapter.py    # When2Call dataset loader
│   └── __init__.py             # Shared CLI argument helpers
│
├── scripts/                    # End-to-end shell scripts
│   ├── train_sae.sh
│   ├── cache_activations.sh
│   ├── feature_discovery_when2call.sh
│   ├── linear_probe_combined.sh
│   ├── eval_suppress_accuracy.sh
│   ├── eval_amcs.sh
│   ├── eval_baseline_rfi.sh
│   └── sweep_amcs_topn.sh
│
├── checkpoint/                 # Pre-trained SAE checkpoints (per model)
│   └── {MODEL_NAME}/
│       ├── stage1/             # General-corpus SAE
│       └── stage2/             # Tool-call-task SAE
│
└── outputs/                    # Experiment outputs (auto-created)
    └── {MODEL_NAME}/
        ├── activations/
        ├── analysis/
        ├── amcs/
        └── steer_accuracy/
```

---

## Supported Models

Pre-trained SAE checkpoints are provided for all eight models:

| Model | SAE Layer | Dict Size |
|---|---|---|
| `gemma-3-1b-it` | 17 | 9216 |
| `gemma-3-4b-it` | 29 | 20480 |
| `gemma-4-E2B-it` | 30 | 12288 |
| `gemma-4-E4B-it` | 30 | 20480 |
| `Ministral-3-3B-Instruct-2512` | 21 | 24576 |
| `Ministral-3-8B-Instruct-2512` | 31 | 32768 |
| `Qwen3.5-4B` | 25 | 20480 |
| `Qwen3.5-9B` | 25 | 32768 |

You can find the checkpoints in [toolcalling-sae](https://huggingface.co/SKwra/toolcalling-sae)

---

## Installation

```bash
git clone https://github.com/YOUR_ORG/Agent-Tool-Use-MI.git
cd Agent-Tool-Use-MI
pip install -r requirements.txt
```

**Key dependencies:**

```
torch>=2.1.0
transformers>=4.40.0
accelerate>=0.27.0
datasets>=2.18.0
einops>=0.7.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
```

---

## Dataset

This project uses the **When2Call** benchmark. Place the data under `data/raw/When2Call/data/` with the following structure:

```
data/raw/When2Call/data/
├── train/
│   ├── when2call_train_pref.jsonl   # 9K samples (3K CALL + 6K NO_CALL)
│   └── when2call_train_sft.jsonl    # 15K NO_CALL samples
└── test/
    └── when2call_test_mcq.jsonl     # 3652 MCQ samples (evaluation only)
```

Each test sample has four possible labels: `direct_answer`, `tool_call`, `request_for_info`, `cannot_answer`.

---

## Usage

### Step 1 — Train SAE (two-stage)

```bash
export MODEL_PATH=/path/to/Qwen3.5-4B
export PRETRAIN_DIR=/path/to/openwebtext2
export WHEN2CALL_TRAIN=/path/to/when2call/train
bash scripts/train_sae.sh
```

- **Stage 1**: trains on 50M tokens of general corpus (e.g., OpenWebText2).
- **Stage 2**: fine-tunes on ~5M tokens of When2Call (`pref` + `sft` splits).

Checkpoints are saved to `checkpoint/{MODEL_NAME}/stage{1,2}/`.

### Step 2 — Extract activations

```bash
export MODEL_PATH=/path/to/Qwen3.5-4B
bash scripts/cache_activations.sh
```

Extracts hidden-state activations at the action boundary for all MCQ test samples. Output: `outputs/{MODEL_NAME}/activations/when2call_mcq/layer_{L}_activations.pt`.

### Step 3 — Feature discovery (H1)

```bash
# Discover tool-call features
CONCEPT=tool_call bash scripts/feature_discovery_when2call.sh

# Discover request-for-info features
CONCEPT=request_for_info bash scripts/feature_discovery_when2call.sh
```

Computes `mean_diff` and AUROC per SAE feature. Outputs top-K feature lists and visualizations (AUROC distribution, mean-diff bar charts, UMAP) to `outputs/{MODEL_NAME}/analysis/feature_discovery/{concept}/`.

### Step 4 — Linear separability (H1)

```bash
bash scripts/linear_probe_combined.sh
```

Trains logistic regression probes on top-K features and reports cross-validated AUC. Output: `outputs/{MODEL_NAME}/analysis/linear_probe/`.

### Step 5 — Bias quantification (H2)

```bash
python -m utils_validation.bias.tc_bias_logistic \
  --layer 25 \
  --sae-path checkpoint/Qwen3.5-4B/stage2/Qwen3.5-4B-L25-d20480-5M-stage2.pt \
  --activations-dir outputs/Qwen3.5-4B/activations/when2call_mcq \
  --feature-discovery-dir outputs/Qwen3.5-4B/analysis/feature_discovery \
  --output-dir outputs/Qwen3.5-4B/analysis/rfi_confusion
```

Fits a logistic regression `P(pred=TC) = sigmoid(β · margin + β₀)` where `margin = tc_act − rfi_act` is the difference in TC vs. RFI feature activations. The intercept `β₀ > 0` quantifies the intrinsic TC bias, and `δ = -β₀/β` is the decision-boundary shift used later by AMCS. Also produces scatter visualizations showing the predicted-TC rate as a function of feature activation margin. Output: `outputs/{MODEL_NAME}/analysis/rfi_confusion/`.

### Step 6 — Causal steering (H3)

```bash
# suppress tool-call features only
bash scripts/eval_suppress_accuracy.sh Qwen3.5-4B suppress_tc cuda:0

# promote request-for-info features only
bash scripts/eval_suppress_accuracy.sh Qwen3.5-4B promote_rfi cuda:0

# both simultaneously
bash scripts/eval_suppress_accuracy.sh Qwen3.5-4B both cuda:0
```

Applies activation steering at inference time and measures accuracy on the MCQ test set. Output: `outputs/{MODEL_NAME}/steer_accuracy/`.

### Step 7 — AMCS bias correction

```bash
bash scripts/sweep_amcs_topn.sh
```

Runs Adaptive Margin-Calibrated Steering (AMCS), which uses a single data-driven parameter `δ = -β₀/β` to construct a fixed steering vector that neutralizes the intrinsic TC bias. No validation-set search is required.

---

## Key Concepts

### Action Boundary

The hidden state extracted at the position where the model has consumed the full context (system prompt + tools + user message) and is about to produce the first assistant token. This is the point at which tool-call vs. no-call decisions are encoded.

### Two-Stage SAE Training

| Stage | Corpus | Tokens | Purpose |
|---|---|---|---|
| Stage 1 | OpenWebText2 | 50M | Learn general language features |
| Stage 2 | When2Call (pref + sft) | ~5M | Specialize for tool-call decision features |

### Intrinsic TC Bias (H2)

To quantify the model's built-in preference for tool calls, a logistic regression is fitted on samples where the ground truth is either `tool_call` or `request_for_info`:

```
P(pred=TC) = sigmoid(β · margin + β₀)
margin = tc_act − rfi_act
```

`tc_act` and `rfi_act` are the summed top-N SAE feature activations for the TC and RFI concepts respectively. The intercept `β₀ > 0` indicates that even when both concepts are equally activated, the model still prefers to call a tool. The decision-boundary shift `δ = -β₀/β` measures how much the RFI signal must exceed the TC signal before the model switches away from tool-calling.

`δ` is the sole data-driven parameter fed into AMCS.

### AMCS (Adaptive Margin-Calibrated Steering)

A closed-form, inference-time bias correction method. The steering vector is:

```
v = Σ wᵢ · n · α · δ · dᵢ
```

where `δ` is derived from the dataset-measured intrinsic bias `β₀`, `α` allocates budget between TC-suppression and RFI-promotion, and `dᵢ` are the top-N SAE decoder directions. No per-sample gating or semantic hyperparameter tuning is needed.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
