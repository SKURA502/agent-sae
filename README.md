# Agent SAE Tool-use MI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

Mechanistic interpretability of tool-use decisions in LLM agents using Sparse Autoencoders (SAE).

## Research Questions

> Which internal representations in an LLM gate tool-call decisions?

Three hypotheses — H1 (gating features), H3 (causal controllability), H2 (evidence accumulation).
See [docs/goal.md](docs/goal.md) for the full research design.

## Project Structure

```
Agent-Tool-Use-MI/
├── configs/            # model_config.yaml (stable model metadata)
├── controller/         # agent loop, tool schema, sandbox tools
├── tasks/              # dataset adapters: When2Call, BFCL, synthetic
├── run/                # activation extraction, SAE Stage 2 training
├── sae/                # two-stage SAE training, feature extraction
├── analysis/           # correlation, linear probe, steering, visualization
├── scripts/            # pipeline scripts, quick test
├── data/               # raw datasets (not committed)
└── outputs/            # checkpoints, activations, analysis results
```

## Model

Primary: **Qwen3.5-4B-Instruct** (`hidden_size=2560, num_layers=32`)

Hook layers: `L24` (`int(32×3/4)`) and `L26` (`int(32×5/6)`)

SAE: `dict_size = hidden_size × 8 = 20480`, `k = hidden_size // 32 = 80`

## Datasets

| File | Size | Label | Use |
|------|------|-------|-----|
| `when2call_train_pref.jsonl` | 9K | 3K CALL + 6K NO_CALL | SAE Stage 2 training |
| `when2call_test_mcq.jsonl` | 3,652 | tool_call / cannot_answer / request_for_info | H1/H3 evaluation |
| `BFCL_v4_irrelevance.json` | 240 | NO_CALL | H1 generalization |
| `BFCL_v4_live_irrelevance.json` | 884 | NO_CALL | H1 generalization |
| `BFCL_v4_simple_python.json` | 400 | CALL | H1 generalization |
| `BFCL_v4_live_simple.json` | 258 | CALL | H1 generalization |

When2Call label parsing: `chosen_response.content` containing `<TOOLCALL>` → CALL, otherwise NO_CALL.
MCQ evaluation uses binary subset only (tool_call vs cannot_answer, 2,590 samples; request_for_info excluded).

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Smoke test (no GPU required)

```bash
python scripts/quick_test.py
```

### Stage 1 SAE training (OpenWebText2, 50M tokens)

```bash
python -m sae.train_sae stage1 \
  --model Qwen/Qwen3.5-4B-Instruct \
  --layer 24 \
  --target-tokens 50000000 \
  --data-dir ./data/raw/pretrain \
  --output-dir ./outputs/sae_checkpoints \
  --device cuda
```

Repeat for `--layer 26`.

### Stage 2 SAE training (When2Call Pref, balanced 3K+3K)

```bash
python -m run.cache_activations stage2 \
  --model Qwen/Qwen3.5-4B-Instruct \
  --dataset when2call \
  --data-path ./data/raw/when2call \
  --split train_pref \
  --layers 24 26 \
  --stage1-dir ./outputs/sae_checkpoints/stage1 \
  --output-dir ./outputs/sae_checkpoints \
  --learning-rate 5e-4 \
  --batch-size 4096 \
  --num-epochs 3 \
  --balance \
  --device cuda
```

### Extract test activations (for H1/H3 analysis)

```bash
# When2Call MCQ binary subset
python -m run.cache_activations extract \
  --model Qwen/Qwen3.5-4B-Instruct \
  --dataset when2call \
  --data-path ./data/raw/when2call \
  --split test_mcq \
  --layers 24 26 \
  --output-dir ./outputs/activations/when2call_mcq \
  --device cuda

# BFCL generalization
python -m run.cache_activations extract \
  --model Qwen/Qwen3.5-4B-Instruct \
  --dataset bfcl \
  --data-path ./data/raw/bfcl \
  --layers 24 26 \
  --output-dir ./outputs/activations/bfcl_gen \
  --device cuda
```

### H1: Correlation analysis + Linear probe

```bash
python -m analysis.correlation_analysis \
  --sae-path ./outputs/sae_checkpoints/stage2/<ckpt>.pt \
  --data-path ./outputs/activations/when2call_mcq/layer_24_activations.pt \
  --layer 24 \
  --output-dir ./outputs/analysis/layer_24/when2call_mcq \
  --device cuda

python -m analysis.linear_probe \
  --sae-path ./outputs/sae_checkpoints/stage2/<ckpt>.pt \
  --data-path ./outputs/activations/when2call_mcq/layer_24_activations.pt \
  --layer 24 \
  --output-dir ./outputs/analysis/layer_24/when2call_mcq \
  --device cuda
```

### H3: Steering experiments

```bash
bash scripts/run_steering.sh
```

### Full pipeline (Stage 1 → Stage 2 → H1 → H3)

```bash
bash scripts/run_pipeline.sh
```

## Checkpoint Naming

- Stage 1: `{model}-L{layer}-d{dict_size}-{tokens}M-stage1.pt`
- Stage 2: `{model}-L{layer}-d{dict_size}-{tokens}M-stage2.pt`

## Key Metrics

| Metric | Hypothesis | Target |
|--------|-----------|--------|
| Top-feature AUROC | H1 | > 0.75 |
| Linear probe AUC (K=50) | H1 | > 0.80 |
| Decision flip rate | H3 | > 20% |
| Δperplexity | H3 | < 20% |

## License

MIT License. See [LICENSE](LICENSE).
