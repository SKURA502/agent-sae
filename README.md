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
├── run/                # When2Call adapter, Stage 2 data iterator,
│                       # activation extraction, rollout generation
├── sae/                # two-stage SAE training, feature extraction
├── analysis/           # correlation, linear probe, steering, visualization
├── scripts/            # pipeline script, quick test
├── data/               # raw datasets (not committed)
│   └── raw/When2Call/data/
│       ├── train/      # when2call_train_pref.jsonl, when2call_train_sft.jsonl
│       └── test/       # when2call_test_mcq.jsonl
└── outputs/            # checkpoints, activations, analysis results
```

## Model

Primary: **Qwen3.5-4B** (`hidden_size=2560, num_layers=32`)

Hook layers: `L24` (`int(32×3/4)`) and `L26` (`int(32×5/6)`)

SAE: `dict_size = hidden_size × 8 = 20480`, `k = hidden_size // 32 = 80`

## Datasets

| File | Size | Label | Use |
|------|------|-------|-----|
| `when2call_train_pref.jsonl` | 9K | 3K CALL + 6K NO_CALL | SAE Stage 2 training + feature discovery |
| `when2call_train_sft.jsonl` | 15K | all NO_CALL | SAE Stage 2 training |
| `when2call_test_mcq.jsonl` | 3,652 | 1,295 tool_call / 1,295 cannot_answer / 1,062 request_for_info | H1/H3 evaluation only |

Training/test split: the entire `test_mcq` file (including `request_for_info`) is excluded from Stage 2 training and feature discovery.
H1/H3 evaluation uses the binary subset only (tool_call vs cannot_answer, 2,590 samples).

Feature discovery uses mean activation difference: `E[f|CALL] − E[f|NO_CALL]` over the pref split, no sampling required.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Stage 1 SAE training (OpenWebText2, 50M tokens)

```bash
python -m sae.train_sae stage1 \
  --model Qwen/Qwen3.5-4B \
  --layer 24 \
  --target-tokens 50000000 \
  --data-dir ./data/raw/pretrain \
  --output-dir ./outputs/sae_checkpoints \
  --device cuda
```

Repeat for `--layer 26`.

### Stage 2 SAE training (When2Call pref + sft, action-boundary activations)

```bash
python -m sae.train_sae stage2 \
  --model Qwen/Qwen3.5-4B \
  --layer 24 \
  --data-dir ./data/raw/When2Call/data/train \
  --stage1-checkpoint ./outputs/sae_checkpoints/stage1/<ckpt>.pt \
  --target-tokens 5000000 \
  --output-dir ./outputs/sae_checkpoints \
  --device cuda
```

Repeat for `--layer 26`. `--stage1-checkpoint` is optional; omit for random init.

### Extract test activations (for H1/H3 analysis)

```bash
python -m run.cache_activations extract \
  --model Qwen/Qwen3.5-4B \
  --dataset when2call \
  --data-path ./data/raw/When2Call/data/test \
  --split test_mcq \
  --layers 24 26 \
  --output-dir ./outputs/activations/when2call_mcq \
  --hook-position last \
  --device cuda
```

### H1: Feature discovery + Linear probe

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
