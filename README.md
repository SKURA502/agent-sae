# Agent SAE Tool-use

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

Mechanistic interpretability framework for understanding tool-use decisions in LLM agents with Sparse Autoencoders (SAE).

## Overview

This repository provides a research workflow to:

- generate agent rollouts for tool-use tasks,
- train SAEs with a two-stage pipeline,
- analyze feature-level CALL/NO_CALL signals,
- run steering experiments for causal checks.

## Project structure

```text
Agent-Tool-Use-MI/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── main.py
├── controller/
│   ├── __init__.py
│   ├── agent_loop.py
│   ├── output_parser.py
│   ├── tool_schema.py
│   └── sandbox_tools/
│       ├── __init__.py
│       ├── calculator.py
│       ├── lookup.py
│       ├── search.py
│       └── tool_utils.py
├── tasks/
│   ├── __init__.py
│   ├── base_adapter.py
│   ├── bfcl_adapter.py
│   ├── synthetic_generator.py
│   └── when2call_adapter.py
├── run/
│   ├── __init__.py
│   ├── cache_activations.py
│   ├── generate_rollouts.py
│   └── rollout_logger.py
├── sae/
│   ├── __init__.py
│   ├── feature_extraction.py
│   ├── pretrain_data.py
│   ├── sae_model.py
│   └── train_sae.py
├── analysis/
│   ├── __init__.py
│   ├── correlation_analysis.py
│   ├── linear_probe.py
│   ├── steering.py
│   └── visualization.py
├── scripts/
│   ├── quick_test.py
│   ├── model_config.yaml
│   ├── run_pipeline.sh
│   └── run_steering.sh
├── data/
│   ├── raw/
│   ├── processed/
│   └── rollouts/
└── outputs/
    ├── analysis_results/
    ├── figures/
    └── sae_checkpoints/
```

## Installation

```bash
pip install -r requirements.txt
```

## CLI entrypoints

You can use both entry styles:

1. Unified entry: `python main.py ...`
2. Module entry: `python -m ...`

`main.py` forwards training-related commands to module CLIs to keep logic centralized.

## Configuration

### Model presets

- File: `configs/model_config.yaml`
- Purpose: stable model metadata (name, hidden size, default hook layers, etc.)

### Experiment settings

Use command line args for dataset, layers, training hyperparameters, output paths, and runtime options.

## Quick start

### A. Generate rollouts

```bash
python main.py generate-rollouts \
  --model-key llama3-8b \
  --dataset synthetic \
  --num-samples 1000 \
  --layers 24 27 \
  --output-dir ./outputs/rollouts \
  --device cuda
```

Override model preset directly:

```bash
python main.py generate-rollouts \
  --model meta-llama/Llama-3-8B-Instruct \
  --dataset when2call \
  --data-path ./data/raw/when2call \
  --split test \
  --num-samples 1000
```

### B. Stage 1 SAE training

```bash
python -m sae.train_sae stage1 \
  --model meta-llama/Llama-3-8B-Instruct \
  --layers 24 27 \
  --target-tokens 50000000 \
  --data-dir ./data/raw/pretrain \
  --output-dir ./outputs/sae_checkpoints \
  --device cuda
```

Equivalent via unified entry:

```bash
python main.py train-sae stage1 --model meta-llama/Llama-3-8B-Instruct --layers 24 27
```

### C. Stage 2 streaming training

```bash
python -m run.cache_activations train \
  --model meta-llama/Llama-3-8B-Instruct \
  --dataset synthetic \
  --num-samples 2000 \
  --layers 24 27 \
  --stage1-dir ./outputs/sae_checkpoints/stage1 \
  --output-dir ./outputs/sae_checkpoints \
  --target-tokens 50000000 \
  --batch-size 4096 \
  --learning-rate 5e-5 \
  --device cuda
```

Equivalent via unified entry:

```bash
python main.py cache-activations train --model meta-llama/Llama-3-8B-Instruct --dataset synthetic --layers 24 27 --stage1-dir ./outputs/sae_checkpoints/stage1
```

## Analysis

`main.py analyze` requires serialized activations via `--data-path`:

```bash
python main.py analyze \
  --analysis-type correlation \
  --sae-path ./outputs/sae_checkpoints/stage2/<your-stage2-ckpt>.pt \
  --data-path ./outputs/activations/layer_24_activations.pt \
  --layer 24 \
  --output-dir ./outputs/analysis/layer_24
```

The same interface applies to `probe` and `visualize`.

## Scripts

- `scripts/run_pipeline.sh`: two-stage pipeline script aligned with current checkpoint naming.
- `scripts/run_steering.sh`: resolves stage2 checkpoints from `./outputs/sae_checkpoints/stage2`.
- `scripts/quick_test.py`: quick API-level sanity checks.

## Notes

Checkpoint naming conventions:

- Stage1: `*-layer{L}-*-stage1.pt`
- Stage2: `*-layer{L}-*-stage2.pt`

Both `sae.train_sae` and `run.cache_activations` use the same naming pattern.

## Contributing

Issues and pull requests are welcome.

## License

MIT License. See [LICENSE](LICENSE).
