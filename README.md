# Agent SAE Tool-use

**Mechanistic Interpretability of Tool-use Decision in LLM Agents**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides a research framework for understanding **why LLMs decide to use tools** through the lens of Sparse Autoencoders (SAE). We study the mechanistic basis of tool-call gating in language model agents.

## 🎯 Research Questions

1. **Existence**: Are there specific SAE features that activate before tool-call decisions?
2. **Predictability**: Can a small set of features (~50) predict CALL vs NO_CALL with high accuracy?
3. **Causality**: Does steering/ablating these features flip the model's decision?

## 🏗️ Architecture

```
agent_sae_tooluse/
├── configs/                    # Configuration files
│   ├── model_config.yaml       # Model paths & hook layers
│   ├── sae_config.yaml         # SAE training hyperparameters
│   └── task_config.yaml        # Dataset & tool configurations
├── controller/                 # Agent controller
│   ├── tool_schema.py          # Pydantic tool definitions
│   ├── output_parser.py        # LLM output parsing
│   ├── agent_loop.py           # Main agent loop with activation caching
│   └── sandbox_tools/          # Sandbox tool implementations
├── tasks/                      # Data adapters
│   ├── when2call_adapter.py    # When2Call dataset
│   ├── bfcl_adapter.py         # BFCL dataset
│   └── synthetic_generator.py  # Synthetic data generation
├── run/                        # Rollout generation
│   ├── generate_rollouts.py    # Batch rollout generation
│   └── cache_activations.py    # Activation caching for SAE
├── sae/                        # SAE training
│   ├── sae_model.py            # TopK SAE implementation
│   ├── train_sae.py            # SAE trainer with WandB
│   └── feature_extraction.py   # Feature analysis & AUROC
├── analysis/                   # Mechanistic analysis
│   ├── correlation_analysis.py # Feature-decision correlation
│   ├── linear_probe.py         # Predictability verification
│   ├── steering.py             # Causal intervention
│   └── visualization.py        # Paper figure generation
├── scripts/                    # Convenience scripts
│   ├── run_pipeline.sh         # Full experiment pipeline
│   ├── run_steering.sh         # Steering experiments
│   └── quick_test.py           # Module verification
├── main.py                     # CLI entry point
└── requirements.txt            # Dependencies
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/agent-sae-tooluse.git
cd agent-sae-tooluse

# Create virtual environment
conda create -n agent-sae python=3.10 -y
conda activate agent-sae

# Install dependencies
pip install -r requirements.txt
```

### Run Full Pipeline

```bash
# Run the complete experiment pipeline
bash scripts/run_pipeline.sh
```

### Step-by-Step Execution

```bash
# 1. Generate rollouts with activation caching
python main.py generate-rollouts \
    --model meta-llama/Llama-3-8B-Instruct \
    --dataset synthetic \
    --num-samples 10000 \
    --output-dir ./outputs/rollouts

# 2. Cache activations at specific layers
python main.py cache-activations \
    --model meta-llama/Llama-3-8B-Instruct \
    --rollout-dir ./outputs/rollouts \
    --layers "20,24" \
    --output-dir ./outputs/activations

# 3. Train SAE
python main.py train-sae \
    --data-path ./outputs/activations/layer_24_activations.pt \
    --dict-size 32768 \
    --k 128 \
    --epochs 10 \
    --output-dir ./outputs/sae_models/layer_24 \
    --wandb

# 4. Run correlation analysis
python main.py analyze \
    --analysis-type correlation \
    --sae-path ./outputs/sae_models/layer_24/best_model.pt \
    --data-path ./outputs/activations/layer_24_activations.pt \
    --layer 24 \
    --output-dir ./outputs/analysis

# 5. Run linear probe
python main.py analyze \
    --analysis-type probe \
    --sae-path ./outputs/sae_models/layer_24/best_model.pt \
    --data-path ./outputs/activations/layer_24_activations.pt \
    --layer 24 \
    --output-dir ./outputs/analysis

# 6. Generate visualizations
python main.py analyze \
    --analysis-type visualize \
    --sae-path ./outputs/sae_models/layer_24/best_model.pt \
    --data-path ./outputs/activations/layer_24_activations.pt \
    --layer 24 \
    --output-dir ./outputs/analysis
```

### Quick Test

```bash
# Verify all modules are working
python scripts/quick_test.py
```

## 📊 Supported Models

| Model | Hidden Size | Hook Layers (3/4, 5/6) |
|-------|-------------|------------------------|
| Llama-3-8B-Instruct | 4096 | 24, 27 |
| Qwen-3-8B | 4096 | 24, 27 |
| Qwen-3-14B | 5120 | 30, 34 |
| Gemma-3-4B | 2560 | 21, 24 |
| Gemma-3-12B | 3584 | 24, 27 |

## 📈 Key Concepts

### Action Boundary Window

We define the **Action Boundary Window** as the critical region around tool-call decisions:

- **W_pre**: 20 tokens before the tool_call output token
- **W_post**: 10 tokens after (for learning from feedback)

### SAE Configuration

- **Architecture**: TopK SAE with ReLU activation
- **Dictionary Size**: `hidden_size × 8` (expansion factor)
- **Sparsity**: `k = hidden_size / 32` active features
- **Hook Points**: Residual stream at 3/4 and 5/6 layer depth

### Analysis Metrics

1. **AUROC**: Per-feature discrimination between CALL and NO_CALL
2. **Mean Difference**: E[f|CALL] - E[f|NO_CALL]
3. **Flip Rate**: Proportion of decisions changed by steering

## 📁 Datasets

| Dataset | Purpose | Features |
|---------|---------|----------|
| **When2Call** | CALL vs NO_CALL gating | Minimal, focused on decision boundary |
| **BFCL** | Function calling benchmark | Diverse API definitions |
| **API-Bank** | End-to-end agent evaluation | Multi-step reasoning |
| **Synthetic** | Controlled experiments | Customizable difficulty |

## 🔬 Experiment Workflow

```
┌─────────────────┐
│  Generate Data  │  ← When2Call / BFCL / Synthetic
└────────┬────────┘
         ▼
┌─────────────────┐
│  Run Rollouts   │  ← Agent loop with activation caching
└────────┬────────┘
         ▼
┌─────────────────┐
│  Train SAE      │  ← TopK SAE on residual stream
└────────┬────────┘
         ▼
┌─────────────────┐
│  Analyze        │  ← Correlation, probe, steering
└────────┬────────┘
         ▼
┌─────────────────┐
│  Visualize      │  ← Paper figures
└─────────────────┘
```

## 📝 Configuration

### Model Configuration (`configs/model_config.yaml`)

```yaml
models:
  llama3_8b:
    name: "meta-llama/Llama-3-8B-Instruct"
    hidden_size: 4096
    num_layers: 32
    hook_layers: [24, 27]  # 3/4 and 5/6 depth
```

### SAE Configuration (`configs/sae_config.yaml`)

```yaml
sae:
  expansion_factor: 8
  k_divisor: 32
  learning_rate: 1.0e-4
  batch_size: 4096
  num_epochs: 10
```

## 📊 Expected Results

Based on our hypotheses:

| Metric | Expected | Significance |
|--------|----------|--------------|
| Top-50 features AUROC | > 0.85 | High discriminability |
| Linear probe accuracy | > 90% | Predictable from few features |
| Steering flip rate | > 30% | Causal influence confirmed |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Anthropic SAE Research](https://www.anthropic.com/) for pioneering SAE interpretability
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) for activation hooking inspiration
- [When2Call](https://arxiv.org/abs/2410.03161) for the benchmark dataset

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{agent_sae_tooluse,
  title = {Agent SAE Tool-use: Mechanistic Interpretability of Tool-use Decision in LLM Agents},
  year = {2026},
  url = {https://github.com/YOUR_USERNAME/agent-sae-tooluse}
}
```

## 📧 Contact

For questions or collaborations, please open an issue or contact the maintainers.
