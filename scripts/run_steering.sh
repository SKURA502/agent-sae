#!/bin/bash
# Steering 实验脚本

set -e

# 配置
MODEL_PATH="meta-llama/Llama-3-8B-Instruct"
SAE_PATH=$(python - << 'PY'
from pathlib import Path

stage2_dir = Path("./outputs/sae_checkpoints/stage2")
matches = sorted(stage2_dir.glob("*-layer24-*-stage2.pt"))
print(matches[0] if matches else "")
PY
)
OUTPUT_DIR="./outputs/steering"
DEVICE="cuda"

if [[ -z "$SAE_PATH" ]]; then
    echo "No Stage 2 SAE checkpoint found for layer 24 under ./outputs/sae_checkpoints/stage2"
    exit 1
fi

echo "================================================"
echo "Steering Experiments"
echo "================================================"
echo "Using SAE checkpoint: $SAE_PATH"

# 从分析结果获取 top features
ANALYSIS_DIR="./outputs/analysis/layer_24"
TOP_FEATURES=$(python -c "
import json
with open('$ANALYSIS_DIR/analysis_summary.json') as f:
    data = json.load(f)
print(','.join(map(str, data['top_20_call_features'][:5])))
")

echo "Testing features: $TOP_FEATURES"

# 运行 steering 实验
python -c "
import torch
from analysis import SteeringExperiment

exp = SteeringExperiment(
    sae_path='$SAE_PATH',
    model_name='$MODEL_PATH',
    hook_layer=24,
    device='$DEVICE',
)

# 生成测试 prompts
prompts = [
    'What is 123 * 456?',
    'What is the capital of France?',
    'Calculate the square root of 144.',
    'Who wrote Romeo and Juliet?',
    'Search for the latest news about AI.',
]

features = [int(x) for x in '$TOP_FEATURES'.split(',')]
strengths = [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]

results = exp.run_steering(
    prompts=prompts,
    feature_indices=features[:3],
    strengths=strengths,
    output_dir='$OUTPUT_DIR',
)

print('Steering complete!')
print(f'Baseline call rate: {results[\"baseline\"][\"call_rate\"]:.2f}')
for e in results['experiments']:
    print(f'Feature {e[\"feature_idx\"]}, strength {e[\"strength\"]}: flip_rate={e[\"flip_rate\"][\"total_flip_rate\"]:.2f}')
"

echo ""
echo "Results saved to: $OUTPUT_DIR"
