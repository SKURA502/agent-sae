#!/bin/bash
# ================================================================
# H3 Steering 实验脚本
# 对 top 门控 features 进行激活干预，测量决策翻转率
# 模型：Qwen3.5-4B-Instruct，Hook 层：L24 / L26
# ================================================================

set -e

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B-Instruct}"
OUTPUT_BASE="${OUTPUT_BASE:-./outputs}"
WHEN2CALL_DIR="${WHEN2CALL_DIR:-./data/raw/when2call}"
DEVICE="${DEVICE:-cuda}"

LAYERS="${LAYERS:-24 26}"

echo "================================================================"
echo "H3 Steering Experiments"
echo "  Model : $MODEL_PATH"
echo "  Layers: $LAYERS"
echo "================================================================"

for LAYER in $LAYERS; do
  # 查找 Stage 2 checkpoint
  SAE_PATH=$(python -c "
from pathlib import Path
base = Path('$OUTPUT_BASE/sae_checkpoints/stage2')
matches = sorted(base.glob('*-L${LAYER}-*-stage2.pt'))
if not matches:
    matches = sorted(base.glob('*-layer${LAYER}-*-stage2.pt'))
print(matches[0] if matches else '')
")

  if [[ -z "$SAE_PATH" ]]; then
    echo "  ✗ No Stage 2 checkpoint for layer $LAYER under $OUTPUT_BASE/sae_checkpoints/stage2"
    continue
  fi

  echo ""
  echo "Layer $LAYER  |  SAE: $SAE_PATH"

  # 从相关性分析结果取 top-5 CALL 门控特征
  ANALYSIS_DIR="$OUTPUT_BASE/analysis/layer_${LAYER}/when2call_mcq"
  TOP_FEATURES=$(python -c "
import json
from pathlib import Path
p = Path('$ANALYSIS_DIR/analysis_summary.json')
if p.exists():
    d = json.loads(p.read_text())
    print(','.join(map(str, d.get('top_20_call_features', [])[:5])))
else:
    print('')
")

  if [[ -z "$TOP_FEATURES" ]]; then
    echo "  ✗ No analysis summary at $ANALYSIS_DIR — run run_pipeline.sh first"
    continue
  fi

  echo "  Top features: $TOP_FEATURES"

  python -c "
import sys
sys.path.insert(0, '.')
from analysis import SteeringExperiment
from tasks import When2CallAdapter
from tasks.base_adapter import DecisionLabel

# When2Call MCQ 二类子集（排除 request_for_info）
adapter = When2CallAdapter('$WHEN2CALL_DIR', split='test_mcq')
adapter.load()
samples = [s for s in adapter if s.label in (DecisionLabel.CALL, DecisionLabel.NO_CALL)]
# 均匀采样 200 条（CALL 100 + NO_CALL 100）
call_s  = [s for s in samples if s.label == DecisionLabel.CALL][:100]
nocall_s = [s for s in samples if s.label == DecisionLabel.NO_CALL][:100]
prompts = [s.instruction for s in call_s + nocall_s]
print(f'Steering prompts: {len(prompts)} ({len(call_s)} CALL + {len(nocall_s)} NO_CALL)')

exp = SteeringExperiment(
    sae_path='$SAE_PATH',
    model_name='$MODEL_PATH',
    hook_layer=$LAYER,
    device='$DEVICE',
)

features = [int(x) for x in '$TOP_FEATURES'.split(',')]
# 正向 steering（增强 CALL 特征）和负向（抑制）
strengths = [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0]

results = exp.run_steering(
    prompts=prompts,
    feature_indices=features,
    strengths=strengths,
    output_dir='$OUTPUT_BASE/steering/layer_$LAYER',
)

print(f'Baseline call rate: {results[\"baseline\"][\"call_rate\"]:.3f}')
print()
for e in results['experiments']:
    fr = e['flip_rate']
    print(f'  feat={e[\"feature_idx\"]:5d}  alpha={e[\"strength\"]:+5.1f}  '
          f'total_flip={fr[\"total_flip_rate\"]:.3f}  '
          f'no_call→call={fr[\"no_call_to_call\"]:.3f}  '
          f'call→no_call={fr[\"call_to_no_call\"]:.3f}')
"
done

echo ""
echo "Steering results saved to: $OUTPUT_BASE/steering/"
echo "================================================================"
