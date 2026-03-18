#!/bin/bash
set -e

# ── 配置 ─────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B-Instruct}"
DATA_BASE="${DATA_BASE:-./data/raw}"
OUTPUT_BASE="${OUTPUT_BASE:-./outputs}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"

PRETRAIN_DIR="$DATA_BASE/pretrain"
WHEN2CALL_TRAIN="$DATA_BASE/When2Call/data/train"
WHEN2CALL_TEST="$DATA_BASE/When2Call/data/test"

STAGE1_TARGET_TOKENS=50000000   # 50M tokens
STAGE2_TARGET_TOKENS=5000000    # ~5M tokens（When2Call pref+sft 全量约一轮）
STAGE2_LR=5e-4
STAGE2_BATCH=4096

LAYERS="24 26"

echo "================================================================"
echo "Agent-Tool-Use-MI Pipeline"
echo "  Model : $MODEL_PATH"
echo "  Layers: $LAYERS"
echo "================================================================"

# ── Step 1：Stage 1 SAE 训练（OpenWebText2，50M tokens）──────────
echo "▶ Step 1: SAE Stage 1 training (${STAGE1_TARGET_TOKENS} tokens)"

for LAYER in $LAYERS; do
  echo "  Layer $LAYER ..."
  python -m sae.train_sae stage1 \
    --model "$MODEL_PATH" \
    --layer "$LAYER" \
    --output-dir "$OUTPUT_BASE/sae_checkpoints" \
    --target-tokens $STAGE1_TARGET_TOKENS \
    --seq-length 1024 \
    --inference-batch-size 32 \
    --batch-size 4096 \
    --learning-rate 1e-5 \
    --data-dir "$PRETRAIN_DIR" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --use-swanlab
done

echo "  Stage 1 checkpoints:"
find "$OUTPUT_BASE/sae_checkpoints/stage1" -name "*-stage1.pt" 2>/dev/null | sort

# ── Step 2：Stage 2 SAE 训练（When2Call pref+sft，action boundary streaming）─
echo "▶ Step 2: SAE Stage 2 training (When2Call pref+sft, ${STAGE2_TARGET_TOKENS} tokens)"

for LAYER in $LAYERS; do
  echo "  Layer $LAYER ..."
  S1_CKPT=$(find "$OUTPUT_BASE/sae_checkpoints/stage1" \
    -name "*-L${LAYER}-*-stage1.pt" 2>/dev/null | sort | head -1)

  python -m sae.train_sae stage2 \
    --model "$MODEL_PATH" \
    --layer "$LAYER" \
    --data-dir "$WHEN2CALL_TRAIN" \
    --stage1-checkpoint "$S1_CKPT" \
    --output-dir "$OUTPUT_BASE/sae_checkpoints" \
    --target-tokens $STAGE2_TARGET_TOKENS \
    --learning-rate $STAGE2_LR \
    --batch-size $STAGE2_BATCH \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --use-swanlab
done

echo "  Stage 2 checkpoints:"
find "$OUTPUT_BASE/sae_checkpoints/stage2" -name "*-stage2.pt" 2>/dev/null | sort

# ── Step 3：提取测试集激活（H1/H3：When2Call MCQ 二类子集）───────
echo "▶ Step 3: Extract activations — When2Call MCQ (H1/H3)"

python -m run.cache_activations extract \
  --model "$MODEL_PATH" \
  --dataset when2call \
  --data-path "$WHEN2CALL_TEST" \
  --split test_mcq \
  --num-samples -1 \
  --layers $LAYERS \
  --output-dir "$OUTPUT_BASE/activations/when2call_mcq" \
  --hook-position last \
  --device "$DEVICE" \
  --dtype "$DTYPE"

# ── Step 4：特征发现（H1）────────────────────────────────────────
echo "▶ Step 4: H1 — Feature discovery (mean activation diff) + Linear probe"

for LAYER in $LAYERS; do
  SAE_PATH=$(python -c "
from pathlib import Path
matches = sorted(Path('$OUTPUT_BASE/sae_checkpoints/stage2').glob('*-L${LAYER}-*-stage2.pt'))
print(matches[0] if matches else '')
")
  if [[ -z "$SAE_PATH" ]]; then
    echo "  Warning: no Stage 2 checkpoint for layer $LAYER, skipping"
    continue
  fi

  echo "  Layer $LAYER, SAE: $SAE_PATH"

  python -m analysis.correlation_analysis \
    --sae-path "$SAE_PATH" \
    --data-path "$OUTPUT_BASE/activations/when2call_mcq/layer_${LAYER}_activations.pt" \
    --layer "$LAYER" \
    --output-dir "$OUTPUT_BASE/analysis/layer_${LAYER}/when2call_mcq" \
    --device "$DEVICE"

  python -m analysis.linear_probe \
    --sae-path "$SAE_PATH" \
    --data-path "$OUTPUT_BASE/activations/when2call_mcq/layer_${LAYER}_activations.pt" \
    --layer "$LAYER" \
    --output-dir "$OUTPUT_BASE/analysis/layer_${LAYER}/when2call_mcq" \
    --device "$DEVICE"
done

# ── Step 5：Steering 实验（H3）───────────────────────────────────
echo "▶ Step 5: H3 — Steering / Ablation"

for LAYER in $LAYERS; do
  SAE_PATH=$(python -c "
from pathlib import Path
matches = sorted(Path('$OUTPUT_BASE/sae_checkpoints/stage2').glob('*-L${LAYER}-*-stage2.pt'))
print(matches[0] if matches else '')
")
  if [[ -z "$SAE_PATH" ]]; then
    echo "  Warning: no Stage 2 checkpoint for layer $LAYER, skipping"
    continue
  fi

  TOP_FEATURES=$(python -c "
import json
from pathlib import Path
summary = Path('$OUTPUT_BASE/analysis/layer_${LAYER}/when2call_mcq/analysis_summary.json')
if summary.exists():
    data = json.loads(summary.read_text())
    print(','.join(map(str, data.get('top_20_call_features', [])[:5])))
else:
    print('')
")
  if [[ -z "$TOP_FEATURES" ]]; then
    echo "  Warning: no analysis summary for layer $LAYER, skipping steering"
    continue
  fi

  echo "  Layer $LAYER, features: $TOP_FEATURES"

  python -c "
import sys, torch
sys.path.insert(0, '.')
from analysis import SteeringExperiment
from run.when2call_adapter import When2CallAdapter, DecisionLabel

adapter = When2CallAdapter('$WHEN2CALL_TEST', split='test_mcq')
adapter.load()
samples = [s for s in adapter if s.label in (DecisionLabel.CALL, DecisionLabel.NO_CALL)][:200]
prompts = [s.instruction for s in samples]

exp = SteeringExperiment(
    sae_path='$SAE_PATH',
    model_name='$MODEL_PATH',
    hook_layer=$LAYER,
    device='$DEVICE',
)
features = [int(x) for x in '$TOP_FEATURES'.split(',')]
strengths = [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]

results = exp.run_steering(
    prompts=prompts,
    feature_indices=features[:3],
    strengths=strengths,
    output_dir='$OUTPUT_BASE/steering/layer_$LAYER',
)
print(f'Baseline call rate: {results[\"baseline\"][\"call_rate\"]:.3f}')
for e in results['experiments']:
    print(f'  feat={e[\"feature_idx\"]} alpha={e[\"strength\"]:+.1f} '
          f'flip={e[\"flip_rate\"][\"total_flip_rate\"]:.3f}')
"
done

echo "================================================================"
echo "Pipeline complete!  Results: $OUTPUT_BASE"
echo "================================================================"
