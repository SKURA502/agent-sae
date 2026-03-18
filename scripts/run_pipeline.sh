#!/bin/bash
set -e

# ── 配置 ─────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B-Instruct}"
DATA_BASE="${DATA_BASE:-./data/raw}"
OUTPUT_BASE="${OUTPUT_BASE:-./outputs}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"

PRETRAIN_DIR="$DATA_BASE/pretrain"
TOOLUSE_DIR="$DATA_BASE/tooluse"
WHEN2CALL_DIR="$DATA_BASE/when2call"
BFCL_DIR="$DATA_BASE/bfcl"

STAGE1_TARGET_TOKENS=50000000   # 50M tokens
STAGE2_TARGET_TOKENS=10000000   # ~10M tokens（When2Call Pref 全量对话文本）
STAGE2_LR=5e-4
STAGE2_BATCH=4096

LAYERS="24 26"                  # 两个 hook 层

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

# ── Step 2：Stage 2 SAE 训练（When2Call 工具调用对话文本，全 token 流式）──
echo "▶ Step 2: SAE Stage 2 training (tool-use JSONL, full-text streaming, ${STAGE2_TARGET_TOKENS} tokens)"

for LAYER in $LAYERS; do
  echo "  Layer $LAYER ..."
  S1_CKPT=$(find "$OUTPUT_BASE/sae_checkpoints/stage1" \
    -name "*-L${LAYER}-*-stage1.pt" 2>/dev/null | sort | head -1)

  python -m sae.train_sae stage2 \
    --model "$MODEL_PATH" \
    --layer "$LAYER" \
    --data-dir "$TOOLUSE_DIR" \
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

# ── Step 3：提取测试集激活（H1 主测：When2Call MCQ 二类子集）────
echo "▶ Step 3: Extract activations — When2Call MCQ (H1 主测)"

python -m run.cache_activations extract \
  --model "$MODEL_PATH" \
  --dataset when2call \
  --data-path "$WHEN2CALL_DIR" \
  --split test_mcq \
  --num-samples -1 \
  --layers $LAYERS \
  --output-dir "$OUTPUT_BASE/activations/when2call_mcq" \
  --device "$DEVICE" \
  --dtype "$DTYPE"

# ── Step 4：提取泛化测试集激活（BFCL Irrelevance + Simple）───────
echo "▶ Step 4: Extract activations — BFCL generalization (H1 泛化)"

python -m run.cache_activations extract \
  --model "$MODEL_PATH" \
  --dataset bfcl \
  --data-path "$BFCL_DIR" \
  --num-samples -1 \
  --layers $LAYERS \
  --output-dir "$OUTPUT_BASE/activations/bfcl_gen" \
  --device "$DEVICE" \
  --dtype "$DTYPE"

# ── Step 5：特征发现与线性探针（H1）─────────────────────────────
echo "▶ Step 5: H1 — Correlation analysis + Linear probe"

for LAYER in $LAYERS; do
  # 找到对应 Stage 2 checkpoint
  SAE_PATH=$(python -c "
from pathlib import Path
matches = sorted(Path('$OUTPUT_BASE/sae_checkpoints/stage2').glob('*-L${LAYER}-*-stage2.pt'))
if not matches:
    matches = sorted(Path('$OUTPUT_BASE/sae_checkpoints/stage2').glob('*-layer${LAYER}-*-stage2.pt'))
print(matches[0] if matches else '')
")
  if [[ -z "$SAE_PATH" ]]; then
    echo "  Warning: no Stage 2 checkpoint for layer $LAYER, skipping"
    continue
  fi

  echo "  Layer $LAYER, SAE: $SAE_PATH"

  # 相关性分析
  python -m analysis.correlation_analysis \
    --sae-path "$SAE_PATH" \
    --data-path "$OUTPUT_BASE/activations/when2call_mcq/layer_${LAYER}_activations.pt" \
    --layer "$LAYER" \
    --output-dir "$OUTPUT_BASE/analysis/layer_${LAYER}/when2call_mcq" \
    --device "$DEVICE"

  # 线性探针
  python -m analysis.linear_probe \
    --sae-path "$SAE_PATH" \
    --data-path "$OUTPUT_BASE/activations/when2call_mcq/layer_${LAYER}_activations.pt" \
    --layer "$LAYER" \
    --output-dir "$OUTPUT_BASE/analysis/layer_${LAYER}/when2call_mcq" \
    --device "$DEVICE"

  # BFCL 泛化验证
  python -m analysis.correlation_analysis \
    --sae-path "$SAE_PATH" \
    --data-path "$OUTPUT_BASE/activations/bfcl_gen/layer_${LAYER}_activations.pt" \
    --layer "$LAYER" \
    --output-dir "$OUTPUT_BASE/analysis/layer_${LAYER}/bfcl_gen" \
    --device "$DEVICE"
done

# ── Step 6：Steering 实验（H3）───────────────────────────────────
echo "▶ Step 6: H3 — Steering / Ablation"

for LAYER in $LAYERS; do
  SAE_PATH=$(python -c "
from pathlib import Path
matches = sorted(Path('$OUTPUT_BASE/sae_checkpoints/stage2').glob('*-L${LAYER}-*-stage2.pt'))
if not matches:
    matches = sorted(Path('$OUTPUT_BASE/sae_checkpoints/stage2').glob('*-layer${LAYER}-*-stage2.pt'))
print(matches[0] if matches else '')
")
  if [[ -z "$SAE_PATH" ]]; then
    echo "  Warning: no Stage 2 checkpoint for layer $LAYER, skipping"
    continue
  fi

  # 从分析结果取 top-5 CALL 门控特征
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
from tasks import When2CallAdapter

# 加载 MCQ 二类子集作为 steering 测试集（排除 request_for_info）
from tasks.base_adapter import DecisionLabel
adapter = When2CallAdapter('$WHEN2CALL_DIR', split='test_mcq')
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
