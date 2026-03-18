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


