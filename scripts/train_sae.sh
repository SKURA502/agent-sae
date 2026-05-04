#!/bin/bash
set -e

# ── 配置 ─────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-gpfs2/safelens-share-gpfs2/source/model/google/gemma-3-4b-it}"
DATA_BASE="${DATA_BASE:-./data/raw}"
CKPT_BASE="${CKPT_BASE:-./checkpoint}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"

MODEL_NAME=$(basename "$MODEL_PATH" | sed 's/___/./g')
STAGE1_DIR="$CKPT_BASE/$MODEL_NAME/stage1"
STAGE2_DIR="$CKPT_BASE/$MODEL_NAME/stage2"

mkdir -p "$STAGE1_DIR" "$STAGE2_DIR"

PRETRAIN_DIR="${PRETRAIN_DIR:-/mnt/shared-storage-gpfs2/safelens-share-gpfs2/source/dataset/Agent_tool_intrep_pretrain}"
WHEN2CALL_TRAIN="${WHEN2CALL_TRAIN:-/mnt/shared-storage-gpfs2/safelens-share-gpfs2/source/dataset/when2call/train}"

STAGE1_TARGET_TOKENS=50000000   # 50M tokens
STAGE2_TARGET_TOKENS=5000000    # ~5M tokens（When2Call pref+sft 全量约一轮）
STAGE2_LR=5e-4
STAGE2_BATCH=4096
STAGE2_SEQ_LENGTH="${STAGE2_SEQ_LENGTH:-2048}"

LAYERS="29"

echo "================================================================"
echo "SAE Training"
echo "  Model      : $MODEL_PATH"
echo "  Layers     : $LAYERS"
echo "  Stage1 dir : $STAGE1_DIR"
echo "  Stage2 dir : $STAGE2_DIR"
echo "================================================================"

# ── Step 1：Stage 1 SAE 训练（OpenWebText2，50M tokens）──────────
echo "▶ Step 1: SAE Stage 1 training (${STAGE1_TARGET_TOKENS} tokens)"

for LAYER in $LAYERS; do
  echo "  Layer $LAYER ..."
  S1_EXISTING=$(find "$STAGE1_DIR" -name "*-L${LAYER}-*-stage1.pt" 2>/dev/null | sort | head -1)
  if [ -n "$S1_EXISTING" ] && [ "${FORCE_STAGE1:-0}" != "1" ]; then
    echo "  Layer $LAYER: Stage 1 checkpoint already exists, skipping (set FORCE_STAGE1=1 to retrain)"
    echo "    $S1_EXISTING"
  else
    python -m sae.train_sae stage1 \
      --model "$MODEL_PATH" \
      --layer "$LAYER" \
      --output-dir "$CKPT_BASE/$MODEL_NAME" \
      --target-tokens $STAGE1_TARGET_TOKENS \
      --seq-length 1024 \
      --inference-batch-size 64 \
      --batch-size 16384 \
      --learning-rate 5e-4 \
      --data-dir "$PRETRAIN_DIR" \
      --device "$DEVICE" \
      --dtype "$DTYPE"
  fi
done

echo "  Stage 1 checkpoints:"
find "$STAGE1_DIR" -name "*-stage1.pt" 2>/dev/null | sort || true

# ── Step 2：Stage 2 SAE 训练（When2Call pref+sft，action boundary streaming）─
echo "▶ Step 2: SAE Stage 2 training (When2Call pref+sft, ${STAGE2_TARGET_TOKENS} tokens)"

for LAYER in $LAYERS; do
  echo "  Layer $LAYER ..."
  S1_CKPT=$(find "$STAGE1_DIR" \
    -name "*-L${LAYER}-*-stage1.pt" 2>/dev/null | sort | head -1)
  S2_EXISTING=$(find "$STAGE2_DIR" -name "*-L${LAYER}-*-stage2.pt" 2>/dev/null | sort | head -1)
  if [ -n "$S2_EXISTING" ] && [ "${FORCE_STAGE2:-0}" != "1" ]; then
    echo "  Layer $LAYER: Stage 2 checkpoint already exists, skipping (set FORCE_STAGE2=1 to retrain)"
    echo "    $S2_EXISTING"
  else
    python -m sae.train_sae stage2 \
      --model "$MODEL_PATH" \
      --layer "$LAYER" \
      --data-dir "$WHEN2CALL_TRAIN" \
      --stage1-checkpoint "$S1_CKPT" \
      --output-dir "$CKPT_BASE/$MODEL_NAME" \
      --target-tokens $STAGE2_TARGET_TOKENS \
      --seq-length $STAGE2_SEQ_LENGTH \
      --learning-rate $STAGE2_LR \
      --batch-size $STAGE2_BATCH \
      --device "$DEVICE" \
      --dtype "$DTYPE"
  fi
done

echo "  Stage 2 checkpoints:"
find "$STAGE2_DIR" -name "*-stage2.pt" 2>/dev/null | sort || true

# ── Plot loss curves ──────────────────────────────────────────────
echo "▶ Plotting loss curves"

for LAYER in $LAYERS; do
  S1_JSON=$(find "$STAGE1_DIR" -name "*-L${LAYER}-*-stage1_stats.json" 2>/dev/null | sort | head -1)
  S2_JSON=$(find "$STAGE2_DIR" -name "*-L${LAYER}-*-stage2_stats.json" 2>/dev/null | sort | head -1)

  if [ -z "$S1_JSON" ] || [ -z "$S2_JSON" ]; then
    echo "  Layer $LAYER: stats JSON not found, skipping plot"
    continue
  fi

  OUT="$CKPT_BASE/$MODEL_NAME/loss_curve_L${LAYER}.png"
  echo "  Layer $LAYER → $OUT"
  python utils/plot_loss.py \
    --stage1-json "$S1_JSON" \
    --stage2-json "$S2_JSON" \
    --output "$OUT" \
    --title "Two-Stage SAE Loss: ${MODEL_NAME} (Layer ${LAYER})"
done

echo "================================================================"
echo "SAE training complete."
echo "  Stage 1: $STAGE1_DIR"
echo "  Stage 2: $STAGE2_DIR"
echo "  Plots  : $CKPT_BASE/$MODEL_NAME/loss_curve_L*.png"
echo "================================================================"
