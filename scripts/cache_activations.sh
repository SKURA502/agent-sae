#!/bin/bash
set -e

# ── 配置 ─────────────────────────────────────────────────────────
# Set SOURCE_ROOT in your environment or in a .env file (see .env.example).
SOURCE_ROOT="${SOURCE_ROOT:-}"
MODEL_PATH="${MODEL_PATH:-$SOURCE_ROOT/model/google/gemma-3-1b-it}"
DATA_BASE="${DATA_BASE:-$SOURCE_ROOT/dataset}"
OUTPUT_BASE="${OUTPUT_BASE:-/data/Agent-Tool-Use-MI}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"

WHEN2CALL_TEST="$DATA_BASE/when2call/test"

LAYERS="17"
HOOK_POSITION="last"   # last | last_t | all
LAST_T=8               # 仅 hook-position=last_t 时生效
MAX_LENGTH=2048
NUM_SAMPLES="${NUM_SAMPLES:--1}"   # -1 = 全量；设小值快速测试，如 NUM_SAMPLES=50

# judge 配置
JUDGE_MODEL="${JUDGE_MODEL:-$SOURCE_ROOT/model/Qwen/Qwen3.5-27B}"
JUDGE_DEVICE="${JUDGE_DEVICE:-cuda:0}"          # 多卡时主模型和 judge 分开放
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-512}"

MODEL_NAME="$(basename "$MODEL_PATH")"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_BASE/outputs/$MODEL_NAME/activations/when2call_mcq}"

echo "================================================================"
echo "Cache Activations (When2Call MCQ test set)"
echo "  Model        : $MODEL_PATH"
echo "  Data         : $WHEN2CALL_TEST"
echo "  Layers       : $LAYERS"
echo "  Hook position: $HOOK_POSITION"
echo "  Judge model  : $JUDGE_MODEL"
echo "  Judge device : $JUDGE_DEVICE"
echo "  Judge max tok: $JUDGE_MAX_NEW_TOKENS"
echo "  Output dir   : $OUTPUT_DIR"
echo "================================================================"

python -m run.cache_activations extract \
  --model                "$MODEL_PATH" \
  --dataset              when2call \
  --data-path            "$WHEN2CALL_TEST" \
  --split                test_mcq \
  --num-samples          "$NUM_SAMPLES" \
  --layers               $LAYERS \
  --output-dir           "$OUTPUT_DIR" \
  --hook-position        "$HOOK_POSITION" \
  --last-t               $LAST_T \
  --max-length           $MAX_LENGTH \
  --device               "$DEVICE" \
  --dtype                "$DTYPE" \
  --judge-model          "$JUDGE_MODEL" \
  --judge-device         "$JUDGE_DEVICE" \
  --judge-max-new-tokens "$JUDGE_MAX_NEW_TOKENS"

echo "================================================================"
echo "Complete. Files:"
find "$OUTPUT_DIR" -type f 2>/dev/null | sort
echo "================================================================"
