#!/bin/bash
set -e

# ── 配置 ─────────────────────────────────────────────────────────────
# Set SOURCE_ROOT in your environment or in a .env file (see .env.example).
SOURCE_ROOT="${SOURCE_ROOT:-}"
MODEL_PATH="${MODEL_PATH:-$SOURCE_ROOT/model/Qwen/Qwen3.5-4B}"
SAE_PATH="${SAE_PATH:-/data/Agent-Tool-Use-MI/checkpoint/Qwen3.5-4B/stage2/Qwen3.5-4B-L25-d20480-5M-stage2.pt}"
DATA_BASE="${DATA_BASE:-$SOURCE_ROOT/dataset}"
OUTPUT_BASE="${OUTPUT_BASE:-/data/Agent-Tool-Use-MI}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"

WHEN2CALL_TEST="$DATA_BASE/when2call/test"
LAYER="${LAYER:-25}"

JUDGE_MODEL="${JUDGE_MODEL:-$SOURCE_ROOT/model/Qwen/Qwen3.5-27B}"
JUDGE_DEVICE="${JUDGE_DEVICE:-cuda:0}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-1024}"

NUM_SAMPLES="${NUM_SAMPLES:--1}"

MODEL_NAME="$(basename "$MODEL_PATH")"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_BASE/outputs/analysis/$MODEL_NAME/amcs}"

# ── AMCS 参数 ─────────────────────────────────────────────────────────
# DELTA 来自 logistic 分析，一般不需要改动
# ALPHA 是 TC/RFI 预算分配比例，可以在 [0,1] 内调整
DELTA="${DELTA:--0.533}"
ALPHA="${ALPHA:-0.5}"
S_MIN_TC="${S_MIN_TC:-0.40}"
S_MAX_RFI="${S_MAX_RFI:-3.0}"
Z_MIN="${Z_MIN:-0.01}"
TOP_N_FEATURES="${TOP_N_FEATURES:-25}"

# 特征来源目录（默认从 outputs/{MODEL_NAME}/analysis/ 推导）
FEATURE_DISCOVERY_DIR="${FEATURE_DISCOVERY_DIR:-$OUTPUT_BASE/outputs/$MODEL_NAME/analysis/feature_discovery}"
RFI_CONFUSION_DIR="${RFI_CONFUSION_DIR:-$OUTPUT_BASE/outputs/$MODEL_NAME/analysis/rfi_confusion}"

echo "================================================================"
echo "AMCS Bias Correction Evaluation (When2Call test set)"
echo "  Model    : $MODEL_PATH"
echo "  SAE      : $SAE_PATH"
echo "  Layer    : $LAYER"
echo "  Data     : $WHEN2CALL_TEST"
echo "  Samples  : $NUM_SAMPLES"
echo "  Judge    : $JUDGE_MODEL"
echo "  Output   : $OUTPUT_DIR"
echo "  δ=$DELTA  α=$ALPHA  clamp=[$S_MIN_TC, $S_MAX_RFI]  z_min=$Z_MIN  top_n=$TOP_N_FEATURES"
echo "  FeatureDisc: $FEATURE_DISCOVERY_DIR"
echo "  RFI Conf   : $RFI_CONFUSION_DIR"
echo "================================================================"

python -m run.eval_amcs_accuracy \
  --model                "$MODEL_PATH"           \
  --sae-path             "$SAE_PATH"             \
  --layer                "$LAYER"                \
  --data-path            "$WHEN2CALL_TEST"       \
  --num-samples          "$NUM_SAMPLES"          \
  --output-dir           "$OUTPUT_DIR"           \
  --device               "$DEVICE"               \
  --dtype                "$DTYPE"                \
  --judge-model          "$JUDGE_MODEL"          \
  --judge-device         "$JUDGE_DEVICE"         \
  --judge-max-new-tokens "$JUDGE_MAX_NEW_TOKENS" \
  --delta                "$DELTA"                \
  --alpha                "$ALPHA"                \
  --s-min-tc             "$S_MIN_TC"             \
  --s-max-rfi            "$S_MAX_RFI"            \
  --z-min                "$Z_MIN"                \
  --top-n-features       "$TOP_N_FEATURES"       \
  --feature-discovery-dir "$FEATURE_DISCOVERY_DIR" \
  --rfi-confusion-dir    "$RFI_CONFUSION_DIR"

echo "================================================================"
echo "Complete. Output files:"
find "$OUTPUT_DIR" -type f 2>/dev/null | sort
echo "================================================================"
