#!/bin/bash

# ── 全局配置 ──────────────────────────────────────────────────────────
_MODEL_BASE="/mnt/shared-storage-gpfs2/safelens-share-gpfs2/source/model"

DATA_BASE="${DATA_BASE:-/mnt/shared-storage-gpfs2/safelens-share-gpfs2/source/dataset}"
OUTPUT_BASE="${OUTPUT_BASE:-/data/Agent-Tool-Use-MI}"
DEVICE="${DEVICE:-cuda:2}"
DTYPE="${DTYPE:-bfloat16}"

WHEN2CALL_TEST="$DATA_BASE/when2call/test"

JUDGE_MODEL="${JUDGE_MODEL:-/mnt/shared-storage-gpfs2/safelens-share-gpfs2/source/model/Qwen/Qwen3.5-27B}"
JUDGE_DEVICE="${JUDGE_DEVICE:-cuda:2}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-1024}"

NUM_SAMPLES="${NUM_SAMPLES:--1}"

# ── 写死的 6 个模型 ───────────────────────────────────────────────────
declare -A MODEL_PATHS
MODEL_PATHS["gemma-3-1b-it"]="$_MODEL_BASE/google/gemma-3-1b-it"
MODEL_PATHS["gemma-3-4b-it"]="$_MODEL_BASE/google/gemma-3-4b-it"
MODEL_PATHS["Ministral-3-3B-Instruct-2512"]="$_MODEL_BASE/mistralai/Ministral-3-3B-Instruct-2512"
MODEL_PATHS["Ministral-3-8B-Instruct-2512"]="$_MODEL_BASE/mistralai/Ministral-3-8B-Instruct-2512"
MODEL_PATHS["Qwen3.5-4B"]="$_MODEL_BASE/Qwen/Qwen3.5-4B"
MODEL_PATHS["Qwen3.5-9B"]="$_MODEL_BASE/Qwen/Qwen3.5-9B"

MODELS=(
  gemma-3-1b-it
  gemma-3-4b-it
  Ministral-3-3B-Instruct-2512
  Ministral-3-8B-Instruct-2512
  Qwen3.5-4B
  Qwen3.5-9B
)

# ── 逐模型串行运行 ────────────────────────────────────────────────────
_failed=()

for MODEL_NAME in "${MODELS[@]}"; do
    MODEL_PATH="${MODEL_PATHS[$MODEL_NAME]}"
    OUTPUT_DIR="$OUTPUT_BASE/outputs/$MODEL_NAME/baseline_rfi"

    echo "================================================================"
    echo "Prompt-Hint RFI Baseline (When2Call test set)"
    echo "  Model      : $MODEL_PATH"
    echo "  Data       : $WHEN2CALL_TEST"
    echo "  Samples    : $NUM_SAMPLES"
    echo "  Judge      : $JUDGE_MODEL"
    echo "  Output     : $OUTPUT_DIR"
    echo "================================================================"

    if python -m run.eval_baseline_rfi \
          --model                "$MODEL_PATH"             \
          --data-path            "$WHEN2CALL_TEST"         \
          --num-samples          "$NUM_SAMPLES"            \
          --output-dir           "$OUTPUT_DIR"             \
          --device               "$DEVICE"                 \
          --dtype                "$DTYPE"                  \
          --judge-model          "$JUDGE_MODEL"            \
          --judge-device         "$JUDGE_DEVICE"           \
          --judge-max-new-tokens "$JUDGE_MAX_NEW_TOKENS"; then
        echo "================================================================"
        echo "Complete [$MODEL_NAME]. Output files:"
        find "$OUTPUT_DIR" -type f 2>/dev/null | sort
        echo "================================================================"
    else
        echo "================================================================"
        echo "FAILED [$MODEL_NAME]"
        echo "================================================================"
        _failed+=("$MODEL_NAME")
    fi

done

# ── 最终汇报 ──────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "All models done."
if [ ${#_failed[@]} -gt 0 ]; then
    echo "  FAILED: ${_failed[*]}"
    exit 1
else
    echo "  All succeeded."
fi
echo "================================================================"
