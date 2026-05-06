#!/bin/bash
set -e

# ── 用法 ──────────────────────────────────────────────────────────────
if [[ -z "$1" ]]; then
    echo "Usage: $0 <MODEL_NAME> [DEVICE]"
    echo ""
    echo "  DEVICE: cuda:0 (default) | cuda:1 | ..."
    echo ""
    echo "Available MODEL_NAME values:"
    echo "  gemma-3-1b-it"
    echo "  gemma-3-4b-it"
    echo "  gemma-4-E2B-it"
    echo "  gemma-4-E4B-it"
    echo "  Ministral-3-3B-Instruct-2512"
    echo "  Ministral-3-8B-Instruct-2512"
    echo "  Qwen3.5-4B"
    echo "  Qwen3.5-9B"
    exit 1
fi

MODEL_NAME="$1"
DEVICE="${2:-${DEVICE:-cuda:0}}"

# ── MODEL_PATH / SAE_PATH / LAYER 查表（写死）────────────────────────
# Set SOURCE_ROOT in your environment or in a .env file (see .env.example).
SOURCE_ROOT="${SOURCE_ROOT:-}"
_MODEL_BASE="${SOURCE_ROOT}/model"
_SAE_BASE="/data/Agent-Tool-Use-MI/checkpoint"

case "$MODEL_NAME" in
  gemma-3-1b-it)
    MODEL_PATH="$_MODEL_BASE/google/gemma-3-1b-it"
    SAE_PATH="$_SAE_BASE/gemma-3-1b-it/stage2/gemma-3-1b-it-L17-d9216-5M-stage2.pt"
    LAYER=17
    ;;
  gemma-3-4b-it)
    MODEL_PATH="$_MODEL_BASE/google/gemma-3-4b-it"
    SAE_PATH="$_SAE_BASE/gemma-3-4b-it/stage2/gemma-3-4b-it-L29-d20480-5M-stage2.pt"
    LAYER=29
    ;;
  gemma-4-E2B-it)
    MODEL_PATH="$_MODEL_BASE/google/gemma-4-E2B-it"
    SAE_PATH="$_SAE_BASE/gemma-4-E2B-it/stage2/gemma-4-E2B-it-L30-d12288-5M-stage2.pt"
    LAYER=30
    ;;
  gemma-4-E4B-it)
    MODEL_PATH="$_MODEL_BASE/google/gemma-4-E4B-it"
    SAE_PATH="$_SAE_BASE/gemma-4-E4B-it/stage2/gemma-4-E4B-it-L30-d20480-5M-stage2.pt"
    LAYER=30
    ;;
  Ministral-3-3B-Instruct-2512)
    MODEL_PATH="$_MODEL_BASE/mistralai/Ministral-3-3B-Instruct-2512"
    SAE_PATH="$_SAE_BASE/Ministral-3-3B-Instruct-2512/stage2/Ministral-3-3B-Instruct-2512-L21-d24576-5M-stage2.pt"
    LAYER=21
    ;;
  Ministral-3-8B-Instruct-2512)
    MODEL_PATH="$_MODEL_BASE/mistralai/Ministral-3-8B-Instruct-2512"
    SAE_PATH="$_SAE_BASE/Ministral-3-8B-Instruct-2512/stage2/Ministral-3-8B-Instruct-2512-L31-d32768-5M-stage2.pt"
    LAYER=31
    ;;
  Qwen3.5-4B)
    MODEL_PATH="$_MODEL_BASE/Qwen/Qwen3.5-4B"
    SAE_PATH="$_SAE_BASE/Qwen3.5-4B/stage2/Qwen3.5-4B-L25-d20480-5M-stage2.pt"
    LAYER=25
    ;;
  Qwen3.5-9B)
    MODEL_PATH="$_MODEL_BASE/Qwen/Qwen3.5-9B"
    SAE_PATH="$_SAE_BASE/Qwen3.5-9B/stage2/Qwen3.5-9B-L25-d32768-5M-stage2.pt"
    LAYER=25
    ;;
  *)
    echo "Error: unknown MODEL_NAME '$MODEL_NAME'" >&2
    exit 1
    ;;
esac

# ── 其余配置 ──────────────────────────────────────────────────────────
DATA_BASE="${DATA_BASE:-$SOURCE_ROOT/dataset}"
OUTPUT_BASE="${OUTPUT_BASE:-/data/Agent-Tool-Use-MI}"
DTYPE="${DTYPE:-bfloat16}"

WHEN2CALL_TEST="$DATA_BASE/when2call/test"

JUDGE_MODEL="${JUDGE_MODEL:-$SOURCE_ROOT/model/Qwen/Qwen3.5-27B}"
JUDGE_DEVICE="${JUDGE_DEVICE:-$DEVICE}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-1024}"

NUM_SAMPLES="${NUM_SAMPLES:--1}"

OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_BASE/outputs/$MODEL_NAME/amcs}"

# ── top_n sweep 参数 ──────────────────────────────────────────────────
if [[ "$MODEL_NAME" == gemma-* ]]; then
    TOP_N_VALUES="${TOP_N_VALUES:-6}"
else
    TOP_N_VALUES="${TOP_N_VALUES:-30}"
fi
ALPHA_VALUES="${ALPHA_VALUES:-0.2 0.6 1.0}"

FEATURE_DISCOVERY_DIR="${FEATURE_DISCOVERY_DIR:-$OUTPUT_BASE/outputs/$MODEL_NAME/analysis/feature_discovery}"
RFI_CONFUSION_DIR="${RFI_CONFUSION_DIR:-$OUTPUT_BASE/outputs/$MODEL_NAME/analysis/rfi_confusion}"
ACTIVATIONS_DIR="${ACTIVATIONS_DIR:-$OUTPUT_BASE/outputs/$MODEL_NAME/activations/when2call_mcq}"

echo "================================================================"
echo "AMCS top_n Sweep (When2Call test set)"
echo "  Model      : $MODEL_PATH"
echo "  SAE        : $SAE_PATH"
echo "  Layer      : $LAYER"
echo "  Data       : $WHEN2CALL_TEST"
echo "  Samples    : $NUM_SAMPLES"
echo "  Judge      : $JUDGE_MODEL"
echo "  Output     : $OUTPUT_DIR"
echo "  top_n vals : $TOP_N_VALUES"
echo "  α values  : $ALPHA_VALUES"
echo "  FeatureDisc: $FEATURE_DISCOVERY_DIR"
echo "  RFI Conf   : $RFI_CONFUSION_DIR"
echo "  Activations: $ACTIVATIONS_DIR"
echo "================================================================"

# ── 先生成 rfi_confusion（如已存在则跳过）────────────────────────────
RFI_JSON="$RFI_CONFUSION_DIR/rfi_confusion_layer${LAYER}.json"
if [[ -f "$RFI_JSON" ]]; then
    echo "rfi_confusion already exists, skipping: $RFI_JSON"
else
    echo "================================================================"
    echo "Step 1/2: generate rfi_confusion_layer${LAYER}.json"
    echo "================================================================"
    python -m utils_validation.analyze_rfi_confusion \
      --layer                "$LAYER"                  \
      --sae-path             "$SAE_PATH"               \
      --activations-dir      "$ACTIVATIONS_DIR"        \
      --feature-discovery-dir "$FEATURE_DISCOVERY_DIR" \
      --output-dir           "$RFI_CONFUSION_DIR"
fi

echo "================================================================"
echo "Step 2/2: AMCS top_n Sweep"
echo "================================================================"
python -m run.sweep_amcs_topn \
  --model                "$MODEL_PATH"             \
  --sae-path             "$SAE_PATH"               \
  --layer                "$LAYER"                  \
  --data-path            "$WHEN2CALL_TEST"         \
  --num-samples          "$NUM_SAMPLES"            \
  --output-dir           "$OUTPUT_DIR"             \
  --device               "$DEVICE"                 \
  --dtype                "$DTYPE"                  \
  --judge-model          "$JUDGE_MODEL"            \
  --judge-device         "$JUDGE_DEVICE"           \
  --judge-max-new-tokens "$JUDGE_MAX_NEW_TOKENS"   \
  --top-n-values         $TOP_N_VALUES             \
  --alpha                $ALPHA_VALUES              \
  --feature-discovery-dir "$FEATURE_DISCOVERY_DIR" \
  --rfi-confusion-dir    "$RFI_CONFUSION_DIR"      \
  --activations-dir      "$ACTIVATIONS_DIR"

# ── top_n acc 汇总 + 平均值（每个 alpha 一张表）─────────────────────
echo ""
echo "================================================================"
echo "top_n Accuracy Summary (all alpha values)"
echo "================================================================"
for _alpha in $ALPHA_VALUES; do
    RESULT_JSON="$OUTPUT_DIR/amcs_alpha${_alpha}_results.json"
    echo ""
    echo "--- α=${_alpha} ---"
    python3 - "$RESULT_JSON" <<'PYEOF'
import json, sys

path = sys.argv[1]
with open(path) as f:
    data = json.load(f)

sweep = data["sweep"]

print(f"  {'top_n':>6}  {'AMCS acc':>10}")
print(f"  {'-'*20}")

accs = []
for row in sweep:
    top_n = row["top_n"]
    acc   = row["amcs"]["accuracy"]
    accs.append(acc)
    acc_str = f"{acc:.4f}" if acc is not None else "   N/A"
    print(f"  {top_n:>6}  {acc_str:>10}")

valid_accs = [a for a in accs if a is not None]
if valid_accs:
    avg = sum(valid_accs) / len(valid_accs)
    print(f"  {'-'*20}")
    print(f"  {'avg':>6}  {avg:>10.4f}")
    print(f"\n  avg AMCS acc : {avg:.4f}  (over {len(valid_accs)} top_n values)")
PYEOF
done

echo "================================================================"
echo "Sweep complete. Output files:"
find "$OUTPUT_DIR" -type f 2>/dev/null | sort
echo "================================================================"
