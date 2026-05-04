#!/bin/bash
set -e

# ── 用法 ──────────────────────────────────────────────────────────────
if [[ -z "$1" ]]; then
    echo "Usage: $0 <MODEL_NAME> [MODE] [DEVICE]"
    echo ""
    echo "  MODE:   both (default) | promote_rfi | suppress_tc"
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
MODE="${2:-both}"
DEVICE="${3:-${DEVICE:-cuda:0}}"

case "$MODE" in
  both|promote_rfi|suppress_tc) ;;
  *)
    echo "Error: unknown MODE '$MODE'. Must be both, promote_rfi, or suppress_tc." >&2
    exit 1
    ;;
esac

# ── MODEL_PATH / SAE_PATH / LAYER 查表（写死）────────────────────────
_MODEL_BASE="/mnt/shared-storage-gpfs2/safelens-share-gpfs2/source/model"
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
DATA_BASE="${DATA_BASE:-/mnt/shared-storage-gpfs2/safelens-share-gpfs2/source/dataset}"
OUTPUT_BASE="${OUTPUT_BASE:-/data/Agent-Tool-Use-MI}"
DTYPE="${DTYPE:-bfloat16}"

WHEN2CALL_TEST="$DATA_BASE/when2call/test"

JUDGE_MODEL="${JUDGE_MODEL:-/mnt/shared-storage-gpfs2/safelens-share-gpfs2/source/model/Qwen/Qwen3.5-27B}"
JUDGE_DEVICE="${JUDGE_DEVICE:-$DEVICE}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-1024}"

NUM_SAMPLES="${NUM_SAMPLES:--1}"

OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_BASE/outputs/$MODEL_NAME/steer_accuracy}"

# ── steering 参数 ─────────────────────────────────────────────────────
if [[ "$MODEL_NAME" == gemma-* ]]; then
    TOP_N="${TOP_N:-6}"
else
    TOP_N="${TOP_N:-30}"
fi
PROMOTE_RFI_STRENGTHS="${PROMOTE_RFI_STRENGTHS:-1.2 1.6 2.0}"
SUPPRESS_TC_STRENGTHS="${SUPPRESS_TC_STRENGTHS:-0.0 0.4 0.8}"

FEATURE_DISCOVERY_DIR="${FEATURE_DISCOVERY_DIR:-$OUTPUT_BASE/outputs/$MODEL_NAME/analysis/feature_discovery}"

echo "================================================================"
echo "SAE Steering Accuracy Sweep (When2Call test set)"
echo "  Model      : $MODEL_PATH"
echo "  SAE        : $SAE_PATH"
echo "  Layer      : $LAYER"
echo "  Data       : $WHEN2CALL_TEST"
echo "  Samples    : $NUM_SAMPLES"
echo "  Judge      : $JUDGE_MODEL"
echo "  Output     : $OUTPUT_DIR"
echo "  Top-N      : $TOP_N"
echo "  Mode       : $MODE"
echo "  promote_rfi strengths: $PROMOTE_RFI_STRENGTHS"
echo "  suppress_tc strengths: $SUPPRESS_TC_STRENGTHS"
echo "  FeatureDisc: $FEATURE_DISCOVERY_DIR"
echo "================================================================"

# ── Step: promote_rfi ────────────────────────────────────────────────
if [[ "$MODE" == "both" || "$MODE" == "promote_rfi" ]]; then
echo "================================================================"
echo "promote_rfi  (strengths=$PROMOTE_RFI_STRENGTHS)"
echo "================================================================"
python -m run.eval_suppress_accuracy \
  --model                "$MODEL_PATH"             \
  --sae-path             "$SAE_PATH"               \
  --layer                "$LAYER"                  \
  --mode                 promote_rfi               \
  --top-n                $TOP_N                    \
  --data-path            "$WHEN2CALL_TEST"         \
  --num-samples          "$NUM_SAMPLES"            \
  --output-dir           "$OUTPUT_DIR"             \
  --device               "$DEVICE"                 \
  --dtype                "$DTYPE"                  \
  --judge-model          "$JUDGE_MODEL"            \
  --judge-device         "$JUDGE_DEVICE"           \
  --judge-max-new-tokens "$JUDGE_MAX_NEW_TOKENS"   \
  --feature-discovery-dir "$FEATURE_DISCOVERY_DIR" \
  --strengths            $PROMOTE_RFI_STRENGTHS
fi

# ── Step: suppress_tc ────────────────────────────────────────────────
if [[ "$MODE" == "both" || "$MODE" == "suppress_tc" ]]; then
echo "================================================================"
echo "suppress_tc  (strengths=$SUPPRESS_TC_STRENGTHS)"
echo "================================================================"
python -m run.eval_suppress_accuracy \
  --model                "$MODEL_PATH"             \
  --sae-path             "$SAE_PATH"               \
  --layer                "$LAYER"                  \
  --mode                 suppress_tc               \
  --top-n                $TOP_N                    \
  --data-path            "$WHEN2CALL_TEST"         \
  --num-samples          "$NUM_SAMPLES"            \
  --output-dir           "$OUTPUT_DIR"             \
  --device               "$DEVICE"                 \
  --dtype                "$DTYPE"                  \
  --judge-model          "$JUDGE_MODEL"            \
  --judge-device         "$JUDGE_DEVICE"           \
  --judge-max-new-tokens "$JUDGE_MAX_NEW_TOKENS"   \
  --feature-discovery-dir "$FEATURE_DISCOVERY_DIR" \
  --strengths            $SUPPRESS_TC_STRENGTHS
fi

# ── 汇总：promote_rfi ────────────────────────────────────────────────
if [[ "$MODE" == "both" || "$MODE" == "promote_rfi" ]]; then
_PRFI_TAG=$(echo "$PROMOTE_RFI_STRENGTHS" | tr ' ' '-')
RESULT_JSON="$OUTPUT_DIR/promote_rfi_strength${_PRFI_TAG}_sweep.json"
echo ""
echo "================================================================"
echo "top_n Accuracy Summary  [mode=promote_rfi  strengths=$PROMOTE_RFI_STRENGTHS]"
echo "================================================================"
python3 - "$RESULT_JSON" "$PROMOTE_RFI_STRENGTHS" <<'PYEOF'
import json, sys

path      = sys.argv[1]
strengths = sys.argv[2].split()

with open(path) as f:
    entries = json.load(f)

for strength in strengths:
    print(f"\n  strength={strength}")
    print(f"  {'top_n':>6}  {'acc':>8}  {'tc_acc':>8}  {'rfi_acc':>9}")
    print(f"  {'-'*38}")
    accs, tc_accs, rfi_accs = [], [], []
    for entry in entries:
        top_n = entry["top_n"]
        r = entry["results"].get(strength) or entry["results"].get(str(float(strength)))
        if r is None:
            print(f"  {top_n:>6}  {'N/A':>8}  {'N/A':>8}  {'N/A':>9}")
            continue
        acc     = r.get("accuracy")
        pc      = r.get("per_class") or {}
        tc_acc  = pc.get("tool_call")
        rfi_acc = pc.get("request_for_info")
        accs.append(acc)
        tc_accs.append(tc_acc)
        rfi_accs.append(rfi_acc)
        def _fmt(v): return f"{v:.4f}" if v is not None else "    N/A"
        print(f"  {top_n:>6}  {_fmt(acc):>8}  {_fmt(tc_acc):>8}  {_fmt(rfi_acc):>9}")
    print(f"  {'-'*38}")
    def _avg(lst):
        v = [x for x in lst if x is not None]
        return sum(v) / len(v) if v else None
    for label, lst in [("avg", accs), ("tc avg", tc_accs), ("rfi avg", rfi_accs)]:
        a = _avg(lst)
        print(f"  {label:>6}  {_fmt(a):>8}")
PYEOF
fi

# ── 汇总：suppress_tc ────────────────────────────────────────────────
if [[ "$MODE" == "both" || "$MODE" == "suppress_tc" ]]; then
_STC_TAG=$(echo "$SUPPRESS_TC_STRENGTHS" | tr ' ' '-')
RESULT_JSON="$OUTPUT_DIR/suppress_tc_strength${_STC_TAG}_sweep.json"
echo ""
echo "================================================================"
echo "top_n Accuracy Summary  [mode=suppress_tc  strengths=$SUPPRESS_TC_STRENGTHS]"
echo "================================================================"
python3 - "$RESULT_JSON" "$SUPPRESS_TC_STRENGTHS" <<'PYEOF'
import json, sys

path      = sys.argv[1]
strengths = sys.argv[2].split()

with open(path) as f:
    entries = json.load(f)

for strength in strengths:
    print(f"\n  strength={strength}")
    print(f"  {'top_n':>6}  {'acc':>8}  {'tc_acc':>8}  {'rfi_acc':>9}")
    print(f"  {'-'*38}")
    accs, tc_accs, rfi_accs = [], [], []
    for entry in entries:
        top_n = entry["top_n"]
        r = entry["results"].get(strength) or entry["results"].get(str(float(strength)))
        if r is None:
            print(f"  {top_n:>6}  {'N/A':>8}  {'N/A':>8}  {'N/A':>9}")
            continue
        acc     = r.get("accuracy")
        pc      = r.get("per_class") or {}
        tc_acc  = pc.get("tool_call")
        rfi_acc = pc.get("request_for_info")
        accs.append(acc)
        tc_accs.append(tc_acc)
        rfi_accs.append(rfi_acc)
        def _fmt(v): return f"{v:.4f}" if v is not None else "    N/A"
        print(f"  {top_n:>6}  {_fmt(acc):>8}  {_fmt(tc_acc):>8}  {_fmt(rfi_acc):>9}")
    print(f"  {'-'*38}")
    def _avg(lst):
        v = [x for x in lst if x is not None]
        return sum(v) / len(v) if v else None
    for label, lst in [("avg", accs), ("tc avg", tc_accs), ("rfi avg", rfi_accs)]:
        a = _avg(lst)
        print(f"  {label:>6}  {_fmt(a):>8}")
PYEOF
fi

echo ""
echo "================================================================"
echo "Complete. Output files:"
find "$OUTPUT_DIR" -type f 2>/dev/null | sort
echo "================================================================"


