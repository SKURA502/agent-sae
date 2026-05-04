#!/bin/bash
# Feature Discovery: 发现 tool_call 或 request_for_info 相关的 SAE 概念
#
# 用法：
#   bash feature_discovery_when2call.sh                        # 默认 tool_call
#   CONCEPT=request_for_info bash feature_discovery_when2call.sh
#
# 输出（以 tool_call 为例）：
#   $OUTPUT_BASE/outputs/$MODEL_NAME/analysis/feature_discovery/tool_call/feature_scores_layer{L}.json
#   $OUTPUT_BASE/outputs/$MODEL_NAME/analysis/feature_discovery/tool_call/top_features_layer{L}.json
#   $OUTPUT_BASE/outputs/$MODEL_NAME/analysis/feature_discovery/tool_call/feature_discovery_layer{L}.png
#   $OUTPUT_BASE/outputs/$MODEL_NAME/analysis/feature_discovery/tool_call/top_features_bar_layer{L}.png
#   $OUTPUT_BASE/outputs/$MODEL_NAME/analysis/feature_discovery/tool_call/decoder_umap_layer{L}.png
set -e

# ── 配置 ──────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-gpfs2/safelens-share-gpfs2/source/model/google/gemma-4-E4B-it}"
OUTPUT_BASE="${OUTPUT_BASE:-/data/Agent-Tool-Use-MI}"
LAYER="${LAYER:-30}"
CONCEPT="${CONCEPT:-request_for_info}"   # tool_call | request_for_info
SAE_PATH="${SAE_PATH:-$OUTPUT_BASE/checkpoint/gemma-4-E4B-it/stage2/gemma-4-E4B-it-L30-d20480-5M-stage2.pt}"

MODEL_NAME="$(basename "$MODEL_PATH")"
ACTIVATIONS_DIR="${ACTIVATIONS_DIR:-$OUTPUT_BASE/outputs/$MODEL_NAME/activations/when2call_mcq}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_BASE/outputs/$MODEL_NAME/analysis/feature_discovery/$CONCEPT}"
DEVICE="${DEVICE:-cuda}"

# 联合筛选阈值
MIN_MEAN_DIFF="${MIN_MEAN_DIFF:-0.0}"   # mean_diff 必须 > 此值
MIN_AUROC="${MIN_AUROC:-0.50}"           # AUROC 必须 > 此值
# 筛选后最多保留 top-K 特征
TOP_K="${TOP_K:-30}"
# UMAP 高亮的 feature 数（取 top_k 的前 N 个）
UMAP_TOP_K="${UMAP_TOP_K:-20}"
UMAP_N_NEIGHBORS="${UMAP_N_NEIGHBORS:-15}"
UMAP_MIN_DIST="${UMAP_MIN_DIST:-0.1}"

ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-512}"

echo "================================================================"
echo "Feature Discovery: $CONCEPT concept analysis"
echo "  Model          : $MODEL_PATH"
echo "  Layer          : $LAYER"
echo "  Concept        : $CONCEPT"
echo "  SAE checkpoint : $SAE_PATH"
echo "  Activations    : $ACTIVATIONS_DIR"
echo "  Output dir     : $OUTPUT_DIR"
echo "  Filter         : mean_diff > $MIN_MEAN_DIFF  AND  auroc > $MIN_AUROC"
echo "  top-K          : $TOP_K  (after filter)"
echo "  UMAP highlight : $UMAP_TOP_K  features"
echo "================================================================"

cd "$OUTPUT_BASE"

python -m analysis.feature_discovery \
  --layer              "$LAYER" \
  --concept            "$CONCEPT" \
  --sae-path           "$SAE_PATH" \
  --activations-dir    "$ACTIVATIONS_DIR" \
  --output-dir         "$OUTPUT_DIR" \
  --top-k              "$TOP_K" \
  --min-mean-diff      "$MIN_MEAN_DIFF" \
  --min-auroc          "$MIN_AUROC" \
  --umap-top-k         "$UMAP_TOP_K" \
  --umap-n-neighbors   "$UMAP_N_NEIGHBORS" \
  --umap-min-dist      "$UMAP_MIN_DIST" \
  --device             "$DEVICE" \
  --encode-batch-size  "$ENCODE_BATCH_SIZE"

echo "================================================================"
echo "Done. Output files:"
find "$OUTPUT_DIR" -name "*layer${LAYER}*" 2>/dev/null | sort
echo "================================================================"
