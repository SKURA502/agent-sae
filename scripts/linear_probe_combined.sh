#!/bin/bash
# Run linear probe for all six target models (both concept labels), then
# generate a single combined 2×3 figure comparing all results.
#
# Prerequisites: feature_scores_layer{L}.json must already exist for each
#   model (produced by feature_discovery_when2call.sh).
#
# Outputs per model:
#   $OUTPUT_BASE/outputs/$MODEL_NAME/analysis/linear_probe/$LABEL/
#     linear_probe_layer{L}.json
#
# Combined figure:
#   $OUTPUT_BASE/outputs/linear_probe_combined.png
#
# Usage:
#   bash scripts/linear_probe_combined.sh
#   SKIP_EXISTING=false bash scripts/linear_probe_combined.sh   # force re-run
set -e

# ── Global settings ───────────────────────────────────────────────────────────
OUTPUT_BASE="${OUTPUT_BASE:-/data/Agent-Tool-Use-MI}"
DEVICE="${DEVICE:-cuda}"
TOP_K="${TOP_K:-50}"
K_VALUES="${K_VALUES:-1 2 3 4 5 10 20 30 50 100}"
CV_FOLDS="${CV_FOLDS:-5}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-512}"
# Set to "true" to skip a run if the output JSON already exists
SKIP_EXISTING="${SKIP_EXISTING:-true}"

MODEL_BASE="/mnt/shared-storage-gpfs2/safelens-share-gpfs2/source/model"

# ── Model registry ────────────────────────────────────────────────────────────
# Format: "org_subdir/model_name:layer:sae_checkpoint_filename"
MODELS=(
    "Qwen/Qwen3.5-4B:25:Qwen3.5-4B-L25-d20480-5M-stage2.pt"
    "Qwen/Qwen3.5-9B:25:Qwen3.5-9B-L25-d32768-5M-stage2.pt"
    "mistralai/Ministral-3-3B-Instruct-2512:21:Ministral-3-3B-Instruct-2512-L21-d24576-5M-stage2.pt"
    "mistralai/Ministral-3-8B-Instruct-2512:31:Ministral-3-8B-Instruct-2512-L31-d32768-5M-stage2.pt"
    "google/gemma-3-1b-it:17:gemma-3-1b-it-L17-d9216-5M-stage2.pt"
    "google/gemma-3-4b-it:29:gemma-3-4b-it-L29-d20480-5M-stage2.pt"
)

LABELS=("tool_call" "request_for_info")

# ── Run probes ────────────────────────────────────────────────────────────────
for entry in "${MODELS[@]}"; do
    IFS=':' read -r model_rel layer sae_file <<< "$entry"
    model_path="$MODEL_BASE/$model_rel"
    model_name="$(basename "$model_rel")"
    sae_path="$OUTPUT_BASE/checkpoint/$model_name/stage2/$sae_file"
    activations_dir="$OUTPUT_BASE/outputs/$model_name/activations/when2call_mcq"

    for label in "${LABELS[@]}"; do
        feature_scores="$OUTPUT_BASE/outputs/$model_name/analysis/feature_discovery/$label/feature_scores_layer${layer}.json"
        output_dir="$OUTPUT_BASE/outputs/$model_name/analysis/linear_probe/$label"
        out_json="$output_dir/linear_probe_layer${layer}.json"

        echo "================================================================"
        echo "  Model : $model_name  |  Label: $label  |  Layer: $layer"

        if [ "$SKIP_EXISTING" = "true" ] && [ -f "$out_json" ]; then
            echo "  SKIP  : output already exists → $out_json"
            continue
        fi

        if [ ! -f "$feature_scores" ]; then
            echo "  SKIP  : feature scores not found → $feature_scores"
            continue
        fi

        if [ ! -d "$activations_dir" ]; then
            echo "  SKIP  : activations dir not found → $activations_dir"
            continue
        fi

        if [ ! -f "$sae_path" ]; then
            echo "  SKIP  : SAE checkpoint not found → $sae_path"
            continue
        fi

        cd "$OUTPUT_BASE"
        python -m analysis.linear_probe \
            --layer               "$layer" \
            --sae-path            "$sae_path" \
            --activations-dir     "$activations_dir" \
            --feature-scores-path "$feature_scores" \
            --output-dir          "$output_dir" \
            --label               "$label" \
            --top-k               "$TOP_K" \
            --k-values            $K_VALUES \
            --cv-folds            "$CV_FOLDS" \
            --device              "$DEVICE" \
            --encode-batch-size   "$ENCODE_BATCH_SIZE"
    done
done

# ── Combined figure ───────────────────────────────────────────────────────────
echo "================================================================"
echo "Generating combined 2×3 figure..."
cd "$OUTPUT_BASE"
python -m analysis.plot_linear_probe_combined \
    --output-base "$OUTPUT_BASE" \
    --output-path "$OUTPUT_BASE/outputs/linear_probe_combined.pdf"

echo "================================================================"
echo "Done."
ls -lh "$OUTPUT_BASE/outputs/linear_probe_combined.png" 2>/dev/null || true
echo "================================================================"
