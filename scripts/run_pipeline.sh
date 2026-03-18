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

# ── Step 3b：提取训练集激活（特征发现用：When2Call Pref 全量）────
echo "▶ Step 3b: Extract activations — When2Call Pref (feature discovery)"

python -m run.cache_activations extract \
  --model "$MODEL_PATH" \
  --dataset when2call \
  --data-path "$WHEN2CALL_TRAIN" \
  --split train_pref \
  --num-samples -1 \
  --layers $LAYERS \
  --output-dir "$OUTPUT_BASE/activations/when2call_pref" \
  --hook-position last \
  --device "$DEVICE" \
  --dtype "$DTYPE"

# ── Step 4：H1 特征发现（When2Call Pref 全量，计算 mean_diff + AUROC）──
echo "▶ Step 4: H1 Feature discovery"

H1_TOP_K="${H1_TOP_K:-100}"

for LAYER in $LAYERS; do
  S2_CKPT=$(find "$OUTPUT_BASE/sae_checkpoints/stage2" \
    -name "*-L${LAYER}-*-stage2.pt" 2>/dev/null | sort | head -1)
  echo "  Layer $LAYER (SAE: $S2_CKPT)..."
  python -m analysis.feature_discovery \
    --layer "$LAYER" \
    --sae-path "$S2_CKPT" \
    --activations-dir "$OUTPUT_BASE/activations/when2call_pref" \
    --output-dir "$OUTPUT_BASE/analysis" \
    --top-k "$H1_TOP_K" \
    --device "$DEVICE"
done

# ── Step 5：H1 线性探针（When2Call MCQ 二类子集，logistic regression）──
echo "▶ Step 5: H1 Linear probe"

for LAYER in $LAYERS; do
  S2_CKPT=$(find "$OUTPUT_BASE/sae_checkpoints/stage2" \
    -name "*-L${LAYER}-*-stage2.pt" 2>/dev/null | sort | head -1)
  python -m analysis.linear_probe \
    --layer "$LAYER" \
    --sae-path "$S2_CKPT" \
    --activations-dir "$OUTPUT_BASE/activations/when2call_mcq" \
    --feature-scores-path "$OUTPUT_BASE/analysis/feature_scores_layer${LAYER}.json" \
    --output-dir "$OUTPUT_BASE/analysis" \
    --top-k 50 \
    --k-values 10 20 50 100 \
    --device "$DEVICE"
done

# ── Step 6：H3 Steering（When2Call MCQ 二类子集，top-5 features per layer）
echo "▶ Step 6: H3 Steering / ablation"

for LAYER in $LAYERS; do
  S2_CKPT=$(find "$OUTPUT_BASE/sae_checkpoints/stage2" \
    -name "*-L${LAYER}-*-stage2.pt" 2>/dev/null | sort | head -1)

  # 取 top-5 feature 索引（从 feature_discovery 输出中提取）
  TOP_FEATS=$(python3 -c "
import json
with open('$OUTPUT_BASE/analysis/top_features_layer${LAYER}.json') as f:
    d = json.load(f)
print(' '.join(str(x['feature_idx']) for x in d[:5]))
")

  echo "  Layer $LAYER top-5 features: $TOP_FEATS"
  python -m analysis.steering \
    --model "$MODEL_PATH" \
    --sae-path "$S2_CKPT" \
    --layer "$LAYER" \
    --feature-indices $TOP_FEATS \
    --alphas 0.5 1.0 2.0 5.0 \
    --output-dir "$OUTPUT_BASE/analysis" \
    --num-samples 500 \
    --device "$DEVICE" \
    --dtype "$DTYPE"
done

# ── Step 7（可选）：H2 Rollout 生成 + 轨迹分析 ───────────────────
# 需要 agent loop 完整跑通后执行；默认跳过，设置 RUN_H2=1 启用
if [ "${RUN_H2:-0}" = "1" ]; then
  echo "▶ Step 7: H2 Rollout generation"

  H2_DOMAIN="${H2_DOMAIN:-retail}"
  H2_N_EPISODES="${H2_N_EPISODES:-100}"
  TAU2_DATA_DIR="${TAU2_DATA_DIR:-$DATA_BASE/tau2-bench-main/data/tau2}"

  for LAYER in $LAYERS; do
    S2_CKPT=$(find "$OUTPUT_BASE/sae_checkpoints/stage2" \
      -name "*-L${LAYER}-*-stage2.pt" 2>/dev/null | sort | head -1)

    # 取 top-3 gate features for trajectory tracking
    TOP_FEATS=$(python3 -c "
import json
with open('$OUTPUT_BASE/analysis/top_features_layer${LAYER}.json') as f:
    d = json.load(f)
print(' '.join(str(x['feature_idx']) for x in d[:3]))
")

    echo "  Layer $LAYER rollout generation..."
    python -m run.generate_rollouts \
      --model "$MODEL_PATH" \
      --domain "$H2_DOMAIN" \
      --tau2-data-dir "$TAU2_DATA_DIR" \
      --n-episodes "$H2_N_EPISODES" \
      --output-dir "$OUTPUT_BASE/rollouts/layer${LAYER}" \
      --layers "$LAYER" \
      --max-steps 10 \
      --seed 300 \
      --device "$DEVICE" \
      --dtype "$DTYPE"

    echo "  Layer $LAYER trajectory analysis..."
    python -m analysis.trajectory_analysis \
      --rollouts-dir "$OUTPUT_BASE/rollouts/layer${LAYER}" \
      --sae-path "$S2_CKPT" \
      --layer "$LAYER" \
      --feature-indices $TOP_FEATS \
      --output-dir "$OUTPUT_BASE/analysis" \
      --device "$DEVICE"
  done
fi

echo "================================================================"
echo "Pipeline complete. Results in $OUTPUT_BASE/analysis/"
echo "================================================================"
