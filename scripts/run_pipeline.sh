#!/bin/bash
# 两阶段 SAE 实验脚本（流式 Stage 2）

set -e

# 配置
MODEL_PATH="meta-llama/Llama-3-8B-Instruct"
OUTPUT_BASE="./outputs"
DEVICE="cuda"
LAYER=24
NUM_SAMPLES=2000
TARGET_TOKENS=50000000

echo "================================================"
echo "Agent SAE Tool-use Two-Stage Pipeline"
echo "================================================"

# Step 1: 生成 Rollouts
echo ""
echo "Step 1: Generating rollouts..."
python main.py generate-rollouts \
    --model $MODEL_PATH \
    --dataset synthetic \
    --num-samples $NUM_SAMPLES \
    --hook-layers $LAYER \
    --output-dir $OUTPUT_BASE/rollouts \
    --device $DEVICE \
    --streaming

# Step 2: Stage 1 训练
echo ""
echo "Step 2: Training SAE Stage 1..."
python -m sae.train_sae stage1 \
    --model $MODEL_PATH \
    --layer $LAYER \
    --output-dir $OUTPUT_BASE/sae_checkpoints \
    --target-tokens $TARGET_TOKENS \
    --seq-length 1024 \
    --batch-size 32 \
    --sae-batch-size 4096 \
    --learning-rate 1e-4 \
    --data-dir ./data/raw/100M \
    --decoder-norm-interval 10 \
    --device $DEVICE \
    --dtype float32

# Step 3: Stage 2 流式训练（复用 Stage 1 检查点）
echo ""
echo "Step 3: Training SAE Stage 2 (streaming)..."
python -m run.cache_activations train \
    --model $MODEL_PATH \
    --dataset synthetic \
    --num-samples $NUM_SAMPLES \
    --layer $LAYER \
    --stage1-dir $OUTPUT_BASE/sae_checkpoints/stage1 \
    --output-dir $OUTPUT_BASE/sae_checkpoints \
    --target-tokens $TARGET_TOKENS \
    --buffer-size 8192 \
    --batch-size 4096 \
    --learning-rate 5e-5 \
    --num-epochs 10 \
    --decoder-norm-interval 10 \
    --device $DEVICE \
    --dtype float32

echo ""
echo "Step 4: Locating Stage 2 checkpoints..."
python - << 'PY'
from pathlib import Path

base = Path("./outputs/sae_checkpoints/stage2")
if not base.exists():
    print("No stage2 checkpoint directory found.")
else:
    files = sorted(base.glob("*-stage2.pt"))
    if not files:
        print("No stage2 checkpoints found.")
    else:
        print("Stage2 checkpoints:")
        for p in files:
            print(f"  {p}")
PY

echo ""
echo "================================================"
echo "Pipeline complete!"
echo "Results saved to: $OUTPUT_BASE"
echo "================================================"
