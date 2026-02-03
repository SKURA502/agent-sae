#!/bin/bash
# 完整实验运行脚本

set -e

# 配置
MODEL_PATH="meta-llama/Llama-3-8B-Instruct"
OUTPUT_BASE="./outputs"
DEVICE="cuda"
NUM_SAMPLES=10000

echo "================================================"
echo "Agent SAE Tool-use Research Pipeline"
echo "================================================"

# Step 1: 生成 Rollouts
echo ""
echo "Step 1: Generating rollouts..."
python main.py generate-rollouts \
    --model $MODEL_PATH \
    --dataset synthetic \
    --num-samples $NUM_SAMPLES \
    --output-dir $OUTPUT_BASE/rollouts \
    --device $DEVICE

# Step 2: 缓存激活值
echo ""
echo "Step 2: Caching activations..."
python main.py cache-activations \
    --model $MODEL_PATH \
    --rollout-dir $OUTPUT_BASE/rollouts \
    --layers "20,24" \
    --output-dir $OUTPUT_BASE/activations \
    --device $DEVICE

# Step 3: 训练 SAE
echo ""
echo "Step 3: Training SAE..."
for LAYER in 20 24; do
    echo "Training SAE for layer $LAYER..."
    python main.py train-sae \
        --data-path $OUTPUT_BASE/activations/layer_${LAYER}_activations.pt \
        --dict-size 32768 \
        --k 128 \
        --epochs 10 \
        --output-dir $OUTPUT_BASE/sae_models/layer_$LAYER \
        --device $DEVICE \
        --wandb
done

# Step 4: 相关性分析
echo ""
echo "Step 4: Running correlation analysis..."
for LAYER in 20 24; do
    echo "Analyzing layer $LAYER..."
    python main.py analyze \
        --analysis-type correlation \
        --sae-path $OUTPUT_BASE/sae_models/layer_$LAYER/best_model.pt \
        --data-path $OUTPUT_BASE/activations/layer_${LAYER}_activations.pt \
        --layer $LAYER \
        --output-dir $OUTPUT_BASE/analysis/layer_$LAYER \
        --device $DEVICE
done

# Step 5: 线性探测
echo ""
echo "Step 5: Running linear probe..."
for LAYER in 20 24; do
    python main.py analyze \
        --analysis-type probe \
        --sae-path $OUTPUT_BASE/sae_models/layer_$LAYER/best_model.pt \
        --data-path $OUTPUT_BASE/activations/layer_${LAYER}_activations.pt \
        --layer $LAYER \
        --output-dir $OUTPUT_BASE/analysis/layer_$LAYER \
        --device $DEVICE
done

# Step 6: 可视化
echo ""
echo "Step 6: Generating visualizations..."
for LAYER in 20 24; do
    python main.py analyze \
        --analysis-type visualize \
        --sae-path $OUTPUT_BASE/sae_models/layer_$LAYER/best_model.pt \
        --data-path $OUTPUT_BASE/activations/layer_${LAYER}_activations.pt \
        --layer $LAYER \
        --output-dir $OUTPUT_BASE/analysis/layer_$LAYER \
        --device $DEVICE
done

echo ""
echo "================================================"
echo "Pipeline complete!"
echo "Results saved to: $OUTPUT_BASE"
echo "================================================"
