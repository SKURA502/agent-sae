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

# Step 3: 训练 SAE（两阶段）
echo ""
echo "Step 3: Training SAE (Stage 1 - pretrain corpus)..."
python -m sae.train_sae stage1 \
    --model $MODEL_PATH \
    --layers 20 24 \
    --output-dir $OUTPUT_BASE/sae_models \
    --target-tokens 50000000 \
    --data-dir data/raw \
    --decoder-norm-interval 10 \
    --device $DEVICE

echo "Step 3b: Training SAE (Stage 2 - tool-use data)..."
python -m sae.train_sae stage2 \
    --model $MODEL_PATH \
    --stage1-dir $OUTPUT_BASE/sae_models/stage1 \
    --layers 20 24 \
    --output-dir $OUTPUT_BASE/sae_models \
    --learning-rate 5e-5 \
    --num-epochs 10 \
    --decoder-norm-interval 10 \
    --device $DEVICE

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
