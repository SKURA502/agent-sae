"""
SAE Module - Sparse Autoencoder 训练和分析

包括：
- TopK SAE 模型
- 两阶段 SAE 训练（预训练语料 + Tool-use）
- 特征提取和分析
- 流式激活处理（不保存 hidden states 到磁盘）
"""

from .sae_model import TopKSAE, SAEConfig
from .train_sae import SAETrainer, TrainingConfig, TwoStageTrainer
from .feature_extraction import FeatureExtractor
from .pretrain_data import (
    PretrainConfig,
    ActivationStreamer,
    LocalJsonlDataset,
    ActivationBuffer,
    create_pretrain_data_iterator,
)

__all__ = [
    # SAE 模型
    "TopKSAE",
    "SAEConfig",
    # 训练器
    "SAETrainer",
    "TrainingConfig",
    "TwoStageTrainer",
    # 特征提取
    "FeatureExtractor",
    # 预训练数据
    "PretrainConfig",
    "ActivationStreamer",
    "LocalJsonlDataset",
    "ActivationBuffer",
    "create_pretrain_data_iterator",
]
