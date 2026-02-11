"""
Run Module - 实验运行脚本

包括：
- Rollout 生成（流式模式，不保存激活到磁盘）
- 流式激活处理
- 日志记录
"""

from .generate_rollouts import (
    RolloutGenerator,
    ToolUseActivationCollector,
)
from .cache_activations import (
    StreamingConfig,
    ActivationBuffer,
    create_streaming_data_pipeline,
    StreamingActivationDataset,
    train_sae_streaming,
)
from .rollout_logger import RolloutLogger

__all__ = [
    # Rollout 生成
    "RolloutGenerator",
    "ToolUseActivationCollector",
    # 流式激活处理
    "StreamingConfig",
    "ActivationBuffer",
    "create_streaming_data_pipeline",
    "StreamingActivationDataset",
    "train_sae_streaming",
    # 日志
    "RolloutLogger",
]
