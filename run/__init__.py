"""
Run Module - 实验运行脚本

包括：
- Rollout 生成（流式模式，不保存激活到磁盘）
- 流式激活处理（Stage 2 训练 + 激活提取）
- 日志记录
"""

from .generate_rollouts import RolloutGenerator
from .cache_activations import (
    StreamingConfig,
    create_streaming_data_pipeline,
    create_stage2_data_generator,
)
from .rollout_logger import RolloutLogger

__all__ = [
    # Rollout 生成
    "RolloutGenerator",
    # 激活处理
    "StreamingConfig",
    "create_streaming_data_pipeline",
    "create_stage2_data_generator",
    # 日志
    "RolloutLogger",
]
