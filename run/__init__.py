"""
Run Module - 实验运行脚本

包括：
- Rollout 生成（流式模式，不保存激活到磁盘）
- 流式激活处理
- 日志记录
"""

from .generate_rollouts import RolloutGenerator
from .cache_activations import (
    StreamingConfig,
    create_streaming_data_pipeline,
)
from .rollout_logger import RolloutLogger

__all__ = [
    # Rollout 生成
    "RolloutGenerator",
    # 流式激活处理
    "StreamingConfig",
    "create_streaming_data_pipeline",
    # 日志
    "RolloutLogger",
]
