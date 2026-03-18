"""
Run Module - 实验运行脚本

包括：
- When2Call 数据加载 + Stage 2 SAE 训练数据迭代器
- Rollout 生成（流式模式，不保存激活到磁盘）
- 激活提取（H1/H3 分析用）
- 日志记录
"""

from .when2call_adapter import (
    DecisionLabel,
    TaskSample,
    When2CallAdapter,
    create_stage2_data_iterator,
)
from .generate_rollouts import RolloutGenerator
from .cache_activations import StreamingConfig, create_streaming_data_pipeline
from .rollout_logger import RolloutLogger

__all__ = [
    # 数据
    "DecisionLabel",
    "TaskSample",
    "When2CallAdapter",
    "create_stage2_data_iterator",
    # Rollout 生成
    "RolloutGenerator",
    # 激活处理
    "StreamingConfig",
    "create_streaming_data_pipeline",
    # 日志
    "RolloutLogger",
]
