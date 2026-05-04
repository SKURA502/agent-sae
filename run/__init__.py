"""
Run Module - 实验运行脚本

包括：
- When2Call 数据加载 + Stage 2 SAE 训练数据迭代器
- Rollout 生成（流式模式，不保存激活到磁盘）
- 激活提取（H1/H3 分析用）
- 日志记录
"""

from utils.when2call_adapter import (
    DecisionLabel,
    TaskSample,
    When2CallAdapter,
    create_stage2_data_iterator,
)
from .cache_activations import create_streaming_data_pipeline

__all__ = [
    # 数据
    "DecisionLabel",
    "TaskSample",
    "When2CallAdapter",
    "create_stage2_data_iterator",
    # 激活处理
    "create_streaming_data_pipeline",
]
