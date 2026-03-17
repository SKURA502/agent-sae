"""
Tasks Module - 数据集适配器

提供统一的数据集接口，支持：
- When2Call（pref / MCQ 格式）
- BFCL（v4 扁平格式及旧版格式）
- 合成数据生成
"""

from .base_adapter import BaseAdapter, TaskSample, DecisionLabel
from .when2call_adapter import When2CallAdapter
from .bfcl_adapter import BFCLAdapter
from .synthetic_generator import SyntheticGenerator

__all__ = [
    "BaseAdapter",
    "TaskSample",
    "DecisionLabel",
    "When2CallAdapter",
    "BFCLAdapter",
    "SyntheticGenerator",
]
