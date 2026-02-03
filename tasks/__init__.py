"""
Tasks Module - 数据集适配器

提供统一的数据集接口，支持：
- When2Call
- BFCL
- API-Bank
- 合成数据生成
"""

from .base_adapter import BaseAdapter, TaskSample
from .when2call_adapter import When2CallAdapter
from .bfcl_adapter import BFCLAdapter
from .synthetic_generator import SyntheticGenerator

__all__ = [
    "BaseAdapter",
    "TaskSample",
    "When2CallAdapter",
    "BFCLAdapter",
    "SyntheticGenerator",
]
