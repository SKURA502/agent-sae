"""
Base Adapter - 数据集适配器基类

定义统一的数据集接口。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union


class DecisionLabel(str, Enum):
    """决策标签"""
    CALL = "call"           # 应该调用工具
    NO_CALL = "no_call"     # 不应该调用工具
    UNCERTAIN = "uncertain" # 不确定（可能都行）


@dataclass
class TaskSample:
    """统一的任务样本格式"""
    # 基本信息
    sample_id: str
    instruction: str
    context: Optional[str] = None
    
    # 工具信息
    tool_schemas: List[Dict[str, Any]] = field(default_factory=list)
    available_tools: List[str] = field(default_factory=list)
    
    # 标签
    label: DecisionLabel = DecisionLabel.UNCERTAIN
    expected_tool: Optional[str] = None
    expected_args: Optional[Dict[str, Any]] = None
    expected_response: Optional[str] = None
    
    # 元数据
    source_dataset: str = ""
    difficulty: Optional[str] = None
    category: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "sample_id": self.sample_id,
            "instruction": self.instruction,
            "context": self.context,
            "tool_schemas": self.tool_schemas,
            "available_tools": self.available_tools,
            "label": self.label.value,
            "expected_tool": self.expected_tool,
            "expected_args": self.expected_args,
            "expected_response": self.expected_response,
            "source_dataset": self.source_dataset,
            "difficulty": self.difficulty,
            "category": self.category,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskSample":
        """从字典创建"""
        data = data.copy()
        if "label" in data and isinstance(data["label"], str):
            data["label"] = DecisionLabel(data["label"])
        return cls(**data)


class BaseAdapter(ABC):
    """数据集适配器基类"""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        split: str = "test",
        num_samples: int = -1,
        seed: int = 42,
    ):
        """
        Args:
            data_path: 数据集路径
            split: 数据集划分 (train/val/test)
            num_samples: 采样数量，-1 表示全部
            seed: 随机种子
        """
        self.data_path = Path(data_path)
        self.split = split
        self.num_samples = num_samples
        self.seed = seed
        
        self._samples: List[TaskSample] = []
        self._loaded = False
    
    @property
    @abstractmethod
    def name(self) -> str:
        """数据集名称"""
        pass
    
    @abstractmethod
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """加载原始数据"""
        pass
    
    @abstractmethod
    def _convert_sample(self, raw_sample: Dict[str, Any], idx: int) -> TaskSample:
        """将原始样本转换为统一格式"""
        pass
    
    def load(self) -> "BaseAdapter":
        """加载数据集"""
        if self._loaded:
            return self
        
        print(f"Loading {self.name} from {self.data_path}")
        
        raw_data = self._load_raw_data()
        
        # 采样
        if self.num_samples > 0 and self.num_samples < len(raw_data):
            import random
            random.seed(self.seed)
            raw_data = random.sample(raw_data, self.num_samples)
        
        # 转换
        self._samples = [
            self._convert_sample(sample, idx) 
            for idx, sample in enumerate(raw_data)
        ]
        
        self._loaded = True
        print(f"Loaded {len(self._samples)} samples")
        
        return self
    
    def __len__(self) -> int:
        if not self._loaded:
            self.load()
        return len(self._samples)
    
    def __getitem__(self, idx: int) -> TaskSample:
        if not self._loaded:
            self.load()
        return self._samples[idx]
    
    def __iter__(self) -> Iterator[TaskSample]:
        if not self._loaded:
            self.load()
        return iter(self._samples)
    
    def get_samples(
        self,
        label: Optional[DecisionLabel] = None,
        category: Optional[str] = None,
    ) -> List[TaskSample]:
        """获取符合条件的样本"""
        if not self._loaded:
            self.load()
        
        samples = self._samples
        
        if label is not None:
            samples = [s for s in samples if s.label == label]
        
        if category is not None:
            samples = [s for s in samples if s.category == category]
        
        return samples
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        if not self._loaded:
            self.load()
        
        label_counts = {}
        category_counts = {}
        tool_counts = {}
        
        for sample in self._samples:
            # Label 统计
            label_key = sample.label.value
            label_counts[label_key] = label_counts.get(label_key, 0) + 1
            
            # Category 统计
            if sample.category:
                category_counts[sample.category] = category_counts.get(sample.category, 0) + 1
            
            # Tool 统计
            if sample.expected_tool:
                tool_counts[sample.expected_tool] = tool_counts.get(sample.expected_tool, 0) + 1
        
        return {
            "name": self.name,
            "total_samples": len(self._samples),
            "label_distribution": label_counts,
            "category_distribution": category_counts,
            "tool_distribution": tool_counts,
        }
    
    def save_processed(self, output_path: Union[str, Path]):
        """保存处理后的数据"""
        import json
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in self._samples:
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
        
        print(f"Saved {len(self._samples)} samples to {output_path}")
