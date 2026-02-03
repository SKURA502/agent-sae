"""
BFCL Adapter - Berkeley Function Calling Leaderboard 数据集适配器

BFCL 是函数调用基准，覆盖串行/并行/多语言等类型。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_adapter import BaseAdapter, TaskSample, DecisionLabel


class BFCLAdapter(BaseAdapter):
    """BFCL 数据集适配器"""
    
    @property
    def name(self) -> str:
        return "BFCL"
    
    def __init__(
        self,
        data_path: str,
        split: str = "test",
        num_samples: int = -1,
        seed: int = 42,
        categories: Optional[List[str]] = None,
    ):
        """
        Args:
            data_path: 数据集路径
            split: 数据集划分
            num_samples: 采样数量
            seed: 随机种子
            categories: 要加载的类别（如 simple, parallel, multiple 等）
        """
        super().__init__(data_path, split, num_samples, seed)
        self.categories = categories or ["simple", "parallel", "multiple"]
    
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """加载 BFCL 数据"""
        samples = []
        
        # BFCL 通常按类别组织
        for category in self.categories:
            category_path = self.data_path / category
            
            if category_path.exists():
                for file_path in category_path.glob("*.json"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        try:
                            data = json.load(f)
                            if isinstance(data, list):
                                for item in data:
                                    item["_category"] = category
                                    samples.append(item)
                            else:
                                data["_category"] = category
                                samples.append(data)
                        except json.JSONDecodeError:
                            continue
        
        # 如果按类别加载没有找到，尝试直接读取
        if not samples:
            for file_path in self.data_path.glob("*.json"):
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            samples.extend(data)
                        else:
                            samples.append(data)
                    except json.JSONDecodeError:
                        continue
        
        return samples
    
    def _convert_sample(self, raw_sample: Dict[str, Any], idx: int) -> TaskSample:
        """转换 BFCL 样本"""
        # 提取用户查询
        instruction = raw_sample.get("user_query", raw_sample.get("question", ""))
        context = raw_sample.get("context", None)
        
        # 提取函数定义
        functions = raw_sample.get("functions", raw_sample.get("function", []))
        if not isinstance(functions, list):
            functions = [functions]
        
        tool_schemas = []
        available_tools = []
        
        for func in functions:
            if isinstance(func, dict):
                tool_schemas.append(func)
                name = func.get("name", "")
                if name:
                    available_tools.append(name)
        
        # BFCL 样本通常都需要函数调用
        label = DecisionLabel.CALL
        
        # 提取 ground truth
        ground_truth = raw_sample.get("ground_truth", raw_sample.get("expected_output", []))
        if not isinstance(ground_truth, list):
            ground_truth = [ground_truth]
        
        expected_tool = None
        expected_args = None
        
        if ground_truth:
            first_call = ground_truth[0]
            if isinstance(first_call, dict):
                expected_tool = first_call.get("name", first_call.get("function"))
                expected_args = first_call.get("arguments", first_call.get("parameters"))
            elif isinstance(first_call, str):
                # 可能是函数调用的字符串表示
                expected_tool = first_call
        
        category = raw_sample.get("_category", raw_sample.get("category", "unknown"))
        
        return TaskSample(
            sample_id=raw_sample.get("id", f"bfcl_{idx:06d}"),
            instruction=instruction,
            context=context,
            tool_schemas=tool_schemas,
            available_tools=available_tools,
            label=label,
            expected_tool=expected_tool,
            expected_args=expected_args,
            source_dataset="bfcl",
            category=category,
            metadata={
                "ground_truth_full": ground_truth,
                "num_functions": len(functions),
            },
        )
