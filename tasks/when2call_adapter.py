"""
When2Call Adapter - When2Call 数据集适配器

When2Call 是专门评估"何时该/不该调用工具"的数据集。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_adapter import BaseAdapter, TaskSample, DecisionLabel


class When2CallAdapter(BaseAdapter):
    """When2Call 数据集适配器"""
    
    @property
    def name(self) -> str:
        return "When2Call"
    
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """加载原始数据"""
        data_file = self.data_path / f"{self.split}.jsonl"
        
        if not data_file.exists():
            # 尝试其他格式
            for suffix in [".json", ".jsonl", ""]:
                alt_file = self.data_path / f"{self.split}{suffix}"
                if alt_file.exists():
                    data_file = alt_file
                    break
            else:
                # 如果没有 split 文件，尝试读取整个目录
                data_file = self.data_path / "data.jsonl"
                if not data_file.exists():
                    raise FileNotFoundError(
                        f"Cannot find data file in {self.data_path}"
                    )
        
        samples = []
        
        if data_file.suffix == ".jsonl":
            with open(data_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
        else:
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples = data
                else:
                    samples = data.get("data", data.get("samples", [data]))
        
        return samples
    
    def _convert_sample(self, raw_sample: Dict[str, Any], idx: int) -> TaskSample:
        """转换 When2Call 样本"""
        # 提取基本信息
        instruction = raw_sample.get("instruction", raw_sample.get("query", ""))
        context = raw_sample.get("context", raw_sample.get("input", None))
        
        # 提取工具信息
        tools = raw_sample.get("tools", raw_sample.get("functions", []))
        tool_schemas = []
        available_tools = []
        
        for tool in tools:
            if isinstance(tool, dict):
                tool_schemas.append(tool)
                tool_name = tool.get("name", tool.get("function", {}).get("name", ""))
                if tool_name:
                    available_tools.append(tool_name)
        
        # 提取标签
        should_call = raw_sample.get("should_call", raw_sample.get("label", None))
        
        if should_call is True or should_call == "call" or should_call == 1:
            label = DecisionLabel.CALL
        elif should_call is False or should_call == "no_call" or should_call == 0:
            label = DecisionLabel.NO_CALL
        else:
            label = DecisionLabel.UNCERTAIN
        
        # 提取期望的工具调用
        expected_tool = None
        expected_args = None
        expected_response = None
        
        if label == DecisionLabel.CALL:
            ground_truth = raw_sample.get("ground_truth", raw_sample.get("expected", {}))
            if isinstance(ground_truth, dict):
                expected_tool = ground_truth.get("tool", ground_truth.get("name"))
                expected_args = ground_truth.get("arguments", ground_truth.get("args"))
        elif label == DecisionLabel.NO_CALL:
            expected_response = raw_sample.get("expected_response", raw_sample.get("answer"))
        
        # 提取元数据
        category = raw_sample.get("category", raw_sample.get("type"))
        difficulty = raw_sample.get("difficulty", raw_sample.get("level"))
        
        return TaskSample(
            sample_id=raw_sample.get("id", f"w2c_{idx:06d}"),
            instruction=instruction,
            context=context,
            tool_schemas=tool_schemas,
            available_tools=available_tools,
            label=label,
            expected_tool=expected_tool,
            expected_args=expected_args,
            expected_response=expected_response,
            source_dataset="when2call",
            difficulty=difficulty,
            category=category,
            metadata={
                "original_id": raw_sample.get("id"),
                "split": self.split,
            },
        )
    
    @staticmethod
    def create_mock_data(output_path: Path, num_samples: int = 100):
        """创建模拟数据（用于测试）"""
        import random
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 模拟的工具定义
        tools = [
            {
                "name": "search",
                "description": "在知识库中搜索信息",
                "parameters": {
                    "query": {"type": "string", "description": "搜索查询"}
                },
                "required": ["query"]
            },
            {
                "name": "calculator",
                "description": "执行数学计算",
                "parameters": {
                    "expression": {"type": "string", "description": "数学表达式"}
                },
                "required": ["expression"]
            },
        ]
        
        # 需要调用工具的问题
        call_templates = [
            ("2023年诺贝尔物理学奖获得者是谁？", "search", {"query": "2023诺贝尔物理学奖"}),
            ("计算 sqrt(144) + 25 * 3", "calculator", {"expression": "sqrt(144) + 25 * 3"}),
            ("最新的GPT-5发布日期是什么时候？", "search", {"query": "GPT-5 发布日期"}),
            ("15% of 250 是多少？", "calculator", {"expression": "0.15 * 250"}),
        ]
        
        # 不需要调用工具的问题
        no_call_templates = [
            ("什么是机器学习？", "机器学习是人工智能的一个分支..."),
            ("Python 是什么编程语言？", "Python 是一种高级编程语言..."),
            ("2 + 2 等于多少？", "4"),
            ("说一句问候语", "你好！有什么可以帮助你的吗？"),
        ]
        
        samples = []
        
        for i in range(num_samples):
            if random.random() < 0.5:
                # 生成需要工具的样本
                template = random.choice(call_templates)
                samples.append({
                    "id": f"mock_{i:06d}",
                    "instruction": template[0],
                    "tools": tools,
                    "should_call": True,
                    "ground_truth": {
                        "tool": template[1],
                        "arguments": template[2]
                    },
                    "category": "tool_required",
                })
            else:
                # 生成不需要工具的样本
                template = random.choice(no_call_templates)
                samples.append({
                    "id": f"mock_{i:06d}",
                    "instruction": template[0],
                    "tools": tools,
                    "should_call": False,
                    "expected_response": template[1],
                    "category": "direct_answer",
                })
        
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        print(f"Created {num_samples} mock samples at {output_path}")
