"""
Synthetic Generator - 合成数据生成器

生成用于训练和测试的合成任务数据。
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .base_adapter import TaskSample, DecisionLabel


@dataclass
class GeneratorConfig:
    """生成器配置"""
    num_samples: int = 1000
    call_ratio: float = 0.5  # CALL 样本的比例
    seed: int = 42
    
    # 任务类型分布
    task_types: List[str] = None
    
    def __post_init__(self):
        if self.task_types is None:
            self.task_types = [
                "factual_query",      # 事实查询（需要 search）
                "calculation",         # 计算（需要 calculator）
                "entity_lookup",       # 实体查询（需要 lookup）
                "common_knowledge",    # 常识问题（不需要工具）
                "simple_math",         # 简单数学（不需要工具）
                "greeting",            # 问候语（不需要工具）
            ]


class SyntheticGenerator:
    """合成数据生成器"""
    
    # 需要工具的模板
    TOOL_REQUIRED_TEMPLATES = {
        "factual_query": [
            ("谁是{year}年诺贝尔{field}奖的获得者？", "search", {"query": "{year}年诺贝尔{field}奖获得者"}),
            ("{entity}是什么时候成立的？", "search", {"query": "{entity} 成立时间"}),
            ("{topic}的最新研究进展是什么？", "search", {"query": "{topic} 最新研究进展"}),
            ("{person}的主要成就是什么？", "search", {"query": "{person} 主要成就"}),
            ("{concept}是由谁提出的？", "search", {"query": "{concept} 提出者"}),
        ],
        "calculation": [
            ("计算 {expr}", "calculator", {"expression": "{expr}"}),
            ("{num1} 的 {percent}% 是多少？", "calculator", {"expression": "{num1} * {percent} / 100"}),
            ("如果 {num1} 除以 {num2}，结果是多少？", "calculator", {"expression": "{num1} / {num2}"}),
            ("{num1} 的平方根是多少？", "calculator", {"expression": "sqrt({num1})"}),
            ("计算 {num1} 的 {num2} 次方", "calculator", {"expression": "pow({num1}, {num2})"}),
        ],
        "entity_lookup": [
            ("{person}的出生年份是什么？", "lookup", {"key": "{person}"}),
            ("查找关于{entity}的信息", "lookup", {"key": "{entity}"}),
            ("{term}的定义是什么？", "lookup", {"key": "{term}", "database": "definitions"}),
            ("{constant}的值是多少？", "lookup", {"key": "{constant}", "database": "facts"}),
        ],
    }
    
    # 不需要工具的模板
    NO_TOOL_TEMPLATES = {
        "common_knowledge": [
            ("什么是机器学习？", "机器学习是人工智能的一个分支，通过数据训练模型来做出预测或决策。"),
            ("Python 是什么？", "Python 是一种高级编程语言，以简洁和易读著称。"),
            ("什么是神经网络？", "神经网络是一种模仿人脑结构的计算模型，由多层人工神经元组成。"),
            ("解释一下什么是深度学习", "深度学习是机器学习的子领域，使用多层神经网络来学习数据的层次表示。"),
            ("什么是自然语言处理？", "自然语言处理是人工智能领域，专注于让计算机理解和生成人类语言。"),
        ],
        "simple_math": [
            ("2 + 2 等于多少？", "4"),
            ("10 减 7 是多少？", "3"),
            ("5 乘以 4 等于？", "20"),
            ("100 除以 10 是多少？", "10"),
            ("1 加 1 等于几？", "2"),
        ],
        "greeting": [
            ("你好", "你好！有什么可以帮助你的吗？"),
            ("早上好", "早上好！今天有什么需要我帮忙的吗？"),
            ("你是谁？", "我是一个AI助手，可以回答问题和帮助完成各种任务。"),
            ("你能做什么？", "我可以回答问题、进行计算、搜索信息等。"),
            ("谢谢", "不客气！如果还有其他问题，随时可以问我。"),
        ],
    }
    
    # 填充变量
    FILL_VALUES = {
        "year": ["2020", "2021", "2022", "2023", "2024"],
        "field": ["物理学", "化学", "生理学或医学", "文学", "和平", "经济学"],
        "entity": ["OpenAI", "Google DeepMind", "Anthropic", "Meta AI"],
        "topic": ["大语言模型", "强化学习", "计算机视觉", "自然语言处理"],
        "person": ["Albert Einstein", "Marie Curie", "Alan Turing", "Geoffrey Hinton"],
        "concept": ["Transformer", "注意力机制", "反向传播", "卷积神经网络"],
        "expr": ["15 + 27 * 3", "sqrt(256) + 100", "2 ** 10 - 24", "sin(0) + cos(0)"],
        "num1": ["100", "256", "500", "1024", "42"],
        "num2": ["2", "4", "8", "16", "3"],
        "percent": ["15", "20", "25", "50", "10"],
        "term": ["transformer", "sparse autoencoder", "mechanistic interpretability"],
        "constant": ["speed_of_light", "pi_value", "avogadro_number"],
    }
    
    # 工具 Schema
    TOOL_SCHEMAS = [
        {
            "name": "search",
            "description": "在知识库中搜索相关信息",
            "parameters": {
                "query": {"type": "string", "description": "搜索查询词"},
                "top_k": {"type": "integer", "description": "返回结果数量"}
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
        {
            "name": "lookup",
            "description": "在键值数据库中查找信息",
            "parameters": {
                "key": {"type": "string", "description": "查询键"},
                "database": {"type": "string", "description": "数据库名称"}
            },
            "required": ["key"]
        },
    ]
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        random.seed(self.config.seed)
    
    def _fill_template(self, template: str) -> str:
        """填充模板中的变量"""
        result = template
        for key, values in self.FILL_VALUES.items():
            placeholder = "{" + key + "}"
            while placeholder in result:
                result = result.replace(placeholder, random.choice(values), 1)
        return result
    
    def _generate_call_sample(self, idx: int) -> TaskSample:
        """生成需要工具调用的样本"""
        task_type = random.choice(list(self.TOOL_REQUIRED_TEMPLATES.keys()))
        template = random.choice(self.TOOL_REQUIRED_TEMPLATES[task_type])
        
        instruction = self._fill_template(template[0])
        tool_name = template[1]
        
        # 填充参数模板
        args = {}
        for key, value in template[2].items():
            args[key] = self._fill_template(value)
        
        return TaskSample(
            sample_id=f"syn_{idx:06d}",
            instruction=instruction,
            tool_schemas=self.TOOL_SCHEMAS,
            available_tools=["search", "calculator", "lookup"],
            label=DecisionLabel.CALL,
            expected_tool=tool_name,
            expected_args=args,
            source_dataset="synthetic",
            category=task_type,
        )
    
    def _generate_no_call_sample(self, idx: int) -> TaskSample:
        """生成不需要工具调用的样本"""
        task_type = random.choice(list(self.NO_TOOL_TEMPLATES.keys()))
        template = random.choice(self.NO_TOOL_TEMPLATES[task_type])
        
        return TaskSample(
            sample_id=f"syn_{idx:06d}",
            instruction=template[0],
            tool_schemas=self.TOOL_SCHEMAS,
            available_tools=["search", "calculator", "lookup"],
            label=DecisionLabel.NO_CALL,
            expected_response=template[1],
            source_dataset="synthetic",
            category=task_type,
        )
    
    def generate(self) -> List[TaskSample]:
        """生成样本"""
        samples = []
        num_call = int(self.config.num_samples * self.config.call_ratio)
        num_no_call = self.config.num_samples - num_call
        
        for i in range(num_call):
            samples.append(self._generate_call_sample(i))
        
        for i in range(num_no_call):
            samples.append(self._generate_no_call_sample(num_call + i))
        
        # 打乱顺序
        random.shuffle(samples)
        
        # 重新分配 ID
        for i, sample in enumerate(samples):
            sample.sample_id = f"syn_{i:06d}"
        
        print(f"Generated {len(samples)} synthetic samples "
              f"(CALL: {num_call}, NO_CALL: {num_no_call})")
        
        return samples
    
    def save(self, samples: List[TaskSample], output_path: str):
        """保存生成的样本"""
        import json
        from pathlib import Path
        
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
        
        print(f"Saved to {output_path}")
