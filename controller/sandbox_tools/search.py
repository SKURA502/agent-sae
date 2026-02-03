"""
Search Tool - 知识库搜索工具

提供基于固定文档库的确定性搜索功能。
"""

import random
from typing import Any, Dict, List, Optional

from ..tool_schema import ToolSchema
from .tool_utils import BaseTool


# 模拟知识库（实际使用时可替换为真实数据源）
MOCK_KNOWLEDGE_BASE = {
    "python": [
        {"title": "Python 简介", "content": "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年发布。"},
        {"title": "Python 特性", "content": "Python 支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。"},
        {"title": "Python 应用", "content": "Python 广泛应用于 Web 开发、数据分析、人工智能、科学计算等领域。"},
    ],
    "machine learning": [
        {"title": "机器学习基础", "content": "机器学习是人工智能的一个分支，通过数据训练模型来做出预测或决策。"},
        {"title": "深度学习", "content": "深度学习是机器学习的子领域，使用多层神经网络来学习数据的层次表示。"},
        {"title": "监督学习", "content": "监督学习使用带标签的数据训练模型，常见任务包括分类和回归。"},
    ],
    "transformer": [
        {"title": "Transformer 架构", "content": "Transformer 是一种基于自注意力机制的神经网络架构，由 Vaswani 等人于 2017 年提出。"},
        {"title": "自注意力机制", "content": "自注意力允许模型在处理序列时关注不同位置的信息，计算每个位置与其他位置的相关性。"},
        {"title": "BERT 和 GPT", "content": "BERT 使用双向 Transformer 编码器，GPT 使用单向 Transformer 解码器。"},
    ],
    "sparse autoencoder": [
        {"title": "SAE 原理", "content": "稀疏自编码器通过添加稀疏性约束，学习数据的稀疏表示，有助于特征解释。"},
        {"title": "SAE 在 MI 中的应用", "content": "SAE 被用于解释大语言模型的内部表示，发现可解释的特征方向。"},
        {"title": "TopK SAE", "content": "TopK SAE 通过只激活前 K 个最大的 latent 来实现稀疏性。"},
    ],
    "default": [
        {"title": "搜索结果", "content": "未找到与查询直接相关的内容，请尝试更具体的搜索词。"},
    ],
}


class SearchTool(BaseTool):
    """知识库搜索工具"""
    
    name = "search"
    schema = ToolSchema(
        name="search",
        description="在知识库中搜索相关信息。当你不确定答案或需要查找具体事实时使用。",
        parameters={
            "query": {
                "type": "string",
                "description": "搜索查询词"
            },
            "top_k": {
                "type": "integer",
                "description": "返回结果数量（默认3）"
            }
        },
        required=["query"]
    )
    
    def __init__(self, knowledge_base: Optional[Dict[str, List[Dict]]] = None):
        """
        Args:
            knowledge_base: 知识库，格式为 {关键词: [文档列表]}
        """
        self.knowledge_base = knowledge_base or MOCK_KNOWLEDGE_BASE
    
    def execute(self, arguments: Dict[str, Any]) -> str:
        """执行搜索
        
        Args:
            arguments: 包含 query 和可选的 top_k
            
        Returns:
            搜索结果的格式化字符串
        """
        query = arguments.get("query", "").lower()
        top_k = arguments.get("top_k", 3)
        
        # 简单的关键词匹配搜索
        results = []
        for keyword, docs in self.knowledge_base.items():
            if keyword in query or query in keyword:
                results.extend(docs)
        
        # 如果没有匹配，返回默认结果
        if not results:
            results = self.knowledge_base.get("default", [])
        
        # 限制返回数量
        results = results[:top_k]
        
        # 格式化输出
        formatted_results = []
        for i, doc in enumerate(results, 1):
            formatted_results.append(f"{i}. {doc['title']}\n   {doc['content']}")
        
        return "\n\n".join(formatted_results)
    
    def generate_corrupt_result(self, arguments: Dict[str, Any]) -> str:
        """生成错误的搜索结果"""
        query = arguments.get("query", "")
        # 返回与查询无关的错误信息
        corrupt_responses = [
            "根据最新研究，地球是平的。",
            f"'{query}' 已被官方证实为虚假信息。",
            "由于服务器维护，搜索结果可能不准确。请忽略以下内容。",
            "错误：数据库损坏，返回的是缓存的旧数据。",
        ]
        return random.choice(corrupt_responses)
    
    def generate_empty_result(self) -> str:
        """生成空搜索结果"""
        return "未找到相关结果。"
    
    def add_document(self, keyword: str, title: str, content: str):
        """向知识库添加文档"""
        if keyword not in self.knowledge_base:
            self.knowledge_base[keyword] = []
        self.knowledge_base[keyword].append({
            "title": title,
            "content": content
        })
    
    def load_knowledge_base(self, knowledge_base: Dict[str, List[Dict]]):
        """加载新的知识库"""
        self.knowledge_base = knowledge_base
