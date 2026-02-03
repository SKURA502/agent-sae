"""
Lookup Tool - 键值查询工具

提供基于固定数据库的确定性键值查询功能。
"""

import random
from typing import Any, Dict, Optional

from ..tool_schema import ToolSchema
from .tool_utils import BaseTool


# 模拟数据库
MOCK_DATABASES = {
    "entities": {
        "Albert Einstein": {
            "birth_year": 1879,
            "death_year": 1955,
            "nationality": "German-American",
            "field": "Theoretical Physics",
            "known_for": "Theory of Relativity, E=mc²"
        },
        "Marie Curie": {
            "birth_year": 1867,
            "death_year": 1934,
            "nationality": "Polish-French",
            "field": "Physics, Chemistry",
            "known_for": "Radioactivity, Polonium, Radium"
        },
        "Alan Turing": {
            "birth_year": 1912,
            "death_year": 1954,
            "nationality": "British",
            "field": "Computer Science, Mathematics",
            "known_for": "Turing Machine, Enigma decryption"
        },
        "Geoffrey Hinton": {
            "birth_year": 1947,
            "nationality": "British-Canadian",
            "field": "Computer Science, Cognitive Psychology",
            "known_for": "Deep Learning, Backpropagation, Boltzmann Machines"
        },
    },
    "facts": {
        "speed_of_light": "299,792,458 m/s",
        "earth_radius": "6,371 km",
        "pi_value": "3.14159265358979",
        "avogadro_number": "6.022 × 10²³",
        "planck_constant": "6.626 × 10⁻³⁴ J·s",
    },
    "definitions": {
        "transformer": "A neural network architecture based on self-attention mechanism",
        "sparse autoencoder": "An autoencoder with sparsity constraints to learn interpretable representations",
        "mechanistic interpretability": "The study of understanding neural network behavior through analyzing internal mechanisms",
        "agent": "A system that perceives its environment and takes actions to achieve goals",
    },
}


class LookupTool(BaseTool):
    """键值查询工具"""
    
    name = "lookup"
    schema = ToolSchema(
        name="lookup",
        description="在键值数据库中查找特定条目。用于查询已知的实体信息。",
        parameters={
            "key": {
                "type": "string",
                "description": "要查找的键名"
            },
            "database": {
                "type": "string",
                "description": "数据库名称（可选，默认搜索所有数据库）"
            }
        },
        required=["key"]
    )
    
    def __init__(self, databases: Optional[Dict[str, Dict]] = None):
        """
        Args:
            databases: 数据库字典，格式为 {数据库名: {键: 值}}
        """
        self.databases = databases or MOCK_DATABASES
    
    def execute(self, arguments: Dict[str, Any]) -> str:
        """执行查询
        
        Args:
            arguments: 包含 key 和可选的 database
            
        Returns:
            查询结果的格式化字符串
        """
        key = arguments.get("key", "")
        database_name = arguments.get("database")
        
        if not key:
            raise ValueError("查询键不能为空")
        
        # 在指定数据库或所有数据库中搜索
        if database_name:
            if database_name not in self.databases:
                return f"数据库 '{database_name}' 不存在"
            result = self._search_in_database(key, self.databases[database_name])
            if result:
                return self._format_result(key, result)
        else:
            # 搜索所有数据库
            for db_name, db in self.databases.items():
                result = self._search_in_database(key, db)
                if result:
                    return self._format_result(key, result, db_name)
        
        return f"未找到键 '{key}' 的相关信息"
    
    def _search_in_database(self, key: str, database: Dict) -> Optional[Any]:
        """在单个数据库中搜索"""
        # 精确匹配
        if key in database:
            return database[key]
        
        # 不区分大小写匹配
        key_lower = key.lower()
        for k, v in database.items():
            if k.lower() == key_lower:
                return v
        
        # 部分匹配
        for k, v in database.items():
            if key_lower in k.lower() or k.lower() in key_lower:
                return v
        
        return None
    
    def _format_result(
        self, 
        key: str, 
        result: Any, 
        database_name: Optional[str] = None
    ) -> str:
        """格式化查询结果"""
        header = f"查询: {key}"
        if database_name:
            header += f" (来源: {database_name})"
        
        if isinstance(result, dict):
            items = [f"  - {k}: {v}" for k, v in result.items()]
            content = "\n".join(items)
        else:
            content = f"  {result}"
        
        return f"{header}\n{content}"
    
    def generate_corrupt_result(self, arguments: Dict[str, Any]) -> str:
        """生成错误的查询结果"""
        key = arguments.get("key", "")
        corrupt_responses = [
            f"'{key}' 的信息已过时，不建议引用。",
            f"警告：'{key}' 的数据存在争议。",
            f"'{key}' 相关信息：[数据已删除]",
            f"根据记录，'{key}' 不存在。请检查拼写。",
        ]
        return random.choice(corrupt_responses)
    
    def generate_empty_result(self) -> str:
        """生成空结果"""
        return "查询无结果"
    
    def add_entry(self, database: str, key: str, value: Any):
        """添加条目"""
        if database not in self.databases:
            self.databases[database] = {}
        self.databases[database][key] = value
    
    def get_available_databases(self) -> list:
        """获取可用数据库列表"""
        return list(self.databases.keys())
