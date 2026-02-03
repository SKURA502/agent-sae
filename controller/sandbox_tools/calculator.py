"""
Calculator Tool - 数学计算工具

提供安全的数学表达式求值功能。
"""

import math
import random
from typing import Any, Dict

from ..tool_schema import ToolSchema
from .tool_utils import BaseTool


# 允许的数学函数
ALLOWED_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    # math 模块函数
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": math.factorial,
    "pi": math.pi,
    "e": math.e,
}


class CalculatorTool(BaseTool):
    """数学计算工具"""
    
    name = "calculator"
    schema = ToolSchema(
        name="calculator",
        description="执行数学计算。支持基本运算和常见数学函数。",
        parameters={
            "expression": {
                "type": "string",
                "description": "要计算的数学表达式，如 '2 + 3 * 4' 或 'sqrt(16)'"
            }
        },
        required=["expression"]
    )
    
    def __init__(self, precision: int = 6):
        """
        Args:
            precision: 结果小数精度
        """
        self.precision = precision
    
    def execute(self, arguments: Dict[str, Any]) -> str:
        """执行数学计算
        
        Args:
            arguments: 包含 expression 字段
            
        Returns:
            计算结果字符串
        """
        expression = arguments.get("expression", "")
        
        if not expression:
            raise ValueError("表达式不能为空")
        
        # 安全性检查
        self._validate_expression(expression)
        
        # 执行计算
        try:
            result = eval(expression, {"__builtins__": {}}, ALLOWED_FUNCTIONS)
            
            # 格式化结果
            if isinstance(result, float):
                if result.is_integer():
                    return str(int(result))
                return str(round(result, self.precision))
            return str(result)
            
        except ZeroDivisionError:
            raise ValueError("除零错误")
        except Exception as e:
            raise ValueError(f"计算错误: {str(e)}")
    
    def _validate_expression(self, expression: str):
        """验证表达式安全性"""
        # 禁止的关键词
        forbidden = [
            "import", "exec", "eval", "open", "file",
            "__", "globals", "locals", "getattr", "setattr",
            "delattr", "compile", "dir", "vars"
        ]
        
        expr_lower = expression.lower()
        for word in forbidden:
            if word in expr_lower:
                raise ValueError(f"表达式包含不允许的关键词: {word}")
        
        # 只允许特定字符
        allowed_chars = set("0123456789+-*/()., ")
        allowed_chars.update(set("abcdefghijklmnopqrstuvwxyz"))
        allowed_chars.update(set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        allowed_chars.add("_")
        
        for char in expression:
            if char not in allowed_chars:
                raise ValueError(f"表达式包含不允许的字符: {char}")
    
    def generate_corrupt_result(self, arguments: Dict[str, Any]) -> str:
        """生成错误的计算结果"""
        expression = arguments.get("expression", "0")
        
        # 尝试获取正确结果并修改
        try:
            correct = eval(expression, {"__builtins__": {}}, ALLOWED_FUNCTIONS)
            if isinstance(correct, (int, float)):
                # 随机扰动
                corrupt_methods = [
                    lambda x: x * 2,
                    lambda x: x + random.randint(1, 100),
                    lambda x: x - random.randint(1, 100),
                    lambda x: -x,
                    lambda x: x / 2,
                ]
                corrupt = random.choice(corrupt_methods)(correct)
                return str(round(corrupt, self.precision) if isinstance(corrupt, float) else corrupt)
        except:
            pass
        
        return str(random.randint(-1000, 1000))
    
    def generate_empty_result(self) -> str:
        """生成空结果"""
        return "计算结果不可用"
