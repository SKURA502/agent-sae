"""
Calculator Tool - Mathematical Calculation Tool

Provides safe mathematical expression evaluation functionality.
"""

import math
import random
from typing import Any, Dict

from ..tool_schema import ToolSchema
from .tool_utils import BaseTool


# Allowed mathematical functions
ALLOWED_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    # math module functions
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
    """Mathematical Calculation Tool"""
    
    name = "calculator"
    schema = ToolSchema(
        name="calculator",
        description="Perform mathematical calculations. Supports basic operations and common mathematical functions.",
        parameters={
            "expression": {
                "type": "string",
                "description": "Mathematical expression to calculate, e.g. '2 + 3 * 4' or 'sqrt(16)'"
            }
        },
        required=["expression"]
    )
    
    def __init__(self, precision: int = 6):
        """
        Args:
            precision: Decimal precision for results
        """
        self.precision = precision
    
    def execute(self, arguments: Dict[str, Any]) -> str:
        """Execute mathematical calculation
        
        Args:
            arguments: Contains expression field
            
        Returns:
            Calculation result string
        """
        expression = arguments.get("expression", "")
        
        if not expression:
            raise ValueError("Expression cannot be empty")
        
        # Security check
        self._validate_expression(expression)
        
        # Execute calculation
        try:
            result = eval(expression, {"__builtins__": {}}, ALLOWED_FUNCTIONS)
            
            # Format result
            if isinstance(result, float):
                if result.is_integer():
                    return str(int(result))
                return str(round(result, self.precision))
            return str(result)
            
        except ZeroDivisionError:
            raise ValueError("Division by zero error")
        except Exception as e:
            raise ValueError(f"Calculation error: {str(e)}")
    
    def _validate_expression(self, expression: str):
        """Validate expression safety"""
        # Forbidden keywords
        forbidden = [
            "import", "exec", "eval", "open", "file",
            "__", "globals", "locals", "getattr", "setattr",
            "delattr", "compile", "dir", "vars"
        ]
        
        expr_lower = expression.lower()
        for word in forbidden:
            if word in expr_lower:
                raise ValueError(f"Expression contains forbidden keyword: {word}")
        
        # Only allow specific characters
        allowed_chars = set("0123456789+-*/()., ")
        allowed_chars.update(set("abcdefghijklmnopqrstuvwxyz"))
        allowed_chars.update(set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        allowed_chars.add("_")
        
        for char in expression:
            if char not in allowed_chars:
                raise ValueError(f"Expression contains forbidden character: {char}")
    
    def generate_corrupt_result(self, arguments: Dict[str, Any]) -> str:
        """Generate corrupt calculation result"""
        expression = arguments.get("expression", "0")
        
        # Try to get correct result and modify
        try:
            correct = eval(expression, {"__builtins__": {}}, ALLOWED_FUNCTIONS)
            if isinstance(correct, (int, float)):
                # Random perturbation
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
        """Generate empty result"""
        return "Calculation result unavailable"
