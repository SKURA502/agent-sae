"""
Tool Schema Definitions using Pydantic

定义工具调用的数据结构，包括：
- ToolSchema: 工具的 JSON Schema 定义
- ToolCall: 模型输出的工具调用请求
- ToolResult: 工具执行结果
- AgentDecision: Agent 的决策类型
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class DecisionType(str, Enum):
    """Agent 决策类型"""
    CALL = "call"           # 调用工具
    NO_CALL = "no_call"     # 不调用工具，直接回答
    ABSTAIN = "abstain"     # 拒绝回答
    CLARIFY = "clarify"     # 请求澄清


class ToolSchema(BaseModel):
    """工具的 JSON Schema 定义"""
    name: str = Field(..., description="工具名称")
    description: str = Field(..., description="工具功能描述")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="参数的 JSON Schema"
    )
    required: List[str] = Field(
        default_factory=list,
        description="必需参数列表"
    )
    
    def to_openai_format(self) -> Dict[str, Any]:
        """转换为 OpenAI function calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                }
            }
        }
    
    def to_prompt_format(self) -> str:
        """转换为 prompt 中的工具描述格式"""
        params_desc = []
        for param_name, param_info in self.parameters.items():
            param_type = param_info.get("type", "any")
            param_desc = param_info.get("description", "")
            required_mark = "*" if param_name in self.required else ""
            params_desc.append(f"  - {param_name}{required_mark} ({param_type}): {param_desc}")
        
        params_str = "\n".join(params_desc) if params_desc else "  (无参数)"
        return f"**{self.name}**: {self.description}\n参数:\n{params_str}"


class ToolCall(BaseModel):
    """模型输出的工具调用请求"""
    name: str = Field(..., description="要调用的工具名称")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="工具调用参数"
    )
    
    def validate_against_schema(self, schema: ToolSchema) -> tuple[bool, Optional[str]]:
        """验证工具调用是否符合 schema"""
        # 检查必需参数
        for required_param in schema.required:
            if required_param not in self.arguments:
                return False, f"缺少必需参数: {required_param}"
        
        # 检查参数类型（简化版本）
        for param_name, param_value in self.arguments.items():
            if param_name in schema.parameters:
                expected_type = schema.parameters[param_name].get("type")
                if expected_type and not self._check_type(param_value, expected_type):
                    return False, f"参数 {param_name} 类型错误，期望 {expected_type}"
        
        return True, None
    
    @staticmethod
    def _check_type(value: Any, expected_type: str) -> bool:
        """简化的类型检查"""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type is None:
            return True  # 未知类型，跳过检查
        return isinstance(value, expected_python_type)


class ToolResult(BaseModel):
    """工具执行结果"""
    tool_name: str = Field(..., description="工具名称")
    success: bool = Field(..., description="是否执行成功")
    result: Any = Field(None, description="执行结果")
    error: Optional[str] = Field(None, description="错误信息")
    
    def to_observation(self) -> str:
        """转换为 Agent 可读的 observation 格式"""
        if self.success:
            return f"[Tool Result - {self.tool_name}]\n{self.result}"
        else:
            return f"[Tool Error - {self.tool_name}]\n{self.error}"


class AgentDecision(BaseModel):
    """Agent 的决策"""
    decision_type: DecisionType = Field(..., description="决策类型")
    tool_call: Optional[ToolCall] = Field(None, description="工具调用（如果是 CALL）")
    response: Optional[str] = Field(None, description="直接回复（如果是 NO_CALL）")
    confidence: Optional[float] = Field(None, description="置信度（可选）")
    
    @property
    def is_tool_call(self) -> bool:
        return self.decision_type == DecisionType.CALL


# ============== 预定义工具 Schema ==============

SEARCH_SCHEMA = ToolSchema(
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

CALCULATOR_SCHEMA = ToolSchema(
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

LOOKUP_SCHEMA = ToolSchema(
    name="lookup",
    description="在键值数据库中查找特定条目。用于查询已知的实体信息。",
    parameters={
        "key": {
            "type": "string",
            "description": "要查找的键名"
        },
        "database": {
            "type": "string",
            "description": "数据库名称（可选）"
        }
    },
    required=["key"]
)

ROUTE_SCHEMA = ToolSchema(
    name="route",
    description="计算两地之间的路线和时间。",
    parameters={
        "origin": {
            "type": "string",
            "description": "起点位置"
        },
        "destination": {
            "type": "string",
            "description": "终点位置"
        },
        "mode": {
            "type": "string",
            "description": "出行方式: driving, walking, transit"
        }
    },
    required=["origin", "destination"]
)


def get_tool_schemas(tool_names: Optional[List[str]] = None) -> List[ToolSchema]:
    """获取工具 Schema 列表
    
    Args:
        tool_names: 要获取的工具名称列表，None 表示获取所有工具
        
    Returns:
        ToolSchema 列表
    """
    all_schemas = {
        "search": SEARCH_SCHEMA,
        "calculator": CALCULATOR_SCHEMA,
        "lookup": LOOKUP_SCHEMA,
        "route": ROUTE_SCHEMA,
    }
    
    if tool_names is None:
        return list(all_schemas.values())
    
    return [all_schemas[name] for name in tool_names if name in all_schemas]


def get_tool_schema_by_name(name: str) -> Optional[ToolSchema]:
    """根据名称获取单个工具 Schema"""
    schemas = get_tool_schemas([name])
    return schemas[0] if schemas else None
