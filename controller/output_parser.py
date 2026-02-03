"""
Output Parser - 解析 LLM 输出为结构化决策

支持多种输出格式：
1. JSON 格式（推荐）
2. XML/Tag 格式
3. 自然语言格式（需要额外解析）
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from .tool_schema import (
    AgentDecision,
    DecisionType,
    ToolCall,
    ToolSchema,
    get_tool_schema_by_name,
)


class ParseStatus(str, Enum):
    """解析状态"""
    SUCCESS = "success"
    FORMAT_ERROR = "format_error"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN_TOOL = "unknown_tool"


@dataclass
class ParseResult:
    """解析结果"""
    status: ParseStatus
    decision: Optional[AgentDecision]
    raw_output: str
    error_message: Optional[str] = None
    
    @property
    def is_success(self) -> bool:
        return self.status == ParseStatus.SUCCESS


class OutputParser:
    """LLM 输出解析器"""
    
    # JSON 代码块正则
    JSON_BLOCK_PATTERN = re.compile(
        r'```(?:json)?\s*\n?(.*?)\n?```',
        re.DOTALL | re.IGNORECASE
    )
    
    # 直接 JSON 对象正则
    JSON_OBJECT_PATTERN = re.compile(
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
        re.DOTALL
    )
    
    def __init__(
        self,
        available_tools: Optional[List[str]] = None,
        strict_mode: bool = True,
    ):
        """
        Args:
            available_tools: 可用工具名称列表
            strict_mode: 严格模式，要求输出必须是有效 JSON
        """
        self.available_tools = available_tools or ["search", "calculator", "lookup", "route"]
        self.strict_mode = strict_mode
    
    def parse(self, output: str) -> ParseResult:
        """解析 LLM 输出
        
        期望的 JSON 格式：
        {
            "decision": "call" | "no_call" | "abstain" | "clarify",
            "tool_call": {  // 仅当 decision == "call" 时
                "name": "tool_name",
                "arguments": {...}
            },
            "response": "..."  // 仅当 decision == "no_call" 时
        }
        """
        output = output.strip()
        
        # 尝试提取 JSON
        json_str = self._extract_json(output)
        
        if json_str is None:
            # 无法提取 JSON，尝试判断是否为直接回复
            if self.strict_mode:
                return ParseResult(
                    status=ParseStatus.FORMAT_ERROR,
                    decision=None,
                    raw_output=output,
                    error_message="无法从输出中提取有效的 JSON"
                )
            else:
                # 非严格模式：将输出视为直接回复
                return ParseResult(
                    status=ParseStatus.SUCCESS,
                    decision=AgentDecision(
                        decision_type=DecisionType.NO_CALL,
                        response=output
                    ),
                    raw_output=output
                )
        
        # 解析 JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            return ParseResult(
                status=ParseStatus.FORMAT_ERROR,
                decision=None,
                raw_output=output,
                error_message=f"JSON 解析失败: {str(e)}"
            )
        
        # 验证并构建决策
        return self._build_decision(data, output)
    
    def _extract_json(self, output: str) -> Optional[str]:
        """从输出中提取 JSON 字符串"""
        # 1. 尝试匹配 ```json ... ``` 代码块
        match = self.JSON_BLOCK_PATTERN.search(output)
        if match:
            return match.group(1).strip()
        
        # 2. 尝试匹配直接的 JSON 对象
        match = self.JSON_OBJECT_PATTERN.search(output)
        if match:
            return match.group(0)
        
        # 3. 整个输出可能就是 JSON
        if output.startswith('{') and output.endswith('}'):
            return output
        
        return None
    
    def _build_decision(self, data: Dict[str, Any], raw_output: str) -> ParseResult:
        """从解析的 JSON 构建决策对象"""
        # 获取决策类型
        decision_str = data.get("decision", "").lower()
        
        # 映射决策类型
        decision_mapping = {
            "call": DecisionType.CALL,
            "tool_call": DecisionType.CALL,
            "no_call": DecisionType.NO_CALL,
            "nocall": DecisionType.NO_CALL,
            "direct": DecisionType.NO_CALL,
            "answer": DecisionType.NO_CALL,
            "abstain": DecisionType.ABSTAIN,
            "refuse": DecisionType.ABSTAIN,
            "clarify": DecisionType.CLARIFY,
            "ask": DecisionType.CLARIFY,
        }
        
        decision_type = decision_mapping.get(decision_str)
        
        if decision_type is None:
            # 尝试从其他字段推断
            if "tool_call" in data or "tool" in data or "function_call" in data:
                decision_type = DecisionType.CALL
            elif "response" in data or "answer" in data:
                decision_type = DecisionType.NO_CALL
            else:
                return ParseResult(
                    status=ParseStatus.FORMAT_ERROR,
                    decision=None,
                    raw_output=raw_output,
                    error_message=f"无法识别的决策类型: {decision_str}"
                )
        
        # 构建决策对象
        if decision_type == DecisionType.CALL:
            return self._build_tool_call_decision(data, raw_output)
        elif decision_type == DecisionType.NO_CALL:
            response = data.get("response") or data.get("answer", "")
            return ParseResult(
                status=ParseStatus.SUCCESS,
                decision=AgentDecision(
                    decision_type=DecisionType.NO_CALL,
                    response=response
                ),
                raw_output=raw_output
            )
        else:
            return ParseResult(
                status=ParseStatus.SUCCESS,
                decision=AgentDecision(
                    decision_type=decision_type,
                    response=data.get("response", "")
                ),
                raw_output=raw_output
            )
    
    def _build_tool_call_decision(
        self, 
        data: Dict[str, Any], 
        raw_output: str
    ) -> ParseResult:
        """构建工具调用决策"""
        # 提取工具调用信息
        tool_data = data.get("tool_call") or data.get("tool") or data.get("function_call")
        
        if tool_data is None:
            # 尝试从顶层提取
            tool_name = data.get("name") or data.get("tool_name") or data.get("function")
            tool_args = data.get("arguments") or data.get("args") or data.get("parameters", {})
        else:
            tool_name = tool_data.get("name") or tool_data.get("function")
            tool_args = tool_data.get("arguments") or tool_data.get("args") or tool_data.get("parameters", {})
        
        if not tool_name:
            return ParseResult(
                status=ParseStatus.FORMAT_ERROR,
                decision=None,
                raw_output=raw_output,
                error_message="工具调用缺少工具名称"
            )
        
        # 验证工具是否可用
        if tool_name not in self.available_tools:
            return ParseResult(
                status=ParseStatus.UNKNOWN_TOOL,
                decision=None,
                raw_output=raw_output,
                error_message=f"未知工具: {tool_name}"
            )
        
        # 确保 arguments 是字典
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except json.JSONDecodeError:
                tool_args = {"input": tool_args}
        
        # 构建 ToolCall
        tool_call = ToolCall(name=tool_name, arguments=tool_args)
        
        # 验证参数
        schema = get_tool_schema_by_name(tool_name)
        if schema:
            is_valid, error_msg = tool_call.validate_against_schema(schema)
            if not is_valid:
                return ParseResult(
                    status=ParseStatus.VALIDATION_ERROR,
                    decision=None,
                    raw_output=raw_output,
                    error_message=error_msg
                )
        
        return ParseResult(
            status=ParseStatus.SUCCESS,
            decision=AgentDecision(
                decision_type=DecisionType.CALL,
                tool_call=tool_call
            ),
            raw_output=raw_output
        )
    
    @staticmethod
    def format_expected_output() -> str:
        """返回期望的输出格式说明（用于 prompt）"""
        return """请以以下 JSON 格式回复：

如果需要调用工具：
```json
{
    "decision": "call",
    "tool_call": {
        "name": "工具名称",
        "arguments": {
            "参数名": "参数值"
        }
    }
}
```

如果可以直接回答：
```json
{
    "decision": "no_call",
    "response": "你的回答"
}
```
"""
