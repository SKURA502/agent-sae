"""
Tool Utilities - 工具执行器和噪声注入

提供统一的工具执行接口和噪声配置。
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from ..tool_schema import ToolCall, ToolResult, ToolSchema


@dataclass
class NoiseConfig:
    """噪声注入配置"""
    p_fail: float = 0.0      # 工具执行失败概率
    p_empty: float = 0.0     # 返回空结果概率
    p_corrupt: float = 0.0   # 返回错误结果概率
    seed: Optional[int] = None
    
    def __post_init__(self):
        if self.seed is not None:
            random.seed(self.seed)
        
        # 确保概率和不超过1
        total = self.p_fail + self.p_empty + self.p_corrupt
        if total > 1.0:
            raise ValueError(f"噪声概率之和不能超过1.0, 当前为: {total}")


class BaseTool(ABC):
    """工具基类"""
    
    name: str
    schema: ToolSchema
    
    @abstractmethod
    def execute(self, arguments: Dict[str, Any]) -> Any:
        """执行工具
        
        Args:
            arguments: 工具参数
            
        Returns:
            执行结果
        """
        pass
    
    @abstractmethod
    def generate_corrupt_result(self, arguments: Dict[str, Any]) -> Any:
        """生成错误结果（用于噪声注入）"""
        pass
    
    @abstractmethod
    def generate_empty_result(self) -> Any:
        """生成空结果"""
        pass


class ToolExecutor:
    """工具执行器 - 统一管理工具执行和噪声注入"""
    
    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None,
        noise_config: Optional[NoiseConfig] = None,
    ):
        """
        Args:
            tools: 可用工具列表
            noise_config: 噪声配置
        """
        self.tools: Dict[str, BaseTool] = {}
        self.noise_config = noise_config or NoiseConfig()
        self.execution_log: List[Dict[str, Any]] = []
        
        if tools:
            for tool in tools:
                self.register_tool(tool)
    
    def register_tool(self, tool: BaseTool):
        """注册工具"""
        self.tools[tool.name] = tool
    
    def get_available_tools(self) -> List[str]:
        """获取可用工具名称列表"""
        return list(self.tools.keys())
    
    def get_tool_schemas(self) -> List[ToolSchema]:
        """获取所有工具的 Schema"""
        return [tool.schema for tool in self.tools.values()]
    
    def execute(self, tool_call: ToolCall) -> ToolResult:
        """执行工具调用
        
        Args:
            tool_call: 工具调用请求
            
        Returns:
            工具执行结果
        """
        tool_name = tool_call.name
        arguments = tool_call.arguments
        
        # 检查工具是否存在
        if tool_name not in self.tools:
            result = ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"未知工具: {tool_name}"
            )
            self._log_execution(tool_call, result, "unknown_tool")
            return result
        
        tool = self.tools[tool_name]
        
        # 应用噪声
        noise_type = self._apply_noise()
        
        if noise_type == "fail":
            result = ToolResult(
                tool_name=tool_name,
                success=False,
                error="工具执行失败（模拟错误）"
            )
            self._log_execution(tool_call, result, "noise_fail")
            return result
        
        if noise_type == "empty":
            result = ToolResult(
                tool_name=tool_name,
                success=True,
                result=tool.generate_empty_result()
            )
            self._log_execution(tool_call, result, "noise_empty")
            return result
        
        if noise_type == "corrupt":
            result = ToolResult(
                tool_name=tool_name,
                success=True,
                result=tool.generate_corrupt_result(arguments)
            )
            self._log_execution(tool_call, result, "noise_corrupt")
            return result
        
        # 正常执行
        try:
            exec_result = tool.execute(arguments)
            result = ToolResult(
                tool_name=tool_name,
                success=True,
                result=exec_result
            )
            self._log_execution(tool_call, result, "success")
            return result
        except Exception as e:
            result = ToolResult(
                tool_name=tool_name,
                success=False,
                error=str(e)
            )
            self._log_execution(tool_call, result, "execution_error")
            return result
    
    def _apply_noise(self) -> Optional[str]:
        """根据噪声配置决定是否注入噪声
        
        Returns:
            噪声类型: "fail", "empty", "corrupt", 或 None（正常执行）
        """
        rand = random.random()
        
        if rand < self.noise_config.p_fail:
            return "fail"
        
        rand -= self.noise_config.p_fail
        if rand < self.noise_config.p_empty:
            return "empty"
        
        rand -= self.noise_config.p_empty
        if rand < self.noise_config.p_corrupt:
            return "corrupt"
        
        return None
    
    def _log_execution(
        self, 
        tool_call: ToolCall, 
        result: ToolResult, 
        status: str
    ):
        """记录工具执行日志"""
        self.execution_log.append({
            "tool_name": tool_call.name,
            "arguments": tool_call.arguments,
            "success": result.success,
            "result": result.result if result.success else None,
            "error": result.error,
            "status": status,
        })
    
    def get_execution_log(self) -> List[Dict[str, Any]]:
        """获取执行日志"""
        return self.execution_log
    
    def clear_log(self):
        """清空执行日志"""
        self.execution_log = []
    
    def set_noise_config(self, noise_config: NoiseConfig):
        """更新噪声配置"""
        self.noise_config = noise_config
