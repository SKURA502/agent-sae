"""
Tool Utilities - Tool Executor and Noise Injection

Provides unified tool execution interface and noise configuration.
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from ..tool_schema import ToolCall, ToolResult, ToolSchema


@dataclass
class NoiseConfig:
    """Noise injection configuration"""
    p_fail: float = 0.0      # Tool execution failure probability
    p_empty: float = 0.0     # Return empty result probability
    p_corrupt: float = 0.0   # Return corrupt result probability
    seed: Optional[int] = None
    
    def __post_init__(self):
        if self.seed is not None:
            random.seed(self.seed)
        
        # Ensure probability sum does not exceed 1
        total = self.p_fail + self.p_empty + self.p_corrupt
        if total > 1.0:
            raise ValueError(f"Sum of noise probabilities cannot exceed 1.0, current: {total}")


class BaseTool(ABC):
    """Base Tool Class"""
    
    name: str
    schema: ToolSchema
    
    @abstractmethod
    def execute(self, arguments: Dict[str, Any]) -> Any:
        """Execute tool
        
        Args:
            arguments: Tool parameters
            
        Returns:
            Execution result
        """
        pass
    
    @abstractmethod
    def generate_corrupt_result(self, arguments: Dict[str, Any]) -> Any:
        """Generate corrupt result (for noise injection)"""
        pass
    
    @abstractmethod
    def generate_empty_result(self) -> Any:
        """Generate empty result"""
        pass


class ToolExecutor:
    """Tool Executor - Unified management of tool execution and noise injection"""
    
    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None,
        noise_config: Optional[NoiseConfig] = None,
    ):
        """
        Args:
            tools: List of available tools
            noise_config: Noise configuration
        """
        self.tools: Dict[str, BaseTool] = {}
        self.noise_config = noise_config or NoiseConfig()
        self.execution_log: List[Dict[str, Any]] = []
        
        if tools:
            for tool in tools:
                self.register_tool(tool)
    
    def register_tool(self, tool: BaseTool):
        """Register tool"""
        self.tools[tool.name] = tool
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names"""
        return list(self.tools.keys())
    
    def get_tool_schemas(self) -> List[ToolSchema]:
        """Get all tool schemas"""
        return [tool.schema for tool in self.tools.values()]
    
    def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute tool call
        
        Args:
            tool_call: Tool call request
            
        Returns:
            Tool execution result
        """
        tool_name = tool_call.name
        arguments = tool_call.arguments
        
        # Check if tool exists
        if tool_name not in self.tools:
            result = ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Unknown tool: {tool_name}"
            )
            self._log_execution(tool_call, result, "unknown_tool")
            return result
        
        tool = self.tools[tool_name]
        
        # Apply noise
        noise_type = self._apply_noise()
        
        if noise_type == "fail":
            result = ToolResult(
                tool_name=tool_name,
                success=False,
                error="Tool execution failed (simulated error)"
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
        
        # Normal execution
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
        """Decide whether to inject noise based on noise config
        
        Returns:
            Noise type: "fail", "empty", "corrupt", or None (normal execution)
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
        """Log tool execution"""
        self.execution_log.append({
            "tool_name": tool_call.name,
            "arguments": tool_call.arguments,
            "success": result.success,
            "result": result.result if result.success else None,
            "error": result.error,
            "status": status,
        })
    
    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Get execution log"""
        return self.execution_log
    
    def clear_log(self):
        """Clear execution log"""
        self.execution_log = []
    
    def set_noise_config(self, noise_config: NoiseConfig):
        """Update noise configuration"""
        self.noise_config = noise_config
