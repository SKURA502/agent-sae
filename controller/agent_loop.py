"""
Agent Loop - Agent 核心循环

实现 LLM Agent 的闭环交互：
prompt → LLM → action → env → observation → LLM → ...

支持：
- 激活缓存（用于 SAE 训练）
- 详细日志记录
- 多种终止条件
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .tool_schema import (
    AgentDecision,
    DecisionType,
    ToolSchema,
    ToolResult,
    get_tool_schemas,
)
from .output_parser import OutputParser, ParseResult
from .sandbox_tools import ToolExecutor, NoiseConfig, SearchTool, CalculatorTool, LookupTool


@dataclass
class AgentConfig:
    """Agent 配置"""
    # 模型配置
    model_name_or_path: str = "meta-llama/Llama-3-8B-Instruct"
    device: str = "cuda"
    dtype: str = "bfloat16"
    
    # 生成配置
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    
    # Agent 配置
    max_steps: int = 10
    available_tools: List[str] = field(default_factory=lambda: ["search", "calculator", "lookup"])
    
    # 激活缓存配置
    cache_activations: bool = True
    hook_layers: List[int] = field(default_factory=lambda: [24, 27])
    w_pre_window: int = 20  # tool_call 输出前的 token 数
    w_post_window: int = 10
    
    # 噪声配置
    noise_config: Optional[NoiseConfig] = None


@dataclass
class StepRecord:
    """单步记录"""
    step_id: int
    prompt: str
    model_output: str
    parse_result: ParseResult
    decision_type: DecisionType
    tool_call: Optional[Dict[str, Any]] = None
    tool_result: Optional[Dict[str, Any]] = None
    activations: Optional[Dict[str, torch.Tensor]] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class EpisodeResult:
    """Episode 结果"""
    episode_id: str
    task_input: Dict[str, Any]
    steps: List[StepRecord]
    final_decision: DecisionType
    final_response: Optional[str]
    success: bool
    total_steps: int
    total_tool_calls: int
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化的字典"""
        return {
            "episode_id": self.episode_id,
            "task_input": self.task_input,
            "steps": [
                {
                    "step_id": s.step_id,
                    "decision_type": s.decision_type.value,
                    "tool_call": s.tool_call,
                    "tool_result": s.tool_result,
                    # 不包含激活（太大）
                }
                for s in self.steps
            ],
            "final_decision": self.final_decision.value,
            "final_response": self.final_response,
            "success": self.success,
            "total_steps": self.total_steps,
            "total_tool_calls": self.total_tool_calls,
        }


class ActivationCache:
    """激活缓存器"""
    
    def __init__(self, layers: List[int]):
        self.layers = layers
        self.activations: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}
        self.hooks = []
    
    def _create_hook(self, layer_idx: int):
        """创建 hook 函数"""
        def hook(module, input, output):
            # 存储残差流激活
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            # 只保留最后的 token 位置（节省内存）
            self.activations[layer_idx].append(hidden_states[:, -1:, :].detach().cpu())
        return hook
    
    def register_hooks(self, model):
        """注册 hook"""
        for layer_idx in self.layers:
            # 获取层模块（需要根据具体模型结构调整）
            try:
                layer = model.model.layers[layer_idx]
                hook = layer.register_forward_hook(self._create_hook(layer_idx))
                self.hooks.append(hook)
            except (AttributeError, IndexError) as e:
                print(f"Warning: Could not register hook for layer {layer_idx}: {e}")
    
    def remove_hooks(self):
        """移除所有 hook"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_activations(self, window_size: int = 20) -> Dict[int, torch.Tensor]:
        """获取最近 window_size 个 token 的激活"""
        result = {}
        for layer_idx, acts in self.activations.items():
            if acts:
                # 拼接并取最后 window_size 个
                concatenated = torch.cat(acts, dim=1)
                result[layer_idx] = concatenated[:, -window_size:, :]
        return result
    
    def clear(self):
        """清空缓存"""
        self.activations = {l: [] for l in self.layers}


class AgentLoop:
    """Agent 核心循环"""
    
    SYSTEM_PROMPT_TEMPLATE = """You are a helpful AI assistant. You can use the following tools to help answer questions:

{tool_descriptions}

{output_format}

Important rules:
1. If you are unsure about the answer, or the question requires looking up specific facts, use the tools
2. If you already have enough information to answer directly, do not use tools
3. You can only call one tool at a time
4. You must strictly follow the JSON format for output"""

    def __init__(
        self,
        config: AgentConfig,
        model: Optional[AutoModelForCausalLM] = None,
        tokenizer: Optional[AutoTokenizer] = None,
    ):
        """
        Args:
            config: Agent 配置
            model: 预加载的模型（可选）
            tokenizer: 预加载的 tokenizer（可选）
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        
        # 初始化工具执行器
        self._init_tools()
        
        # 初始化输出解析器
        self.parser = OutputParser(available_tools=config.available_tools)
        
        # 激活缓存
        self.activation_cache: Optional[ActivationCache] = None
        
        # Episode 计数
        self.episode_count = 0
    
    def _init_tools(self):
        """初始化工具"""
        tools = []
        if "search" in self.config.available_tools:
            tools.append(SearchTool())
        if "calculator" in self.config.available_tools:
            tools.append(CalculatorTool())
        if "lookup" in self.config.available_tools:
            tools.append(LookupTool())
        
        self.tool_executor = ToolExecutor(
            tools=tools,
            noise_config=self.config.noise_config or NoiseConfig()
        )
    
    def load_model(self):
        """加载模型"""
        if self.model is not None:
            return
        
        print(f"Loading model: {self.config.model_name_or_path}")
        
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=dtype_map.get(self.config.dtype, torch.bfloat16),
            device_map=self.config.device,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Model loaded successfully")
    
    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        tool_schemas = get_tool_schemas(self.config.available_tools)
        tool_descriptions = "\n\n".join([s.to_prompt_format() for s in tool_schemas])
        output_format = OutputParser.format_expected_output()
        
        return self.SYSTEM_PROMPT_TEMPLATE.format(
            tool_descriptions=tool_descriptions,
            output_format=output_format,
        )
    
    def _build_conversation(
        self,
        instruction: str,
        context: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """构建对话历史"""
        messages = [{"role": "system", "content": self._build_system_prompt()}]
        
        # 添加历史记录
        if history:
            messages.extend(history)
        
        # 构建用户消息
        user_content = instruction
        if context:
            user_content = f"Context: {context}\n\nQuestion: {instruction}"
        
        messages.append({"role": "user", "content": user_content})
        
        return messages
    
    @torch.no_grad()
    def _generate(self, messages: List[Dict[str, str]]) -> str:
        """生成模型输出"""
        # 使用 chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback: 简单拼接
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            prompt += "\nassistant:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature if self.config.do_sample else None,
            top_p=self.config.top_p if self.config.do_sample else None,
            do_sample=self.config.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        # 解码输出（只取生成的部分）
        generated = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        
        return response.strip()
    
    def run_episode(
        self,
        instruction: str,
        context: Optional[str] = None,
        episode_id: Optional[str] = None,
    ) -> EpisodeResult:
        """运行一个 episode
        
        Args:
            instruction: 用户指令
            context: 可选的背景信息
            episode_id: episode ID
            
        Returns:
            EpisodeResult
        """
        if self.model is None:
            self.load_model()
        
        # 初始化激活缓存
        if self.config.cache_activations:
            self.activation_cache = ActivationCache(self.config.hook_layers)
            self.activation_cache.register_hooks(self.model)
        
        # 生成 episode ID
        if episode_id is None:
            episode_id = f"ep_{self.episode_count:06d}"
            self.episode_count += 1
        
        # 初始化
        steps: List[StepRecord] = []
        history: List[Dict[str, str]] = []
        total_tool_calls = 0
        
        try:
            for step_id in range(self.config.max_steps):
                # 构建对话
                messages = self._build_conversation(instruction, context, history)
                
                # 生成输出
                model_output = self._generate(messages)
                
                # 解析输出
                parse_result = self.parser.parse(model_output)
                
                # 获取激活
                activations = None
                if self.config.cache_activations and self.activation_cache:
                    activations = self.activation_cache.get_activations(
                        self.config.w_pre_window
                    )
                    self.activation_cache.clear()
                
                # 创建步骤记录
                step_record = StepRecord(
                    step_id=step_id,
                    prompt=messages[-1]["content"],
                    model_output=model_output,
                    parse_result=parse_result,
                    decision_type=parse_result.decision.decision_type if parse_result.decision else DecisionType.NO_CALL,
                    activations=activations,
                )
                
                # 处理决策
                if parse_result.is_success and parse_result.decision:
                    decision = parse_result.decision
                    
                    if decision.is_tool_call:
                        # 执行工具调用
                        tool_call = decision.tool_call
                        step_record.tool_call = {
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                        }
                        
                        tool_result = self.tool_executor.execute(tool_call)
                        step_record.tool_result = {
                            "success": tool_result.success,
                            "result": tool_result.result,
                            "error": tool_result.error,
                        }
                        
                        total_tool_calls += 1
                        
                        # 更新历史
                        history.append({"role": "assistant", "content": model_output})
                        history.append({"role": "user", "content": tool_result.to_observation()})
                        
                    else:
                        # 直接回答，结束循环
                        steps.append(step_record)
                        break
                else:
                    # 解析失败，尝试继续（非严格模式）或结束
                    if not self.parser.strict_mode:
                        steps.append(step_record)
                        break
                
                steps.append(step_record)
            
            # 确定最终结果
            final_step = steps[-1] if steps else None
            final_decision = final_step.decision_type if final_step else DecisionType.NO_CALL
            final_response = None
            if final_step and final_step.parse_result.decision:
                final_response = final_step.parse_result.decision.response
            
            return EpisodeResult(
                episode_id=episode_id,
                task_input={"instruction": instruction, "context": context},
                steps=steps,
                final_decision=final_decision,
                final_response=final_response,
                success=final_decision == DecisionType.NO_CALL and final_response is not None,
                total_steps=len(steps),
                total_tool_calls=total_tool_calls,
            )
            
        finally:
            # 清理
            if self.activation_cache:
                self.activation_cache.remove_hooks()
                self.activation_cache = None
    
    def run_batch(
        self,
        tasks: List[Dict[str, Any]],
        output_path: Optional[str] = None,
    ) -> List[EpisodeResult]:
        """批量运行 episode
        
        Args:
            tasks: 任务列表，每个任务包含 instruction 和可选的 context
            output_path: 结果输出路径
            
        Returns:
            EpisodeResult 列表
        """
        results = []
        
        for i, task in enumerate(tasks):
            print(f"Running episode {i+1}/{len(tasks)}")
            
            result = self.run_episode(
                instruction=task.get("instruction", ""),
                context=task.get("context"),
                episode_id=task.get("episode_id"),
            )
            results.append(result)
            
            # 保存中间结果
            if output_path:
                self._save_result(result, output_path)
        
        return results
    
    def _save_result(self, result: EpisodeResult, output_path: str):
        """保存单个结果"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "a") as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
