"""
Generate Rollouts - 生成 Agent Rollouts

批量运行 Agent 并记录 episode。
支持运行时推理模式，激活直接流式传输到 SAE 训练，不保存到磁盘。
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import torch
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from controller import AgentLoop, AgentConfig
from controller.sandbox_tools import NoiseConfig
from tasks import When2CallAdapter, BFCLAdapter, SyntheticGenerator, TaskSample, DecisionLabel
from .rollout_logger import RolloutLogger, EpisodeLog, StepLog, create_step_log


# 激活回调类型：接收 (episode_id, step_id, decision_type, activations)
ActivationCallback = Callable[[str, int, str, Dict[int, torch.Tensor]], None]


class RolloutGenerator:
    """Rollout 生成器
    
    支持两种模式：
    1. 日志模式：只记录 rollout 日志，不处理激活
    2. 流式模式：将激活实时传递给回调函数（用于 SAE 训练）
    """
    
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        cache_activations: bool = True,
        hook_layers: Optional[List[int]] = None,
        noise_config: Optional[NoiseConfig] = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        activation_callback: Optional[ActivationCallback] = None,
    ):
        """
        Args:
            model_name: 模型名称或路径
            output_dir: 输出目录（用于日志）
            cache_activations: 是否缓存激活（用于流式传输）
            hook_layers: 要 hook 的层
            noise_config: 噪声配置
            device: 设备
            dtype: 数据类型
            activation_callback: 激活回调函数（流式模式）
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_activations = cache_activations
        self.hook_layers = hook_layers or [24, 27]
        self.noise_config = noise_config
        self.activation_callback = activation_callback
        
        # 创建 Agent 配置
        self.agent_config = AgentConfig(
            model_name_or_path=model_name,
            device=device,
            dtype=dtype,
            cache_activations=cache_activations,
            hook_layers=self.hook_layers,
            noise_config=noise_config,
        )
        
        self.agent: Optional[AgentLoop] = None
        self.logger: Optional[RolloutLogger] = None
    
    def init_agent(self):
        """初始化 Agent"""
        if self.agent is None:
            print(f"Initializing agent with model: {self.model_name}")
            self.agent = AgentLoop(self.agent_config)
            self.agent.load_model()
    
    def run(
        self,
        samples: List[TaskSample],
        experiment_name: str = "rollout",
        show_progress: bool = True,
    ) -> List[EpisodeLog]:
        """运行 rollout 生成
        
        Args:
            samples: 任务样本列表
            experiment_name: 实验名称
            show_progress: 是否显示进度条
            
        Returns:
            EpisodeLog 列表
        """
        self.init_agent()
        
        # 初始化日志器
        self.logger = RolloutLogger(
            output_dir=str(self.output_dir),
            experiment_name=experiment_name,
        )
        
        results = []
        iterator = tqdm(samples, desc="Generating rollouts") if show_progress else samples
        
        try:
            for sample in iterator:
                episode_log = self._run_single(sample)
                results.append(episode_log)
                self.logger.log_episode(episode_log)
        finally:
            self.logger.close()
        
        return results
    
    def run_streaming(
        self,
        samples: List[TaskSample],
        show_progress: bool = True,
    ) -> Generator[Tuple[EpisodeLog, Dict[int, torch.Tensor]], None, None]:
        """流式运行 rollout 生成
        
        每个 episode 完成后 yield 日志和激活，用于实时 SAE 训练。
        激活不会保存到磁盘。
        
        Args:
            samples: 任务样本列表
            show_progress: 是否显示进度条
            
        Yields:
            (episode_log, activations) 元组
            activations: {layer_idx: [num_steps, window_size, hidden_dim]}
        """
        self.init_agent()
        
        iterator = tqdm(samples, desc="Streaming rollouts") if show_progress else samples
        
        for sample in iterator:
            episode_log, activations = self._run_single_with_activations(sample)
            yield episode_log, activations
    
    def _run_single(self, sample: TaskSample) -> EpisodeLog:
        """运行单个样本"""
        episode_log = EpisodeLog(
            episode_id=sample.sample_id,
            instruction=sample.instruction,
            context=sample.context,
            available_tools=sample.available_tools,
            label=sample.label.value,
            expected_tool=sample.expected_tool,
            expected_args=sample.expected_args,
            model_name=self.model_name,
            source_dataset=sample.source_dataset,
            sample_id=sample.sample_id,
        )
        
        # 运行 Agent
        result = self.agent.run_episode(
            instruction=sample.instruction,
            context=sample.context,
            episode_id=sample.sample_id,
        )
        
        # 转换步骤记录
        for step in result.steps:
            # 如果有回调函数，传递激活
            if self.activation_callback and step.activations:
                self.activation_callback(
                    sample.sample_id,
                    step.step_id,
                    step.decision_type.value,
                    step.activations,
                )
            
            step_log = create_step_log(
                step_id=step.step_id,
                model_output=step.model_output,
                decision_type=step.decision_type.value,
                tool_name=step.tool_call.get("name") if step.tool_call else None,
                tool_args=step.tool_call.get("arguments") if step.tool_call else None,
                tool_result=step.tool_result,
                activation_info=None,  # 不保存激活路径
            )
            episode_log.steps.append(step_log)
        
        # 填充结果信息
        episode_log.final_decision = result.final_decision.value
        episode_log.final_response = result.final_response
        episode_log.total_steps = result.total_steps
        episode_log.total_tool_calls = result.total_tool_calls
        
        # 判断成功（基于标签）
        if sample.label == DecisionLabel.CALL:
            episode_log.success = (
                result.total_tool_calls > 0 and
                (sample.expected_tool is None or 
                 any(s.tool_name == sample.expected_tool for s in episode_log.steps))
            )
        elif sample.label == DecisionLabel.NO_CALL:
            episode_log.success = result.total_tool_calls == 0
        else:
            episode_log.success = result.final_response is not None
        
        return episode_log
    
    def _run_single_with_activations(
        self,
        sample: TaskSample,
    ) -> Tuple[EpisodeLog, Dict[int, List[torch.Tensor]]]:
        """运行单个样本并返回激活
        
        Returns:
            (episode_log, activations)
            activations: {layer_idx: [激活列表]}
        """
        episode_log = EpisodeLog(
            episode_id=sample.sample_id,
            instruction=sample.instruction,
            context=sample.context,
            available_tools=sample.available_tools,
            label=sample.label.value,
            expected_tool=sample.expected_tool,
            expected_args=sample.expected_args,
            model_name=self.model_name,
            source_dataset=sample.source_dataset,
            sample_id=sample.sample_id,
        )
        
        # 收集该 episode 的所有激活
        episode_activations: Dict[int, List[torch.Tensor]] = {
            layer: [] for layer in self.hook_layers
        }
        
        # 运行 Agent
        result = self.agent.run_episode(
            instruction=sample.instruction,
            context=sample.context,
            episode_id=sample.sample_id,
        )
        
        # 转换步骤记录并收集激活
        for step in result.steps:
            if step.activations:
                for layer, acts in step.activations.items():
                    if layer in episode_activations:
                        episode_activations[layer].append(acts)
            
            step_log = create_step_log(
                step_id=step.step_id,
                model_output=step.model_output,
                decision_type=step.decision_type.value,
                tool_name=step.tool_call.get("name") if step.tool_call else None,
                tool_args=step.tool_call.get("arguments") if step.tool_call else None,
                tool_result=step.tool_result,
                activation_info=None,
            )
            episode_log.steps.append(step_log)
        
        # 填充结果信息
        episode_log.final_decision = result.final_decision.value
        episode_log.final_response = result.final_response
        episode_log.total_steps = result.total_steps
        episode_log.total_tool_calls = result.total_tool_calls
        
        # 判断成功
        if sample.label == DecisionLabel.CALL:
            episode_log.success = (
                result.total_tool_calls > 0 and
                (sample.expected_tool is None or 
                 any(s.tool_name == sample.expected_tool for s in episode_log.steps))
            )
        elif sample.label == DecisionLabel.NO_CALL:
            episode_log.success = result.total_tool_calls == 0
        else:
            episode_log.success = result.final_response is not None
        
        return episode_log, episode_activations


class ToolUseActivationCollector:
    """Tool-use 激活收集器
    
    用于收集 tool-use 任务的激活，支持流式传输到 SAE 训练。
    不保存任何数据到磁盘。
    """
    
    def __init__(
        self,
        model_name: str,
        layers: List[int],
        device: str = "cuda",
        dtype: str = "bfloat16",
        buffer_size: int = 8192,
    ):
        """
        Args:
            model_name: 模型名称
            layers: 要收集的层
            device: 设备
            dtype: 数据类型
            buffer_size: 激活缓冲区大小
        """
        self.model_name = model_name
        self.layers = layers
        self.device = device
        self.dtype = dtype
        self.buffer_size = buffer_size
        
        # 激活缓冲区
        self._buffers: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}
        self._labels: List[int] = []  # 0=NO_CALL, 1=CALL
        self._current_size = 0
    
    def _add_to_buffer(
        self,
        activations: Dict[int, torch.Tensor],
        decision: str,
    ):
        """添加激活到缓冲区"""
        for layer, acts in activations.items():
            if layer in self._buffers:
                # 展平为 2D: [batch * seq, hidden]
                if len(acts.shape) == 3:
                    acts = acts.view(-1, acts.shape[-1])
                self._buffers[layer].append(acts)
        
        # 记录标签
        label = 1 if decision == "call" else 0
        batch_size = next(iter(activations.values())).shape[0] if activations else 0
        self._labels.extend([label] * batch_size)
        
        # 更新大小
        if self._buffers:
            first_layer = self.layers[0]
            self._current_size = sum(t.shape[0] for t in self._buffers[first_layer])
    
    def collect_from_samples(
        self,
        samples: List[TaskSample],
        output_dir: Optional[str] = None,
    ) -> Generator[Tuple[Dict[int, torch.Tensor], torch.Tensor], None, None]:
        """从样本收集激活
        
        Args:
            samples: 任务样本
            output_dir: 日志输出目录（可选）
            
        Yields:
            (activations, labels) 当缓冲区满时
            activations: {layer: [buffer_size, hidden_dim]}
            labels: [buffer_size] 0=NO_CALL, 1=CALL
        """
        generator = RolloutGenerator(
            model_name=self.model_name,
            output_dir=output_dir or "./outputs/rollouts",
            cache_activations=True,
            hook_layers=self.layers,
            device=self.device,
            dtype=self.dtype,
        )
        
        for episode_log, activations in generator.run_streaming(samples):
            # 获取决策
            decision = episode_log.final_decision
            
            # 合并该 episode 的所有激活
            for layer in self.layers:
                if activations.get(layer):
                    combined = torch.cat(activations[layer], dim=1)
                    self._add_to_buffer({layer: combined}, decision)
            
            # 检查缓冲区是否满
            if self._current_size >= self.buffer_size:
                yield self._flush_buffer()
        
        # 最后刷新缓冲区
        if self._current_size > 0:
            yield self._flush_buffer()
    
    def _flush_buffer(self) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        """刷新缓冲区"""
        result = {}
        for layer in self.layers:
            if self._buffers[layer]:
                result[layer] = torch.cat(self._buffers[layer], dim=0)
        
        labels = torch.tensor(self._labels)
        
        # 清空缓冲区
        self._buffers = {l: [] for l in self.layers}
        self._labels = []
        self._current_size = 0
        
        return result, labels


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="Generate Agent Rollouts")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--output-dir", type=str, default="./data/rollouts", help="Output directory for logs")
    parser.add_argument("--dataset", type=str, default="synthetic", 
                        choices=["synthetic", "when2call", "bfcl"],
                        help="Dataset to use")
    parser.add_argument("--data-path", type=str, default=None, help="Dataset path")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--hook-layers", type=int, nargs="+", default=[24, 27], 
                        help="Layers to hook for activation")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--experiment-name", type=str, default=None, help="Experiment name")
    parser.add_argument("--streaming", action="store_true", 
                        help="Use streaming mode (activations not saved to disk)")
    
    args = parser.parse_args()
    
    # 加载数据
    if args.dataset == "synthetic":
        generator_cls = SyntheticGenerator()
        samples = generator_cls.generate()[:args.num_samples]
    elif args.dataset == "when2call":
        adapter = When2CallAdapter(args.data_path or "./data/raw/when2call")
        adapter.load()
        samples = list(adapter)[:args.num_samples]
    elif args.dataset == "bfcl":
        adapter = BFCLAdapter(args.data_path or "./data/raw/bfcl")
        adapter.load()
        samples = list(adapter)[:args.num_samples]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    print(f"Loaded {len(samples)} samples")
    
    # 运行 rollout 生成
    generator = RolloutGenerator(
        model_name=args.model,
        output_dir=args.output_dir,
        cache_activations=True,  # 总是缓存用于流式处理
        hook_layers=args.hook_layers,
        device=args.device,
    )
    
    experiment_name = args.experiment_name or f"{args.dataset}_{len(samples)}"
    
    if args.streaming:
        # 流式模式：打印统计信息
        call_count = 0
        no_call_count = 0
        
        for episode_log, activations in generator.run_streaming(samples):
            if episode_log.final_decision == "call":
                call_count += 1
            else:
                no_call_count += 1
        
        print(f"\nStreaming complete!")
        print(f"  CALL: {call_count}, NO_CALL: {no_call_count}")
        print("Note: Activations were not saved to disk (streaming mode)")
    else:
        # 标准模式：只保存日志
        results = generator.run(samples, experiment_name=experiment_name)
        
        print(f"Generated {len(results)} rollouts")
        print(f"Logs saved to {args.output_dir}")
        print("Note: Activations are not saved to disk. Use streaming API for SAE training.")


if __name__ == "__main__":
    main()
