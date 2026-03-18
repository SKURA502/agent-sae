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

from controller import AgentLoop, AgentConfig
from controller.sandbox_tools import NoiseConfig
from .when2call_adapter import TaskSample, DecisionLabel
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
                episode_log, _ = self._run_single(sample)
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
        
        Args:
            samples: 任务样本列表
            show_progress: 是否显示进度条
            
        Yields:
            (episode_log, activations) 元组
        """
        self.init_agent()
        
        iterator = tqdm(samples, desc="Streaming rollouts") if show_progress else samples
        
        for sample in iterator:
            episode_log, activations = self._run_single(sample, collect_activations=True)
            yield episode_log, activations
    
    def _run_single(
        self,
        sample: TaskSample,
        collect_activations: bool = False,
    ) -> Tuple[EpisodeLog, Optional[Dict[int, List[torch.Tensor]]]]:
        """运行单个样本

        Args:
            sample: 任务样本
            collect_activations: 是否收集激活

        Returns:
            (episode_log, activations) — activations 为 None 当 collect_activations=False
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
        
        # 初始化激活收集器
        episode_activations = None
        if collect_activations:
            episode_activations = {layer: [] for layer in self.hook_layers}
        
        # 运行 Agent
        result = self.agent.run_episode(
            instruction=sample.instruction,
            context=sample.context,
            episode_id=sample.sample_id,
        )
        
        # 转换步骤记录
        for step in result.steps:
            # 收集激活
            if collect_activations and step.activations:
                for layer, acts in step.activations.items():
                    if layer in episode_activations:
                        episode_activations[layer].append(acts)

            # 回调函数
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


def main():
    """命令行入口"""
    from utils import add_common_args, add_dataset_args, load_samples

    parser = argparse.ArgumentParser(description="Generate Agent Rollouts")
    add_common_args(parser)
    add_dataset_args(parser)
    parser.add_argument("--output-dir", type=str, default="./data/rollouts")
    parser.add_argument("--hook-layers", type=int, nargs="+", default=[24, 27])
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming mode (activations not saved to disk)")
    
    args = parser.parse_args()
    
    samples = load_samples(args.dataset, args.data_path, args.num_samples)
    print(f"Loaded {len(samples)} samples")
    
    generator = RolloutGenerator(
        model_name=args.model,
        output_dir=args.output_dir,
        cache_activations=True,
        hook_layers=args.hook_layers,
        device=args.device,
    )
    
    experiment_name = args.experiment_name or f"{args.dataset}_{len(samples)}"
    
    if args.streaming:
        call_count = 0
        no_call_count = 0
        
        for episode_log, activations in generator.run_streaming(samples):
            if episode_log.final_decision == "call":
                call_count += 1
            else:
                no_call_count += 1
        
        print(f"\nStreaming complete!")
        print(f"  CALL: {call_count}, NO_CALL: {no_call_count}")
    else:
        results = generator.run(samples, experiment_name=experiment_name)
        print(f"Generated {len(results)} rollouts")
        print(f"Logs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
