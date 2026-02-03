"""
Generate Rollouts - 生成 Agent Rollouts

批量运行 Agent 并记录 episode，包括激活缓存。
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from controller import AgentLoop, AgentConfig
from controller.sandbox_tools import NoiseConfig
from tasks import When2CallAdapter, BFCLAdapter, SyntheticGenerator, TaskSample, DecisionLabel
from .rollout_logger import RolloutLogger, EpisodeLog, StepLog, create_step_log


class RolloutGenerator:
    """Rollout 生成器"""
    
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        cache_activations: bool = True,
        hook_layers: Optional[List[int]] = None,
        noise_config: Optional[NoiseConfig] = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ):
        """
        Args:
            model_name: 模型名称或路径
            output_dir: 输出目录
            cache_activations: 是否缓存激活
            hook_layers: 要 hook 的层
            noise_config: 噪声配置
            device: 设备
            dtype: 数据类型
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_activations = cache_activations
        self.hook_layers = hook_layers or [24, 27]
        self.noise_config = noise_config
        
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
        
        # 激活保存目录
        self.activation_dir = self.output_dir / "activations"
        self.activation_dir.mkdir(exist_ok=True)
    
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
            # 保存激活
            activation_info = None
            if self.cache_activations and step.activations:
                activation_info = self._save_activations(
                    sample.sample_id,
                    step.step_id,
                    step.activations,
                )
            
            step_log = create_step_log(
                step_id=step.step_id,
                model_output=step.model_output,
                decision_type=step.decision_type.value,
                tool_name=step.tool_call.get("name") if step.tool_call else None,
                tool_args=step.tool_call.get("arguments") if step.tool_call else None,
                tool_result=step.tool_result,
                activation_info=activation_info,
            )
            episode_log.steps.append(step_log)
        
        # 填充结果信息
        episode_log.final_decision = result.final_decision.value
        episode_log.final_response = result.final_response
        episode_log.total_steps = result.total_steps
        episode_log.total_tool_calls = result.total_tool_calls
        
        # 判断成功（基于标签）
        if sample.label == DecisionLabel.CALL:
            # 应该调用工具
            episode_log.success = (
                result.total_tool_calls > 0 and
                (sample.expected_tool is None or 
                 any(s.tool_name == sample.expected_tool for s in episode_log.steps))
            )
        elif sample.label == DecisionLabel.NO_CALL:
            # 不应该调用工具
            episode_log.success = result.total_tool_calls == 0
        else:
            # 不确定
            episode_log.success = result.final_response is not None
        
        return episode_log
    
    def _save_activations(
        self,
        episode_id: str,
        step_id: int,
        activations: Dict[int, torch.Tensor],
    ) -> Dict[str, Any]:
        """保存激活到磁盘"""
        filename = f"{episode_id}_step{step_id}.pt"
        filepath = self.activation_dir / filename
        
        # 保存为 PyTorch 格式
        torch.save(activations, filepath)
        
        # 获取形状信息
        shapes = {str(k): list(v.shape) for k, v in activations.items()}
        
        return {
            "path": str(filepath),
            "shape": shapes,
        }


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="Generate Agent Rollouts")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--output-dir", type=str, default="./data/rollouts", help="Output directory")
    parser.add_argument("--dataset", type=str, default="synthetic", 
                        choices=["synthetic", "when2call", "bfcl"],
                        help="Dataset to use")
    parser.add_argument("--data-path", type=str, default=None, help="Dataset path")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--cache-activations", action="store_true", help="Cache activations")
    parser.add_argument("--hook-layers", type=int, nargs="+", default=[24, 27], 
                        help="Layers to hook")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--experiment-name", type=str, default=None, help="Experiment name")
    
    args = parser.parse_args()
    
    # 加载数据
    if args.dataset == "synthetic":
        generator = SyntheticGenerator()
        samples = generator.generate()[:args.num_samples]
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
        cache_activations=args.cache_activations,
        hook_layers=args.hook_layers,
        device=args.device,
    )
    
    experiment_name = args.experiment_name or f"{args.dataset}_{len(samples)}"
    results = generator.run(samples, experiment_name=experiment_name)
    
    print(f"Generated {len(results)} rollouts")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
