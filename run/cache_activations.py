"""
Streaming Activations - 流式激活处理工具

提供运行时推理的激活流式处理，用于 SAE 训练。
不保存 hidden states 到磁盘，而是实时流式传输到 SAE 训练。
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import torch
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class StreamingConfig:
    """流式处理配置"""
    # 缓冲区大小（累积多少样本后 yield）
    buffer_size: int = 8192
    # 是否平衡 CALL/NO_CALL
    balance: bool = True
    # 层列表
    layers: List[int] = None
    # 设备
    device: str = "cuda"
    
    def __post_init__(self):
        if self.layers is None:
            self.layers = [24, 27]


class ActivationBuffer:
    """激活缓冲区
    
    累积激活数据，支持按决策类型平衡采样。
    """
    
    def __init__(
        self,
        layers: List[int],
        buffer_size: int = 8192,
        balance: bool = True,
    ):
        """
        Args:
            layers: 要收集的层
            buffer_size: 缓冲区大小
            balance: 是否平衡 CALL/NO_CALL
        """
        self.layers = layers
        self.buffer_size = buffer_size
        self.balance = balance
        
        # 分别存储 CALL 和 NO_CALL 激活
        self._call_buffers: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}
        self._no_call_buffers: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}
        
        self._call_count = 0
        self._no_call_count = 0
    
    def add(
        self,
        activations: Dict[int, torch.Tensor],
        decision: str,
    ):
        """添加激活
        
        Args:
            activations: {layer: [batch, seq, hidden] or [batch, hidden]}
            decision: 决策类型 ("call" or "no_call")
        """
        is_call = decision.lower() == "call"
        buffers = self._call_buffers if is_call else self._no_call_buffers
        
        for layer, acts in activations.items():
            if layer not in self.layers:
                continue
            
            # 展平为 2D
            if len(acts.shape) == 3:
                acts = acts.view(-1, acts.shape[-1])
            elif len(acts.shape) == 1:
                acts = acts.unsqueeze(0)
            
            buffers[layer].append(acts.cpu())
        
        # 更新计数
        sample_count = acts.shape[0] if len(acts.shape) >= 1 else 1
        if is_call:
            self._call_count += sample_count
        else:
            self._no_call_count += sample_count
    
    @property
    def total_size(self) -> int:
        """当前总大小"""
        return self._call_count + self._no_call_count
    
    @property
    def is_ready(self) -> bool:
        """缓冲区是否达到指定大小"""
        if self.balance:
            # 平衡模式：需要两类都有足够数据
            min_per_class = self.buffer_size // 2
            return self._call_count >= min_per_class and self._no_call_count >= min_per_class
        else:
            return self.total_size >= self.buffer_size
    
    def get_and_clear(self) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        """获取并清空缓冲区
        
        Returns:
            (activations, labels)
            activations: {layer: [total, hidden]}
            labels: [total] 0=NO_CALL, 1=CALL
        """
        result = {}
        labels = []
        
        if self.balance:
            # 平衡采样
            min_count = min(self._call_count, self._no_call_count)
            samples_per_class = min(min_count, self.buffer_size // 2)
            
            for layer in self.layers:
                call_acts = torch.cat(self._call_buffers[layer], dim=0)[:samples_per_class]
                no_call_acts = torch.cat(self._no_call_buffers[layer], dim=0)[:samples_per_class]
                result[layer] = torch.cat([call_acts, no_call_acts], dim=0)
            
            labels = [1] * samples_per_class + [0] * samples_per_class
        else:
            # 不平衡：全部返回
            for layer in self.layers:
                all_acts = []
                if self._call_buffers[layer]:
                    all_acts.append(torch.cat(self._call_buffers[layer], dim=0))
                if self._no_call_buffers[layer]:
                    all_acts.append(torch.cat(self._no_call_buffers[layer], dim=0))
                if all_acts:
                    result[layer] = torch.cat(all_acts, dim=0)
            
            labels = [1] * self._call_count + [0] * self._no_call_count
        
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        # 清空缓冲区
        self._call_buffers = {l: [] for l in self.layers}
        self._no_call_buffers = {l: [] for l in self.layers}
        self._call_count = 0
        self._no_call_count = 0
        
        return result, labels_tensor
    
    def get_remaining(self) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        """获取剩余数据（不平衡）"""
        if self.total_size == 0:
            return {}, torch.tensor([])
        
        result = {}
        for layer in self.layers:
            all_acts = []
            if self._call_buffers[layer]:
                all_acts.append(torch.cat(self._call_buffers[layer], dim=0))
            if self._no_call_buffers[layer]:
                all_acts.append(torch.cat(self._no_call_buffers[layer], dim=0))
            if all_acts:
                result[layer] = torch.cat(all_acts, dim=0)
        
        labels = [1] * self._call_count + [0] * self._no_call_count
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        # 清空
        self._call_buffers = {l: [] for l in self.layers}
        self._no_call_buffers = {l: [] for l in self.layers}
        self._call_count = 0
        self._no_call_count = 0
        
        return result, labels_tensor


def create_streaming_data_pipeline(
    rollout_generator,
    samples: list,
    config: StreamingConfig,
) -> Generator[Tuple[Dict[int, torch.Tensor], torch.Tensor], None, None]:
    """创建流式数据管道
    
    将 rollout 生成器的输出转换为 SAE 训练所需的格式。
    
    Args:
        rollout_generator: RolloutGenerator 实例
        samples: 任务样本列表
        config: 流式处理配置
        
    Yields:
        (activations, labels) 当缓冲区满时
    """
    buffer = ActivationBuffer(
        layers=config.layers,
        buffer_size=config.buffer_size,
        balance=config.balance,
    )
    
    for episode_log, activations in rollout_generator.run_streaming(samples):
        # 获取决策
        decision = episode_log.final_decision
        
        # 合并该 episode 的激活
        merged = {}
        for layer in config.layers:
            if activations.get(layer):
                merged[layer] = torch.cat(activations[layer], dim=0)
        
        if merged:
            buffer.add(merged, decision)
        
        # 检查缓冲区
        if buffer.is_ready:
            yield buffer.get_and_clear()
    
    # 返回剩余数据
    remaining = buffer.get_remaining()
    if remaining[0]:
        yield remaining


class StreamingActivationDataset:
    """流式激活数据集
    
    用于将流式激活数据转换为 PyTorch 可用的格式。
    """
    
    def __init__(
        self,
        activation_generator: Generator[Tuple[Dict[int, torch.Tensor], torch.Tensor], None, None],
        layer: int,
    ):
        """
        Args:
            activation_generator: 激活生成器
            layer: 目标层
        """
        self.generator = activation_generator
        self.layer = layer
        self._buffer = None
        self._labels = None
        self._index = 0
    
    def __iter__(self):
        """迭代器"""
        for activations, labels in self.generator:
            if self.layer in activations:
                data = activations[self.layer]
                for i in range(len(data)):
                    yield data[i], labels[i]

def _load_samples(dataset: str, data_path: Optional[str], num_samples: int):
    """按数据集类型加载样本。"""
    from tasks import SyntheticGenerator, When2CallAdapter, BFCLAdapter

    if dataset == "synthetic":
        generator = SyntheticGenerator()
        return generator.generate()[:num_samples]

    if dataset == "when2call":
        adapter = When2CallAdapter(data_path or "./data/raw/when2call")
    elif dataset == "bfcl":
        adapter = BFCLAdapter(data_path or "./data/raw/bfcl")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    adapter.load()
    return list(adapter)[:num_samples]


def _find_stage1_checkpoints(stage1_dir: str, layers: List[int]) -> Dict[int, str]:
    """按 train_sae 统一命名规则查找 Stage 1 检查点。"""
    checkpoints: Dict[int, str] = {}
    checkpoint_dir = Path(stage1_dir)

    for layer in layers:
        matches = sorted(checkpoint_dir.glob(f"*-layer{layer}-*-stage1.pt"))
        if matches:
            checkpoints[layer] = str(matches[0])
        else:
            print(f"Warning: Stage 1 checkpoint not found for layer {layer}")

    return checkpoints


def _estimate_total_steps(num_samples: int, batch_size: int, buffer_size: int) -> int:
    """估算流式训练总步数，用于学习率调度。"""
    estimated_points = max(num_samples * 20, buffer_size)
    return max(1, estimated_points // max(batch_size, 1))


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="Streaming Activations - 流式激活处理工具\n"
                    "用于运行时推理，不保存 hidden states 到磁盘"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # train 命令：流式 SAE 训练（Stage 2）
    train_parser = subparsers.add_parser("train", help="流式训练 SAE (Stage 2)")
    train_parser.add_argument("--model", type=str, required=True,
                              help="LLM model name or path")
    train_parser.add_argument("--dataset", type=str, default="synthetic",
                              choices=["synthetic", "when2call", "bfcl"],
                              help="Dataset to use")
    train_parser.add_argument("--data-path", type=str, default=None,
                              help="Dataset path")
    train_parser.add_argument("--num-samples", type=int, default=1000,
                              help="Number of samples")
    train_parser.add_argument("--layers", type=int, nargs="+", default=[24, 27],
                              help="Layers to train")
    train_parser.add_argument("--output-dir", type=str, 
                              default="./outputs/sae_checkpoints",
                              help="Output directory")
    train_parser.add_argument("--stage1-dir", type=str, required=True,
                              help="Stage 1 checkpoint directory")
    train_parser.add_argument("--buffer-size", type=int, default=8192,
                              help="Activation buffer size")
    train_parser.add_argument("--target-tokens", type=int, default=50_000_000,
                              help="Target tokens for Stage 2 checkpoint naming")
    train_parser.add_argument("--learning-rate", type=float, default=5e-5,
                              help="Stage 2 learning rate")
    train_parser.add_argument("--batch-size", type=int, default=4096,
                              help="SAE training batch size")
    train_parser.add_argument("--num-epochs", type=int, default=10,
                              help="Stage 2 epochs (for config tracking)")
    train_parser.add_argument("--decoder-norm-interval", type=int, default=10,
                              help="Apply decoder unit norm every N steps")
    train_parser.add_argument("--use-swanlab", action="store_true",
                              help="Use SwanLab for logging")
    train_parser.add_argument("--balance", action="store_true",
                              help="Balance CALL/NO_CALL samples")
    train_parser.add_argument("--device", type=str, default="cuda",
                              help="Device")
    train_parser.add_argument("--dtype", type=str, default="float32",
                              help="Model dtype for rollout inference")
    
    # info 命令：显示信息
    info_parser = subparsers.add_parser("info", help="显示工具信息")
    
    args = parser.parse_args()
    
    if args.command == "train":
        from run.generate_rollouts import RolloutGenerator
        from sae.train_sae import TwoStageTrainer

        samples = _load_samples(args.dataset, args.data_path, args.num_samples)
        print(f"Loaded {len(samples)} samples")
        if not args.stage1_dir:
            raise ValueError("--stage1-dir is required for stage2 streaming training")

        stage1_checkpoints = _find_stage1_checkpoints(args.stage1_dir, args.layers)

        trainer = TwoStageTrainer(
            model_name_or_path=args.model,
            layers=args.layers,
            output_dir=args.output_dir,
            device=args.device,
        )

        tooluse_config = {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "target_tokens": args.target_tokens,
            "decoder_norm_interval": args.decoder_norm_interval,
            "use_swanlab": args.use_swanlab,
        }
        trainer.train_stage2(stage1_checkpoints=stage1_checkpoints, tooluse_config=tooluse_config)

        streaming_config = StreamingConfig(
            buffer_size=args.buffer_size,
            balance=args.balance,
            layers=args.layers,
            device=args.device,
        )

        rollout_gen = RolloutGenerator(
            model_name=args.model,
            output_dir=args.output_dir,
            cache_activations=True,
            hook_layers=args.layers,
            device=args.device,
            dtype=args.dtype,
        )

        total_steps = _estimate_total_steps(args.num_samples, args.batch_size, args.buffer_size)

        for layer in args.layers:
            print(f"\n=== Streaming Stage 2 training for layer {layer} ===")
            data_pipeline = create_streaming_data_pipeline(rollout_gen, samples, streaming_config)

            def layer_generator():
                for activations, _ in data_pipeline:
                    if layer in activations:
                        yield {layer: activations[layer]}

            layer_trainer = trainer.sae_trainers[layer]
            layer_trainer.train_streaming(layer_generator(), layer=layer, total_steps=total_steps)
            layer_trainer.save_checkpoint(f"{layer_trainer.config.experiment_name}.pt")

        print("\nStreaming training complete!")
    
    elif args.command == "info" or args.command is None:
        print("""
流式激活处理工具
================

本工具支持运行时推理模式，不保存 hidden states 到磁盘。
激活在推理过程中实时流式传输到 SAE 训练。

使用方式：
---------

1. Stage 2 流式训练（tool-use 数据）:
   python -m run.cache_activations train \\
       --model meta-llama/Llama-3-8B-Instruct \\
       --dataset when2call \\
       --layers 24 27 \\
       --stage1-dir ./outputs/sae_checkpoints/stage1 \\
    --target-tokens 50000000 \
       --output-dir ./outputs/sae_checkpoints/stage2

2. 编程接口：
   from run.cache_activations import create_streaming_data_pipeline, StreamingConfig
   from run.generate_rollouts import RolloutGenerator
   
   generator = RolloutGenerator(model_name, output_dir, ...)
   config = StreamingConfig(layers=[24, 27], buffer_size=8192)
   
   for activations, labels in create_streaming_data_pipeline(generator, samples, config):
       # activations: {layer: [batch, hidden]}
       # labels: [batch] 0=NO_CALL, 1=CALL
       pass

核心功能：
---------
- ActivationBuffer: 激活缓冲区，支持平衡采样
- create_streaming_data_pipeline: 创建流式数据管道
- train 子命令：复用 sae.train_sae.TwoStageTrainer 进行 Stage 2 初始化

磁盘空间节省：
------------
由于不保存 hidden states，相比传统方法可节省大量磁盘空间。
例如：50k episodes * 10 steps * 20 tokens * 4096 dim * 4 bytes ≈ 160GB
使用流式模式，这些数据不会被写入磁盘。
        """)


if __name__ == "__main__":
    main()
