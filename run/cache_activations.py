"""
Streaming Activations - 流式激活处理工具

提供运行时推理的激活流式处理，用于 SAE 训练。
不保存 hidden states 到磁盘，而是实时流式传输到 SAE 训练。
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

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


def train_sae_streaming(
    model_name: str,
    samples: list,
    layers: List[int],
    output_dir: str,
    sae_config: Optional[Dict[str, Any]] = None,
    streaming_config: Optional[StreamingConfig] = None,
    stage1_checkpoints: Optional[Dict[int, str]] = None,
    device: str = "cuda",
    dtype: str = "bfloat16",
):
    """流式训练 SAE（Stage 2）
    
    从 tool-use 任务中流式收集激活并训练 SAE。
    
    Args:
        model_name: LLM 模型名称
        samples: 任务样本
        layers: 要训练的层
        output_dir: 输出目录
        sae_config: SAE 配置
        streaming_config: 流式处理配置
        stage1_checkpoints: Stage 1 检查点路径
        device: 设备
        dtype: 数据类型
    """
    from run.generate_rollouts import RolloutGenerator
    from sae.train_sae import SAETrainer, TrainingConfig
    from sae.sae_model import TopKSAE
    
    streaming_config = streaming_config or StreamingConfig(layers=layers)
    sae_config = sae_config or {}
    
    # 创建 rollout 生成器
    rollout_gen = RolloutGenerator(
        model_name=model_name,
        output_dir=output_dir,
        cache_activations=True,
        hook_layers=layers,
        device=device,
        dtype=dtype,
    )
    
    # 为每层训练 SAE
    for layer in layers:
        print(f"\n=== Training SAE for layer {layer} ===")
        
        # 加载 Stage 1 检查点（如果有）
        sae_model = None
        if stage1_checkpoints and layer in stage1_checkpoints:
            checkpoint_path = stage1_checkpoints[layer]
            if Path(checkpoint_path).exists():
                print(f"Loading Stage 1 checkpoint: {checkpoint_path}")
                sae_model = TopKSAE.load(checkpoint_path, device=device)
        
        # 创建流式数据管道
        data_pipeline = create_streaming_data_pipeline(
            rollout_gen,
            samples,
            streaming_config,
        )
        
        # 创建训练器
        # 注意：这里需要预先知道 input_dim
        # 可以从 Stage 1 模型获取，或者从第一个 batch 推断
        input_dim = sae_model.config.hidden_size if sae_model else sae_config.get("input_dim", 4096)
        
        train_config = TrainingConfig(
            input_dim=input_dim,
            dict_size=sae_config.get("dict_size", input_dim * 8),
            k=sae_config.get("k", input_dim // 32),
            learning_rate=sae_config.get("learning_rate", 5e-5),
            batch_size=sae_config.get("batch_size", 4096),
            output_dir=output_dir,
            experiment_name=f"stage2_layer{layer}",
            device=device,
        )
        
        trainer = SAETrainer(train_config)
        
        if sae_model is not None:
            trainer.model = sae_model
            trainer.optimizer = torch.optim.AdamW(
                trainer.model.parameters(),
                lr=train_config.learning_rate,
            )
        
        # 流式训练
        def layer_generator():
            for activations, labels in data_pipeline:
                if layer in activations:
                    yield {layer: activations[layer]}, labels
        
        total_steps = len(samples) * 10 // train_config.batch_size  # 估计
        trainer.train_streaming(layer_generator(), layer, total_steps)
    
    print("\nStreaming training complete!")


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
    train_parser.add_argument("--stage1-dir", type=str, default=None,
                              help="Stage 1 checkpoint directory")
    train_parser.add_argument("--buffer-size", type=int, default=8192,
                              help="Activation buffer size")
    train_parser.add_argument("--balance", action="store_true",
                              help="Balance CALL/NO_CALL samples")
    train_parser.add_argument("--device", type=str, default="cuda",
                              help="Device")
    
    # info 命令：显示信息
    info_parser = subparsers.add_parser("info", help="显示工具信息")
    
    args = parser.parse_args()
    
    if args.command == "train":
        # 加载数据样本
        from tasks import SyntheticGenerator, When2CallAdapter, BFCLAdapter
        
        if args.dataset == "synthetic":
            gen = SyntheticGenerator()
            samples = gen.generate()[:args.num_samples]
        elif args.dataset == "when2call":
            adapter = When2CallAdapter(args.data_path or "./data/raw/when2call")
            adapter.load()
            samples = list(adapter)[:args.num_samples]
        elif args.dataset == "bfcl":
            adapter = BFCLAdapter(args.data_path or "./data/raw/bfcl")
            adapter.load()
            samples = list(adapter)[:args.num_samples]
        
        print(f"Loaded {len(samples)} samples")
        
        # 查找 Stage 1 检查点
        stage1_checkpoints = None
        if args.stage1_dir:
            stage1_dir = Path(args.stage1_dir)
            stage1_checkpoints = {}
            for layer in args.layers:
                path = stage1_dir / f"stage1_layer{layer}_final.pt"
                if path.exists():
                    stage1_checkpoints[layer] = str(path)
        
        # 流式训练
        streaming_config = StreamingConfig(
            buffer_size=args.buffer_size,
            balance=args.balance,
            layers=args.layers,
            device=args.device,
        )
        
        train_sae_streaming(
            model_name=args.model,
            samples=samples,
            layers=args.layers,
            output_dir=args.output_dir,
            streaming_config=streaming_config,
            stage1_checkpoints=stage1_checkpoints,
            device=args.device,
        )
    
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
- train_sae_streaming: 流式训练 SAE (Stage 2)

磁盘空间节省：
------------
由于不保存 hidden states，相比传统方法可节省大量磁盘空间。
例如：50k episodes * 10 steps * 20 tokens * 4096 dim * 4 bytes ≈ 160GB
使用流式模式，这些数据不会被写入磁盘。
        """)


if __name__ == "__main__":
    main()
