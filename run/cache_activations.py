"""
Streaming Activations - 流式激活处理工具

提供运行时推理的激活流式处理，用于 SAE Stage 2 训练。
不保存 hidden states 到磁盘，而是实时流式传输到 SAE 训练。
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, Optional

import torch

from sae.pretrain_data import ActivationBuffer


@dataclass
class StreamingConfig:
    """流式处理配置"""
    # 缓冲区大小（累积多少样本后 yield）
    buffer_size: int = 8192
    # 目标层
    layer: int = 24
    # 设备
    device: str = "cuda"


def create_streaming_data_pipeline(
    rollout_generator,
    samples: list,
    config: StreamingConfig,
) -> Generator[Dict[int, torch.Tensor], None, None]:
    """创建流式数据管道

    将 rollout 生成器的输出转换为 SAE 训练所需的格式。

    Args:
        rollout_generator: RolloutGenerator 实例
        samples: 任务样本列表
        config: 流式处理配置

    Yields:
        {layer: [buffer_size, hidden_dim]} 当缓冲区满时
    """
    buffer = ActivationBuffer(
        buffer_size=config.buffer_size,
        layers=[config.layer],
    )

    for episode_log, activations in rollout_generator.run_streaming(samples):
        # 合并该 episode 的激活
        merged = {}
        if activations.get(config.layer):
            merged[config.layer] = torch.cat(activations[config.layer], dim=0)

        if merged:
            buffer.add(merged)

        # 缓冲区满了，yield 数据
        if buffer.is_ready():
            yield buffer.get_and_clear()

    # 清空剩余
    if buffer.current_size > 0:
        yield buffer.get_and_clear()


def _find_stage1_checkpoint(stage1_dir: str, layer: int) -> Optional[str]:
    """查找 Stage 1 检查点。"""
    checkpoint_dir = Path(stage1_dir)
    matches = sorted(checkpoint_dir.glob(f"*-layer{layer}-*-stage1.pt"))
    if matches:
        return str(matches[0])
    print(f"Warning: Stage 1 checkpoint not found for layer {layer}")
    return None


def _estimate_total_steps(num_samples: int, batch_size: int, buffer_size: int) -> int:
    """估算流式训练总步数，用于学习率调度。"""
    estimated_points = max(num_samples * 20, buffer_size)
    return max(1, estimated_points // max(batch_size, 1))


def main():
    """命令行入口 — 流式 SAE Stage 2 训练"""
    from utils import add_common_args, add_sae_args, add_dataset_args, load_samples

    parser = argparse.ArgumentParser(
        description="流式 SAE Stage 2 训练，不保存 hidden states 到磁盘"
    )
    add_common_args(parser)
    add_sae_args(parser)
    add_dataset_args(parser)
    parser.add_argument("--layer", type=int, default=24, help="Target layer")
    parser.add_argument("--output-dir", type=str,
                        default="./outputs/sae_checkpoints")
    parser.add_argument("--stage1-dir", type=str, required=True,
                        help="Stage 1 checkpoint directory")
    parser.add_argument("--buffer-size", type=int, default=8192)

    args = parser.parse_args()

    from run.generate_rollouts import RolloutGenerator
    from sae.train_sae import TwoStageTrainer

    samples = load_samples(args.dataset, args.data_path, args.num_samples)
    print(f"Loaded {len(samples)} samples")

    stage1_checkpoint = _find_stage1_checkpoint(args.stage1_dir, args.layer)

    trainer = TwoStageTrainer(
        model_name_or_path=args.model,
        layer=args.layer,
        output_dir=args.output_dir,
        device=args.device,
        dtype=args.dtype,
    )

    tooluse_config = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "target_tokens": args.target_tokens,
        "use_swanlab": args.use_swanlab,
    }
    trainer.init_stage2(stage1_checkpoint, tooluse_config)

    streaming_config = StreamingConfig(
        buffer_size=args.buffer_size,
        layer=args.layer,
        device=args.device,
    )

    rollout_gen = RolloutGenerator(
        model_name=args.model,
        output_dir=args.output_dir,
        cache_activations=True,
        hook_layers=[args.layer],
        device=args.device,
        dtype=args.dtype,
    )

    total_steps = _estimate_total_steps(
        args.num_samples, args.batch_size, args.buffer_size
    )

    print(f"\n=== Streaming Stage 2 training for layer {args.layer} ===")
    data_pipeline = create_streaming_data_pipeline(
        rollout_gen, samples, streaming_config
    )

    trainer.sae_trainer.train_streaming(
        data_pipeline, layer=args.layer, total_steps=total_steps
    )

    ckpt_name = f"{trainer.sae_trainer.config.experiment_name}.pt"
    trainer.sae_trainer.save_checkpoint(ckpt_name)

    print("\nStreaming training complete!")


if __name__ == "__main__":
    main()
