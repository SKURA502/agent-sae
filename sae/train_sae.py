"""
Train SAE - SAE 训练脚本

两阶段训练：
- Stage 1: 通用预训练语料（本地 JSONL）激活
- Stage 2: Tool-use 任务激活

支持运行时推理流式训练，避免保存 hidden states 到磁盘。
使用 SwanLab 记录训练指标。
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False

# 项目根目录（相对于本文件: sae/ -> Agent-Tool-Use-MI/）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

from .sae_model import TopKSAE, SAEConfig


def _make_checkpoint_name(
    model_name: str,
    layer: int,
    dict_size: int,
    target_tokens: int,
    stage: str,
) -> str:
    """生成清晰的检查点文件名。

    格式: {LLM}-layer{L}-d{dict_size}-{tokens_short}-{stage}.pt
    例如: Llama-3-8B-Instruct-layer24-d32768-100M-stage1.pt
    """
    # 从模型路径中提取简短名称
    short_name = model_name.rstrip("/").split("/")[-1]

    # 将 token 数转为可读字符串
    if target_tokens >= 1_000_000_000:
        tok_str = f"{target_tokens / 1e9:.0f}B"
    elif target_tokens >= 1_000_000:
        tok_str = f"{target_tokens / 1e6:.0f}M"
    elif target_tokens >= 1_000:
        tok_str = f"{target_tokens / 1e3:.0f}K"
    else:
        tok_str = str(target_tokens)

    return f"{short_name}-layer{layer}-d{int(dict_size)}-{tok_str}-{stage}.pt"


@dataclass
class TrainingConfig:
    """训练配置"""
    # 模型配置
    input_dim: int = 4096
    dict_size: int = 32768
    k: int = 128

    # 训练配置
    learning_rate: float = 1e-5
    batch_size: int = 4096
    num_epochs: int = 1
    warmup_ratio: float = 0.1

    # Decoder unit-norm 间隔（每 N 步做一次 decoder 列归一化）
    decoder_norm_interval: int = 10

    # 日志配置
    log_interval: int = 10

    # 输出配置
    output_dir: str = "./outputs/sae_checkpoints"
    experiment_name: str = "sae_training"

    # SwanLab 配置
    use_swanlab: bool = False
    swanlab_project: str = "agent-tool-use"

    # 设备配置
    device: str = "cuda"
    dtype: str = "float32"

    def to_sae_config(self) -> SAEConfig:
        """转换为 SAE 配置"""
        return SAEConfig(
            input_dim=self.input_dim,
            dict_size=self.dict_size,
            k=self.k,
            device=self.device,
            dtype=self.dtype,
        )


class SAETrainer:
    """SAE 训练器"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._swanlab_active = False

        # 初始化模型
        sae_config = config.to_sae_config()
        self.model = TopKSAE(sae_config)

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
        )

        # 训练状态
        self.global_step = 0

        # SwanLab
        if config.use_swanlab and SWANLAB_AVAILABLE:
            swanlab.init(
                project=config.swanlab_project,
                experiment_name=config.experiment_name,
                config=vars(config),
                mode="offline",
                logdir="swanlog",
            )
            self._swanlab_active = True

    def _finish_swanlab(self):
        if self._swanlab_active and SWANLAB_AVAILABLE:
            swanlab.finish()
            self._swanlab_active = False

    def train(
        self,
        train_data: torch.Tensor,
    ) -> Dict[str, Any]:
        """训练 SAE

        Args:
            train_data: [num_samples, input_dim] 训练数据

        Returns:
            训练统计信息
        """
        print(f"Training SAE with {len(train_data)} samples")
        print(f"Model config: dict_size={self.config.dict_size}, k={self.config.k}")

        # 创建 DataLoader
        train_dataset = TensorDataset(train_data)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        # 学习率调度器
        num_training_steps = len(train_loader) * self.config.num_epochs
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        scheduler = self._get_scheduler(num_training_steps, num_warmup_steps)

        # 训练循环
        training_stats: Dict[str, list] = {
            "train_losses": [],
        }

        try:
            for epoch in range(self.config.num_epochs):
                epoch_loss = self._train_epoch(train_loader, scheduler, epoch)
                training_stats["train_losses"].append(epoch_loss)
                print(
                    f"Epoch {epoch+1}/{self.config.num_epochs}: "
                    f"train_loss={epoch_loss:.4f}"
                )

            # 保存训练统计
            stats_path = self.output_dir / f"{self.config.experiment_name}_stats.json"
            with open(stats_path, "w") as f:
                json.dump(training_stats, f, indent=2)

            return training_stats
        finally:
            self._finish_swanlab()

    def _train_epoch(
        self,
        train_loader: DataLoader,
        scheduler: Any,
        epoch: int,
    ) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch_idx, (batch,) in enumerate(pbar):
            metrics = self.train_batch(batch, scheduler)

            total_loss += metrics["loss"]
            num_batches += 1

            # 更新进度条
            pbar.set_postfix({"loss": f"{metrics['loss']:.4f}"})

            # 日志
            if self.global_step % self.config.log_interval == 0:
                log_data = {
                    "train/loss": metrics["loss"],
                    "train/reconstruction_loss": metrics["reconstruction_loss"],
                    "train/mean_activation": metrics["mean_activation"],
                    "train/lr": metrics["lr"],
                    "global_step": self.global_step,
                }
                if self.config.use_swanlab and SWANLAB_AVAILABLE:
                    swanlab.log(log_data)
                else:
                    tqdm.write(
                        f"[Step {self.global_step}] "
                        f"loss={metrics['loss']:.4f} "
                        f"recon={metrics['reconstruction_loss']:.4f} "
                        f"mean_act={metrics['mean_activation']:.4f} "
                        f"lr={metrics['lr']:.2e}"
                    )

        return total_loss / max(num_batches, 1)

    def _get_scheduler(
        self,
        num_training_steps: int,
        num_warmup_steps: int,
    ):
        """获取学习率调度器（线性 warmup + 线性衰减）"""
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step)
                / float(max(1, num_training_steps - num_warmup_steps)),
            )

        return LambdaLR(self.optimizer, lr_lambda)

    def save_checkpoint(self, name: str):
        """保存检查点"""
        checkpoint_path = self.output_dir / name
        self.model.save(str(checkpoint_path))
        print(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """从检查点加载模型"""
        print(f"Loading checkpoint from {checkpoint_path}")
        self.model = TopKSAE.load(checkpoint_path, device=self.config.device)

        # 重新创建优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )
        print("Checkpoint loaded successfully")

    def train_batch(
        self,
        batch: torch.Tensor,
        scheduler: Any,
    ) -> Dict[str, float]:
        """执行单个 batch 训练步骤。"""
        batch = batch.to(self.config.device)
        self.optimizer.zero_grad(set_to_none=True)

        loss, loss_dict = self.model.compute_loss(batch)
        loss.backward()

        self.optimizer.step()
        scheduler.step()

        self.global_step += 1

        if self.global_step % self.config.decoder_norm_interval == 0:
            self.model._normalize_decoder()

        return {
            "loss": float(loss.item()),
            "reconstruction_loss": float(loss_dict["loss"]),
            "mean_activation": float(loss_dict["mean_activation"]),
            "lr": float(scheduler.get_last_lr()[0]),
            "global_step": float(self.global_step),
        }

    def train_streaming(
        self,
        activation_generator: Generator[Dict[int, torch.Tensor], None, None],
        layer: int,
        total_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """流式训练 SAE

        从激活生成器中流式获取数据进行训练，适用于运行时推理模式。

        Args:
            activation_generator: 激活生成器，yield {layer_idx: [batch, hidden_dim]}
            layer: 要训练的层
            total_steps: 总训练步数估计（用于学习率调度）

        Returns:
            训练统计信息
        """
        print(f"Starting streaming training for layer {layer}")
        print(f"Model config: dict_size={self.config.dict_size}, k={self.config.k}")

        if total_steps is None:
            total_steps = 10000

        # 学习率调度器
        num_warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = self._get_scheduler(total_steps, num_warmup_steps)

        training_stats: Dict[str, list] = {
            "interval_avg_losses": [],
            "steps": [],
        }

        self.model.train()
        running_loss = 0.0
        num_batches = 0

        pbar = tqdm(total=total_steps, desc="Streaming training")

        try:
            for activations in activation_generator:
                if layer not in activations:
                    continue

                batch_data = activations[layer]

                # 如果是 3D，展平
                if len(batch_data.shape) == 3:
                    batch_data = batch_data.view(-1, batch_data.shape[-1])

                # 分批训练
                for i in range(0, len(batch_data), self.config.batch_size):
                    batch = batch_data[i : i + self.config.batch_size]
                    metrics = self.train_batch(batch, scheduler)

                    running_loss += metrics["loss"]
                    num_batches += 1

                    pbar.update(1)
                    pbar.set_postfix({"loss": f"{metrics['loss']:.4f}"})

                    # 日志
                    if self.global_step % self.config.log_interval == 0 and num_batches > 0:
                        avg_loss = running_loss / num_batches
                        training_stats["interval_avg_losses"].append(avg_loss)
                        training_stats["steps"].append(self.global_step)

                        log_data = {
                            "train/loss": avg_loss,
                            "train/mean_activation": metrics["mean_activation"],
                            "train/lr": metrics["lr"],
                            "global_step": self.global_step,
                        }
                        if self.config.use_swanlab and SWANLAB_AVAILABLE:
                            swanlab.log(log_data)
                        else:
                            tqdm.write(
                                f"[Step {self.global_step}] "
                                f"loss={avg_loss:.4f} "
                                f"mean_act={metrics['mean_activation']:.4f} "
                                f"lr={metrics['lr']:.2e}"
                            )

                        running_loss = 0.0
                        num_batches = 0

            # 保存训练统计
            stats_path = self.output_dir / f"{self.config.experiment_name}_stats.json"
            with open(stats_path, "w") as f:
                json.dump(training_stats, f, indent=2)

            return training_stats
        finally:
            pbar.close()
            self._finish_swanlab()


class TwoStageTrainer:
    """两阶段 SAE 训练器

    Stage 1: 通用预训练语料激活
    Stage 2: Tool-use 任务激活
    """

    def __init__(
        self,
        model_name_or_path: str,
        layer: int,
        output_dir: str = "./outputs/sae_checkpoints",
        device: str = "cuda",
        dtype: str = "float32",
    ):
        """
        Args:
            model_name_or_path: LLM 模型路径
            layer: 要训练 SAE 的目标层
            output_dir: 输出目录
            device: 设备
            dtype: 数据类型
        """
        self.model_name_or_path = model_name_or_path
        self.layer = int(layer)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.dtype = dtype

        # LLM 模型（延迟加载）
        self.model = None
        self.tokenizer = None

        self.sae_trainer: Optional[SAETrainer] = None

    def _infer_hidden_size(self) -> int:
        """推断 LLM 文本 hidden size（优先 text_config，兼容多模态模型）。"""
        config = getattr(self.model, "config", None)

        if config is not None:
            text_config = getattr(config, "text_config", None)
            if text_config is not None and hasattr(text_config, "hidden_size"):
                return int(text_config.hidden_size)

            for attr in ("hidden_size", "d_model", "n_embd"):
                if hasattr(config, attr):
                    value = getattr(config, attr)
                    if value is not None:
                        return int(value)

        # 结构兜底：尝试从输入 embedding 读取
        for emb_path in (
            "model.embed_tokens",
            "language_model.model.embed_tokens",
            "model.language_model.model.embed_tokens",
            "transformer.wte",
        ):
            cur = self.model
            ok = True
            for part in emb_path.split("."):
                if not hasattr(cur, part):
                    ok = False
                    break
                cur = getattr(cur, part)
            if ok and hasattr(cur, "embedding_dim"):
                return int(cur.embedding_dim)

        return 4096

    def _load_llm(self):
        """加载 LLM 模型"""
        if self.model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading LLM: {self.model_name_or_path}")

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=dtype_map.get(self.dtype, torch.bfloat16),
            device_map=self.device,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 获取 hidden size（兼容多模态模型，如 Gemma3）
        self.hidden_size = self._infer_hidden_size()

        print(f"LLM loaded. Hidden size: {self.hidden_size}")

    def train_stage1(
        self,
        pretrain_config: Optional[Dict[str, Any]] = None,
        sae_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[int, str]:
        """第一阶段训练：通用预训练语料

        Args:
            pretrain_config: 预训练数据配置
            sae_config: SAE 配置

        Returns:
            {layer: checkpoint_path} 每层的检查点路径
        """
        from .pretrain_data import PretrainConfig, create_pretrain_data_iterator

        self._load_llm()

        pretrain_config = pretrain_config or {}
        sae_config = sae_config or {}

        seq_length = pretrain_config.get("seq_length", 1024)
        sample_position = pretrain_config.get("sample_position", "all")
        # 当使用所有位置时，positions_per_seq 等于 seq_length
        positions_per_seq = (
            seq_length if sample_position == "all"
            else pretrain_config.get("positions_per_seq", 8)
        )

        pt_config = PretrainConfig(
            data_dir=pretrain_config.get(
                "data_dir",
                str(_PROJECT_ROOT / "data" / "raw" / "100M"),
            ),
            target_tokens=pretrain_config.get("target_tokens", 50_000_000),
            seq_length=seq_length,
            sample_position=sample_position,
            positions_per_seq=positions_per_seq,
        )

        # 计算字典大小（仅当传入值非 None 时覆盖默认）
        requested_dict_size = sae_config.get("dict_size")
        requested_k = sae_config.get("k")
        dict_size = int(requested_dict_size) if requested_dict_size is not None else self.hidden_size * 8
        k = int(requested_k) if requested_k is not None else self.hidden_size // 32
        target_tokens = pt_config.target_tokens

        print("Stage 1: Training SAE on pretrain corpus")
        print(f"  Target tokens: {target_tokens:,}")
        print(f"  Dict size: {dict_size}, K: {k}")

        layer = self.layer
        print(f"\n=== Training layer {layer} ===")

        ckpt_name = _make_checkpoint_name(
            self.model_name_or_path, layer, dict_size, target_tokens, "stage1"
        )

        train_config = TrainingConfig(
            input_dim=self.hidden_size,
            dict_size=dict_size,
            k=k,
            learning_rate=sae_config.get("learning_rate", 1e-5),
            batch_size=sae_config.get("batch_size", 4096),
            decoder_norm_interval=sae_config.get("decoder_norm_interval", 10),
            log_interval=sae_config.get("log_interval", 1),
            output_dir=str(self.output_dir / "stage1"),
            experiment_name=ckpt_name.replace(".pt", ""),
            use_swanlab=sae_config.get("use_swanlab", False),
            swanlab_project=sae_config.get("swanlab_project", "agent-tool-use"),
            device=self.device,
            dtype=self.dtype,
        )

        trainer = SAETrainer(train_config)
        self.sae_trainer = trainer

        if pt_config.sample_position == "all":
            estimated_samples = target_tokens
        else:
            estimated_samples = (
                target_tokens // pt_config.seq_length * pt_config.positions_per_seq
            )
        total_steps = max(estimated_samples // train_config.batch_size, 1)

        activation_gen = create_pretrain_data_iterator(
            model=self.model,
            tokenizer=self.tokenizer,
            config=pt_config,
            layers=[layer],
            batch_size=pretrain_config.get("batch_size", 32),
            buffer_size=sae_config.get("buffer_size", 8192),
            device=self.device,
        )

        trainer.train_streaming(activation_gen, layer, total_steps)
        trainer.save_checkpoint(ckpt_name)

        print("\nStage 1 training complete!")
        return {layer: str(self.output_dir / "stage1" / ckpt_name)}

    def init_stage2(
        self,
        stage1_checkpoint: Optional[str],
        tooluse_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[int, str]:
        """第二阶段初始化：Tool-use 任务激活训练器（由外部提供数据）

        Args:
            stage1_checkpoint: Stage 1 检查点路径
            tooluse_config: Tool-use 训练配置

        Returns:
            {layer: checkpoint_path} 每层的检查点路径（预期最终保存位置）
        """
        self._load_llm()

        tooluse_config = tooluse_config or {}
        target_tokens = tooluse_config.get("target_tokens", 50_000_000)

        print("Stage 2: Training SAE on tool-use data")

        layer = self.layer
        print(f"\n=== Training layer {layer} ===")

        requested_dict_size = tooluse_config.get("dict_size")
        requested_k = tooluse_config.get("k")
        if stage1_checkpoint and Path(stage1_checkpoint).exists():
            print(f"Loading Stage 1 checkpoint: {stage1_checkpoint}")
            sae_model = TopKSAE.load(stage1_checkpoint, device=self.device)
        else:
            print(
                f"Warning: No Stage 1 checkpoint for layer {layer}, "
                "training from scratch"
            )
            sae_model = None

        if sae_model:
            dict_size = sae_model.config.dict_size
            k_val = sae_model.config.k
            if requested_dict_size is not None and int(requested_dict_size) != int(dict_size):
                print(
                    f"Warning: stage1 checkpoint dict_size={dict_size}，"
                    f"忽略传入的 --dict-size={requested_dict_size}"
                )
            if requested_k is not None and int(requested_k) != int(k_val):
                print(
                    f"Warning: stage1 checkpoint k={k_val}，"
                    f"忽略传入的 --k={requested_k}"
                )
        else:
            dict_size = int(requested_dict_size) if requested_dict_size is not None else self.hidden_size * 8
            k_val = int(requested_k) if requested_k is not None else self.hidden_size // 32

        print(f"  Dict size: {dict_size}, K: {k_val}")

        ckpt_name = _make_checkpoint_name(
            self.model_name_or_path, layer, dict_size, target_tokens, "stage2"
        )

        train_config = TrainingConfig(
            input_dim=self.hidden_size,
            dict_size=dict_size,
            k=k_val,
            learning_rate=tooluse_config.get("learning_rate", 5e-5),
            batch_size=tooluse_config.get("batch_size", 4096),
            num_epochs=tooluse_config.get("num_epochs", 1),
            decoder_norm_interval=tooluse_config.get("decoder_norm_interval", 10),
            log_interval=tooluse_config.get("log_interval", 1),
            output_dir=str(self.output_dir / "stage2"),
            experiment_name=ckpt_name.replace(".pt", ""),
            use_swanlab=tooluse_config.get("use_swanlab", False),
            swanlab_project=tooluse_config.get("swanlab_project", "agent-tool-use"),
            device=self.device,
            dtype=self.dtype,
        )

        trainer = SAETrainer(train_config)

        if sae_model is not None:
            trainer.model = sae_model
            trainer.optimizer = torch.optim.AdamW(
                trainer.model.parameters(),
                lr=train_config.learning_rate,
            )

        self.sae_trainer = trainer

        return {layer: str(self.output_dir / "stage2" / ckpt_name)}

    def train_stage2(
        self,
        stage1_checkpoint: Optional[str],
        tooluse_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[int, str]:
        """兼容旧接口：等价于 init_stage2。"""
        return self.init_stage2(stage1_checkpoint, tooluse_config)

    def train_stage2_streaming(
        self,
        stage1_checkpoint: Optional[str],
        activation_generator: Generator[Dict[int, torch.Tensor], None, None],
        total_steps: Optional[int] = None,
        tooluse_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[int, str]:
        """第二阶段流式训练

        Args:
            stage1_checkpoints: Stage 1 检查点路径
            activation_generator: Tool-use 激活生成器
            total_steps: 总训练步数
            tooluse_config: 配置

        Returns:
            {layer: checkpoint_path}
        """
        checkpoint_paths = self.train_stage2(stage1_checkpoint, tooluse_config)
        tooluse_config = tooluse_config or {}
        target_tokens = tooluse_config.get("target_tokens", 50_000_000)

        layer = self.layer
        if self.sae_trainer is None:
            raise RuntimeError("Stage2 trainer is not initialized")

        trainer = self.sae_trainer
        trainer.train_streaming(activation_generator, layer, total_steps)

        dict_size = trainer.config.dict_size
        ckpt_name = _make_checkpoint_name(
            self.model_name_or_path, layer, dict_size, target_tokens, "stage2"
        )
        trainer.save_checkpoint(ckpt_name)

        return checkpoint_paths


def main():
    """命令行入口 — 仅保留两阶段训练"""
    parser = argparse.ArgumentParser(description="Train SAE (Two-Stage)")
    subparsers = parser.add_subparsers(dest="command", help="训练阶段")

    # ---- Stage 1 ----
    s1 = subparsers.add_parser("stage1", help="Stage 1: 预训练语料训练")
    s1.add_argument("--model", type=str, required=True, help="LLM model name or path")
    s1.add_argument("--layer", type=int, default=24,
                     help="Target layer to train SAE on")
    s1.add_argument("--output-dir", type=str, default="./outputs/sae_checkpoints")
    s1.add_argument("--target-tokens", type=int, default=50_000_000,
                     help="Target number of tokens (50-100M recommended)")
    s1.add_argument("--seq-length", type=int, default=1024, help="Context window size for LLM inference")
    s1.add_argument("--batch-size", type=int, default=32, help="Batch size for LLM inference forward pass")
    s1.add_argument("--sae-batch-size", type=int, default=4096,
                     help="Batch size for SAE training (number of token activations)")
    s1.add_argument("--learning-rate", type=float, default=1e-5)
    s1.add_argument("--dict-size", type=int, default=None,
                     help="SAE dictionary size (default: hidden_size * 8)")
    s1.add_argument("--k", type=int, default=None,
                     help="SAE top-k sparsity (default: hidden_size // 32)")
    s1.add_argument("--data-dir", type=str,
                     default=str(_PROJECT_ROOT / "data" / "raw" / "100M"),
                     help="本地 JSONL 数据目录")
    s1.add_argument("--decoder-norm-interval", type=int, default=10,
                     help="每 N 步进行一次 decoder unit norm")
    s1.add_argument("--use-swanlab", action="store_true", help="Use SwanLab for logging")
    s1.add_argument("--device", type=str, default="cuda")
    s1.add_argument("--dtype", type=str, default="float32")

    # ---- Stage 2 ----
    s2 = subparsers.add_parser("stage2", help="Stage 2: Tool-use 数据训练")
    s2.add_argument("--model", type=str, required=True, help="LLM model name or path")
    s2.add_argument("--stage1-dir", type=str, required=True,
                     help="Stage 1 checkpoint directory")
    s2.add_argument("--layer", type=int, default=24,
                    help="Target layer to train SAE on")
    s2.add_argument("--output-dir", type=str, default="./outputs/sae_checkpoints")
    s2.add_argument("--target-tokens", type=int, default=50_000_000)
    s2.add_argument("--learning-rate", type=float, default=5e-5,
                     help="Learning rate (usually smaller than stage1)")
    s2.add_argument("--dict-size", type=int, default=None,
                     help="SAE dictionary size (only used when no stage1 checkpoint)")
    s2.add_argument("--k", type=int, default=None,
                     help="SAE top-k sparsity (only used when no stage1 checkpoint)")
    s2.add_argument("--num-epochs", type=int, default=1)
    s2.add_argument("--batch-size", type=int, default=4096, help="Batch size for SAE training (number of token activations)")
    s2.add_argument("--decoder-norm-interval", type=int, default=10,
                     help="每 N 步进行一次 decoder unit norm")
    s2.add_argument("--use-swanlab", action="store_true", help="Use SwanLab for logging")
    s2.add_argument("--device", type=str, default="cuda")
    s2.add_argument("--dtype", type=str, default="float32")

    args = parser.parse_args()

    if args.command == "stage1":
        trainer = TwoStageTrainer(
            model_name_or_path=args.model,
            layer=args.layer,
            output_dir=args.output_dir,
            device=args.device,
            dtype=args.dtype,
        )

        pretrain_config = {
            "data_dir": args.data_dir,
            "target_tokens": args.target_tokens,
            "seq_length": args.seq_length,
            "batch_size": args.batch_size,
        }

        sae_config = {
            "batch_size": args.sae_batch_size,
            "learning_rate": args.learning_rate,
            "dict_size": args.dict_size,
            "k": args.k,
            "decoder_norm_interval": args.decoder_norm_interval,
            "use_swanlab": args.use_swanlab,
        }

        checkpoints = trainer.train_stage1(pretrain_config, sae_config)

        print("\nStage 1 complete! Checkpoints:")
        for layer, path in checkpoints.items():
            print(f"  Layer {layer}: {path}")

    elif args.command == "stage2":
        # 查找 Stage 1 检查点
        stage1_dir = Path(args.stage1_dir)
        stage1_checkpoint: Optional[str] = None
        matches = list(stage1_dir.glob(f"*-layer{args.layer}-*-stage1.pt"))
        if matches:
            stage1_checkpoint = str(matches[0])
        else:
            print(f"Warning: Stage 1 checkpoint not found for layer {args.layer}")

        trainer = TwoStageTrainer(
            model_name_or_path=args.model,
            layer=args.layer,
            output_dir=args.output_dir,
            device=args.device,
            dtype=args.dtype,
        )

        tooluse_config = {
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "dict_size": args.dict_size,
            "k": args.k,
            "target_tokens": args.target_tokens,
            "decoder_norm_interval": args.decoder_norm_interval,
            "use_swanlab": args.use_swanlab,
        }

        checkpoints = trainer.init_stage2(stage1_checkpoint, tooluse_config)

        print("\nStage 2 initialized! Ready for tool-use data.")
        print("Use the streaming API to provide tool-use activations.")
        for layer, path in checkpoints.items():
            print(f"  Layer {layer}: {path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()