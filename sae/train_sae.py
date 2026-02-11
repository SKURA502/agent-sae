"""
Train SAE - SAE 训练脚本

支持两阶段训练：
- Stage 1: 通用预训练语料（OpenWebText2）激活
- Stage 2: Tool-use 任务激活

支持运行时推理流式训练，避免保存 hidden states 到磁盘。
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import sys
sys.path.append(str(Path(__file__).parent.parent))

from .sae_model import TopKSAE, SAEConfig


@dataclass
class TrainingConfig:
    """训练配置"""
    # 模型配置
    input_dim: int = 4096
    dict_size: int = 32768
    k: int = 128
    
    # 训练配置
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 4096
    num_epochs: int = 10
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # 数据配置
    val_ratio: float = 0.1
    
    # 日志配置
    log_interval: int = 100
    eval_interval: int = 500
    save_interval: int = 1000
    
    # 输出配置
    output_dir: str = "./outputs/sae_checkpoints"
    experiment_name: str = "sae_training"
    
    # WandB 配置
    use_wandb: bool = False
    wandb_project: str = "agent-sae-tooluse"
    wandb_entity: Optional[str] = None
    
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
        
        # 初始化模型
        sae_config = config.to_sae_config()
        self.model = TopKSAE(sae_config)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # 训练状态
        self.global_step = 0
        self.best_val_loss = float("inf")
        
        # WandB
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.experiment_name,
                config=vars(config),
            )
    
    def train(
        self,
        train_data: torch.Tensor,
        val_data: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """训练 SAE
        
        Args:
            train_data: [num_samples, input_dim] 训练数据
            val_data: [num_val_samples, input_dim] 验证数据（可选）
            labels: [num_samples] 标签（用于监控，可选）
            
        Returns:
            训练统计信息
        """
        print(f"Training SAE with {len(train_data)} samples")
        print(f"Model config: dict_size={self.config.dict_size}, k={self.config.k}")
        
        # 准备验证集
        if val_data is None:
            val_size = int(len(train_data) * self.config.val_ratio)
            indices = torch.randperm(len(train_data))
            val_indices = indices[:val_size]
            train_indices = indices[val_size:]
            
            val_data = train_data[val_indices]
            train_data = train_data[train_indices]
            
            if labels is not None:
                val_labels = labels[val_indices]
                train_labels = labels[train_indices]
            else:
                val_labels = None
                train_labels = None
        else:
            train_labels = labels
            val_labels = None
        
        # 创建 DataLoader
        train_dataset = TensorDataset(train_data)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        
        # 学习率调度器
        num_training_steps = len(train_loader) * self.config.num_epochs
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
        scheduler = self._get_scheduler(num_training_steps, num_warmup_steps)
        
        # 训练循环
        training_stats = {
            "train_losses": [],
            "val_losses": [],
            "sparsity": [],
        }
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = self._train_epoch(
                train_loader,
                scheduler,
                epoch,
            )
            
            training_stats["train_losses"].append(epoch_loss)
            
            # 验证
            if val_data is not None:
                val_loss = self._evaluate(val_data)
                training_stats["val_losses"].append(val_loss)
                
                print(f"Epoch {epoch+1}/{self.config.num_epochs}: "
                      f"train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}")
                
                # 保存最佳模型
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint("best")
            else:
                print(f"Epoch {epoch+1}/{self.config.num_epochs}: "
                      f"train_loss={epoch_loss:.4f}")
        
        # 保存最终模型
        self._save_checkpoint("final")
        
        # 保存训练统计
        stats_path = self.output_dir / f"{self.config.experiment_name}_stats.json"
        with open(stats_path, "w") as f:
            json.dump(training_stats, f, indent=2)
        
        return training_stats
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        scheduler: Any,
        epoch: int,
    ) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (batch,) in enumerate(pbar):
            batch = batch.to(self.config.device)
            
            # 前向传播
            loss, loss_dict = self.model.compute_loss(batch, return_components=True)
            
            # 反向传播
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                
                self.optimizer.step()
                scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "sparsity": f"{loss_dict['sparsity']:.2%}",
            })
            
            # 日志
            if self.global_step % self.config.log_interval == 0:
                if self.config.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/reconstruction_loss": loss_dict["reconstruction_loss"],
                        "train/sparsity": loss_dict["sparsity"],
                        "train/lr": scheduler.get_last_lr()[0],
                        "global_step": self.global_step,
                    })
            
            # 保存检查点
            if self.global_step % self.config.save_interval == 0:
                self._save_checkpoint(f"step_{self.global_step}")
        
        return total_loss / num_batches
    
    def _evaluate(self, val_data: torch.Tensor) -> float:
        """评估"""
        self.model.eval()
        
        with torch.no_grad():
            val_data = val_data.to(self.config.device)
            
            # 分批评估
            batch_size = self.config.batch_size
            total_loss = 0
            num_batches = 0
            
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i+batch_size]
                loss = self.model.compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "val/loss": avg_loss,
                "global_step": self.global_step,
            })
        
        return avg_loss
    
    def _get_scheduler(
        self,
        num_training_steps: int,
        num_warmup_steps: int,
    ):
        """获取学习率调度器"""
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step) /
                float(max(1, num_training_steps - num_warmup_steps)),
            )
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def _save_checkpoint(self, name: str):
        """保存检查点"""
        checkpoint_path = self.output_dir / f"{self.config.experiment_name}_{name}.pt"
        self.model.save(str(checkpoint_path))
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """从检查点加载模型
        
        Args:
            checkpoint_path: 检查点路径
        """
        print(f"Loading checkpoint from {checkpoint_path}")
        self.model = TopKSAE.load(checkpoint_path, device=self.config.device)
        
        # 重新创建优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        print("Checkpoint loaded successfully")
    
    def train_streaming(
        self,
        activation_generator: Generator[Dict[int, torch.Tensor], None, None],
        layer: int,
        total_steps: Optional[int] = None,
        val_data: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """流式训练 SAE
        
        从激活生成器中流式获取数据进行训练，适用于运行时推理模式。
        
        Args:
            activation_generator: 激活生成器，yield {layer_idx: [batch, hidden_dim]}
            layer: 要训练的层
            total_steps: 总训练步数估计（用于学习率调度）
            val_data: 验证数据
            
        Returns:
            训练统计信息
        """
        print(f"Starting streaming training for layer {layer}")
        print(f"Model config: dict_size={self.config.dict_size}, k={self.config.k}")
        
        # 估计总步数
        if total_steps is None:
            total_steps = 10000  # 默认值
        
        # 学习率调度器
        num_warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = self._get_scheduler(total_steps, num_warmup_steps)
        
        # 训练统计
        training_stats = {
            "train_losses": [],
            "val_losses": [],
            "sparsity": [],
            "steps": [],
        }
        
        self.model.train()
        running_loss = 0
        num_batches = 0
        
        pbar = tqdm(total=total_steps, desc="Streaming training")
        
        for activations in activation_generator:
            if layer not in activations:
                continue
            
            batch_data = activations[layer]
            
            # 如果是 3D，展平
            if len(batch_data.shape) == 3:
                batch_data = batch_data.view(-1, batch_data.shape[-1])
            
            # 分批训练
            for i in range(0, len(batch_data), self.config.batch_size):
                batch = batch_data[i:i+self.config.batch_size].to(self.config.device)
                
                # 前向传播
                loss, loss_dict = self.model.compute_loss(batch, return_components=True)
                
                # 反向传播
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                if (num_batches + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                    
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                
                running_loss += loss.item() * self.config.gradient_accumulation_steps
                num_batches += 1
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "sparsity": f"{loss_dict['sparsity']:.2%}",
                })
                
                # 日志
                if self.global_step % self.config.log_interval == 0:
                    avg_loss = running_loss / num_batches
                    training_stats["train_losses"].append(avg_loss)
                    training_stats["steps"].append(self.global_step)
                    
                    if self.config.use_wandb and WANDB_AVAILABLE:
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/sparsity": loss_dict["sparsity"],
                            "train/lr": scheduler.get_last_lr()[0],
                            "global_step": self.global_step,
                        })
                    
                    running_loss = 0
                    num_batches = 0
                
                # 验证
                if val_data is not None and self.global_step % self.config.eval_interval == 0:
                    val_loss = self._evaluate(val_data)
                    training_stats["val_losses"].append(val_loss)
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint("best")
                
                # 保存检查点
                if self.global_step % self.config.save_interval == 0:
                    self._save_checkpoint(f"step_{self.global_step}")
        
        pbar.close()
        
        # 保存最终模型
        self._save_checkpoint("final")
        
        # 保存训练统计
        stats_path = self.output_dir / f"{self.config.experiment_name}_stats.json"
        with open(stats_path, "w") as f:
            json.dump(training_stats, f, indent=2)
        
        return training_stats


class TwoStageTrainer:
    """两阶段 SAE 训练器
    
    Stage 1: 通用预训练语料激活
    Stage 2: Tool-use 任务激活
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        layers: List[int],
        output_dir: str = "./outputs/sae_checkpoints",
        device: str = "cuda",
        dtype: str = "bfloat16",
    ):
        """
        Args:
            model_name_or_path: LLM 模型路径
            layers: 要训练 SAE 的层
            output_dir: 输出目录
            device: 设备
            dtype: 数据类型
        """
        self.model_name_or_path = model_name_or_path
        self.layers = layers
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.dtype = dtype
        
        # LLM 模型（延迟加载）
        self.model = None
        self.tokenizer = None
        
        # SAE 训练器（每层一个）
        self.sae_trainers: Dict[int, SAETrainer] = {}
    
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
        
        # 获取 hidden size
        if hasattr(self.model.config, 'hidden_size'):
            self.hidden_size = self.model.config.hidden_size
        elif hasattr(self.model.config, 'd_model'):
            self.hidden_size = self.model.config.d_model
        else:
            self.hidden_size = 4096  # 默认值
        
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
        
        # 默认配置
        pretrain_config = pretrain_config or {}
        sae_config = sae_config or {}
        
        pt_config = PretrainConfig(
            dataset_name=pretrain_config.get("dataset", "Skylion007/openwebtext"),
            target_tokens=pretrain_config.get("target_tokens", 50_000_000),
            seq_length=pretrain_config.get("seq_length", 1024),
            sample_position=pretrain_config.get("sample_position", "random"),
            positions_per_seq=pretrain_config.get("positions_per_seq", 8),
        )
        
        # 计算字典大小
        dict_size = sae_config.get("dict_size", self.hidden_size * 8)
        k = sae_config.get("k", self.hidden_size // 32)
        
        print(f"Stage 1: Training SAE on pretrain corpus")
        print(f"  Target tokens: {pt_config.target_tokens:,}")
        print(f"  Dict size: {dict_size}, K: {k}")
        
        checkpoint_paths = {}
        
        for layer in self.layers:
            print(f"\n=== Training layer {layer} ===")
            
            # 创建 SAE 训练器
            train_config = TrainingConfig(
                input_dim=self.hidden_size,
                dict_size=dict_size,
                k=k,
                learning_rate=sae_config.get("learning_rate", 1e-4),
                batch_size=sae_config.get("batch_size", 4096),
                log_interval=sae_config.get("log_interval", 100),
                save_interval=sae_config.get("save_interval", 1000),
                output_dir=str(self.output_dir / "stage1"),
                experiment_name=f"stage1_layer{layer}",
                device=self.device,
            )
            
            trainer = SAETrainer(train_config)
            self.sae_trainers[layer] = trainer
            
            # 创建激活生成器
            activation_gen = create_pretrain_data_iterator(
                model=self.model,
                tokenizer=self.tokenizer,
                config=pt_config,
                layers=[layer],
                batch_size=pretrain_config.get("batch_size", 32),
                buffer_size=sae_config.get("buffer_size", 8192),
                device=self.device,
            )
            
            # 估计总步数
            estimated_samples = pt_config.target_tokens // pt_config.seq_length * pt_config.positions_per_seq
            total_steps = estimated_samples // train_config.batch_size
            
            # 流式训练
            trainer.train_streaming(activation_gen, layer, total_steps)
            
            # 记录检查点路径
            checkpoint_paths[layer] = str(
                self.output_dir / "stage1" / f"stage1_layer{layer}_final.pt"
            )
        
        print("\nStage 1 training complete!")
        return checkpoint_paths
    
    def train_stage2(
        self,
        stage1_checkpoints: Dict[int, str],
        tooluse_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[int, str]:
        """第二阶段训练：Tool-use 任务激活
        
        Args:
            stage1_checkpoints: Stage 1 检查点路径
            tooluse_config: Tool-use 训练配置
            
        Returns:
            {layer: checkpoint_path} 每层的检查点路径
        """
        self._load_llm()
        
        tooluse_config = tooluse_config or {}
        
        print("Stage 2: Training SAE on tool-use data")
        
        checkpoint_paths = {}
        
        for layer in self.layers:
            print(f"\n=== Training layer {layer} ===")
            
            # 从 Stage 1 加载检查点
            stage1_path = stage1_checkpoints.get(layer)
            if stage1_path and Path(stage1_path).exists():
                print(f"Loading Stage 1 checkpoint: {stage1_path}")
                sae_model = TopKSAE.load(stage1_path, device=self.device)
            else:
                print(f"Warning: No Stage 1 checkpoint for layer {layer}, training from scratch")
                sae_model = None
            
            # 创建训练配置
            train_config = TrainingConfig(
                input_dim=self.hidden_size,
                dict_size=sae_model.config.dict_size if sae_model else self.hidden_size * 8,
                k=sae_model.config.k if sae_model else self.hidden_size // 32,
                learning_rate=tooluse_config.get("learning_rate", 5e-5),
                batch_size=tooluse_config.get("batch_size", 4096),
                num_epochs=tooluse_config.get("num_epochs", 10),
                log_interval=tooluse_config.get("log_interval", 100),
                save_interval=tooluse_config.get("save_interval", 1000),
                output_dir=str(self.output_dir / "stage2"),
                experiment_name=f"stage2_layer{layer}",
                device=self.device,
            )
            
            trainer = SAETrainer(train_config)
            
            # 如果有 Stage 1 模型，加载它
            if sae_model is not None:
                trainer.model = sae_model
                trainer.optimizer = torch.optim.AdamW(
                    trainer.model.parameters(),
                    lr=train_config.learning_rate,
                    weight_decay=train_config.weight_decay,
                )
            
            self.sae_trainers[layer] = trainer
            
            # 记录检查点路径（Stage 2 训练由外部提供数据）
            checkpoint_paths[layer] = str(
                self.output_dir / "stage2" / f"stage2_layer{layer}_final.pt"
            )
        
        return checkpoint_paths
    
    def train_stage2_streaming(
        self,
        stage1_checkpoints: Dict[int, str],
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
        # 先初始化 Stage 2 训练器
        checkpoint_paths = self.train_stage2(stage1_checkpoints, tooluse_config)
        
        # 流式训练每层
        for layer in self.layers:
            trainer = self.sae_trainers[layer]
            
            # 需要为每层重新创建生成器或使用共享数据
            # 这里假设 activation_generator 包含所有层的数据
            trainer.train_streaming(activation_generator, layer, total_steps)
        
        return checkpoint_paths


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="Train SAE")
    subparsers = parser.add_subparsers(dest="command", help="训练模式")
    
    # 传统模式：从文件加载数据训练
    legacy_parser = subparsers.add_parser("legacy", help="从缓存文件训练（传统模式）")
    legacy_parser.add_argument("--data-path", type=str, required=True,
                               help="Path to activation data (.pt file)")
    legacy_parser.add_argument("--layer", type=int, required=True,
                               help="Layer to train on")
    legacy_parser.add_argument("--output-dir", type=str, default="./outputs/sae_checkpoints",
                               help="Output directory")
    legacy_parser.add_argument("--dict-size", type=int, default=32768,
                               help="Dictionary size")
    legacy_parser.add_argument("--k", type=int, default=128,
                               help="TopK value")
    legacy_parser.add_argument("--batch-size", type=int, default=4096,
                               help="Batch size")
    legacy_parser.add_argument("--num-epochs", type=int, default=10,
                               help="Number of epochs")
    legacy_parser.add_argument("--learning-rate", type=float, default=1e-4,
                               help="Learning rate")
    legacy_parser.add_argument("--experiment-name", type=str, default="sae",
                               help="Experiment name")
    legacy_parser.add_argument("--use-wandb", action="store_true",
                               help="Use WandB for logging")
    legacy_parser.add_argument("--device", type=str, default="cuda",
                               help="Device")
    
    # Stage 1：预训练语料
    stage1_parser = subparsers.add_parser("stage1", help="Stage 1: 预训练语料训练")
    stage1_parser.add_argument("--model", type=str, required=True,
                               help="LLM model name or path")
    stage1_parser.add_argument("--layers", type=int, nargs="+", default=[24, 27],
                               help="Layers to train SAE on")
    stage1_parser.add_argument("--output-dir", type=str, default="./outputs/sae_checkpoints",
                               help="Output directory")
    stage1_parser.add_argument("--target-tokens", type=int, default=50_000_000,
                               help="Target number of tokens (50-100M recommended)")
    stage1_parser.add_argument("--seq-length", type=int, default=1024,
                               help="Sequence length")
    stage1_parser.add_argument("--batch-size", type=int, default=32,
                               help="Inference batch size")
    stage1_parser.add_argument("--sae-batch-size", type=int, default=4096,
                               help="SAE training batch size")
    stage1_parser.add_argument("--learning-rate", type=float, default=1e-4,
                               help="Learning rate")
    stage1_parser.add_argument("--dataset", type=str, default="Skylion007/openwebtext",
                               help="Pretrain dataset (HuggingFace)")
    stage1_parser.add_argument("--device", type=str, default="cuda",
                               help="Device")
    stage1_parser.add_argument("--dtype", type=str, default="bfloat16",
                               help="Model dtype")
    
    # Stage 2：Tool-use 数据（从 Stage 1 继续）
    stage2_parser = subparsers.add_parser("stage2", help="Stage 2: Tool-use 数据训练")
    stage2_parser.add_argument("--model", type=str, required=True,
                               help="LLM model name or path")
    stage2_parser.add_argument("--stage1-dir", type=str, required=True,
                               help="Stage 1 checkpoint directory")
    stage2_parser.add_argument("--layers", type=int, nargs="+", default=[24, 27],
                               help="Layers to train SAE on")
    stage2_parser.add_argument("--output-dir", type=str, default="./outputs/sae_checkpoints",
                               help="Output directory")
    stage2_parser.add_argument("--learning-rate", type=float, default=5e-5,
                               help="Learning rate (usually smaller than stage1)")
    stage2_parser.add_argument("--num-epochs", type=int, default=10,
                               help="Number of epochs")
    stage2_parser.add_argument("--batch-size", type=int, default=4096,
                               help="SAE training batch size")
    stage2_parser.add_argument("--device", type=str, default="cuda",
                               help="Device")
    stage2_parser.add_argument("--dtype", type=str, default="bfloat16",
                               help="Model dtype")
    
    args = parser.parse_args()
    
    if args.command == "legacy" or args.command is None:
        # 兼容传统模式
        if not hasattr(args, 'data_path') or args.data_path is None:
            parser.print_help()
            return
        
        # 加载数据
        print(f"Loading data from {args.data_path}")
        data = torch.load(args.data_path)
        
        layer_key = f"layer_{args.layer}"
        if layer_key not in data:
            raise ValueError(f"Layer {args.layer} not found in data. "
                            f"Available: {list(data.keys())}")
        
        activations = data[layer_key]
        labels = data.get("labels")
        
        print(f"Loaded {len(activations)} samples with shape {activations.shape}")
        
        # 创建配置
        config = TrainingConfig(
            input_dim=activations.shape[-1],
            dict_size=args.dict_size,
            k=args.k,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir,
            experiment_name=f"{args.experiment_name}_layer{args.layer}",
            use_wandb=args.use_wandb,
            device=args.device,
        )
        
        # 训练
        trainer = SAETrainer(config)
        
        # 如果数据是 3D [batch, seq, hidden]，展平为 2D
        if len(activations.shape) == 3:
            activations = activations.view(-1, activations.shape[-1])
        
        stats = trainer.train(activations, labels=labels)
        
        print("Training complete!")
        print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    
    elif args.command == "stage1":
        # Stage 1: 预训练语料
        trainer = TwoStageTrainer(
            model_name_or_path=args.model,
            layers=args.layers,
            output_dir=args.output_dir,
            device=args.device,
            dtype=args.dtype,
        )
        
        pretrain_config = {
            "dataset": args.dataset,
            "target_tokens": args.target_tokens,
            "seq_length": args.seq_length,
            "batch_size": args.batch_size,
        }
        
        sae_config = {
            "batch_size": args.sae_batch_size,
            "learning_rate": args.learning_rate,
        }
        
        checkpoints = trainer.train_stage1(pretrain_config, sae_config)
        
        print("\nStage 1 complete! Checkpoints:")
        for layer, path in checkpoints.items():
            print(f"  Layer {layer}: {path}")
    
    elif args.command == "stage2":
        # Stage 2: Tool-use 数据
        # 查找 Stage 1 检查点
        stage1_dir = Path(args.stage1_dir)
        stage1_checkpoints = {}
        for layer in args.layers:
            checkpoint_pattern = f"stage1_layer{layer}_final.pt"
            checkpoint_path = stage1_dir / checkpoint_pattern
            if checkpoint_path.exists():
                stage1_checkpoints[layer] = str(checkpoint_path)
            else:
                print(f"Warning: Stage 1 checkpoint not found for layer {layer}")
        
        trainer = TwoStageTrainer(
            model_name_or_path=args.model,
            layers=args.layers,
            output_dir=args.output_dir,
            device=args.device,
            dtype=args.dtype,
        )
        
        tooluse_config = {
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
        }
        
        checkpoints = trainer.train_stage2(stage1_checkpoints, tooluse_config)
        
        print("\nStage 2 initialized! Ready for tool-use data.")
        print("Use the streaming API to provide tool-use activations.")
        for layer, path in checkpoints.items():
            print(f"  Layer {layer}: {path}")


if __name__ == "__main__":
    main()
