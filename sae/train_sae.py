"""
Train SAE - SAE 训练脚本

支持多种训练配置和监控。
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="Train SAE")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to activation data (.pt file)")
    parser.add_argument("--layer", type=int, required=True,
                        help="Layer to train on")
    parser.add_argument("--output-dir", type=str, default="./outputs/sae_checkpoints",
                        help="Output directory")
    parser.add_argument("--dict-size", type=int, default=32768,
                        help="Dictionary size")
    parser.add_argument("--k", type=int, default=128,
                        help="TopK value")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--experiment-name", type=str, default="sae",
                        help="Experiment name")
    parser.add_argument("--use-wandb", action="store_true",
                        help="Use WandB for logging")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device")
    
    args = parser.parse_args()
    
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


if __name__ == "__main__":
    main()
