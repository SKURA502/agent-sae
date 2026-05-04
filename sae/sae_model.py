"""
SAE Model - TopK Sparse Autoencoder
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEConfig:
    """SAE 配置"""
    input_dim: int = 4096
    dict_size: int = 32768
    k: int = 128
    device: str = "cuda"
    dtype: str = "bfloat16"
    
    def get_torch_dtype(self) -> torch.dtype:
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.dtype, torch.bfloat16)       


class TopKSAE(nn.Module):
    """
    TopK Sparse Autoencoder    
    """
    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        
        self.decoder = nn.Linear(config.dict_size, config.input_dim, bias=False)
        self._normalize_decoder() 
        self.encoder = nn.Linear(config.input_dim, config.dict_size, bias=False)
        self.encoder.weight.data = self.decoder.weight.data.T.clone()
        self.pre_bias = nn.Parameter(torch.zeros(config.input_dim))

        self.to(config.device)
        self.to(config.get_torch_dtype())
    
    def _normalize_decoder(self):
        """归一化 decoder 权重（每列归一化）"""
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(
                self.decoder.weight.data, dim=0
            )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码：input -> latent activations
        Args:
            x: [batch, input_dim]
        Returns:
            latent: [batch, dict_size] 稀疏激活
        """
        centered_x = x - self.pre_bias
        pre_activation = self.encoder(centered_x)
        
        topk_values, topk_indices = torch.topk(
            pre_activation, k=self.config.k, dim=-1
        )
        
        latents = torch.zeros_like(pre_activation)
        latents.scatter_(-1, topk_indices, F.relu(topk_values))
        
        return latents
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """解码：latent -> reconstruction
        Args:
            latent: [batch, dict_size]
        Returns:
            reconstruction: [batch, input_dim]
        """
        return self.decoder(latents) + self.pre_bias 
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(x)
        x_hat = self.decode(latents)
        return x_hat, latents
    
    def compute_loss(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """计算损失
        Args:
            x: [batch, input_dim]
        Returns:
            (loss, loss_dict)
        """
        x_hat, latent = self.forward(x)
        loss = (((x_hat - x) ** 2).mean(dim=-1) / (x**2).mean(dim=-1)).mean()
        loss_dict = {
            "mean_activation": latent[latent > 0].mean().item() if (latent > 0).any() else 0,
        }
        return loss, loss_dict
        
    def get_feature_activations(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取特征激活及其索引
        Args:
            x: [batch, input_dim]
        Returns:
            values: [batch, k] TopK 激活值
            indices: [batch, k] TopK 特征索引
        """
        centered_x = x - self.pre_bias
        pre_activation = self.encoder(centered_x)
        topk_values, topk_indices = torch.topk(
            pre_activation, k=self.config.k, dim=-1
        )
        return F.relu(topk_values), topk_indices
    
    def get_decoder_vectors(self, indices: torch.Tensor) -> torch.Tensor:
        """获取指定特征的 decoder 向量
        Args:
            indices: [batch, k] 特征索引
        Returns:
            vectors: [batch, k, input_dim] decoder 向量
        """
        return self.decoder.weight[:, indices].permute(1, 2, 0)
    
    def steer(
        self,
        x: torch.Tensor,
        feature_idx: int,
        strength: float,
    ) -> torch.Tensor:
        """在原始 activation 上直接加减 decoder 方向来 steer 指定特征。

        h_new = h + (strength - 1) * z_i * d_i

        其中 z_i 是 feature_idx 的 TopK 编码激活值，d_i 是其 decoder 列向量。
        不走完整 encode→decode，避免 TopK 重建误差污染其他特征。
        strength=1.0 时变化量为 0（真正的 baseline）。

        Args:
            x: [batch, input_dim] 输入激活
            feature_idx: 要 steering 的特征索引
            strength: 目标特征缩放系数（1.0 不变，>1 增强，<1 抑制，0 完全消除）
        Returns:
            steered_x: [batch, input_dim] 调整后的激活
        """
        with torch.no_grad():
            latents = self.encode(x)                         # [batch, dict_size]
            z_i = latents[:, feature_idx]                   # [batch]
            d_i = self.decoder.weight[:, feature_idx]       # [input_dim]
            delta = (strength - 1) * z_i.unsqueeze(-1) * d_i.unsqueeze(0)
            return x + delta

    def steer_multi(
        self,
        x: torch.Tensor,
        feature_indices: list,
        strengths: list,
    ) -> torch.Tensor:
        """在原始 activation 上直接加减 decoder 方向来同时 steer 多个特征。

        h_new = h + sum_i (strength_i - 1) * z_i * d_i

        不走完整 encode→decode，避免 TopK 重建误差污染其他特征。
        strength=1.0 时对应特征的变化量为 0（真正的 baseline）。

        Args:
            x: [batch, input_dim] 输入激活
            feature_indices: 要 steering 的特征索引列表
            strengths: 对应的缩放系数列表
        Returns:
            steered_x: [batch, input_dim] 调整后的激活
        """
        with torch.no_grad():
            latents = self.encode(x)                         # [batch, dict_size]
            delta = torch.zeros_like(x)
            for feat_idx, strength in zip(feature_indices, strengths):
                z_i = latents[:, feat_idx]                  # [batch]
                d_i = self.decoder.weight[:, feat_idx]      # [input_dim]
                delta += (strength - 1) * z_i.unsqueeze(-1) * d_i.unsqueeze(0)
            return x + delta
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            "config": self.config,
            "state_dict": self.state_dict(),
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "TopKSAE":
        """加载模型"""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        config.device = device
        
        model = cls(config)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(config.get_torch_dtype())
        
        return model


