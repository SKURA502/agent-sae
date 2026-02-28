"""
SAE Model - TopK Sparse Autoencoder

实现用于 LLM 激活分析的稀疏自编码器。
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
    # 输入维度（模型 hidden_size）
    input_dim: int = 4096
    # 字典大小（latent 维度）
    dict_size: int = 32768
    # TopK 的 K 值
    k: int = 128
    # 设备
    device: str = "cuda"
    # 数据类型
    dtype: str = "float32"
    
    def get_torch_dtype(self) -> torch.dtype:
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.dtype, torch.float32)


class TopKSAE(nn.Module):
    """TopK Sparse Autoencoder
    
    使用 TopK 激活实现稀疏性，而不是 L1 正则化。
    """
    
    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        
        # Encoder: input_dim -> dict_size
        self.encoder = nn.Linear(config.input_dim, config.dict_size, bias=False)
        
        # Decoder: dict_size -> input_dim
        self.decoder = nn.Linear(config.dict_size, config.input_dim, bias=False)

        # Learnable parameters
        self.pre_bias = nn.Parameter(torch.zeros(config.input_dim))
        
        # 初始化
        self._init_weights()
        
        # 移动到设备
        self.to(config.device)
        self.to(config.get_torch_dtype())
    
    def _init_weights(self):
        """初始化权重"""
        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data.T.clone()
    
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
        # 线性变换
        centered_x = x - self.pre_bias
        pre_activation = self.encoder(centered_x)
        
        # TopK 稀疏化
        topk_values, topk_indices = torch.topk(
            pre_activation, k=self.config.k, dim=-1
        )
        
        # 创建稀疏激活
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
        """前向传播
        
        Args:
            x: [batch, input_dim]
            
        Returns:
            x_hat: [batch, input_dim]
            latent: [batch, dict_size]
        """
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
        
        # 重建损失 (Normalized_MSE)
        loss = (((x_hat - x) ** 2).mean(dim=-1) / (x**2).mean(dim=-1)).mean()

        loss_dict = {
            "loss": loss.item(),
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
        # decoder.weight: [input_dim, dict_size]
        # 需要取 indices 对应的列
        return self.decoder.weight[:, indices].permute(1, 2, 0)
    
    def steer(
        self,
        x: torch.Tensor,
        feature_idx: int,
        strength: float,
    ) -> torch.Tensor:
        """对输入进行特征 steering
        
        Args:
            x: [batch, input_dim] 输入激活
            feature_idx: 要 steering 的特征索引
            strength: steering 强度（正数增强，负数抑制）
            
        Returns:
            steered: [batch, input_dim] steering 后的激活
        """
        # 获取 decoder 向量
        decoder_vector = self.decoder.weight[:, feature_idx]  # [input_dim]
        
        # 添加 steering
        steered = x + strength * decoder_vector.unsqueeze(0)
        
        return steered
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            "config": self.config,
            "state_dict": self.state_dict(),
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "TopKSAE":
        """加载模型"""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]
        config.device = device
        
        model = cls(config)
        model.load_state_dict(checkpoint["state_dict"])
        
        return model


