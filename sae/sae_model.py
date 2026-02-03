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
    # 是否使用 encoder bias
    use_encoder_bias: bool = True
    # 是否使用 decoder bias
    use_decoder_bias: bool = True
    # 是否归一化 decoder
    normalize_decoder: bool = True
    # 初始化标准差
    init_std: float = 0.02
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
        self.encoder = nn.Linear(
            config.input_dim,
            config.dict_size,
            bias=config.use_encoder_bias,
        )
        
        # Decoder: dict_size -> input_dim
        self.decoder = nn.Linear(
            config.dict_size,
            config.input_dim,
            bias=config.use_decoder_bias,
        )
        
        # 初始化
        self._init_weights()
        
        # 移动到设备
        self.to(config.device)
        self.to(config.get_torch_dtype())
    
    def _init_weights(self):
        """初始化权重"""
        # Xavier 初始化
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        
        if self.config.use_encoder_bias:
            nn.init.zeros_(self.encoder.bias)
        if self.config.use_decoder_bias:
            nn.init.zeros_(self.decoder.bias)
        
        # 归一化 decoder
        if self.config.normalize_decoder:
            self._normalize_decoder()
    
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
        pre_activation = self.encoder(x)
        
        # TopK 稀疏化
        topk_values, topk_indices = torch.topk(
            pre_activation, k=self.config.k, dim=-1
        )
        
        # 创建稀疏激活
        sparse_activation = torch.zeros_like(pre_activation)
        sparse_activation.scatter_(-1, topk_indices, F.relu(topk_values))
        
        return sparse_activation
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """解码：latent -> reconstruction
        
        Args:
            latent: [batch, dict_size]
            
        Returns:
            reconstruction: [batch, input_dim]
        """
        return self.decoder(latent)
    
    def forward(
        self,
        x: torch.Tensor,
        return_latent: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """前向传播
        
        Args:
            x: [batch, input_dim]
            return_latent: 是否返回 latent 激活
            
        Returns:
            reconstruction: [batch, input_dim]
            latent: [batch, dict_size] (可选)
        """
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        
        if return_latent:
            return reconstruction, latent
        return reconstruction, None
    
    def compute_loss(
        self,
        x: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor:
        """计算损失
        
        Args:
            x: [batch, input_dim]
            return_components: 是否返回损失组成
            
        Returns:
            loss: 标量损失
            或 (loss, loss_dict) 如果 return_components=True
        """
        reconstruction, latent = self.forward(x, return_latent=True)
        
        # 重建损失 (MSE)
        reconstruction_loss = F.mse_loss(reconstruction, x)
        
        # 总损失（TopK SAE 不需要 L1 正则化，稀疏性由 TopK 保证）
        total_loss = reconstruction_loss
        
        if return_components:
            loss_dict = {
                "reconstruction_loss": reconstruction_loss.item(),
                "sparsity": (latent > 0).float().mean().item(),
                "mean_activation": latent[latent > 0].mean().item() if (latent > 0).any() else 0,
            }
            return total_loss, loss_dict
        
        return total_loss
    
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
        pre_activation = self.encoder(x)
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


class VanillaSAE(nn.Module):
    """Vanilla SAE with L1 regularization (for comparison)"""
    
    def __init__(self, config: SAEConfig, l1_coef: float = 1e-3):
        super().__init__()
        self.config = config
        self.l1_coef = l1_coef
        
        self.encoder = nn.Linear(
            config.input_dim,
            config.dict_size,
            bias=config.use_encoder_bias,
        )
        
        self.decoder = nn.Linear(
            config.dict_size,
            config.input_dim,
            bias=config.use_decoder_bias,
        )
        
        self._init_weights()
        self.to(config.device)
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        if self.config.use_encoder_bias:
            nn.init.zeros_(self.encoder.bias)
        if self.config.use_decoder_bias:
            nn.init.zeros_(self.decoder.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(x))
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)
    
    def forward(
        self,
        x: torch.Tensor,
        return_latent: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        
        if return_latent:
            return reconstruction, latent
        return reconstruction, None
    
    def compute_loss(
        self,
        x: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor:
        reconstruction, latent = self.forward(x, return_latent=True)
        
        reconstruction_loss = F.mse_loss(reconstruction, x)
        l1_loss = latent.abs().mean()
        
        total_loss = reconstruction_loss + self.l1_coef * l1_loss
        
        if return_components:
            loss_dict = {
                "reconstruction_loss": reconstruction_loss.item(),
                "l1_loss": l1_loss.item(),
                "sparsity": (latent > 0).float().mean().item(),
            }
            return total_loss, loss_dict
        
        return total_loss
