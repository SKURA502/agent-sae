"""
Feature Extraction - SAE 特征提取和分析

从 SAE 中提取特征激活，用于后续分析。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from .sae_model import TopKSAE


class FeatureExtractor:
    """SAE 特征提取器"""
    
    def __init__(
        self,
        sae_path: str,
        device: str = "cuda",
    ):
        """
        Args:
            sae_path: SAE 模型路径
            device: 设备
        """
        self.device = device
        self.sae = TopKSAE.load(sae_path, device=device)
        self.sae.eval()
        
        print(f"Loaded SAE from {sae_path}")
        print(f"Dict size: {self.sae.config.latent_size}, K: {self.sae.config.k}")
    
    @torch.no_grad()
    def extract_features(
        self,
        activations: torch.Tensor,
        batch_size: int = 1024,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """提取 SAE 特征
        
        Args:
            activations: [num_samples, hidden_dim] 输入激活
            batch_size: 批大小
            
        Returns:
            values: [num_samples, k] TopK 激活值
            indices: [num_samples, k] TopK 特征索引
        """
        all_values = []
        all_indices = []
        
        for i in range(0, len(activations), batch_size):
            batch = activations[i:i+batch_size].to(self.device)
            values, indices = self.sae.get_feature_activations(batch)
            all_values.append(values.cpu())
            all_indices.append(indices.cpu())
        
        return torch.cat(all_values, dim=0), torch.cat(all_indices, dim=0)
    
    @torch.no_grad()
    def get_full_activations(
        self,
        activations: torch.Tensor,
        batch_size: int = 1024,
    ) -> torch.Tensor:
        """获取完整的 SAE 激活（稀疏）
        
        Args:
            activations: [num_samples, hidden_dim]
            batch_size: 批大小
            
        Returns:
            sae_activations: [num_samples, dict_size]
        """
        all_acts = []
        
        for i in range(0, len(activations), batch_size):
            batch = activations[i:i+batch_size].to(self.device)
            sae_act = self.sae.encode(batch)
            all_acts.append(sae_act.cpu())
        
        return torch.cat(all_acts, dim=0)
    
    def compute_feature_statistics(
        self,
        activations: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """计算特征统计信息
        
        Args:
            activations: 输入激活
            labels: 标签 (0=NO_CALL, 1=CALL)
            
        Returns:
            统计信息字典
        """
        sae_acts = self.get_full_activations(activations)
        
        stats = {
            "num_samples": len(activations),
            "dict_size": self.sae.config.latent_size,
            "k": self.sae.config.k,
        }
        
        # 全局统计
        stats["global"] = {
            "mean_activation": sae_acts[sae_acts > 0].mean().item(),
            "sparsity": (sae_acts > 0).float().mean().item(),
            "active_features": (sae_acts.sum(dim=0) > 0).sum().item(),
        }
        
        # 每个特征的激活频率
        feature_freq = (sae_acts > 0).float().mean(dim=0)
        stats["feature_frequency"] = feature_freq.tolist()
        
        # 如果有标签，计算与决策相关的统计
        if labels is not None:
            call_mask = labels == 1
            no_call_mask = labels == 0
            
            call_acts = sae_acts[call_mask]
            no_call_acts = sae_acts[no_call_mask]
            
            # 每个特征在 CALL vs NO_CALL 中的差异
            call_mean = call_acts.mean(dim=0)
            no_call_mean = no_call_acts.mean(dim=0)
            
            stats["call_vs_no_call"] = {
                "mean_diff": (call_mean - no_call_mean).tolist(),
                "call_mean": call_mean.tolist(),
                "no_call_mean": no_call_mean.tolist(),
            }
            
            # 计算每个特征的 AUROC（简化版）
            auroc_scores = self._compute_auroc(sae_acts, labels)
            stats["feature_auroc"] = auroc_scores.tolist()
        
        return stats
    
    def _compute_auroc(
        self,
        activations: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """计算每个特征的 AUROC
        
        简化版本：使用 Mann-Whitney U 统计量近似
        """
        n_features = activations.shape[1]
        auroc = torch.zeros(n_features)
        
        call_acts = activations[labels == 1]
        no_call_acts = activations[labels == 0]
        
        n1, n0 = len(call_acts), len(no_call_acts)
        
        if n1 == 0 or n0 == 0:
            return auroc
        
        for i in range(n_features):
            # 计算 U 统计量
            f1 = call_acts[:, i]
            f0 = no_call_acts[:, i]
            
            # 使用广播计算
            u = (f1.unsqueeze(1) > f0.unsqueeze(0)).float().mean()
            auroc[i] = u
        
        return auroc
    
    def get_top_features_for_decision(
        self,
        stats: Dict[str, Any],
        top_k: int = 50,
    ) -> Dict[str, List[int]]:
        """获取与决策最相关的特征
        
        Args:
            stats: 特征统计信息
            top_k: 返回的特征数量
            
        Returns:
            字典包含：
            - call_features: 与 CALL 正相关的特征
            - no_call_features: 与 NO_CALL 正相关的特征
            - high_auroc_features: AUROC 最高的特征
        """
        result = {}
        
        if "call_vs_no_call" in stats:
            mean_diff = torch.tensor(stats["call_vs_no_call"]["mean_diff"])
            
            # CALL 正相关
            call_features = mean_diff.argsort(descending=True)[:top_k]
            result["call_features"] = call_features.tolist()
            
            # NO_CALL 正相关
            no_call_features = mean_diff.argsort()[:top_k]
            result["no_call_features"] = no_call_features.tolist()
        
        if "feature_auroc" in stats:
            auroc = torch.tensor(stats["feature_auroc"])
            
            # 高 AUROC（远离 0.5）
            auroc_diff = (auroc - 0.5).abs()
            high_auroc = auroc_diff.argsort(descending=True)[:top_k]
            result["high_auroc_features"] = high_auroc.tolist()
            
            # AUROC > 0.5 的（预测 CALL）
            call_auroc = auroc.argsort(descending=True)[:top_k]
            result["call_predictive_features"] = call_auroc.tolist()
            
            # AUROC < 0.5 的（预测 NO_CALL）
            no_call_auroc = auroc.argsort()[:top_k]
            result["no_call_predictive_features"] = no_call_auroc.tolist()
        
        return result
    
    @torch.no_grad()
    def get_top_activating_examples(
        self,
        feature_idx: int,
        activations: torch.Tensor,
        top_k: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取某个特征激活最强的样本
        
        Args:
            feature_idx: 特征索引
            activations: 输入激活
            top_k: 返回的样本数量
            
        Returns:
            indices: 样本索引
            values: 对应的激活值
        """
        sae_acts = self.get_full_activations(activations)
        feature_acts = sae_acts[:, feature_idx]
        
        top_values, top_indices = torch.topk(feature_acts, k=top_k)
        
        return top_indices, top_values
    
    def get_decoder_vector(self, feature_idx: int) -> torch.Tensor:
        """获取某个特征的 decoder 向量"""
        return self.sae.decoder.weight[:, feature_idx].cpu()
    
    def save_feature_stats(
        self,
        stats: Dict[str, Any],
        output_path: str,
    ):
        """保存特征统计"""
        with open(output_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved feature statistics to {output_path}")
