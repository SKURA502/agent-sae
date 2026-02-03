"""
Correlation Analysis - 特征与决策的相关性分析

计算 SAE 特征与 tool-call 决策之间的相关性。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from sae import TopKSAE, FeatureExtractor


class CorrelationAnalyzer:
    """特征-决策相关性分析器"""
    
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
        self.feature_extractor = FeatureExtractor(sae_path, device)
        self.device = device
    
    def analyze(
        self,
        activations: torch.Tensor,
        labels: torch.Tensor,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """完整的相关性分析
        
        Args:
            activations: [num_samples, hidden_dim] 输入激活
            labels: [num_samples] 标签 (0=NO_CALL, 1=CALL)
            output_dir: 输出目录（可选）
            
        Returns:
            分析结果字典
        """
        print(f"Analyzing {len(activations)} samples")
        
        # 提取 SAE 特征
        print("Extracting SAE features...")
        sae_acts = self.feature_extractor.get_full_activations(activations)
        
        results = {
            "num_samples": len(activations),
            "num_call": (labels == 1).sum().item(),
            "num_no_call": (labels == 0).sum().item(),
            "dict_size": sae_acts.shape[1],
        }
        
        # 1. 均值差异分析
        print("Computing mean differences...")
        mean_diff_results = self._compute_mean_differences(sae_acts, labels)
        results["mean_diff"] = mean_diff_results
        
        # 2. AUROC 分析
        print("Computing AUROC scores...")
        auroc_results = self._compute_auroc(sae_acts, labels)
        results["auroc"] = auroc_results
        
        # 3. 激活频率分析
        print("Computing activation frequencies...")
        freq_results = self._compute_activation_frequency(sae_acts, labels)
        results["frequency"] = freq_results
        
        # 4. 统计显著性检验
        print("Computing statistical significance...")
        significance_results = self._compute_significance(sae_acts, labels)
        results["significance"] = significance_results
        
        # 5. 识别 top features
        results["top_features"] = self._identify_top_features(results)
        
        # 保存结果
        if output_dir:
            self._save_results(results, output_dir)
        
        return results
    
    def _compute_mean_differences(
        self,
        sae_acts: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, Any]:
        """计算 CALL vs NO_CALL 的均值差异"""
        call_mask = labels == 1
        no_call_mask = labels == 0
        
        call_mean = sae_acts[call_mask].mean(dim=0)
        no_call_mean = sae_acts[no_call_mask].mean(dim=0)
        
        diff = call_mean - no_call_mean
        
        return {
            "call_mean": call_mean.numpy().tolist(),
            "no_call_mean": no_call_mean.numpy().tolist(),
            "difference": diff.numpy().tolist(),
            "top_positive": diff.argsort(descending=True)[:100].tolist(),
            "top_negative": diff.argsort()[:100].tolist(),
        }
    
    def _compute_auroc(
        self,
        sae_acts: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, Any]:
        """计算每个特征的 AUROC"""
        n_features = sae_acts.shape[1]
        auroc_scores = np.zeros(n_features)
        
        labels_np = labels.numpy()
        
        for i in tqdm(range(n_features), desc="Computing AUROC"):
            feature_acts = sae_acts[:, i].numpy()
            
            # 跳过全零特征
            if feature_acts.std() == 0:
                auroc_scores[i] = 0.5
                continue
            
            try:
                auroc_scores[i] = roc_auc_score(labels_np, feature_acts)
            except ValueError:
                auroc_scores[i] = 0.5
        
        return {
            "scores": auroc_scores.tolist(),
            "top_call_predictive": np.argsort(auroc_scores)[::-1][:100].tolist(),
            "top_no_call_predictive": np.argsort(auroc_scores)[:100].tolist(),
            "mean_auroc": float(np.mean(auroc_scores)),
            "std_auroc": float(np.std(auroc_scores)),
        }
    
    def _compute_activation_frequency(
        self,
        sae_acts: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, Any]:
        """计算激活频率差异"""
        call_mask = labels == 1
        no_call_mask = labels == 0
        
        # 激活频率 = 非零激活的比例
        call_freq = (sae_acts[call_mask] > 0).float().mean(dim=0)
        no_call_freq = (sae_acts[no_call_mask] > 0).float().mean(dim=0)
        
        freq_diff = call_freq - no_call_freq
        
        return {
            "call_frequency": call_freq.numpy().tolist(),
            "no_call_frequency": no_call_freq.numpy().tolist(),
            "frequency_diff": freq_diff.numpy().tolist(),
            "top_call_specific": freq_diff.argsort(descending=True)[:100].tolist(),
            "top_no_call_specific": freq_diff.argsort()[:100].tolist(),
        }
    
    def _compute_significance(
        self,
        sae_acts: torch.Tensor,
        labels: torch.Tensor,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """计算统计显著性（t-test）"""
        call_mask = labels == 1
        no_call_mask = labels == 0
        
        call_acts = sae_acts[call_mask].numpy()
        no_call_acts = sae_acts[no_call_mask].numpy()
        
        n_features = sae_acts.shape[1]
        p_values = np.zeros(n_features)
        t_stats = np.zeros(n_features)
        
        for i in range(n_features):
            # 两样本 t 检验
            t_stat, p_val = stats.ttest_ind(
                call_acts[:, i],
                no_call_acts[:, i],
                equal_var=False  # Welch's t-test
            )
            t_stats[i] = t_stat
            p_values[i] = p_val
        
        # Bonferroni 校正
        significant_mask = p_values < (alpha / n_features)
        
        return {
            "t_statistics": t_stats.tolist(),
            "p_values": p_values.tolist(),
            "significant_features": np.where(significant_mask)[0].tolist(),
            "num_significant": int(significant_mask.sum()),
            "alpha": alpha,
            "corrected_alpha": alpha / n_features,
        }
    
    def _identify_top_features(
        self,
        results: Dict[str, Any],
        top_k: int = 50,
    ) -> Dict[str, List[int]]:
        """综合识别最重要的特征"""
        # 从多个指标综合
        mean_diff = np.array(results["mean_diff"]["difference"])
        auroc = np.array(results["auroc"]["scores"])
        freq_diff = np.array(results["frequency"]["frequency_diff"])
        
        # 计算综合得分
        # 归一化各指标
        def normalize(x):
            return (x - x.mean()) / (x.std() + 1e-8)
        
        # 对于 CALL 相关的特征：高均值差、高 AUROC、高频率差
        call_score = normalize(mean_diff) + normalize(auroc - 0.5) + normalize(freq_diff)
        
        # 对于 NO_CALL 相关的特征：低均值差、低 AUROC、低频率差
        no_call_score = -call_score
        
        return {
            "call_gate_features": np.argsort(call_score)[::-1][:top_k].tolist(),
            "no_call_gate_features": np.argsort(no_call_score)[::-1][:top_k].tolist(),
            "call_scores": call_score.tolist(),
        }
    
    def _save_results(self, results: Dict[str, Any], output_dir: str):
        """保存分析结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存完整结果
        with open(output_path / "correlation_analysis.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # 保存摘要
        summary = {
            "num_samples": results["num_samples"],
            "num_call": results["num_call"],
            "num_no_call": results["num_no_call"],
            "num_significant_features": results["significance"]["num_significant"],
            "mean_auroc": results["auroc"]["mean_auroc"],
            "top_20_call_features": results["top_features"]["call_gate_features"][:20],
            "top_20_no_call_features": results["top_features"]["no_call_gate_features"][:20],
        }
        
        with open(output_path / "analysis_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to {output_path}")


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Correlation Analysis")
    parser.add_argument("--sae-path", type=str, required=True,
                        help="Path to SAE model")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to activation data")
    parser.add_argument("--layer", type=int, required=True,
                        help="Layer to analyze")
    parser.add_argument("--output-dir", type=str, default="./outputs/analysis_results",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device")
    
    args = parser.parse_args()
    
    # 加载数据
    data = torch.load(args.data_path)
    activations = data[f"layer_{args.layer}"]
    labels = data["labels"]
    
    print(f"Loaded {len(activations)} samples")
    
    # 分析
    analyzer = CorrelationAnalyzer(args.sae_path, device=args.device)
    results = analyzer.analyze(activations, labels, output_dir=args.output_dir)
    
    print(f"\nAnalysis complete!")
    print(f"Number of significant features: {results['significance']['num_significant']}")
    print(f"Top 10 CALL gate features: {results['top_features']['call_gate_features'][:10]}")


if __name__ == "__main__":
    main()
