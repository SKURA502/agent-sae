"""
Linear Probe - 线性探测实验

使用少量 SAE 特征预测 tool-call 决策。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from sae import FeatureExtractor


class LinearProbe:
    """线性探测器 - 验证少量特征的预测能力"""
    
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
    
    def probe(
        self,
        activations: torch.Tensor,
        labels: torch.Tensor,
        feature_subsets: Optional[Dict[str, List[int]]] = None,
        k_values: Optional[List[int]] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """运行线性探测实验
        
        Args:
            activations: 输入激活
            labels: 标签
            feature_subsets: 要测试的特征子集 {名称: [特征索引]}
            k_values: 要测试的 K 值列表（使用 top-K 特征）
            output_dir: 输出目录
            
        Returns:
            探测结果
        """
        print(f"Running linear probe on {len(activations)} samples")
        
        # 提取 SAE 特征
        sae_acts = self.feature_extractor.get_full_activations(activations)
        labels_np = labels.numpy()
        
        results = {
            "num_samples": len(activations),
            "experiments": [],
        }
        
        # 1. 使用所有特征的基线
        print("Testing with all features...")
        all_features_result = self._train_and_evaluate(
            sae_acts.numpy(),
            labels_np,
            name="all_features",
        )
        results["experiments"].append(all_features_result)
        results["all_features_auc"] = all_features_result["auc"]
        
        # 2. 测试不同 K 值
        if k_values is None:
            k_values = [5, 10, 20, 50, 100, 200, 500]
        
        print(f"Testing K values: {k_values}")
        k_results = self._test_k_values(sae_acts.numpy(), labels_np, k_values)
        results["k_experiments"] = k_results
        
        # 3. 测试指定的特征子集
        if feature_subsets:
            print(f"Testing feature subsets: {list(feature_subsets.keys())}")
            for name, indices in feature_subsets.items():
                subset_acts = sae_acts[:, indices].numpy()
                subset_result = self._train_and_evaluate(
                    subset_acts,
                    labels_np,
                    name=name,
                )
                results["experiments"].append(subset_result)
        
        # 4. 特征重要性分析
        print("Computing feature importance...")
        importance_result = self._compute_feature_importance(
            sae_acts.numpy(),
            labels_np,
        )
        results["feature_importance"] = importance_result
        
        # 保存结果
        if output_dir:
            self._save_results(results, output_dir)
        
        return results
    
    def _train_and_evaluate(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        name: str = "probe",
        test_size: float = 0.2,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """训练并评估线性分类器"""
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # 训练逻辑回归
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # 计算指标
        auc = roc_auc_score(y_test, y_prob)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # 交叉验证
        cv_scores = cross_val_score(
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            features, labels,
            cv=cv_folds,
            scoring="roc_auc",
        )
        
        return {
            "name": name,
            "num_features": features.shape[1],
            "auc": float(auc),
            "accuracy": float(accuracy),
            "f1": float(f1),
            "cv_auc_mean": float(cv_scores.mean()),
            "cv_auc_std": float(cv_scores.std()),
        }
    
    def _test_k_values(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        k_values: List[int],
    ) -> List[Dict[str, Any]]:
        """测试不同 K 值（top-K 特征）"""
        # 首先计算特征重要性
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(features, labels)
        importance = np.abs(model.coef_[0])
        
        # 按重要性排序
        sorted_indices = np.argsort(importance)[::-1]
        
        results = []
        for k in tqdm(k_values, desc="Testing K values"):
            if k > len(sorted_indices):
                k = len(sorted_indices)
            
            top_k_indices = sorted_indices[:k]
            subset_features = features[:, top_k_indices]
            
            result = self._train_and_evaluate(
                subset_features,
                labels,
                name=f"top_{k}",
            )
            result["k"] = k
            result["feature_indices"] = top_k_indices.tolist()
            results.append(result)
        
        return results
    
    def _compute_feature_importance(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        top_k: int = 100,
    ) -> Dict[str, Any]:
        """计算特征重要性"""
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(features, labels)
        
        importance = model.coef_[0]
        abs_importance = np.abs(importance)
        
        # 排序
        sorted_indices = np.argsort(abs_importance)[::-1]
        
        return {
            "importance_scores": importance.tolist(),
            "top_positive": np.argsort(importance)[::-1][:top_k].tolist(),
            "top_negative": np.argsort(importance)[:top_k].tolist(),
            "top_absolute": sorted_indices[:top_k].tolist(),
        }
    
    def _save_results(self, results: Dict[str, Any], output_dir: str):
        """保存结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / "linear_probe_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_path}")
    
    def get_minimal_feature_set(
        self,
        activations: torch.Tensor,
        labels: torch.Tensor,
        target_auc: float = 0.90,
        max_features: int = 100,
    ) -> Tuple[List[int], float]:
        """找到达到目标 AUC 所需的最小特征集
        
        Returns:
            feature_indices: 特征索引列表
            achieved_auc: 达到的 AUC
        """
        sae_acts = self.feature_extractor.get_full_activations(activations)
        features = sae_acts.numpy()
        labels_np = labels.numpy()
        
        # 获取特征重要性排序
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(features, labels_np)
        importance = np.abs(model.coef_[0])
        sorted_indices = np.argsort(importance)[::-1]
        
        # 逐步增加特征
        for k in range(1, min(max_features + 1, len(sorted_indices) + 1)):
            top_k_indices = sorted_indices[:k]
            subset_features = features[:, top_k_indices]
            
            # 交叉验证 AUC
            cv_scores = cross_val_score(
                LogisticRegression(max_iter=1000, class_weight="balanced"),
                subset_features, labels_np,
                cv=5,
                scoring="roc_auc",
            )
            mean_auc = cv_scores.mean()
            
            if mean_auc >= target_auc:
                return top_k_indices.tolist(), float(mean_auc)
        
        # 如果达不到目标，返回 max_features
        return sorted_indices[:max_features].tolist(), float(mean_auc)


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Linear Probe")
    parser.add_argument("--sae-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output-dir", type=str, default="./outputs/analysis_results")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # 加载数据
    data = torch.load(args.data_path)
    activations = data[f"layer_{args.layer}"]
    labels = data["labels"]
    
    # 运行探测
    probe = LinearProbe(args.sae_path, device=args.device)
    results = probe.probe(activations, labels, output_dir=args.output_dir)
    
    print("\nLinear Probe Results:")
    print(f"All features AUC: {results['all_features_auc']:.4f}")
    
    for exp in results["k_experiments"]:
        print(f"Top-{exp['k']} features AUC: {exp['auc']:.4f}")


if __name__ == "__main__":
    main()
