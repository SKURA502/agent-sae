"""
Visualization - 可视化工具

生成论文所需的核心图表。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 延迟导入可视化库
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class Visualizer:
    """可视化器 - 生成分析图表"""
    
    def __init__(self, output_dir: str, style: str = "paper"):
        """
        Args:
            output_dir: 输出目录
            style: 图表风格 ("paper", "presentation", "notebook")
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.style = style
        
        if MATPLOTLIB_AVAILABLE:
            self._setup_matplotlib_style()
    
    def _setup_matplotlib_style(self):
        """设置 matplotlib 风格"""
        if self.style == "paper":
            plt.rcParams.update({
                "font.size": 12,
                "axes.labelsize": 14,
                "axes.titlesize": 16,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "legend.fontsize": 12,
                "figure.figsize": (8, 6),
                "figure.dpi": 150,
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
            })
            sns.set_style("whitegrid")
    
    def plot_feature_separability(
        self,
        auroc_scores: List[float],
        mean_diff: List[float],
        top_k: int = 100,
        filename: str = "feature_separability.pdf",
    ):
        """Fig 1: 特征可分离性
        
        Args:
            auroc_scores: 每个特征的 AUROC 分数
            mean_diff: 每个特征的 CALL - NO_CALL 均值差
            top_k: 显示的 top 特征数量
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available, skipping plot")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        auroc_np = np.array(auroc_scores)
        diff_np = np.array(mean_diff)
        
        # (a) AUROC 分布
        ax1 = axes[0]
        ax1.hist(auroc_np, bins=50, edgecolor="black", alpha=0.7)
        ax1.axvline(0.5, color="red", linestyle="--", label="Random (0.5)")
        ax1.axvline(auroc_np.mean(), color="green", linestyle="-", label=f"Mean ({auroc_np.mean():.2f})")
        ax1.set_xlabel("AUROC Score")
        ax1.set_ylabel("Number of Features")
        ax1.set_title("(a) AUROC Distribution")
        ax1.legend()
        
        # (b) Top features AUROC
        ax2 = axes[1]
        sorted_indices = np.argsort(np.abs(auroc_np - 0.5))[::-1][:top_k]
        sorted_auroc = auroc_np[sorted_indices]
        
        colors = ["#d62728" if a > 0.5 else "#1f77b4" for a in sorted_auroc]
        ax2.bar(range(top_k), sorted_auroc - 0.5, color=colors, alpha=0.7)
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.set_xlabel("Feature Rank")
        ax2.set_ylabel("AUROC - 0.5")
        ax2.set_title("(b) Top Features by AUROC")
        
        # (c) 均值差分布
        ax3 = axes[2]
        sorted_diff = np.sort(diff_np)[::-1][:top_k]
        ax3.bar(range(top_k), sorted_diff, alpha=0.7)
        ax3.axhline(0, color="black", linewidth=0.5)
        ax3.set_xlabel("Feature Rank")
        ax3.set_ylabel("Mean Activation Difference")
        ax3.set_title("(c) Mean Difference (CALL - NO_CALL)")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()
        
        print(f"Saved: {self.output_dir / filename}")
    
    def plot_linear_probe_auc(
        self,
        k_values: List[int],
        auc_values: List[float],
        auc_std: Optional[List[float]] = None,
        filename: str = "linear_probe_auc.pdf",
    ):
        """Fig 1 补充: 线性探测 AUC vs K
        
        Args:
            k_values: K 值列表
            auc_values: 对应的 AUC 值
            auc_std: AUC 标准差（可选）
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available, skipping plot")
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if auc_std:
            ax.errorbar(k_values, auc_values, yerr=auc_std, 
                       marker="o", capsize=3, capthick=1)
        else:
            ax.plot(k_values, auc_values, marker="o")
        
        ax.axhline(0.9, color="red", linestyle="--", alpha=0.5, label="AUC = 0.9")
        ax.axhline(0.95, color="green", linestyle="--", alpha=0.5, label="AUC = 0.95")
        
        ax.set_xlabel("Number of Features (K)")
        ax.set_ylabel("AUC Score")
        ax.set_title("Linear Probe: AUC vs Number of Features")
        ax.legend()
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()
        
        print(f"Saved: {self.output_dir / filename}")
    
    def plot_steering_results(
        self,
        strengths: List[float],
        flip_rates: List[float],
        call_rates: List[float],
        baseline_call_rate: float,
        filename: str = "steering_results.pdf",
    ):
        """Fig 2: 因果干预效果
        
        Args:
            strengths: steering 强度列表
            flip_rates: 对应的决策翻转率
            call_rates: 对应的工具调用率
            baseline_call_rate: 基线工具调用率
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available, skipping plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # (a) Flip rate vs strength
        ax1 = axes[0]
        ax1.plot(strengths, flip_rates, marker="o", linewidth=2)
        ax1.set_xlabel("Steering Strength (α)")
        ax1.set_ylabel("Decision Flip Rate")
        ax1.set_title("(a) Decision Flip Rate vs Steering Strength")
        ax1.grid(True, alpha=0.3)
        
        # (b) Tool call rate vs strength
        ax2 = axes[1]
        ax2.plot(strengths, call_rates, marker="o", linewidth=2, label="Steered")
        ax2.axhline(baseline_call_rate, color="red", linestyle="--", 
                   label=f"Baseline ({baseline_call_rate:.2f})")
        ax2.set_xlabel("Steering Strength (α)")
        ax2.set_ylabel("Tool Call Rate")
        ax2.set_title("(b) Tool Call Rate vs Steering Strength")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()
        
        print(f"Saved: {self.output_dir / filename}")
    
    def plot_dynamics(
        self,
        successful_trajectories: List[List[float]],
        failed_trajectories: List[List[float]],
        feature_name: str = "Gate Feature",
        filename: str = "feature_dynamics.pdf",
    ):
        """Fig 3: 动态门控
        
        Args:
            successful_trajectories: 成功 episode 的特征轨迹列表
            failed_trajectories: 失败 episode 的特征轨迹列表
            feature_name: 特征名称
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available, skipping plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 计算平均轨迹
        max_len = max(
            max(len(t) for t in successful_trajectories) if successful_trajectories else 0,
            max(len(t) for t in failed_trajectories) if failed_trajectories else 0,
        )
        
        def compute_mean_trajectory(trajectories, max_len):
            if not trajectories:
                return None, None
            
            # 填充到相同长度
            padded = []
            for t in trajectories:
                if len(t) < max_len:
                    t = t + [np.nan] * (max_len - len(t))
                padded.append(t[:max_len])
            
            arr = np.array(padded)
            mean = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0)
            
            return mean, std
        
        steps = np.arange(max_len)
        
        # 成功轨迹
        if successful_trajectories:
            mean_s, std_s = compute_mean_trajectory(successful_trajectories, max_len)
            ax.plot(steps, mean_s, color="green", linewidth=2, label="Successful Episodes")
            ax.fill_between(steps, mean_s - std_s, mean_s + std_s, 
                           color="green", alpha=0.2)
        
        # 失败轨迹
        if failed_trajectories:
            mean_f, std_f = compute_mean_trajectory(failed_trajectories, max_len)
            ax.plot(steps, mean_f, color="red", linewidth=2, label="Failed Episodes")
            ax.fill_between(steps, mean_f - std_f, mean_f + std_f, 
                           color="red", alpha=0.2)
        
        ax.set_xlabel("Step")
        ax.set_ylabel(f"{feature_name} Activation")
        ax.set_title(f"Feature Dynamics: {feature_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()
        
        print(f"Saved: {self.output_dir / filename}")
    
    def plot_heatmap(
        self,
        data: np.ndarray,
        row_labels: List[str],
        col_labels: List[str],
        title: str = "Heatmap",
        filename: str = "heatmap.pdf",
    ):
        """通用热力图"""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available, skipping plot")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(
            data,
            xticklabels=col_labels,
            yticklabels=row_labels,
            cmap="RdBu_r",
            center=0,
            ax=ax,
            annot=True if data.shape[0] * data.shape[1] < 100 else False,
            fmt=".2f",
        )
        
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()
        
        print(f"Saved: {self.output_dir / filename}")
    
    def create_summary_table(
        self,
        results: Dict[str, Any],
        filename: str = "summary_table.tex",
    ):
        """生成 LaTeX 表格"""
        # 简单的 LaTeX 表格生成
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Experiment Summary}",
            r"\begin{tabular}{lcc}",
            r"\hline",
            r"Metric & Value & Std \\",
            r"\hline",
        ]
        
        for key, value in results.items():
            if isinstance(value, (int, float)):
                lines.append(f"{key} & {value:.4f} & - \\\\")
            elif isinstance(value, dict) and "mean" in value:
                lines.append(f"{key} & {value['mean']:.4f} & {value.get('std', 0):.4f} \\\\")
        
        lines.extend([
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ])
        
        with open(self.output_dir / filename, "w") as f:
            f.write("\n".join(lines))
        
        print(f"Saved: {self.output_dir / filename}")


def main():
    """示例用法"""
    viz = Visualizer("./outputs/figures")
    
    # 示例数据
    np.random.seed(42)
    auroc = np.random.beta(2, 2, 1000)  # 集中在 0.5 附近
    auroc[:50] = np.random.uniform(0.7, 0.95, 50)  # 一些高 AUROC 特征
    auroc[50:100] = np.random.uniform(0.05, 0.3, 50)  # 一些低 AUROC 特征
    
    mean_diff = np.random.randn(1000) * 0.1
    mean_diff[:50] = np.random.uniform(0.3, 0.8, 50)
    mean_diff[50:100] = np.random.uniform(-0.8, -0.3, 50)
    
    viz.plot_feature_separability(auroc.tolist(), mean_diff.tolist())
    
    k_values = [5, 10, 20, 50, 100, 200, 500]
    auc_values = [0.65, 0.72, 0.78, 0.85, 0.89, 0.92, 0.94]
    viz.plot_linear_probe_auc(k_values, auc_values)
    
    print("Visualization examples complete!")


if __name__ == "__main__":
    main()
