"""
Cache Activations - 激活缓存工具

从已生成的 rollout 中批量提取和缓存激活，用于 SAE 训练。
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))


class ActivationDataset(Dataset):
    """激活数据集"""
    
    def __init__(
        self,
        activation_dir: str,
        layers: Optional[List[int]] = None,
        decision_filter: Optional[str] = None,
    ):
        """
        Args:
            activation_dir: 激活文件目录
            layers: 要加载的层（None 表示全部）
            decision_filter: 过滤决策类型 (call/no_call)
        """
        self.activation_dir = Path(activation_dir)
        self.layers = layers
        self.decision_filter = decision_filter
        
        # 扫描所有激活文件
        self.files = list(self.activation_dir.glob("*.pt"))
        
        # 如果有过滤条件，需要加载对应的日志
        if decision_filter:
            self._filter_by_decision()
        
        print(f"Found {len(self.files)} activation files")
    
    def _filter_by_decision(self):
        """根据决策类型过滤"""
        # 查找对应的日志文件
        log_dir = self.activation_dir.parent
        log_files = list(log_dir.glob("*_rollouts.jsonl"))
        
        if not log_files:
            print("Warning: No log files found, cannot filter by decision")
            return
        
        # 加载日志获取决策信息
        valid_episodes = set()
        for log_file in log_files:
            with open(log_file, "r") as f:
                for line in f:
                    if line.strip():
                        episode = json.loads(line)
                        if episode.get("final_decision") == self.decision_filter:
                            valid_episodes.add(episode["episode_id"])
        
        # 过滤文件
        filtered_files = []
        for file in self.files:
            # 从文件名提取 episode_id
            episode_id = file.stem.rsplit("_step", 1)[0]
            if episode_id in valid_episodes:
                filtered_files.append(file)
        
        print(f"Filtered to {len(filtered_files)} files with decision={self.decision_filter}")
        self.files = filtered_files
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_path = self.files[idx]
        activations = torch.load(file_path, map_location="cpu")
        
        # 过滤层
        if self.layers is not None:
            activations = {k: v for k, v in activations.items() if k in self.layers}
        
        return activations


class ActivationCacher:
    """激活缓存器 - 用于准备 SAE 训练数据"""
    
    def __init__(
        self,
        activation_dir: str,
        output_path: str,
        layers: List[int],
        window_size: int = 20,
    ):
        """
        Args:
            activation_dir: 原始激活目录
            output_path: 输出路径
            layers: 要处理的层
            window_size: 时间窗口大小
        """
        self.activation_dir = Path(activation_dir)
        self.output_path = Path(output_path)
        self.layers = layers
        self.window_size = window_size
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def process(
        self,
        decision_filter: Optional[str] = None,
        max_samples: int = -1,
        balance: bool = True,
    ) -> Dict[str, Any]:
        """处理激活数据
        
        Args:
            decision_filter: 决策类型过滤
            max_samples: 最大样本数
            balance: 是否平衡 CALL/NO_CALL
            
        Returns:
            处理统计信息
        """
        if balance:
            return self._process_balanced(max_samples)
        else:
            return self._process_filtered(decision_filter, max_samples)
    
    def _process_filtered(
        self,
        decision_filter: Optional[str],
        max_samples: int,
    ) -> Dict[str, Any]:
        """按过滤条件处理"""
        dataset = ActivationDataset(
            str(self.activation_dir),
            layers=self.layers,
            decision_filter=decision_filter,
        )
        
        if max_samples > 0:
            indices = list(range(min(max_samples, len(dataset))))
        else:
            indices = list(range(len(dataset)))
        
        return self._collect_and_save(dataset, indices)
    
    def _process_balanced(self, max_samples: int) -> Dict[str, Any]:
        """平衡 CALL/NO_CALL 处理"""
        # 分别加载两类
        call_dataset = ActivationDataset(
            str(self.activation_dir),
            layers=self.layers,
            decision_filter="call",
        )
        
        no_call_dataset = ActivationDataset(
            str(self.activation_dir),
            layers=self.layers,
            decision_filter="no_call",
        )
        
        # 确定每类样本数
        min_count = min(len(call_dataset), len(no_call_dataset))
        if max_samples > 0:
            samples_per_class = min(max_samples // 2, min_count)
        else:
            samples_per_class = min_count
        
        print(f"Using {samples_per_class} samples per class")
        
        # 收集激活
        all_activations = {layer: [] for layer in self.layers}
        all_labels = []
        
        # CALL 样本
        for i in tqdm(range(samples_per_class), desc="Loading CALL samples"):
            acts = call_dataset[i]
            for layer in self.layers:
                if layer in acts:
                    all_activations[layer].append(acts[layer])
            all_labels.append(1)  # CALL = 1
        
        # NO_CALL 样本
        for i in tqdm(range(samples_per_class), desc="Loading NO_CALL samples"):
            acts = no_call_dataset[i]
            for layer in self.layers:
                if layer in acts:
                    all_activations[layer].append(acts[layer])
            all_labels.append(0)  # NO_CALL = 0
        
        # 合并并保存
        result = {}
        for layer in self.layers:
            if all_activations[layer]:
                result[f"layer_{layer}"] = torch.cat(all_activations[layer], dim=0)
        
        result["labels"] = torch.tensor(all_labels)
        
        # 保存
        torch.save(result, self.output_path)
        
        stats = {
            "total_samples": len(all_labels),
            "call_samples": sum(all_labels),
            "no_call_samples": len(all_labels) - sum(all_labels),
            "layers": self.layers,
            "output_path": str(self.output_path),
        }
        
        print(f"Saved {stats['total_samples']} samples to {self.output_path}")
        
        return stats
    
    def _collect_and_save(
        self,
        dataset: ActivationDataset,
        indices: List[int],
    ) -> Dict[str, Any]:
        """收集并保存激活"""
        all_activations = {layer: [] for layer in self.layers}
        
        for idx in tqdm(indices, desc="Collecting activations"):
            acts = dataset[idx]
            for layer in self.layers:
                if layer in acts:
                    all_activations[layer].append(acts[layer])
        
        # 合并
        result = {}
        for layer in self.layers:
            if all_activations[layer]:
                result[f"layer_{layer}"] = torch.cat(all_activations[layer], dim=0)
        
        # 保存
        torch.save(result, self.output_path)
        
        stats = {
            "total_samples": len(indices),
            "layers": self.layers,
            "output_path": str(self.output_path),
        }
        
        print(f"Saved to {self.output_path}")
        
        return stats


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="Cache Activations for SAE Training")
    parser.add_argument("--activation-dir", type=str, required=True, 
                        help="Directory containing activation files")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Output file path")
    parser.add_argument("--layers", type=int, nargs="+", default=[24, 27],
                        help="Layers to process")
    parser.add_argument("--max-samples", type=int, default=-1,
                        help="Maximum number of samples")
    parser.add_argument("--balance", action="store_true",
                        help="Balance CALL/NO_CALL samples")
    parser.add_argument("--decision-filter", type=str, default=None,
                        choices=["call", "no_call"],
                        help="Filter by decision type")
    
    args = parser.parse_args()
    
    cacher = ActivationCacher(
        activation_dir=args.activation_dir,
        output_path=args.output_path,
        layers=args.layers,
    )
    
    stats = cacher.process(
        decision_filter=args.decision_filter,
        max_samples=args.max_samples,
        balance=args.balance,
    )
    
    print("Processing complete!")
    print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
