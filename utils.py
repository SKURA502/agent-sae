"""共享 CLI 参数和工具函数"""

import argparse
from typing import List, Optional


def add_common_args(parser: argparse.ArgumentParser):
    """添加通用 CLI 参数：--model, --device, --dtype"""
    parser.add_argument("--model", type=str, required=True, help="LLM model name or path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32")


def add_sae_args(parser: argparse.ArgumentParser):
    """添加 SAE 训练相关 CLI 参数"""
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=4096, help="SAE training batch size")
    parser.add_argument("--dict-size", type=int, default=None,
                        help="SAE dictionary size (default: hidden_size * 8)")
    parser.add_argument("--k", type=int, default=None,
                        help="SAE top-k sparsity (default: hidden_size // 32)")
    parser.add_argument("--target-tokens", type=int, default=50_000_000)
    parser.add_argument("--use-swanlab", action="store_true", help="Use SwanLab for logging")


def add_dataset_args(parser: argparse.ArgumentParser):
    """添加数据集相关 CLI 参数"""
    parser.add_argument("--dataset", type=str, default="synthetic",
                        choices=["synthetic", "when2call", "bfcl"])
    parser.add_argument("--data-path", type=str, default=None, help="Dataset path")
    parser.add_argument("--num-samples", type=int, default=1000)


def load_samples(dataset: str, data_path: Optional[str], num_samples: int):
    """按数据集类型加载样本。"""
    from tasks import SyntheticGenerator, When2CallAdapter, BFCLAdapter

    if dataset == "synthetic":
        return SyntheticGenerator().generate()[:num_samples]

    if dataset == "when2call":
        adapter = When2CallAdapter(data_path or "./data/raw/when2call")
    elif dataset == "bfcl":
        adapter = BFCLAdapter(data_path or "./data/raw/bfcl")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    adapter.load()
    return list(adapter)[:num_samples]
