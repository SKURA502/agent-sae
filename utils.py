"""共享 CLI 参数和工具函数"""

import argparse
from typing import List, Optional


def add_common_args(parser: argparse.ArgumentParser):
    """添加通用 CLI 参数：--model, --device, --dtype"""
    parser.add_argument("--model", type=str, required=True, help="LLM model name or path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")


def add_sae_args(parser: argparse.ArgumentParser):
    """添加 SAE 训练相关 CLI 参数"""
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=8192, help="SAE training batch size")
    parser.add_argument("--buffer-size", type=int, default=None,
                        help="Streaming activation buffer size (default: same as --batch-size)")
    parser.add_argument("--drop-last", action="store_true",
                        help="Drop final incomplete SAE batch in streaming training")
    parser.add_argument("--dict-size", type=int, default=None,
                        help="SAE dictionary size (default: hidden_size * 8)")
    parser.add_argument("--k", type=int, default=None,
                        help="SAE top-k sparsity (default: hidden_size // 32)")
    parser.add_argument("--target-tokens", type=int, default=50_000_000)
    parser.add_argument("--use-swanlab", action="store_true", help="Use SwanLab for logging")


def add_stage_args(parser: argparse.ArgumentParser):
    """stage1 / stage2 共享参数。"""
    add_common_args(parser)
    add_sae_args(parser)
    parser.add_argument("--layer", type=int, default=23, help="Target layer of the LLM to attach SAE(starting from 0)")
    parser.add_argument("--output-dir", type=str,
                        default="./outputs/sae_checkpoints")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing training JSONL files")
    parser.add_argument("--seq-length", type=int, default=1024,
                        help="Tokenization sequence length")
    parser.add_argument("--inference-batch-size", type=int, default=32,
                        help="LLM inference batch size for activation extraction")


def add_dataset_args(parser: argparse.ArgumentParser):
    """添加数据集相关 CLI 参数"""
    parser.add_argument("--dataset", type=str, default="when2call",
                        choices=["when2call"])
    parser.add_argument("--data-path", type=str, default=None,
                        help="Dataset path (default: data/raw/When2Call/data/test)")
    parser.add_argument("--split", type=str, default="test_mcq",
                        help="Dataset split: train_pref / train_sft / test_mcq")
    parser.add_argument("--num-samples", type=int, default=-1,
                        help="Number of samples to load (-1 = all)")


def load_samples(dataset: str, data_path: Optional[str], num_samples: int,
                 split: str = "test_mcq"):
    """加载 When2Call 样本，默认加载 test_mcq 二类子集（排除 UNCERTAIN）。"""
    from run.when2call_adapter import When2CallAdapter, DecisionLabel

    default_path = {
        "test_mcq": "./data/raw/When2Call/data/test",
        "train_pref": "./data/raw/When2Call/data/train",
        "train_sft": "./data/raw/When2Call/data/train",
    }.get(split, "./data/raw/When2Call/data/test")

    adapter = When2CallAdapter(data_path or default_path, split=split)
    adapter.load()
    samples = [s for s in adapter if s.label != DecisionLabel.UNCERTAIN]
    if num_samples > 0:
        samples = samples[:num_samples]
    return samples
