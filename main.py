"""
Agent SAE Tool-use Research Framework

主入口文件，整合所有模块。
"""

import argparse
import subprocess
import sys
import yaml
from pathlib import Path

from tasks import When2CallAdapter, BFCLAdapter, SyntheticGenerator
from run import RolloutGenerator
from analysis import CorrelationAnalyzer, LinearProbe, Visualizer


def load_model_config(config_path: str) -> dict:
    """加载模型配置文件。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_model_name(args) -> str:
    """优先使用 --model，否则从 model_config 读取模型名。"""
    if args.model:
        return args.model

    config = load_model_config(args.model_config)
    model_key = args.model_key or config.get("current_model")
    if not model_key:
        raise ValueError("No model selected. Please pass --model or --model-key.")

    model_info = config.get("models", {}).get(model_key)
    if not model_info:
        raise ValueError(f"Model key not found in model config: {model_key}")

    model_name = model_info.get("name")
    if not model_name:
        raise ValueError(f"Model config missing 'name' for key: {model_key}")

    return model_name


def cmd_generate_rollouts(args):
    """生成 rollouts"""
    model_name = resolve_model_name(args)
    
    generator = RolloutGenerator(
        model_name=model_name,
        output_dir=args.output_dir,
        hook_layers=args.layers,
        device=args.device,
        dtype=args.dtype,
    )
    
    # 加载数据集
    if args.dataset == "when2call":
        adapter = When2CallAdapter(args.data_path or "./data/raw/when2call", split=args.split)
        adapter.load()
        samples = list(adapter)[:args.num_samples]
    elif args.dataset == "bfcl":
        adapter = BFCLAdapter(args.data_path or "./data/raw/bfcl", split=args.split)
        adapter.load()
        samples = list(adapter)[:args.num_samples]
    else:
        generator_config = None
        if args.seed is not None:
            from tasks.synthetic_generator import GeneratorConfig
            generator_config = GeneratorConfig(num_samples=args.num_samples, seed=args.seed)
        adapter = SyntheticGenerator(config=generator_config)
        samples = adapter.generate()[:args.num_samples]
    
    generator.run(samples, experiment_name=args.experiment_name)


def cmd_cache_activations(args):
    """透传到 run.cache_activations CLI。"""
    forward_args = args.cache_args if args.cache_args else ["--help"]
    cmd = [sys.executable, "-m", "run.cache_activations", *forward_args]
    subprocess.run(cmd, check=True)


def cmd_train_sae(args):
    """透传到 sae.train_sae CLI。"""
    if not args.train_args:
        cmd = [sys.executable, "-m", "sae.train_sae", "--help"]
    else:
        cmd = [sys.executable, "-m", "sae.train_sae", *args.train_args]
    subprocess.run(cmd, check=True)


def cmd_analyze(args):
    """运行分析"""
    import torch
    
    data = torch.load(args.data_path)
    activations = data[f"layer_{args.layer}"]
    labels = data["labels"]
    
    if args.analysis_type == "correlation":
        analyzer = CorrelationAnalyzer(args.sae_path, device=args.device)
        results = analyzer.analyze(activations, labels, output_dir=args.output_dir)
        print(f"Significant features: {results['significance']['num_significant']}")
        
    elif args.analysis_type == "probe":
        probe = LinearProbe(args.sae_path, device=args.device)
        results = probe.probe(activations, labels, output_dir=args.output_dir)
        print(f"All features AUC: {results['all_features_auc']:.4f}")
        
    elif args.analysis_type == "visualize":
        # 加载之前的分析结果
        import json
        with open(Path(args.output_dir) / "correlation_analysis.json") as f:
            corr_results = json.load(f)
        
        viz = Visualizer(args.output_dir)
        viz.plot_feature_separability(
            corr_results["auroc"]["scores"],
            corr_results["mean_diff"]["difference"],
        )


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description="Agent SAE Tool-use Research")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")
    
    # generate-rollouts
    p_rollout = subparsers.add_parser("generate-rollouts", help="Generate rollouts")
    p_rollout.add_argument("--model", type=str, default=None)
    p_rollout.add_argument("--model-key", type=str, default=None,
                          help="Model key from model_config.yaml (fallback when --model is not set)")
    p_rollout.add_argument("--model-config", type=str,
                          default=str(Path(__file__).parent / "configs" / "model_config.yaml"),
                          help="Path to model_config.yaml")
    p_rollout.add_argument("--dataset", type=str, default="synthetic",
                          choices=["when2call", "bfcl", "synthetic"])
    p_rollout.add_argument("--data-path", type=str, default=None,
                          help="Dataset root path for when2call/bfcl")
    p_rollout.add_argument("--split", type=str, default="test",
                          help="Dataset split for adapters")
    p_rollout.add_argument("--num-samples", type=int, default=1000)
    p_rollout.add_argument("--layers", type=int, nargs="+", default=[24, 27],
                          help="Hook layers")
    p_rollout.add_argument("--experiment-name", type=str, default="rollout")
    p_rollout.add_argument("--seed", type=int, default=None,
                          help="Synthetic generator seed")
    p_rollout.add_argument("--output-dir", type=str, default="./outputs/rollouts")
    p_rollout.add_argument("--device", type=str, default="cuda")
    p_rollout.add_argument("--dtype", type=str, default="bfloat16")
    
    # cache-activations
    p_cache = subparsers.add_parser(
        "cache-activations",
        help="Forward to python -m run.cache_activations"
    )
    p_cache.add_argument("cache_args", nargs=argparse.REMAINDER,
                        help="Arguments forwarded to run.cache_activations")
    
    # train-sae
    p_train = subparsers.add_parser(
        "train-sae",
        help="Forward to python -m sae.train_sae"
    )
    p_train.add_argument("train_args", nargs=argparse.REMAINDER,
                        help="Arguments forwarded to sae.train_sae")
    
    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Run analysis")
    p_analyze.add_argument("--analysis-type", type=str, required=True,
                          choices=["correlation", "probe", "visualize"])
    p_analyze.add_argument("--sae-path", type=str, required=True)
    p_analyze.add_argument("--data-path", type=str, required=True)
    p_analyze.add_argument("--layer", type=int, required=True)
    p_analyze.add_argument("--output-dir", type=str, default="./outputs/analysis")
    p_analyze.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    if args.command == "generate-rollouts":
        cmd_generate_rollouts(args)
    elif args.command == "cache-activations":
        cmd_cache_activations(args)
    elif args.command == "train-sae":
        cmd_train_sae(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
