"""
Agent SAE Tool-use Research Framework

主入口文件，整合所有模块。
"""

import argparse
import yaml
from pathlib import Path

from controller import AgentLoop, AgentConfig
from tasks import When2CallAdapter, BFCLAdapter, SyntheticGenerator
from run import RolloutGenerator, ActivationCacher
from sae import TopKSAE, FeatureExtractor
from analysis import CorrelationAnalyzer, LinearProbe, SteeringExperiment, Visualizer


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def cmd_generate_rollouts(args):
    """生成 rollouts"""
    config = load_config(args.config) if args.config else {}
    
    generator = RolloutGenerator(
        model_name_or_path=args.model or config.get("model_path"),
        output_dir=args.output_dir,
        device=args.device,
    )
    
    # 加载数据集
    if args.dataset == "when2call":
        adapter = When2CallAdapter(config.get("when2call_path", ""))
    elif args.dataset == "bfcl":
        adapter = BFCLAdapter(config.get("bfcl_path", ""))
    else:
        adapter = SyntheticGenerator()
    
    samples = adapter.get_samples(args.num_samples)
    generator.generate_batch(samples)


def cmd_cache_activations(args):
    """缓存激活值"""
    cacher = ActivationCacher(
        model_name_or_path=args.model,
        hook_layers=list(map(int, args.layers.split(","))),
        output_dir=args.output_dir,
        device=args.device,
    )
    
    cacher.cache_from_rollout_dir(args.rollout_dir)


def cmd_train_sae(args):
    """训练 SAE（两阶段训练入口）"""
    from sae.train_sae import TwoStageTrainer

    print("SAE training now uses two-stage mode only.")
    print("Please run: python -m sae.train_sae stage1 / stage2")
    print("See sae/train_sae.py --help for details.")


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
    p_rollout.add_argument("--model", type=str, required=True)
    p_rollout.add_argument("--dataset", type=str, default="synthetic",
                          choices=["when2call", "bfcl", "synthetic"])
    p_rollout.add_argument("--num-samples", type=int, default=1000)
    p_rollout.add_argument("--output-dir", type=str, default="./outputs/rollouts")
    p_rollout.add_argument("--config", type=str, default=None)
    p_rollout.add_argument("--device", type=str, default="cuda")
    
    # cache-activations
    p_cache = subparsers.add_parser("cache-activations", help="Cache activations")
    p_cache.add_argument("--model", type=str, required=True)
    p_cache.add_argument("--rollout-dir", type=str, required=True)
    p_cache.add_argument("--layers", type=str, default="20,24",
                        help="Comma-separated layer indices")
    p_cache.add_argument("--output-dir", type=str, default="./outputs/activations")
    p_cache.add_argument("--device", type=str, default="cuda")
    
    # train-sae
    p_train = subparsers.add_parser("train-sae", help="Train SAE (prints two-stage usage)")
    p_train.add_argument("--device", type=str, default="cuda")
    
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
