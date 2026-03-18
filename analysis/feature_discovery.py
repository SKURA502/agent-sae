"""
Feature Discovery - H1 特征发现

计算每个 SAE feature 相对于 CALL/NO_CALL 标签的 mean_diff 和 AUROC，筛选 top-K 门控 feature。

CLI:
  python -m analysis.feature_discovery \\
    --layer 24 \\
    --sae-path outputs/sae_checkpoints/stage2/layer24/best.pt \\
    --activations-dir outputs/activations/train_pref \\
    --output-dir outputs/analysis \\
    --top-k 100

输入:
  cache_activations.py 输出的 layer_{L}_activations.pt，格式：
    {"layer_{L}": Tensor[N, hidden], "labels": Tensor[N]}  (1=CALL, 0=NO_CALL)

输出:
  outputs/analysis/feature_scores_layer{L}.json   - 全部特征的 mean_diff + AUROC
  outputs/analysis/top_features_layer{L}.json     - top-K 特征（按 |mean_diff| 排序）
  outputs/analysis/feature_discovery_layer{L}.png - AUROC 分布直方图（Fig 1a）
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# ─────────────────────── data loading ──────────────────────────────

def load_activations(
    activations_dir: Path, layer: int, device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """加载 cache_activations.py 的输出。

    Returns:
        acts:   [N, hidden_size]  float32
        labels: [N]               long (1=CALL, 0=NO_CALL)
    """
    path = activations_dir / f"layer_{layer}_activations.pt"
    ckpt = torch.load(path, map_location=device, weights_only=True)
    acts = ckpt[f"layer_{layer}"].to(torch.float32)
    labels = ckpt["labels"].long()
    return acts, labels


# ─────────────────────── SAE encoding ──────────────────────────────

def encode_activations(
    sae, acts: torch.Tensor, batch_size: int = 512
) -> torch.Tensor:
    """分批通过 SAE encode，返回 [N, dict_size] 稀疏激活。"""
    latents_list = []
    for i in range(0, len(acts), batch_size):
        batch = acts[i : i + batch_size].to(sae.config.device)
        with torch.no_grad():
            lat = sae.encode(batch)
        latents_list.append(lat.cpu())
    return torch.cat(latents_list, dim=0)


# ─────────────────────── scoring ───────────────────────────────────

def compute_feature_scores(
    latents: torch.Tensor, labels: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray]:
    """计算每个 feature 的 mean_diff 和 AUROC。

    Returns:
        mean_diff: [dict_size]  E[f|CALL] − E[f|NO_CALL]
        auroc:     [dict_size]  per-feature AUROC（对 CALL 正类）
    """
    lat_np = latents.numpy().astype(np.float32)
    labels_np = labels.numpy()

    call_mask = labels_np == 1
    no_call_mask = labels_np == 0

    mean_call = lat_np[call_mask].mean(axis=0)
    mean_no_call = lat_np[no_call_mask].mean(axis=0)
    mean_diff = mean_call - mean_no_call

    n_features = lat_np.shape[1]
    auroc = np.full(n_features, 0.5, dtype=np.float32)

    for i in tqdm(range(n_features), desc="Computing AUROC", leave=False):
        col = lat_np[:, i]
        if col.max() == col.min():
            continue
        try:
            auroc[i] = roc_auc_score(labels_np, col)
        except Exception:
            pass

    return mean_diff, auroc


# ─────────────────────── plotting ──────────────────────────────────

def _plot_distributions(mean_diff: np.ndarray, auroc: np.ndarray, layer: int, out_path: Path):
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].hist(auroc, bins=60, edgecolor="black", alpha=0.75)
        axes[0].axvline(0.5, color="red", linestyle="--", label="chance")
        axes[0].axvline(0.75, color="green", linestyle="--", label="target 0.75")
        axes[0].set_xlabel("AUROC")
        axes[0].set_ylabel("# Features")
        axes[0].set_title(f"Feature AUROC Distribution (layer {layer})")
        axes[0].legend()

        axes[1].hist(mean_diff, bins=60, edgecolor="black", alpha=0.75)
        axes[1].axvline(0, color="red", linestyle="--")
        axes[1].set_xlabel("Mean Diff (E[f|CALL] − E[f|NO_CALL])")
        axes[1].set_ylabel("# Features")
        axes[1].set_title(f"Feature Mean Diff Distribution (layer {layer})")

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved figure → {out_path}")
    except ImportError:
        print("matplotlib not available, skipping figure")


# ─────────────────────── CLI ───────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="H1 特征发现：计算 per-feature mean_diff 和 AUROC"
    )
    parser.add_argument("--layer", type=int, required=True, help="目标层索引")
    parser.add_argument("--sae-path", type=str, required=True, help="SAE checkpoint 路径")
    parser.add_argument("--activations-dir", type=str, required=True,
                        help="cache_activations.py 的输出目录")
    parser.add_argument("--output-dir", type=str, default="./outputs/analysis")
    parser.add_argument("--top-k", type=int, default=100, help="保存 top-K 特征（按 |mean_diff|）")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=512)
    args = parser.parse_args()

    from sae.sae_model import TopKSAE

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SAE from {args.sae_path}")
    sae = TopKSAE.load(args.sae_path, device=args.device)
    sae.eval()

    print(f"Loading activations: layer {args.layer} from {args.activations_dir}")
    acts, labels = load_activations(Path(args.activations_dir), args.layer, device="cpu")
    n_call = int(labels.sum())
    n_no_call = int((labels == 0).sum())
    print(f"  {len(acts)} samples: {n_call} CALL + {n_no_call} NO_CALL")

    print("Encoding activations through SAE...")
    latents = encode_activations(sae, acts, batch_size=args.encode_batch_size)
    print(f"  Latents: {latents.shape}")

    print("Computing feature scores...")
    mean_diff, auroc = compute_feature_scores(latents, labels)

    # 所有特征分数
    all_scores = [
        {"feature_idx": int(i), "mean_diff": float(mean_diff[i]), "auroc": float(auroc[i])}
        for i in range(len(mean_diff))
    ]
    out_scores = output_dir / f"feature_scores_layer{args.layer}.json"
    with open(out_scores, "w") as f:
        json.dump(all_scores, f)
    print(f"Saved {len(all_scores)} feature scores → {out_scores}")

    # Top-K（按 |mean_diff| 降序）
    sorted_by_diff = sorted(all_scores, key=lambda x: abs(x["mean_diff"]), reverse=True)
    top_features = sorted_by_diff[: args.top_k]
    out_top = output_dir / f"top_features_layer{args.layer}.json"
    with open(out_top, "w") as f:
        json.dump(top_features, f, indent=2)
    print(f"Saved top-{args.top_k} features → {out_top}")

    print(f"\nTop 10 features (|mean_diff| ranking):")
    for entry in top_features[:10]:
        print(f"  feat {entry['feature_idx']:5d}: "
              f"mean_diff={entry['mean_diff']:+.4f}  auroc={entry['auroc']:.4f}")

    _plot_distributions(
        mean_diff, auroc, args.layer,
        output_dir / f"feature_discovery_layer{args.layer}.png"
    )


if __name__ == "__main__":
    main()
