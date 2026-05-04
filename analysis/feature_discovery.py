"""
Feature Discovery - H1 特征发现

计算每个 SAE feature 相对于 TOOL_CALL/OTHER 标签的 mean_diff 和 AUROC，筛选 top-K 门控 feature。

CLI:
  python -m analysis.feature_discovery \\
    --layer 25 \\
    --sae-path outputs/sae_checkpoints/stage2/layer25/best.pt \\
    --activations-dir outputs/activations/when2call_mcq \\
    --output-dir outputs/analysis \\
    --top-k 30

输入:
  cache_activations.py 输出的 layer_{L}_activations.pt，格式：
    {"layer_{L}": Tensor[N, hidden], "labels": Tensor[N]}
    labels: 1=TOOL_CALL, 0/2/3=OTHER（非 tool_call 类别）

输出:
  outputs/analysis/feature_scores_layer{L}.json        - 全部特征的 mean_diff + AUROC
  outputs/analysis/top_features_layer{L}.json          - top-K 特征（按 |mean_diff| 排序）
  outputs/analysis/feature_discovery_layer{L}.png      - AUROC 分布 + mean_diff 分布图
  outputs/analysis/top_features_bar_layer{L}.png       - top-K 特征 mean_diff 柱状图
  outputs/analysis/decoder_umap_layer{L}.png           - SAE decoder latent UMAP + KDE
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
        batch = acts[i : i + batch_size].to(sae.config.device, dtype=sae.config.get_torch_dtype())
        with torch.no_grad():
            lat = sae.encode(batch)
        latents_list.append(lat.cpu())
    return torch.cat(latents_list, dim=0)


# ─────────────────────── scoring ───────────────────────────────────

CONCEPT_LABEL = {
    "tool_call": 1,
    "request_for_info": 2,
}


def compute_feature_scores(
    latents: torch.Tensor, labels: torch.Tensor, positive_label: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算每个 feature 的 mean_diff 和 AUROC。

    positive_label: 正类标签索引（1=tool_call, 2=request_for_info）
    其余标签均视为负类（OTHER）

    Returns:
        mean_diff: [dict_size]  E[f|POSITIVE] − E[f|OTHER]
        auroc:     [dict_size]  per-feature AUROC（对正类）
    """
    lat_np = latents.float().numpy().astype(np.float32)
    labels_np = labels.numpy()

    call_mask = labels_np == positive_label
    no_call_mask = labels_np != positive_label
    binary_labels = call_mask.astype(np.int32)

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
            auroc[i] = roc_auc_score(binary_labels, col)
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


def _plot_top_features_bar(top_features: list, layer: int, out_path: Path, concept: str = "tool_call"):
    """双指标柱状图：柱高=mean_diff，柱色=AUROC（colormap），右轴折线也显示 AUROC。

    筛选后的 top_features 已经同时满足 mean_diff 和 AUROC 阈值，按 mean_diff 降序。
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        if not top_features:
            print("No features to plot in bar chart (empty list)")
            return

        entries = top_features
        n = len(entries)
        feat_ids = [str(e["feature_idx"]) for e in entries]
        diffs = np.array([e["mean_diff"] for e in entries])
        aurocs = np.array([e["auroc"] for e in entries])

        # 柱子颜色用 AUROC 映射到 colormap（越高越红）
        norm = mcolors.Normalize(vmin=max(0.5, aurocs.min() - 0.02), vmax=min(1.0, aurocs.max() + 0.02))
        cmap = cm.get_cmap("RdYlGn")
        bar_colors = [cmap(norm(a)) for a in aurocs]

        fig, ax1 = plt.subplots(figsize=(max(10, n * 0.5), 5))

        # ── 主轴：mean_diff 柱状图 ──
        bars = ax1.bar(
            range(n), diffs, color=bar_colors,
            edgecolor="white", linewidth=0.5, zorder=2,
        )
        ax1.axhline(0, color="black", linewidth=0.8, zorder=1)
        ax1.set_xticks(range(n))
        ax1.set_xticklabels(feat_ids, rotation=90, fontsize=7)
        ax1.set_xlabel("Feature Index")
        ax1.set_ylabel(f"Mean Diff  (E[f|{concept}] − E[f|other])", color="black")
        ax1.tick_params(axis="y", labelcolor="black")

        # ── 副轴：AUROC 折线 + 散点 ──
        ax2 = ax1.twinx()
        ax2.plot(range(n), aurocs, color="#333333", linewidth=1.2,
                 linestyle="--", zorder=3, label="AUROC")
        ax2.scatter(range(n), aurocs, c=bar_colors, s=40,
                    edgecolors="#333333", linewidths=0.5, zorder=4)
        ax2.set_ylabel("AUROC", color="#333333")
        ax2.tick_params(axis="y", labelcolor="#333333")
        ax2.set_ylim(0.5, 1.05)
        ax2.axhline(0.5, color="#333333", linewidth=0.5, linestyle=":")

        # ── Colorbar 说明 AUROC 颜色 ──
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax2, pad=0.08, fraction=0.03)
        cbar.set_label("AUROC (bar color)", fontsize=8)

        ax1.set_title(
            f"Top-{n} {concept} Features  (layer {layer})\n"
            f"bars = mean_diff,  color & right axis = AUROC",
            fontsize=10,
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved bar chart → {out_path}")
    except ImportError:
        print("matplotlib not available, skipping bar chart")


def _plot_decoder_umap(
    sae,
    top_feature_indices: list,
    layer: int,
    out_path: Path,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    concept: str = "tool_call",
):
    """把 SAE decoder latent（每列=一个 concept）做 UMAP 降维，
    背景用 KDE 显示密度，top tool_call feature 高亮。

    decoder.weight: [input_dim, dict_size] → 每列是一个 concept 的方向向量
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from scipy.stats import gaussian_kde
    except ImportError:
        print("matplotlib/scipy not available, skipping UMAP plot")
        return

    try:
        import umap as umap_lib
    except ImportError:
        print("umap-learn not available (pip install umap-learn), skipping UMAP plot")
        return

    print("Extracting decoder vectors...")
    with torch.no_grad():
        # decoder.weight: [input_dim, dict_size] → T → [dict_size, input_dim]
        W = sae.decoder.weight.detach().float().cpu().T.numpy()  # [dict_size, input_dim]

    dict_size = W.shape[0]
    print(f"  Decoder matrix: {W.shape}  →  running UMAP...")

    reducer = umap_lib.UMAP(
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        n_components=2,
        metric="cosine",
        random_state=42,
        verbose=False,
    )
    embedding = reducer.fit_transform(W)  # [dict_size, 2]
    print("  UMAP done.")

    top_set = set(top_feature_indices)
    is_top = np.array([i in top_set for i in range(dict_size)], dtype=bool)

    fig, ax = plt.subplots(figsize=(10, 8))

    # ── KDE 背景（全部 concept）──
    kde = gaussian_kde(embedding.T, bw_method=0.15)
    x_min, x_max = embedding[:, 0].min() - 1, embedding[:, 0].max() + 1
    y_min, y_max = embedding[:, 1].min() - 1, embedding[:, 1].max() + 1
    grid_x, grid_y = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    grid_pts = np.vstack([grid_x.ravel(), grid_y.ravel()])
    density = kde(grid_pts).reshape(200, 200)
    ax.contourf(grid_x, grid_y, density, levels=15, cmap="Blues", alpha=0.45)

    # ── 所有 concept（灰色小点）──
    ax.scatter(
        embedding[~is_top, 0], embedding[~is_top, 1],
        s=3, c="gray", alpha=0.25, linewidths=0, label="all concepts",
    )

    # ── top tool_call concept（高亮）──
    if is_top.any():
        sc = ax.scatter(
            embedding[is_top, 0], embedding[is_top, 1],
            s=60, c="#E07B39", alpha=0.9, linewidths=0.5,
            edgecolors="white", zorder=5, label=f"top {concept} features ({is_top.sum()})",
        )
        # 标注 feature idx
        for idx in top_feature_indices:
            ax.annotate(
                str(idx),
                xy=(embedding[idx, 0], embedding[idx, 1]),
                fontsize=6, color="#C04000",
                xytext=(3, 3), textcoords="offset points",
            )

    ax.set_title(f"SAE Decoder Latent UMAP  (layer {layer},  cosine,  KDE background)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved UMAP plot → {out_path}")


# ─────────────────────── CLI ───────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="特征发现：计算 per-feature mean_diff 和 AUROC（指定概念 vs OTHER）"
    )
    parser.add_argument("--layer", type=int, required=True, help="目标层索引")
    parser.add_argument("--sae-path", type=str, required=True, help="SAE checkpoint 路径")
    parser.add_argument("--activations-dir", type=str, required=True,
                        help="cache_activations.py 的输出目录")
    parser.add_argument("--output-dir", type=str, default="./outputs/analysis")
    parser.add_argument("--concept", type=str, default="tool_call",
                        choices=list(CONCEPT_LABEL.keys()),
                        help="目标概念：tool_call 或 request_for_info（默认 tool_call）")
    parser.add_argument("--top-k", type=int, default=30,
                        help="联合筛选后最多保留 top-K 特征（按 mean_diff 降序）")
    parser.add_argument("--min-mean-diff", type=float, default=0.0,
                        help="筛选阈值：mean_diff 必须 > 此值（默认 0，即只保留正向相关）")
    parser.add_argument("--min-auroc", type=float, default=0.6,
                        help="筛选阈值：AUROC 必须 > 此值（默认 0.6）")
    parser.add_argument("--umap-top-k", type=int, default=20,
                        help="UMAP 图中高亮的 top-K feature 数量")
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=512)
    args = parser.parse_args()

    concept = args.concept
    positive_label = CONCEPT_LABEL[concept]

    from sae.sae_model import TopKSAE

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SAE from {args.sae_path}")
    sae = TopKSAE.load(args.sae_path, device=args.device)
    sae.eval()

    print(f"Loading activations: layer {args.layer} from {args.activations_dir}")
    acts, labels = load_activations(Path(args.activations_dir), args.layer, device="cpu")
    labels_np = labels.numpy()
    n_pos = int((labels_np == positive_label).sum())
    n_other = int((labels_np != positive_label).sum())
    print(f"  {len(acts)} samples: {n_pos} {concept.upper()} + {n_other} OTHER "
          f"(label dist: { {v: int((labels_np==v).sum()) for v in sorted(set(labels_np.tolist()))} })")

    print("Encoding activations through SAE...")
    latents = encode_activations(sae, acts, batch_size=args.encode_batch_size)
    print(f"  Latents: {latents.shape}")

    print("Computing feature scores...")
    mean_diff, auroc = compute_feature_scores(latents, labels, positive_label=positive_label)

    # 所有特征分数
    all_scores = [
        {"feature_idx": int(i), "mean_diff": float(mean_diff[i]), "auroc": float(auroc[i])}
        for i in range(len(mean_diff))
    ]
    out_scores = output_dir / f"feature_scores_layer{args.layer}.json"
    with open(out_scores, "w") as f:
        json.dump(all_scores, f)
    print(f"Saved {len(all_scores)} feature scores → {out_scores}")

    # 联合筛选：mean_diff > min_mean_diff  AND  auroc > min_auroc，按 mean_diff 降序取 top-K
    filtered = [
        e for e in all_scores
        if e["mean_diff"] > args.min_mean_diff and e["auroc"] > args.min_auroc
    ]
    filtered.sort(key=lambda x: x["mean_diff"], reverse=True)
    top_features = filtered[: args.top_k]

    print(f"\n  Filter: mean_diff > {args.min_mean_diff},  auroc > {args.min_auroc}")
    print(f"  {len(filtered)} features passed filter → keeping top-{len(top_features)}")

    out_top = output_dir / f"top_features_layer{args.layer}.json"
    with open(out_top, "w") as f:
        json.dump(top_features, f, indent=2)
    print(f"Saved top features → {out_top}")

    print(f"\nTop 10 {concept} features:")
    for entry in top_features[:10]:
        print(f"  feat {entry['feature_idx']:5d}: "
              f"mean_diff={entry['mean_diff']:+.4f}  auroc={entry['auroc']:.4f}")

    # ── Fig 1: AUROC + mean_diff 分布 ──
    _plot_distributions(
        mean_diff, auroc, args.layer,
        output_dir / f"feature_discovery_layer{args.layer}.png"
    )

    # ── Fig 2: top-K 柱状图（mean_diff + AUROC 双指标）──
    _plot_top_features_bar(
        top_features, args.layer,
        output_dir / f"top_features_bar_layer{args.layer}.png",
        concept=concept,
    )

    # ── Fig 3: decoder UMAP + KDE ──
    umap_top_indices = [e["feature_idx"] for e in top_features[: args.umap_top_k]]
    _plot_decoder_umap(
        sae,
        umap_top_indices,
        args.layer,
        output_dir / f"decoder_umap_layer{args.layer}.png",
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        concept=concept,
    )


if __name__ == "__main__":
    main()
