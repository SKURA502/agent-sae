"""
Linear Probe - H1 线性可分性验证

用 top-K SAE 特征训练逻辑回归，评估 CALL/NO_CALL 线性可分性。目标：K=50 时 AUC > 0.80。

CLI:
  python -m analysis.linear_probe \\
    --layer 24 \\
    --sae-path outputs/sae_checkpoints/stage2/layer24/best.pt \\
    --activations-dir outputs/activations/test_mcq \\
    --feature-scores-path outputs/analysis/feature_scores_layer24.json \\
    --output-dir outputs/analysis \\
    --top-k 50

输出:
  outputs/analysis/linear_probe_layer{L}.json  - CV AUC by K 及全集 AUC
  outputs/analysis/linear_probe_layer{L}.png   - AUC vs K 曲线 + ROC 曲线（Fig 1b）
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold


def main():
    parser = argparse.ArgumentParser(
        description="H1 线性探针：用 top-K SAE 特征训练逻辑回归"
    )
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--sae-path", type=str, required=True)
    parser.add_argument("--activations-dir", type=str, required=True)
    parser.add_argument("--feature-scores-path", type=str, required=True,
                        help="feature_scores_layer{L}.json（来自 feature_discovery.py）")
    parser.add_argument("--output-dir", type=str, default="./outputs/analysis")
    parser.add_argument("--top-k", type=int, default=50, help="主评估 K 值")
    parser.add_argument("--k-values", type=int, nargs="+", default=[10, 20, 50, 100],
                        help="AUC vs K 曲线的 K 列表")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=512)
    args = parser.parse_args()

    from sae.sae_model import TopKSAE
    from analysis.feature_discovery import load_activations, encode_activations

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SAE from {args.sae_path}")
    sae = TopKSAE.load(args.sae_path, device=args.device)
    sae.eval()

    print(f"Loading activations: layer {args.layer}")
    acts, labels = load_activations(Path(args.activations_dir), args.layer, device="cpu")
    print(f"  {len(acts)} samples: {int(labels.sum())} CALL + {int((labels==0).sum())} NO_CALL")

    print("Encoding through SAE...")
    latents = encode_activations(sae, acts, batch_size=args.encode_batch_size)
    lat_np = latents.numpy().astype(np.float32)
    labels_np = labels.numpy()

    print(f"Loading feature scores from {args.feature_scores_path}")
    with open(args.feature_scores_path) as f:
        all_scores = json.load(f)
    all_scores.sort(key=lambda x: abs(x["mean_diff"]), reverse=True)
    all_indices = [s["feature_idx"] for s in all_scores]

    # ── AUC vs K ──────────────────────────────────────────────────────
    k_values = sorted(set(args.k_values + [args.top_k]))
    k_values = [k for k in k_values if k <= len(all_indices)]

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    results_by_k = {}

    for k in k_values:
        feat_indices = all_indices[:k]
        X = lat_np[:, feat_indices]

        fold_aucs = []
        for train_idx, val_idx in cv.split(X, labels_np):
            clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
            clf.fit(X[train_idx], labels_np[train_idx])
            prob = clf.predict_proba(X[val_idx])[:, 1]
            fold_aucs.append(roc_auc_score(labels_np[val_idx], prob))

        mean_auc = float(np.mean(fold_aucs))
        std_auc = float(np.std(fold_aucs))
        results_by_k[k] = {"mean_auc": mean_auc, "std_auc": std_auc}
        print(f"  K={k:4d}: mean AUC = {mean_auc:.4f} ± {std_auc:.4f}")

    # ── Full-set ROC with top-K ────────────────────────────────────────
    X_main = lat_np[:, all_indices[: args.top_k]]
    clf_main = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf_main.fit(X_main, labels_np)
    probs_main = clf_main.predict_proba(X_main)[:, 1]
    full_auc = float(roc_auc_score(labels_np, probs_main))
    print(f"\nFull-set AUC (top-{args.top_k}): {full_auc:.4f}")

    out_json = output_dir / f"linear_probe_layer{args.layer}.json"
    with open(out_json, "w") as f:
        json.dump(
            {
                "layer": args.layer,
                "top_k": args.top_k,
                "full_auc": full_auc,
                "cv_results_by_k": {str(k): v for k, v in results_by_k.items()},
            },
            f,
            indent=2,
        )
    print(f"Saved results → {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ks = sorted(results_by_k.keys())
        aucs = [results_by_k[k]["mean_auc"] for k in ks]
        stds = [results_by_k[k]["std_auc"] for k in ks]

        axes[0].errorbar(ks, aucs, yerr=stds, marker="o", capsize=4)
        axes[0].axhline(0.80, color="green", linestyle="--", label="target 0.80")
        axes[0].set_xlabel("K (number of features)")
        axes[0].set_ylabel("Mean AUC (5-fold CV)")
        axes[0].set_title(f"Linear Probe AUC vs K (layer {args.layer})")
        axes[0].legend()

        fpr, tpr, _ = roc_curve(labels_np, probs_main)
        axes[1].plot(fpr, tpr, label=f"AUC={full_auc:.3f}")
        axes[1].plot([0, 1], [0, 1], "k--", label="chance")
        axes[1].set_xlabel("FPR")
        axes[1].set_ylabel("TPR")
        axes[1].set_title(f"ROC Curve (top-{args.top_k} features, layer {args.layer})")
        axes[1].legend()

        plt.tight_layout()
        out_fig = output_dir / f"linear_probe_layer{args.layer}.png"
        plt.savefig(out_fig, dpi=150)
        plt.close()
        print(f"Saved figure → {out_fig}")
    except ImportError:
        print("matplotlib not available, skipping figure")


if __name__ == "__main__":
    main()
