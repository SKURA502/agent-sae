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
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
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
    parser.add_argument("--balance-classes", action="store_true",
                        help="随机下采样正样本使正负样本数量一致（保留全部负样本）")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=512)
    parser.add_argument("--label", type=str, default="tool_call",
                        choices=["tool_call", "request_for_info"],
                        help="正类标签，决定预测目标（默认 tool_call）")
    args = parser.parse_args()

    from sae.sae_model import TopKSAE
    from analysis.feature_discovery import load_activations, encode_activations

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SAE from {args.sae_path}")
    sae = TopKSAE.load(args.sae_path, device=args.device)
    sae.eval()

    from utils.templates import MCQ_CHOICES
    target_idx = MCQ_CHOICES.index(args.label)

    print(f"Loading activations: layer {args.layer}")
    acts, labels = load_activations(Path(args.activations_dir), args.layer, device="cpu")
    n_target = int((labels == target_idx).sum())
    print(f"  {len(acts)} samples: {n_target} {args.label.upper()} + {len(acts)-n_target} OTHER")

    print("Encoding through SAE...")
    latents = encode_activations(sae, acts, batch_size=args.encode_batch_size)
    lat_np = latents.float().numpy().astype(np.float32)
    acts_np = acts.float().numpy().astype(np.float32)
    # 二值化：目标 label=1，其余=0
    labels_np = (labels.numpy() == target_idx).astype(np.int32)

    n_pos = int(labels_np.sum())
    n_neg = int((labels_np == 0).sum())
    pos_rate = n_pos / len(labels_np)
    print(f"  Class balance: {n_pos} {args.label} ({pos_rate:.1%}) / {n_neg} other ({1-pos_rate:.1%})")

    keep = None
    if args.balance_classes:
        rng = np.random.default_rng(42)
        pos_indices = np.where(labels_np == 1)[0]
        neg_indices = np.where(labels_np == 0)[0]
        sampled_pos = rng.choice(pos_indices, size=n_neg, replace=False)
        keep = np.sort(np.concatenate([sampled_pos, neg_indices]))
        lat_np = lat_np[keep]
        acts_np = acts_np[keep]
        labels_np = labels_np[keep]
        n_pos = int(labels_np.sum())
        n_neg = int((labels_np == 0).sum())
        pos_rate = n_pos / len(labels_np)
        print(f"  After balancing: {n_pos} tool_call / {n_neg} other (total={len(labels_np)})")

    print(f"Loading feature scores from {args.feature_scores_path}")
    with open(args.feature_scores_path) as f:
        all_scores = json.load(f)
    # 与 feature_discovery 一致：按 mean_diff 降序（已是正向筛选后的列表）
    all_scores.sort(key=lambda x: x["mean_diff"], reverse=True)
    all_indices = [s["feature_idx"] for s in all_scores]

    dict_size = lat_np.shape[1]

    # ── AUC vs K ──────────────────────────────────────────────────────
    k_values = sorted(set(args.k_values + [args.top_k]))
    k_values = [k for k in k_values if k <= len(all_indices)]

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    results_by_k = {}

    for k in k_values:
        feat_indices = all_indices[:k]
        X = lat_np[:, feat_indices]

        fold_aucs, fold_f1_neg, fold_f1_pos = [], [], []
        for train_idx, val_idx in cv.split(X, labels_np):
            y_val = labels_np[val_idx]
            clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
            clf.fit(X[train_idx], labels_np[train_idx])
            prob = clf.predict_proba(X[val_idx])[:, 1]
            pred = clf.predict(X[val_idx])
            fold_aucs.append(roc_auc_score(y_val, prob))
            fold_f1_neg.append(f1_score(y_val, pred, pos_label=0, zero_division=0))
            fold_f1_pos.append(f1_score(y_val, pred, pos_label=1, zero_division=0))

        mean_auc  = float(np.mean(fold_aucs))
        std_auc   = float(np.std(fold_aucs))
        mean_f1_neg = float(np.mean(fold_f1_neg))
        std_f1_neg  = float(np.std(fold_f1_neg))
        mean_f1_pos = float(np.mean(fold_f1_pos))
        results_by_k[k] = {
            "mean_auc": mean_auc, "std_auc": std_auc,
            "mean_f1_neg": mean_f1_neg, "std_f1_neg": std_f1_neg,
            "mean_f1_pos": mean_f1_pos,
        }
        print(f"  K={k:4d}: AUC={mean_auc:.4f}±{std_auc:.4f}  "
              f"F1_neg={mean_f1_neg:.4f}±{std_f1_neg:.4f}  "
              f"F1_pos={mean_f1_pos:.4f}")

    # ── Raw activation baseline ────────────────────────────────────────
    print("\nComputing raw activation baseline...")
    raw_fold_aucs = []
    for train_idx, val_idx in cv.split(acts_np, labels_np):
        clf_raw = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf_raw.fit(acts_np[train_idx], labels_np[train_idx])
        prob_raw = clf_raw.predict_proba(acts_np[val_idx])[:, 1]
        raw_fold_aucs.append(roc_auc_score(labels_np[val_idx], prob_raw))
    raw_mean_auc = float(np.mean(raw_fold_aucs))
    raw_std_auc  = float(np.std(raw_fold_aucs))
    print(f"  Raw activation baseline: AUC={raw_mean_auc:.4f}±{raw_std_auc:.4f}")

    # ── Random SAE features baseline (5 seeds, per K) ─────────────────
    print("\nComputing random SAE features baseline (5 seeds × K)...")
    n_random_seeds = 20
    random_results_by_k = {}

    for k in k_values:
        seed_aucs = []
        for seed in range(n_random_seeds):
            rng_seed = np.random.default_rng(seed * 1000 + k)
            rand_indices = rng_seed.choice(dict_size, size=k, replace=False).tolist()
            X_rand = lat_np[:, rand_indices]
            fold_aucs_rand = []
            for train_idx, val_idx in cv.split(X_rand, labels_np):
                clf_rand = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
                clf_rand.fit(X_rand[train_idx], labels_np[train_idx])
                prob_rand = clf_rand.predict_proba(X_rand[val_idx])[:, 1]
                fold_aucs_rand.append(roc_auc_score(labels_np[val_idx], prob_rand))
            seed_aucs.append(float(np.mean(fold_aucs_rand)))
        random_results_by_k[k] = {
            "mean_auc": float(np.mean(seed_aucs)),
            "std_auc":  float(np.std(seed_aucs)),
            "seed_aucs": seed_aucs,
        }
        print(f"  K={k:4d}: random AUC={np.mean(seed_aucs):.4f}±{np.std(seed_aucs):.4f}")

    # ── CV out-of-fold ROC with top-K ─────────────────────────────────
    # 使用 cross-validated out-of-fold 预测画 ROC，避免 full-set 数据泄露
    X_main = lat_np[:, all_indices[: args.top_k]]
    oof_probs_main = np.zeros(len(labels_np), dtype=np.float32)
    oof_probs_raw  = np.zeros(len(labels_np), dtype=np.float32)
    for train_idx, val_idx in cv.split(X_main, labels_np):
        # Top-K SAE features
        clf_main = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf_main.fit(X_main[train_idx], labels_np[train_idx])
        oof_probs_main[val_idx] = clf_main.predict_proba(X_main[val_idx])[:, 1]
        # Raw hidden state
        clf_raw = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf_raw.fit(acts_np[train_idx], labels_np[train_idx])
        oof_probs_raw[val_idx] = clf_raw.predict_proba(acts_np[val_idx])[:, 1]

    full_auc = float(roc_auc_score(labels_np, oof_probs_main))
    full_f1_pos = float(f1_score(labels_np, (oof_probs_main >= 0.5).astype(int), pos_label=1, zero_division=0))
    full_f1_neg = float(f1_score(labels_np, (oof_probs_main >= 0.5).astype(int), pos_label=0, zero_division=0))
    full_auc_raw = float(roc_auc_score(labels_np, oof_probs_raw))
    print(f"\nCV out-of-fold (top-{args.top_k}):")
    print(f"  AUC   = {full_auc:.4f}")
    print(f"  F1_pos= {full_f1_pos:.4f}")
    print(f"  F1_neg= {full_f1_neg:.4f}")
    print(f"  Raw activation CV AUC = {full_auc_raw:.4f}")

    # CV out-of-fold random SAE ROC curves (top_k features, n_random_seeds seeds)
    random_roc_curves = []  # list of (fpr, tpr, auc)
    for seed in range(n_random_seeds):
        rng_seed = np.random.default_rng(seed * 1000 + args.top_k)
        rand_indices = rng_seed.choice(dict_size, size=args.top_k, replace=False).tolist()
        X_rand_full = lat_np[:, rand_indices]
        oof_probs_rand = np.zeros(len(labels_np), dtype=np.float32)
        for train_idx, val_idx in cv.split(X_rand_full, labels_np):
            clf_rand = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
            clf_rand.fit(X_rand_full[train_idx], labels_np[train_idx])
            oof_probs_rand[val_idx] = clf_rand.predict_proba(X_rand_full[val_idx])[:, 1]
        auc_rand = float(roc_auc_score(labels_np, oof_probs_rand))
        fpr_r, tpr_r, _ = roc_curve(labels_np, oof_probs_rand)
        random_roc_curves.append((fpr_r, tpr_r, auc_rand))

    balance_suffix = "_bal_pos" if args.balance_classes else ""
    out_json = output_dir / f"linear_probe_layer{args.layer}{balance_suffix}.json"
    with open(out_json, "w") as f:
        json.dump(
            {
                "layer": args.layer,
                "top_k": args.top_k,
                "class_balance": {"n_pos": n_pos, "n_neg": n_neg, "pos_rate": round(pos_rate, 4)},
                "full_auc": full_auc,
                "full_f1_pos": full_f1_pos,
                "full_f1_neg": full_f1_neg,
                "raw_activation_baseline": {
                    "mean_auc": raw_mean_auc,
                    "std_auc": raw_std_auc,
                    "cv_auc": full_auc_raw,
                },
                "random_sae_baseline": {
                    str(k): v for k, v in random_results_by_k.items()
                },
                "cv_results_by_k": {str(k): v for k, v in results_by_k.items()},
            },
            f,
            indent=2,
        )
    print(f"Saved results → {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from matplotlib import rcParams

        rcParams.update({
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.4,
            "lines.linewidth": 1.8,
        })

        BLUE   = "#2166AC"
        ORANGE = "#D6604D"
        GRAY   = "#888888"

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # ── Left: AUC vs K (log x-axis) ───────────────────────────────
        ks   = sorted(results_by_k.keys())
        aucs = [results_by_k[k]["mean_auc"] for k in ks]

        axes[0].plot(ks, aucs, marker="o", markersize=5, color=BLUE,
                     label="Top-$K$ SAE features")

        axes[0].axhline(raw_mean_auc, color=ORANGE, linestyle="--", linewidth=1.6,
                        label=f"Raw hidden state (AUC = {raw_mean_auc:.3f})")

        rand_ks    = sorted(random_results_by_k.keys())
        rand_means = np.array([random_results_by_k[k]["mean_auc"] for k in rand_ks])
        axes[0].plot(rand_ks, rand_means, color=GRAY, linestyle=":", marker="s",
                     markersize=4,
                     label=f"Random SAE features (mean of {n_random_seeds} seeds)")

        axes[0].set_xscale("log")
        axes[0].xaxis.set_major_formatter(mticker.ScalarFormatter())
        axes[0].xaxis.set_minor_formatter(mticker.NullFormatter())
        axes[0].set_xticks(ks)
        axes[0].set_xlabel("Number of features $K$")
        axes[0].set_ylabel("Mean AUROC (5-fold CV)")
        axes[0].set_title(f"Linear Probe: AUROC vs. $K$  (layer {args.layer})")
        axes[0].legend(loc="center right", framealpha=0.9)

        # ── Right: ROC curves ──────────────────────────────────────────
        rand_aucs = [auc_r for _, _, auc_r in random_roc_curves]

        # random SAE curves first (behind everything)
        for i, (fpr_r, tpr_r, _) in enumerate(random_roc_curves):
            axes[1].plot(fpr_r, tpr_r, color=GRAY, alpha=0.2, linewidth=0.9,
                         label=f"Random SAE ({n_random_seeds} seeds, "
                               f"mean AUC = {np.mean(rand_aucs):.3f})" if i == 0 else "_nolegend_",
                         zorder=1)

        fpr_raw, tpr_raw, _ = roc_curve(labels_np, oof_probs_raw)
        axes[1].plot(fpr_raw, tpr_raw, color=ORANGE, linestyle="--", linewidth=1.8,
                     label=f"Raw hidden state (AUC = {full_auc_raw:.3f})", zorder=3)

        fpr, tpr, _ = roc_curve(labels_np, oof_probs_main)
        axes[1].plot(fpr, tpr, color=BLUE, linewidth=2.0,
                     label=f"Top-{args.top_k} SAE features (AUC = {full_auc:.3f})", zorder=4)

        axes[1].plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.0,
                     label="Chance", zorder=2)

        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title(f"ROC Curve  (layer {args.layer})")
        axes[1].legend(loc="lower right", framealpha=0.9)

        plt.tight_layout(pad=2.0)
        out_fig = output_dir / f"linear_probe_layer{args.layer}{balance_suffix}.png"
        plt.savefig(out_fig, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved figure → {out_fig}")
    except ImportError:
        print("matplotlib not available, skipping figure")


if __name__ == "__main__":
    main()
