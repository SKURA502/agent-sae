"""
用 logistic regression 量化 TC 内禀偏置（TC tendency > RFI tendency）。

对所有 GT ∈ {tool_call, request_for_info} 的样本，拟合：
    P(pred=TC) = sigmoid(β * margin + β₀)
    margin = tc_act - rfi_act  (decoder-norm scaled)

β₀ > 0  → 在 margin=0 时模型仍偏向 TC（TC tendency > RFI tendency）
决策边界 margin* = -β₀ / β（RFI 需要超过 TC 多少才能拉平到 50%）

同时输出：
  - logistic curve 图（GT=TC / GT=RFI 分开着色，拟合曲线叠加）
  - 按 margin 分 bin 的实际 TC 预测率（用于验证 logistic fit）

用法：
    python -m utils_validation.tc_bias_logistic --layer 25
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


MCQ_CHOICES = ("direct_answer", "tool_call", "request_for_info", "cannot_answer")
LABEL = {c: i for i, c in enumerate(MCQ_CHOICES)}


def encode_activations(sae, acts: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
    latents = []
    for i in range(0, len(acts), batch_size):
        batch = acts[i : i + batch_size].to(
            sae.config.device, dtype=sae.config.get_torch_dtype()
        )
        with torch.no_grad():
            latents.append(sae.encode(batch).cpu())
    return torch.cat(latents, dim=0)


def load_top_feature_indices(path: Path) -> list:
    with open(path) as f:
        return [e["feature_idx"] for e in json.load(f)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=25)
    parser.add_argument("--sae-path", type=str,
        default="/data/Agent-Tool-Use-MI/checkpoint/Qwen3.5-4B/stage2/"
                "Qwen3.5-4B-L25-d20480-5M-stage2.pt")
    parser.add_argument("--activations-dir", type=str,
        default="/data/Agent-Tool-Use-MI/outputs/Qwen3.5-4B/activations/when2call_mcq")
    parser.add_argument("--feature-discovery-dir", type=str,
        default="/data/Agent-Tool-Use-MI/outputs/Qwen3.5-4B/analysis/feature_discovery")
    parser.add_argument("--output-dir", type=str,
        default="/data/Agent-Tool-Use-MI/outputs/Qwen3.5-4B/analysis/rfi_confusion")
    parser.add_argument("--scatter-top-n", type=int, default=25)
    parser.add_argument("--n-bins", type=int, default=20,
        help="margin 分 bin 数量（用于实际决策率验证）")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--encode-batch-size", type=int, default=512)
    args = parser.parse_args()

    layer  = args.layer
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ── 加载激活 ─────────────────────────────────────────────────────
    acts_path = Path(args.activations_dir) / f"layer_{layer}_activations.pt"
    print(f"Loading activations: {acts_path}")
    ckpt      = torch.load(acts_path, map_location="cpu", weights_only=True)
    acts      = ckpt[f"layer_{layer}"].float()
    labels    = ckpt["labels"].long().numpy()
    gt_labels = ckpt["gt_labels"].long().numpy()
    print(f"  Total samples: {len(acts)}")

    # ── 加载 SAE，编码 ────────────────────────────────────────────────
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from sae.sae_model import TopKSAE

    print(f"Loading SAE: {args.sae_path}")
    sae = TopKSAE.load(args.sae_path, device=device)
    sae.eval()
    latents = encode_activations(sae, acts, batch_size=args.encode_batch_size)
    with torch.no_grad():
        decoder_norms = sae.decoder.weight.detach().float().cpu().norm(dim=0)
    latents = latents * decoder_norms

    # ── 加载 top features ─────────────────────────────────────────────
    fd_root = Path(args.feature_discovery_dir)
    tc_features  = load_top_feature_indices(
        fd_root / "tool_call"        / f"top_features_layer{layer}.json")
    rfi_features = load_top_feature_indices(
        fd_root / "request_for_info" / f"top_features_layer{layer}.json")
    sn = args.scatter_top_n
    tc_feat  = tc_features[:sn]
    rfi_feat = rfi_features[:sn]
    print(f"Using top-{sn} features: TC={len(tc_feat)}, RFI={len(rfi_feat)}")

    # ── 筛选 GT ∈ {TC, RFI} 的样本 ───────────────────────────────────
    gt_filter = (
        (gt_labels == LABEL["tool_call"]) |
        (gt_labels == LABEL["request_for_info"])
    )
    sub_latents  = latents[gt_filter].float()
    sub_labels   = labels[gt_filter]    # 模型预测
    sub_gt       = gt_labels[gt_filter]

    tc_act  = sub_latents[:, tc_feat].mean(dim=1).numpy()
    rfi_act = sub_latents[:, rfi_feat].mean(dim=1).numpy()
    margin  = tc_act - rfi_act          # 正值 = TC 激活领先

    # 因变量：pred=TC → 1，pred=RFI → 0（排除其他预测）
    tc_rfi_pred = (sub_labels == LABEL["tool_call"]) | (sub_labels == LABEL["request_for_info"])
    y      = (sub_labels[tc_rfi_pred] == LABEL["tool_call"]).astype(float)
    X      = margin[tc_rfi_pred]
    gt_sub = sub_gt[tc_rfi_pred]
    print(f"  Regression samples (pred∈{{TC,RFI}}): {len(y)}"
          f"  (pred=TC: {int(y.sum())}, pred=RFI: {int((1-y).sum())})")

    # ── Logistic Regression (raw) ─────────────────────────────────────
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(fit_intercept=True, max_iter=1000)
    clf.fit(X.reshape(-1, 1), y)
    beta   = float(clf.coef_[0][0])
    beta_0 = float(clf.intercept_[0])
    boundary = -beta_0 / beta

    # ── Logistic Regression (standardized: divide by std, keep mean) ──
    X_std = float(X.std())
    X_z   = X / X_std          # 只除以 std，不减均值，保留方向信息
    clf_z = LogisticRegression(fit_intercept=True, max_iter=1000)
    clf_z.fit(X_z.reshape(-1, 1), y)
    beta_z   = float(clf_z.coef_[0][0])
    beta_0_z = float(clf_z.intercept_[0])
    boundary_z = -beta_0_z / beta_z   # 单位：margin 的标准差

    print("\n" + "=" * 55)
    print("  [Raw]  P(pred=TC) = σ(β·margin + β₀)")
    print(f"  β  = {beta:.4f}  (sensitivity to margin)")
    print(f"  β₀ = {beta_0:.4f}  (intrinsic TC bias, >0 = TC tendency)")
    print(f"  Decision boundary  margin* = {boundary:.4f}")
    print(f"  → RFI needs to exceed TC by {-boundary:.3f} units for 50/50 decision"
          if boundary < 0 else
          f"  → TC needs to exceed RFI by {boundary:.3f} units for 50/50 decision")
    print()
    print("  [Standardized]  P(pred=TC) = σ(β_z·(margin/σ) + β₀_z)")
    print(f"  margin σ = {X_std:.4f}")
    print(f"  β_z  = {beta_z:.4f}")
    print(f"  β₀_z = {beta_0_z:.4f}  (comparable across models)")
    print(f"  P(TC | margin=0) = σ(β₀_z) = {1/(1+np.exp(-beta_0_z)):.4f}")
    print(f"  Decision boundary = {boundary_z:.4f} σ")
    print("=" * 55)

    # ── Binned actual decision rate (raw & standardized) ─────────────
    bin_edges = np.linspace(X.min(), X.max(), args.n_bins + 1)
    bin_centers, bin_rates, bin_counts = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (X >= lo) & (X < hi)
        if mask.sum() > 5:
            bin_centers.append((lo + hi) / 2)
            bin_rates.append(y[mask].mean())
            bin_counts.append(int(mask.sum()))

    bin_edges_z = np.linspace(X_z.min(), X_z.max(), args.n_bins + 1)
    bin_centers_z, bin_rates_z, bin_counts_z = [], [], []
    for lo, hi in zip(bin_edges_z[:-1], bin_edges_z[1:]):
        mask = (X_z >= lo) & (X_z < hi)
        if mask.sum() > 5:
            bin_centers_z.append((lo + hi) / 2)
            bin_rates_z.append(y[mask].mean())
            bin_counts_z.append(int(mask.sum()))

    # ── 绘图 ─────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    COLORS = {
        LABEL["tool_call"]:        "#D55E00",
        LABEL["request_for_info"]: "#0072B2",
    }

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(19, 5),
                             gridspec_kw={"wspace": 0.38})

    # ── 左图：raw logistic curve + binned empirical rates ────────────
    ax = axes[0]
    margin_range = np.linspace(X.min() - 0.1, X.max() + 0.1, 300)
    p_tc = 1 / (1 + np.exp(-(beta * margin_range + beta_0)))
    ax.plot(margin_range, p_tc, color="#333333", linewidth=1.8,
            label=f"Logistic fit\n$\\beta$={beta:.2f}, $\\beta_0$={beta_0:+.2f}")

    ax.scatter(bin_centers, bin_rates, s=[c / 5 for c in bin_counts],
               color="#666666", alpha=0.7, zorder=3,
               label="Empirical rate (size ∝ n)")

    ax.axvline(boundary, color="#AA44BB", linewidth=1.2, linestyle="--", alpha=0.8,
               label=f"Boundary = {boundary:.3f}")
    ax.axvline(0, color="#888888", linewidth=0.8, linestyle=":", alpha=0.6,
               label="TC = RFI (margin = 0)")
    ax.axhline(0.5, color="#888888", linewidth=0.6, linestyle=":", alpha=0.5)

    ax.set_xlabel("margin  =  TC_act − RFI_act", labelpad=6)
    ax.set_ylabel("P(pred = tool_call)", labelpad=6)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        f"[Raw]  TC decision prob vs margin\n"
        f"Layer {layer}  ·  $n$ = {len(y)}",
        pad=8,
    )
    ax.legend(fontsize=8.5, frameon=True, framealpha=0.92, edgecolor="#cccccc")

    # ── 中图：standardized logistic curve ────────────────────────────
    ax = axes[1]
    margin_range_z = np.linspace(X_z.min() - 0.1, X_z.max() + 0.1, 300)
    p_tc_z = 1 / (1 + np.exp(-(beta_z * margin_range_z + beta_0_z)))
    p_at_0 = 1 / (1 + np.exp(-beta_0_z))
    ax.plot(margin_range_z, p_tc_z, color="#1A6B3C", linewidth=1.8,
            label=f"Logistic fit\n$\\beta_z$={beta_z:.2f}, $\\beta_{{0z}}$={beta_0_z:+.2f}")

    ax.scatter(bin_centers_z, bin_rates_z, s=[c / 5 for c in bin_counts_z],
               color="#666666", alpha=0.7, zorder=3,
               label="Empirical rate (size ∝ n)")

    ax.axvline(boundary_z, color="#AA44BB", linewidth=1.2, linestyle="--", alpha=0.8,
               label=f"Boundary = {boundary_z:.3f} σ")
    ax.axvline(0, color="#888888", linewidth=0.8, linestyle=":", alpha=0.6,
               label=f"margin=0  →  P(TC)={p_at_0:.3f}")
    ax.axhline(0.5, color="#888888", linewidth=0.6, linestyle=":", alpha=0.5)
    ax.axhline(p_at_0, color="#1A6B3C", linewidth=0.8, linestyle=":", alpha=0.6)

    ax.set_xlabel("margin / σ  (standardized)", labelpad=6)
    ax.set_ylabel("P(pred = tool_call)", labelpad=6)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        f"[Standardized]  TC decision prob vs margin/σ\n"
        f"Layer {layer}  ·  σ = {X_std:.4f}",
        pad=8,
    )
    ax.legend(fontsize=8.5, frameon=True, framealpha=0.92, edgecolor="#cccccc")

    # ── 右图：margin 分布（GT=TC vs GT=RFI，按 pred 着色） ─────────────
    ax = axes[2]
    for gt_val, gt_name, linestyle in [
        (LABEL["tool_call"],        "GT=TC",  "-"),
        (LABEL["request_for_info"], "GT=RFI", "--"),
    ]:
        gt_mask = gt_sub == gt_val
        for pred_val, pred_name, color, alpha in [
            (LABEL["tool_call"],        "pred=TC",  "#D55E00", 0.7),
            (LABEL["request_for_info"], "pred=RFI", "#0072B2", 0.7),
        ]:
            pred_mask = (sub_labels[tc_rfi_pred] == pred_val)
            m = gt_mask & pred_mask
            if m.sum() < 5:
                continue
            vals = X[m]
            bins = np.linspace(X.min(), X.max(), 40)
            counts_hist, edges = np.histogram(vals, bins=bins)
            centers = (edges[:-1] + edges[1:]) / 2
            lbl = f"{gt_name}, {pred_name} (n={m.sum()})"
            ax.plot(centers, counts_hist, color=color, linestyle=linestyle,
                    linewidth=1.3, alpha=alpha, label=lbl)

    ax.axvline(boundary, color="#AA44BB", linewidth=1.2, linestyle="--", alpha=0.8,
               label=f"margin* = {boundary:.3f}")
    ax.axvline(0, color="#888888", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.set_xlabel("margin  =  TC_act − RFI_act", labelpad=6)
    ax.set_ylabel("Count", labelpad=6)
    ax.set_title(
        f"Margin distribution by GT & prediction\nLayer {layer}",
        pad=8,
    )
    ax.legend(fontsize=7.5, frameon=True, framealpha=0.92, edgecolor="#cccccc")

    fig.suptitle(
        f"TC Intrinsic Bias  ·  "
        f"Raw: $\\beta_0$={beta_0:+.3f}, boundary={boundary:.3f}  ·  "
        f"Std: $\\beta_{{0z}}$={beta_0_z:+.3f}, P(TC|margin=0)={p_at_0:.3f}",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    out_path = Path(args.output_dir) / f"tc_bias_logistic_layer{layer}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {out_path}")

    # ── 保存数值结果 ───────────────────────────────────────────────────
    result = {
        "layer": layer,
        "n_samples": len(y),
        "logistic_raw": {
            "beta":            round(beta, 6),
            "beta_0":          round(beta_0, 6),
            "boundary_margin": round(boundary, 6),
            "interpretation": (
                f"RFI must exceed TC by {-boundary:.3f} units for 50/50 decision"
                if boundary < 0 else
                f"TC must exceed RFI by {boundary:.3f} units for 50/50 decision"
            ),
        },
        "logistic_standardized": {
            "margin_std":       round(X_std, 6),
            "beta_z":           round(beta_z, 6),
            "beta_0_z":         round(beta_0_z, 6),
            "boundary_sigma":   round(boundary_z, 6),
            "p_tc_at_margin_0": round(p_at_0, 6),
            "note": "margin divided by std(margin); beta_0_z and p_tc_at_margin_0 are comparable across models",
        },
        "empirical": {
            "tc_win_rate_when_tc_gt_rfi":   float(y[X > 0].mean()) if (X > 0).sum() > 0 else None,
            "rfi_win_rate_when_rfi_gt_tc":  float((1 - y)[X < 0].mean()) if (X < 0).sum() > 0 else None,
        },
    }
    json_path = Path(args.output_dir) / f"tc_bias_logistic_layer{layer}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved JSON → {json_path}")

    # ── 终端摘要 ─────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  TC win rate  (margin > 0): {result['empirical']['tc_win_rate_when_tc_gt_rfi']:.1%}")
    print(f"  RFI win rate (margin < 0): {result['empirical']['rfi_win_rate_when_rfi_gt_tc']:.1%}")
    print(f"  Asymmetry: TC wins {result['empirical']['tc_win_rate_when_tc_gt_rfi'] - result['empirical']['rfi_win_rate_when_rfi_gt_tc']:.1%} more when it has the advantage")
    print(f"  [Cross-model] β₀_z = {beta_0_z:+.4f}  |  P(TC|margin=0) = {p_at_0:.4f}")
    print(f"{'─'*55}")


if __name__ == "__main__":
    main()
