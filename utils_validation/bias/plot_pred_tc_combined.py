#!/usr/bin/env python3
"""
Combined pred-TC scatter+marginal figure for all 6 models (2 rows × 3 cols).

Layout:
  col 0  : legend
  col 1-3: Qwen3.5 / Gemma-3 / Ministral-3  (one family per col)
  row 0  : smaller model in each family
  row 1  : larger model in each family

Usage:
    python -m utils_validation.plot_pred_tc_combined
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Ellipse, Patch
import seaborn as sns

BASE       = Path("/data/Agent-Tool-Use-MI")
MODEL_BASE = Path("/mnt/shared-storage-gpfs2/safelens-share-gpfs2/source/model")

MCQ_CHOICES = ("direct_answer", "tool_call", "request_for_info", "cannot_answer")
LABEL = {c: i for i, c in enumerate(MCQ_CHOICES)}

# ── 与 linear_probe_combined.py 一致的极浅家族背景色 ──────────────────
FAMILY_BG = {
    "Qwen3.5":    "#FEFAF8",   # 10% 橙
    "Gemma-3":    "#F8FBF9",   # 10% 绿
    "Ministral-3": "#F9FCFE",  # 10% 蓝
}

FAMILY_COLOR = {
    "Qwen3.5":    "#F5D0BA",   # 橙
    "Gemma-3":    "#B5D9C0",   # 绿
    "Ministral-3": "#C5DFF0",  # 蓝
}

# ── 散点颜色 ──────────────────────────────────────────────────────────
COLOR_TC  = "#F4ACBB"   # 浅粉：GT=TC,  pred=TC  (correct)
COLOR_NC  = "#A8D0E6"   # 浅蓝：GT=NC,  pred=TC  (incorrect)

# ── 模型配置（顺序：col 0=Qwen, col 1=Gemma, col 2=Ministral；row 0=小, row 1=大）
MODELS = [
    dict(row=0, col=0, id="Qwen3.5-4B",  layer=25,
         sae="Qwen3.5-4B-L25-d20480-5M-stage2.pt",
         label="Qwen3.5-4B",   family="Qwen3.5",    axis_range=(0, 2.5)),
    dict(row=1, col=0, id="Qwen3.5-9B",  layer=25,
         sae="Qwen3.5-9B-L25-d32768-5M-stage2.pt",
         label="Qwen3.5-9B",   family="Qwen3.5",    axis_range=(1, 9)),
    dict(row=0, col=1, id="gemma-3-1b-it", layer=17,
         sae="gemma-3-1b-it-L17-d9216-5M-stage2.pt",
         label="Gemma-3-1B",   family="Gemma-3",    axis_range=(500, 1000)),
    dict(row=1, col=1, id="gemma-3-4b-it", layer=29,
         sae="gemma-3-4b-it-L29-d20480-5M-stage2.pt",
         label="Gemma-3-4B",   family="Gemma-3",    axis_range=(3000, 5000)),
    dict(row=0, col=2, id="Ministral-3-3B-Instruct-2512", layer=21,
         sae="Ministral-3-3B-Instruct-2512-L21-d24576-5M-stage2.pt",
         label="Ministral-3B", family="Ministral-3", axis_range=(0.0, 1.0)),
    dict(row=1, col=2, id="Ministral-3-8B-Instruct-2512", layer=31,
         sae="Ministral-3-8B-Instruct-2512-L31-d32768-5M-stage2.pt",
         label="Ministral-8B", family="Ministral-3", axis_range=(0, 3)),
]


# ── helpers ───────────────────────────────────────────────────────────

def _encode(sae, acts, batch_size=512):
    latents = []
    for i in range(0, len(acts), batch_size):
        b = acts[i:i+batch_size].to(sae.config.device, dtype=sae.config.get_torch_dtype())
        with torch.no_grad():
            latents.append(sae.encode(b).cpu())
    return torch.cat(latents, dim=0)


def _top_feature_indices(path, top_n):
    with open(path) as f:
        data = json.load(f)
    return [e["feature_idx"] for e in data[:top_n]]


def _nice_ticks(lo, hi):
    span = hi - lo
    raw_step = span / 4
    mag = 10 ** np.floor(np.log10(raw_step))
    norm = raw_step / mag
    if norm <= 1.5:
        step = mag
    elif norm <= 3.0:
        step = 2 * mag
    elif norm <= 7.0:
        step = 5 * mag
    else:
        step = 10 * mag
    start = np.round(lo / step) * step
    ticks = np.arange(start, hi + step * 0.01, step)
    return ticks[(ticks >= lo - 1e-9) & (ticks <= hi + 1e-9)]


def _cov_ellipse(ax, x, y, n_std=2.0, **kwargs):
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(np.abs(vals))
    ax.add_patch(Ellipse(xy=(x.mean(), y.mean()),
                         width=w, height=h, angle=angle, **kwargs))


def load_model_data(m, top_n=25):
    mid   = m["id"]
    layer = m["layer"]
    acts_path     = BASE / "outputs" / mid / "activations" / "when2call_mcq" / f"layer_{layer}_activations.pt"
    tc_feat_path  = BASE / "outputs" / mid / "analysis" / "feature_discovery" / "tool_call" / f"top_features_layer{layer}.json"
    nc_feat_path  = BASE / "outputs" / mid / "analysis" / "feature_discovery" / "request_for_info" / f"top_features_layer{layer}.json"
    sae_path      = BASE / "checkpoint" / mid / "stage2" / m["sae"]

    for p in [acts_path, tc_feat_path, nc_feat_path, sae_path]:
        if not p.exists():
            print(f"  [skip] missing: {p}")
            return None

    print(f"  Loading {mid}...")
    ckpt      = torch.load(acts_path, map_location="cpu", weights_only=True)
    acts      = ckpt[f"layer_{layer}"].float()
    labels    = ckpt["labels"].long().numpy()
    gt_labels = ckpt["gt_labels"].long().numpy()

    sys.path.insert(0, str(BASE))
    from sae.sae_model import TopKSAE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae    = TopKSAE.load(str(sae_path), device=device)
    sae.eval()
    latents = _encode(sae, acts)
    with torch.no_grad():
        d_norms = sae.decoder.weight.detach().float().cpu().norm(dim=0)
    latents = latents * d_norms

    tc_feats  = _top_feature_indices(tc_feat_path,  top_n)
    nc_feats  = _top_feature_indices(nc_feat_path,  top_n)

    pred_tc   = labels    == LABEL["tool_call"]
    gt_filter = (gt_labels == LABEL["tool_call"]) | (gt_labels == LABEL["request_for_info"])
    mask      = pred_tc & gt_filter

    sub = latents[mask].float()
    tc_act = sub[:, tc_feats].mean(dim=1).numpy()
    nc_act = sub[:, nc_feats].mean(dim=1).numpy()
    return {"tc_act": tc_act, "nc_act": nc_act, "sub_gt": gt_labels[mask]}


def draw_panel(ax_main, ax_top, ax_right, data, bg_color,
               show_xticks=False, show_yticks=False, axis_range=None):
    for ax in [ax_main, ax_top, ax_right]:
        ax.set_facecolor(bg_color)

    if data is None:
        ax_main.text(0.5, 0.5, "No data", ha="center", va="center",
                     transform=ax_main.transAxes, fontsize=8, color="#AAAAAA")
    else:
        tc_act, nc_act, sub_gt = data["tc_act"], data["nc_act"], data["sub_gt"]
        centers = {}
        for gt_val, color in [(LABEL["tool_call"], COLOR_TC),
                               (LABEL["request_for_info"], COLOR_NC)]:
            m = sub_gt == gt_val
            if m.sum() < 3:
                continue
            x, y = tc_act[m], nc_act[m]
            centers[gt_val] = (x.mean(), y.mean())
            ax_main.scatter(x, y, c=color, s=5, alpha=0.45, linewidths=0, zorder=2)
            _cov_ellipse(ax_main, x, y, n_std=2.0,
                         facecolor="none", edgecolor=color,
                         linewidth=1.4, linestyle="--", zorder=3)
            sns.kdeplot(x=x, ax=ax_top,   color=color, fill=True, alpha=0.55, linewidth=0.8)
            sns.kdeplot(y=y, ax=ax_right, color=color, fill=True, alpha=0.55, linewidth=0.8)

        if LABEL["tool_call"] in centers and LABEL["request_for_info"] in centers:
            cx0, cy0 = centers[LABEL["tool_call"]]
            cx1, cy1 = centers[LABEL["request_for_info"]]
            ax_main.annotate("", xy=(cx1, cy1), xytext=(cx0, cy0),
                             arrowprops=dict(arrowstyle="-|>", color="#444444",
                                             lw=1.2, mutation_scale=10),
                             zorder=4)

        if axis_range is not None:
            lo, hi = axis_range
            pad = (hi - lo) * 0.1
            lo, hi = lo - pad, hi + pad
        else:
            xl, yl = ax_main.get_xlim(), ax_main.get_ylim()
            lo = min(xl[0], yl[0])
            hi = max(xl[1], yl[1])
        ax_main.set_xlim(lo, hi)
        ax_main.set_ylim(lo, hi)
        ticks = _nice_ticks(lo, hi)
        ax_main.set_xticks(ticks)
        ax_main.set_yticks(ticks)

    # ── 主图坐标轴 ────────────────────────────────────────────────────
    ax_main.grid(True, linestyle="--", linewidth=0.4, color="#cccccc", zorder=0)
    ax_main.set_axisbelow(True)
    ax_main.set_xlabel("")
    ax_main.set_ylabel("")
    ax_main.tick_params(labelsize=10.5, direction="in", length=2)
    if not show_xticks:
        ax_main.set_xticklabels([])
    if not show_yticks:
        ax_main.set_yticklabels([])
    for sp in ax_main.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.7)

    # ── 边缘轴 ────────────────────────────────────────────────────────
    for ax_m in [ax_top, ax_right]:
        for sp in ax_m.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.7)
        ax_m.tick_params(left=False, bottom=False,
                         labelleft=False, labelbottom=False)
        ax_m.set_xlabel("")
        ax_m.set_ylabel("")


# ── main ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-path",
                    default=str(BASE / "outputs" / "pred_tc_combined.pdf"))
    ap.add_argument("--top-n", type=int, default=25)
    args = ap.parse_args()

    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif", "font.size": 9,
    })

    # ── 加载数据 ──────────────────────────────────────────────────────
    model_map = {(m["row"], m["col"]): m for m in MODELS}
    all_data  = {(m["row"], m["col"]): load_model_data(m, args.top_n) for m in MODELS}

    # ── 构建图形 ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 7))
    outer = GridSpec(2, 4, figure=fig,
                     width_ratios=[0.22, 1, 1, 1],
                     hspace=0.10, wspace=0.20,
                     left=0.01, right=0.98, top=0.91, bottom=0.05)

    # ── legend 列 ─────────────────────────────────────────────────────
    ax_leg = fig.add_subplot(outer[:, 0])
    ax_leg.axis("off")

    family_patches = [
        Patch(facecolor=FAMILY_COLOR[fam], edgecolor="#888888",
              linewidth=0.7, label=fam)
        for fam in ["Qwen3.5", "Gemma-3", "Ministral-3"]
    ]
    scatter_patches = [
        Patch(facecolor=COLOR_TC,  alpha=0.8, label="GT=TC,\nResp=TC"),
        Patch(facecolor=COLOR_NC,  alpha=0.8, label="GT=NC,\nResp=TC"),
    ]
    leg1 = ax_leg.legend(handles=family_patches,
                         title="Model family",
                         loc="center", bbox_to_anchor=(0.5, 0.75),
                         frameon=True, framealpha=0.9, edgecolor="#cccccc",
                         fontsize=10, title_fontsize=11,
                         borderpad=0.6, labelspacing=0.5,
                         handlelength=0.8, handleheight=1.2)
    leg2 = ax_leg.legend(handles=scatter_patches,
                         title="GT → Resp",
                         loc="center", bbox_to_anchor=(0.5, 0.25),
                         frameon=True, framealpha=0.9, edgecolor="#cccccc",
                         fontsize=10, title_fontsize=11,
                         borderpad=0.5, labelspacing=0.5,
                         handlelength=1.2, handleheight=3.0)
    ax_leg.add_artist(leg1)

    # ── 绘制每个 panel ────────────────────────────────────────────────
    for row in range(2):
        for col in range(3):
            m    = model_map[(row, col)]
            data = all_data[(row, col)]
            bg   = FAMILY_BG[m["family"]]

            inner = GridSpecFromSubplotSpec(
                2, 2,
                subplot_spec=outer[row, col + 1],
                width_ratios=[4, 1], height_ratios=[1, 4],
                hspace=0.0, wspace=0.0,
            )
            ax_main  = fig.add_subplot(inner[1, 0])
            ax_top   = fig.add_subplot(inner[0, 0], sharex=ax_main)
            ax_right = fig.add_subplot(inner[1, 1], sharey=ax_main)
            fig.add_subplot(inner[0, 1]).set_visible(False)

            draw_panel(ax_main, ax_top, ax_right, data, bg,
                       show_xticks=True,
                       show_yticks=True,
                       axis_range=m["axis_range"])

            # 列标题（第一行顶部）
            if row == 0:
                ax_top.set_title(m["family"], fontsize=14,
                                 fontweight="bold", pad=4)
            # 模型名
            ax_main.text(0.04, 0.96, m["label"],
                         transform=ax_main.transAxes,
                         fontsize=12, va="top", ha="left",
                         color="#333333")

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
