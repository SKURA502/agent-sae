#!/usr/bin/env python3
"""
Combined scatter+marginal figure for all 6 models (2 rows × 3 cols).
Filters to samples where Resp ∈ {TC, NC}; colors by Resp (GT ignored).

Layout:
  col 0  : legend
  col 1-3: Qwen3.5 / Gemma-3 / Ministral-3  (one family per col)
  row 0  : smaller model in each family
  row 1  : larger model in each family

Usage:
    python -m utils_validation.plot_pred_tc_nc_combined
"""

import argparse
import json
import os
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
MODEL_BASE = Path(os.environ.get("SOURCE_ROOT", "")) / "model"

MCQ_CHOICES = ("direct_answer", "tool_call", "request_for_info", "cannot_answer")
LABEL = {c: i for i, c in enumerate(MCQ_CHOICES)}

FAMILY_BG = {
    "Qwen3.5":    "#FFFCFB",   # 橙
    "Gemma-3":    "#FBFEFC",   # 绿
    "Ministral-3": "#FBFDFF",  # 蓝
}

FAMILY_COLOR = {
    "Qwen3.5":    "#F5D0BA",   # 橙
    "Gemma-3":    "#B5D9C0",   # 绿
    "Ministral-3": "#C5DFF0",  # 蓝
}

COLOR_TC  = "#F5A623"   # 琥珀黄：Resp=TC
COLOR_NC = "#B39DDB"   # 薰衣草紫：Resp=NC

COLOR_TC_REGION       = "#FFFEF5"   # 极浅黄：对角线下方（TC feature dominant）
COLOR_NC_REGION      = "#F5F0FF"   # 极浅薰衣草：对角线上方（NC feature dominant）
COLOR_TC_REGION_LEG  = "#FFF9C4"   # legend 用，稍深黄
COLOR_NC_REGION_LEG  = "#E8DAFF"   # legend 用，稍深薰衣草

MODELS = [
    dict(row=0, col=0, id="Qwen3.5-4B",  layer=25,
         sae="Qwen3.5-4B-L25-d20480-5M-stage2.pt",
         label="Qwen3.5-4B",   family="Qwen3.5",   axis_range=(0, 3)),
    dict(row=1, col=0, id="Qwen3.5-9B",  layer=25,
         sae="Qwen3.5-9B-L25-d32768-5M-stage2.pt",
         label="Qwen3.5-9B",   family="Qwen3.5",   axis_range=(0, 10)),
    dict(row=0, col=1, id="gemma-3-1b-it", layer=17,
         sae="gemma-3-1b-it-L17-d9216-5M-stage2.pt",
         label="Gemma-3-1B",   family="Gemma-3",   axis_range=(400, 1000)),
    dict(row=1, col=1, id="gemma-3-4b-it", layer=29,
         sae="gemma-3-4b-it-L29-d20480-5M-stage2.pt",
         label="Gemma-3-4B",   family="Gemma-3",   axis_range=(3000, 5000)),
    dict(row=0, col=2, id="Ministral-3-3B-Instruct-2512", layer=21,
         sae="Ministral-3-3B-Instruct-2512-L21-d24576-5M-stage2.pt",
         label="Ministral-3B", family="Ministral-3", axis_range=(0.0, 1.5)),
    dict(row=1, col=2, id="Ministral-3-8B-Instruct-2512", layer=31,
         sae="Ministral-3-8B-Instruct-2512-L31-d32768-5M-stage2.pt",
         label="Ministral-8B", family="Ministral-3", axis_range=(0, 4)),
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
    nc_feat_path = BASE / "outputs" / mid / "analysis" / "feature_discovery" / "request_for_info" / f"top_features_layer{layer}.json"
    sae_path      = BASE / "checkpoint" / mid / "stage2" / m["sae"]

    for p in [acts_path, tc_feat_path, nc_feat_path, sae_path]:
        if not p.exists():
            print(f"  [skip] missing: {p}")
            return None

    print(f"  Loading {mid}...")
    ckpt   = torch.load(acts_path, map_location="cpu", weights_only=True)
    acts   = ckpt[f"layer_{layer}"].float()
    labels = ckpt["labels"].long().numpy()

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
    nc_feats = _top_feature_indices(nc_feat_path, top_n)

    mask = (labels == LABEL["tool_call"]) | (labels == LABEL["request_for_info"])

    sub      = latents[mask].float()
    tc_act   = sub[:, tc_feats].mean(dim=1).numpy()
    nc_act  = sub[:, nc_feats].mean(dim=1).numpy()
    sub_resp = labels[mask]
    return {"tc_act": tc_act, "nc_act": nc_act, "sub_resp": sub_resp}


def draw_panel(ax_main, ax_top, ax_right, data, bg_color,
               show_xticks=False, show_yticks=False, axis_range=None):
    for ax in [ax_main, ax_top, ax_right]:
        ax.set_facecolor(bg_color)

    if data is None:
        ax_main.text(0.5, 0.5, "No data", ha="center", va="center",
                     transform=ax_main.transAxes, fontsize=8, color="#AAAAAA")
    else:
        tc_act, nc_act, sub_resp = data["tc_act"], data["nc_act"], data["sub_resp"]
        for resp_val, color, n_std in [(LABEL["tool_call"], COLOR_TC, 2.5),
                                        (LABEL["request_for_info"], COLOR_NC, 2.0)]:
            m = sub_resp == resp_val
            if m.sum() < 3:
                continue
            x, y = tc_act[m], nc_act[m]
            ax_main.scatter(x, y, c=color, s=5, alpha=0.45, linewidths=0, zorder=2)
            _cov_ellipse(ax_main, x, y, n_std=n_std,
                         facecolor="none", edgecolor=color,
                         linewidth=1.4, linestyle="--", zorder=3)
            sns.kdeplot(x=x, ax=ax_top,   color=color, fill=True, alpha=0.55, linewidth=0.8)
            sns.kdeplot(y=y, ax=ax_right, color=color, fill=True, alpha=0.55, linewidth=0.8)

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
        # 对角线下方：TC-dominant；上方：NC-dominant
        ax_main.fill_between([lo, hi], [lo, hi], lo,
                             color=COLOR_TC_REGION,  alpha=1.0, zorder=0)
        ax_main.fill_between([lo, hi], [lo, hi], hi,
                             color=COLOR_NC_REGION, alpha=1.0, zorder=0)
        ax_main.plot([lo, hi], [lo, hi], color="#888888", lw=0.8,
                     ls="--", alpha=0.5, zorder=1)

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
                    default=str(BASE / "outputs" / "pred_tc_nc_combined.pdf"))
    ap.add_argument("--top-n", type=int, default=25)
    args = ap.parse_args()

    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif", "font.size": 9,
    })

    model_map = {(m["row"], m["col"]): m for m in MODELS}
    all_data  = {(m["row"], m["col"]): load_model_data(m, args.top_n) for m in MODELS}

    fig = plt.figure(figsize=(13, 7))
    outer = GridSpec(2, 4, figure=fig,
                     width_ratios=[0.22, 1, 1, 1],
                     hspace=0.10, wspace=0.20,
                     left=0.01, right=0.98, top=0.91, bottom=0.05)

    ax_leg = fig.add_subplot(outer[:, 0])
    ax_leg.axis("off")

    scatter_patches = [
        Patch(facecolor=COLOR_TC,  alpha=0.8, label="Resp=TC"),
        Patch(facecolor=COLOR_NC, alpha=0.8, label="Resp=NC"),
    ]
    region_patches = [
        Patch(facecolor=COLOR_TC_REGION_LEG,  edgecolor="#cccccc", linewidth=0.7, label="TC-dominant"),
        Patch(facecolor=COLOR_NC_REGION_LEG, edgecolor="#cccccc", linewidth=0.7, label="NC-dominant"),
    ]
    leg2 = ax_leg.legend(handles=scatter_patches,
                         title="Response",
                         loc="center", bbox_to_anchor=(0.5, 0.65),
                         frameon=True, framealpha=0.9, edgecolor="#cccccc",
                         fontsize=10, title_fontsize=11,
                         borderpad=0.5, labelspacing=0.5,
                         handlelength=1.1, handleheight=1.2)
    leg3 = ax_leg.legend(handles=region_patches,
                         title="Region",
                         loc="center", bbox_to_anchor=(0.5, 0.30),
                         frameon=True, framealpha=0.9, edgecolor="#cccccc",
                         fontsize=10, title_fontsize=11,
                         borderpad=0.5, labelspacing=0.5,
                         handlelength=1.1, handleheight=1.2)
    ax_leg.add_artist(leg2)

    for row in range(2):
        for col in range(3):
            m    = model_map[(row, col)]
            data = all_data[(row, col)]
            bg   = "white"

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
