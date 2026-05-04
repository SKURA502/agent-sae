#!/usr/bin/env python3
"""
SAE training loss curves grouped by model family (3 subplots).
Output: /data/Agent-Tool-Use-MI/checkpoint/loss_curves_combined.pdf

Usage:
    python utils_validation/loss/plot_loss_curves.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from matplotlib.lines import Line2D

BASE = Path("/data/Agent-Tool-Use-MI/checkpoint")
OUT  = BASE / "loss_curves_combined.pdf"

FAMILIES = [
    dict(label="Qwen", models=[
        dict(id="Qwen3.5-4B",
             s1="stage1/Qwen3.5-4B-L25-d20480-50M-stage1_stats.json",
             s2="stage2/Qwen3.5-4B-L25-d20480-5M-stage2_stats.json",
             label="Qwen3.5-4B", color="#C8580E"),
        dict(id="Qwen3.5-9B",
             s1="stage1/Qwen3___5-9B-L25-d32768-50M-stage1_stats.json",
             s2="stage2/Qwen3___5-9B-L25-d32768-5M-stage2_stats.json",
             label="Qwen3.5-9B", color="#E8901A"),
    ]),
    dict(label="Gemma", models=[
        dict(id="gemma-3-1b-it",
             s1="stage1/gemma-3-1b-it-L17-d9216-50M-stage1_stats.json",
             s2="stage2/gemma-3-1b-it-L17-d9216-5M-stage2_stats.json",
             label="Gemma-3-1B", color="#1A7340"),
        dict(id="gemma-3-4b-it",
             s1="stage1/gemma-3-4b-it-L29-d20480-50M-stage1_stats.json",
             s2="stage2/gemma-3-4b-it-L29-d20480-5M-stage2_stats.json",
             label="Gemma-3-4B", color="#3DAA6B"),
    ]),
    dict(label="Ministral", models=[
        dict(id="Ministral-3-3B-Instruct-2512",
             s1="stage1/Ministral-3-3B-Instruct-2512-L21-d24576-50M-stage1_stats.json",
             s2="stage2/Ministral-3-3B-Instruct-2512-L21-d24576-5M-stage2_stats.json",
             label="Ministral-3B", color="#0D4EA6"),
        dict(id="Ministral-3-8B-Instruct-2512",
             s1="stage1/Ministral-3-8B-Instruct-2512-L31-d32768-50M-stage1_stats.json",
             s2="stage2/Ministral-3-8B-Instruct-2512-L31-d32768-5M-stage2_stats.json",
             label="Ministral-8B", color="#3A7FC1"),
    ]),
]


def load_stats(path):
    with open(path) as f:
        d = json.load(f)
    return np.array(d["steps"], dtype=float), np.array(d["interval_avg_losses"], dtype=float)


def ema_smooth(y, alpha=0.05):
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i - 1]
    return out


def main():
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype":  42,
        "font.family":  "serif",
        "font.size":    13,
    })

    n = len(FAMILIES)
    fig, axes = plt.subplots(n, 1, figsize=(9, 3.5 * n),
                             sharex=False, sharey=False)

    for ax, fam in zip(axes, FAMILIES):
        x_min = np.inf
        x_max = -np.inf
        boundaries = []

        for m in fam["models"]:
            x1, y1 = load_stats(BASE / m["id"] / m["s1"])
            x2, y2 = load_stats(BASE / m["id"] / m["s2"])

            boundary = float(x1[-1])
            x2p = x2 + boundary
            boundaries.append(boundary)

            alpha1 = 0.02 if len(y1) > 1000 else 0.08
            y1s = ema_smooth(y1, alpha=alpha1)
            y2s = ema_smooth(y2, alpha=0.15)

            color = m["color"]

            # raw traces (very faint)
            ax.plot(x1,  y1,  color=color, lw=0.6, alpha=0.18, zorder=1)
            ax.plot(x2p, y2,  color=color, lw=0.6, alpha=0.18, zorder=1)

            # smoothed traces
            ax.plot(x1,  y1s, color=color, lw=1.8, zorder=2,
                    solid_capstyle="round", label=m["label"])
            ax.plot(x2p, y2s, color=color, lw=1.8, zorder=2, linestyle="--",
                    dash_capstyle="round")

            x_min = min(x_min, x1[0])
            x_max = max(x_max, x2p[-1])

        # Stage boundary: use first model as reference for shading/labels;
        # draw a vertical line per model (coincides if steps are identical)
        boundary_ref = boundaries[0]
        ax.axvspan(x_min, boundary_ref, color="#aaaaaa", alpha=0.07, zorder=0)
        for b in set(boundaries):
            ax.axvline(b, color="#777777", lw=0.9, ls=":", zorder=3)

        trans = blended_transform_factory(ax.transData, ax.transAxes)
        mid1 = (x_min + boundary_ref) / 2
        mid2 = (boundary_ref + x_max) / 2
        for xm, txt in [(mid1, "Stage 1"), (mid2, "Stage 2")]:
            ax.text(xm, 0.97, txt, transform=trans,
                    ha="center", va="top", fontsize=12,
                    color="#555555", style="italic")

        # Per-model color legend (upper right); preserve it so stage legend
        # can be added separately to the last subplot
        model_handles = [
            Line2D([0], [0], color=m["color"], lw=1.8, label=m["label"])
            for m in fam["models"]
        ]
        leg = ax.legend(handles=model_handles, loc="center right", fontsize=12,
                        framealpha=0.90, edgecolor="#cccccc",
                        handlelength=1.8, borderpad=0.6)
        ax.add_artist(leg)

        ax.set_ylabel("Loss", fontsize=14, labelpad=6)
        ax.set_xlim(x_min, x_max)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune="both"))
        ax.tick_params(labelsize=12, direction="in", length=3)
        ax.grid(True, linestyle="--", linewidth=0.4, color="#cccccc", zorder=0)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_linewidth(0.7)

        if ax is not axes[-1]:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Training Step", fontsize=15, labelpad=6)

    # Stage style legend at bottom-left of the last subplot
    stage_handles = [
        Line2D([0], [0], color="#555555", lw=1.8,
               label="Stage 1  (pre-training)"),
        Line2D([0], [0], color="#555555", lw=1.8, linestyle="--",
               label="Stage 2  (fine-tuning)"),
    ]
    axes[-1].legend(handles=stage_handles,
                    loc="lower left", fontsize=12,
                    framealpha=0.92, edgecolor="#cccccc",
                    handlelength=2.2, borderpad=0.7)

    plt.tight_layout(h_pad=0.4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight")
    plt.close()
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()
