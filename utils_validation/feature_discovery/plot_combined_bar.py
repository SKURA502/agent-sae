#!/usr/bin/env python3
"""
Combined top-feature bar chart: tool_call (left) and No Call (right).

Both panels share the same AUROC colormap normalization so colors are
directly comparable across concepts.

Usage:
    python plot_combined_bar.py [--layer 25] [--top-n 20] [--out combined_top_features_bar_layer25.png]
    python plot_combined_bar.py --data-dir /path/to/feature_discovery
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import rcParams

HERE = Path(__file__).parent
DEFAULT_DATA_DIR = (
    HERE.parent / "outputs" / "Qwen3.5-9B" / "analysis" / "feature_discovery"
)

# ── typography & style ────────────────────────────────────────────────────────
rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":          12,
    "axes.titlesize":     15,
    "axes.titleweight":   "bold",
    "axes.labelsize":     14,
    "axes.labelweight":   "bold",
    "xtick.labelsize":    10,
    "ytick.labelsize":    11,
    "legend.fontsize":    11,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "lines.linewidth":    1.2,
    "pdf.fonttype":       42,
    "svg.fonttype":       "none",
})

CONCEPT_LABELS = {
    "tool_call":        "Tool Call",
    "request_for_info": "No Call",
}
CMAP_NAME = "RdYlGn"


def load_features(data_dir: Path, layer: int, concept: str, top_n: int):
    path = data_dir / concept / f"top_features_layer{layer}.json"
    with open(path) as f:
        data = json.load(f)
    data.sort(key=lambda x: x["mean_diff"], reverse=True)
    return data[:top_n]


def _draw_panel(ax_bar, ax_auroc, entries, norm, cmap, concept: str, layer: int,
                show_ylabel: bool):
    n = len(entries)
    feat_ids  = [str(e["feature_idx"]) for e in entries]
    diffs     = np.array([e["mean_diff"] for e in entries])
    aurocs    = np.array([e["auroc"]    for e in entries])
    colors    = [cmap(norm(a)) for a in aurocs]

    # ── bars: mean_diff ───────────────────────────────────────────────
    ax_bar.bar(range(n), diffs, color=colors, width=0.72,
               linewidth=0, zorder=2)
    ax_bar.axhline(0, color="#444444", linewidth=0.6, zorder=1)

    ax_bar.yaxis.grid(True, linestyle="--", linewidth=0.4, color="#cccccc", zorder=0)
    ax_bar.set_axisbelow(True)

    ax_bar.set_xticks(range(n))
    ax_bar.set_xticklabels(feat_ids, rotation=45, ha="right")
    ax_bar.set_xlim(-0.6, n - 0.4)

    if show_ylabel:
        ax_bar.set_ylabel("Mean Activation Difference")
    ax_bar.set_xlabel("SAE Feature Index")

    for spine in ["top", "right"]:
        ax_bar.spines[spine].set_visible(False)

    # ── AUROC overlay on twin axis ────────────────────────────────────
    ax_auroc.plot(range(n), aurocs, color="#222222", linewidth=1.0,
                  linestyle="--", zorder=4)
    ax_auroc.scatter(range(n), aurocs, c=colors, s=28,
                     edgecolors="#222222", linewidths=0.4, zorder=5)
    ax_auroc.set_ylim(0.50, 1.05)
    ax_auroc.axhline(0.5, color="#999999", linewidth=0.5, linestyle=":")

    # hide all twin-axis ticks/labels — colorbar is the sole AUROC reference
    ax_auroc.tick_params(axis="y", left=False, right=False,
                         labelleft=False, labelright=False)
    ax_auroc.set_yticklabels([])
    for spine in ["top", "right"]:
        ax_auroc.spines[spine].set_visible(False)

    ax_bar.set_title(
        f"Top-{n} {CONCEPT_LABELS[concept]} Features  (Layer {layer})",
        pad=6,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer",    type=int, default=25)
    parser.add_argument("--top-n",    type=int, default=30,
                        help="Number of top features to show per concept (default: all 30)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Directory containing tool_call/ and request_for_info/ subdirs")
    parser.add_argument("--out",      default=None,
                        help="Output filename (default: combined_top_features_bar_layer{L}.png)")
    args = parser.parse_args()

    data_dir = args.data_dir
    concepts = ["tool_call", "request_for_info"]
    all_data  = {c: load_features(data_dir, args.layer, c, args.top_n) for c in concepts}

    # ── shared colormap normalization ─────────────────────────────────
    all_aurocs = [e["auroc"] for c in concepts for e in all_data[c]]
    vmin = max(0.50, min(all_aurocs) - 0.02)
    vmax = min(1.00, max(all_aurocs) + 0.02)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(CMAP_NAME)

    n = args.top_n
    # scale figure width to number of bars
    fig_w = max(14, n * 0.9)
    fig, axes = plt.subplots(
        1, 2,
        figsize=(fig_w, 5.6),
        gridspec_kw={"wspace": 0.095},
    )

    twin_axes = [ax.twinx() for ax in axes]

    for i, concept in enumerate(concepts):
        _draw_panel(
            axes[i], twin_axes[i],
            all_data[concept], norm, cmap, concept, args.layer,
            show_ylabel=(i == 0),
        )

    # ── shared colorbar ───────────────────────────────────────────────
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), pad=0.02, fraction=0.018, aspect=30)
    cbar.set_label("AUROC", labelpad=6)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(width=0.5)

    # ── panel labels ─────────────────────────────────────────────────
    for ax, letter in zip(axes, ["(a)", "(b)"]):
        ax.text(-0.08, 1.06, letter, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top", ha="left")

    out_name = args.out or f"combined_top_features_bar_layer{args.layer}.png"
    out_path = data_dir / out_name
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")

    # also save PDF for paper submission
    pdf_path = out_path.with_suffix(".pdf")
    fig2, axes2 = plt.subplots(
        1, 2,
        figsize=(fig_w, 4.6),
        gridspec_kw={"wspace": 0.095},
    )
    twin_axes2 = [ax.twinx() for ax in axes2]
    for i, concept in enumerate(concepts):
        _draw_panel(axes2[i], twin_axes2[i], all_data[concept], norm, cmap,
                    concept, args.layer, show_ylabel=(i == 0))
    sm2 = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm2.set_array([])
    cbar2 = fig2.colorbar(sm2, ax=axes2.tolist(), pad=0.01, fraction=0.018, aspect=30)
    cbar2.set_label("AUROC", labelpad=6)
    cbar2.outline.set_visible(False)
    cbar2.ax.tick_params(width=0.5)
    for ax, letter in zip(axes2, ["(a)", "(b)"]):
        ax.text(-0.08, 1.06, letter, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top", ha="left")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved → {pdf_path}")


if __name__ == "__main__":
    main()
