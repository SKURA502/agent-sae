#!/usr/bin/env python3
"""
Multi-model combined top-feature bar chart.

5 models stacked vertically (one row each).
Left column = Tool Call, right column = No Call.
Column headers shown once at the top; model name centered above each row.

Usage:
    python plot_multi_model_bar.py [--top-n 20] [--out multi_model_bar.png]
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
OUTPUTS_DIR = HERE.parent.parent / "outputs"

rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    14,   # y-axis title: +2
    "axes.labelweight":  "bold",
    "xtick.labelsize":   8,
    "ytick.labelsize":   10,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth":   1.2,
    "pdf.fonttype":      42,   # TrueType embedding — vector fonts in PDF
    "svg.fonttype":      "none",
})

# (model_dir_name, layer)
MODELS = [
    ("gemma-3-1b-it",                17),
    ("gemma-3-4b-it",                29),
    ("Ministral-3-3B-Instruct-2512", 21),
    ("Ministral-3-8B-Instruct-2512", 31),
    ("Qwen3.5-4B",                   25),
]

CONCEPT_LABELS = {
    "tool_call":        "Tool Call",
    "request_for_info": "No Call",
}
CONCEPTS = ["tool_call", "request_for_info"]
CMAP_NAME = "RdYlGn"


def load_features(model: str, layer: int, concept: str, top_n: int) -> list:
    path = (OUTPUTS_DIR / model / "analysis" / "feature_discovery"
            / concept / f"top_features_layer{layer}.json")
    with open(path) as f:
        data = json.load(f)
    data.sort(key=lambda x: x["mean_diff"], reverse=True)
    return data[:top_n]


def _draw_panel(ax_bar, ax_auroc, entries, norm, cmap, show_ylabel: bool):
    n = len(entries)
    feat_ids = [str(e["feature_idx"]) for e in entries]
    diffs    = np.array([e["mean_diff"] for e in entries])
    aurocs   = np.array([e["auroc"]    for e in entries])
    colors   = [cmap(norm(a)) for a in aurocs]

    ax_bar.bar(range(n), diffs, color=colors, width=0.72, linewidth=0, zorder=2)
    ax_bar.axhline(0, color="#444444", linewidth=0.6, zorder=1)
    ax_bar.yaxis.grid(True, linestyle="--", linewidth=0.4, color="#cccccc", zorder=0)
    ax_bar.set_axisbelow(True)

    ax_bar.set_xticks(range(n))
    ax_bar.set_xticklabels(feat_ids, rotation=45, ha="right")
    ax_bar.set_xlim(-0.6, n - 0.4)


    for spine in ["top", "right"]:
        ax_bar.spines[spine].set_visible(False)

    # AUROC overlay on twin axis
    ax_auroc.plot(range(n), aurocs, color="#222222", linewidth=1.0,
                  linestyle="--", zorder=4)
    ax_auroc.scatter(range(n), aurocs, c=colors, s=28,
                     edgecolors="#222222", linewidths=0.4, zorder=5)
    ax_auroc.set_ylim(0.50, 1.05)
    ax_auroc.axhline(0.5, color="#999999", linewidth=0.5, linestyle=":")
    ax_auroc.tick_params(axis="y", left=False, right=False,
                         labelleft=False, labelright=False)
    ax_auroc.set_yticklabels([])
    for spine in ["top", "right"]:
        ax_auroc.spines[spine].set_visible(False)


def _build_figure(all_data, norm, cmap, top_n, fig_w, fig_h):
    n_rows = len(MODELS)
    fig, axes = plt.subplots(n_rows, 2, figsize=(fig_w, fig_h))
    fig.subplots_adjust(
        left=0.07, right=0.92,
        top=0.88,  bottom=0.04,
        hspace=0.55, wspace=0.10,
    )
    twin_axes = [[ax.twinx() for ax in row] for row in axes]

    for row_idx, (model_name, _) in enumerate(MODELS):
        for col_idx, concept in enumerate(CONCEPTS):
            _draw_panel(
                axes[row_idx, col_idx],
                twin_axes[row_idx][col_idx],
                all_data[model_name][concept],
                norm, cmap,
                show_ylabel=(col_idx == 0),
            )

    # Shared colorbar attached to rightmost column
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:, 1].tolist(),
                        pad=0.02, fraction=0.04, aspect=40)
    cbar.set_label("AUROC", labelpad=6)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(width=0.5)

    # Model name centered above each row
    for row_idx, (model_name, _) in enumerate(MODELS):
        pos_l = axes[row_idx, 0].get_position()
        pos_r = axes[row_idx, 1].get_position()
        cx = (pos_l.x0 + pos_r.x1) / 2
        fig.text(cx, pos_l.y1 + 0.007, model_name,
                 ha="center", va="bottom", fontsize=18, fontweight="bold")

    # Column headers above the first row (above model name)
    pos0l = axes[0, 0].get_position()
    pos0r = axes[0, 1].get_position()
    header_y = pos0l.y1 + 0.040
    fig.text((pos0l.x0 + pos0l.x1) / 2, header_y,
             f"Top-{top_n} {CONCEPT_LABELS['tool_call']} Features",
             ha="center", va="bottom", fontsize=20, fontweight="bold")
    fig.text((pos0r.x0 + pos0r.x1) / 2, header_y,
             f"Top-{top_n} {CONCEPT_LABELS['request_for_info']} Features",
             ha="center", va="bottom", fontsize=20, fontweight="bold")

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=20,
                        help="Number of top features per concept per model (default: 20)")
    parser.add_argument("--out",   default=None,
                        help="Output PDF filename (saved under outputs/)")
    args = parser.parse_args()

    top_n = args.top_n

    all_data = {}
    for model, layer in MODELS:
        all_data[model] = {
            concept: load_features(model, layer, concept, top_n)
            for concept in CONCEPTS
        }

    # Shared colormap across all models and concepts
    all_aurocs = [e["auroc"] for model, _ in MODELS
                  for concept in CONCEPTS for e in all_data[model][concept]]
    vmin = max(0.50, min(all_aurocs) - 0.02)
    vmax = min(1.00, max(all_aurocs) + 0.02)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps[CMAP_NAME]

    fig_w = max(14, top_n * 0.9)
    fig_h = len(MODELS) * 3.6

    out_stem = args.out or f"multi_model_top_features_bar_top{top_n}"
    out_stem = out_stem.removesuffix(".pdf")
    pdf_path = OUTPUTS_DIR / f"{out_stem}.pdf"

    fig = _build_figure(all_data, norm, cmap, top_n, fig_w, fig_h)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {pdf_path}")


if __name__ == "__main__":
    main()
