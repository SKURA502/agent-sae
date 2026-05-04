"""
Feature attribution plots grouped by model family.
One figure per family (Qwen / Gemma / Ministral), 2 rows × 2 cols layout.
- Model label shown vertically in the gap between the two panels per row.
- Subplot titles only on the top row.
- Legend: lower right for Gemma, upper right for others.

Output: /data/Agent-Tool-Use-MI/outputs/feature_attribution_{Family}.pdf

Usage:
    python -m utils_validation.bias.replot_feature_attribution
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

BASE_OUTPUTS = Path("/data/Agent-Tool-Use-MI/outputs")
TOP_N = 15

# Colors match plot_loss_curves.py
FAMILIES = [
    dict(label="Qwen", models=[
        dict(id="Qwen3.5-4B",                   layer=25, label="Qwen3.5-4B",  color="#C8580E"),
        dict(id="Qwen3.5-9B",                   layer=25, label="Qwen3.5-9B",  color="#E8901A"),
    ]),
    dict(label="Gemma", models=[
        dict(id="gemma-3-1b-it",                layer=17, label="Gemma-3-1B",  color="#1A7340"),
        dict(id="gemma-3-4b-it",                layer=29, label="Gemma-3-4B",  color="#3DAA6B"),
    ]),
    dict(label="Ministral", models=[
        dict(id="Ministral-3-3B-Instruct-2512", layer=21, label="Ministral-3B", color="#0D4EA6"),
        dict(id="Ministral-3-8B-Instruct-2512", layer=31, label="Ministral-8B", color="#3A7FC1"),
    ]),
]


def load_data(model_id: str, layer: int) -> dict:
    path = BASE_OUTPUTS / model_id / "analysis" / "rfi_confusion" / f"rfi_confusion_layer{layer}.json"
    with open(path) as f:
        return json.load(f)


def draw_panel(ax, feat_ids, fail_vals, succ_vals, diff_vals,
               n_fail, n_succ, title, color,
               show_ylabel: bool, show_title: bool, show_xlabel: bool, legend_loc: str):
    n   = len(feat_ids)
    x   = np.arange(n)
    w   = 0.34
    lbl = [str(fid) for fid in feat_ids]

    ax.bar(x - w / 2, fail_vals, w,
           label=f"TC fail  ($n$={n_fail})",
           color=color, alpha=0.88, linewidth=0, zorder=3)
    ax.bar(x + w / 2, succ_vals, w,
           label=f"NC succ  ($n$={n_succ})",
           color=color, alpha=0.32, linewidth=0, zorder=3)

    ymax = max(max(fail_vals), max(succ_vals))
    for xi, (fv, sv, dv) in enumerate(zip(fail_vals, succ_vals, diff_vals)):
        top = max(fv, sv) + ymax * 0.03
        ax.text(xi, top, f"Δ{dv:+.1f}",
                ha="center", va="bottom", fontsize=7, color="#444444")

    ax.axhline(0, color="#888888", linewidth=0.6, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(lbl, rotation=45, fontsize=8, ha="right")
    if show_xlabel:
        ax.set_xlabel("SAE Feature Index", labelpad=6)
    if show_ylabel:
        ax.set_ylabel("Mean activation", labelpad=6)
    if show_title:
        ax.set_title(title, pad=8)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(axis="y", which="major", zorder=0)
    ax.grid(axis="y", which="minor", zorder=0, alpha=0.4)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.legend(frameon=True, framealpha=0.92, edgecolor="#cccccc",
              loc=legend_loc, handlelength=1.2, handletextpad=0.5)


def plot_family(fam: dict):
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.size":         10,
        "axes.titlesize":    13,
        "axes.labelsize":    12,
        "xtick.labelsize":   8,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "axes.linewidth":    0.8,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "grid.color":        "#e0e0e0",
        "grid.linewidth":    0.6,
        "pdf.fonttype":      42,
        "svg.fonttype":      "none",
    })

    legend_loc = "lower right" if fam["label"] == "Gemma" else "upper right"
    n_models   = len(fam["models"])

    fig, axes = plt.subplots(
        n_models, 2,
        figsize=(14, 4.2 * n_models),
        gridspec_kw={"wspace": 0.22, "hspace": 0.20},
    )

    for row, m in enumerate(fam["models"]):
        data   = load_data(m["id"], m["layer"])
        n_fail = data["pred_tool_call"]["n_samples"]
        n_succ = data["pred_request_for_info"]["n_samples"]

        tc_entries  = data["feature_attribution"]["tc_features_overactivated"][:TOP_N]
        rfi_entries = data["feature_attribution"]["rfi_features_underactivated"][:TOP_N]

        tc_ids   = [e["feature_idx"] for e in tc_entries]
        tc_fail  = [e["fail_mean"]   for e in tc_entries]
        tc_succ  = [e["succ_mean"]   for e in tc_entries]
        tc_diff  = [e["diff"]        for e in tc_entries]

        rfi_ids  = [e["feature_idx"] for e in rfi_entries]
        rfi_fail = [e["fail_mean"]   for e in rfi_entries]
        rfi_succ = [e["succ_mean"]   for e in rfi_entries]
        rfi_diff = [e["diff"]        for e in rfi_entries]

        show_title = (row == 0)
        ax_tc  = axes[row][0]
        ax_rfi = axes[row][1]

        show_xlabel = not show_title
        draw_panel(ax_tc,  tc_ids,  tc_fail,  tc_succ,  tc_diff,
                   n_fail, n_succ,
                   "TC features — overactivated in failure",
                   m["color"], show_ylabel=True,
                   show_title=show_title, show_xlabel=show_xlabel, legend_loc=legend_loc)
        draw_panel(ax_rfi, rfi_ids, rfi_fail, rfi_succ, rfi_diff,
                   n_fail, n_succ,
                   "NC features — underactivated in failure",
                   m["color"], show_ylabel=False,
                   show_title=show_title, show_xlabel=show_xlabel, legend_loc=legend_loc)

    # Place each model's label vertically in the gap between TC and NC panels
    fig.canvas.draw()
    for row, m in enumerate(fam["models"]):
        pos_tc  = axes[row][0].get_position()
        pos_rfi = axes[row][1].get_position()
        mid_x = (pos_tc.x1 + pos_rfi.x0) / 2
        mid_y = (pos_tc.y0  + pos_tc.y1)  / 2
        fig.text(mid_x, mid_y, m["label"],
                 ha="center", va="center",
                 fontsize=12, fontweight="bold", color=m["color"],
                 rotation=90)

    out_path = BASE_OUTPUTS / f"feature_attribution_{fam['label']}.pdf"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    for fam in FAMILIES:
        plot_family(fam)


if __name__ == "__main__":
    main()
