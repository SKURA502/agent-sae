"""
Produces a 2×3 summary figure of the linear probe analysis across six models.

Layout:
  rows = concept labels  (tool_call, request_for_info)
  cols = model families  (Qwen3.5,  Ministral-3,  Gemma-3)

Each panel shows AUC vs. K for two model sizes within the family, with three
curves per model: Top-K related SAE features, raw hidden-state (upper bound),
and random-SAE baseline.  Within a model, all three curves share one colour;
the condition is distinguished by line style and marker.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Global rcParams ───────────────────────────────────────────────────────────
_RC = {
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
    "font.family":          "serif",
    "font.size":            11,
    "axes.titlesize":       20,       # 2× original 10
    "axes.titleweight":     "bold",
    "axes.labelsize":       18,       # 2× original 9
    "legend.fontsize":      11,
    "xtick.labelsize":      15,       # 2× original 7.5
    "ytick.labelsize":      15,       # 2× original 7.5
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.grid":            True,
    "grid.linestyle":       "--",
    "grid.alpha":           0.30,
    "grid.color":           "#CCCCCC",
}


# ── Model-family configuration ────────────────────────────────────────────────
SERIES = [
    {
        "name": "Qwen3.5",
        "models": [
            {"id": "Qwen3.5-4B",  "layer": 25, "short": "4B", "color": "#F2C0AB"},  # 橙
            {"id": "Qwen3.5-9B",  "layer": 25, "short": "9B", "color": "#C07048"},
        ],
    },
    {
        "name": "Gemma-3",
        "models": [
            {"id": "gemma-3-1b-it", "layer": 17, "short": "1B", "color": "#98C6A8"},  # 绿
            {"id": "gemma-3-4b-it", "layer": 29, "short": "4B", "color": "#3A8860"},
        ],
    },
    {
        "name": "Ministral-3",
        "models": [
            {"id": "Ministral-3-3B-Instruct-2512", "layer": 21, "short": "3B", "color": "#9DC3DB"},  # 蓝
            {"id": "Ministral-3-8B-Instruct-2512", "layer": 31, "short": "8B", "color": "#3D77A8"},
        ],
    },
]

LABELS = [
    ("tool_call",        "Tool Call"),
    ("request_for_info", "No Call"),
]

# Three conditions: same colour per model, distinguished by line/marker style.
_COND = {
    "select": dict(ls="--",          mk="^",  ms=7.0, lw=2.0, z=4, tag="Top-$K$ related SAE"),
    "upper":  dict(ls="-",           mk="",   ms=0.0, lw=1.7, z=3, tag="Raw (upper bound)"),
    "random": dict(ls=(0, (3, 2)),   mk="o",  ms=6.5, lw=1.5, z=2, tag="Random SAE"),
}


# ── Custom y-axis scale: equally-spaced ticks at 0.5 0.65 0.8 0.9 1.0 ────────
# Piecewise-linear forward/inverse maps data values to equally-spaced display
# positions so the five tick marks are visually equidistant.
_YTICKS = [0.5, 0.65, 0.8, 0.9, 1.0]
_D_BP   = np.array([0.44,  0.50, 0.65, 0.80, 0.90, 1.00, 1.03])
_P_BP   = np.array([-0.40, 0.00, 1.00, 2.00, 3.00, 4.00, 4.30])

def _yfwd(y): return np.interp(y, _D_BP, _P_BP)
def _yinv(x): return np.interp(x, _P_BP, _D_BP)


# ── Data helpers ──────────────────────────────────────────────────────────────
def _load(base: Path, model_id: str, label: str, layer: int) -> Optional[dict]:
    p = (base / "outputs" / model_id / "analysis" / "linear_probe"
         / label / f"linear_probe_layer{layer}.json")
    return json.loads(p.read_text()) if p.exists() else None


# ── Per-panel drawing ─────────────────────────────────────────────────────────
def _draw(ax: plt.Axes, series: dict, label: str, base: Path,
          legend_y: float = 0.60) -> None:
    handles, leg_labels = [], []

    # ── Determine K values and equal-spaced positions from first available data ──
    ks: list = []
    positions: list = []
    for m in series["models"]:
        d = _load(base, m["id"], label, m["layer"])
        if d is not None:
            ks = sorted(int(k) for k in d["cv_results_by_k"])
            positions = list(range(len(ks)))
            break

    # ── Band helper (uses positions for equal spacing) ────────────────────────
    def _band(c, mu_arr, sd_arr, alpha=0.10):
        ax.fill_between(
            positions,
            np.clip(mu_arr - sd_arr, 0.0, 1.0),
            np.clip(mu_arr + sd_arr, 0.0, 1.0),
            color=c, alpha=alpha, zorder=0, linewidth=0,
        )

    # ── Draw each model in the series ─────────────────────────────────────────
    missing = []
    for m in series["models"]:
        d = _load(base, m["id"], label, m["layer"])
        if d is None:
            missing.append(f"{series['name']} {m['short']}")
            continue

        c   = m["color"]
        nm  = m["short"]
        kst = [str(k) for k in ks]

        mu_sel = np.array([d["cv_results_by_k"][k]["mean_auc"]     for k in kst])
        sd_sel = np.array([d["cv_results_by_k"][k]["std_auc"]      for k in kst])
        mu_rnd = np.array([d["random_sae_baseline"][k]["mean_auc"] for k in kst])
        sd_rnd = np.array([d["random_sae_baseline"][k]["std_auc"]  for k in kst])
        raw_mu = d["raw_activation_baseline"]["mean_auc"]
        raw_sd = d["raw_activation_baseline"]["std_auc"]

        # Top-K related SAE (select concept)
        s = _COND["select"]
        h, = ax.plot(positions, mu_sel, color=c, ls=s["ls"], marker=s["mk"],
                     ms=s["ms"], lw=s["lw"], zorder=s["z"], clip_on=True)
        _band(c, mu_sel, sd_sel, alpha=0.13)
        handles.append(h)
        leg_labels.append(f"{nm} — {s['tag']}")

        # Raw hidden state (upper bound, flat)
        s = _COND["upper"]
        mu_up = np.full(len(positions), raw_mu)
        h, = ax.plot(positions, mu_up, color=c, ls=s["ls"], marker=s["mk"],
                     ms=s["ms"], lw=s["lw"], zorder=s["z"], clip_on=True)
        _band(c, mu_up, np.full(len(positions), raw_sd), alpha=0.08)
        handles.append(h)
        leg_labels.append(f"{nm} — {s['tag']}")

        # Random SAE baseline
        s = _COND["random"]
        h, = ax.plot(positions, mu_rnd, color=c, ls=s["ls"], marker=s["mk"],
                     ms=s["ms"], lw=s["lw"], zorder=s["z"], clip_on=True)
        _band(c, mu_rnd, sd_rnd, alpha=0.10)
        handles.append(h)
        leg_labels.append(f"{nm} — {s['tag']}")

    # Show "No data" for missing models
    for i, name in enumerate(missing):
        ax.text(0.5, 0.52 - i * 0.12, f"No data\n{name}",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#AAAAAA", style="italic")

    # ── Chance reference line ─────────────────────────────────────────────────
    ax.axhline(0.5, color="#C0C0C0", lw=0.9, ls=":", zorder=0)

    # ── Axes formatting: equal-spaced x ticks ────────────────────────────────
    if positions:
        ax.set_xticks(positions)
        ax.set_xticklabels([str(k) for k in ks], rotation=0, ha="center")
        ax.set_xlim(-0.6, len(positions) - 0.4)
    ax.set_yscale("function", functions=(_yfwd, _yinv))
    ax.set_ylim(0.44, 1.03)
    ax.set_yticks(_YTICKS)

    # ── Legend (center right) ─────────────────────────────────────────────────
    # Row-major with ncol = n_data  →  each row = one condition, each col = one model
    n_data = len(handles) // 3 if handles else 0
    if handles:
        per_model = [handles[i * 3: i * 3 + 3] for i in range(n_data)]
        per_model_lbl = [leg_labels[i * 3: i * 3 + 3] for i in range(n_data)]
        flat_h = [per_model[m][cond] for cond in [1, 0, 2] for m in range(n_data)]
        flat_l = [per_model_lbl[m][cond] for cond in [1, 0, 2] for m in range(n_data)]
        ax.legend(
            flat_h, flat_l,
            ncol=1,
            loc="center right",
            bbox_to_anchor=(0.99, legend_y),
            framealpha=0.88,
            fontsize=11,
            borderpad=0.55,
            handlelength=2.2,
            labelspacing=0.30,
            columnspacing=0.80,
        )


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Combined linear probe figure (6 models × 2 labels)"
    )
    ap.add_argument("--output-base", default="/data/Agent-Tool-Use-MI",
                    help="Root directory containing outputs/ and checkpoint/")
    ap.add_argument("--output-path", default=None,
                    help="Destination PNG (default: <output-base>/outputs/linear_probe_combined.png)")
    args = ap.parse_args()

    base     = Path(args.output_base)
    out_path = (Path(args.output_path) if args.output_path
                else base / "outputs" / "linear_probe_combined.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(_RC)

    nr, nc = len(LABELS), len(SERIES)
    fig, axes = plt.subplots(nr, nc, figsize=(20, 10))

    for r, (lbl_key, lbl_title) in enumerate(LABELS):
        for c, ser in enumerate(SERIES):
            ax = axes[r][c]
            _draw(ax, ser, lbl_key, base,
                  legend_y=0.60 if r == 0 else 0.51)

            ax.set_title(f"{ser['name']} — {lbl_title}")

            if c > 0:
                ax.tick_params(labelleft=False)
            if r < nr - 1:
                ax.tick_params(labelbottom=False)

    # rect=[left, bottom, right, top] reserves margins for the shared labels.
    # Adjust the label x/y positions and rect edges together to control distance.
    plt.tight_layout(pad=2.0, h_pad=1.0, w_pad=0.5, rect=[0.005, 0.02, 1.0, 1.0])
    fig.text(0.5,  0.01, "Number of features $K$",
             ha="center", va="bottom", fontsize=18, fontweight="bold")
    fig.text(0.00, 0.5,  "Mean AUROC (5-fold CV)",
             ha="left",   va="center", fontsize=18, fontweight="bold",
             rotation="vertical")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined figure → {out_path}")


if __name__ == "__main__":
    main()
