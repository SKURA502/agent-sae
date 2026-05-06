#!/usr/bin/env python3
"""
Combined UMAP visualization for Tool Call (tc) and No Call / request_for_info (nc)
discovered SAE features, across 6 models.

Layout: 3 rows × 2 cols
  Row 0 : Qwen3.5      (4B | 9B)
  Row 1 : Gemma-3      (1B | 4B)
  Row 2 : Ministral-3  (3B | 8B)

Each subplot shows a single decoder-latent UMAP of the model's SAE, with
Tool Call features and No Call features highlighted using the model family's
dark and light colours respectively.

Usage:
  cd /data/Agent-Tool-Use-MI
  python utils_validation/feature_discovery/plot_umap_tc_nc_combined.py
  python utils_validation/feature_discovery/plot_umap_tc_nc_combined.py \
      --output-path outputs/umap_tc_nc_combined.pdf \
      --umap-top-k 20 \
      --device cuda
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

# ── rcParams ──────────────────────────────────────────────────────────────────
_RC = {
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "font.family":        "serif",
    "font.size":          16,
    "axes.titlesize":     17,
    "axes.titleweight":   "bold",
    "axes.labelsize":     15,
    "legend.fontsize":    14,
    "xtick.labelsize":    13,
    "ytick.labelsize":    13,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
}
plt.rcParams.update(_RC)


# ── Model / family config (mirrors linear_probe_combined.sh) ──────────────────
OUTPUT_BASE = Path("/data/Agent-Tool-Use-MI")
CHECKPOINT_BASE = OUTPUT_BASE / "checkpoint"
MODEL_BASE = Path(os.environ.get("SOURCE_ROOT", "")) / "model"

SERIES = [
    {
        "name": "Qwen3.5",
        # dark = tool_call highlight, light = no_call highlight
        "tc_color":  "#C07048",   # dark orange
        "nc_color":  "#F2C0AB",   # light orange
        "kde_cmap":  "Oranges",
        "models": [
            {
                "id":    "Qwen3.5-4B",
                "layer": 25,
                "sae":   "Qwen3.5-4B-L25-d20480-5M-stage2.pt",
                "short": "Qwen3.5-4B",
            },
            {
                "id":    "Qwen3.5-9B",
                "layer": 25,
                "sae":   "Qwen3.5-9B-L25-d32768-5M-stage2.pt",
                "short": "Qwen3.5-9B",
            },
        ],
    },
    {
        "name": "Gemma-3",
        "tc_color":  "#3A8860",   # dark green
        "nc_color":  "#98C6A8",   # light green
        "kde_cmap":  "Greens",
        "models": [
            {
                "id":    "gemma-3-1b-it",
                "layer": 17,
                "sae":   "gemma-3-1b-it-L17-d9216-5M-stage2.pt",
                "short": "Gemma-3-1B",
            },
            {
                "id":    "gemma-3-4b-it",
                "layer": 29,
                "sae":   "gemma-3-4b-it-L29-d20480-5M-stage2.pt",
                "short": "Gemma-3-4B",
            },
        ],
    },
    {
        "name": "Ministral-3",
        "tc_color":  "#3D77A8",   # dark blue
        "nc_color":  "#9DC3DB",   # light blue
        "kde_cmap":  "Blues",
        "models": [
            {
                "id":    "Ministral-3-3B-Instruct-2512",
                "layer": 21,
                "sae":   "Ministral-3-3B-Instruct-2512-L21-d24576-5M-stage2.pt",
                "short": "Ministral-3B",
            },
            {
                "id":    "Ministral-3-8B-Instruct-2512",
                "layer": 31,
                "sae":   "Ministral-3-8B-Instruct-2512-L31-d32768-5M-stage2.pt",
                "short": "Ministral-8B",
            },
        ],
    },
]

CONCEPTS = [
    ("tool_call",        "Tool Call"),
    ("request_for_info", "No Call"),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_top_features(model_id: str, layer: int, concept: str, top_k: int) -> list[int]:
    path = (
        OUTPUT_BASE / "outputs" / model_id
        / "analysis" / "feature_discovery" / concept
        / f"top_features_layer{layer}.json"
    )
    if not path.exists():
        print(f"  [WARN] top_features not found: {path}", file=sys.stderr)
        return []
    with open(path) as f:
        data = json.load(f)
    return [e["feature_idx"] for e in data[:top_k]]


def load_decoder_matrix(model_id: str, sae_filename: str, device: str) -> np.ndarray:
    """Load SAE checkpoint and return decoder weight matrix [dict_size, input_dim]."""
    sae_path = CHECKPOINT_BASE / model_id / "stage2" / sae_filename
    if not sae_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {sae_path}")

    # Add project root to path so 'sae' package is importable
    project_root = str(OUTPUT_BASE)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from sae.sae_model import TopKSAE  # noqa: PLC0415

    print(f"  Loading SAE: {sae_path.name}")
    sae = TopKSAE.load(str(sae_path), device=device)
    sae.eval()
    with torch.no_grad():
        # decoder.weight: [input_dim, dict_size] → transpose → [dict_size, input_dim]
        W = sae.decoder.weight.detach().float().cpu().T.numpy()
    print(f"  Decoder matrix: {W.shape}")
    del sae
    torch.cuda.empty_cache() if device != "cpu" else None
    return W


def run_umap(W: np.ndarray, n_neighbors: int, min_dist: float) -> np.ndarray:
    """UMAP on decoder rows (each row = one feature direction). Returns [dict_size, 2]."""
    try:
        import umap as umap_lib
    except ImportError:
        raise ImportError("umap-learn not installed. Run: pip install umap-learn")

    print(f"  Running UMAP on {W.shape[0]} features …")
    reducer = umap_lib.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="cosine",
        random_state=42,
        verbose=False,
    )
    embedding = reducer.fit_transform(W)
    print("  UMAP done.")
    return embedding


def plot_subplot(
    ax: plt.Axes,
    embedding: np.ndarray,
    tc_indices: list[int],
    nc_indices: list[int],
    tc_color: str,
    nc_color: str,
    kde_cmap: str,
    title: str,
    layer: int,
):
    from scipy.stats import gaussian_kde

    dict_size = embedding.shape[0]
    tc_set = set(tc_indices)
    nc_set = set(nc_indices)

    is_tc = np.array([i in tc_set for i in range(dict_size)], dtype=bool)
    is_nc = np.array([i in nc_set for i in range(dict_size)], dtype=bool)
    is_bg = ~is_tc & ~is_nc  # neither

    # KDE background — neutral grey to avoid colour clash with highlights
    try:
        kde = gaussian_kde(embedding.T, bw_method=0.15)
        x_min, x_max = embedding[:, 0].min() - 1, embedding[:, 0].max() + 1
        y_min, y_max = embedding[:, 1].min() - 1, embedding[:, 1].max() + 1
        grid_x, grid_y = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        density = kde(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(200, 200)
        ax.contourf(grid_x, grid_y, density, levels=12, cmap="Greys", alpha=0.40)
    except Exception as e:
        print(f"  [WARN] KDE failed: {e}", file=sys.stderr)

    # Background dots — slightly darker so highlights pop
    ax.scatter(
        embedding[is_bg, 0], embedding[is_bg, 1],
        s=3, c="#999999", alpha=0.30, linewidths=0, rasterized=True,
    )

    # No Call features — light family colour with dark outline for contrast
    if is_nc.any():
        ax.scatter(
            embedding[is_nc, 0], embedding[is_nc, 1],
            s=130, c=nc_color, alpha=1.0, linewidths=1.5,
            edgecolors=tc_color, zorder=4, marker="s",
            label=f"No Call  ({is_nc.sum()})",
        )

    # Tool Call features — dark family colour with white outline
    if is_tc.any():
        ax.scatter(
            embedding[is_tc, 0], embedding[is_tc, 1],
            s=150, c=tc_color, alpha=1.0, linewidths=1.5,
            edgecolors="white", zorder=5, marker="^",
            label=f"Tool Call  ({is_tc.sum()})",
        )

    ax.set_title(f"{title}  (L{layer})", fontsize=16, fontweight="bold")
    ax.set_xlabel("UMAP-1", fontsize=14)
    ax.set_ylabel("UMAP-2", fontsize=14)
    ax.tick_params(labelsize=8)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Combined UMAP: Tool Call vs No Call features across 6 models"
    )
    parser.add_argument(
        "--output-path", type=str,
        default=str(OUTPUT_BASE / "outputs" / "umap_tc_nc_combined.pdf"),
        help="Output file path (.pdf or .png)",
    )
    parser.add_argument("--umap-top-k",     type=int,   default=20,
                        help="Number of top features to highlight per concept")
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist",  type=float, default=0.1)
    parser.add_argument("--device",         type=str,   default="cuda")
    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(SERIES)       # 3 families
    n_cols = 2                 # 2 models per family
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(14, 18),
        constrained_layout=True,
    )

    for row_idx, family in enumerate(SERIES):
        family_name  = family["name"]
        tc_color     = family["tc_color"]
        nc_color     = family["nc_color"]
        kde_cmap     = family["kde_cmap"]

        for col_idx, model_cfg in enumerate(family["models"]):
            ax         = axes[row_idx, col_idx]
            model_id   = model_cfg["id"]
            layer      = model_cfg["layer"]
            short_name = model_cfg["short"]
            sae_fn     = model_cfg["sae"]

            print(f"\n{'='*60}")
            print(f"  {family_name}  |  {short_name}  |  Layer {layer}")
            print(f"{'='*60}")

            # Load decoder → UMAP
            try:
                W = load_decoder_matrix(model_id, sae_fn, args.device)
            except FileNotFoundError as e:
                print(f"  [SKIP] {e}", file=sys.stderr)
                ax.set_visible(False)
                continue

            embedding = run_umap(W, args.umap_n_neighbors, args.umap_min_dist)
            del W

            # Load top features for each concept
            tc_indices = load_top_features(model_id, layer, "tool_call",        args.umap_top_k)
            nc_indices = load_top_features(model_id, layer, "request_for_info", args.umap_top_k)
            print(f"  tool_call top-{args.umap_top_k}: {len(tc_indices)} features")
            print(f"  request_for_info top-{args.umap_top_k}: {len(nc_indices)} features")

            plot_subplot(
                ax, embedding,
                tc_indices, nc_indices,
                tc_color, nc_color, kde_cmap,
                title=short_name,
                layer=layer,
            )

            # Per-subplot legend
            ax.legend(loc="upper right", fontsize=13, framealpha=0.7, markerscale=1.2)

        # Row label (family name) on the left side
        axes[row_idx, 0].set_ylabel(
            f"{family_name}\nUMAP-2",
            fontsize=15,
            fontweight="bold",
            labelpad=8,
        )

    # Cross-family legend patches (concept shapes are universal)
    tc_patch = mpatches.Patch(color="#555555", label="▲ Tool Call (dark shade per family)")
    nc_patch = mpatches.Patch(color="#AAAAAA", label="■ No Call  (light shade per family)")
    fig.legend(
        handles=[tc_patch, nc_patch],
        loc="lower center",
        ncol=2,
        fontsize=14,
        framealpha=0.8,
        bbox_to_anchor=(0.5, -0.06),
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n{'='*60}")
    print(f"Saved → {output_path}")

    # Also save PNG alongside if PDF was requested
    if output_path.suffix == ".pdf":
        png_path = output_path.with_suffix(".png")
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {png_path}")

    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
