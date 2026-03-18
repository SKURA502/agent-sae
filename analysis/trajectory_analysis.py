"""
Trajectory Analysis - H2 门控 feature 强度随 agent step 演化分析

加载 generate_rollouts.py 保存的 per-step 激活，通过 SAE encode 计算各 feature 强度，
绘制成功 vs 失败 episode 的轨迹曲线（Fig 3）。

前置条件：
  generate_rollouts.py 运行时保存了激活文件：
    {rollouts_dir}/activations/{episode_id}_layer{L}.pt  → Tensor[n_steps, hidden]

CLI:
  python -m analysis.trajectory_analysis \\
    --rollouts-dir data/rollouts \\
    --sae-path outputs/sae_checkpoints/stage2/layer24/best.pt \\
    --layer 24 \\
    --feature-indices 42 17 \\
    --output-dir outputs/analysis

输出:
  outputs/analysis/trajectory_analysis_layer{L}.json  - 轨迹数据
  outputs/analysis/trajectory_feat{F}_layer{L}.png    - Fig 3（CALL vs NO_CALL episode 对比）
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


# ─────────────────────── helpers ───────────────────────────────────

def load_episode_activations(
    activations_dir: Path, episode_id: str, layer: int
) -> Optional[torch.Tensor]:
    """加载单个 episode 的 per-step 激活。返回 [n_steps, hidden] 或 None。"""
    path = activations_dir / f"{episode_id}_layer{layer}.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=True)


def compute_feature_trajectory(
    sae, acts: torch.Tensor, feature_indices: List[int]
) -> Dict[int, List[float]]:
    """对一个 episode 的 per-step 激活，计算各 feature 的强度序列。

    Args:
        acts: [n_steps, hidden]
    Returns:
        {feature_idx: [strength_step_0, strength_step_1, ...]}
    """
    acts_f32 = acts.to(torch.float32).to(sae.config.device)
    with torch.no_grad():
        latents = sae.encode(acts_f32)  # [n_steps, dict_size]
    return {f: latents[:, f].cpu().tolist() for f in feature_indices}


# ─────────────────────── plotting ──────────────────────────────────

def _plot_trajectories(
    call_trajs: List[Dict[int, List[float]]],
    no_call_trajs: List[Dict[int, List[float]]],
    feat_idx: int,
    layer: int,
    out_path: Path,
):
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        def _draw(ax, trajs, title, color):
            if not trajs:
                ax.set_title(f"{title}\n(no data)")
                return
            max_len = max(len(t[feat_idx]) for t in trajs)
            arr = np.full((len(trajs), max_len), np.nan)
            for i, t in enumerate(trajs):
                seq = t[feat_idx]
                arr[i, : len(seq)] = seq
            mean = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0)
            steps = np.arange(max_len)
            for row in arr:
                valid = ~np.isnan(row)
                ax.plot(steps[valid], row[valid], alpha=0.15, color=color)
            ax.plot(steps, mean, color=color, linewidth=2, label="mean")
            ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=color)
            ax.set_xlabel("Agent Step")
            ax.set_ylabel(f"Feature {feat_idx} Activation")
            ax.set_title(title)
            ax.legend()

        _draw(axes[0], call_trajs, f"CALL Episodes (n={len(call_trajs)})", "steelblue")
        _draw(axes[1], no_call_trajs, f"NO_CALL Episodes (n={len(no_call_trajs)})", "salmon")

        plt.suptitle(f"Feature {feat_idx} Activation Trajectory (layer {layer})")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved figure → {out_path}")
    except ImportError:
        print("matplotlib not available, skipping figure")


# ─────────────────────── CLI ───────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="H2 轨迹分析：追踪门控 feature 强度随 agent step 演化"
    )
    parser.add_argument("--rollouts-dir", type=str, required=True)
    parser.add_argument("--sae-path", type=str, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--feature-indices", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, default="./outputs/analysis")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-episodes", type=int, default=-1,
                        help="最多分析的 episode 数（-1 = all）")
    args = parser.parse_args()

    from sae.sae_model import TopKSAE
    from run.rollout_logger import RolloutLogger

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rollouts_dir = Path(args.rollouts_dir)
    activations_dir = rollouts_dir / "activations"

    print(f"Loading SAE from {args.sae_path}")
    sae = TopKSAE.load(args.sae_path, device=args.device)
    sae.eval()

    print(f"Loading rollout logs from {rollouts_dir}")
    logger = RolloutLogger(str(rollouts_dir))
    all_logs = logger.load_all()
    if args.max_episodes > 0:
        all_logs = all_logs[: args.max_episodes]
    print(f"  {len(all_logs)} episodes loaded")

    # ── Compute feature trajectories ──────────────────────────────────
    call_trajectories: List[Dict[int, List[float]]] = []
    no_call_trajectories: List[Dict[int, List[float]]] = []
    missing = 0

    for log in all_logs:
        acts = load_episode_activations(activations_dir, log.episode_id, args.layer)
        if acts is None or len(acts) == 0:
            missing += 1
            continue

        traj = compute_feature_trajectory(sae, acts, args.feature_indices)
        last_decision = log.steps[-1]["decision"] if log.steps else "NO_CALL"
        if last_decision == "CALL":
            call_trajectories.append(traj)
        else:
            no_call_trajectories.append(traj)

    if missing:
        print(f"  Warning: {missing} episodes missing activation files")
    print(f"  {len(call_trajectories)} CALL episodes, {len(no_call_trajectories)} NO_CALL episodes")

    # ── Save JSON ─────────────────────────────────────────────────────
    results = {
        "feature_indices": args.feature_indices,
        "layer": args.layer,
        "n_call_episodes": len(call_trajectories),
        "n_no_call_episodes": len(no_call_trajectories),
        "call_trajectories": call_trajectories,
        "no_call_trajectories": no_call_trajectories,
    }
    out_json = output_dir / f"trajectory_analysis_layer{args.layer}.json"
    with open(out_json, "w") as f:
        json.dump(results, f)
    print(f"Saved trajectory data → {out_json}")

    # ── Plot (Fig 3) ──────────────────────────────────────────────────
    for feat_idx in args.feature_indices:
        _plot_trajectories(
            call_trajectories, no_call_trajectories,
            feat_idx, args.layer,
            output_dir / f"trajectory_feat{feat_idx}_layer{args.layer}.png",
        )


if __name__ == "__main__":
    main()
