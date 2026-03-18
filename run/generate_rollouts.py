"""
Generate Rollouts - H2 sandbox rollout 批量生成

RolloutGenerator 封装 AgentLoop + RolloutLogger：
  - run_streaming(samples) → yield (EpisodeLog, activations_dict)，兼容 create_streaming_data_pipeline
  - generate(n_episodes, task_source, seed) → 批量生成并保存 rollout + per-step 激活

激活保存格式（H2 轨迹分析用）：
  data/rollouts/activations/{episode_id}_layer{L}.pt  → Tensor[n_steps, hidden]

CLI:
  python -m run.generate_rollouts \\
    --model Qwen/Qwen3.5-4B \\
    --domain retail \\
    --n-episodes 100 \\
    --output-dir data/rollouts \\
    --layers 24 26
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Tuple

import torch
from tqdm import tqdm

from .rollout_logger import EpisodeLog, RolloutLogger


# ─────────────────────── tau2-bench prompt loader ──────────────────

def load_tau2_prompts(domain: str, tau2_data_dir: str, n: Optional[int] = None) -> List[str]:
    """从 tau2-bench tasks.json 提取 reason_for_call 作为 task prompt seed。

    tau2-bench tasks.json 路径：{tau2_data_dir}/domains/{domain}/tasks.json
    每条 task 的 user_scenario.instructions.reason_for_call 描述具体业务场景。
    """
    import json
    tasks_path = Path(tau2_data_dir) / "domains" / domain / "tasks.json"
    with open(tasks_path) as f:
        tasks = json.load(f)

    prompts = []
    for task in tasks:
        scenario = task.get("user_scenario", {})
        instructions = scenario.get("instructions", {}) if isinstance(scenario, dict) else {}
        reason = instructions.get("reason_for_call", "")
        task_instr = instructions.get("task_instructions", "")
        prompt = (reason or task_instr or task.get("description", "")).strip()
        if prompt:
            prompts.append(prompt)

    return prompts[:n] if n is not None else prompts


# ─────────────────────── RolloutGenerator ──────────────────────────

class RolloutGenerator:
    """H2 rollout 批量生成器。"""

    def __init__(
        self,
        agent_loop,
        logger: RolloutLogger,
        streamer=None,                  # 保留接口，暂未使用（激活由 agent_loop 直接收集）
    ):
        self.agent_loop = agent_loop
        self.logger = logger
        self.streamer = streamer

    def run_streaming(
        self, samples
    ) -> Generator[Tuple[EpisodeLog, Dict[int, List[torch.Tensor]]], None, None]:
        """从 samples 逐个生成 rollout，yield (EpisodeLog, activations)。

        activations: {layer: [act_step0, act_step1, ...]}，每个 act 为 [1, hidden]
        兼容 create_streaming_data_pipeline 的接口。
        """
        for i, sample in enumerate(samples):
            task_prompt = getattr(sample, "instruction", str(sample))
            episode_id = f"ep_{i:06d}"

            steps = list(self.agent_loop.run_episode(task_prompt, seed=i))

            # 收集 per-layer 激活列表
            activations: Dict[int, List[torch.Tensor]] = {}
            for step in steps:
                for layer, act in step.activations.items():
                    activations.setdefault(layer, []).append(act)

            log = self.logger.log_episode(
                episode_id=episode_id,
                task_prompt=task_prompt,
                steps=steps,
                metadata={"sample_id": getattr(sample, "sample_id", i)},
            )
            yield log, activations

    def generate(
        self,
        n_episodes: int,
        task_source: Iterable[str],
        seed: int = 300,
        save_activations: bool = True,
    ) -> List[EpisodeLog]:
        """批量生成 rollout，可选保存 per-step 激活到磁盘（H2 轨迹分析用）。

        Args:
            n_episodes:        目标 episode 数量
            task_source:       task prompt 列表（tau2-bench user_scenario 等）
            seed:              随机种子基准（每个 episode 用 seed+i）
            save_activations:  是否保存 per-step 激活（需要 agent_loop.config.layers_to_track 非空）
        """
        task_list = list(task_source)
        if not task_list:
            raise ValueError("task_source is empty")

        activations_dir = self.logger.output_dir / "activations"
        if save_activations and self.agent_loop.config.layers_to_track:
            activations_dir.mkdir(parents=True, exist_ok=True)

        logs = []
        for i in tqdm(range(n_episodes), desc="Generating rollouts"):
            task_prompt = task_list[i % len(task_list)]
            episode_id = f"ep_{i:06d}"

            steps = list(self.agent_loop.run_episode(task_prompt, seed=seed + i))

            # 收集并保存 per-step 激活
            if save_activations and self.agent_loop.config.layers_to_track:
                per_layer: Dict[int, List[torch.Tensor]] = {}
                for step in steps:
                    for layer, act in step.activations.items():
                        per_layer.setdefault(layer, []).append(act)

                for layer, act_list in per_layer.items():
                    stacked = torch.cat(act_list, dim=0)  # [n_steps, hidden]
                    out_path = activations_dir / f"{episode_id}_layer{layer}.pt"
                    torch.save(stacked, out_path)

            log = self.logger.log_episode(
                episode_id=episode_id,
                task_prompt=task_prompt,
                steps=steps,
            )
            logs.append(log)

        return logs


# ─────────────────────── CLI ───────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="H2 Rollout 生成：用 tau2-bench 任务作为 prompt 种子"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--domain", type=str, default="retail",
        choices=["retail", "airline", "telecom", "mock"],
    )
    parser.add_argument(
        "--tau2-data-dir", type=str,
        default="data/raw/tau2-bench-main/data/tau2",
        help="tau2-bench data 目录（包含 domains/ 子目录）",
    )
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="data/rollouts")
    parser.add_argument("--layers", type=int, nargs="+", default=[24, 26])
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--p-fail", type=float, default=0.0)
    parser.add_argument("--p-empty", type=float, default=0.0)
    parser.add_argument("--p-corrupt", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--no-save-activations", action="store_true",
                        help="不保存 per-step 激活到磁盘")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from controller.agent_loop import AgentConfig, AgentLoop

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, device_map=args.device, trust_remote_code=True
    )
    llm.eval()

    config = AgentConfig(
        max_steps=args.max_steps,
        p_fail=args.p_fail,
        p_empty=args.p_empty,
        p_corrupt=args.p_corrupt,
        layers_to_track=args.layers,
        device=args.device,
    )
    agent_loop = AgentLoop(llm, tokenizer, config)

    logger = RolloutLogger(args.output_dir)
    generator = RolloutGenerator(agent_loop, logger)

    print(f"Loading tau2-bench prompts: domain={args.domain} from {args.tau2_data_dir}")
    prompts = load_tau2_prompts(args.domain, args.tau2_data_dir)
    print(f"  Loaded {len(prompts)} task prompts")
    if not prompts:
        print("Error: no prompts loaded, check --tau2-data-dir", file=sys.stderr)
        sys.exit(1)

    print(f"Generating {args.n_episodes} rollouts...")
    logs = generator.generate(
        args.n_episodes,
        prompts,
        seed=args.seed,
        save_activations=not args.no_save_activations,
    )

    n_call = sum(1 for log in logs for step in log.steps if step["decision"] == "CALL")
    n_total_steps = sum(len(log.steps) for log in logs)
    print(
        f"\nDone. {len(logs)} episodes | {n_total_steps} total steps | "
        f"{n_call} CALL ({n_call / max(n_total_steps, 1) * 100:.1f}%)"
    )
    print(f"Rollouts → {Path(args.output_dir) / 'rollouts.jsonl'}")


if __name__ == "__main__":
    main()
