"""
Steering / Ablation - H3 因果 steering 实验

对 MCQ 二类子集（2590 样本）中的每个样本：
  1. baseline 推理，记录 CALL/NO_CALL 决策
  2. 在目标层注册 SteeringHook，放大/抑制目标 feature
  3. 再次推理，记录 steered 决策
  4. 统计翻转率，计算 Δperplexity

CLI:
  python -m analysis.steering \\
    --model Qwen/Qwen3.5-4B \\
    --sae-path outputs/sae_checkpoints/stage2/layer24/best.pt \\
    --layer 24 \\
    --feature-indices 42 17 \\
    --alphas 0.5 1.0 2.0 5.0 \\
    --output-dir outputs/analysis \\
    --num-samples 500

输出:
  outputs/analysis/steering_results.json  - 翻转率 + Δperplexity
  outputs/analysis/steering_layer{L}.png  - Fig 2：翻转率 vs α + 质量 vs α
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm


# ─────────────────────── data structures ───────────────────────────

@dataclass
class SteeringResult:
    feature_idx: int
    alpha: float
    n_samples: int
    n_call_to_nocall: int
    n_nocall_to_call: int
    flip_rate: float
    delta_perplexity: float   # (mean_ppl_steered / mean_ppl_base) - 1，正数表示质量下降


# ─────────────────────── steering hook ─────────────────────────────

class SteeringHook:
    """注册到目标层，在 action boundary 最后 token 处修改激活。

    strength > 1 → 放大 feature（增强 CALL 倾向）
    strength = 0 → 抑制 feature（减弱 CALL 倾向）
    """

    def __init__(self, sae, feature_idx: int, strength: float):
        self.sae = sae
        self.feature_idx = feature_idx
        self.strength = strength
        self._handle = None

    def __call__(self, module, input, output):
        h = output[0].clone()  # [batch, seq, hidden]
        last = h[:, -1, :].unsqueeze(0).to(self.sae.config.device)
        steered = self.sae.steer(last, self.feature_idx, self.strength)
        h[:, -1, :] = steered.squeeze(0).to(h.dtype)
        return (h,) + output[1:]

    def register(self, layer_module):
        self._handle = layer_module.register_forward_hook(self)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ─────────────────────── helpers ───────────────────────────────────

def _get_layer_module(llm, layer: int):
    if hasattr(llm, "model") and hasattr(llm.model, "layers"):
        return llm.model.layers[layer]
    if hasattr(llm, "layers"):
        return llm.layers[layer]
    raise RuntimeError("Cannot locate model layers")


def _parse_decision(text: str) -> str:
    """从生成文本中解析 CALL/NO_CALL 决策（与 when2call_adapter 保持一致）。"""
    if "<TOOLCALL>" in text.upper():
        return "CALL"
    return "NO_CALL"


def _generate(llm, tokenizer, input_ids: torch.Tensor, max_new_tokens: int) -> str:
    """运行推理，返回新生成的文本。"""
    with torch.no_grad():
        out = llm.generate(
            input_ids.to(llm.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _token_cross_entropy(llm, tokenizer, text: str) -> Optional[float]:
    """计算文本的 per-token cross-entropy（perplexity 的对数形式）。"""
    ids = tokenizer.encode(text, return_tensors="pt").to(llm.device)
    if ids.shape[1] < 2:
        return None
    with torch.no_grad():
        loss = llm(ids, labels=ids).loss
    return float(loss.item())


# ─────────────────────── experiment ────────────────────────────────

def run_steering_experiment(
    llm,
    tokenizer,
    sae,
    layer: int,
    feature_idx: int,
    alphas: List[float],
    samples,
    max_new_tokens: int = 64,
) -> List[SteeringResult]:
    """对单个 feature 的所有 alpha 运行 steering 实验。"""
    from run.cache_activations import _build_action_boundary_input_ids

    layer_module = _get_layer_module(llm, layer)
    results = []

    for alpha in alphas:
        n_call_to_nocall = 0
        n_nocall_to_call = 0
        ce_base_list: List[float] = []
        ce_steer_list: List[float] = []

        for sample in tqdm(samples, desc=f"feat={feature_idx} α={alpha:.1f}", leave=False):
            try:
                input_ids = _build_action_boundary_input_ids(tokenizer, sample)

                # Baseline
                base_text = _generate(llm, tokenizer, input_ids, max_new_tokens)
                base_decision = _parse_decision(base_text)

                # Steered（strength = 1 + alpha 放大）
                hook = SteeringHook(sae, feature_idx, 1.0 + alpha)
                hook.register(layer_module)
                try:
                    steer_text = _generate(llm, tokenizer, input_ids, max_new_tokens)
                finally:
                    hook.remove()
                steer_decision = _parse_decision(steer_text)

                if base_decision == "CALL" and steer_decision == "NO_CALL":
                    n_call_to_nocall += 1
                elif base_decision == "NO_CALL" and steer_decision == "CALL":
                    n_nocall_to_call += 1

                # Perplexity proxy（用生成文本的 cross-entropy）
                if base_text.strip() and steer_text.strip():
                    ce_base = _token_cross_entropy(llm, tokenizer, base_text)
                    ce_steer = _token_cross_entropy(llm, tokenizer, steer_text)
                    if ce_base is not None and ce_steer is not None and ce_base > 0:
                        ce_base_list.append(ce_base)
                        ce_steer_list.append(ce_steer)

            except Exception as e:
                tqdm.write(f"Warning: skip sample: {e}")

        n = len(samples)
        flip_rate = (n_call_to_nocall + n_nocall_to_call) / max(n, 1)

        if ce_base_list:
            delta_ppl = float(
                (sum(ce_steer_list) / len(ce_steer_list))
                / (sum(ce_base_list) / len(ce_base_list))
                - 1.0
            )
        else:
            delta_ppl = 0.0

        result = SteeringResult(
            feature_idx=feature_idx,
            alpha=alpha,
            n_samples=n,
            n_call_to_nocall=n_call_to_nocall,
            n_nocall_to_call=n_nocall_to_call,
            flip_rate=flip_rate,
            delta_perplexity=delta_ppl,
        )
        results.append(result)
        print(
            f"  α={alpha:.2f}: flip_rate={flip_rate:.3f} "
            f"(C→NC={n_call_to_nocall}, NC→C={n_nocall_to_call})  "
            f"Δppl={delta_ppl:+.3f}"
        )

    return results


# ─────────────────────── plotting ──────────────────────────────────

def _plot_results(all_results: List[SteeringResult], layer: int, out_path: Path):
    try:
        import matplotlib.pyplot as plt

        feat_ids = sorted(set(r.feature_idx for r in all_results))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for feat_idx in feat_ids:
            feat_res = sorted(
                [r for r in all_results if r.feature_idx == feat_idx],
                key=lambda r: r.alpha,
            )
            alphas_plt = [r.alpha for r in feat_res]
            flip_rates = [r.flip_rate for r in feat_res]
            delta_ppls = [r.delta_perplexity for r in feat_res]

            axes[0].plot(alphas_plt, flip_rates, marker="o", label=f"feat {feat_idx}")
            axes[1].plot(alphas_plt, delta_ppls, marker="o", label=f"feat {feat_idx}")

        axes[0].axhline(0.20, color="green", linestyle="--", label="target 20%")
        axes[0].set_xlabel("Steering strength α")
        axes[0].set_ylabel("Flip rate")
        axes[0].set_title(f"H3 Flip Rate vs α (layer {layer})")
        axes[0].legend()

        axes[1].axhline(0.10, color="red", linestyle="--", label="Δppl limit 10%")
        axes[1].set_xlabel("Steering strength α")
        axes[1].set_ylabel("Δ CE (relative)")
        axes[1].set_title(f"H3 Language Quality vs α (layer {layer})")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved figure → {out_path}")
    except ImportError:
        print("matplotlib not available, skipping figure")


# ─────────────────────── CLI ───────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="H3 Steering 实验：测量决策翻转率和语言质量变化")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--sae-path", type=str, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--feature-indices", type=int, nargs="+", required=True,
                        help="目标 feature 索引（来自 feature_discovery 输出）")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--output-dir", type=str, default="./outputs/analysis")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae.sae_model import TopKSAE
    from utils import load_samples

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, device_map=args.device, trust_remote_code=True
    )
    llm.eval()

    print(f"Loading SAE from {args.sae_path}")
    sae = TopKSAE.load(args.sae_path, device=args.device)
    sae.eval()

    samples = load_samples("when2call", args.data_path, args.num_samples, split="test_mcq")
    print(f"Loaded {len(samples)} MCQ samples (CALL + NO_CALL)")

    all_results: List[SteeringResult] = []
    for feat_idx in args.feature_indices:
        print(f"\nSteering feature {feat_idx}...")
        results = run_steering_experiment(
            llm, tokenizer, sae, args.layer, feat_idx,
            args.alphas, samples, args.max_new_tokens,
        )
        all_results.extend(results)

    out_json = output_dir / "steering_results.json"
    with open(out_json, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nSaved results → {out_json}")

    _plot_results(all_results, args.layer, output_dir / f"steering_layer{args.layer}.png")


if __name__ == "__main__":
    main()
