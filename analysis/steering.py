"""
Steering / Ablation - H3 因果 steering 实验

两种实验模式：
  amplify: 放大目标 feature（strength = 1 + alpha），增强 tool_call 倾向
  ablate:  将目标 feature 持续置 0（strength = 0），抑制 tool_call 倾向

决策评估：log-prob 对四个 MCQ 选项打分（与 cache_activations.py 完全一致），
          覆盖全部 test_mcq 非 UNCERTAIN 样本（tool_call / direct / request_for_info / cannot_answer）。

注意：log-prob scoring 时 hook 的 position = context_len - 1（action boundary），
      自回归 generate 时 position = -1（每步最后一个 token）。

Case study：保存前 N 个样本的完整记录（prompt / 生成文本 / 四选项 log-prob，steer 前后对比）。

CLI:
  # amplify 实验（多个 alpha）
  python -m analysis.steering \\
    --model /path/to/model --sae-path .../best.pt \\
    --layer 25 --feature-indices 42 17 \\
    --mode amplify --alphas 0.5 1.0 2.0 5.0 \\
    --data-path $SOURCE_ROOT/dataset/when2call/test \\
    --output-dir outputs/analysis/steering

  # ablate 实验
  python -m analysis.steering \\
    --mode ablate ...

  # 同时运行两种实验
  python -m analysis.steering \\
    --mode both ...

输出:
  $output_dir/steering_results.json          - 全部实验翻转率 + 准确率
  $output_dir/case_studies_feat{X}.json      - 每个 feature 的 case study
  $output_dir/steering_layer{L}.png          - 翻转率 / 准确率图
"""

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm


# ─────────────────────── data structures ───────────────────────────

@dataclass
class CaseStudy:
    """单个样本在一次 steering 实验中的完整记录。"""
    sample_id: str
    correct_answer: str          # 数据集标注的正确答案（tool_call / direct / ...）
    feature_idx: int
    mode: str                    # "amplify" | "ablate"
    strength: float              # 实际使用的 strength 值
    prompt_text: str             # decode 后的 context（可能含 special tokens）
    # ── baseline（无 hook）──
    base_pred: str               # log-prob 打分最高的选项
    base_scores: Dict[str, float]  # 四选项 normalized log-prob
    base_response: str           # 自回归生成的文本
    # ── steered（有 hook）──
    steer_pred: str
    steer_scores: Dict[str, float]
    steer_response: str


@dataclass
class SteeringResult:
    """单次实验（feature × mode × strength）的汇总统计。"""
    feature_idx: int
    mode: str               # "amplify" | "ablate"
    strength: float         # 实际 strength（amplify: 1+alpha；ablate: 0）
    alpha: float            # alpha 参数（ablate 为 0）
    n_samples: int
    # 翻转统计
    n_to_tool_call: int     # no_call → call
    n_from_tool_call: int   # call → no_call
    # 目标方向翻转率：
    #   amplify → n_to_tool_call / n  （应 call 但没 call → call）
    #   ablate  → n_from_tool_call / n（不应 call 但 call 了 → no_call）
    flip_rate: float
    targeted_flip_rate: float
    base_accuracy: float = 0.0
    steered_accuracy: float = 0.0


# ─────────────────────── steering hook ─────────────────────────────

class SteeringHook:
    """注册到目标层，在指定 position 处修改激活。

    position=-1  : 最后一个 token（自回归 generate 时用）
    position=k   : 绝对位置 k（log-prob scoring 时设为 context_len - 1）

    strength > 1 → 放大 feature（amplify）
    strength = 0 → 完全抑制 feature（ablate）

    支持单特征（feature_idx: int）或多特征联合（feature_idx: List[int]，strength: List[float]）。
    多特征时在单次 encode-decode 内同时修改，避免多次重建误差累积。
    """

    def __init__(self, sae, feature_idx, strength, position: int = -1,
                 all_positions: bool = False):
        self.sae = sae
        # 统一存为列表，方便统一处理
        if isinstance(feature_idx, int):
            self.feature_indices = [feature_idx]
            self.strengths = [strength]
        else:
            self.feature_indices = list(feature_idx)
            self.strengths = [strength] * len(self.feature_indices) if isinstance(strength, float) else list(strength)
        self.position = position
        self.all_positions = all_positions  # True → 对全序列所有 token 都 steer
        self._handle = None

    def __call__(self, module, input, output):
        # output 可能是 tuple（大多数 transformer block）或直接是 tensor（部分 Qwen 层）
        if isinstance(output, torch.Tensor):
            h = output.clone()
            rest = None
        else:
            h = output[0].clone()
            rest = output[1:]

        # h 可能是 [batch, seq, hidden]（3D）或 [seq, hidden]（2D，无 batch 维）
        if h.dim() == 3:
            if self.all_positions:
                batch, seq, hidden = h.shape
                # 展平为 [batch*seq, hidden]，统一过 SAE，再还原
                target = h.reshape(batch * seq, hidden)
                steered = self.sae.steer_multi(
                    target.to(self.sae.config.device), self.feature_indices, self.strengths
                )
                h = steered.reshape(batch, seq, hidden).to(h.dtype)
            else:
                target = h[:, self.position, :]          # [batch, hidden]
                steered = self.sae.steer_multi(
                    target.to(self.sae.config.device), self.feature_indices, self.strengths
                )
                h[:, self.position, :] = steered.to(h.dtype)
        elif h.dim() == 2:
            if self.all_positions:
                seq, hidden = h.shape
                steered = self.sae.steer_multi(
                    h.to(self.sae.config.device), self.feature_indices, self.strengths
                )
                h = steered.to(h.dtype)
            else:
                target = h[self.position, :].unsqueeze(0)  # [1, hidden]
                steered = self.sae.steer_multi(
                    target.to(self.sae.config.device), self.feature_indices, self.strengths
                )
                h[self.position, :] = steered.squeeze(0).to(h.dtype)
        else:
            # 未预期的维度，直接跳过 steering
            pass

        if rest is None:
            return h
        return (h,) + rest

    def register(self, layer_module):
        self._handle = layer_module.register_forward_hook(self)
        return self

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


def _generate(llm, tokenizer, input_ids: torch.Tensor, max_new_tokens: int,
              hook: Optional[SteeringHook] = None,
              layer_module=None) -> str:
    """自回归生成，可选地附加 position=-1 的 steering hook。"""
    if hook is not None and layer_module is not None:
        hook.position = -1
        hook.register(layer_module)
    try:
        with torch.no_grad():
            out = llm.generate(
                input_ids.to(llm.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        if hook is not None:
            hook.remove()
    new_tokens = out[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _parse_toolcall(text: str) -> str:
    """根据生成文本判断决策：含 <tool_call> 标签 → 'call'，否则 → 'no_call'。"""
    import re
    return "call" if re.search(r"<tool_call\b", text, re.IGNORECASE) else "no_call"


def _score_choices_steered(
    llm,
    tokenizer,
    context_ids: torch.Tensor,
    answer_texts: Dict[str, str],
    device: str,
    hook: Optional[SteeringHook] = None,
    layer_module=None,
) -> Dict[str, float]:
    """对四个 MCQ 选项计算 log-prob（仅用于 case study 补充分析）。

    hook 注册在 action boundary 位置（context_len - 1）。
    返回 {choice: score} 字典，不返回 pred（pred 由生成文本决定）。
    """
    from run.cache_activations import _score_choices

    if hook is not None and layer_module is not None:
        hook.position = context_ids.shape[1] - 1
        hook.register(layer_module)
    try:
        _, scores = _score_choices(llm, tokenizer, context_ids, answer_texts, device)
    finally:
        if hook is not None:
            hook.remove()
    return scores


# ─────────────────────── experiment ────────────────────────────────

def run_experiment(
    llm,
    tokenizer,
    sae,
    layer: int,
    feature_idx,              # int（单特征）或 List[int]（联合 steering）
    mode: str,                # "amplify" | "ablate"
    alphas: List[float],      # 仅 amplify 有效
    samples,
    device: str,
    max_new_tokens: int = 512,
    all_positions: bool = False,  # True → 对全序列所有 token 都 steer
) -> Tuple[List[SteeringResult], List[CaseStudy]]:
    """对单个或多个 feature（联合）运行一种实验模式（amplify 或 ablate）。

    feature_idx 可以是 int（单特征）或 List[int]（联合 steering，所有特征用同一 strength）。

    样本已按缓存 label 筛选：
      amplify 样本：模型原本不 call → base_decision 固定为 "no_call"
      ablate  样本：模型原本 call   → base_decision 固定为 "call"

    Returns:
        results:      每个 (mode, strength) 的汇总统计
        case_studies: 前 num_case_studies 个样本的 case study
    """
    from utils.templates import build_context_input_ids as _build_context_input_ids
    from run.cache_activations import (
        _get_answer_texts,
    )

    # 统一为列表，用于打印和 hook 构建
    feature_indices = [feature_idx] if isinstance(feature_idx, int) else list(feature_idx)
    # SteeringResult 里的 feature_idx 字段用逗号拼接字符串表示联合特征
    feat_label = feature_indices[0] if len(feature_indices) == 1 else feature_indices

    layer_module = _get_layer_module(llm, layer)
    all_results: List[SteeringResult] = []
    case_studies: List[CaseStudy] = []

    # 确定 (alpha, strength) 列表
    if mode == "ablate":
        experiments = [(0.0, 0.0)]
    else:
        experiments = [(a, 1.0 + a) for a in sorted(alphas)]

    feat_desc = str(feature_indices[0]) if len(feature_indices) == 1 else f"joint{len(feature_indices)}"

    for alpha, strength in experiments:
        n_to_tool_call = 0
        n_from_tool_call = 0
        n_valid = 0  # 成功处理的样本数（异常跳过不计）
        n_base_correct = 0
        n_steer_correct = 0

        for i, sample in enumerate(tqdm(
            samples,
            desc=f"feat={feat_desc} {mode} str={strength:.1f}",
            leave=False,
        )):
            correct_answer = str(
                (sample.metadata or {}).get("correct_answer") or sample.label.value
            )
            # correct_answer 是否为 tool_call（用于准确率统计）
            correct_is_call = (correct_answer == "tool_call")

            try:
                context_ids = _build_context_input_ids(tokenizer, sample)

                # base_decision 由缓存 label 决定，无需重新生成
                # amplify 样本：原本不 call；ablate 样本：原本 call
                base_decision = "no_call" if mode == "amplify" else "call"
                n_valid += 1

                # ── Steered：生成文本（hook 在每步最后一个 token，或全序列）──
                steer_hook = SteeringHook(sae, feature_indices, strength,
                                          all_positions=all_positions)
                steer_resp = _generate(
                    llm, tokenizer, context_ids, max_new_tokens,
                    hook=steer_hook, layer_module=layer_module,
                )
                steer_decision = _parse_toolcall(steer_resp)

                # 翻转统计
                flipped = base_decision != steer_decision
                if base_decision == "no_call" and steer_decision == "call":
                    n_to_tool_call += 1
                elif base_decision == "call" and steer_decision == "no_call":
                    n_from_tool_call += 1

                # 准确率统计
                base_pred_is_call = (base_decision == "call")
                steer_pred_is_call = (steer_decision == "call")
                if base_pred_is_call == correct_is_call:
                    n_base_correct += 1
                if steer_pred_is_call == correct_is_call:
                    n_steer_correct += 1

                # ── Case study：翻转时生成 base_resp 供对比查看 ──
                if flipped:
                    base_resp = _generate(
                        llm, tokenizer, context_ids, max_new_tokens,
                        hook=None, layer_module=None,
                    )
                    prompt_text = tokenizer.decode(
                        context_ids[0], skip_special_tokens=False
                    )
                    answer_texts = _get_answer_texts(sample)
                    # log-prob：baseline（无 hook，action boundary 位置）
                    base_scores = _score_choices_steered(
                        llm, tokenizer, context_ids, answer_texts, device,
                        hook=None, layer_module=None,
                    )
                    # log-prob：steered（hook 在 context_len-1，或全序列）
                    lp_hook = SteeringHook(sae, feature_indices, strength,
                                           all_positions=all_positions)
                    steer_scores = _score_choices_steered(
                        llm, tokenizer, context_ids, answer_texts, device,
                        hook=lp_hook, layer_module=layer_module,
                    )
                    case_studies.append(CaseStudy(
                        sample_id=sample.sample_id,
                        correct_answer=correct_answer,
                        feature_idx=feat_label,
                        mode=mode,
                        strength=strength,
                        prompt_text=prompt_text,
                        base_pred=base_decision,
                        base_scores={k: float(v) for k, v in base_scores.items()},
                        base_response=base_resp,
                        steer_pred=steer_decision,
                        steer_scores={k: float(v) for k, v in steer_scores.items()},
                        steer_response=steer_resp,
                    ))

            except Exception as e:
                tqdm.write(
                    f"Warning: skip sample {getattr(sample, 'sample_id', i)}: {e}"
                )

        if mode == "amplify":
            targeted_flip_rate = n_to_tool_call / max(n_valid, 1)
        else:
            targeted_flip_rate = n_from_tool_call / max(n_valid, 1)

        result = SteeringResult(
            feature_idx=feat_label,
            mode=mode,
            strength=strength,
            alpha=alpha,
            n_samples=n_valid,
            n_to_tool_call=n_to_tool_call,
            n_from_tool_call=n_from_tool_call,
            flip_rate=targeted_flip_rate,
            targeted_flip_rate=targeted_flip_rate,
            base_accuracy=n_base_correct / max(n_valid, 1),
            steered_accuracy=n_steer_correct / max(n_valid, 1),
        )
        all_results.append(result)
        print(
            f"  {mode} str={strength:.2f} (n={n_valid}/{len(samples)}): "
            f"flip_rate={targeted_flip_rate:.3f}  "
            f"(→call={n_to_tool_call}, call→={n_from_tool_call})"
        )

    return all_results, case_studies


# ─────────────────────── plotting ──────────────────────────────────

def _plot_results(all_results: List[SteeringResult], layer: int, out_path: Path):
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        amplify = [r for r in all_results if r.mode == "amplify"]
        ablate  = [r for r in all_results if r.mode == "ablate"]
        def _hashable(fid):
            return tuple(fid) if isinstance(fid, list) else fid
        feat_ids = sorted(set(_hashable(r.feature_idx) for r in all_results),
                          key=lambda x: (str(x),))

        n_plots = 2 + (1 if ablate else 0)
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        def _match(r, fid):
            return _hashable(r.feature_idx) == fid

        # ── amplify: flip rate vs alpha ──
        ax = axes[0]
        for fid in feat_ids:
            rows = sorted(
                [r for r in amplify if _match(r, fid)],
                key=lambda r: r.alpha,
            )
            if rows:
                ax.plot([r.alpha for r in rows], [r.flip_rate for r in rows],
                        marker="o", label=f"feat {fid}")
        ax.axhline(0.20, color="green", linestyle="--", alpha=0.6, label="20% ref")
        ax.set_xlabel("α (strength = 1+α)")
        ax.set_ylabel("Flip rate")
        ax.set_title(f"Amplify: Flip Rate vs α  (layer {layer})")
        ax.legend(fontsize=8)

        # ── amplify: accuracy vs alpha ──
        ax = axes[1]
        for fid in feat_ids:
            rows = sorted(
                [r for r in amplify if _match(r, fid)],
                key=lambda r: r.alpha,
            )
            if rows:
                ax.plot([r.alpha for r in rows], [r.base_accuracy for r in rows],
                        linestyle="--", marker="s", alpha=0.5, label=f"feat {fid} base")
                ax.plot([r.alpha for r in rows], [r.steered_accuracy for r in rows],
                        linestyle="-", marker="o", label=f"feat {fid} steered")
        ax.set_xlabel("α")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Amplify: Accuracy vs α  (layer {layer})")
        ax.legend(fontsize=8)

        # ── ablate: bar chart（flip rate + accuracy change per feature）──
        if ablate and len(axes) > 2:
            ax = axes[2]
            x = np.arange(len(feat_ids))
            width = 0.3
            flip_rates = [
                next((r.flip_rate for r in ablate if _match(r, fid)), 0)
                for fid in feat_ids
            ]
            acc_deltas = [
                next(
                    (r.steered_accuracy - r.base_accuracy
                     for r in ablate if _match(r, fid)), 0
                )
                for fid in feat_ids
            ]
            ax.bar(x - width / 2, flip_rates, width, label="flip rate", color="#4C8BE0")
            ax.bar(x + width / 2, acc_deltas, width, label="Δacc (steered−base)",
                   color=["#E07B39" if d >= 0 else "#c0392b" for d in acc_deltas])
            ax.set_xticks(x)
            def _fid_label(fid):
                return "joint:" + ",".join(str(i) for i in fid) if isinstance(fid, tuple) else f"feat {fid}"
            ax.set_xticklabels([_fid_label(fid) for fid in feat_ids], rotation=15)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(f"Ablate: Flip Rate & Δacc  (layer {layer})")
            ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved figure → {out_path}")
    except ImportError:
        print("matplotlib not available, skipping figure")


# ─────────────────────── CLI ───────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="H3 Steering 实验：amplify / ablate feature，测量 MCQ 决策翻转率和准确率变化"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--sae-path", type=str, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--feature-indices", type=int, nargs="+", required=True,
                        help="目标 feature 索引（来自 feature_discovery 输出）")
    parser.add_argument("--joint", action="store_true",
                        help="将所有 --feature-indices 作为一组联合 steering（单次 encode-decode 内同时修改），"
                             "否则（默认）每个 feature 独立实验")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["amplify", "ablate", "both"],
                        help="实验模式：amplify / ablate / both")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 5.0],
                        help="amplify 实验的 alpha 列表（strength = 1 + alpha）")
    parser.add_argument("--output-dir", type=str, default="./outputs/analysis/steering")
    parser.add_argument("--data-path", type=str, default=None,
                        help="When2Call test 目录（含 when2call_test_mcq.jsonl）")
    parser.add_argument("--num-samples", type=int, default=-1,
                        help="样本数（-1 = 全量，包含所有4类非 UNCERTAIN 样本）")
    parser.add_argument("--num-case-studies", type=int, default=20,
                        help="每个 feature 保存的 case study 数量")
    parser.add_argument("--case-study-alpha", type=float, default=2.0,
                        help="case study 采集时使用的 alpha（amplify 模式）")
    parser.add_argument("--max-new-tokens", type=int, default=10240,
                        help="case study 生成文本的最大 token 数")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--all-positions", action="store_true",
                        help="对全序列所有 token 位置都做 steer（默认只 steer 最后一个 token）")
    parser.add_argument("--activation-path", type=str, default=None,
                        help="预缓存的激活 .pt 文件路径（含 labels / gt_labels），"
                             "用于筛选 amplify/ablate 目标样本。"
                             "amplify 目标：gt=tool_call 且 model 预测非 tool_call；"
                             "ablate 目标：gt≠tool_call 且 model 预测 tool_call。")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae.sae_model import TopKSAE
    from utils import load_samples

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=args.device,
        trust_remote_code=True,
    )
    llm.eval()

    print(f"Loading SAE from {args.sae_path}")
    sae = TopKSAE.load(args.sae_path, device=args.device)
    sae.eval()

    # 加载全部4类非 UNCERTAIN 样本（tool_call / direct / request_for_info / cannot_answer）
    samples = load_samples("when2call", args.data_path, args.num_samples, split="test_mcq")
    from utils.when2call_adapter import DecisionLabel
    label_dist = {
        lbl.value: sum(1 for s in samples if s.label == lbl)
        for lbl in DecisionLabel if lbl != DecisionLabel.UNCERTAIN
    }
    print(f"Loaded {len(samples)} samples: {label_dist}")

    # ── 按激活缓存中的 labels / gt_labels 筛选目标样本 ──────────────────
    # MCQ_CHOICES = ("direct_answer", "tool_call", "request_for_info", "cannot_answer")
    # index 1 = tool_call
    _TOOL_CALL_IDX = 1
    amplify_samples = samples   # 默认：用全部样本
    ablate_samples  = samples

    if args.activation_path:
        act_data = torch.load(args.activation_path, map_location="cpu", weights_only=False)
        pred_labels = act_data["labels"]   # 模型 MCQ 预测（0-3）
        gt_labels   = act_data["gt_labels"]  # 数据集标注（0-3）

        if len(pred_labels) != len(samples):
            print(
                f"WARNING: activation file has {len(pred_labels)} entries "
                f"but {len(samples)} samples loaded — skipping sample filtering."
            )
        else:
            # amplify：应该 call 但模型没有 call
            amp_mask = (gt_labels == _TOOL_CALL_IDX) & (pred_labels != _TOOL_CALL_IDX)
            amp_indices = amp_mask.nonzero(as_tuple=True)[0].tolist()
            amplify_samples = [samples[i] for i in amp_indices]

            # ablate：不应该 call 但模型 call 了
            abl_mask = (gt_labels != _TOOL_CALL_IDX) & (pred_labels == _TOOL_CALL_IDX)
            abl_indices = abl_mask.nonzero(as_tuple=True)[0].tolist()
            ablate_samples = [samples[i] for i in abl_indices]

            print(
                f"Activation-based sample filtering:\n"
                f"  amplify targets (gt=tool_call, pred≠tool_call): {len(amplify_samples)}\n"
                f"  ablate  targets (gt≠tool_call, pred=tool_call):  {len(ablate_samples)}"
            )

    modes = (
        ["amplify", "ablate"] if args.mode == "both"
        else [args.mode]
    )

    all_results: List[SteeringResult] = []
    all_case_studies: List[CaseStudy] = []

    def _save_case_studies_by_alpha(case_studies: List[CaseStudy], prefix: str):
        """将 case study 列表按 (mode, strength) 分组，每组单独写一个文件。"""
        from collections import defaultdict
        groups: Dict[tuple, List[CaseStudy]] = defaultdict(list)
        for cs in case_studies:
            groups[(cs.mode, cs.strength)].append(cs)
        for (mode, strength), cs_list in sorted(groups.items()):
            alpha = strength - 1.0 if mode == "amplify" else 0.0
            alpha_tag = f"alpha{alpha:.2f}".rstrip("0").rstrip(".")
            cs_path = output_dir / f"{prefix}_{mode}_{alpha_tag}.json"
            with open(cs_path, "w", encoding="utf-8") as f:
                json.dump([asdict(c) for c in cs_list], f,
                          ensure_ascii=False, indent=2)
            print(f"Saved {len(cs_list)} case studies → {cs_path}")

    def _samples_for_mode(mode: str):
        """根据模式返回对应的目标样本集合。"""
        if mode == "amplify":
            return amplify_samples
        else:  # ablate
            return ablate_samples

    if args.joint:
        # ── 联合模式：所有 feature 在同一次 encode-decode 里一起 steer ──
        feat_key = "_".join(str(i) for i in args.feature_indices)
        print(f"\n=== Joint Steering: features {args.feature_indices} ===")
        joint_case_studies: List[CaseStudy] = []
        for mode in modes:
            mode_samples = _samples_for_mode(mode)
            print(f"--- {mode} ({len(mode_samples)} samples) ---")
            results, cs = run_experiment(
                llm, tokenizer, sae,
                layer=args.layer,
                feature_idx=args.feature_indices,   # 传入列表
                mode=mode,
                alphas=args.alphas,
                samples=mode_samples,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                all_positions=args.all_positions,
            )
            all_results.extend(results)
            joint_case_studies.extend(cs)
        if joint_case_studies:
            _save_case_studies_by_alpha(joint_case_studies, f"case_studies_joint_{feat_key}")
        all_case_studies.extend(joint_case_studies)
    else:
        # ── 默认模式：每个 feature 独立实验 ──
        for feat_idx in args.feature_indices:
            feat_case_studies: List[CaseStudy] = []
            print(f"\n=== Feature {feat_idx} ===")
            for mode in modes:
                mode_samples = _samples_for_mode(mode)
                print(f"--- {mode} ({len(mode_samples)} samples) ---")
                results, cs = run_experiment(
                    llm, tokenizer, sae,
                    layer=args.layer,
                    feature_idx=feat_idx,
                    mode=mode,
                    alphas=args.alphas,
                    samples=mode_samples,
                    device=args.device,
                    max_new_tokens=args.max_new_tokens,
                    all_positions=args.all_positions,
                )
                all_results.extend(results)
                feat_case_studies.extend(cs)

            if feat_case_studies:
                _save_case_studies_by_alpha(feat_case_studies, f"case_studies_feat{feat_idx}")
            all_case_studies.extend(feat_case_studies)

    # 汇总结果
    out_json = output_dir / "steering_results.json"
    with open(out_json, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nSaved steering results → {out_json}")

    _plot_results(all_results, args.layer, output_dir / f"steering_layer{args.layer}.png")


if __name__ == "__main__":
    main()
