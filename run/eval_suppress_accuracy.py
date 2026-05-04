"""
eval_suppress_accuracy.py
──────────────────────────────────────────────────────────────────────────
用 SAE steering 测试 judge-based MCQ 准确率随 steering 强度的变化。

两种模式（--mode）：
  suppress_tc   抑制 top-N tool_call 概念（strength < 1.0 减弱，0.0 = 归零）
  promote_rfi   促进 top-N request_for_info 概念（strength > 1.0 增强）

features 从 feature_discovery JSON 自动加载，通过 --top-n 指定数量。

用法示例：
  # 抑制 TC，sweep strength
  python -m run.eval_suppress_accuracy \\
    --mode suppress_tc --top-n 25 \\
    --strengths 1.0 0.5 0.1 0.0 \\
    --model  /path/to/Qwen3.5-4B \\
    --sae-path /path/to/best.pt --layer 25

  # 促进 RFI，sweep strength
  python -m run.eval_suppress_accuracy \\
    --mode promote_rfi --top-n 25 \\
    --strengths 1.0 1.5 2.0 3.0 \\
    --model  /path/to/Qwen3.5-4B \\
    --sae-path /path/to/best.pt --layer 25
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.templates import MCQ_CHOICES, build_context_input_ids as _build_context_input_ids
from run.cache_activations import _clear_sampling_generation_config, _judge_classify

MODES = ("suppress_tc", "promote_rfi")


# ─────────────────────── feature loading ───────────────────────────

def load_top_features(feature_discovery_dir: Path, mode: str, layer: int, top_n: int) -> List[int]:
    """从 feature_discovery JSON 加载 top-N feature 索引。"""
    label = "tool_call" if mode == "suppress_tc" else "request_for_info"
    path  = feature_discovery_dir / label / f"top_features_layer{layer}.json"
    with open(path) as f:
        entries = json.load(f)
    indices = [e["feature_idx"] for e in entries[:top_n]]
    print(f"  Loaded {len(indices)} {label} features from {path.name}")
    return indices


# ─────────────────────── steering hook ─────────────────────────────

class SteerHook:
    """对每步生成的所有 token 位置应用 SAE feature steering。

    strength < 1.0：抑制（suppress）
    strength > 1.0：增强（promote）
    strength = 1.0：基线（无变化）
    """

    def __init__(self, sae, feature_indices: List[int], strength: float):
        self.sae             = sae
        self.feature_indices = feature_indices
        self.strengths       = [strength] * len(feature_indices)
        self._handle         = None

    def __call__(self, module, input, output):
        if isinstance(output, torch.Tensor):
            h, rest = output.clone(), None
        else:
            h, rest = output[0].clone(), output[1:]

        orig_device, orig_dtype = h.device, h.dtype

        if h.dim() == 3:
            batch, seq, hidden = h.shape
            flat    = h.reshape(batch * seq, hidden)
            steered = self.sae.steer_multi(flat.to(self.sae.config.device),
                                           self.feature_indices, self.strengths)
            h = steered.reshape(batch, seq, hidden).to(device=orig_device, dtype=orig_dtype)
        elif h.dim() == 2:
            steered = self.sae.steer_multi(h.to(self.sae.config.device),
                                           self.feature_indices, self.strengths)
            h = steered.to(device=orig_device, dtype=orig_dtype)

        return h if rest is None else (h,) + rest

    def register(self, layer_module):
        self._handle = layer_module.register_forward_hook(self)
        return self

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ─────────────────────── generation ────────────────────────────────

def generate_with_hook(llm, tokenizer, context_ids, device, max_new_tokens,
                       hook=None, layer_module=None) -> str:
    input_ids = context_ids.to(device)
    if hook is not None and layer_module is not None:
        hook.register(layer_module)
    try:
        with torch.no_grad():
            out_ids = llm.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        if hook is not None:
            hook.remove()
    return tokenizer.decode(out_ids[0, context_ids.shape[1]:], skip_special_tokens=True)


# ─────────────────────── perplexity ────────────────────────────────

def compute_response_perplexity(
    llm, tokenizer, context_ids: torch.Tensor, response_text: str, device: str
) -> Optional[float]:
    """用 unsteered 模型计算 response_text 的 perplexity（越高越不自然）。"""
    response_ids = tokenizer.encode(
        response_text, add_special_tokens=False, return_tensors="pt"
    ).to(device)
    if response_ids.shape[1] == 0:
        return None
    context_ids = context_ids.to(device)
    full_ids = torch.cat([context_ids, response_ids], dim=1)
    labels = full_ids.clone()
    labels[:, :context_ids.shape[1]] = -100  # 只对 response 部分算 loss
    with torch.no_grad():
        loss = llm(full_ids, labels=labels).loss
    return math.exp(loss.item())


# ─────────────────────── accuracy evaluation ───────────────────────

def evaluate_accuracy(
    llm, tokenizer, judge_model, judge_tokenizer,
    layer_module, sae, samples,
    device: str, judge_device: str,
    feature_indices: List[int],
    strength: float,
    max_new_tokens: int = 256,
    desc: str = "",
) -> Dict:
    """生成 + judge 分类 + 统计准确率 + perplexity。strength=1.0 等价于 baseline。"""
    hook = SteerHook(sae, feature_indices, strength) if strength != 1.0 else None

    labels_pred: List[str] = []
    labels_gt:   List[str] = []
    ppl_list:    List[float] = []
    judge_failed = skipped = 0

    for sample in tqdm(samples, desc=desc or "评估"):
        try:
            context_ids = _build_context_input_ids(tokenizer, sample)
            gt_label = (sample.metadata or {}).get("correct_answer")
            if gt_label not in MCQ_CHOICES:
                continue

            response = generate_with_hook(
                llm, tokenizer, context_ids, device, max_new_tokens,
                hook=hook, layer_module=layer_module,
            )
            choice, _ = _judge_classify(judge_model, judge_tokenizer, sample, response, judge_device)
            if choice is None:
                judge_failed += 1
                continue

            ppl = compute_response_perplexity(llm, tokenizer, context_ids, response, device)
            if ppl is not None:
                ppl_list.append(ppl)

            labels_pred.append(choice)
            labels_gt.append(gt_label)

        except Exception as e:
            tqdm.write(f"  Warning: skip {getattr(sample, 'sample_id', '?')}: {e!r}")
            skipped += 1

    if not labels_pred:
        return {"n_valid": 0, "accuracy": None, "per_class": {},
                "judge_failed": judge_failed, "skipped": skipped,
                "ppl_mean": None, "ppl_median": None, "ppl_std": None}

    n_correct = sum(p == g for p, g in zip(labels_pred, labels_gt))
    per_class  = {c: {"correct": 0, "total": 0} for c in MCQ_CHOICES}
    for pred, gt in zip(labels_pred, labels_gt):
        if gt in per_class:
            per_class[gt]["total"] += 1
            if pred == gt:
                per_class[gt]["correct"] += 1

    ppl_stats = {
        "ppl_mean":   round(statistics.mean(ppl_list),   2) if ppl_list else None,
        "ppl_median": round(statistics.median(ppl_list), 2) if ppl_list else None,
        "ppl_std":    round(statistics.stdev(ppl_list),  2) if len(ppl_list) > 1 else None,
    }

    return {
        "n_valid":      len(labels_pred),
        "n_correct":    n_correct,
        "accuracy":     round(n_correct / len(labels_pred), 4),
        "per_class":    {c: round(v["correct"] / v["total"], 4) if v["total"] else None
                         for c, v in per_class.items()},
        "judge_failed": judge_failed,
        "skipped":      skipped,
        **ppl_stats,
    }


# ─────────────────────── helpers ───────────────────────────────────

def _get_layer_module(llm, layer: int):
    import torch.nn as nn

    def _get_attr_by_path(obj, path: str):
        for attr in path.split("."):
            if not hasattr(obj, attr):
                return None
            obj = getattr(obj, attr)
        return obj

    paths = [
        "model.layers",
        "language_model.model.layers",
        "model.language_model.layers",
        "model.model.layers",
        "layers",
        "transformer.h",
        "model.decoder.layers",
    ]
    for path in paths:
        container = _get_attr_by_path(llm, path)
        if isinstance(container, (list, nn.ModuleList)) and len(container) > 0:
            return container[layer]
    top_attrs = [k for k, _ in llm.named_children()]
    sub_attrs  = {k: [sk for sk, _ in v.named_children()] for k, v in llm.named_children()}
    raise RuntimeError(
        f"无法定位模型层（layer={layer}）。\n"
        f"  顶层子模块   : {top_attrs}\n"
        f"  二层子模块   : {sub_attrs}\n"
        f"  请在 _get_layer_module 的 paths 列表中添加对应路径。"
    )


def _print_result(mode: str, top_n: int, strength: float, r: Dict):
    if r["accuracy"] is None:
        print(f"[{mode}] top_n={top_n:<3} strength={strength:<5} | no valid samples")
        return
    tc  = r["per_class"].get("tool_call")        or 0.0
    rfi = r["per_class"].get("request_for_info") or 0.0
    ppl = r.get("ppl_mean")
    ppl_str = f"{ppl:>8.2f}" if ppl is not None else "     n/a"
    print(f"[{mode}] top_n={top_n:<3} strength={strength:<5} | "
          f"acc={r['accuracy']:.4f}  TC={tc:.4f}  RFI={rfi:.4f} | ppl={ppl_str}")


def _save_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ─────────────────────── main ──────────────────────────────────────

def main():
    from run.cache_activations import DEFAULT_JUDGE_MODEL

    parser = argparse.ArgumentParser(description="SAE steering accuracy sweep")
    parser.add_argument("--model",    required=True)
    parser.add_argument("--sae-path", required=True)
    parser.add_argument("--layer",    type=int, required=True)
    parser.add_argument("--mode",     required=True, choices=MODES,
                        help="suppress_tc: 抑制 tool_call 概念；promote_rfi: 促进 rfi 概念")
    parser.add_argument("--top-n",    type=int, nargs="+", default=[25],
                        help="使用 top-N 个 feature，可指定多个（默认 25）")
    parser.add_argument("--strengths", type=float, nargs="+",
                        help="steering 强度列表（suppress_tc 默认 1.0 0.5 0.1 0.0；"
                             "promote_rfi 默认 1.0 1.5 2.0 3.0）")
    parser.add_argument("--data-path",   default=None)
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--output-dir",  default=None,
                        help="结果保存目录，默认 outputs/<model_name>/steer_accuracy")
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--dtype",       default="bfloat16")
    parser.add_argument("--judge-model",          default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-device",         default="cuda")
    parser.add_argument("--judge-max-new-tokens", type=int, default=256)
    parser.add_argument("--feature-discovery-dir", default=None,
                        help="feature_discovery 根目录，默认从 sae-path 推导")
    args = parser.parse_args()

    # 默认 strengths
    if args.strengths is None:
        args.strengths = [1.0, 0.5, 0.1, 0.0] if args.mode == "suppress_tc" \
                    else [1.0, 1.5, 2.0, 3.0]

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from sae.sae_model import TopKSAE
    from utils import load_samples

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)

    model_name = Path(args.model).name
    sae_root   = Path(args.sae_path).resolve().parent.parent.parent.parent

    fd_dir = Path(args.feature_discovery_dir) if args.feature_discovery_dir \
             else sae_root / "outputs" / model_name / "analysis" / "feature_discovery"

    output_dir = Path(args.output_dir) if args.output_dir \
                 else sae_root / "outputs" / model_name / "steer_accuracy"
    output_dir.mkdir(parents=True, exist_ok=True)

    def _load_model_with_fp8_dequant(model_path, device, dtype):
        """加载模型，自动将 FP8 量化 config 设为 dequantize=True 以避免缺少 FP8 kernel。"""
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        for sub in [cfg, getattr(cfg, "text_config", None)]:
            if sub is None:
                continue
            qc = getattr(sub, "quantization_config", None)
            if qc is None:
                continue
            is_fp8 = (isinstance(qc, dict) and qc.get("quant_method") == "fp8") or \
                     (not isinstance(qc, dict) and getattr(qc, "quant_method", None) == "fp8")
            if is_fp8:
                if isinstance(qc, dict):
                    qc["dequantize"] = True
                else:
                    qc.dequantize = True
                print(f"  FP8 quantization detected → dequantize=True ({type(sub).__name__})")
        load_kwargs = dict(config=cfg, torch_dtype=dtype, device_map=device, trust_remote_code=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        except ValueError:
            from transformers import AutoModelForImageTextToText
            print("  AutoModelForCausalLM failed, retrying with AutoModelForImageTextToText ...")
            model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
        return model

    # ── 加载模型 ──────────────────────────────────────────────────────
    print(f"Loading model : {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = _load_model_with_fp8_dequant(args.model, args.device, torch_dtype)
    llm.eval()
    _clear_sampling_generation_config(llm)

    print(f"Loading judge : {args.judge_model}")
    judge_tok = AutoTokenizer.from_pretrained(args.judge_model, trust_remote_code=True)
    if judge_tok.pad_token is None:
        judge_tok.pad_token = judge_tok.eos_token
    judge_model = _load_model_with_fp8_dequant(args.judge_model, args.judge_device, torch_dtype)
    judge_model.eval()
    _clear_sampling_generation_config(judge_model)

    print(f"Loading SAE   : {args.sae_path}")
    sae = TopKSAE.load(args.sae_path, device=args.device)
    sae.eval()

    layer_module = _get_layer_module(llm, args.layer)
    print(f"Hook layer    : {args.layer}  ({layer_module.__class__.__name__})")

    # ── 加载数据 ──────────────────────────────────────────────────────
    samples = load_samples("when2call", args.data_path, args.num_samples, split="test_mcq")
    samples = [s for s in samples if (s.metadata or {}).get("correct_answer") != "cannot_answer"]
    print(f"Samples       : {len(samples)}  (cannot_answer excluded)")

    print(f"\nMode      : {args.mode}")
    print(f"Top-N vals: {args.top_n}")
    print(f"Strengths : {args.strengths}")

    # ── 逐 top_n × strength 评估 ──────────────────────────────────────
    all_entries = []
    for top_n in args.top_n:
        feature_indices = load_top_features(fd_dir, args.mode, args.layer, top_n)
        print(f"\n{'='*60}\ntop_n={top_n}  features={feature_indices[:5]}{'...' if len(feature_indices) > 5 else ''}")

        results_all = {}
        for strength in args.strengths:
            desc = f"[{args.mode}] top_n={top_n} strength={strength}"
            r = evaluate_accuracy(
                llm, tokenizer, judge_model, judge_tok,
                layer_module, sae, samples,
                device=args.device, judge_device=args.judge_device,
                feature_indices=feature_indices,
                strength=strength,
                max_new_tokens=args.judge_max_new_tokens,
                desc=desc,
            )
            results_all[str(strength)] = r
            _print_result(args.mode, top_n, strength, r)

        entry = {
            "mode":            args.mode,
            "top_n":           top_n,
            "feature_indices": feature_indices,
            "layer":           args.layer,
            "model":           args.model,
            "sae_path":        args.sae_path,
            "judge_model":     args.judge_model,
            "results":         results_all,
        }
        all_entries.append(entry)

    strength_tag = "-".join(str(s) for s in args.strengths)
    out_path = output_dir / f"{args.mode}_strength{strength_tag}_sweep.json"
    _save_json(out_path, all_entries)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
