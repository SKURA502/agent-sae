"""
eval_amcs_accuracy.py
─────────────────────────────────────────────────────────────────────────
AMCS（Adaptive Margin-Calibrated Steering）准确率评估脚本。

算法：消除模型内禀 TC 偏置 β₀，使 TC 与 RFI 倾向保持中立。
  - 预计算固定 steering 向量 v = Σ w_i·n·α·δ·d_i，对所有位置直接加
  - δ = -β₀/β 是唯一来自数据的参数，其余为归一化常数
  - 无门控条件，无需在验证集上搜索语义超参数

评估：主模型生成 → judge 分类 → 与 GT 比对，输出 per-class 准确率。

用法：
  python -m analysis.eval_amcs_accuracy \\
    --model  /path/to/Qwen3.5-4B \\
    --sae-path /path/to/best.pt \\
    --layer 25 \\
    --data-path /path/to/when2call/test \\
    --judge-model /path/to/Qwen3.5-27B \\
    --output-dir ./outputs/analysis/amcs
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.templates import MCQ_CHOICES, build_context_input_ids as _build_context_input_ids
from run.cache_activations import _clear_sampling_generation_config, _judge_classify


# ─────────────────────── 特征加载 ─────────────────────────────────────

def load_amcs_features(
    feature_discovery_dir: Path,
    rfi_confusion_dir: Path,
    layer: int,
    top_n: int = 25,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """从 JSON 文件加载 AMCS 所需的特征索引与归因权重。

    feature_discovery_dir/{tool_call,request_for_info}/top_features_layer{layer}.json
        → 提供 top-N 特征索引（按 mean_diff 排序）

    rfi_confusion_dir/rfi_confusion_layer{layer}.json
        → 提供 feature_attribution.{tc,rfi}_features 的 diff 值作为权重
          若某特征在该文件中无记录，回退使用 feature_discovery 的 mean_diff

    Returns:
        tc_features:  {feature_idx: normalized_weight}  (权重和 = 1)
        rfi_features: {feature_idx: normalized_weight}
    """
    tc_disc_path  = feature_discovery_dir / "tool_call"        / f"top_features_layer{layer}.json"
    rfi_disc_path = feature_discovery_dir / "request_for_info" / f"top_features_layer{layer}.json"
    rc_path       = rfi_confusion_dir     / f"rfi_confusion_layer{layer}.json"

    with open(tc_disc_path)  as f: tc_disc  = json.load(f)[:top_n]
    with open(rfi_disc_path) as f: rfi_disc = json.load(f)[:top_n]
    with open(rc_path)       as f: rc       = json.load(f)

    fa = rc["feature_attribution"]
    tc_diff_map  = {e["feature_idx"]: abs(e["diff"])
                    for e in fa["tc_features_overactivated"]}
    rfi_diff_map = {e["feature_idx"]: abs(e["diff"])
                    for e in fa["rfi_features_underactivated"]}

    def _build(disc_list, diff_map):
        raw = {e["feature_idx"]: diff_map.get(e["feature_idx"], e["mean_diff"])
               for e in disc_list}
        total = sum(raw.values())
        return {k: round(v / total, 6) for k, v in raw.items()}

    return _build(tc_disc, tc_diff_map), _build(rfi_disc, rfi_diff_map)


# ─────────────────────── 算法配置 ─────────────────────────────────────

@dataclass
class AMCSConfig:
    """AMCS 算法参数。

    delta 是唯一来自数据的参数（= -β₀/β，来自 logistic regression）。
    tc_features / rfi_features 由 load_amcs_features() 动态加载。
    """
    delta: float = -0.533   # margin 修正量，消除内禀 TC 偏置
    alpha: float = 0.5      # TC/RFI 预算分配比例（0.5 = 对称）

    # 由 load_amcs_features() 填充（feature_idx → 归一化权重）
    tc_features:  Dict[int, float] = field(default_factory=dict)
    rfi_features: Dict[int, float] = field(default_factory=dict)


# ─────────────────────── Hook ──────────────────────────────────────────

class AMCSHook:
    """AMCS Steering Hook。

    在 __init__ 中预计算固定 steering 向量，每步生成时直接加到所有位置。

    Δh = Σ_i w_i·n_tc·α·δ·d_i  (TC 抑制, δ<0)
       + Σ_j v_j·n_rfi·(1-α)·(-δ)·d_j  (RFI 增强, -δ>0)

    n_tc/n_rfi 消掉 tc_act/rfi_act 定义中的均值 1/n，使总 margin 偏移 = δ。
    """

    def __init__(self, sae, config: AMCSConfig):
        self.config  = config
        self._handle = None

        n_tc  = len(config.tc_features)
        n_rfi = len(config.rfi_features)
        W = sae.decoder.weight.detach().float()  # [hidden, dict_size]

        vec = torch.zeros(W.shape[0])
        for feat_i, w_i in config.tc_features.items():
            vec += (w_i * n_tc * config.alpha * config.delta) * W[:, feat_i].cpu()
        for feat_j, v_j in config.rfi_features.items():
            vec += (v_j * n_rfi * (1.0 - config.alpha) * (-config.delta)) * W[:, feat_j].cpu()

        self._steering_vec = vec  # [hidden], float32, cpu

    def __call__(self, module, input, output):
        if isinstance(output, torch.Tensor):
            h, rest = output.clone(), None
        else:
            h, rest = output[0].clone(), output[1:]

        if h.dim() not in (2, 3):
            return output

        sv = self._steering_vec.to(device=h.device, dtype=h.dtype)
        h = h + sv  # broadcast over batch & seq

        return h if rest is None else (h,) + rest

    def register(self, layer_module):
        self._handle = layer_module.register_forward_hook(self)
        return self

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ─────────────────────── 生成 ─────────────────────────────────────────

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


# ─────────────────────── perplexity ──────────────────────────────────

def compute_response_perplexity(
    llm, tokenizer, context_ids: torch.Tensor, response_text: str, device: str
) -> Optional[float]:
    response_ids = tokenizer.encode(
        response_text, add_special_tokens=False, return_tensors="pt"
    ).to(device)
    if response_ids.shape[1] == 0:
        return None
    context_ids = context_ids.to(device)
    full_ids = torch.cat([context_ids, response_ids], dim=1)
    labels = full_ids.clone()
    labels[:, :context_ids.shape[1]] = -100
    with torch.no_grad():
        loss = llm(full_ids, labels=labels).loss
    return math.exp(loss.item())


# ─────────────────────── 评估 ─────────────────────────────────────────

def evaluate_accuracy(
    llm, tokenizer, judge_model, judge_tokenizer,
    layer_module, sae, samples,
    device: str, judge_device: str,
    amcs_config: Optional[AMCSConfig],
    max_new_tokens: int = 256,
    desc: str = "",
) -> Dict:
    """生成 + judge 分类 + 统计 per-class 准确率。amcs_config=None 为 baseline。"""
    hook = AMCSHook(sae, amcs_config) if amcs_config is not None else None

    labels_pred, labels_gt = [], []
    ppl_list: List[float] = []
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

    return {
        "n_valid":      len(labels_pred),
        "n_correct":    n_correct,
        "accuracy":     round(n_correct / len(labels_pred), 4),
        "per_class":    {c: round(v["correct"]/v["total"], 4) if v["total"] else None
                         for c, v in per_class.items()},
        "judge_failed": judge_failed,
        "skipped":      skipped,
        "ppl_mean":     round(statistics.mean(ppl_list),   2) if ppl_list else None,
        "ppl_median":   round(statistics.median(ppl_list), 2) if ppl_list else None,
        "ppl_std":      round(statistics.stdev(ppl_list),  2) if len(ppl_list) > 1 else None,
        "labels_pred":  labels_pred,
        "labels_gt":    labels_gt,
    }


# ─────────────────────── 工具函数 ─────────────────────────────────────

def _get_attr_by_path(obj, path: str):
    for attr in path.split("."):
        if not hasattr(obj, attr):
            return None
        obj = getattr(obj, attr)
    return obj


def _get_layer_module(llm, layer: int):
    import torch.nn as nn
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
    # 兜底：打印两层结构帮助调试
    top_attrs  = [k for k, _ in llm.named_children()]
    sub_attrs  = {k: [sk for sk, _ in v.named_children()]
                  for k, v in llm.named_children()}
    raise RuntimeError(
        f"无法定位模型层（layer={layer}）。\n"
        f"  顶层子模块   : {top_attrs}\n"
        f"  二层子模块   : {sub_attrs}\n"
        f"  请在 _get_layer_module 的 paths 列表中添加对应路径。"
    )


def _print_result(tag: str, r: Dict):
    if r["accuracy"] is None:
        print(f"  {tag}: no valid samples"); return
    print(f"  {tag}: {r['n_correct']}/{r['n_valid']} = {r['accuracy']:.4f}"
          f"  [judge_failed={r['judge_failed']}, skipped={r['skipped']}]")
    for c, a in r["per_class"].items():
        if a is not None:
            print(f"    {c:<24} {a:.4f}  ({a*100:.1f}%)")
    if r.get("ppl_mean") is not None:
        print(f"    {'ppl':<24} mean={r['ppl_mean']:.2f}  median={r['ppl_median']:.2f}  std={r.get('ppl_std')}")


def _print_comparison(baseline: Dict, amcs: Dict):
    print(f"\n{'='*58}")
    print(f"  {'Label':<24} {'Baseline':>10} {'AMCS':>10} {'Δ':>8}")
    print(f"  {'-'*54}")
    for c in MCQ_CHOICES:
        b, a = baseline["per_class"].get(c), amcs["per_class"].get(c)
        if b is not None and a is not None:
            d = a - b
            print(f"  {c:<24} {b:>10.4f} {a:>10.4f} {d:>+8.4f}")
    b_acc, a_acc = baseline["accuracy"], amcs["accuracy"]
    if b_acc is not None and a_acc is not None:
        print(f"  {'Overall':.<24} {b_acc:>10.4f} {a_acc:>10.4f} {a_acc-b_acc:>+8.4f}")
    print(f"{'='*58}")


# ─────────────────────── CLI ──────────────────────────────────────────

def main():
    from run.cache_activations import DEFAULT_JUDGE_MODEL

    parser = argparse.ArgumentParser(description="AMCS 偏置修正评估")
    parser.add_argument("--model",       required=True)
    parser.add_argument("--sae-path",    required=True)
    parser.add_argument("--layer",       type=int, required=True)
    parser.add_argument("--data-path",   default=None)
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--output-dir",  default="./outputs/analysis/amcs")
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--dtype",       default="bfloat16")
    parser.add_argument("--judge-model",          default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-device",         default="cuda")
    parser.add_argument("--judge-max-new-tokens", type=int, default=256)
    parser.add_argument("--no-baseline", action="store_true")
    # 算法参数
    parser.add_argument("--delta", type=float, default=-0.533,
                        help="margin 修正量（= -β₀/β，默认 -0.533）")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="TC/RFI 预算分配比例（默认 0.5）")
    # 特征来源（动态加载）
    parser.add_argument("--feature-discovery-dir", default=None,
                        help="feature_discovery 根目录（含 tool_call/ 和 request_for_info/ 子目录）")
    parser.add_argument("--rfi-confusion-dir", default=None,
                        help="rfi_confusion 目录（含 rfi_confusion_layer{layer}.json）")
    parser.add_argument("--top-n-features", type=int, default=25,
                        help="每侧使用 top-N 特征（默认 25）")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae.sae_model import TopKSAE
    from utils import load_samples

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model : {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, device_map=args.device, trust_remote_code=True)
    llm.eval()
    _clear_sampling_generation_config(llm)

    print(f"Loading judge : {args.judge_model}")
    judge_tok = AutoTokenizer.from_pretrained(args.judge_model, trust_remote_code=True)
    if judge_tok.pad_token is None:
        judge_tok.pad_token = judge_tok.eos_token
    judge_model = AutoModelForCausalLM.from_pretrained(
        args.judge_model, torch_dtype=torch_dtype,
        device_map=args.judge_device, trust_remote_code=True)
    judge_model.eval()
    _clear_sampling_generation_config(judge_model)

    print(f"Loading SAE   : {args.sae_path}")
    sae = TopKSAE.load(args.sae_path, device=args.device)
    sae.eval()

    layer_module = _get_layer_module(llm, args.layer)
    print(f"Hook layer    : {args.layer}  ({layer_module.__class__.__name__})")

    samples = load_samples("when2call", args.data_path, args.num_samples, split="test_mcq")
    print(f"Samples       : {len(samples)}")

    # ── 加载特征 ──────────────────────────────────────────────────────
    model_name = Path(args.model).name
    _sae_base  = Path(args.sae_path).resolve().parent.parent.parent.parent
    _analysis  = _sae_base / "outputs" / model_name / "analysis"

    fd_dir = Path(args.feature_discovery_dir) if args.feature_discovery_dir else _analysis / "feature_discovery"
    rc_dir = Path(args.rfi_confusion_dir)      if args.rfi_confusion_dir      else _analysis / "rfi_confusion"

    print(f"Feature dir   : {fd_dir}")
    print(f"RFI confusion : {rc_dir}")
    tc_feats, rfi_feats = load_amcs_features(fd_dir, rc_dir, args.layer, args.top_n_features)
    print(f"TC  features  : {len(tc_feats)} (top-{args.top_n_features})")
    print(f"RFI features  : {len(rfi_feats)} (top-{args.top_n_features})")

    amcs_config = AMCSConfig(
        delta=args.delta, alpha=args.alpha,
        tc_features=tc_feats, rfi_features=rfi_feats,
    )
    print(f"\nAMCS: δ={amcs_config.delta}  α={amcs_config.alpha}"
          f"  top_n={args.top_n_features}")

    results_all = {}

    if not args.no_baseline:
        print("\n" + "="*58 + "\nBaseline")
        r = evaluate_accuracy(
            llm, tokenizer, judge_model, judge_tok, layer_module, sae, samples,
            device=args.device, judge_device=args.judge_device,
            amcs_config=None, max_new_tokens=args.judge_max_new_tokens, desc="baseline",
        )
        results_all["baseline"] = r
        _print_result("baseline", r)

    print("\n" + "="*58 + "\nAMCS")
    r = evaluate_accuracy(
        llm, tokenizer, judge_model, judge_tok, layer_module, sae, samples,
        device=args.device, judge_device=args.judge_device,
        amcs_config=amcs_config, max_new_tokens=args.judge_max_new_tokens, desc="AMCS",
    )
    results_all["amcs"] = r
    _print_result("AMCS", r)

    if "baseline" in results_all:
        _print_comparison(results_all["baseline"], results_all["amcs"])

    out_path = output_dir / "amcs_accuracy_results.json"
    existing = _load_json_list(out_path)
    existing.append({
        "model": args.model, "sae_path": args.sae_path,
        "layer": args.layer, "judge_model": args.judge_model,
        "n_samples": len(samples),
        "amcs_config": {
            "delta": amcs_config.delta, "alpha": amcs_config.alpha,
        },
        "results": {k: _strip_labels(v) for k, v in results_all.items()},
    })
    _save_json(out_path, existing)
    print(f"\nSaved → {out_path}  ({len(existing)} 条记录)")


def _strip_labels(r: Dict) -> Dict:
    return {k: v for k, v in r.items() if k not in ("labels_pred", "labels_gt")}


def _save_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _load_json_list(path: Path) -> list:
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    except Exception:
        return []


if __name__ == "__main__":
    main()
