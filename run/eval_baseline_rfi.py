"""
eval_baseline_rfi.py
──────────────────────────────────────────────────────────────────────────
Prompt-hint baseline：在用户 prompt 末尾追加提示语，鼓励模型在适当时
请求更多信息（request_for_info），以此作为 SAE steering 的对照 baseline。

用法示例：
  python -m run.eval_baseline_rfi \\
    --model /path/to/Qwen3.5-4B \\
    --data-path /path/to/when2call/test \\
    --output-dir /path/to/outputs/baseline_rfi
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.templates import MCQ_CHOICES, build_context_input_ids as _build_base_input_ids
from run.cache_activations import _clear_sampling_generation_config, _judge_classify, _load_model

# ─────────────────────── hint 文本 ─────────────────────────────────

HINT = (
    "\nNote: When the user's request lacks necessary details, "
    "ask before taking action."
)


# ─────────────────────── prompt builder ────────────────────────────

def build_context_input_ids_with_hint(
    tokenizer,
    sample,
    max_length: int = 2048,
) -> torch.Tensor:
    original_instruction = sample.instruction
    try:
        sample.instruction = (sample.instruction or "") + HINT
        return _build_base_input_ids(tokenizer, sample, max_length=max_length)
    finally:
        sample.instruction = original_instruction


# ─────────────────────── perplexity of generated response ──────────

def compute_response_perplexity(
    llm, tokenizer, context_ids: torch.Tensor, response_text: str, device: str
) -> Optional[float]:
    """Return exp(mean NLL per token) for response_text given context_ids."""
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)
    if not response_ids:
        return None
    resp_tensor = torch.tensor([response_ids], dtype=torch.long)
    full_ids = torch.cat([context_ids, resp_tensor], dim=1).to(device)
    context_len = context_ids.shape[1]
    n_resp = len(response_ids)

    with torch.no_grad():
        logits = llm(full_ids).logits  # [1, seq_len, vocab_size]

    # logits[i] predicts token[i+1]; response tokens start at context_len
    pred_logits = logits[0, context_len - 1 : context_len - 1 + n_resp, :]
    log_probs = F.log_softmax(pred_logits, dim=-1)
    resp_token_ids = full_ids[0, context_len:]
    token_log_probs = log_probs[range(n_resp), resp_token_ids]
    return math.exp(-token_log_probs.mean().item())


# ─────────────────────── generation ────────────────────────────────

def generate_response(llm, tokenizer, context_ids, device, max_new_tokens) -> str:
    input_ids = context_ids.to(device)
    with torch.no_grad():
        out_ids = llm.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out_ids[0, context_ids.shape[1]:], skip_special_tokens=True)


# ─────────────────────── accuracy evaluation ───────────────────────

def evaluate_accuracy(
    llm, tokenizer, judge_model, judge_tokenizer,
    samples,
    device: str, judge_device: str,
    max_new_tokens: int = 256,
) -> Dict:
    """生成 + judge 分类 + 统计准确率，同时计算生成回复的 perplexity。"""
    labels_pred: List[str] = []
    labels_gt:   List[str] = []
    ppl_values:  List[float] = []
    judge_failed = skipped = 0

    for sample in tqdm(samples):
        try:
            context_ids = build_context_input_ids_with_hint(tokenizer, sample)
            gt_label = (sample.metadata or {}).get("correct_answer")
            if gt_label not in MCQ_CHOICES:
                continue

            response = generate_response(llm, tokenizer, context_ids, device, max_new_tokens)

            ppl = compute_response_perplexity(llm, tokenizer, context_ids, response, device)
            if ppl is not None:
                ppl_values.append(ppl)

            choice, _ = _judge_classify(judge_model, judge_tokenizer, sample, response, judge_device)
            if choice is None:
                judge_failed += 1
                continue

            labels_pred.append(choice)
            labels_gt.append(gt_label)

        except Exception as e:
            tqdm.write(f"  Warning: skip {getattr(sample, 'sample_id', '?')}: {e!r}")
            skipped += 1

    if not labels_pred:
        return {"n_valid": 0, "accuracy": None, "per_class": {},
                "mean_ppl": None, "judge_failed": judge_failed, "skipped": skipped}

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
        "per_class":    {c: round(v["correct"] / v["total"], 4) if v["total"] else None
                         for c, v in per_class.items()},
        "mean_ppl":     round(sum(ppl_values) / len(ppl_values), 4) if ppl_values else None,
        "judge_failed": judge_failed,
        "skipped":      skipped,
    }


# ─────────────────────── helpers ───────────────────────────────────

def _print_result(r: Dict):
    if r["accuracy"] is None:
        print("  no valid samples"); return
    ppl_str = f", mean_ppl={r['mean_ppl']}" if r.get("mean_ppl") is not None else ""
    print(f"  {r['n_correct']}/{r['n_valid']} = {r['accuracy']:.4f}"
          f"  [judge_failed={r['judge_failed']}, skipped={r['skipped']}{ppl_str}]")
    for c, a in r["per_class"].items():
        if a is not None:
            print(f"    {c:<24} {a:.4f}")


def _save_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ─────────────────────── main ──────────────────────────────────────

def main():
    from run.cache_activations import DEFAULT_JUDGE_MODEL

    parser = argparse.ArgumentParser(description="Prompt-hint RFI baseline")
    parser.add_argument("--model",    required=True)
    parser.add_argument("--data-path",   default=None)
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--output-dir",  default=None)
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--dtype",       default="bfloat16")
    parser.add_argument("--judge-model",          default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-device",         default="cuda")
    parser.add_argument("--judge-max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    from utils import load_samples

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)

    model_name = Path(args.model).name

    output_dir = Path(args.output_dir) if args.output_dir \
                 else Path("/data/Agent-Tool-Use-MI/outputs") / model_name / "baseline_rfi"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model : {args.model}")
    llm, tokenizer = _load_model(args.model, args.device, torch_dtype)
    llm.eval()
    _clear_sampling_generation_config(llm)

    print(f"Loading judge : {args.judge_model}")
    judge_model, judge_tok = _load_model(args.judge_model, args.judge_device, torch_dtype)
    judge_model.eval()
    _clear_sampling_generation_config(judge_model)

    samples = load_samples("when2call", args.data_path, args.num_samples, split="test_mcq")
    samples = [s for s in samples if (s.metadata or {}).get("correct_answer") != "cannot_answer"]
    print(f"Samples       : {len(samples)}  (cannot_answer excluded)")

    r = evaluate_accuracy(
        llm, tokenizer, judge_model, judge_tok,
        samples,
        device=args.device, judge_device=args.judge_device,
        max_new_tokens=args.judge_max_new_tokens,
    )
    _print_result(r)

    _save_json(output_dir / "result.json", {
        "model":       args.model,
        "judge_model": args.judge_model,
        "hint":        HINT,
        "result":      r,
    })
    print(f"\nSaved results → {output_dir}")


if __name__ == "__main__":
    main()
