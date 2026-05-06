"""
analyze_perclass_ppl.py
────────────────────────────────────────────────────────────────────────
验证假设：baseline 下，request_for_info response 的 PPL 是否本身就高于
tool_call response？

若成立，则 AMCS aggregate PPL 升高可由 RFI 预测比例增加来解释，
而非 steering 本身损害了模型流畅性。

用法：
python -m run.analyze_perclass_ppl \
--model $SOURCE_ROOT/model/Qwen/Qwen3.5-4B \
--judge-model $SOURCE_ROOT/model/Qwen/Qwen3.5-27B \
--data-path $SOURCE_ROOT/dataset/when2call/test \
--device cuda:1 \
--judge-device cuda:1 \
--output-dir ./outputs/Qwen3.5-4B/analysis/perclass_ppl
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.templates import MCQ_CHOICES, build_context_input_ids as _build_context_input_ids
from run.cache_activations import _clear_sampling_generation_config, _judge_classify


def compute_ppl(llm, tokenizer, context_ids: torch.Tensor,
                response_text: str, device: str):
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


def main():
    from run.cache_activations import DEFAULT_JUDGE_MODEL

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       required=True)
    parser.add_argument("--data-path",   default=None)
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--output-dir",  default="./outputs/analysis/perclass_ppl")
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--dtype",       default="bfloat16")
    parser.add_argument("--judge-model",          default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-device",         default="cuda")
    parser.add_argument("--judge-max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
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

    samples = load_samples("when2call", args.data_path, args.num_samples, split="test_mcq")
    print(f"Samples: {len(samples)}")

    # ── 收集每个样本的 (predicted_class, ppl) ──────────────────────
    by_pred: dict[str, List[float]] = defaultdict(list)
    judge_failed = skipped = 0

    for sample in tqdm(samples, desc="baseline"):
        try:
            context_ids = _build_context_input_ids(tokenizer, sample)

            with torch.no_grad():
                out_ids = llm.generate(
                    context_ids.to(args.device),
                    attention_mask=torch.ones_like(context_ids).to(args.device),
                    max_new_tokens=args.judge_max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(
                out_ids[0, context_ids.shape[1]:], skip_special_tokens=True
            )

            choice, _ = _judge_classify(
                judge_model, judge_tok, sample, response, args.judge_device
            )
            if choice is None:
                judge_failed += 1
                continue

            ppl = compute_ppl(llm, tokenizer, context_ids, response, args.device)
            if ppl is not None:
                by_pred[choice].append(ppl)

        except Exception as e:
            tqdm.write(f"  Warning: {e!r}")
            skipped += 1

    # ── 统计 ────────────────────────────────────────────────────────
    def _stats(ppls: List[float]) -> dict:
        return {
            "n":      len(ppls),
            "mean":   round(statistics.mean(ppls),   4),
            "median": round(statistics.median(ppls), 4),
            "std":    round(statistics.stdev(ppls),  4) if len(ppls) > 1 else None,
        }

    results = {c: _stats(by_pred[c]) for c in MCQ_CHOICES if by_pred[c]}
    all_ppls = [p for ppls in by_pred.values() for p in ppls]
    results["overall"] = _stats(all_ppls)

    print(f"\n{'='*55}")
    print(f"  {'Class':<24}  {'mean':>7}  {'median':>7}  {'n':>6}")
    print(f"  {'─'*51}")
    for c in MCQ_CHOICES:
        if c in results:
            s = results[c]
            print(f"  {c:<24}  {s['mean']:>7.4f}  {s['median']:>7.4f}  {s['n']:>6}")
    print(f"  {'─'*51}")
    s = results["overall"]
    print(f"  {'overall':<24}  {s['mean']:>7.4f}  {s['median']:>7.4f}  {s['n']:>6}")
    print(f"{'='*55}")

    tc_mean  = results.get("tool_call",        {}).get("mean")
    rfi_mean = results.get("request_for_info", {}).get("mean")
    if tc_mean and rfi_mean:
        print(f"\n  PPL(RFI) - PPL(TC) = {rfi_mean - tc_mean:+.4f}")
        if rfi_mean > tc_mean:
            print("  → 假设成立：RFI response 本身 PPL 更高")
            print("    AMCS 的 aggregate PPL 升高可由 RFI 预测比例增加来解释")
        else:
            print("  → 假设不成立")

    out_path = output_dir / "perclass_ppl_baseline.json"
    with open(out_path, "w") as f:
        json.dump({"by_pred": results, "judge_failed": judge_failed, "skipped": skipped}, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
