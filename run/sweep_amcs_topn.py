"""
sweep_amcs_topn.py
─────────────────────────────────────────────────────────────────────────
Sweep over top_n feature hyperparameter for AMCS evaluation.

For each top_n:
  1. Compute delta = -β₀/β from cached activations via logistic regression
     (fast, no LLM inference)
  2. Run AMCS evaluation with that top_n and delta

LLM and judge models are loaded once; evaluation loops over top_n.

Usage:
  python -m run.sweep_amcs_topn \\
    --model  /path/to/Qwen3.5-4B \\
    --sae-path /path/to/best.pt \\
    --layer 25 \\
    --top-n-values 5 10 15 20 25 30 \\
    --output-dir ./outputs/analysis/Qwen3.5-4B/amcs_sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run.cache_activations import _clear_sampling_generation_config, DEFAULT_JUDGE_MODEL
from run.eval_amcs_accuracy import (
    AMCSConfig,
    evaluate_accuracy,
    load_amcs_features,
    _get_layer_module,
    _load_json_list,
    _print_result,
    _save_json,
    _strip_labels,
)

MCQ_CHOICES = ("direct_answer", "tool_call", "request_for_info", "cannot_answer")
MCQ_LABEL = {c: i for i, c in enumerate(MCQ_CHOICES)}


# ─────────────────────── delta computation ────────────────────────────

def _encode_activations(sae, acts: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
    latents = []
    for i in range(0, len(acts), batch_size):
        batch = acts[i : i + batch_size].to(
            sae.config.device, dtype=sae.config.get_torch_dtype()
        )
        with torch.no_grad():
            latents.append(sae.encode(batch).cpu().float())
    return torch.cat(latents, dim=0)


def compute_delta(
    latents: torch.Tensor,      # [N, dict_size]  decoder-norm scaled, on CPU
    labels: torch.Tensor,       # [N]  model predictions (from cached activations)
    gt_labels: torch.Tensor,    # [N]  ground-truth
    tc_feat_indices: List[int],
    rfi_feat_indices: List[int],
) -> Tuple[float, float, float]:
    """Compute delta = -β₀/β via logistic regression on cached latents.

    Returns (beta, beta_0, delta).
    """
    from sklearn.linear_model import LogisticRegression

    tc_idx  = MCQ_LABEL["tool_call"]
    rfi_idx = MCQ_LABEL["request_for_info"]

    labels_np   = labels.numpy()
    gt_np       = gt_labels.numpy()

    # keep GT ∈ {TC, RFI}
    gt_mask  = (gt_np == tc_idx) | (gt_np == rfi_idx)
    sub_lat  = latents[gt_mask].numpy()
    sub_pred = labels_np[gt_mask]

    tc_act  = sub_lat[:, tc_feat_indices].mean(axis=1)
    rfi_act = sub_lat[:, rfi_feat_indices].mean(axis=1)
    margin  = tc_act - rfi_act

    # keep pred ∈ {TC, RFI}
    pred_mask = (sub_pred == tc_idx) | (sub_pred == rfi_idx)
    y = (sub_pred[pred_mask] == tc_idx).astype(float)
    X = margin[pred_mask]

    clf = LogisticRegression(fit_intercept=True, max_iter=1000)
    clf.fit(X.reshape(-1, 1), y)
    beta   = float(clf.coef_[0][0])
    beta_0 = float(clf.intercept_[0])
    delta  = -beta_0 / beta
    return beta, beta_0, delta


# ─────────────────────── CLI ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AMCS top_n sweep")
    parser.add_argument("--model",       required=True)
    parser.add_argument("--sae-path",    required=True)
    parser.add_argument("--layer",       type=int, required=True)
    parser.add_argument("--data-path",   default=None)
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--output-dir",  default="./outputs/analysis/amcs_sweep")
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--dtype",       default="bfloat16")
    parser.add_argument("--judge-model",          default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-device",         default="cuda")
    parser.add_argument("--judge-max-new-tokens", type=int, default=256)
    parser.add_argument("--top-n-values", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30])
    parser.add_argument("--activations-dir", default=None,
                        help="Directory with layer_N_activations.pt. "
                             "Defaults to <sae_root>/outputs/<model>/activations/when2call_mcq")
    # fixed AMCS params (not swept)
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.5])
    # feature source dirs
    parser.add_argument("--feature-discovery-dir", default=None)
    parser.add_argument("--rfi-confusion-dir",     default=None)
    args = parser.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from sae.sae_model import TopKSAE
    from utils import load_samples

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)
    output_dir = Path(args.output_dir)
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

    # ── resolve dirs ──────────────────────────────────────────────────
    model_name = Path(args.model).name
    sae_root   = Path(args.sae_path).resolve().parent.parent.parent.parent
    analysis   = sae_root / "outputs" / model_name / "analysis"

    fd_dir   = Path(args.feature_discovery_dir) if args.feature_discovery_dir \
               else analysis / "feature_discovery"
    rc_dir   = Path(args.rfi_confusion_dir) if args.rfi_confusion_dir \
               else analysis / "rfi_confusion"
    acts_dir = Path(args.activations_dir) if args.activations_dir \
               else sae_root / "outputs" / model_name / "activations" / "when2call_mcq"

    # ── load models once ──────────────────────────────────────────────
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

    samples = load_samples("when2call", args.data_path, args.num_samples, split="test_mcq")
    samples = [s for s in samples if (s.metadata or {}).get("correct_answer") != "cannot_answer"]
    print(f"Samples       : {len(samples)}  (cannot_answer excluded)")

    # ── load all raw features (up to max available) ───────────────────
    with open(fd_dir / "tool_call"        / f"top_features_layer{args.layer}.json") as f:
        all_tc_raw  = json.load(f)
    with open(fd_dir / "request_for_info" / f"top_features_layer{args.layer}.json") as f:
        all_rfi_raw = json.load(f)
    max_available = min(len(all_tc_raw), len(all_rfi_raw))
    top_n_values  = sorted(n for n in args.top_n_values if n <= max_available)
    if len(top_n_values) < len(args.top_n_values):
        skipped = [n for n in args.top_n_values if n > max_available]
        print(f"  Warning: skipping top_n {skipped} (only {max_available} features available)")

    # ── load cached activations and encode through SAE (once) ─────────
    acts_path = acts_dir / f"layer_{args.layer}_activations.pt"
    print(f"\nLoading cached activations: {acts_path}")
    ckpt        = torch.load(acts_path, map_location="cpu", weights_only=True)
    cached_acts = ckpt[f"layer_{args.layer}"].float()
    cached_pred = ckpt["labels"].long()
    cached_gt   = ckpt["gt_labels"].long()
    print(f"  Shape: {cached_acts.shape}")

    print("Encoding activations through SAE ...")
    latents = _encode_activations(sae, cached_acts)
    with torch.no_grad():
        decoder_norms = sae.decoder.weight.detach().float().cpu().norm(dim=0)
    latents = latents * decoder_norms   # [N, dict_size]  decoder-norm scaled

    # ── step 1: compute delta for each top_n (fast) ───────────────────
    print(f"\n{'='*60}")
    print(f"  Computing delta per top_n from cached activations")
    print(f"  {'top_n':>6}  {'n_reg':>7}  {'β':>8}  {'β₀':>8}  {'δ':>8}")
    print(f"  {'-'*46}")

    top_n_stats: Dict[int, dict] = {}
    for top_n in top_n_values:
        tc_idx_list  = [e["feature_idx"] for e in all_tc_raw[:top_n]]
        rfi_idx_list = [e["feature_idx"] for e in all_rfi_raw[:top_n]]
        beta, beta_0, delta = compute_delta(
            latents, cached_pred, cached_gt, tc_idx_list, rfi_idx_list
        )
        top_n_stats[top_n] = {"beta": beta, "beta_0": beta_0, "delta": delta}
        print(f"  {top_n:>6}  {'':>7}  {beta:>8.4f}  {beta_0:>+8.4f}  {delta:>+8.4f}")

    # ── step 2: AMCS eval per alpha × top_n ──────────────────────────
    alpha_values = sorted(set(args.alpha))
    print(f"\nAlpha values  : {alpha_values}")

    def _fmt(v, fmt):
        return format(v, fmt) if v is not None else "N/A".rjust(len(format(0, fmt)))

    for alpha in alpha_values:
        out_path = output_dir / f"amcs_alpha{alpha}_results.json"
        print(f"\n{'='*72}")
        print(f"  Alpha = {alpha}")
        print(f"{'='*72}")

        sweep_rows = []
        for top_n in top_n_values:
            stats = top_n_stats[top_n]
            delta = stats["delta"]
            print(f"\n{'='*60}\ntop_n={top_n}  α={alpha}  δ={delta:+.4f}")

            tc_feats, rfi_feats = load_amcs_features(fd_dir, rc_dir, args.layer, top_n)
            amcs_config = AMCSConfig(
                delta=delta,
                alpha=alpha,
                tc_features=tc_feats,
                rfi_features=rfi_feats,
            )

            r = evaluate_accuracy(
                llm, tokenizer, judge_model, judge_tok, layer_module, sae, samples,
                device=args.device, judge_device=args.judge_device,
                amcs_config=amcs_config, max_new_tokens=args.judge_max_new_tokens,
                desc=f"AMCS α={alpha} top_n={top_n}",
            )
            _print_result(f"AMCS α={alpha} top_n={top_n}", r)

            entry = {
                "top_n":  top_n,
                "delta":  round(delta, 6),
                "beta":   round(stats["beta"],   6),
                "beta_0": round(stats["beta_0"], 6),
                "amcs":   _strip_labels(r),
            }
            sweep_rows.append(entry)

        # ── summary table for this alpha ──────────────────────────────
        print(f"\n{'='*72}")
        print(f"  Alpha = {alpha}")
        print(f"  {'top_n':>6}  {'δ':>8}  {'AMCS':>10}  {'RFI':>7}  {'CA':>7}  {'TC':>7}")
        print(f"  {'-'*55}")
        for e in sweep_rows:
            a   = e["amcs"]
            rfi = a["per_class"].get("request_for_info") or 0.0
            ca  = a["per_class"].get("cannot_answer")    or 0.0
            tc  = a["per_class"].get("tool_call")        or 0.0
            print(f"  {e['top_n']:>6}  {_fmt(e['delta'],'+8.4f')}"
                  f"  {_fmt(a['accuracy'],'>10.4f')}"
                  f"  {rfi:>7.4f}  {ca:>7.4f}  {tc:>7.4f}")
        print(f"{'='*72}")
        output = {
            "alpha": alpha,
            "sweep": sweep_rows,
        }
        _save_json(out_path, output)
        print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
