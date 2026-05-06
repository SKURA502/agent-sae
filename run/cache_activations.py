"""
Cache Activations - 激活提取（分析用）

子命令：
  extract  从测试集（MCQ / BFCL）提取并保存激活到磁盘，供后续分析使用

Action boundary：模型读完完整上下文（system + tools + user question）、即将生成
assistant response 的最后一个 token 处的残差流激活。

标签判定方式：
  judge    让主模型生成一段回答，再用 LLM-as-judge 模型（JUDGE_PROMPT）对回答分类，
           输出 direct_answer / tool_call / request_for_info / cannot_answer 之一。
"""

import argparse
import json
import os
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple
import torch
from tqdm import tqdm

from sae.pretrain_data import ActivationBuffer, ActivationStreamer
from utils.templates import (
    JUDGE_PROMPT,
    MCQ_CHOICES,
    build_context_input_ids as _build_context_input_ids,
)


# ─────────────────────── config ────────────────────────────────────

@dataclass
class StreamingConfig:
    """（旧接口兼容）流式处理配置"""
    buffer_size: int = 8192
    layer: int = 24
    device: str = "cuda"


# ─────────────────────── judge-based labeling ──────────────────────


DEFAULT_JUDGE_MODEL = os.path.join(os.environ.get("SOURCE_ROOT", ""), "model/Qwen/Qwen3.5-27B")


def _clear_sampling_generation_config(model) -> None:
    """清除模型 generation_config 中的采样参数（temperature/top_p/top_k）。

    Qwen 等模型的 generation_config.json 预设了这些字段；当 do_sample=False 时
    HuggingFace 会发出 'generation flags are not valid' warning。直接置 None 即可消除。
    """
    gc = getattr(model, "generation_config", None)
    if gc is None:
        return
    for attr in ("temperature", "top_p", "top_k", "max_length"):
        if getattr(gc, attr, None) is not None:
            setattr(gc, attr, None)


def _generate_response(model, tokenizer, context_ids: torch.Tensor, device: str,
                       max_new_tokens: int = 256) -> str:
    """用主模型对 context 生成一段回答，供 judge 分类使用。"""
    context_ids_dev = context_ids.to(device)
    # 显式传入全 1 attention_mask，避免 pad_token_id==eos_token_id 时
    # HuggingFace generate() 将 prompt 内的 <|im_end|> 误判为 padding 而 mask 掉
    attention_mask = torch.ones_like(context_ids_dev)
    with torch.no_grad():
        out_ids = model.generate(
            context_ids_dev,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = out_ids[0, context_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def _judge_classify(
    judge_model,
    judge_tokenizer,
    sample,
    model_response: str,
    device: str,
) -> Tuple[Optional[str], str]:
    """调用 judge 模型，用 JUDGE_PROMPT 对主模型回答分类。

    Returns:
        (mcq_choice_key, raw_judge_output)
        mcq_choice_key 是 MCQ_CHOICES 中的 key（失败时为 None）
    """
    meta = sample.metadata or {}
    tools_raw: List = meta.get("original_tools_raw") or []
    tool_strs = [t if isinstance(t, str) else json.dumps(t, ensure_ascii=False) for t in tools_raw]
    tools_str = "\n".join(tool_strs) if tool_strs else "(none)"
    user_question = sample.instruction or "(no question)"

    prompt = JUDGE_PROMPT.format(tools_str, user_question, model_response)
    messages = [{"role": "user", "content": prompt}]

    try:
        tokenized = judge_tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", enable_thinking=False,
        )
    except TypeError:
        tokenized = judge_tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt",
        )

    # apply_chat_template returns BatchEncoding (dict-like), not a tensor
    if isinstance(tokenized, torch.Tensor):
        input_ids = tokenized.to(device)
        attention_mask = torch.ones_like(input_ids)
    else:
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        else:
            attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out_ids = judge_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=judge_tokenizer.eos_token_id,
        )

    new_ids = out_ids[0, input_ids.shape[1]:]
    raw = judge_tokenizer.decode(new_ids, skip_special_tokens=True)

    # 解析 JSON，提取 classification 字段
    try:
        m = re.search(r'\{[^}]+\}', raw)
        if m:
            classification = json.loads(m.group()).get("classification", "").strip()
            mapped = classification if classification in MCQ_CHOICES else None
            return mapped, raw
    except (json.JSONDecodeError, AttributeError):
        pass

    return None, raw


# ─────────────────────── legacy pipeline (rollout-based) ───────────

def create_streaming_data_pipeline(
    rollout_generator,
    samples: list,
    config: StreamingConfig,
) -> Generator[Dict[int, torch.Tensor], None, None]:
    """从 rollout 生成器提取激活（兼容旧接口）。"""
    buffer = ActivationBuffer(buffer_size=config.buffer_size, layers=[config.layer])

    for _episode_log, activations in rollout_generator.run_streaming(samples):
        merged: Dict[int, torch.Tensor] = {}
        if activations.get(config.layer):
            merged[config.layer] = torch.cat(activations[config.layer], dim=0)
        if merged:
            buffer.add(merged)
        if buffer.is_ready():
            yield buffer.get_and_clear()

    if buffer.current_size > 0:
        yield buffer.get_and_clear()


# ─────────────────────── cmd_extract ───────────────────────────────

def _load_model(model_path: str, device: str, torch_dtype):
    """加载模型，兼容 CausalLM / ImageTextToText / FP8 量化等情形。"""
    import os
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    # ── virtiofs 兼容补丁 ──────────────────────────────────────────────
    # 在部分 Kata 容器环境中，virtiofs 文件系统的 os.path.isdir 对
    # 绝对路径可能返回 False，导致 transformers 的 cached_files 跳过
    # 本地文件查找、直接调用 hf_hub_download，而 validate_repo_id
    # 又会拒绝含多个 / 的绝对路径。两个补丁联合解决此问题：
    #   1) os.path.isdir fallback：isdir 失败时用 os.path.exists 兜底
    #   2) validate_repo_id bypass：对本地目录跳过 repo ID 格式校验
    _orig_isdir = os.path.isdir
    def _patched_isdir(s):
        result = _orig_isdir(s)
        if not result and isinstance(s, str) and s.startswith('/') and os.path.exists(s):
            return True
        return result
    os.path.isdir = _patched_isdir

    from huggingface_hub.utils._validators import validate_repo_id as _orig_validate
    def _patched_validate(repo_id):
        if os.path.isdir(repo_id):
            return
        _orig_validate(repo_id)
    import huggingface_hub.utils._validators as _hv
    _hv.validate_repo_id = _patched_validate

    # ── 修复装饰器闭包捕获问题 ────────────────────────────────────────
    # @validate_hf_hub_args 装饰器在模块导入时就把 validate_repo_id
    # 闭包捕获到了 _inner_fn 中。虽然 _inner_fn 通过 LOAD_GLOBAL 查找
    # validate_repo_id（而非闭包变量），且 _hv.__dict__ 就是
    # _inner_fn.__globals__，所以修改 _hv.validate_repo_id 理论上能生效，
    # 但为了确保所有已导入的装饰器 wrapper 都能使用补丁版本，
    # 我们也直接修改 hf_hub_download.__globals__ 中的引用。
    try:
        from huggingface_hub.file_download import hf_hub_download
        if 'validate_repo_id' in hf_hub_download.__globals__:
            hf_hub_download.__globals__['validate_repo_id'] = _patched_validate
    except Exception as e:
        print(f"  Warning: failed to patch hf_hub_download globals: {e}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

    # FP8 量化权重反量化
    def _disable_fp8(cfg):
        qc = getattr(cfg, "quantization_config", None)
        if qc is None:
            return
        if isinstance(qc, dict):
            if qc.get("quant_method") == "fp8":
                print("  Setting dequantize=True on FP8 quantization config")
                qc["dequantize"] = True
        else:
            if getattr(qc, "quant_method", None) == "fp8":
                print("  Setting dequantize=True on FP8 quantization config")
                qc.dequantize = True

    _disable_fp8(model_config)
    text_cfg = getattr(model_config, "text_config", None)
    if text_cfg is not None:
        _disable_fp8(text_cfg)

    load_kwargs = dict(
        config=model_config,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
        local_files_only=True,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    except ValueError:
        from transformers import AutoModelForImageTextToText
        print("  AutoModelForCausalLM failed, retrying with AutoModelForImageTextToText ...")
        model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)

    return model, tokenizer


def cmd_extract(args):
    """Extract：提取激活并保存到磁盘，用于 H1/H3 分析。"""
    from utils import load_samples

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)

    print(f"Loading model: {args.model}")
    llm, tokenizer = _load_model(args.model, args.device, torch_dtype)
    llm.eval()
    _clear_sampling_generation_config(llm)

    # ── judge 模型 ───────────────────────────────────────────────────
    print(f"Loading judge model: {args.judge_model}")
    judge_model, judge_tokenizer = _load_model(args.judge_model, args.judge_device, torch_dtype)
    judge_model.eval()
    _clear_sampling_generation_config(judge_model)

    samples = load_samples(args.dataset, args.data_path, args.num_samples)
    print(f"Loaded {len(samples)} samples")

    layers: List[int] = args.layers
    streamer = ActivationStreamer(llm, tokenizer, layers, args.device)

    hook_position = args.hook_position
    last_t = args.last_t
    if hook_position == "last":
        positions = [-1]
    elif hook_position == "last_t":
        positions = list(range(-last_t, 0))
    else:  # "all"
        positions = None

    acts_by_layer: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}
    labels_list: List[int] = []
    correct_answers_list: List[Optional[str]] = []   # 数据集 ground-truth
    skipped = 0
    DEBUG_CASES = 10  # 每个 ground-truth label 打印 N 条，方便人工确认 label 正确性
    _DEBUG_LABELS = {"tool_call", "cannot_answer", "request_for_info"}
    debug_counts: Dict[str, int] = {lbl: 0 for lbl in _DEBUG_LABELS}

    _debug_dir = Path(args.output_dir if args.output_dir is not None else "./outputs/judge")
    _debug_dir.mkdir(parents=True, exist_ok=True)
    _debug_file = open(_debug_dir / "debug_cases.txt", "w", encoding="utf-8")
    print(f"Debug cases will be written to: {_debug_dir / 'debug_cases.txt'}")

    def _dbg(*a, **kw):
        print(*a, **kw, file=_debug_file)

    for idx, sample in enumerate(tqdm(samples, desc="提取激活")):
        try:
            input_ids = _build_context_input_ids(
                tokenizer, sample, max_length=getattr(args, "max_length", 2048)
            )

            # ── 激活提取（action boundary = context 末尾）────────────────
            raw_acts = streamer.extract_activations(input_ids, positions)
            # 先存入临时 dict，labeling 成功后再 commit，避免激活与标签长度不一致
            pending_acts: Dict[int, torch.Tensor] = {}
            for l in layers:
                if l in raw_acts:
                    act = raw_acts[l]  # [1, n_pos, hidden]
                    if hook_position == "last":
                        act = act[:, 0, :]    # [1, hidden]
                    else:
                        act = act.mean(dim=1)  # [1, hidden]
                    pending_acts[l] = act.cpu()

            # ── 标签：judge 打标 ──────────────────────────────────────
            gt_label = sample.metadata.get("correct_answer")

            model_response = _generate_response(
                llm, tokenizer, input_ids, args.device,
                max_new_tokens=args.judge_max_new_tokens,
            )
            choice, judge_raw = _judge_classify(
                judge_model, judge_tokenizer, sample, model_response, args.judge_device,
            )
            label = MCQ_CHOICES.index(choice) if choice in MCQ_CHOICES else -1
            for l, act in pending_acts.items():
                acts_by_layer[l].append(act)
            labels_list.append(label)
            correct_answers_list.append(gt_label)

            if gt_label in _DEBUG_LABELS and debug_counts[gt_label] < DEBUG_CASES:
                debug_counts[gt_label] += 1
                prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
                _dbg(f"\n{'='*70}")
                _dbg(f"[Case {idx}]  gt={gt_label!r} ({debug_counts[gt_label]}/{DEBUG_CASES})"
                     f"  sample_id={getattr(sample, 'sample_id', '?')}")
                _dbg(f"  input_ids shape={input_ids.shape}  dtype={input_ids.dtype}")
                _dbg(f"── PROMPT (last 600 chars) ──")
                _dbg(prompt_text)
                _dbg(f"── MODEL RESPONSE ──")
                _dbg(model_response)
                _dbg(f"── JUDGE OUTPUT ──")
                _dbg(judge_raw)
                _dbg(f"── PREDICTED choice={choice!r}  label={label}  "
                     f"(dataset correct_answer={gt_label!r})")
                _dbg(f"{'='*70}")
                _debug_file.flush()

        except Exception as e:
            tb = traceback.format_exc()
            tqdm.write(f"Warning: skip {getattr(sample, 'sample_id', '?')}: {e!r}\n{tb}")
            skipped += 1

    _debug_file.close()
    streamer.cleanup()

    output_dir = Path(args.output_dir if args.output_dir is not None else "./outputs/judge")
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    # ground-truth label indices (-1 if unavailable)
    gt_indices = [MCQ_CHOICES.index(gt) if gt in MCQ_CHOICES else -1
                  for gt in correct_answers_list]
    gt_tensor = torch.tensor(gt_indices, dtype=torch.long)

    for l in layers:
        if not acts_by_layer[l]:
            continue
        acts_tensor = torch.cat(acts_by_layer[l], dim=0)  # [N, hidden]
        save_path = output_dir / f"layer_{l}_activations.pt"
        torch.save({
            f"layer_{l}": acts_tensor,
            "labels": labels_tensor,
            "gt_labels": gt_tensor,
        }, save_path)
        print(f"Saved layer {l}: {acts_tensor.shape} → {save_path}")

    meta = {
        "dataset": args.dataset,
        "num_samples": len(labels_list),
        "label_map": {i: c for i, c in enumerate(MCQ_CHOICES)},
        "label_counts": {c: int((labels_tensor == i).sum()) for i, c in enumerate(MCQ_CHOICES)},
        "num_score_failed": int((labels_tensor == -1).sum()),
        "layers": layers,
        "hook_position": hook_position,
        "label_method": "judge",
        "scoring": "llm_judge",
        "judge_model": args.judge_model,
    }

    # ── MCQ 准确率统计 ────────────────────────────────────────────────
    valid = [(MCQ_CHOICES[l], gt) for l, gt in zip(labels_list, correct_answers_list)
             if l >= 0 and gt is not None]
    if valid:
        n_correct = sum(pred == gt for pred, gt in valid)
        accuracy = n_correct / len(valid)
        per_class: Dict[str, Dict[str, int]] = {c: {"correct": 0, "total": 0} for c in MCQ_CHOICES}
        for pred, gt in valid:
            if gt in per_class:
                per_class[gt]["total"] += 1
                if pred == gt:
                    per_class[gt]["correct"] += 1
        meta["mcq_accuracy"] = round(accuracy, 4)
        meta["mcq_per_class_accuracy"] = {
            c: round(v["correct"] / v["total"], 4) if v["total"] else None
            for c, v in per_class.items()
        }

        print(f"\n{'='*60}")
        print(f"MCQ Accuracy: {n_correct}/{len(valid)} = {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"  Per-class accuracy (ground-truth class):")
        for c, v in per_class.items():
            if v["total"] > 0:
                acc = v["correct"] / v["total"]
                print(f"    {c:<22} {v['correct']:>4}/{v['total']:<4} = {acc:.3f}")
        print(f"{'='*60}")
    else:
        print("\n(no ground-truth labels available for accuracy calculation)")

    with open(output_dir / "extraction_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    counts_str = ", ".join(f"{c}={meta['label_counts'][c]}" for c in MCQ_CHOICES)
    print(f"\nExtraction complete: {len(labels_list)} samples "
          f"({counts_str}, score_failed={meta['num_score_failed']}, skipped={skipped})")


# ─────────────────────── CLI ───────────────────────────────────────

def main():
    """命令行入口"""
    from utils import add_common_args, add_dataset_args

    parser = argparse.ArgumentParser(description="激活提取（分析用）")
    sub = parser.add_subparsers(dest="command")

    # ── extract ─────────────────────────────────────────────────────
    pe = sub.add_parser("extract", help="提取并保存激活（用于 H1/H3 分析）")
    add_common_args(pe)
    add_dataset_args(pe)
    pe.add_argument("--layers", type=int, nargs="+", default=[24, 26])
    pe.add_argument("--output-dir", type=str, default=None,
                    help="Output directory. Defaults to ./outputs/judge.")
    pe.add_argument("--max-length", type=int, default=2048)
    pe.add_argument("--hook-position", type=str, default="last",
                    choices=["last", "last_t", "all"],
                    help="Which token positions to extract (last: action boundary token, "
                         "last_t: mean of last T tokens, all: mean of full sequence)")
    pe.add_argument("--last-t", type=int, default=8,
                    help="Number of trailing tokens to mean-pool (used with --hook-position=last_t)")
    pe.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL,
                    help="Path to judge model (used with --label-method=judge). "
                         f"Default: {DEFAULT_JUDGE_MODEL}")
    pe.add_argument("--judge-device", type=str, default="cuda",
                    help="Device for judge model (default: cuda). "
                         "Set to 'cuda:1' etc. when main and judge models share a multi-GPU node.")
    pe.add_argument("--judge-max-new-tokens", type=int, default=256,
                    help="Max tokens for main model generation before judge classification (default: 256)")

    args = parser.parse_args()

    if args.command == "extract":
        cmd_extract(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
