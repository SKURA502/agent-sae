"""
Cache Activations - 激活提取（分析用）

子命令：
  extract  从测试集（MCQ / BFCL）提取并保存激活到磁盘，供后续分析使用

Action boundary：模型读完完整上下文（messages + tools）、即将生成 assistant response
的最后一个 token 处的残差流激活。
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional

import torch
from tqdm import tqdm

from sae.pretrain_data import ActivationBuffer, ActivationStreamer


# ─────────────────────── config ────────────────────────────────────

@dataclass
class StreamingConfig:
    """（旧接口兼容）流式处理配置"""
    buffer_size: int = 8192
    layer: int = 24
    device: str = "cuda"


# ─────────────────────── prompt builder ────────────────────────────

def _build_action_boundary_input_ids(
    tokenizer,
    sample,
    max_length: int = 2048,
) -> torch.Tensor:
    """从 TaskSample 构建 action boundary input_ids（不含 assistant response）。

    优先使用 sample.metadata["original_messages"]；如无，则用 instruction 重建。

    Returns:
        input_ids: [1, seq_len]
    """
    meta = sample.metadata or {}
    messages = meta.get("original_messages") or []
    tools = sample.tool_schemas or meta.get("original_tools") or []

    if not messages:
        messages = [{"role": "user", "content": sample.instruction}]

    # 尝试带 tools 参数应用 chat template（Qwen3.5 支持）
    try:
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            tools=tools if tools else None,
        )
    except Exception:
        try:
            ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        except Exception:
            # 最后兜底：直接拼接文本
            text = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
            )
            ids = tokenizer.encode(text, add_special_tokens=True)

    if len(ids) > max_length:
        ids = ids[-max_length:]

    return torch.tensor([ids], dtype=torch.long)


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

def cmd_extract(args):
    """Extract：提取激活并保存到磁盘，用于 H1/H3 分析。"""
    from utils import load_samples
    from .when2call_adapter import DecisionLabel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype,
        device_map=args.device, trust_remote_code=True,
    )
    llm.eval()

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

    for sample in tqdm(samples, desc="提取激活"):
        try:
            input_ids = _build_action_boundary_input_ids(
                tokenizer, sample, max_length=getattr(args, "max_length", 2048)
            )
            raw_acts = streamer.extract_activations(input_ids, positions)
            for l in layers:
                if l in raw_acts:
                    act = raw_acts[l]  # [1, n_pos, hidden] or [1, seq_len, hidden]
                    if hook_position == "last":
                        act = act[:, 0, :]   # [1, hidden]
                    else:
                        act = act.mean(dim=1)  # [1, hidden]
                    acts_by_layer[l].append(act.cpu())
            labels_list.append(1 if sample.label == DecisionLabel.CALL else 0)
        except Exception as e:
            tqdm.write(f"Warning: skip {getattr(sample, 'sample_id', '?')}: {e}")

    streamer.cleanup()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_tensor = torch.tensor(labels_list, dtype=torch.long)

    for l in layers:
        if not acts_by_layer[l]:
            continue
        acts_tensor = torch.cat(acts_by_layer[l], dim=0)  # [N, hidden]
        save_path = output_dir / f"layer_{l}_activations.pt"
        torch.save({f"layer_{l}": acts_tensor, "labels": labels_tensor}, save_path)
        print(f"Saved layer {l}: {acts_tensor.shape} → {save_path}")

    meta = {
        "dataset": args.dataset,
        "num_samples": len(labels_list),
        "num_call": int(labels_tensor.sum()),
        "num_no_call": int((labels_tensor == 0).sum()),
        "layers": layers,
        "hook_position": hook_position,
    }
    with open(output_dir / "extraction_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nExtraction complete: {len(labels_list)} samples, "
          f"{meta['num_call']} CALL + {meta['num_no_call']} NO_CALL")


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
    pe.add_argument("--output-dir", type=str, default="./outputs/activations")
    pe.add_argument("--max-length", type=int, default=2048)
    pe.add_argument("--hook-position", type=str, default="last",
                    choices=["last", "last_t", "all"],
                    help="Which token positions to extract (last: action boundary token, "
                         "last_t: mean of last T tokens, all: mean of full sequence)")
    pe.add_argument("--last-t", type=int, default=8,
                    help="Number of trailing tokens to mean-pool (used with --hook-position=last_t)")

    args = parser.parse_args()

    if args.command == "extract":
        cmd_extract(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
