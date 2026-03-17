"""
Cache Activations - 激活提取与 SAE Stage 2 训练

子命令：
  stage2   从有标签的 When2Call Pref 数据流式提取 action boundary 激活，训练 SAE Stage 2
  extract  从测试集（MCQ / BFCL）提取并保存激活到磁盘，供后续分析使用

Action boundary：模型读完完整上下文（messages + tools）、即将生成 assistant response
的最后一个 token 处的残差流激活。
"""

import argparse
import json
import random
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


# ─────────────────────── activation extraction ─────────────────────

def _extract_last_token_acts(
    streamer: ActivationStreamer,
    input_ids: torch.Tensor,
) -> Dict[int, torch.Tensor]:
    """提取最后一个 token 的激活。

    Args:
        streamer:  ActivationStreamer 实例
        input_ids: [1, seq_len]

    Returns:
        {layer_idx: [1, hidden_dim]}  (CPU tensor)
    """
    streamer._activations = {}
    if not streamer._hooks:
        streamer._register_hooks()

    with torch.no_grad():
        streamer.model(input_ids.to(streamer.device))

    result: Dict[int, torch.Tensor] = {}
    for layer_idx, acts in streamer._activations.items():
        # acts: [batch=1, seq_len, hidden_dim]
        result[layer_idx] = acts[:, -1, :].detach().cpu()  # [1, hidden_dim]
    return result


# ─────────────────────── Stage 2 generator ─────────────────────────

def create_stage2_data_generator(
    model,
    tokenizer,
    samples: list,
    layers: List[int],
    device: str,
    buffer_size: int = 4096,
    max_length: int = 2048,
) -> Generator[Dict[int, torch.Tensor], None, None]:
    """从有标签样本提取 action boundary 激活，流式供 SAE Stage 2 训练。

    Yields:
        {layer_idx: [buffer_size, hidden_dim]}
    """
    streamer = ActivationStreamer(model, tokenizer, layers, device)
    buffer = ActivationBuffer(buffer_size=buffer_size, layers=layers)

    for sample in tqdm(samples, desc="提取 Stage 2 激活"):
        try:
            input_ids = _build_action_boundary_input_ids(tokenizer, sample, max_length)
            acts = _extract_last_token_acts(streamer, input_ids)
            buffer.add(acts)
            if buffer.is_ready():
                yield buffer.get_and_clear()
        except Exception as e:
            sid = getattr(sample, "sample_id", "?")
            tqdm.write(f"Warning: skip sample {sid}: {e}")
            continue

    if buffer.current_size > 0:
        yield buffer.get_and_clear()

    streamer.cleanup()


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


# ─────────────────────── helpers ───────────────────────────────────

def _find_stage1_checkpoint(stage1_dir: str, layer: int) -> Optional[str]:
    """查找 Stage 1 checkpoint（兼容新旧命名）。"""
    d = Path(stage1_dir)
    for pattern in (f"*-L{layer}-*-stage1.pt", f"*-layer{layer}-*-stage1.pt"):
        matches = sorted(d.glob(pattern))
        if matches:
            return str(matches[0])
    print(f"Warning: Stage 1 checkpoint not found for layer {layer} in {stage1_dir}")
    return None


def _get_hidden_size(model) -> int:
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", None)
    hs = (getattr(text_cfg, "hidden_size", None) if text_cfg else None) or getattr(cfg, "hidden_size", None)
    if hs is None:
        raise ValueError("Cannot infer hidden_size from model config")
    return int(hs)


def _balance_samples(samples: list, max_per_class: int, seed: int = 42) -> list:
    """1:1 平衡 CALL / NO_CALL 样本。"""
    from tasks.base_adapter import DecisionLabel
    call_s = [s for s in samples if s.label == DecisionLabel.CALL]
    no_call_s = [s for s in samples if s.label == DecisionLabel.NO_CALL]
    n = min(len(call_s), len(no_call_s), max_per_class)
    rng = random.Random(seed)
    balanced = rng.sample(call_s, n) + rng.sample(no_call_s, n)
    rng.shuffle(balanced)
    return balanced


# ─────────────────────── cmd_stage2 ────────────────────────────────

def cmd_stage2(args):
    """Stage 2：从有标签数据提取激活并训练 SAE。"""
    from utils import load_samples
    from sae.train_sae import TwoStageTrainer
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
    hidden_size = _get_hidden_size(llm)
    print(f"Hidden size: {hidden_size}")

    # 加载并平衡采样
    samples = load_samples(args.dataset, args.data_path, args.num_samples)
    print(f"Loaded {len(samples)} samples")

    if args.balance:
        samples = _balance_samples(samples, args.max_per_class)
        print(f"After balancing: {len(samples)} samples")

    layers: List[int] = args.layers

    # ── 一次性提取所有层的激活 ────────────────────────────────────────
    print(f"Extracting action boundary activations for layers {layers}...")
    streamer = ActivationStreamer(llm, tokenizer, layers, args.device)
    acts_by_layer: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}

    for sample in tqdm(samples, desc="提取激活"):
        try:
            input_ids = _build_action_boundary_input_ids(
                tokenizer, sample, max_length=args.max_length
            )
            acts = _extract_last_token_acts(streamer, input_ids)
            for l in layers:
                if l in acts:
                    acts_by_layer[l].append(acts[l])  # [1, hidden]
        except Exception as e:
            tqdm.write(f"Warning: skip {getattr(sample, 'sample_id', '?')}: {e}")

    streamer.cleanup()

    # ── 每层训练 SAE Stage 2 ──────────────────────────────────────────
    for layer in layers:
        if not acts_by_layer[layer]:
            print(f"No activations for layer {layer}, skipping")
            continue

        train_data = torch.cat(acts_by_layer[layer], dim=0)  # [N, hidden]
        print(f"\n=== Stage 2 SAE | layer {layer} | {len(train_data)} samples ===")

        stage1_ckpt = _find_stage1_checkpoint(args.stage1_dir, layer)

        trainer = TwoStageTrainer(
            model_name_or_path=args.model,
            layer=layer,
            output_dir=args.output_dir,
            device=args.device,
            dtype=args.dtype,
        )
        # 注入已加载 LLM，避免二次加载
        trainer.model = llm
        trainer.tokenizer = tokenizer
        trainer.hidden_size = hidden_size

        tooluse_config = {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "target_tokens": len(train_data),
            "use_swanlab": args.use_swanlab,
        }
        trainer.init_stage2(stage1_ckpt, tooluse_config)

        # 直接在内存数据上训练（不需要流式，Stage 2 数据集小）
        trainer.sae_trainer.train(train_data)
        trainer.sae_trainer.model._normalize_decoder()

        ckpt_name = f"{trainer.sae_trainer.config.experiment_name}.pt"
        trainer.sae_trainer.save_checkpoint(ckpt_name)
        out_path = Path(args.output_dir) / "stage2" / ckpt_name
        print(f"Layer {layer} Stage 2 complete → {out_path}")


# ─────────────────────── cmd_extract ───────────────────────────────

def cmd_extract(args):
    """Extract：提取激活并保存到磁盘，用于 H1/H3 分析。"""
    from utils import load_samples
    from tasks.base_adapter import DecisionLabel
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

    acts_by_layer: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}
    labels_list: List[int] = []

    for sample in tqdm(samples, desc="提取激活"):
        try:
            input_ids = _build_action_boundary_input_ids(
                tokenizer, sample, max_length=getattr(args, "max_length", 2048)
            )
            acts = _extract_last_token_acts(streamer, input_ids)
            for l in layers:
                if l in acts:
                    acts_by_layer[l].append(acts[l])
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
    }
    with open(output_dir / "extraction_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nExtraction complete: {len(labels_list)} samples, "
          f"{meta['num_call']} CALL + {meta['num_no_call']} NO_CALL")


# ─────────────────────── CLI ───────────────────────────────────────

def main():
    """命令行入口"""
    from utils import add_common_args, add_sae_args, add_dataset_args

    parser = argparse.ArgumentParser(description="激活提取与 SAE Stage 2 训练")
    sub = parser.add_subparsers(dest="command")

    # ── stage2 ──────────────────────────────────────────────────────
    p2 = sub.add_parser("stage2", help="Stage 2 SAE 训练（从有标签的工具调用数据）")
    add_common_args(p2)
    add_sae_args(p2)
    add_dataset_args(p2)
    p2.add_argument("--layers", type=int, nargs="+", default=[24, 26],
                    help="要训练 SAE 的目标层（Qwen3.5-4B: 24 and 26）")
    p2.add_argument("--output-dir", type=str, default="./outputs/sae_checkpoints")
    p2.add_argument("--stage1-dir", type=str, required=True,
                    help="Stage 1 checkpoint 所在目录")
    p2.add_argument("--num-epochs", type=int, default=3,
                    help="Stage 2 训练 epoch 数（小数据集建议 3-5）")
    p2.add_argument("--max-length", type=int, default=2048,
                    help="最大输入长度（tokens）")
    p2.add_argument("--balance", action="store_true",
                    help="平衡 CALL/NO_CALL 采样（1:1）")
    p2.add_argument("--max-per-class", type=int, default=3000,
                    help="平衡采样时每类最多样本数")

    # ── extract ─────────────────────────────────────────────────────
    pe = sub.add_parser("extract", help="提取并保存激活（用于 H1/H3 分析）")
    add_common_args(pe)
    add_dataset_args(pe)
    pe.add_argument("--layers", type=int, nargs="+", default=[24, 26])
    pe.add_argument("--output-dir", type=str, default="./outputs/activations")
    pe.add_argument("--max-length", type=int, default=2048)

    args = parser.parse_args()

    if args.command == "stage2":
        cmd_stage2(args)
    elif args.command == "extract":
        cmd_extract(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
