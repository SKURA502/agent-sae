"""
Train SAE - 两阶段训练脚本

Stage 1: 通用预训练语料激活
Stage 2: Tool-use 任务激活

支持运行时推理流式训练；使用 SwanLab 记录指标。
"""

import argparse
import json
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import torch
from tqdm import tqdm

try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False

from .sae_model import TopKSAE, SAEConfig


def pre_process(
    hidden_states: torch.Tensor, 
    eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize hidden states to zero mean and unit variance."""
    mean = hidden_states.mean(dim=-1, keepdim=True)
    std = hidden_states.std(dim=-1, keepdim=True)
    x = (hidden_states - mean) / (std + eps)
    return x, mean, std

# ---------- 流式训练辅助工具 ----------

def _prefetch_generator(gen, maxsize=2):
    """后台线程预取生成器数据，实现推理/训练并行。"""
    q = queue.Queue(maxsize=maxsize)
    sentinel = object()

    def _worker():
        try:
            for item in gen:
                q.put(item)
        finally:
            q.put(sentinel)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is sentinel:
            break
        yield item
    t.join()


class _PendingBuffer:
    """预分配的激活缓冲区，避免反复 torch.cat / slice。"""

    def __init__(self, hidden_dim: int, device: str, dtype: torch.dtype = torch.bfloat16,
                 initial_capacity: int = 16384):
        self.buf = torch.empty(initial_capacity, hidden_dim, device=device, dtype=dtype)
        self.size = 0
        self.hidden_dim = hidden_dim
        self.device = device
        self.dtype = dtype

    def append(self, data: torch.Tensor):
        data = data.to(device=self.device, dtype=self.dtype)
        n = data.shape[0]
        needed = self.size + n
        if needed > self.buf.shape[0]:
            new_cap = max(needed * 2, self.buf.shape[0] * 2)
            new_buf = torch.empty(new_cap, self.hidden_dim, device=self.device, dtype=self.dtype)
            new_buf[:self.size] = self.buf[:self.size]
            self.buf = new_buf
        self.buf[self.size:self.size + n] = data
        self.size += n

    def pop_batch(self, batch_size: int) -> torch.Tensor:
        batch = self.buf[:batch_size].clone()
        remaining = self.size - batch_size
        if remaining > 0:
            self.buf[:remaining] = self.buf[batch_size:self.size].clone()
        self.size = remaining
        return batch

    def pop_all(self) -> torch.Tensor:
        data = self.buf[:self.size].clone()
        self.size = 0
        return data

    def __len__(self) -> int:
        return self.size


def _make_checkpoint_name(
    model_name: str, layer: int, dict_size: int,
    target_tokens: int, stage: str,
) -> str:
    """格式: {LLM}-layer{L}-d{dict_size}-{tokens}M-{stage}.pt"""
    short = model_name.rstrip("/").split("/")[-1]
    tok = f"{target_tokens / 1e6:.0f}M"
    return f"{short}-L{layer}-d{int(dict_size)}-{tok}-{stage}.pt"


def _load_llm(model_name: str, device: str, dtype: str):
    """加载 LLM 并推断 hidden_size，返回 (model, tokenizer, hidden_size)。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    print(f"Loading LLM: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 若模型配置含 FP8 量化，设置 dequantize=True，让 transformers 加载时把
    # FP8 权重反量化为 bfloat16，避免需要 Triton FP8 kernel
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    def _disable_fp8(cfg):
        qc = getattr(cfg, "quantization_config", None)
        if qc is None:
            return
        if isinstance(qc, dict):
            if qc.get("quant_method") == "fp8":
                print("  Setting dequantize=True on FP8 quantization config (Triton FP8 unavailable)")
                qc["dequantize"] = True
        else:
            if getattr(qc, "quant_method", None) == "fp8":
                print("  Setting dequantize=True on FP8 quantization config (Triton FP8 unavailable)")
                qc.dequantize = True

    _disable_fp8(model_config)
    text_cfg = getattr(model_config, "text_config", None)
    if text_cfg is not None:
        _disable_fp8(text_cfg)

    load_kwargs = dict(
        config=model_config,
        torch_dtype=dtype_map.get(dtype, torch.bfloat16),
        device_map=device, trust_remote_code=True,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except ValueError:
        from transformers import AutoModelForImageTextToText
        print("  AutoModelForCausalLM failed, retrying with AutoModelForImageTextToText ...")
        model = AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg = model.config
    text_cfg = getattr(cfg, "text_config", None)
    hs = (getattr(text_cfg, "hidden_size", None) if text_cfg else None) \
        or getattr(cfg, "hidden_size", None)
    if hs is None:
        raise ValueError(f"Cannot infer hidden_size from {type(cfg)}")
    hidden_size = int(hs)
    print(f"LLM loaded. hidden_size={hidden_size}")
    return model, tokenizer, hidden_size


@dataclass
class TrainingConfig:
    """训练配置"""
    input_dim: int = 4096
    dict_size: int = 32768
    k: int = 128
    learning_rate: float = 5e-4
    batch_size: int = 4096
    warmup_ratio: float = 0.1
    stable_ratio: float = 0.8
    decoder_norm_interval: int = 10
    log_interval: int = 10
    drop_last: bool = False
    output_dir: str = "./outputs/sae_checkpoints"
    experiment_name: str = "sae_training"
    use_swanlab: bool = False
    swanlab_project: str = "agent-tool-use"
    device: str = "cuda"
    dtype: str = "bfloat16"

    def to_sae_config(self) -> SAEConfig:
        return SAEConfig(
            input_dim=self.input_dim, dict_size=self.dict_size,
            k=self.k, device=self.device, dtype=self.dtype,
        )


class SAETrainer:
    """SAE 训练器"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._swanlab_active = False

        self.model = TopKSAE(config.to_sae_config())
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate,
        )
        self.global_step = 0

        if config.use_swanlab and SWANLAB_AVAILABLE:
            swanlab.init(
                project=config.swanlab_project,
                experiment_name=config.experiment_name,
                config=vars(config), mode="offline", logdir="swanlog",
            )
            self._swanlab_active = True

    # Core training
    def train_batch(self, batch: torch.Tensor, scheduler: Any) -> Dict[str, float]:
        """单 batch 训练步。"""
        batch = batch.to(self.config.device)
        self.optimizer.zero_grad(set_to_none=True)

        x, _, _ = pre_process(batch)
        loss, loss_dict = self.model.compute_loss(x)
        
        loss.backward()
        self.optimizer.step()
        scheduler.step()
        self.global_step += 1

        if self.global_step % self.config.decoder_norm_interval == 0:
            self.model._normalize_decoder()

        return {
            "loss": float(loss.item()),
            "mean_activation": float(loss_dict["mean_activation"]),
            "lr": float(scheduler.get_last_lr()[0]),
            "global_step": float(self.global_step),
        }

    def train_streaming(
        self,
        activation_generator: Generator[Dict[int, torch.Tensor], None, None],
        layer: int,
        total_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """流式训练：从激活生成器逐批读取并训练（prefetch + 预分配缓冲）。"""
        print(f"Streaming training for layer {layer}, "
              f"dict_size={self.config.dict_size}, k={self.config.k}")

        total_steps = total_steps or 10000
        scheduler = self._make_scheduler(total_steps)

        stats: Dict[str, list] = {"interval_avg_losses": [], "steps": []}
        self.model.train()
        running, n = 0.0, 0
        pbar = tqdm(total=total_steps, desc="Streaming training")
        model_dtype = next(self.model.parameters()).dtype
        pending = _PendingBuffer(
            self.config.input_dim, self.config.device,
            dtype=model_dtype,
            initial_capacity=self.config.batch_size * 4,
        )

        def _run_step(batch: torch.Tensor):
            nonlocal running, n
            m = self.train_batch(batch, scheduler)
            running += m["loss"]
            n += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{m['loss']:.4f}")
            if self.global_step % self.config.log_interval == 0 and n > 0:
                avg = running / n
                stats["interval_avg_losses"].append(avg)
                stats["steps"].append(self.global_step)
                self._log_metrics(avg, m)
                running, n = 0.0, 0

        try:
            for activations in _prefetch_generator(activation_generator):
                if layer not in activations:
                    continue
                data = activations[layer]
                if data.dim() == 3:
                    data = data.view(-1, data.shape[-1])

                pending.append(data)

                while len(pending) >= self.config.batch_size:
                    _run_step(pending.pop_batch(self.config.batch_size))

            if len(pending) > 0 and not self.config.drop_last:
                _run_step(pending.pop_all())

            self._save_stats(stats)
            return stats
        finally:
            pbar.close()
            self._finish_swanlab()

    # Checkpoint
    def save_checkpoint(self, name: str):
        path = self.output_dir / name
        self.model.save(str(path))
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, checkpoint_path: str):
        self.model = TopKSAE.load(checkpoint_path, device=self.config.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.learning_rate,
        )
        print(f"Loaded checkpoint from {checkpoint_path}")

    # Internal helpers
    def _make_scheduler(self, total: int):
        from torch.optim.lr_scheduler import LambdaLR

        total = max(int(total), 1)
        warmup_steps = int(total * self.config.warmup_ratio)
        stable_steps = int(total * self.config.stable_ratio)
        decay_start = min(warmup_steps + stable_steps, total)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            if step < decay_start:
                return 1.0
            return max(0.0, (total - step) / max(1, total - decay_start))

        return LambdaLR(self.optimizer, lr_lambda)

    def _log_metrics(self, avg_loss: float, metrics: Dict[str, float]):
        """流式训练中打日志。"""
        log_data = {
            "train/loss": avg_loss,
            "train/mean_activation": metrics["mean_activation"],
            "train/lr": metrics["lr"],
            "global_step": self.global_step,
        }
        if self._swanlab_active and SWANLAB_AVAILABLE:
            swanlab.log(log_data)
        else:
            tqdm.write(
                f"[Step {self.global_step}] loss={avg_loss:.4f} "
                f"mean_act={metrics['mean_activation']:.4f} "
                f"lr={metrics['lr']:.2e}"
            )

    def _save_stats(self, stats: Dict[str, Any]):
        path = self.output_dir / f"{self.config.experiment_name}_stats.json"
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)

    def _finish_swanlab(self):
        if self._swanlab_active and SWANLAB_AVAILABLE:
            try:
                swanlab.finish()
            except Exception as e:
                print(f"  Warning: swanlab.finish() failed (ignored): {e}")
            self._swanlab_active = False


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="Train SAE (Two-Stage)")
    sub = parser.add_subparsers(dest="command")
    from utils import add_stage_args

    s1 = sub.add_parser("stage1", help="Stage 1: pretrain corpus")
    add_stage_args(s1)

    s2 = sub.add_parser("stage2", help="Stage 2: When2Call pref+sft action-boundary activations")
    add_stage_args(s2)
    s2.add_argument("--stage1-checkpoint", type=str, default=None,
                    help="Stage 1 .pt checkpoint path (optional; random init if omitted)")

    args = parser.parse_args()

    if args.command not in ("stage1", "stage2"):
        parser.print_help()
        return

    from .pretrain_data import PretrainConfig, create_pretrain_data_iterator

    model, tokenizer, hidden_size = _load_llm(args.model, args.device, args.dtype)
    model.eval()

    dict_size = args.dict_size if args.dict_size is not None else hidden_size * 8
    k = args.k if args.k is not None else hidden_size // 32
    layer = args.layer
    stage = args.command
    stage_output_dir = str(Path(args.output_dir) / stage)
    ckpt_name = _make_checkpoint_name(
        args.model, layer, dict_size, args.target_tokens, stage,
    )

    tc = TrainingConfig(
        input_dim=hidden_size,
        dict_size=dict_size,
        k=k,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        drop_last=args.drop_last,
        output_dir=stage_output_dir,
        experiment_name=ckpt_name.replace(".pt", ""),
        use_swanlab=args.use_swanlab,
        device=args.device,
        dtype=args.dtype,
    )
    trainer = SAETrainer(tc)

    if stage == "stage2":
        s1_ckpt = args.stage1_checkpoint
        if s1_ckpt and Path(s1_ckpt).exists():
            print(f"Loading Stage 1 checkpoint: {s1_ckpt}")
            trainer.model = TopKSAE.load(s1_ckpt, device=args.device)
            trainer.optimizer = torch.optim.AdamW(
                trainer.model.parameters(), lr=tc.learning_rate,
            )
        elif s1_ckpt:
            print(f"Warning: Stage 1 checkpoint not found at {s1_ckpt}, "
                  "training from scratch")

    buffer_size = args.buffer_size if args.buffer_size is not None else args.batch_size
    total_steps = max(args.target_tokens // args.batch_size, 1)

    if stage == "stage2":
        from utils.when2call_adapter import create_stage2_data_iterator
        gen = create_stage2_data_iterator(
            model=model, tokenizer=tokenizer,
            data_dir=args.data_dir,
            layers=[layer],
            target_tokens=args.target_tokens,
            max_length=args.seq_length,
            inference_batch_size=args.inference_batch_size,
            buffer_size=buffer_size,
            device=args.device,
            log_dir=stage_output_dir,
        )
    else:
        pt_cfg = PretrainConfig(
            data_dir=args.data_dir,
            target_tokens=args.target_tokens,
            seq_length=args.seq_length,
            sample_position="all",
            positions_per_seq=args.seq_length,
        )
        gen = create_pretrain_data_iterator(
            model=model, tokenizer=tokenizer, config=pt_cfg,
            layers=[layer], batch_size=args.inference_batch_size,
            buffer_size=buffer_size, device=args.device,
        )

    trainer.train_streaming(gen, layer, total_steps)
    trainer.model._normalize_decoder()
    trainer.save_checkpoint(ckpt_name)
    print(f"\n{stage.capitalize()} complete!")
    print(f"  Checkpoint: {stage_output_dir}/{ckpt_name}")


if __name__ == "__main__":
    main()
