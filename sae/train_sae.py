"""
Train SAE - 两阶段训练脚本

Stage 1: 通用预训练语料激活
Stage 2: Tool-use 任务激活

支持运行时推理流式训练；使用 SwanLab 记录指标。
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

from .sae_model import TopKSAE, SAEConfig


def _make_checkpoint_name(
    model_name: str, layer: int, dict_size: int,
    target_tokens: int, stage: str,
) -> str:
    """格式: {LLM}-layer{L}-d{dict_size}-{tokens}M-{stage}.pt"""
    short = model_name.rstrip("/").split("/")[-1]
    tok = f"{target_tokens / 1e6:.0f}M"
    return f"{short}-L{layer}-d{int(dict_size)}-{tok}-{stage}.pt"


@dataclass
class TrainingConfig:
    """训练配置"""
    input_dim: int = 4096
    dict_size: int = 32768
    k: int = 128
    learning_rate: float = 1e-5
    batch_size: int = 4096
    num_epochs: int = 1
    warmup_ratio: float = 0.1
    decoder_norm_interval: int = 10
    log_interval: int = 10
    output_dir: str = "./outputs/sae_checkpoints"
    experiment_name: str = "sae_training"
    use_swanlab: bool = False
    swanlab_project: str = "agent-tool-use"
    device: str = "cuda"
    dtype: str = "float32"

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

        loss, loss_dict = self.model.compute_loss(batch)
        loss.backward()
        self.optimizer.step()
        scheduler.step()
        self.global_step += 1

        if self.global_step % self.config.decoder_norm_interval == 0:
            self.model._normalize_decoder()

        return {
            "loss": float(loss.item()),
            "reconstruction_loss": float(loss_dict["loss"]),
            "mean_activation": float(loss_dict["mean_activation"]),
            "lr": float(scheduler.get_last_lr()[0]),
            "global_step": float(self.global_step),
        }

    def train(self, train_data: torch.Tensor) -> Dict[str, Any]:
        """在给定张量上训练 SAE。"""
        print(f"Training SAE: {len(train_data)} samples, "
              f"dict_size={self.config.dict_size}, k={self.config.k}")

        loader = DataLoader(
            TensorDataset(train_data),
            batch_size=self.config.batch_size,
            shuffle=False, num_workers=0, pin_memory=True,
        )
        total_steps = len(loader) * self.config.num_epochs
        warmup = int(total_steps * self.config.warmup_ratio)
        scheduler = self._make_scheduler(total_steps, warmup)

        stats: Dict[str, list] = {"train_losses": []}
        try:
            for epoch in range(self.config.num_epochs):
                self.model.train()
                running, n = 0.0, 0
                pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
                for (batch,) in pbar:
                    m = self.train_batch(batch, scheduler)
                    running += m["loss"]
                    n += 1
                    pbar.set_postfix(loss=f"{m['loss']:.4f}")
                    self._maybe_log(m)
                avg = running / max(n, 1)
                stats["train_losses"].append(avg)
                print(f"Epoch {epoch+1}/{self.config.num_epochs}: loss={avg:.4f}")

            self._save_stats(stats)
            return stats
        finally:
            self._finish_swanlab()

    def train_streaming(
        self,
        activation_generator: Generator[Dict[int, torch.Tensor], None, None],
        layer: int,
        total_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """流式训练：从激活生成器逐批读取并训练。"""
        print(f"Streaming training for layer {layer}, "
              f"dict_size={self.config.dict_size}, k={self.config.k}")

        total_steps = total_steps or 10000
        warmup = int(total_steps * self.config.warmup_ratio)
        scheduler = self._make_scheduler(total_steps, warmup)

        stats: Dict[str, list] = {"interval_avg_losses": [], "steps": []}
        self.model.train()
        running, n = 0.0, 0
        pbar = tqdm(total=total_steps, desc="Streaming training")

        try:
            for activations in activation_generator:
                if layer not in activations:
                    continue
                data = activations[layer]
                if data.dim() == 3:
                    data = data.view(-1, data.shape[-1])

                for i in range(0, len(data), self.config.batch_size):
                    m = self.train_batch(
                        data[i:i + self.config.batch_size], scheduler,
                    )
                    running += m["loss"]
                    n += 1
                    pbar.update(1)
                    pbar.set_postfix(loss=f"{m['loss']:.4f}")

                    if (self.global_step % self.config.log_interval == 0
                            and n > 0):
                        avg = running / n
                        stats["interval_avg_losses"].append(avg)
                        stats["steps"].append(self.global_step)
                        self._log_metrics(avg, m)
                        running, n = 0.0, 0

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
    def _make_scheduler(self, total: int, warmup: int):
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            return max(0.0, (total - step) / max(1, total - warmup))

        return LambdaLR(self.optimizer, lr_lambda)

    def _maybe_log(self, metrics: Dict[str, float]):
        """Epoch 训练中按 interval 打日志。"""
        if self.global_step % self.config.log_interval != 0:
            return
        log_data = {
            "train/loss": metrics["loss"],
            "train/reconstruction_loss": metrics["reconstruction_loss"],
            "train/mean_activation": metrics["mean_activation"],
            "train/lr": metrics["lr"],
            "global_step": self.global_step,
        }
        if self._swanlab_active and SWANLAB_AVAILABLE:
            swanlab.log(log_data)
        else:
            tqdm.write(
                f"[Step {self.global_step}] loss={metrics['loss']:.4f} "
                f"recon={metrics['reconstruction_loss']:.4f} "
                f"mean_act={metrics['mean_activation']:.4f} "
                f"lr={metrics['lr']:.2e}"
            )

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
            swanlab.finish()
            self._swanlab_active = False


class TwoStageTrainer:
    """两阶段 SAE 训练器（Stage 1: 预训练语料 / Stage 2: Tool-use）"""

    def __init__(
        self, model_name_or_path: str, layer: int,
        output_dir: str = "./outputs/sae_checkpoints",
        device: str = "cuda", dtype: str = "float32",
    ):
        self.model_name_or_path = model_name_or_path
        self.layer = int(layer)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.dtype = dtype
        self.model = None
        self.tokenizer = None
        self.hidden_size: Optional[int] = None
        self.sae_trainer: Optional[SAETrainer] = None

    # LLM loading
    def _load_llm(self):
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading LLM: {self.model_name_or_path}")
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=dtype_map.get(self.dtype, torch.bfloat16),
            device_map=self.device, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 推断 hidden_size（兼容多模态模型，如 Gemma3）
        cfg = self.model.config
        text_cfg = getattr(cfg, "text_config", None)
        hs = (getattr(text_cfg, "hidden_size", None) if text_cfg else None) \
            or getattr(cfg, "hidden_size", None)
        if hs is None:
            raise ValueError(f"无法从 {type(cfg)} 推断 hidden_size")
        self.hidden_size = int(hs)
        print(f"LLM loaded. Hidden size: {self.hidden_size}")

    # Helpers
    def _resolve_dict_k(self, overrides: Dict[str, Any], sae_model=None):
        """从 overrides / checkpoint / 默认值推断 dict_size 与 k。"""
        req_d, req_k = overrides.get("dict_size"), overrides.get("k")
        if sae_model is not None:
            d, k = sae_model.config.dict_size, sae_model.config.k
            if req_d is not None and int(req_d) != int(d):
                print(f"Warning: checkpoint dict_size={d}, "
                      f"ignoring --dict-size={req_d}")
            if req_k is not None and int(req_k) != int(k):
                print(f"Warning: checkpoint k={k}, ignoring --k={req_k}")
            return int(d), int(k)
        d = int(req_d) if req_d is not None else self.hidden_size * 8
        k = int(req_k) if req_k is not None else self.hidden_size // 32
        return d, k

    def _make_train_config(
        self, dict_size: int, k: int, cfg: Dict[str, Any],
        stage: str, ckpt_name: str,
    ) -> TrainingConfig:
        default_lr = 1e-5 if stage == "stage1" else 5e-5
        return TrainingConfig(
            input_dim=self.hidden_size, dict_size=dict_size, k=k,
            learning_rate=cfg.get("learning_rate", default_lr),
            batch_size=cfg.get("batch_size", 4096),
            num_epochs=cfg.get("num_epochs", 1),
            decoder_norm_interval=cfg.get("decoder_norm_interval", 10),
            log_interval=cfg.get("log_interval", 1),
            output_dir=str(self.output_dir / stage),
            experiment_name=ckpt_name.replace(".pt", ""),
            use_swanlab=cfg.get("use_swanlab", False),
            swanlab_project=cfg.get("swanlab_project", "agent-tool-use"),
            device=self.device, dtype=self.dtype,
        )

    def train_stage1(
        self,
        pretrain_config: Optional[Dict[str, Any]] = None,
        sae_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[int, str]:
        """Stage 1: 通用预训练语料训练，返回 {layer: checkpoint_path}。"""
        from .pretrain_data import PretrainConfig, create_pretrain_data_iterator

        self._load_llm()
        pc, sc = pretrain_config or {}, sae_config or {}

        seq_length = pc.get("seq_length", 1024)
        sample_pos = pc.get("sample_position", "all")
        positions_per_seq = (
            seq_length if sample_pos == "all"
            else pc.get("positions_per_seq", 8)
        )
        target_tokens = pc.get("target_tokens", 50_000_000)

        pt_cfg = PretrainConfig(
            data_dir=pc.get(
                "data_dir", str(_PROJECT_ROOT / "data" / "raw" / "100M"),
            ),
            target_tokens=target_tokens, seq_length=seq_length,
            sample_position=sample_pos, positions_per_seq=positions_per_seq,
        )

        dict_size, k = self._resolve_dict_k(sc)
        layer = self.layer
        ckpt_name = _make_checkpoint_name(
            self.model_name_or_path, layer, dict_size, target_tokens, "stage1",
        )

        print(f"Stage 1: layer {layer}, tokens={target_tokens:,}, "
              f"dict_size={dict_size}, k={k}")

        tc = self._make_train_config(dict_size, k, sc, "stage1", ckpt_name)
        trainer = SAETrainer(tc)
        self.sae_trainer = trainer

        est = (target_tokens if sample_pos == "all"
               else target_tokens // seq_length * positions_per_seq)
        total_steps = max(est // tc.batch_size, 1)

        gen = create_pretrain_data_iterator(
            model=self.model, tokenizer=self.tokenizer, config=pt_cfg,
            layers=[layer], batch_size=pc.get("batch_size", 32),
            buffer_size=sc.get("buffer_size", 8192), device=self.device,
        )
        trainer.train_streaming(gen, layer, total_steps)
        trainer.save_checkpoint(ckpt_name)
        print("Stage 1 complete!")
        return {layer: str(self.output_dir / "stage1" / ckpt_name)}

    def init_stage2(
        self,
        stage1_checkpoint: Optional[str],
        tooluse_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[int, str]:
        """Stage 2 初始化：加载 checkpoint、创建 trainer，返回预期路径。"""
        self._load_llm()
        cfg = tooluse_config or {}
        target_tokens = cfg.get("target_tokens", 50_000_000)
        layer = self.layer

        sae_model = None
        if stage1_checkpoint and Path(stage1_checkpoint).exists():
            print(f"Loading Stage 1 checkpoint: {stage1_checkpoint}")
            sae_model = TopKSAE.load(stage1_checkpoint, device=self.device)
        else:
            print(f"Warning: No Stage 1 checkpoint for layer {layer}, "
                  "training from scratch")

        dict_size, k = self._resolve_dict_k(cfg, sae_model)
        ckpt_name = _make_checkpoint_name(
            self.model_name_or_path, layer, dict_size, target_tokens, "stage2",
        )

        print(f"Stage 2: layer {layer}, dict_size={dict_size}, k={k}")

        tc = self._make_train_config(dict_size, k, cfg, "stage2", ckpt_name)
        trainer = SAETrainer(tc)

        if sae_model is not None:
            trainer.model = sae_model
            trainer.optimizer = torch.optim.AdamW(
                trainer.model.parameters(), lr=tc.learning_rate,
            )

        self.sae_trainer = trainer
        return {layer: str(self.output_dir / "stage2" / ckpt_name)}

    def train_stage2(
        self,
        stage1_checkpoint: Optional[str],
        tooluse_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[int, str]:
        """兼容旧接口：等价于 init_stage2。"""
        return self.init_stage2(stage1_checkpoint, tooluse_config)

    def train_stage2_streaming(
        self,
        stage1_checkpoint: Optional[str],
        activation_generator: Generator[Dict[int, torch.Tensor], None, None],
        total_steps: Optional[int] = None,
        tooluse_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[int, str]:
        """Stage 2 流式训练：init + stream + save。"""
        paths = self.init_stage2(stage1_checkpoint, tooluse_config)
        cfg = tooluse_config or {}
        target_tokens = cfg.get("target_tokens", 50_000_000)

        if self.sae_trainer is None:
            raise RuntimeError("Stage2 trainer not initialized")

        trainer = self.sae_trainer
        trainer.train_streaming(activation_generator, self.layer, total_steps)

        ckpt = _make_checkpoint_name(
            self.model_name_or_path, self.layer,
            trainer.config.dict_size, target_tokens, "stage2",
        )
        trainer.save_checkpoint(ckpt)
        return paths


def _add_stage_args(parser: argparse.ArgumentParser):
    """stage1 / stage2 共享参数。"""
    from utils import add_common_args, add_sae_args
    add_common_args(parser)
    add_sae_args(parser)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--output-dir", type=str,
                        default="./outputs/sae_checkpoints")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="Train SAE (Two-Stage)")
    sub = parser.add_subparsers(dest="command")

    s1 = sub.add_parser("stage1", help="Stage 1: pretrain corpus")
    _add_stage_args(s1)
    s1.add_argument("--seq-length", type=int, default=512)
    s1.add_argument("--inference-batch-size", type=int, default=64,
                    help="LLM inference batch size")
    s1.add_argument("--data-dir", type=str,
                    default=str(_PROJECT_ROOT / "data" / "raw" / "100M"))

    s2 = sub.add_parser("stage2", help="Stage 2: tool-use data")
    _add_stage_args(s2)
    s2.add_argument("--stage1-dir", type=str, required=True,
                    help="Stage 1 checkpoint directory")

    args = parser.parse_args()

    if args.command == "stage1":
        trainer = TwoStageTrainer(
            model_name_or_path=args.model, layer=args.layer,
            output_dir=args.output_dir, device=args.device, dtype=args.dtype,
        )
        ckpts = trainer.train_stage1(
            pretrain_config={
                "data_dir": args.data_dir,
                "target_tokens": args.target_tokens,
                "seq_length": args.seq_length,
                "batch_size": args.inference_batch_size,
            },
            sae_config={
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "dict_size": args.dict_size,
                "k": args.k,
                "use_swanlab": args.use_swanlab,
            },
        )
        print("\nStage 1 checkpoints:")
        for layer, path in ckpts.items():
            print(f"  Layer {layer}: {path}")

    elif args.command == "stage2":
        stage1_dir = Path(args.stage1_dir)
        matches = list(stage1_dir.glob(f"*-layer{args.layer}-*-stage1.pt"))
        s1_ckpt = str(matches[0]) if matches else None
        if not s1_ckpt:
            print(f"Warning: Stage 1 checkpoint not found for layer "
                  f"{args.layer}")

        trainer = TwoStageTrainer(
            model_name_or_path=args.model, layer=args.layer,
            output_dir=args.output_dir, device=args.device, dtype=args.dtype,
        )
        ckpts = trainer.init_stage2(s1_ckpt, {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "dict_size": args.dict_size,
            "k": args.k,
            "target_tokens": args.target_tokens,
            "use_swanlab": args.use_swanlab,
        })
        print("\nStage 2 initialized! Use streaming API for tool-use data.")
        for layer, path in ckpts.items():
            print(f"  Layer {layer}: {path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
