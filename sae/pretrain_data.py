"""预训练语料数据加载器 — 支持本地 JSONL，运行时推理提取 hidden states 用于 SAE 训练。"""

import glob, json, os, random
from dataclasses import dataclass
from typing import Dict, Generator, Iterator, List, Optional

import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm


@dataclass
class PretrainConfig:
    """预训练数据配置"""
    data_dir: str = "/data/agent_tool_use/Agent-Tool-Use-MI/data/raw/pretrain"
    target_tokens: int = 50_000_000
    seq_length: int = 1024
    sample_position: str = "all"   # random / last / all
    positions_per_seq: int = 1
    seed: int = 42


class ActivationStreamer:
    """从 LLM 流式提取激活，不落盘，直接供 SAE 训练。"""

    def __init__(self, model, tokenizer, layers: List[int], device: str = "cuda"):
        self.model, self.tokenizer = model, tokenizer
        self.layers, self.device = layers, device
        self._activations: Dict[int, torch.Tensor] = {}
        self._hooks: list = []
        self._layer_container = None

    # ---------- 模型层解析 ----------
    @staticmethod
    def _get_attr_by_path(obj, path: str):
        for attr in path.split("."):
            if not hasattr(obj, attr):
                return None
            obj = getattr(obj, attr)
        return obj

    def _resolve_layer_container(self):
        if self._layer_container is not None:
            return self._layer_container
        for path in ("model.layers", "language_model.model.layers"):
            c = self._get_attr_by_path(self.model, path)
            if isinstance(c, (list, torch.nn.ModuleList)) and len(c) > 0:
                self._layer_container = c
                print(f"检测到层容器: {path} (num_layers={len(c)})")
                return c
        raise AttributeError("无法找到模型的 layer 结构，请检查模型架构。")

    # ---------- Hook 管理 ----------
    def _create_hook(self, layer_idx: int):
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            self._activations[layer_idx] = h.detach()
        return hook

    def _register_hooks(self):
        self._remove_hooks()
        layers = self._resolve_layer_container()
        for idx in self.layers:
            try:
                self._hooks.append(layers[idx].register_forward_hook(self._create_hook(idx)))
            except (AttributeError, IndexError) as e:
                print(f"Warning: 无法为层 {idx} 注册 hook: {e}")

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks, self._activations = [], {}

    def cleanup(self):
        """移除所有 hooks，释放资源。"""
        self._remove_hooks()

    def __del__(self):
        self._remove_hooks()

    # ---------- 激活提取 ----------
    @torch.no_grad()
    def extract_activations(
        self, input_ids: torch.Tensor, positions: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """提取指定位置的激活 → {layer: [batch, positions, hidden]}"""
        self._activations = {}
        if not self._hooks:
            self._register_hooks()
        self.model(input_ids.to(self.device))
        return {
            idx: (acts[:, positions, :] if positions else acts)
            for idx, acts in self._activations.items()
        }

    # ---------- 流式提取 ----------
    def _process_batch(self, batch_texts: List[str], config: PretrainConfig):
        """对一个 batch 做 tokenize → 提取激活 → 过滤，返回 (activations, n_tokens)。"""
        enc = self.tokenizer(
            batch_texts, max_length=config.seq_length,
            truncation=True, padding="max_length", return_tensors="pt",
        )
        input_ids, attn_mask = enc["input_ids"], enc["attention_mask"]
        positions = self._get_sample_positions(attn_mask, config.sample_position, config.positions_per_seq)
        acts = self.extract_activations(input_ids, positions)
        n_tokens = int(attn_mask.sum().item())
        acts = self._filter_and_flatten(acts, input_ids.cpu(), positions)
        return acts, n_tokens

    def stream_activations(
        self, texts: Iterator[str], config: PretrainConfig, batch_size: int = 32,
    ) -> Generator[Dict[int, torch.Tensor], None, None]:
        """流式提取激活 → yields {layer: [N, hidden]}"""
        batch_texts: List[str] = []
        total_tokens = 0
        for text in texts:
            if total_tokens >= config.target_tokens:
                break
            batch_texts.append(text)
            if len(batch_texts) >= batch_size:
                acts, n = self._process_batch(batch_texts, config)
                total_tokens += n
                if acts:
                    yield acts
                batch_texts = []
        if batch_texts:
            acts, _ = self._process_batch(batch_texts, config)
            if acts:
                yield acts

    # ---------- 辅助方法 ----------
    def _filter_and_flatten(
        self, activations: Dict[int, torch.Tensor],
        input_ids_cpu: torch.Tensor, positions: Optional[List[int]],
    ) -> Dict[int, torch.Tensor]:
        """过滤特殊 token 并展平激活（向量化）。"""
        sp_ids = {
            getattr(self.tokenizer, a)
            for a in ('pad_token_id', 'bos_token_id', 'eos_token_id')
            if getattr(self.tokenizer, a, None) is not None
        }
        sampled = input_ids_cpu[:, positions] if positions else input_ids_cpu
        sp_tensor = torch.tensor(list(sp_ids), dtype=sampled.dtype)
        valid_flat = ~torch.isin(sampled, sp_tensor).view(-1)

        result = {}
        for idx, acts in activations.items():
            if acts.ndim == 3:
                mask = valid_flat.to(acts.device)
                filtered = acts.view(-1, acts.shape[-1])[mask]
                if filtered.size(0) > 0:
                    result[idx] = filtered
            else:
                result[idx] = acts
        return result

    def _get_sample_positions(
        self, attention_mask: torch.Tensor, strategy: str, n: int,
    ) -> Optional[List[int]]:
        if strategy == "all":
            return None
        seq_len = attention_mask.shape[1]
        if strategy == "last":
            return [seq_len - 1]
        if strategy == "random":
            return sorted(random.sample(range(seq_len), min(n, seq_len)))
        step = seq_len // n
        return [i * step for i in range(n)]


#  本地 JSONL 数据集
class LocalJsonlDataset(IterableDataset):
    """读取目录下所有 .jsonl，每行解析为一条文本。"""

    def __init__(self, data_dir: str, text_key: str = "text", seed: int = 42):
        self.data_dir, self.text_key, self.seed = data_dir, text_key, seed
        self._files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
        if not self._files:
            raise FileNotFoundError(f"在目录 {data_dir} 中未找到任何 .jsonl 文件")
        print(f"发现 {len(self._files)} 个 jsonl 文件，来自: {data_dir}")

    def __iter__(self) -> Iterator[str]:
        files = list(self._files)
        random.Random(self.seed).shuffle(files)
        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = item.get(self.text_key) or item.get("content") or item.get("body")
                    if text and isinstance(text, str) and len(text) > 10:
                        yield text


#  激活缓冲区
class ActivationBuffer:
    """累积激活数据，达到 buffer_size 后供 SAE 训练。"""

    def __init__(self, buffer_size: int = 8192, layers: Optional[List[int]] = None):
        self.buffer_size, self.layers = buffer_size, layers
        self._buffers: Dict[int, List[torch.Tensor]] = {}
        self._current_size = 0

    def add(self, activations: Dict[int, torch.Tensor]):
        for idx, acts in activations.items():
            if self.layers is not None and idx not in self.layers:
                continue
            self._buffers.setdefault(idx, []).append(acts)
        if self._buffers:
            first = next(iter(self._buffers.values()))
            self._current_size = sum(t.shape[0] for t in first)

    def is_ready(self) -> bool:
        return self._current_size >= self.buffer_size

    def get_and_clear(self) -> Dict[int, torch.Tensor]:
        result = {k: torch.cat(v) for k, v in self._buffers.items() if v}
        self._buffers, self._current_size = {}, 0
        return result

    @property
    def current_size(self) -> int:
        return self._current_size


#  便捷入口
def create_pretrain_data_iterator(
    model, tokenizer, config: PretrainConfig, layers: List[int],
    batch_size: int = 32, buffer_size: int = 8192, device: str = "cuda",
) -> Generator[Dict[int, torch.Tensor], None, None]:
    """组合数据加载 + 激活提取 + 缓冲 → yields {layer: [buffer_size, hidden]}"""
    if buffer_size is None:
        buffer_size = 8192
    if buffer_size <= 0:
        raise ValueError(f"buffer_size 必须为正整数，当前为: {buffer_size}")

    dataset = LocalJsonlDataset(data_dir=config.data_dir, seed=config.seed)
    streamer = ActivationStreamer(model, tokenizer, layers, device)
    buffer = ActivationBuffer(buffer_size, layers)

    total_tokens = 0
    pbar = tqdm(total=config.target_tokens, desc="提取预训练激活")

    for activations in streamer.stream_activations(iter(dataset), config, batch_size):
        buffer.add(activations)
        if activations:
            first_layer = next(iter(activations))
            n = activations[first_layer].shape[0] * config.seq_length // config.positions_per_seq
            total_tokens += n
            pbar.update(n)
        if buffer.is_ready():
            yield buffer.get_and_clear()
        if total_tokens >= config.target_tokens:
            break

    if buffer.current_size > 0:
        yield buffer.get_and_clear()
    pbar.close()
