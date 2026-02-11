"""
Pretrain Data - 通用预训练语料数据加载器

支持 OpenWebText2 等通用预训练语料，用于 SAE 第一阶段训练。
实现运行时推理，避免保存 hidden states 到磁盘。
"""

import random
from dataclasses import dataclass
from typing import Generator, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


@dataclass
class PretrainConfig:
    """预训练数据配置"""
    # 数据集名称
    dataset_name: str = "Skylion007/openwebtext"
    # 数据集子集（如果有）
    dataset_subset: Optional[str] = None
    # 目标 token 数
    target_tokens: int = 50_000_000
    # 序列长度
    seq_length: int = 1024
    # 采样位置策略：random, last, all
    sample_position: str = "random"
    # 每个序列采样的位置数
    positions_per_seq: int = 1
    # 随机种子
    seed: int = 42


class ActivationStreamer:
    """激活流式提取器
    
    从 LLM 中流式提取激活，不保存到磁盘，直接用于 SAE 训练。
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        layers: List[int],
        device: str = "cuda",
    ):
        """
        Args:
            model: HuggingFace 模型
            tokenizer: Tokenizer
            layers: 要提取激活的层索引
            device: 设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers
        self.device = device
        
        # 激活缓存
        self._activations = {}
        self._hooks = []
        
    def _create_hook(self, layer_idx: int):
        """创建 hook 函数"""
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            # 存储激活（detach 并移到 CPU 节省 GPU 内存）
            self._activations[layer_idx] = hidden_states.detach()
        return hook
    
    def _register_hooks(self):
        """注册 hooks"""
        self._remove_hooks()
        
        for layer_idx in self.layers:
            try:
                # 尝试不同的模型结构
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                    layer = self.model.model.layers[layer_idx]
                elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                    layer = self.model.transformer.h[layer_idx]
                elif hasattr(self.model, 'gpt_neox') and hasattr(self.model.gpt_neox, 'layers'):
                    layer = self.model.gpt_neox.layers[layer_idx]
                else:
                    raise AttributeError(f"无法找到模型的 layer 结构")
                    
                hook = layer.register_forward_hook(self._create_hook(layer_idx))
                self._hooks.append(hook)
            except (AttributeError, IndexError) as e:
                print(f"Warning: 无法为层 {layer_idx} 注册 hook: {e}")
    
    def _remove_hooks(self):
        """移除 hooks"""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._activations = {}
    
    @torch.no_grad()
    def extract_activations(
        self,
        input_ids: torch.Tensor,
        positions: Optional[List[int]] = None,
    ) -> dict[int, torch.Tensor]:
        """提取指定位置的激活
        
        Args:
            input_ids: [batch, seq_len] 输入 token ids
            positions: 要提取的位置列表，None 表示所有位置
            
        Returns:
            {layer_idx: [batch, num_positions, hidden_dim]} 激活字典
        """
        self._activations = {}
        self._register_hooks()
        
        try:
            input_ids = input_ids.to(self.device)
            
            # 前向传播
            _ = self.model(input_ids)
            
            # 提取指定位置的激活
            result = {}
            for layer_idx, acts in self._activations.items():
                if positions is not None:
                    result[layer_idx] = acts[:, positions, :].cpu()
                else:
                    result[layer_idx] = acts.cpu()
            
            return result
            
        finally:
            self._remove_hooks()
    
    def stream_activations(
        self,
        texts: Iterator[str],
        config: PretrainConfig,
        batch_size: int = 16,
    ) -> Generator[dict[int, torch.Tensor], None, None]:
        """流式提取激活
        
        Args:
            texts: 文本迭代器
            config: 预训练配置
            batch_size: 批大小
            
        Yields:
            {layer_idx: [batch * positions_per_seq, hidden_dim]} 激活
        """
        batch_texts = []
        total_tokens = 0
        
        for text in texts:
            if total_tokens >= config.target_tokens:
                break
                
            batch_texts.append(text)
            
            if len(batch_texts) >= batch_size:
                # Tokenize
                encodings = self.tokenizer(
                    batch_texts,
                    max_length=config.seq_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                
                input_ids = encodings["input_ids"]
                attention_mask = encodings["attention_mask"]
                
                # 确定采样位置
                positions = self._get_sample_positions(
                    attention_mask,
                    config.sample_position,
                    config.positions_per_seq,
                )
                
                # 提取激活
                activations = self.extract_activations(input_ids, positions)
                
                # 统计 token 数
                total_tokens += attention_mask.sum().item()
                
                # 重塑激活：[batch, positions, hidden] -> [batch * positions, hidden]
                for layer_idx in activations:
                    acts = activations[layer_idx]
                    if len(acts.shape) == 3:
                        activations[layer_idx] = acts.view(-1, acts.shape[-1])
                
                yield activations
                
                batch_texts = []
        
        # 处理剩余的 batch
        if batch_texts:
            encodings = self.tokenizer(
                batch_texts,
                max_length=config.seq_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            
            input_ids = encodings["input_ids"]
            attention_mask = encodings["attention_mask"]
            
            positions = self._get_sample_positions(
                attention_mask,
                config.sample_position,
                config.positions_per_seq,
            )
            
            activations = self.extract_activations(input_ids, positions)
            
            for layer_idx in activations:
                acts = activations[layer_idx]
                if len(acts.shape) == 3:
                    activations[layer_idx] = acts.view(-1, acts.shape[-1])
            
            yield activations
    
    def _get_sample_positions(
        self,
        attention_mask: torch.Tensor,
        strategy: str,
        positions_per_seq: int,
    ) -> List[int]:
        """获取采样位置
        
        Args:
            attention_mask: [batch, seq_len]
            strategy: 采样策略
            positions_per_seq: 每个序列采样的位置数
            
        Returns:
            位置列表
        """
        seq_len = attention_mask.shape[1]
        
        if strategy == "last":
            # 只取最后一个位置
            return [seq_len - 1]
        
        elif strategy == "random":
            # 随机采样位置
            positions = random.sample(range(seq_len), min(positions_per_seq, seq_len))
            return sorted(positions)
        
        elif strategy == "all":
            # 所有位置
            return list(range(seq_len))
        
        else:
            # 默认：均匀分布采样
            step = seq_len // positions_per_seq
            return [i * step for i in range(positions_per_seq)]


class OpenWebTextDataset(IterableDataset):
    """OpenWebText 数据集迭代器"""
    
    def __init__(
        self,
        config: PretrainConfig,
        split: str = "train",
    ):
        """
        Args:
            config: 预训练配置
            split: 数据集分割
        """
        if not DATASETS_AVAILABLE:
            raise ImportError("请安装 datasets: pip install datasets")
        
        self.config = config
        self.split = split
        self._dataset = None
    
    def _load_dataset(self):
        """延迟加载数据集"""
        if self._dataset is None:
            print(f"加载数据集: {self.config.dataset_name}")
            self._dataset = load_dataset(
                self.config.dataset_name,
                self.config.dataset_subset,
                split=self.split,
                streaming=True,  # 使用流式加载，节省内存
            )
            
            # 设置随机种子并 shuffle
            self._dataset = self._dataset.shuffle(seed=self.config.seed)
    
    def __iter__(self) -> Iterator[str]:
        """迭代返回文本"""
        self._load_dataset()
        
        for item in self._dataset:
            # OpenWebText 的文本字段是 "text"
            if "text" in item:
                yield item["text"]
            elif "content" in item:
                yield item["content"]
            else:
                # 尝试获取第一个字符串字段
                for key, value in item.items():
                    if isinstance(value, str) and len(value) > 100:
                        yield value
                        break


class PretrainActivationBuffer:
    """预训练激活缓冲区
    
    用于累积激活数据，达到一定数量后进行 SAE 训练。
    """
    
    def __init__(
        self,
        buffer_size: int = 8192,
        layers: Optional[List[int]] = None,
    ):
        """
        Args:
            buffer_size: 缓冲区大小
            layers: 层索引列表
        """
        self.buffer_size = buffer_size
        self.layers = layers
        self._buffers: dict[int, List[torch.Tensor]] = {}
        self._current_size = 0
    
    def add(self, activations: dict[int, torch.Tensor]):
        """添加激活到缓冲区
        
        Args:
            activations: {layer_idx: [batch, hidden_dim]} 激活
        """
        for layer_idx, acts in activations.items():
            if self.layers is not None and layer_idx not in self.layers:
                continue
            
            if layer_idx not in self._buffers:
                self._buffers[layer_idx] = []
            
            self._buffers[layer_idx].append(acts)
        
        # 更新大小（使用第一个层的数据）
        if self._buffers:
            first_layer = list(self._buffers.keys())[0]
            self._current_size = sum(t.shape[0] for t in self._buffers[first_layer])
    
    def is_ready(self) -> bool:
        """检查缓冲区是否达到指定大小"""
        return self._current_size >= self.buffer_size
    
    def get_and_clear(self) -> dict[int, torch.Tensor]:
        """获取并清空缓冲区
        
        Returns:
            {layer_idx: [total_samples, hidden_dim]} 合并的激活
        """
        result = {}
        for layer_idx, acts_list in self._buffers.items():
            if acts_list:
                result[layer_idx] = torch.cat(acts_list, dim=0)
        
        self._buffers = {}
        self._current_size = 0
        
        return result
    
    @property
    def current_size(self) -> int:
        """当前缓冲区大小"""
        return self._current_size


def create_pretrain_data_iterator(
    model,
    tokenizer,
    config: PretrainConfig,
    layers: List[int],
    batch_size: int = 16,
    buffer_size: int = 8192,
    device: str = "cuda",
) -> Generator[dict[int, torch.Tensor], None, None]:
    """创建预训练数据迭代器
    
    这是一个便捷函数，组合了数据加载、激活提取和缓冲。
    
    Args:
        model: LLM 模型
        tokenizer: Tokenizer
        config: 预训练数据配置
        layers: 要提取的层
        batch_size: 推理批大小
        buffer_size: 激活缓冲区大小
        device: 设备
        
    Yields:
        {layer_idx: [buffer_size, hidden_dim]} 激活
    """
    # 创建数据集
    dataset = OpenWebTextDataset(config)
    
    # 创建激活提取器
    streamer = ActivationStreamer(model, tokenizer, layers, device)
    
    # 创建缓冲区
    buffer = PretrainActivationBuffer(buffer_size, layers)
    
    # 流式提取激活
    total_tokens = 0
    pbar = tqdm(total=config.target_tokens, desc="提取预训练激活")
    
    for activations in streamer.stream_activations(iter(dataset), config, batch_size):
        buffer.add(activations)
        
        # 更新进度
        if activations:
            first_layer = list(activations.keys())[0]
            batch_tokens = activations[first_layer].shape[0] * config.seq_length // config.positions_per_seq
            total_tokens += batch_tokens
            pbar.update(batch_tokens)
        
        # 缓冲区满了，yield 数据
        if buffer.is_ready():
            yield buffer.get_and_clear()
        
        # 达到目标 token 数
        if total_tokens >= config.target_tokens:
            break
    
    # 清空剩余的缓冲区
    if buffer.current_size > 0:
        yield buffer.get_and_clear()
    
    pbar.close()
