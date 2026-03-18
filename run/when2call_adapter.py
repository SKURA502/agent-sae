"""
When2Call 数据加载器 + Stage 2 SAE 训练数据迭代器

数据路径: data/raw/When2Call/data/
  train/when2call_train_pref.jsonl  (9K, 3K CALL + 6K NO_CALL)
  train/when2call_train_sft.jsonl   (15K, 全部 NO_CALL)
  test/when2call_test_mcq.jsonl     (3652条，仅用于 H1/H3 评测，排除出训练)

三种格式差异：
  pref  — tools/messages/chosen_response 字段；action boundary = messages（无末尾 assistant）
  sft   — tools/messages 字段；messages 末尾含 assistant 回复，需 strip
  mcq   — orig_tools/question/correct_answer 字段；无 messages

Stage 2 SAE 训练：pref + sft 全量，对每条样本 apply_chat_template 构建 action boundary
  prompt 文本，通过 ActivationStreamer 提取激活，yield {layer: [N, hidden]}。

特征发现：仅 pref（9K），调用方分类别取 E[f|CALL] − E[f|NO_CALL]。

评测（H1/H3）：加载 mcq，过滤 UNCERTAIN 得 2,590 条二类子集。
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Union

import torch
from tqdm import tqdm


# ─────────────────────────────── data structures ────────────────────────────────

class DecisionLabel(str, Enum):
    CALL = "call"
    NO_CALL = "no_call"
    UNCERTAIN = "uncertain"


@dataclass
class TaskSample:
    sample_id: str
    instruction: str
    context: Optional[str] = None
    tool_schemas: List[Dict[str, Any]] = field(default_factory=list)
    available_tools: List[str] = field(default_factory=list)
    label: DecisionLabel = DecisionLabel.UNCERTAIN
    expected_tool: Optional[str] = None
    expected_args: Optional[Dict[str, Any]] = None
    expected_response: Optional[str] = None
    source_dataset: str = ""
    difficulty: Optional[str] = None
    category: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "instruction": self.instruction,
            "context": self.context,
            "tool_schemas": self.tool_schemas,
            "available_tools": self.available_tools,
            "label": self.label.value,
            "expected_tool": self.expected_tool,
            "expected_args": self.expected_args,
            "expected_response": self.expected_response,
            "source_dataset": self.source_dataset,
            "difficulty": self.difficulty,
            "category": self.category,
            "metadata": self.metadata,
        }


# ─────────────────────────────── helpers ────────────────────────────────────────

def _has_toolcall_tag(text: str) -> bool:
    return bool(re.search(r"<TOOLCALL\b", text, re.IGNORECASE))


def _parse_tools(tools_raw: List[Any]) -> List[Dict]:
    """tools 字段可能是 dict 列表或 JSON 字符串列表，统一转为 dict 列表。"""
    result = []
    for t in tools_raw:
        if isinstance(t, dict):
            result.append(t)
        elif isinstance(t, str):
            try:
                parsed = json.loads(t)
                if isinstance(parsed, dict):
                    result.append(parsed)
            except (json.JSONDecodeError, ValueError):
                pass
    return result


def _strip_trailing_assistant(messages: List[Dict]) -> List[Dict]:
    """移除末尾的 assistant 消息（SFT 格式将完整回复包含在 messages 中）。"""
    msgs = list(messages)
    while msgs and msgs[-1].get("role") == "assistant":
        msgs.pop()
    return msgs


def _extract_last_user_message(messages: List[Dict]) -> str:
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    return part.get("text", "")
        return str(content) if content else ""
    return ""


# ─────────────────────────────── adapter ────────────────────────────────────────

class When2CallAdapter:
    """
    When2Call 数据集适配器。

    split 参数对应文件：
      "train_pref"  → when2call_train_pref.jsonl
      "train_sft"   → when2call_train_sft.jsonl
      "test_mcq"    → when2call_test_mcq.jsonl
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        split: str = "train_pref",
        num_samples: int = -1,
        seed: int = 42,
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.num_samples = num_samples
        self.seed = seed
        self._samples: List[TaskSample] = []
        self._loaded = False

    def load(self) -> "When2CallAdapter":
        if self._loaded:
            return self
        data_file = self._find_file()
        raw_data = self._read_jsonl(data_file)

        if self.num_samples > 0 and len(raw_data) > self.num_samples:
            import random
            random.seed(self.seed)
            raw_data = random.sample(raw_data, self.num_samples)

        self._samples = [self._convert(r, i) for i, r in enumerate(raw_data)]
        self._loaded = True

        call_n = sum(1 for s in self._samples if s.label == DecisionLabel.CALL)
        no_call_n = sum(1 for s in self._samples if s.label == DecisionLabel.NO_CALL)
        unc_n = sum(1 for s in self._samples if s.label == DecisionLabel.UNCERTAIN)
        print(f"Loaded {len(self._samples)} samples [{data_file.name}] "
              f"CALL={call_n} NO_CALL={no_call_n} UNCERTAIN={unc_n}")
        return self

    def _find_file(self) -> Path:
        candidates = [
            self.data_path / f"when2call_{self.split}.jsonl",
            self.data_path / f"{self.split}.jsonl",
            self.data_path / f"when2call_{self.split}.json",
            self.data_path / f"{self.split}.json",
        ]
        for c in candidates:
            if c.exists():
                return c
        available = sorted(p.name for p in self.data_path.iterdir()) if self.data_path.exists() else []
        raise FileNotFoundError(
            f"Cannot find split '{self.split}' in {self.data_path}.\n"
            f"Available files: {available}"
        )

    @staticmethod
    def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return samples

    def _infer_label(self, raw: Dict[str, Any]) -> DecisionLabel:
        # pref 格式: chosen_response 含 <TOOLCALL> → CALL，否则 NO_CALL
        chosen = raw.get("chosen_response") or raw.get("chosen")
        if chosen is not None:
            content = (
                chosen.get("content") or chosen.get("message") or ""
                if isinstance(chosen, dict) else str(chosen)
            )
            return DecisionLabel.CALL if _has_toolcall_tag(content) else DecisionLabel.NO_CALL

        # MCQ 格式: correct_answer 字段
        answer = raw.get("correct_answer") or raw.get("answer") or raw.get("label")
        if answer is not None:
            val = str(answer).lower().strip()
            if val in ("tool_call", "call", "yes", "1", "true"):
                return DecisionLabel.CALL
            if val in ("cannot_answer", "no_call", "no", "0", "false", "direct"):
                return DecisionLabel.NO_CALL
            if val == "request_for_info":
                return DecisionLabel.UNCERTAIN

        # SFT 格式兜底（无 chosen_response / answer 字段，全为 NO_CALL）
        if "sft" in self.split.lower():
            return DecisionLabel.NO_CALL

        return DecisionLabel.UNCERTAIN

    def _convert(self, raw: Dict[str, Any], idx: int) -> TaskSample:
        # ── tools（pref/sft 用 "tools"；mcq 用 "orig_tools"）──────────────
        tools_raw = raw.get("tools") or raw.get("orig_tools") or []
        if not isinstance(tools_raw, list):
            tools_raw = [tools_raw] if tools_raw else []
        tool_schemas = _parse_tools(tools_raw)
        available_tools = [t.get("name", "") for t in tool_schemas if t.get("name")]

        # ── messages / instruction ────────────────────────────────────────
        messages: List[Dict] = raw.get("messages") or []
        # SFT: messages 末尾含 assistant 回复，构建 action boundary 时需 strip
        boundary_messages = _strip_trailing_assistant(messages)
        instruction = _extract_last_user_message(boundary_messages)

        # MCQ 格式无 messages，instruction 在 "question" 字段
        if not instruction:
            instruction = (
                raw.get("question")
                or raw.get("instruction")
                or raw.get("query")
                or ""
            )

        # ── label ─────────────────────────────────────────────────────────
        label = self._infer_label(raw)

        # ── sample id ─────────────────────────────────────────────────────
        sample_id = (
            raw.get("uuid") or raw.get("id") or f"w2c_{self.split}_{idx:06d}"
        )

        return TaskSample(
            sample_id=sample_id,
            instruction=instruction,
            context=raw.get("context"),
            tool_schemas=tool_schemas,
            available_tools=available_tools,
            label=label,
            source_dataset="when2call",
            category=raw.get("category") or raw.get("source"),
            metadata={
                "split": self.split,
                # action boundary prompt 构建用
                "boundary_messages": boundary_messages,
                "original_tools_raw": tools_raw,
            },
        )

    # ── iteration helpers ────────────────────────────────────────────────────

    def __len__(self) -> int:
        if not self._loaded:
            self.load()
        return len(self._samples)

    def __iter__(self) -> Iterator[TaskSample]:
        if not self._loaded:
            self.load()
        return iter(self._samples)

    def __getitem__(self, idx: int) -> TaskSample:
        if not self._loaded:
            self.load()
        return self._samples[idx]

    def get_samples(self, label: Optional[DecisionLabel] = None) -> List[TaskSample]:
        if not self._loaded:
            self.load()
        if label is None:
            return list(self._samples)
        return [s for s in self._samples if s.label == label]


# ─────────────────────────────── Stage 2 data iterator ──────────────────────────

def create_stage2_data_iterator(
    model,
    tokenizer,
    data_dir: Union[str, Path],
    layers: List[int],
    target_tokens: int = 5_000_000,
    max_length: int = 2048,
    inference_batch_size: int = 16,
    buffer_size: int = 8192,
    device: str = "cuda",
) -> Generator[Dict[int, torch.Tensor], None, None]:
    """
    Stage 2 SAE 训练数据迭代器。

    加载 When2Call pref（9K）+ sft（15K），对每条样本用 tokenizer.apply_chat_template
    构建 action boundary 文本，通过 ActivationStreamer 提取残差流激活。

    接口与 create_pretrain_data_iterator 相同：yield {layer_idx: [N, hidden_size]}。

    Args:
        data_dir: 含 when2call_train_pref.jsonl 和 when2call_train_sft.jsonl 的目录
        layers: 要提取激活的层号列表（如 [24, 26]）
        target_tokens: 达到后停止（默认 5M，约覆盖全部训练数据一轮）
        max_length: chat template tokenize 时的截断长度
        inference_batch_size: LLM 推理 batch 大小
        buffer_size: 激活缓冲区大小（tokens）
        device: 推理设备
    """
    from sae.pretrain_data import ActivationStreamer, ActivationBuffer, PretrainConfig

    data_dir = Path(data_dir)

    pref = When2CallAdapter(data_dir, split="train_pref").load()
    sft = When2CallAdapter(data_dir, split="train_sft").load()
    all_samples = list(pref) + list(sft)
    print(f"Stage 2 total: {len(all_samples)} samples "
          f"(pref={len(pref)}, sft={len(sft)})")

    # ── build text prompts using tokenizer ───────────────────────────────
    def _sample_to_text(sample: TaskSample) -> Optional[str]:
        msgs = sample.metadata.get("boundary_messages") or []
        tools = sample.tool_schemas or []
        if not msgs:
            msgs = [{"role": "user", "content": sample.instruction}]
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                tools=tools if tools else None,
            )
        except Exception:
            try:
                return tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                return "\n".join(
                    f"{m.get('role', 'user')}: {m.get('content', '')}" for m in msgs
                )

    def text_iter() -> Iterator[str]:
        for s in all_samples:
            text = _sample_to_text(s)
            if text:
                yield text

    # ── stream activations through existing infrastructure ───────────────
    cfg = PretrainConfig(
        data_dir=str(data_dir),  # 不实际读文件，仅占位
        target_tokens=target_tokens,
        seq_length=max_length,
        sample_position="all",
        positions_per_seq=max_length,
    )

    streamer = ActivationStreamer(model, tokenizer, layers, device)
    act_buffer = ActivationBuffer(buffer_size=buffer_size, layers=layers)

    total_tokens = 0
    pbar = tqdm(total=target_tokens, desc="Stage 2 激活提取", unit="tok")

    try:
        for acts in streamer.stream_activations(text_iter(), cfg, batch_size=inference_batch_size):
            act_buffer.add(acts)
            if acts:
                first_layer = next(iter(acts))
                n = acts[first_layer].shape[0]
                total_tokens += n
                pbar.update(n)
            if act_buffer.is_ready():
                yield act_buffer.get_and_clear()
            if total_tokens >= target_tokens:
                break

        if act_buffer.current_size > 0:
            yield act_buffer.get_and_clear()
    finally:
        pbar.close()
        streamer.cleanup()
