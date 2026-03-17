"""
When2Call Adapter - When2Call 数据集适配器

支持三种格式：
- pref: 偏好数据（when2call_train_pref.jsonl），从 chosen_response 解析 <TOOLCALL> 标签
- sft:  SFT 数据（when2call_train_sft.jsonl），全为 NO_CALL
- mcq:  测试 MCQ 数据（when2call_test_mcq.jsonl），从 answer 字段读取类型
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_adapter import BaseAdapter, TaskSample, DecisionLabel


def _has_toolcall_tag(text: str) -> bool:
    """检查文本是否包含 <TOOLCALL> 标签。"""
    return bool(re.search(r"<TOOLCALL\b", text, re.IGNORECASE))


def _extract_last_user_message(messages: List[Dict]) -> str:
    """从消息列表中提取最后一条 user 消息的文本内容。"""
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            # 多模态消息：取 text 部分
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    return part.get("text", "")
        return str(content) if content else ""
    return ""


class When2CallAdapter(BaseAdapter):
    """When2Call 数据集适配器"""

    @property
    def name(self) -> str:
        return "When2Call"

    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """加载原始数据。

        按优先级依次尝试：
          {data_path}/{split}.jsonl
          {data_path}/when2call_{split}.jsonl
          {data_path}/{split}.json
        """
        candidates = [
            self.data_path / f"{self.split}.jsonl",
            self.data_path / f"when2call_{self.split}.jsonl",
            self.data_path / f"{self.split}.json",
        ]

        data_file: Optional[Path] = None
        for c in candidates:
            if c.exists():
                data_file = c
                break

        if data_file is None:
            available = [p.name for p in self.data_path.iterdir()] if self.data_path.exists() else []
            raise FileNotFoundError(
                f"Cannot find '{self.split}' data in {self.data_path}. "
                f"Available files: {available}"
            )

        samples: List[Dict[str, Any]] = []
        if data_file.suffix == ".jsonl":
            with open(data_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        else:
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                samples = data if isinstance(data, list) else data.get("data", [data])

        return samples

    # ──────────────────────── label inference ────────────────────────

    def _infer_label(self, raw_sample: Dict[str, Any]) -> DecisionLabel:
        """推断样本标签，按优先级依次尝试三种格式。

        1. pref 格式：chosen_response.content 含 <TOOLCALL> → CALL，否则 NO_CALL
        2. MCQ 格式：answer 字段（tool_call / cannot_answer / request_for_info）
        3. 旧格式兼容：should_call 布尔字段
        """
        # 1. pref 格式
        chosen = raw_sample.get("chosen_response") or raw_sample.get("chosen")
        if chosen is not None:
            if isinstance(chosen, dict):
                content = chosen.get("content") or chosen.get("message") or ""
            else:
                content = str(chosen)
            return DecisionLabel.CALL if _has_toolcall_tag(content) else DecisionLabel.NO_CALL

        # 2. MCQ 格式
        answer = raw_sample.get("answer") or raw_sample.get("label")
        if answer is not None:
            val = str(answer).lower().strip()
            if val in ("tool_call", "call", "yes", "1", "true"):
                return DecisionLabel.CALL
            if val in ("cannot_answer", "no_call", "no", "0", "false"):
                return DecisionLabel.NO_CALL
            if val == "request_for_info":
                return DecisionLabel.UNCERTAIN

        # 3. 旧格式兼容
        should_call = raw_sample.get("should_call")
        if should_call is True or should_call == 1:
            return DecisionLabel.CALL
        if should_call is False or should_call == 0:
            return DecisionLabel.NO_CALL

        return DecisionLabel.UNCERTAIN

    # ──────────────────────── sample conversion ──────────────────────

    def _convert_sample(self, raw_sample: Dict[str, Any], idx: int) -> TaskSample:
        """转换 When2Call 样本（支持 pref / MCQ / 旧格式）。"""
        # 工具信息
        tools = raw_sample.get("tools") or raw_sample.get("functions") or []
        if not isinstance(tools, list):
            tools = [tools] if tools else []
        tool_schemas: List[Dict] = [t for t in tools if isinstance(t, dict)]
        available_tools: List[str] = [
            t.get("name") or t.get("function", {}).get("name", "")
            for t in tool_schemas
        ]
        available_tools = [n for n in available_tools if n]

        # 消息历史
        messages: List[Dict] = raw_sample.get("messages") or []

        # instruction：优先从 messages 提取最后一条 user 消息
        instruction = _extract_last_user_message(messages)
        if not instruction:
            instruction = (
                raw_sample.get("instruction")
                or raw_sample.get("query")
                or raw_sample.get("user_request")
                or ""
            )

        # context：原始字段，或将除最后一条 user 消息外的历史序列化
        context: Optional[str] = raw_sample.get("context") or raw_sample.get("input") or None
        if context is None and messages:
            other_msgs = [
                m for m in messages
                if not (m.get("role") == "user" and m.get("content") == instruction)
            ]
            if other_msgs:
                context = json.dumps(other_msgs, ensure_ascii=False)

        # 标签
        label = self._infer_label(raw_sample)

        # 期望输出
        expected_tool: Optional[str] = None
        expected_args: Optional[Dict] = None
        expected_response: Optional[str] = None

        if label == DecisionLabel.CALL:
            gt = raw_sample.get("ground_truth") or raw_sample.get("expected") or {}
            if isinstance(gt, dict):
                expected_tool = gt.get("tool") or gt.get("name")
                expected_args = gt.get("arguments") or gt.get("args")
        elif label == DecisionLabel.NO_CALL:
            resp = raw_sample.get("expected_response") or raw_sample.get("answer")
            # 避免把 MCQ 的 answer label 字符串存为 response
            if isinstance(resp, str) and resp not in (
                "cannot_answer", "request_for_info", "tool_call"
            ):
                expected_response = resp

        return TaskSample(
            sample_id=raw_sample.get("id", f"w2c_{idx:06d}"),
            instruction=instruction,
            context=context,
            tool_schemas=tool_schemas,
            available_tools=available_tools,
            label=label,
            expected_tool=expected_tool,
            expected_args=expected_args,
            expected_response=expected_response,
            source_dataset="when2call",
            difficulty=raw_sample.get("difficulty") or raw_sample.get("level"),
            category=raw_sample.get("category") or raw_sample.get("type"),
            metadata={
                "original_id": raw_sample.get("id"),
                "split": self.split,
                # 保留原始消息，供激活提取时构建 action boundary prompt
                "original_messages": messages if messages else None,
                "original_tools": tools if tools else None,
            },
        )

    # ──────────────────────── mock data ──────────────────────────────

    @staticmethod
    def create_mock_data(output_path: Path, num_samples: int = 100):
        """创建 pref 格式模拟数据（用于测试）。"""
        import random

        output_path.parent.mkdir(parents=True, exist_ok=True)

        tools = [
            {
                "name": "search",
                "description": "在知识库中搜索信息",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "搜索查询"}},
                    "required": ["query"],
                },
            },
            {
                "name": "calculator",
                "description": "执行数学计算",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string", "description": "数学表达式"}},
                    "required": ["expression"],
                },
            },
        ]

        call_templates = [
            ("2023年诺贝尔物理学奖获得者是谁？", "search", {"query": "2023诺贝尔物理学奖"}),
            ("计算 sqrt(144) + 25 * 3", "calculator", {"expression": "sqrt(144) + 25 * 3"}),
            ("最新的 AI 模型发布了什么？", "search", {"query": "latest AI model release"}),
            ("15% of 250 是多少？", "calculator", {"expression": "0.15 * 250"}),
        ]
        no_call_instructions = [
            "什么是机器学习？",
            "Python 是什么编程语言？",
            "2 + 2 等于多少？",
            "说一句问候语",
        ]

        samples = []
        for i in range(num_samples):
            if random.random() < 0.5:
                tmpl = random.choice(call_templates)
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": tmpl[0]},
                ]
                chosen_content = (
                    f"I'll use a tool to find that. "
                    f'<TOOLCALL>{{"name": "{tmpl[1]}", "arguments": {json.dumps(tmpl[2])}}}</TOOLCALL>'
                )
                samples.append({
                    "id": f"mock_{i:06d}",
                    "messages": messages,
                    "tools": tools,
                    "chosen_response": {"role": "assistant", "content": chosen_content},
                    "rejected_response": {"role": "assistant", "content": "I don't know."},
                    "category": "tool_required",
                })
            else:
                instr = random.choice(no_call_instructions)
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": instr},
                ]
                samples.append({
                    "id": f"mock_{i:06d}",
                    "messages": messages,
                    "tools": tools,
                    "chosen_response": {"role": "assistant", "content": "Let me answer directly."},
                    "rejected_response": {"role": "assistant", "content": "<TOOLCALL>...</TOOLCALL>"},
                    "category": "direct_answer",
                })

        with open(output_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        print(f"Created {num_samples} mock pref samples at {output_path}")
