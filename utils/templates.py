"""
Prompt templates for Qwen2.5 and Qwen3.5 tool-calling models.

Qwen2.5 和 Qwen3.5 的工具调用格式不同：
- Qwen2.5：工具调用使用 <tool_callXML 格式，JSON 编码参数
- Qwen3.5：工具调用使用 <sourceXML 格式，<function>/<parameter> 标签编码参数

使用方式：
    from utils.templates import SYSTEM_TEMPLATE_QWEN25, SYSTEM_TEMPLATE_QWEN35
    from utils.templates import get_system_template
    from utils.templates import build_context_input_ids
"""

import json
from typing import List

from jinja2 import UndefinedError

import torch

# ─────────────────────── MCQ 选项常量 ──────────────────────────────

MCQ_CHOICES = ("direct_answer", "tool_call", "request_for_info", "cannot_answer")

# 四个选项的兜底答案文本（当数据集 answers 字段不可用时使用）
OPTION_FALLBACK = {
    "direct_answer":    "I can answer you directly without using any tool.",
    "tool_call":        "I need to call a tool to fulfill your request.",
    "request_for_info": "Could you provide more information so I can help you better?",
    "cannot_answer":    "I'm sorry, I cannot fulfill this request with the available tools.",
}


# ─────────────────────── Default System Prompt（无 tools 支持的模型）──

DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant.
You have access to the following tools described in <tool></tool> which you can use to answer the user's questions.
Only use a tool if it directly answers the user's question.
"""

TOOL_USE_INSTRUCTIONS = """To use a tool, return JSON in the following format:
{"name": "tool_name", "arguments": {"argument1": "value1", "argument2": "value2", ...}}
"""


# ─────────────────────── Qwen2.5 System Template ────────────────────

SYSTEM_TEMPLATE_QWEN25 = """\
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_str}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""


# ─────────────────────── Qwen3.5 System Template ────────────────────

SYSTEM_TEMPLATE_QWEN35 = """\
# Tools

You have access to the following functions:

<tools>
{tools_str}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>"""





# ─────────────────────── Judge Prompt ───────────────────────────────

JUDGE_PROMPT = """\
You are an expert at classifying responses from AI models.

Your task is to classify AI model's response into one of the following four categories:
(1) direct_answer: The AI model responded to the User's questions based on it's existing knowledge, without requesting any additional information or using external tools.
(2) tool_call: The AI model decided to use a tool from the provided one's to help answer the question.
(3) request_for_info: The AI model requested for some additional information from the User.
(4) cannot_answer: The AI model refused to answer the User's questions by acknowledging the lack of required capabilities.

*You should not judge whether the AI model's response is accurate or not. Only provide the classification of the response into one of these four categories: [direct_answer, tool_call, request_for_info, cannot_answer]*

- The tools available to the AI model are given in <AVAILABLE_TOOLS> </AVAILABLE_TOOLS>
- The User's question is provided in <USER_QUESTION> </USER_QUESTION>
- The AI model's response is provided in <AI_MODEL_RESPONSE> </AI_MODEL_RESPONSE> which may or may not invlove a tool call

<AVAILABLE_TOOLS>
{}
</AVAILABLE_TOOLS>

<USER_QUESTION>
{}
</USER_QUESTION>

<AI_MODEL_RESPONSE>
{}
</AI_MODEL_RESPONSE>

Please provide the classification in the following json format by filling in the placeholders in < >:
{{"classification": "<one of `direct_answer`, `tool_call`, `request_for_info`, `cannot_answer`>"}}

Respond only in the prescribed json format with the placeholders filled in."""


# ─────────────────────── Helper ─────────────────────────────────────


def get_system_template(model_name_or_path: str):
    """根据模型名称/路径自动选择对应的 system template。

    识别规则：
    - 路径或名称中包含 "qwen3"（如 Qwen3.5-4B）→ Qwen3.5 格式
    - 其他情况（包括 Qwen2.5）→ Qwen2.5 格式

    Args:
        model_name_or_path: HuggingFace 模型名或本地路径

    Returns:
        对应的 system template 字符串
    """
    key = model_name_or_path.lower()
    if "qwen3" in key:
        return SYSTEM_TEMPLATE_QWEN35
    return SYSTEM_TEMPLATE_QWEN25


# ─────────────────────── Context Builder ────────────────────────────


def build_context_input_ids(
    tokenizer,
    sample,
    max_length: int = 2048,
) -> torch.Tensor:
    """构建 context（system + tools + user question）的 input_ids。

    不含任何 MCQ 选项或指令，作为 log-prob 计算的 context 及激活提取的 action boundary。
    使用 apply_chat_template(tools=) 让 tokenizer 自动渲染工具格式，与 Stage2 训练一致。

    Returns:
        input_ids: [1, seq_len]
    """
    meta = sample.metadata or {}
    tools_raw: List = meta.get("original_tools_raw") or []
    # 解析工具定义为 dict 列表
    tools_parsed = []
    for t in tools_raw:
        if isinstance(t, str):
            try:
                tools_parsed.append(json.loads(t))
            except (json.JSONDecodeError, TypeError):
                pass
        elif isinstance(t, dict):
            tools_parsed.append(t)

    # 归一化为 OpenAI 格式 {"type":"function","function":{...}}
    # 与 stage2 训练时 _normalize_tools_for_template 保持一致
    def _to_openai_format(tool_list):
        result = []
        for t in tool_list:
            if not isinstance(t, dict):
                continue
            if t.get("type") == "function" and "function" in t:
                result.append(t)  # 已是 OpenAI 格式
            elif "name" in t:
                result.append({"type": "function", "function": t})
        return result

    # 裸格式（function 定义本身，不含外层 type/function 包装）
    def _to_bare_format(tool_list):
        result = []
        for t in tool_list:
            if isinstance(t, dict) and "function" in t:
                result.append(t["function"])
            elif isinstance(t, dict):
                result.append(t)
        return result

    tools_openai = _to_openai_format(tools_parsed)   # Gemma/GPT 系模型期望格式
    tools_bare = _to_bare_format(tools_openai)        # 部分模型期望裸格式

    messages = [
        {"role": "user", "content": sample.instruction or "(no question)"},
    ]

    # 检测 chat template 是否原生支持 tools 变量（Gemma-3 等模型不支持）
    _tmpl = getattr(tokenizer, 'chat_template', '') or ''
    _template_has_tools = 'tools' in str(_tmpl)

    def _apply(tools_list, msgs=None):
        """尝试用给定工具列表渲染 chat template。
        只捕获 enable_thinking=False 不被支持的 TypeError，其余异常向外传播。
        """
        if msgs is None:
            msgs = messages
        kwargs = dict(
            tokenize=True, add_generation_prompt=True,
            tools=tools_list if tools_list else None,
        )
        try:
            return tokenizer.apply_chat_template(msgs, enable_thinking=False, **kwargs)
        except TypeError:
            # enable_thinking 参数不被支持，去掉后重试；若仍报错则向外传播
            return tokenizer.apply_chat_template(msgs, **kwargs)

    if tools_openai and not _template_has_tools:
        # chat template 不原生支持 tools（如 Gemma-3）：用统一 system prompt 注入工具定义
        tools_xml = "\n".join(
            "<tool>\n" + json.dumps(t.get("function", t), ensure_ascii=False, indent=2) + "\n</tool>"
            for t in tools_openai
        )
        sys_content = DEFAULT_SYSTEM_PROMPT + tools_xml + "\n" + TOOL_USE_INSTRUCTIONS
        msgs_with_sys = [{"role": "system", "content": sys_content}] + messages
        try:
            result = _apply(None, msgs=msgs_with_sys)
        except Exception:
            result = _apply(None)
    else:
        try:
            # 优先尝试 OpenAI 格式（与训练一致）
            result = _apply(tools_openai)
        except (UndefinedError, KeyError, TypeError):
            try:
                # 回退到裸格式（部分模型 chat template 期望此格式）
                result = _apply(tools_bare)
            except Exception:
                # 两种格式都不兼容，不传 tools 兜底渲染
                result = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                )

    if result is None:
        fallback = "\n\n".join(m["content"] for m in messages)
        ids = tokenizer.encode(fallback, add_special_tokens=False)
    elif isinstance(result, list):
        ids = result
    elif isinstance(result, dict):
        ids = result["input_ids"]
    elif hasattr(result, "input_ids"):
        raw = result.input_ids
        ids = raw.tolist() if hasattr(raw, "tolist") else list(raw)
    else:
        ids = list(result)

    if len(ids) > max_length:
        ids = ids[-max_length:]

    return torch.tensor([ids], dtype=torch.long)
