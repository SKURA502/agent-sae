"""
case_study_visualize.py
────────────────────────────────────────────────────────────────────────
给定多轮对话（system + tools + multi-turn），对整段序列做一次前向传播，
在每个 token 上可视化 top_features_layer25.json 中所有 feature 的
SAE 激活值之和，背景绿色深浅表示激活强度，输出为 PNG 图片。

用法示例：
python analysis/case_study_visualize.py \
    --model /mnt/shared-storage-gpfs2/safelens-share-gpfs2/source/model/Qwen/Qwen3.5-4B \
    --sae-path /data/Agent-Tool-Use-MI/checkpoint/Qwen3.5-4B/stage2/Qwen3.5-4B-L25-d20480-5M-stage2.pt \
    --layer 25 \
    --features-json /data/Agent-Tool-Use-MI/outputs/judge/analysis/feature_discovery/top_features_layer25.json \
    --top-k 1 \
    --zero-pct 50 \
    --gamma 5 \
    --output outputs/judge/analysis/case_study/accumulate-top1.png

★ 在下方"用户修改区域"填写你的对话内容。
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sae.sae_model import TopKSAE


# ════════════════════════════════════════════════════════════════════
#  ★ 用户修改区域 — 在此填写对话内容
# ════════════════════════════════════════════════════════════════════

TOOLS = [
{
"type": "function",
"function": {
"name": "get_current_time",
"description": "Get the current time",
"parameters": {
"type": "object",
"properties": {
"timezone": {
"type": "string",
"description": "The timezone, e.g. America/New_York"
}
},
"required": [
"timezone"
]
}
}
},
{
"type": "function",
"function": {
"name": "search_books",
"description": "Search for books by title or author",
"parameters": {
"type": "object",
"properties": {
"keyword": {
"type": "string",
"description": "The keyword to search for in book titles or authors"
}
},
"required": [
"keyword"
]
}
}
}
]

# system 消息会自动根据 TOOLS 生成，这里只填 user / assistant / tool 轮次
MESSAGES = [
{
"role": "user",
"content": "What's the current time in New York?"
},
{
"tool_calls": [
{
"function": {
"name": "get_current_time",
"arguments": {
"timezone": "America/New_York"
}
}
}
],
"role": "assistant",
"content": ""
},
{
"role": "tool",
"content": "{\"time\": \"10:00 AM\"}"
},
{
"role": "assistant",
"content": "The current time in New York is 10:00 AM."
},
{
"role": "user",
"content": "And what about in London?"
},
{
"tool_calls": [
{
"function": {
"name": "get_current_time",
"arguments": {
"timezone": "Europe/London"
}
}
}
],
"role": "assistant",
"content": "<think>\nOkay, the user first asked for the current time in New York, and I used the get_current_time tool with America/New_York. Now they're asking \"And what about in London?\" so they want the time in London.\n\nI need to check the available tools. The get_current_time function requires a timezone. London's timezone is Europe/London. I should call the same function but with the Europe/London timezone parameter.\n\nMake sure to structure the tool call correctly. The parameters need to be a JSON object with the timezone. So the arguments should be {\"timezone\": \"Europe/London\"}.\n\nDouble-check the timezone name. Yes, it's Europe/London for London, UK. No daylight saving adjustments needed here since the tool probably handles that automatically based on the timezone string.\n\nSo the correct tool call is get_current_time with Europe/London.\n</think>"
}
]

# ════════════════════════════════════════════════════════════════════
#  激活提取
# ════════════════════════════════════════════════════════════════════


def _normalize_tools(tools):
    """把 OpenAI 嵌套格式的 tool 定义拍平为 Qwen chat template 期望的格式。

    OpenAI 格式: {"type": "function", "function": {"name":…, "parameters":…}}
    Qwen 期望:   {"name":…, "description":…, "parameters":…}
    """
    result = []
    for t in tools:
        if isinstance(t, str):
            t = json.loads(t)
        if t.get("type") == "function" and "function" in t:
            result.append(t["function"])
        else:
            result.append(t)
    return result


def _adapt_messages_for_chat_template(messages):
    """将消息格式适配为 apply_chat_template 期望的格式，让模板自动渲染工具调用。

    关键处理：
    1. assistant + tool_calls：
       确保 tool_call 含 "type": "function"，arguments 保持为 dict。
       直接保留 tool_calls 字段，由 apply_chat_template 渲染工具调用格式。
    2. role=="tool"（tool 结果）：
       保留原始 role="tool"，由 apply_chat_template 渲染工具返回格式。
       如有前序 assistant 的 tool_calls，补充 name 字段。
    """
    adapted = []
    # 记录最近的 tool_call names，用于给 tool 消息补充 name 字段
    last_tool_call_names = []

    for msg in messages:
        role = msg.get("role", "")
        new_msg = dict(msg)  # 浅拷贝

        if role == "assistant" and msg.get("tool_calls"):
            # 记录 tool_call names，供后续 tool 消息使用
            last_tool_call_names = []
            adapted_tool_calls = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", {})
                # Qwen3.5 模板用 arguments.items() 遍历参数，必须保持为 dict
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        pass
                last_tool_call_names.append(name)
                adapted_tool_calls.append({
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                })
            new_msg["tool_calls"] = adapted_tool_calls
            adapted.append(new_msg)

        elif role == "tool":
            # 补充 name 字段（从最近的 tool_call 中获取）
            if "name" not in new_msg and last_tool_call_names:
                new_msg["name"] = last_tool_call_names.pop(0)
            adapted.append(new_msg)

        else:
            adapted.append(new_msg)

    return adapted


def get_token_activations(model, tokenizer, messages, tools, layer: int, device: str):
    """前向传播一次，返回 (token_texts, hidden_states, message_boundaries)。

    Returns:
        token_texts : List[str]           每个 token 的原始文本（含特殊 token）
        hidden      : Tensor [seq_len, H] 指定层输出的隐状态
        boundaries  : List[(int, str)]    每条消息的 (起始 token 索引, 角色名)
    """
    adapted_messages = _adapt_messages_for_chat_template(messages)

    # 使用 apply_chat_template(tools=) 让 tokenizer 自动渲染工具格式，
    # 与 Stage2 训练和 cache_activations.py 保持一致
    parsed_tools = []
    for t in tools:
        if isinstance(t, str):
            try:
                parsed_tools.append(json.loads(t))
            except (json.JSONDecodeError, TypeError):
                parsed_tools.append(t)
        else:
            parsed_tools.append(t)

    try:
        text = tokenizer.apply_chat_template(
            adapted_messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
            tools=parsed_tools if parsed_tools else None,
        )
    except TypeError:
        try:
            text = tokenizer.apply_chat_template(
                adapted_messages, tokenize=False, add_generation_prompt=True,
                tools=parsed_tools if parsed_tools else None,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                adapted_messages, tokenize=False, add_generation_prompt=True,
            )

    # 用 adapted_messages 建语义标签列表（保留 tool / assistant+tool_calls 的原始 role）
    # tokenizer 自动生成的 system 消息也要计入
    label_messages = [{"role": "system"}, *adapted_messages]

    ids = tokenizer.encode(text, add_special_tokens=False)
    input_ids = torch.tensor([ids], dtype=torch.long).to(device)

    token_texts = [
        tokenizer.decode([tid], skip_special_tokens=False)
        for tid in ids
    ]

    # 注册 forward hook 采集 layer 输出隐状态
    hidden_capture = []

    def _hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        hidden_capture.append(h.detach().float().cpu())

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer_mod = model.model.layers[layer]
    elif hasattr(model, "layers"):
        layer_mod = model.layers[layer]
    else:
        raise RuntimeError("无法定位模型层，请检查模型结构。")

    handle = layer_mod.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        handle.remove()

    hidden = hidden_capture[0][0]  # [seq_len, H]
    # 用 full_messages 的语义覆盖 token 里读到的原始 role 名
    boundaries = _find_message_boundaries(token_texts, label_messages)
    # add_generation_prompt=True 会在末尾追加 <|im_start|>assistant\n，
    # 产生一个没有对应 label_messages 的多余 boundary，需要去掉，
    # 同时裁掉对应的 token 和 hidden state，避免在可视化中出现多余内容
    if len(boundaries) > len(label_messages):
        gen_prompt_start = boundaries[-1][0]  # 多余 boundary 的起始位置
        token_texts = token_texts[:gen_prompt_start]
        hidden = hidden[:gen_prompt_start]
        boundaries = boundaries[:-1]
    return token_texts, hidden, boundaries


def compute_feature_scores(sae, hidden: torch.Tensor, feature_indices: list, device: str):
    """SAE encode → 返回指定 feature 激活值之和，shape [seq_len]。"""
    h = hidden.to(device).to(sae.config.get_torch_dtype())
    latents = sae.encode(h)                           # [seq_len, dict_size]
    scores = latents[:, feature_indices].sum(dim=-1)  # [seq_len]
    return scores.detach().cpu().float().numpy()


# ════════════════════════════════════════════════════════════════════
#  渲染
# ════════════════════════════════════════════════════════════════════

# 用于把 BPE/SentencePiece 空格前缀替换为普通空格
_BPE_SPACE = str.maketrans({"▁": " ", "Ġ": " "})


def _display(tok: str) -> str:
    """把 token 字符串转为适合显示的文本（保留可见字符）。"""
    d = tok.translate(_BPE_SPACE)
    # 用 ↵ 表示换行，避免真正换行破坏布局
    d = d.replace("\n", "↵").replace("\r", "")
    return d if d else tok  # fallback：保持原样


def _layout_lines(token_texts, norm_scores, max_chars: int):
    """将 token 序列按字符数分行（无角色信息时的 fallback）。"""
    lines = []
    cur_line = []
    cur_len = 0
    for tok, sc in zip(token_texts, norm_scores):
        dt = _display(tok)
        w = max(len(dt), 1)
        if cur_len + w > max_chars and cur_line:
            lines.append(cur_line)
            cur_line = []
            cur_len = 0
        cur_line.append((dt, float(sc)))
        cur_len += w
    if cur_line:
        lines.append(cur_line)
    return lines


# role_key → header 背景色（RGB）
_ROLE_BG = {
    "system":         (0.88, 0.88, 0.88),  # 浅灰
    "user":           (0.78, 0.89, 1.00),  # 浅蓝
    "tool_call":      (1.00, 0.88, 0.65),  # 浅橙（assistant 发出 tool call）
    "tool_response":  (0.88, 0.80, 1.00),  # 浅紫（tool 返回结果）
    "assistant":      (1.00, 0.92, 0.76),  # 浅橙（assistant 最终回答）
}
_ROLE_LABEL = {
    "system":        "SYSTEM",
    "user":          "USER",
    "tool_call":     "TOOL_CALL",
    "tool_response": "TOOL_CALL_RESPONSE",
    "assistant":     "ASSISTANT",
}


def _semantic_role(msg: dict) -> str:
    """根据消息内容返回语义 role_key。"""
    role = msg.get("role", "")
    if role == "assistant":
        return "tool_call" if msg.get("tool_calls") else "assistant"
    if role == "tool":
        return "tool_response"
    return role  # system / user


def _find_message_boundaries(token_texts, full_messages=None):
    """扫描 token 列表，找出每条消息的 (起始索引, role_key)。

    若提供 full_messages，则按消息顺序把语义 role_key（tool_call /
    tool_response / assistant 等）覆盖到对应边界上；
    否则退化为从 token 文本读取原始 role 名。
    """
    boundaries = []
    for i, tok in enumerate(token_texts):
        if "<|im_start|>" in tok:
            raw_role = token_texts[i + 1].strip().lower() if i + 1 < len(token_texts) else ""
            boundaries.append((i, raw_role))

    # 用原始 messages 的语义覆盖 role
    if full_messages and len(full_messages) == len(boundaries):
        boundaries = [
            (idx, _semantic_role(msg))
            for (idx, _), msg in zip(boundaries, full_messages)
        ]
    return boundaries


def _layout_conversation(token_texts, norm_scores, message_boundaries, max_chars: int):
    """按对话角色分块布局，返回行列表。

    每行是一个 dict：
      {"type": "header",  "role": str}
      {"type": "tokens",  "role": str, "items": [(display_text, norm_score)]}
      {"type": "spacer"}
    """
    boundaries = sorted(message_boundaries, key=lambda x: x[0])
    n = len(boundaries)
    rows = []

    for msg_idx, (start_idx, role) in enumerate(boundaries):
        end_idx = boundaries[msg_idx + 1][0] if msg_idx + 1 < n else len(token_texts)

        # 角色 header 行
        rows.append({"type": "header", "role": role})

        # 把这条消息内的 token 分行
        cur_line, cur_len = [], 0
        for tok, sc in zip(token_texts[start_idx:end_idx],
                           norm_scores[start_idx:end_idx]):
            dt = _display(tok)
            w = max(len(dt), 1)
            if cur_len + w > max_chars and cur_line:
                rows.append({"type": "tokens", "role": role, "items": cur_line})
                cur_line, cur_len = [], 0
            cur_line.append((dt, float(sc)))
            cur_len += w
        if cur_line:
            rows.append({"type": "tokens", "role": role, "items": cur_line})

        # 消息间空行
        rows.append({"type": "spacer"})

    return rows


def render_heatmap(
    token_texts,
    scores,
    output_path: str,
    title: str = "Token activation heatmap",
    max_chars_per_line: int = 110,
    font_size: float = 8.0,
    dpi: int = 150,
    gamma: float = 3.0,
    zero_pct: float = 0.0,
    clip_pct: float = 100.0,
    message_boundaries=None,
):
    """将 token 热力图渲染为 PNG 图片。

    当 message_boundaries 不为空时，使用对话排版：
    每条消息前显示角色 header（灰/蓝/橙/紫色横条），消息间有空白分隔。

    clip_pct  : 用第 N 百分位作为 vmax 截顶，超过的饱和为深绿，
                避免极端值把大多数 token 压到同一浅色（默认 100 = 不截）。
    zero_pct  : 低于第 N 百分位的值直接归白（默认 0 = 不截）。
    gamma     : 幂次变换，>1 进一步压缩小值、拉伸大值。
    """
    # ── normalize scores ─────────────────────────────────────────────
    vmin = float(scores.min())
    # 上截：用百分位作为 vmax，极端值饱和为深绿
    vmax_display = float(np.percentile(scores, clip_pct)) if clip_pct < 100 else float(scores.max())
    vmax = float(scores.max())  # 仅用于 colorbar 显示真实范围

    # 下截：低于 zero_pct 百分位的直接归 0
    if zero_pct > 0:
        threshold = float(np.percentile(scores, zero_pct))
        working = np.where(scores <= threshold, 0.0, scores - threshold)
        vmax_display = max(vmax_display - threshold, 1e-8)
    else:
        working = scores - vmin

    # 归一化到 [0, 1]，超过 vmax_display 的饱和到 1
    norm_sc = np.clip(working / (vmax_display + 1e-8), 0.0, 1.0)

    # gamma 变换：进一步压缩小值、拉伸大值
    norm_sc = np.power(norm_sc, gamma)

    # ── 布局 ─────────────────────────────────────────────────────────
    # 每种行的高度（以 token 行为 1 个单位）
    HEADER_H  = 1.5   # 角色 header 行高
    TOKEN_H   = 1.0   # token 行高
    SPACER_H  = 0.5   # 消息间空行高

    if message_boundaries:
        rows = _layout_conversation(token_texts, norm_sc, message_boundaries, max_chars_per_line)
    else:
        # 无角色信息：退化为普通分行
        rows = [{"type": "tokens", "role": "", "items": line}
                for line in _layout_lines(token_texts, norm_sc, max_chars_per_line)]

    # 计算总高度（单位）
    total_h_units = sum(
        HEADER_H if r["type"] == "header" else
        SPACER_H if r["type"] == "spacer" else
        TOKEN_H
        for r in rows
    )

    # ── 计算图片尺寸 ──────────────────────────────────────────────────
    char_w_pt  = font_size * 0.60
    unit_h_pt  = font_size * 1.80   # 1 单位高度对应的 points
    margin_pt  = font_size * 1.5
    cbar_h_pt  = font_size * 4.0
    title_h_pt = font_size * 3.0

    fig_w_pt = max_chars_per_line * char_w_pt + 2 * margin_pt
    fig_h_pt = title_h_pt + total_h_units * unit_h_pt + margin_pt + cbar_h_pt

    fig = plt.figure(figsize=(fig_w_pt / 72, fig_h_pt / 72), dpi=dpi, facecolor="white")

    ax_left   = margin_pt / fig_w_pt
    ax_right  = 1.0 - margin_pt / fig_w_pt
    ax_bottom = (cbar_h_pt + margin_pt * 0.5) / fig_h_pt
    ax_top    = (cbar_h_pt + margin_pt * 0.5 + total_h_units * unit_h_pt) / fig_h_pt

    ax = fig.add_axes([ax_left, ax_bottom, ax_right - ax_left, ax_top - ax_bottom])
    ax.set_xlim(0, max_chars_per_line)
    ax.set_ylim(-total_h_units, 0)
    ax.axis("off")

    cmap = plt.cm.Greens
    y = 0.0  # 当前绘制位置（从顶部往下累减）

    for row in rows:
        rtype = row["type"]

        if rtype == "spacer":
            y -= SPACER_H

        elif rtype == "header":
            role  = row["role"]
            bg    = _ROLE_BG.get(role, (0.85, 0.85, 0.85))
            label = _ROLE_LABEL.get(role, role.upper())

            # 全宽背景横条
            ax.add_patch(patches.Rectangle(
                (0, y - HEADER_H), max_chars_per_line, HEADER_H,
                linewidth=0, facecolor=bg, clip_on=False,
            ))
            # 角色标签
            ax.text(
                1.0, y - HEADER_H / 2,
                label,
                fontsize=font_size,
                fontweight="bold",
                fontfamily="monospace",
                va="center", ha="left",
                color="#333333",
                clip_on=False,
            )
            y -= HEADER_H

        else:  # tokens
            x = 0.0
            for dt, sc in row["items"]:
                w = max(len(dt), 1)
                # sc==0 → 纯白；非零 → 从可见浅绿起步
                color = (1.0, 1.0, 1.0, 1.0) if sc == 0.0 else cmap(sc * 0.85 + 0.15)

                ax.add_patch(patches.Rectangle(
                    (x, y - TOKEN_H * 0.88), w, TOKEN_H * 0.88,
                    linewidth=0, facecolor=color, clip_on=False,
                ))
                ax.text(
                    x + 0.15, y - TOKEN_H * 0.44,
                    dt,
                    fontsize=font_size,
                    fontfamily="monospace",
                    va="center", ha="left",
                    color="white" if sc > 0.72 else "black",
                    clip_on=False,
                )
                x += w
            y -= TOKEN_H

    # ── 标题 ────────────────────────────────────────────────────────
    fig.text(
        0.5, (ax_top + 1.0) / 2,
        title,
        ha="center", va="center",
        fontsize=font_size + 2,
        transform=fig.transFigure,
    )

    # ── colorbar ────────────────────────────────────────────────────
    cax = fig.add_axes([0.08, (margin_pt * 0.3) / fig_h_pt,
                        0.84, (cbar_h_pt * 0.28) / fig_h_pt])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Sum of top SAE feature activations", fontsize=font_size)
    cb.ax.tick_params(labelsize=max(font_size - 1, 6))

    # ── 保存 ────────────────────────────────────────────────────────
    Path(str(output_path)).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    n_token_rows = sum(1 for r in rows if r["type"] == "tokens")
    print(f"Saved  → {output_path}")
    print(f"Tokens : {len(token_texts)}  |  Rows: {n_token_rows}")
    print(f"Score  : min={vmin:.4f}  max={vmax:.4f}  mean={scores.mean():.4f}")


# ════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Token-level SAE feature activation heatmap (PNG output)"
    )
    parser.add_argument("--model",
        type=str, required=True, help="LLM 路径")
    parser.add_argument("--sae-path",
        type=str, required=True, help="SAE checkpoint 路径")
    parser.add_argument("--layer",
        type=int, default=25, help="hook 层索引（默认 25）")
    parser.add_argument("--features-json",
        type=str,
        default="/data/Agent-Tool-Use-MI/outputs/judge/analysis/feature_discovery/top_features_layer25.json",
        help="top_features JSON 路径")
    parser.add_argument("--top-k",
        type=int, default=None,
        help="只使用 JSON 中排名前 k 的 feature（默认使用全部）")
    parser.add_argument("--output",
        type=str, default="case_study.png", help="输出图片路径")
    parser.add_argument("--device",
        type=str, default="cuda")
    parser.add_argument("--dtype",
        type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-chars",
        type=int, default=110, help="每行最大字符数（默认 110）")
    parser.add_argument("--font-size",
        type=float, default=10.0, help="字体大小 pt（默认 10）")
    parser.add_argument("--dpi",
        type=int, default=300, help="输出 DPI（默认 300）")
    parser.add_argument("--gamma",
        type=float, default=3.0,
        help="颜色幂次变换系数，>1 使小值更白、大值更快变深绿（默认 3.0）")
    parser.add_argument("--zero-pct",
        type=float, default=75.0,
        help="低于第 N 百分位的激活值直接置白（默认 75，即只显示前 25%% 高激活）")
    parser.add_argument("--clip-pct",
        type=float, default=95.0,
        help="用第 N 百分位作为颜色上限，超过的饱和为深绿，避免极端值压扁其他颜色（默认 95）")
    parser.add_argument("--title",
        type=str, default="", help="图片标题（默认自动生成）")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from run.cache_activations import _clear_sampling_generation_config

    # ── 加载 feature 索引 ─────────────────────────────────────────────
    with open(args.features_json, encoding="utf-8") as f:
        feature_list = json.load(f)
    feature_indices = [e["feature_idx"] for e in feature_list]
    if args.top_k is not None:
        feature_indices = feature_indices[: args.top_k]
    print(f"Features : {len(feature_indices)} indices loaded from {args.features_json}")

    # ── 加载模型 ──────────────────────────────────────────────────────
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading model  : {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype,
        device_map=args.device, trust_remote_code=True,
    )
    model.eval()
    _clear_sampling_generation_config(model)

    # ── 加载 SAE ─────────────────────────────────────────────────────
    print(f"Loading SAE    : {args.sae_path}")
    sae = TopKSAE.load(args.sae_path, device=args.device)
    sae.eval()

    # ── 前向传播，提取激活 ────────────────────────────────────────────
    print("Forward pass ...")
    token_texts, hidden, boundaries = get_token_activations(
        model, tokenizer, MESSAGES, TOOLS, args.layer, args.device,
    )
    print(token_texts)
    print(f"Sequence length: {len(token_texts)} tokens")
    print(f"Messages found : {len(boundaries)} ({[r for _, r in boundaries]})")

    # ── 计算 feature score ────────────────────────────────────────────
    print("Computing SAE feature scores ...")
    scores = compute_feature_scores(sae, hidden, feature_indices, args.device)

    # ── 渲染并保存 ────────────────────────────────────────────────────
    title = args.title or (
        f"Token activation heatmap — layer {args.layer}, "
        f"{len(feature_indices)} top features"
    )
    render_heatmap(
        token_texts, scores, args.output,
        title=title,
        max_chars_per_line=args.max_chars,
        font_size=args.font_size,
        dpi=args.dpi,
        gamma=args.gamma,
        zero_pct=args.zero_pct,
        clip_pct=args.clip_pct,
        message_boundaries=boundaries,
    )


if __name__ == "__main__":
    main()
