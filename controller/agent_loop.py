"""
Agent Loop - H2 sandbox rollout 生成

最小 agent 循环：
  - 用 chat template 构建上下文（含 sandbox tool schema）
  - 推理模型，解析 CALL/NO_CALL 决策（与 when2call_adapter 相同的 <TOOLCALL> 标记）
  - 若 CALL：执行 sandbox tool，追加结果，继续下一步
  - 若 NO_CALL 或达到 max_steps：结束 episode
  - 可选：在每步 action boundary 处提取激活（H2 轨迹分析用）

Smoke test（无需 LLM）：
  python controller/agent_loop.py
"""

import json
import random
import re
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional

import torch


# ─────────────────────── data structures ───────────────────────────

@dataclass
class AgentConfig:
    max_steps: int = 10
    p_fail: float = 0.0
    p_empty: float = 0.0
    p_corrupt: float = 0.0
    layers_to_track: List[int] = field(default_factory=list)
    max_new_tokens: int = 128
    device: str = "cuda"
    dtype: str = "bfloat16"


@dataclass
class EpisodeStep:
    turn: int
    messages: List[Dict]
    decision: str                           # "CALL" or "NO_CALL"
    tool_name: Optional[str]
    tool_result: Optional[str]
    activations: Dict[int, torch.Tensor]    # layer → [1, hidden]


# ─────────────────────── AgentLoop ─────────────────────────────────

class AgentLoop:
    """最小 agent 循环，可选激活收集（H2 用）。"""

    def __init__(
        self,
        llm,
        tokenizer,
        config: AgentConfig,
        tool_kit=None,
    ):
        self.llm = llm
        self.tokenizer = tokenizer
        self.config = config

        if tool_kit is None:
            from controller.sandbox_tools import SandboxToolKit
            tool_kit = SandboxToolKit(
                p_fail=config.p_fail,
                p_empty=config.p_empty,
                p_corrupt=config.p_corrupt,
            )
        self.tool_kit = tool_kit

    def run_episode(
        self, task_prompt: str, seed: Optional[int] = None
    ) -> Generator[EpisodeStep, None, None]:
        """运行一个 episode，逐步 yield EpisodeStep。"""
        rng = random.Random(seed)
        self.tool_kit.reset(seed=seed)
        schemas = self.tool_kit.get_schemas()

        messages: List[Dict] = [{"role": "user", "content": task_prompt}]

        for turn in range(self.config.max_steps):
            input_ids = self._build_input_ids(messages, schemas)
            decision, generated_text, activations = self._infer(input_ids)

            tool_name = None
            tool_result = None

            if decision == "CALL":
                tool_name = self._parse_tool_name(generated_text, schemas)
                args = self._parse_tool_args(generated_text)
                tool_result = self._execute_tool(tool_name, args, rng)
                messages.append({"role": "assistant", "content": generated_text})
                messages.append({
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": tool_name or "unknown",
                })
            else:
                messages.append({"role": "assistant", "content": generated_text})

            yield EpisodeStep(
                turn=turn,
                messages=list(messages),
                decision=decision,
                tool_name=tool_name,
                tool_result=tool_result,
                activations=activations,
            )

            if decision == "NO_CALL":
                break

    # ─────────────────── internal helpers ──────────────────────────

    def _build_input_ids(self, messages: List[Dict], schemas: List[Dict]) -> torch.Tensor:
        """构建 action boundary input_ids（复用 cache_activations 中的逻辑）。"""
        try:
            ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                tools=schemas if schemas else None,
            )
        except Exception:
            try:
                ids = self.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
            except Exception:
                text = "\n".join(f"{m['role']}: {m.get('content', '')}" for m in messages)
                ids = self.tokenizer.encode(text, add_special_tokens=True)

        return torch.tensor([ids], dtype=torch.long)

    def _infer(self, input_ids: torch.Tensor):
        """运行推理，返回 (decision, text, activations)。"""
        activations: Dict[int, torch.Tensor] = {}
        hooks = []

        for layer in self.config.layers_to_track:
            module = self._get_layer_module(layer)
            hook = module.register_forward_hook(self._make_capture_hook(activations, layer))
            hooks.append(hook)

        try:
            with torch.no_grad():
                out = self.llm.generate(
                    input_ids.to(self.config.device),
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        finally:
            for h in hooks:
                h.remove()

        new_tokens = out[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        decision = "CALL" if "<TOOLCALL>" in text.upper() else "NO_CALL"
        return decision, text, activations

    def _make_capture_hook(self, storage: Dict, layer: int):
        """创建 forward hook，捕获 action boundary 最后 token 的激活。"""
        def hook(module, input, output):
            h = output[0]  # [batch, seq, hidden]
            storage[layer] = h[:, -1, :].detach().cpu()
        return hook

    def _get_layer_module(self, layer: int):
        llm = self.llm
        if hasattr(llm, "model") and hasattr(llm.model, "layers"):
            return llm.model.layers[layer]
        if hasattr(llm, "layers"):
            return llm.layers[layer]
        raise RuntimeError("Cannot locate model layers")

    def _parse_tool_name(self, text: str, schemas: List[Dict]) -> Optional[str]:
        """从生成文本中解析被调用的工具名。"""
        known_names = [s["function"]["name"] for s in schemas]
        for name in known_names:
            if name in text:
                return name
        return known_names[0] if known_names else None

    def _parse_tool_args(self, text: str) -> Dict:
        """尝试从生成文本中解析工具参数（简单 JSON 解析）。"""
        match = re.search(r"\{[^}]+\}", text)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return {}

    def _execute_tool(self, tool_name: Optional[str], args: Dict, rng: random.Random) -> str:
        if tool_name is None:
            return "Error: no tool name specified"
        try:
            return self.tool_kit.execute(tool_name, args, rng)
        except Exception as e:
            return f"Error: {e}"


# ─────────────────────── smoke test ────────────────────────────────

if __name__ == "__main__":
    import os
    import sys
    # 确保从项目根目录可以找到 controller 模块
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("AgentLoop smoke test (mock LLM, no GPU required)...")

    turn_counter = [0]

    class _MockLLM:
        device = "cpu"

        def generate(self, input_ids, **kwargs):
            turn_counter[0] += 1
            # 前两步返回 CALL，第三步返回 NO_CALL
            if turn_counter[0] <= 2:
                token_text = '<TOOLCALL>{"query": "test query"}</TOOLCALL>'
            else:
                token_text = "I have enough information to answer."
            # 模拟返回：原 input + 新 token（用 0 占位，decode 由 tokenizer 处理）
            fake_new = torch.zeros(1, 10, dtype=torch.long)
            return torch.cat([input_ids, fake_new], dim=1)

    class _MockTokenizer:
        eos_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            return [1, 2, 3]

        def encode(self, text, **kwargs):
            return [1, 2, 3]

        def decode(self, tokens, **kwargs):
            if turn_counter[0] <= 2:
                return '<TOOLCALL>{"query": "test query"}</TOOLCALL>'
            return "I have enough information to answer."

    config = AgentConfig(max_steps=5, p_fail=0.0, p_empty=0.1, device="cpu")
    loop = AgentLoop(_MockLLM(), _MockTokenizer(), config)

    for step in loop.run_episode("Find out the capital of France.", seed=42):
        result_preview = repr(step.tool_result)[:40] if step.tool_result else "None"
        print(f"  Step {step.turn}: decision={step.decision}, tool={step.tool_name}, "
              f"result={result_preview}")
        if step.decision == "NO_CALL":
            break

    print("Smoke test passed.")
