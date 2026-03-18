"""
Sandbox Tools - H2 agent loop 使用的模拟工具

工具列表：search / calculator / lookup
每个 episode 随机化工具名（防止模型对工具名 token 产生 shortcut 学习）。
噪声注入：p_fail（抛异常）、p_empty（返回空结果）、p_corrupt（返回乱码）。
"""

import math
import random
import string
from typing import Any, Dict, List, Optional


# ─────────────────────── tool definitions ──────────────────────────

_BASE_TOOLS = [
    {
        "base_name": "search",
        "description": "Search for information on a topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."}
            },
            "required": ["query"],
        },
    },
    {
        "base_name": "calculator",
        "description": "Evaluate a mathematical expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression to evaluate."}
            },
            "required": ["expression"],
        },
    },
    {
        "base_name": "lookup",
        "description": "Look up a value by key in a knowledge base.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "The key to look up."}
            },
            "required": ["key"],
        },
    },
]

_MOCK_SEARCH_RESULTS = [
    "Relevant documents found: [1] Background context about the topic. [2] Additional details available.",
    "Search results: The answer involves multiple factors including context, timing, and available resources.",
    "Found 3 relevant articles. Key finding: the requested information is available in the knowledge base.",
]

_MOCK_LOOKUP_RESULTS = [
    "Value: 42.7 (standard reference, last updated 2024-01-01)",
    "Entry found: category=general, status=active, confidence=high",
    "Key exists: last_updated=2024-06-15, value=confirmed, source=internal_db",
]


# ─────────────────────── helpers ───────────────────────────────────

def _corrupt_string(s: str, rng: random.Random) -> str:
    """随机替换 1/4 字符制造乱码。"""
    chars = list(s)
    n = max(1, len(chars) // 4)
    positions = rng.sample(range(len(chars)), min(n, len(chars)))
    for p in positions:
        chars[p] = rng.choice(string.ascii_letters + "!@#$%^&*")
    return "".join(chars)


# ─────────────────────── SandboxToolKit ────────────────────────────

class SandboxToolKit:
    """三个沙盒工具，每个 episode 随机化名称，支持噪声注入。"""

    def __init__(self, p_fail: float = 0.0, p_empty: float = 0.0, p_corrupt: float = 0.0):
        self.p_fail = p_fail
        self.p_empty = p_empty
        self.p_corrupt = p_corrupt
        self._name_map: Dict[str, str] = {}   # randomized_name → base_name
        self._schemas: List[Dict] = []

    def reset(self, seed: Optional[int] = None):
        """每个 episode 开始时调用，生成新的随机工具名。"""
        rng = random.Random(seed)
        suffixes = [
            "".join(rng.choices(string.ascii_lowercase, k=4))
            for _ in range(len(_BASE_TOOLS))
        ]
        self._name_map = {}
        self._schemas = []
        for tool, suffix in zip(_BASE_TOOLS, suffixes):
            rand_name = f"{tool['base_name']}_{suffix}"
            self._name_map[rand_name] = tool["base_name"]
            self._schemas.append({
                "type": "function",
                "function": {
                    "name": rand_name,
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            })

    def get_schemas(self) -> List[Dict]:
        """返回当前 episode 的工具 schema 列表（随机化名称）。"""
        return self._schemas

    def execute(self, tool_name: str, args: Dict[str, Any], rng: random.Random) -> str:
        """执行工具，应用噪声注入，返回结果字符串。"""
        if rng.random() < self.p_fail:
            raise RuntimeError(f"Tool '{tool_name}' failed: service temporarily unavailable")

        if rng.random() < self.p_empty:
            return ""

        base_name = self._name_map.get(tool_name)
        result = self._dispatch(base_name, args, rng)

        if rng.random() < self.p_corrupt:
            result = _corrupt_string(result, rng)

        return result

    def _dispatch(self, base_name: Optional[str], args: Dict[str, Any], rng: random.Random) -> str:
        if base_name == "search":
            return rng.choice(_MOCK_SEARCH_RESULTS)
        elif base_name == "calculator":
            expr = args.get("expression", "0")
            try:
                result = eval(expr, {"__builtins__": {}}, {"sqrt": math.sqrt, "abs": abs, "pow": pow})
                return f"Result: {result}"
            except Exception:
                return "Error: could not evaluate expression"
        elif base_name == "lookup":
            return rng.choice(_MOCK_LOOKUP_RESULTS)
        else:
            return f"Error: unknown tool '{base_name}'"
