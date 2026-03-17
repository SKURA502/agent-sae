"""
BFCL Adapter - Berkeley Function Calling Leaderboard 数据集适配器

支持 BFCL v4 扁平格式：文件直接位于 data_path 下，命名为 BFCL_v4_{category}.json。

标签推断规则（按文件名）：
  含 "irrelevance" → NO_CALL
  其余（simple, relevance, parallel 等）→ CALL

文件规模（已核实）：
  BFCL_v4_irrelevance.json       240 条  NO_CALL
  BFCL_v4_live_irrelevance.json  884 条  NO_CALL
  BFCL_v4_live_relevance.json     16 条  CALL
  BFCL_v4_simple_python.json     400 条  CALL
  BFCL_v4_live_simple.json       258 条  CALL
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_adapter import BaseAdapter, TaskSample, DecisionLabel


# 文件名包含以下关键字时标注为 NO_CALL
_NO_CALL_KEYWORDS = frozenset(["irrelevance"])


def _label_from_filename(filename: str) -> DecisionLabel:
    """从文件名推断标签。"""
    stem = Path(filename).stem.lower()
    for kw in _NO_CALL_KEYWORDS:
        if kw in stem:
            return DecisionLabel.NO_CALL
    return DecisionLabel.CALL


def _category_from_filename(filename: str) -> str:
    """从文件名提取简短类别名。"""
    stem = Path(filename).stem.lower()  # e.g. bfcl_v4_live_irrelevance
    for prefix in ("bfcl_v4_live_", "bfcl_v4_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


class BFCLAdapter(BaseAdapter):
    """BFCL 数据集适配器（支持 v4 扁平格式及旧版子目录格式）。"""

    @property
    def name(self) -> str:
        return "BFCL"

    def __init__(
        self,
        data_path: str,
        split: str = "test",
        num_samples: int = -1,
        seed: int = 42,
        categories: Optional[List[str]] = None,
    ):
        """
        Args:
            data_path: 数据集根目录（含 BFCL_v4_*.json 文件）
            split: 数据集划分（不影响 v4 文件加载，保留兼容性）
            num_samples: 采样数量（-1 表示全部）
            seed: 随机种子
            categories: 要加载的类别关键字，如 ["irrelevance", "simple_python"]。
                       None 表示加载所有检测到的 BFCL_v4_*.json 文件。
        """
        super().__init__(data_path, split, num_samples, seed)
        self.categories = categories  # None = 全部

    # ──────────────────────── data loading ───────────────────────────

    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """加载 BFCL 数据。优先 v4 扁平格式，回退旧版子目录格式。"""
        v4_files = sorted(self.data_path.glob("BFCL_v4_*.json"))
        if v4_files:
            return self._load_v4(v4_files)
        return self._load_legacy()

    def _load_v4(self, json_files: List[Path]) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        for fp in json_files:
            if self.categories is not None:
                file_stem = fp.stem.lower()
                if not any(cat.lower() in file_stem for cat in self.categories):
                    continue
            data = self._read_json(fp)
            for item in (data if isinstance(data, list) else [data]):
                if not isinstance(item, dict):
                    continue
                item.setdefault("_source_file", fp.name)
                item.setdefault("_category", _category_from_filename(fp.name))
                samples.append(item)
        return samples

    def _load_legacy(self) -> List[Dict[str, Any]]:
        """旧格式：category 子目录下的 JSON 文件。"""
        cats = self.categories or ["simple", "parallel", "multiple", "irrelevance"]
        samples: List[Dict[str, Any]] = []
        for cat in cats:
            cat_path = self.data_path / cat
            if not cat_path.exists():
                continue
            for fp in sorted(cat_path.glob("*.json")):
                data = self._read_json(fp)
                for item in (data if isinstance(data, list) else [data]):
                    if not isinstance(item, dict):
                        continue
                    item.setdefault("_source_file", fp.name)
                    item.setdefault("_category", cat)
                    samples.append(item)
        # 若子目录也没找到，扫描根目录 *.json
        if not samples:
            for fp in sorted(self.data_path.glob("*.json")):
                data = self._read_json(fp)
                for item in (data if isinstance(data, list) else [data]):
                    if not isinstance(item, dict):
                        continue
                    item.setdefault("_source_file", fp.name)
                    item.setdefault("_category", "unknown")
                    samples.append(item)
        return samples

    @staticmethod
    def _read_json(fp: Path) -> Any:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

    # ──────────────────────── sample conversion ──────────────────────

    def _convert_sample(self, raw_sample: Dict[str, Any], idx: int) -> TaskSample:
        """转换 BFCL 样本（v4 及旧格式）。"""
        # 用户请求（多种字段名）
        instruction: str = (
            raw_sample.get("user_request")
            or raw_sample.get("user_query")
            or ""
        )
        if not instruction:
            q_list = raw_sample.get("question") or []
            if isinstance(q_list, list):
                for msg in reversed(q_list):
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        instruction = str(msg.get("content", ""))
                        break
            elif isinstance(q_list, str):
                instruction = q_list
        instruction = str(instruction) if instruction else ""

        # 函数定义（function / functions 两种命名）
        functions = raw_sample.get("function") or raw_sample.get("functions") or []
        if not isinstance(functions, list):
            functions = [functions] if functions else []
        tool_schemas: List[Dict] = [f for f in functions if isinstance(f, dict)]
        available_tools: List[str] = [f.get("name", "") for f in tool_schemas]
        available_tools = [n for n in available_tools if n]

        # 从文件名推断标签
        source_file: str = raw_sample.get("_source_file", "")
        label = _label_from_filename(source_file) if source_file else DecisionLabel.CALL

        # Ground truth
        ground_truth = raw_sample.get("ground_truth") or raw_sample.get("expected_output") or []
        if not isinstance(ground_truth, list):
            ground_truth = [ground_truth]

        expected_tool: Optional[str] = None
        expected_args: Optional[Dict] = None
        if ground_truth:
            first = ground_truth[0]
            if isinstance(first, dict):
                expected_tool = first.get("name") or first.get("function")
                expected_args = first.get("arguments") or first.get("parameters")
            elif isinstance(first, str):
                expected_tool = first

        category: str = raw_sample.get("_category") or raw_sample.get("category") or "unknown"

        return TaskSample(
            sample_id=raw_sample.get("id", f"bfcl_{idx:06d}"),
            instruction=instruction,
            context=None,
            tool_schemas=tool_schemas,
            available_tools=available_tools,
            label=label,
            expected_tool=expected_tool,
            expected_args=expected_args,
            source_dataset="bfcl",
            category=category,
            metadata={
                "ground_truth_full": ground_truth,
                "num_functions": len(functions),
                "source_file": source_file,
            },
        )
