"""
Rollout Logger - H2 episode 日志记录

RolloutLogger 将 episode 步骤序列化为 JSONL 文件。每行为一个完整 episode 的 JSON 记录。
激活张量不写入日志（由 generate_rollouts.py 单独保存到 .pt 文件）。
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EpisodeLog:
    episode_id: str
    task_prompt: str
    steps: List[Dict]           # 序列化后的步骤（不含激活张量）
    reward: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class RolloutLogger:
    """将 EpisodeLog 追加写入 JSONL 文件。"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.output_dir / "rollouts.jsonl"

    def log_episode(
        self,
        episode_id: str,
        task_prompt: str,
        steps,
        reward: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ) -> EpisodeLog:
        """序列化并记录一个 episode，返回 EpisodeLog。"""
        serialized_steps = [
            {
                "turn": step.turn,
                "decision": step.decision,
                "tool_name": step.tool_name,
                "tool_result": step.tool_result,
                # activations 不序列化（张量），由调用方单独保存
            }
            for step in steps
        ]

        log = EpisodeLog(
            episode_id=episode_id,
            task_prompt=task_prompt,
            steps=serialized_steps,
            reward=reward,
            metadata=metadata or {},
        )
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(log)) + "\n")
        return log

    def load_all(self) -> List[EpisodeLog]:
        """加载所有已记录的 episode。"""
        if not self._path.exists():
            return []
        logs = []
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    logs.append(EpisodeLog(**json.loads(line)))
        return logs
