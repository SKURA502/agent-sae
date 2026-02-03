"""
Rollout Logger - 结构化日志记录

记录 episode 的详细信息，支持 JSONL 格式。
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass
class StepLog:
    """单步日志"""
    step_id: int
    timestamp: float
    
    # 输入
    prompt_tokens: int = 0
    
    # 输出
    model_output: str = ""
    output_tokens: int = 0
    
    # 决策
    decision_type: str = ""  # call / no_call / abstain / clarify
    parse_success: bool = True
    parse_error: Optional[str] = None
    
    # 工具调用（如果有）
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_success: Optional[bool] = None
    tool_result: Optional[str] = None
    tool_error: Optional[str] = None
    
    # 激活信息（元数据，实际张量另存）
    activation_saved: bool = False
    activation_path: Optional[str] = None
    activation_shape: Optional[List[int]] = None


@dataclass
class EpisodeLog:
    """Episode 日志"""
    episode_id: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    # 任务信息
    instruction: str = ""
    context: Optional[str] = None
    available_tools: List[str] = field(default_factory=list)
    
    # 标签（ground truth）
    label: str = ""  # call / no_call
    expected_tool: Optional[str] = None
    expected_args: Optional[Dict[str, Any]] = None
    
    # 结果
    steps: List[StepLog] = field(default_factory=list)
    final_decision: str = ""
    final_response: Optional[str] = None
    success: bool = False
    
    # 统计
    total_steps: int = 0
    total_tool_calls: int = 0
    total_tokens: int = 0
    
    # 元数据
    model_name: str = ""
    source_dataset: str = ""
    sample_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于 JSON 序列化）"""
        result = asdict(self)
        # 处理 StepLog 列表
        result["steps"] = [asdict(s) for s in self.steps]
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeLog":
        """从字典创建"""
        data = data.copy()
        if "steps" in data:
            data["steps"] = [StepLog(**s) for s in data["steps"]]
        return cls(**data)


class RolloutLogger:
    """Rollout 日志管理器"""
    
    def __init__(
        self,
        output_dir: str,
        experiment_name: Optional[str] = None,
        flush_interval: int = 10,
    ):
        """
        Args:
            output_dir: 输出目录
            experiment_name: 实验名称（用于生成文件名）
            flush_interval: 多少条记录后 flush 一次
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        
        self.flush_interval = flush_interval
        self._buffer: List[EpisodeLog] = []
        self._total_logged = 0
        
        # 日志文件路径
        self.log_path = self.output_dir / f"{experiment_name}_rollouts.jsonl"
        self.stats_path = self.output_dir / f"{experiment_name}_stats.json"
        
        # 统计信息
        self.stats = {
            "total_episodes": 0,
            "successful_episodes": 0,
            "total_steps": 0,
            "total_tool_calls": 0,
            "decision_distribution": {"call": 0, "no_call": 0, "other": 0},
            "tool_distribution": {},
            "avg_steps_per_episode": 0.0,
        }
    
    def log_episode(self, episode: EpisodeLog):
        """记录一个 episode"""
        episode.end_time = time.time()
        
        # 更新统计
        self._update_stats(episode)
        
        # 添加到缓冲区
        self._buffer.append(episode)
        self._total_logged += 1
        
        # 检查是否需要 flush
        if len(self._buffer) >= self.flush_interval:
            self.flush()
    
    def _update_stats(self, episode: EpisodeLog):
        """更新统计信息"""
        self.stats["total_episodes"] += 1
        if episode.success:
            self.stats["successful_episodes"] += 1
        
        self.stats["total_steps"] += episode.total_steps
        self.stats["total_tool_calls"] += episode.total_tool_calls
        
        # 决策分布
        if episode.final_decision in ["call", "no_call"]:
            self.stats["decision_distribution"][episode.final_decision] += 1
        else:
            self.stats["decision_distribution"]["other"] += 1
        
        # 工具分布
        for step in episode.steps:
            if step.tool_name:
                self.stats["tool_distribution"][step.tool_name] = \
                    self.stats["tool_distribution"].get(step.tool_name, 0) + 1
        
        # 平均步数
        if self.stats["total_episodes"] > 0:
            self.stats["avg_steps_per_episode"] = \
                self.stats["total_steps"] / self.stats["total_episodes"]
    
    def flush(self):
        """将缓冲区写入磁盘"""
        if not self._buffer:
            return
        
        # 写入日志
        with open(self.log_path, "a", encoding="utf-8") as f:
            for episode in self._buffer:
                f.write(json.dumps(episode.to_dict(), ensure_ascii=False) + "\n")
        
        # 写入统计
        with open(self.stats_path, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        self._buffer = []
    
    def close(self):
        """关闭日志器，确保所有数据写入"""
        self.flush()
        print(f"Logged {self._total_logged} episodes to {self.log_path}")
        print(f"Stats saved to {self.stats_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
    
    def load_logs(self) -> List[EpisodeLog]:
        """加载已保存的日志"""
        if not self.log_path.exists():
            return []
        
        episodes = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    episodes.append(EpisodeLog.from_dict(json.loads(line)))
        
        return episodes
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_step_log(
    step_id: int,
    model_output: str,
    decision_type: str,
    tool_name: Optional[str] = None,
    tool_args: Optional[Dict] = None,
    tool_result: Optional[Dict] = None,
    activation_info: Optional[Dict] = None,
) -> StepLog:
    """创建步骤日志的便捷函数"""
    log = StepLog(
        step_id=step_id,
        timestamp=time.time(),
        model_output=model_output,
        decision_type=decision_type,
        tool_name=tool_name,
        tool_args=tool_args,
    )
    
    if tool_result:
        log.tool_success = tool_result.get("success", False)
        log.tool_result = str(tool_result.get("result", ""))
        log.tool_error = tool_result.get("error")
    
    if activation_info:
        log.activation_saved = True
        log.activation_path = activation_info.get("path")
        log.activation_shape = activation_info.get("shape")
    
    return log
