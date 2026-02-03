"""
Steering Experiment - 因果干预实验

通过修改 SAE 特征来验证其对决策的因果影响。
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from sae import TopKSAE
from controller import AgentLoop, AgentConfig, OutputParser
from controller.tool_schema import DecisionType


class SteeringExperiment:
    """Steering 实验 - 验证特征的因果效应"""
    
    def __init__(
        self,
        sae_path: str,
        model_name: str,
        hook_layer: int,
        device: str = "cuda",
    ):
        """
        Args:
            sae_path: SAE 模型路径
            model_name: LLM 模型名称
            hook_layer: 要干预的层
            device: 设备
        """
        self.device = device
        self.hook_layer = hook_layer
        
        # 加载 SAE
        self.sae = TopKSAE.load(sae_path, device=device)
        self.sae.eval()
        
        # Agent 配置（延迟加载）
        self.model_name = model_name
        self.agent: Optional[AgentLoop] = None
        
        # 结果记录
        self.results: List[Dict[str, Any]] = []
    
    def _init_agent(self):
        """初始化 Agent"""
        if self.agent is not None:
            return
        
        config = AgentConfig(
            model_name_or_path=self.model_name,
            device=self.device,
            cache_activations=False,  # 不需要缓存
        )
        self.agent = AgentLoop(config)
        self.agent.load_model()
    
    def run_steering(
        self,
        prompts: List[str],
        feature_indices: List[int],
        strengths: List[float],
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """运行 steering 实验
        
        Args:
            prompts: 测试 prompt 列表
            feature_indices: 要 steering 的特征索引
            strengths: steering 强度列表
            output_dir: 输出目录
            
        Returns:
            实验结果
        """
        self._init_agent()
        
        results = {
            "num_prompts": len(prompts),
            "feature_indices": feature_indices,
            "strengths": strengths,
            "experiments": [],
        }
        
        # Baseline（无干预）
        print("Running baseline...")
        baseline_results = self._run_batch(prompts, intervention=None)
        results["baseline"] = baseline_results
        
        # 对每个特征和强度进行实验
        for feature_idx in tqdm(feature_indices, desc="Features"):
            for strength in strengths:
                print(f"Testing feature {feature_idx}, strength {strength}")
                
                intervention = {
                    "feature_idx": feature_idx,
                    "strength": strength,
                }
                
                exp_results = self._run_batch(prompts, intervention)
                exp_results["feature_idx"] = feature_idx
                exp_results["strength"] = strength
                
                # 计算决策翻转率
                flip_rate = self._compute_flip_rate(
                    baseline_results["decisions"],
                    exp_results["decisions"],
                )
                exp_results["flip_rate"] = flip_rate
                
                results["experiments"].append(exp_results)
        
        # 保存结果
        if output_dir:
            self._save_results(results, output_dir)
        
        return results
    
    def _run_batch(
        self,
        prompts: List[str],
        intervention: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """批量运行，可选干预"""
        decisions = []
        responses = []
        
        # 注册 hook
        hook_handle = None
        if intervention:
            hook_handle = self._register_steering_hook(
                intervention["feature_idx"],
                intervention["strength"],
            )
        
        try:
            for prompt in prompts:
                result = self.agent.run_episode(instruction=prompt)
                decisions.append(result.final_decision.value)
                responses.append(result.final_response)
        finally:
            if hook_handle:
                hook_handle.remove()
        
        # 统计
        call_rate = sum(1 for d in decisions if d == "call") / len(decisions)
        
        return {
            "decisions": decisions,
            "responses": responses,
            "call_rate": call_rate,
            "no_call_rate": 1 - call_rate,
        }
    
    def _register_steering_hook(
        self,
        feature_idx: int,
        strength: float,
    ):
        """注册 steering hook"""
        decoder_vector = self.sae.decoder.weight[:, feature_idx].to(self.device)
        
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # 添加 steering
            steered = hidden_states + strength * decoder_vector
            
            if isinstance(output, tuple):
                return (steered,) + output[1:]
            return steered
        
        # 注册到目标层
        layer = self.agent.model.model.layers[self.hook_layer]
        return layer.register_forward_hook(hook)
    
    def _compute_flip_rate(
        self,
        baseline_decisions: List[str],
        steered_decisions: List[str],
    ) -> Dict[str, float]:
        """计算决策翻转率"""
        call_to_no_call = 0
        no_call_to_call = 0
        unchanged = 0
        
        for base, steered in zip(baseline_decisions, steered_decisions):
            if base == steered:
                unchanged += 1
            elif base == "call" and steered == "no_call":
                call_to_no_call += 1
            elif base == "no_call" and steered == "call":
                no_call_to_call += 1
        
        total = len(baseline_decisions)
        
        return {
            "call_to_no_call": call_to_no_call / total,
            "no_call_to_call": no_call_to_call / total,
            "unchanged": unchanged / total,
            "total_flip_rate": (call_to_no_call + no_call_to_call) / total,
        }
    
    def run_ablation(
        self,
        prompts: List[str],
        feature_indices: List[int],
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """运行 ablation 实验（将特征置零）
        
        等价于 steering with negative strength
        """
        # Ablation 使用负强度
        strengths = [-1.0, -2.0, -5.0]
        return self.run_steering(
            prompts, feature_indices, strengths, output_dir
        )
    
    def find_optimal_strength(
        self,
        prompts: List[str],
        feature_idx: int,
        target_direction: str = "increase_call",
        strength_range: Tuple[float, float] = (-10.0, 10.0),
        steps: int = 20,
    ) -> Tuple[float, float]:
        """找到最优 steering 强度
        
        Args:
            prompts: 测试 prompts
            feature_idx: 特征索引
            target_direction: 目标方向 ("increase_call" 或 "decrease_call")
            strength_range: 强度范围
            steps: 搜索步数
            
        Returns:
            (optimal_strength, achieved_rate)
        """
        self._init_agent()
        
        strengths = torch.linspace(
            strength_range[0], strength_range[1], steps
        ).tolist()
        
        best_strength = 0.0
        best_rate = 0.0
        
        for strength in tqdm(strengths, desc="Searching"):
            intervention = {
                "feature_idx": feature_idx,
                "strength": strength,
            }
            
            results = self._run_batch(prompts, intervention)
            
            if target_direction == "increase_call":
                rate = results["call_rate"]
            else:
                rate = results["no_call_rate"]
            
            if rate > best_rate:
                best_rate = rate
                best_strength = strength
        
        return best_strength, best_rate
    
    def _save_results(self, results: Dict[str, Any], output_dir: str):
        """保存结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / "steering_results.json", "w") as f:
            # 不保存完整 responses（太大）
            save_results = results.copy()
            save_results["baseline"] = {
                k: v for k, v in results["baseline"].items() 
                if k != "responses"
            }
            for exp in save_results["experiments"]:
                exp.pop("responses", None)
            
            json.dump(save_results, f, indent=2)
        
        print(f"Results saved to {output_path}")


class AblationExperiment:
    """Ablation 实验 - 特征抑制"""
    
    def __init__(
        self,
        sae_path: str,
        device: str = "cuda",
    ):
        self.sae = TopKSAE.load(sae_path, device=device)
        self.device = device
    
    def ablate_features(
        self,
        activations: torch.Tensor,
        feature_indices: List[int],
    ) -> torch.Tensor:
        """消融指定特征
        
        Args:
            activations: [batch, hidden_dim] 输入激活
            feature_indices: 要消融的特征索引
            
        Returns:
            ablated: 消融后的激活
        """
        activations = activations.to(self.device)
        
        with torch.no_grad():
            # 编码
            sae_acts = self.sae.encode(activations)
            
            # 消融（置零）
            for idx in feature_indices:
                sae_acts[:, idx] = 0
            
            # 解码
            reconstructed = self.sae.decode(sae_acts)
        
        return reconstructed
    
    def measure_ablation_effect(
        self,
        activations: torch.Tensor,
        feature_indices: List[int],
    ) -> Dict[str, float]:
        """测量消融效果"""
        original = activations.to(self.device)
        ablated = self.ablate_features(activations, feature_indices)
        
        # 计算差异
        diff = (original - ablated).norm(dim=-1).mean()
        relative_diff = diff / original.norm(dim=-1).mean()
        
        return {
            "absolute_diff": diff.item(),
            "relative_diff": relative_diff.item(),
            "num_ablated_features": len(feature_indices),
        }
