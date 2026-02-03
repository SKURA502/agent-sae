#!/usr/bin/env python
"""
快速测试脚本 - 验证各模块是否正常工作
"""

import sys
from pathlib import Path

# 添加项目根目录到 path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """测试所有模块导入"""
    print("Testing imports...")
    
    try:
        from controller import AgentLoop, AgentConfig, OutputParser
        print("  ✓ controller module")
    except Exception as e:
        print(f"  ✗ controller module: {e}")
    
    try:
        from controller.tool_schema import ToolSchema, ToolCall, DecisionType
        print("  ✓ tool_schema")
    except Exception as e:
        print(f"  ✗ tool_schema: {e}")
    
    try:
        from tasks import When2CallAdapter, BFCLAdapter, SyntheticGenerator
        print("  ✓ tasks module")
    except Exception as e:
        print(f"  ✗ tasks module: {e}")
    
    try:
        from run import RolloutGenerator, ActivationCacher
        print("  ✓ run module")
    except Exception as e:
        print(f"  ✗ run module: {e}")
    
    try:
        from sae import TopKSAE, SAETrainer, FeatureExtractor
        print("  ✓ sae module")
    except Exception as e:
        print(f"  ✗ sae module: {e}")
    
    try:
        from analysis import CorrelationAnalyzer, LinearProbe, SteeringExperiment, Visualizer
        print("  ✓ analysis module")
    except Exception as e:
        print(f"  ✗ analysis module: {e}")


def test_tool_schema():
    """测试工具 schema"""
    print("\nTesting tool schema...")
    
    from controller.tool_schema import ToolSchema, ToolCall, ToolResult, DecisionType
    
    # 创建工具
    tool = ToolSchema(
        name="calculator",
        description="Perform calculations",
        parameters={"expression": {"type": "string", "description": "Math expression"}},
        required_params=["expression"],
    )
    print(f"  ✓ Created tool: {tool.name}")
    
    # 创建调用
    call = ToolCall(name="calculator", arguments={"expression": "2+2"})
    print(f"  ✓ Created call: {call.name}({call.arguments})")
    
    # 创建结果
    result = ToolResult(tool_call=call, output="4", success=True)
    print(f"  ✓ Created result: {result.output}")


def test_output_parser():
    """测试输出解析器"""
    print("\nTesting output parser...")
    
    from controller.output_parser import OutputParser
    
    parser = OutputParser()
    
    # 测试 JSON 格式
    json_output = '''
    Let me help you with that.
    
    {"action": "call_tool", "tool_name": "calculator", "arguments": {"expression": "123 * 456"}}
    '''
    
    result = parser.parse(json_output)
    print(f"  ✓ Parsed action: {result.decision.value}")
    if result.tool_call:
        print(f"  ✓ Tool call: {result.tool_call.name}")
    
    # 测试直接回答
    direct_output = '''
    {"action": "respond", "response": "The capital of France is Paris."}
    '''
    
    result = parser.parse(direct_output)
    print(f"  ✓ Parsed direct response: {result.decision.value}")


def test_synthetic_generator():
    """测试合成数据生成"""
    print("\nTesting synthetic generator...")
    
    from tasks import SyntheticGenerator
    
    generator = SyntheticGenerator()
    samples = generator.get_samples(5)
    
    for i, sample in enumerate(samples):
        label_str = "CALL" if sample.label.value == 1 else "NO_CALL"
        print(f"  Sample {i+1}: [{label_str}] {sample.instruction[:50]}...")
    
    print(f"  ✓ Generated {len(samples)} samples")


def test_sae_model():
    """测试 SAE 模型"""
    print("\nTesting SAE model...")
    
    import torch
    from sae import TopKSAE, SAEConfig
    
    config = SAEConfig(
        hidden_size=128,
        dict_size=512,
        k=8,
    )
    
    sae = TopKSAE(config)
    print(f"  ✓ Created SAE: dict_size={config.dict_size}, k={config.k}")
    
    # 测试前向传播
    x = torch.randn(10, 128)
    reconstructed, sae_acts = sae(x)
    
    print(f"  ✓ Forward pass: input {x.shape} -> acts {sae_acts.shape} -> output {reconstructed.shape}")
    
    # 检查稀疏性
    sparsity = (sae_acts != 0).float().sum(dim=-1).mean()
    print(f"  ✓ Average active features: {sparsity:.1f} (target: {config.k})")


def test_visualization():
    """测试可视化"""
    print("\nTesting visualization...")
    
    import numpy as np
    from analysis import Visualizer
    
    viz = Visualizer("./test_outputs", style="paper")
    print(f"  ✓ Created visualizer")
    
    # 生成测试数据
    auroc = np.random.beta(2, 2, 100).tolist()
    mean_diff = np.random.randn(100).tolist()
    
    try:
        viz.plot_feature_separability(auroc, mean_diff, filename="test_separability.pdf")
        print("  ✓ Generated separability plot")
    except Exception as e:
        print(f"  ⚠ Visualization skipped (matplotlib not available): {e}")


def main():
    """运行所有测试"""
    print("=" * 50)
    print("Agent SAE Tool-use - Module Tests")
    print("=" * 50)
    
    test_imports()
    test_tool_schema()
    test_output_parser()
    test_synthetic_generator()
    test_sae_model()
    test_visualization()
    
    print("\n" + "=" * 50)
    print("All tests complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
