#!/usr/bin/env python
"""
快速测试脚本 - 验证各模块正确工作

包含：
1. 模块导入测试
2. When2Call pref 格式标签解析测试（含 <TOOLCALL> 识别）
3. BFCL v4 标签推断测试
4. SAE 模型前向传播测试
5. 冒烟测试：100 pref 样本 → 激活提取模拟 → SAE 更新 1 batch
6. 可视化测试
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

PASS = "✓"
FAIL = "✗"
WARN = "⚠"


def test_imports():
    print("Testing imports...")

    for name, mod_path in [
        ("controller", "controller"),
        ("tool_schema", "controller.tool_schema"),
        ("tasks", "tasks"),
        ("run", "run"),
        ("sae", "sae"),
        ("analysis", "analysis"),
    ]:
        try:
            __import__(mod_path)
            print(f"  {PASS} {name}")
        except Exception as e:
            print(f"  {FAIL} {name}: {e}")


def test_when2call_label_parsing():
    """验证 pref 格式 / MCQ 格式标签解析均正确。"""
    print("\nTesting When2Call label parsing...")

    from tasks.when2call_adapter import When2CallAdapter, _has_toolcall_tag
    from tasks.base_adapter import DecisionLabel

    # _has_toolcall_tag
    assert _has_toolcall_tag('<TOOLCALL>{"name":"search"}</TOOLCALL>'), "should detect TOOLCALL"
    assert not _has_toolcall_tag("Just a plain response."), "should not detect in plain text"
    print(f"  {PASS} _has_toolcall_tag")

    # pref 格式 CALL
    raw_call = {
        "id": "t001",
        "messages": [{"role": "user", "content": "Who won the Nobel Prize?"}],
        "tools": [{"name": "search", "parameters": {}}],
        "chosen_response": {
            "role": "assistant",
            "content": "Let me search. <TOOLCALL>{\"name\": \"search\"}</TOOLCALL>",
        },
    }
    adapter = When2CallAdapter.__new__(When2CallAdapter)
    label = adapter._infer_label(raw_call)
    assert label == DecisionLabel.CALL, f"Expected CALL, got {label}"
    print(f"  {PASS} pref CALL via <TOOLCALL>")

    # pref 格式 NO_CALL
    raw_no_call = {
        "id": "t002",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "tools": [],
        "chosen_response": {"role": "assistant", "content": "The answer is 4."},
    }
    label = adapter._infer_label(raw_no_call)
    assert label == DecisionLabel.NO_CALL, f"Expected NO_CALL, got {label}"
    print(f"  {PASS} pref NO_CALL (no <TOOLCALL>)")

    # MCQ 格式 CALL
    raw_mcq_call = {"answer": "tool_call"}
    label = adapter._infer_label(raw_mcq_call)
    assert label == DecisionLabel.CALL, f"Expected CALL, got {label}"
    print(f"  {PASS} MCQ CALL (answer=tool_call)")

    # MCQ 格式 NO_CALL
    raw_mcq_no_call = {"answer": "cannot_answer"}
    label = adapter._infer_label(raw_mcq_no_call)
    assert label == DecisionLabel.NO_CALL, f"Expected NO_CALL, got {label}"
    print(f"  {PASS} MCQ NO_CALL (answer=cannot_answer)")

    # MCQ 格式 UNCERTAIN
    raw_mcq_rfi = {"answer": "request_for_info"}
    label = adapter._infer_label(raw_mcq_rfi)
    assert label == DecisionLabel.UNCERTAIN, f"Expected UNCERTAIN, got {label}"
    print(f"  {PASS} MCQ UNCERTAIN (answer=request_for_info)")


def test_bfcl_label_inference():
    print("\nTesting BFCL v4 label inference...")

    from tasks.bfcl_adapter import _label_from_filename, _category_from_filename
    from tasks.base_adapter import DecisionLabel

    cases = [
        ("BFCL_v4_irrelevance.json", DecisionLabel.NO_CALL, "irrelevance"),
        ("BFCL_v4_live_irrelevance.json", DecisionLabel.NO_CALL, "irrelevance"),
        ("BFCL_v4_simple_python.json", DecisionLabel.CALL, "simple_python"),
        ("BFCL_v4_live_simple.json", DecisionLabel.CALL, "simple"),
        ("BFCL_v4_live_relevance.json", DecisionLabel.CALL, "relevance"),
    ]
    for fname, expected_label, expected_cat in cases:
        label = _label_from_filename(fname)
        cat = _category_from_filename(fname)
        assert label == expected_label, f"{fname}: expected {expected_label}, got {label}"
        assert cat == expected_cat, f"{fname}: expected category '{expected_cat}', got '{cat}'"
    print(f"  {PASS} All 5 v4 filename → label / category mappings correct")


def test_sae_model():
    print("\nTesting SAE model (Qwen3.5-4B dimensions)...")

    import torch
    from sae import TopKSAE, SAEConfig

    # Qwen3.5-4B: hidden_size=2560, dict_size=20480, k=80
    config = SAEConfig(input_dim=2560, dict_size=20480, k=80, device="cpu", dtype="float32")
    sae = TopKSAE(config)
    print(f"  {PASS} Created SAE: dict_size={config.dict_size}, k={config.k}")

    x = torch.randn(16, 2560)
    x_hat, latents = sae(x)
    assert x_hat.shape == x.shape, f"Shape mismatch: {x_hat.shape}"
    n_active = (latents > 0).float().sum(dim=-1).mean().item()
    assert abs(n_active - config.k) < 5, f"Active features {n_active:.1f} far from k={config.k}"
    print(f"  {PASS} Forward pass OK, avg active features: {n_active:.1f} (target: {config.k})")

    loss, info = sae.compute_loss(x)
    assert loss.item() > 0
    print(f"  {PASS} compute_loss OK: loss={loss.item():.4f}")


def test_smoke_100_samples():
    """
    冒烟测试：生成 100 条 mock pref 样本 → 模拟激活提取 → 完成 1 个 SAE 训练 batch。
    不加载真实 LLM，用随机向量代替激活。
    """
    print("\nSmoke test: 100 pref samples → mock activations → 1 SAE batch...")

    import torch
    from tasks.when2call_adapter import When2CallAdapter
    from tasks.base_adapter import DecisionLabel
    from sae import TopKSAE, SAEConfig, SAETrainer, TrainingConfig

    # 1. 生成 mock 数据
    import tempfile, json, random
    tmp = Path(tempfile.mkdtemp())
    mock_path = tmp / "train_pref.jsonl"
    When2CallAdapter.create_mock_data(mock_path, num_samples=100)
    print(f"  {PASS} Created 100 mock pref samples")

    # 2. 加载并验证标签
    adapter = When2CallAdapter(str(tmp), split="train_pref")
    adapter.load()
    samples = list(adapter)
    call_n = sum(1 for s in samples if s.label == DecisionLabel.CALL)
    no_call_n = sum(1 for s in samples if s.label == DecisionLabel.NO_CALL)
    assert len(samples) == 100, f"Expected 100, got {len(samples)}"
    assert call_n + no_call_n == 100, f"Unexpected labels: {call_n} CALL + {no_call_n} NO_CALL"
    print(f"  {PASS} Loaded 100 samples: {call_n} CALL + {no_call_n} NO_CALL")

    # 3. 模拟激活提取（用随机向量代替真实 LLM 输出）
    hidden_size = 2560  # Qwen3.5-4B
    mock_activations = torch.randn(len(samples), hidden_size)
    print(f"  {PASS} Mock activations: {mock_activations.shape}")

    # 4. 创建小型 SAE 并训练 1 batch（CPU 测试用 float32）
    train_config = TrainingConfig(
        input_dim=hidden_size,
        dict_size=hidden_size * 8,
        k=hidden_size // 32,
        learning_rate=5e-4,
        batch_size=len(samples),
        num_epochs=1,
        device="cpu",
        dtype="float32",
        output_dir=str(tmp / "sae_ckpt"),
    )
    trainer = SAETrainer(train_config)
    stats = trainer.train(mock_activations)
    assert stats["train_losses"][-1] < 10.0, f"Loss too high: {stats['train_losses']}"
    print(f"  {PASS} SAE batch training OK: loss={stats['train_losses'][-1]:.4f}")


def test_visualization():
    print("\nTesting visualization...")

    import numpy as np
    from analysis import Visualizer

    viz = Visualizer("./test_outputs", style="paper")
    auroc = np.random.beta(2, 2, 100).tolist()
    mean_diff = np.random.randn(100).tolist()

    try:
        viz.plot_feature_separability(auroc, mean_diff, filename="test_separability.pdf")
        print(f"  {PASS} Generated separability plot")
    except Exception as e:
        print(f"  {WARN} Visualization skipped (matplotlib unavailable?): {e}")


def main():
    print("=" * 55)
    print("Agent-Tool-Use-MI  Quick Tests")
    print("=" * 55)

    test_imports()
    test_when2call_label_parsing()
    test_bfcl_label_inference()
    test_sae_model()
    test_smoke_100_samples()
    test_visualization()

    print("\n" + "=" * 55)
    print("All tests complete!")
    print("=" * 55)


if __name__ == "__main__":
    main()
