# 研究问题总结：SAE 二阶段训练数据集选择

## 项目背景

本项目研究 LLM 工具调用的机制可解释性（Mechanistic Interpretability），验证三个核心假设：
1. **Tool-call Gating 假设**：LLM 内部存在稀疏 latent features 作为工具调用的门控信号
2. **Evidence Accumulation 假设**：门控是随"信息缺口感知"连续积累的，非单点触发
3. **Causal Controllability 假设**：对门控 features 的干预可显著改变 tool-call 行为

**目标模型**：Qwen3.5-4B/9B
**SAE 类型**：TopK SAE，两阶段训练

完整架构设计参见：`Agent-Tool-Use-MI/docs/architecture.md`

---

## SAE 两阶段训练设计

### Stage 1（已确定）
- **数据集**：OpenWebText2（50-100M tokens）
- **目的**：学习通用语言表征，建立 SAE 字典初始化
- **无需选择**，直接使用

### Stage 2（待确定）
- **目的**：在 Stage 1 基础上微调，学习 tool-use 决策相关的特征
- **关键要求**：数据集必须能提供清晰的 **action boundary 激活**，即模型在"是否调用工具"这一决策节点上的 hidden states
- **当前问题**：需要从以下 5 个候选数据集中选择合适的组合

---

## 候选数据集分析

### 1. When2Call ⭐（当前架构首选）

**位置**：`When2Call/`，论文：`Papers/when2call.pdf`

**任务格式**：
- 输入：用户问题 + 工具规格（名称、描述、参数）
- 输出：4 类决策标签——`tool_call` / `direct` / `request_for_info` / `cannot_answer`

**数据规模**：
| 分割 | 样本量 |
|------|--------|
| Test MCQ | 3,652 |
| Test LLM-Judge | 300 |
| Train SFT | 15,000 |
| Train Preference | 9,000 |
| **合计** | **27,952** |

**数据文件**：
```
When2Call/data/test/when2call_test_mcq.jsonl
When2Call/data/test/when2call_test_llm_judge.jsonl
When2Call/data/train/when2call_train_sft.jsonl
When2Call/data/train/when2call_train_pref.jsonl
```

**优点**：
- 唯一具有**显式 CALL vs NO_CALL 标签**的数据集，且标签粒度达 4 类
- 训练集 27K 样本，体量充足
- 决策边界干净，不混入参数正确性等干扰信号
- 数据源自 BFCL v2 + XLAM，工具多样性良好

**缺点**：
- 单轮决策，缺乏多步 agent 轨迹（对 Evidence Accumulation 假设验证有限）
- 合成数据，真实 API 执行性未验证

**SAE 适用性**：★★★★★（最适合学习 tool-call gate features）

---

### 2. Berkeley Function Calling Leaderboard (BFCL)

**位置**：`gorilla-main/berkeley-function-call-leaderboard/`，论文：`Papers/BFCL.pdf`

**任务格式**：
- 输入：自然语言问题 + API/函数列表
- 输出：精确的函数调用（含参数）或无调用

**数据规模**：23,000+ 评估样本，覆盖 1,000+ 真实 API

**优点**：
- 真实 API 覆盖广（Google、AWS、GitHub 等）
- v4 增加了"Irrelevance"类别（对应 NO_CALL）
- AST 解析验证可执行性

**缺点**：
- **无显式 CALL vs NO_CALL 标签**，需后处理推导
- 决策标签混入了"参数正确性"噪声，决策边界不纯
- 评估集为主，训练集有限

**SAE 适用性**：★★★☆☆（需标签清洗，可作辅助）

---

### 3. Tau²-Bench

**位置**：`tau2-bench-main/`，论文：`Papers/tau2bench.pdf`

**任务格式**：
- 仿真多轮对话（Agent-User 交互），含工具调用轨迹
- 领域：Airlines（50 tasks）、Retail（114）、Telecom（2,285）
- 通过任务成功率（而非逐步标注）评估

**优点**：
- 多步 agent 轨迹，贴近 Evidence Accumulation 假设
- 有真实 user-agent 交互动态

**缺点**：
- **无逐步 CALL vs NO_CALL 标签**
- 长上下文使单点激活提取困难
- 主要用于 RL/多智能体评估，不适合 SAE 监督训练

**SAE 适用性**：★★☆☆☆（标签问题严重，需大量后处理）

---

### 4. VitaBench

**位置**：`vitabench-main/`，论文：`Papers/vitabench.pdf`

**任务格式**：
- 真实场景服务代理任务（外卖、线下购物、在线旅游）
- 400 个任务，66 种工具（27 写工具 + 33 读工具）
- 用 Rubric-based 评估（无显式逐步标注）

**优点**：
- 真实场景复杂度高，环境上下文丰富
- 工具读写分离，可能有不同的 gate 特征

**缺点**：
- **无显式决策标签**
- 任务量少（400），不足以训练 SAE
- 中文原始数据，英文翻译版本

**SAE 适用性**：★★☆☆☆（数据量和标签均不足）

---

### 5. Qwen-Agent

**位置**：`Qwen-Agent-main/`

**性质**：这是一个 agent **开发框架**，不是标注数据集

**用途**：可用于运行 agent、调用工具、生成轨迹数据
**SAE 适用性**：N/A（无标注数据）

---

## 核心问题

当前需要决策的问题如下：

### 问题 1：Stage 2 数据集组合策略

以下哪种策略更合适？

| 方案 | 数据集 | 优点 | 风险 |
|------|--------|------|------|
| **A（单一纯净）** | 仅 When2Call | 标签纯净，决策边界清晰 | 分布单一，泛化性未知 |
| **B（混合扩充）** | When2Call + BFCL（需标签清洗） | 工具多样性更强 | 标签噪声引入 |
| **C（轨迹补充）** | When2Call + Tau²-Bench 轨迹后处理 | 捕获多步积累信号 | 后处理成本高 |

### 问题 2：BFCL 的 NO_CALL 标签怎么提取？

BFCL 本身没有显式"不应该调用"的标签（v4 的 Irrelevance 类别是否足够？），需要确认：
- v4 Irrelevance 样本量有多少？
- 是否可以与 tool_call 样本配对形成 1:1 CALL/NO_CALL？

### 问题 3：多步轨迹数据对 Evidence Accumulation 验证是否必要？

架构文档中 Fig 2 需要展示"gate feature 随 agent step 演化"，但 When2Call 是单轮决策数据。

- 是否需要补充 Tau²-Bench / VitaBench 的多步轨迹数据来验证 H2？
- 或者可以用 When2Call 构造 **伪多步序列**（将多个单轮样本串联）来近似？

### 问题 4：数据量是否足够？

Stage 2 训练预计需要的激活数据量：
- When2Call 训练集：15K SFT + 9K Preference = 24K 样本
- 考虑 CALL:NO_CALL 平衡采样，有效样本约 **12K × 2 = 24K**
- 目标模型（Llama-3-8B）的 hidden_size = 4096，SAE dictionary_size = 32768
- **24K 样本是否足够训练一个稳健的 TopK SAE？**

---

## 当前架构对数据的要求（来自 architecture.md）

```
Stage 2 输入：
- 数据：When2Call / BFCL 任务的 action boundary 激活
- 样本平衡：CALL : NO_CALL ≈ 1:1
- 学习率：5e-5（比 Stage 1 小）
- 训练轮数：1

Stage 2 运行命令（示例）：
python -m run.cache_activations train \
    --model meta-llama/Llama-3-8B-Instruct \
    --dataset when2call \
    --layers 24 27 \
    --stage1-dir ./outputs/sae_checkpoints/stage1 \
    --output-dir ./outputs/sae_checkpoints/stage2 \
    --balance
```

激活提取位置：模型 3/4 或 5/6 位置的残差流（对应 Llama-3-8B 的第 24、27 层）

---

## 需要回答的具体任务

1. **分析 BFCL v4 的 Irrelevance 数据**：统计其样本量，判断是否可构成有效的 NO_CALL 集合
2. **评估 When2Call 24K 样本**对 SAE Stage 2 训练的充分性（参考 SAE 训练规模文献）
3. **提出 Stage 2 数据集最终方案**，说明理由，包括：
   - 具体使用哪些数据文件
   - CALL/NO_CALL 如何构造及平衡
   - 是否需要补充多步轨迹数据

---

## 相关文件索引

| 文件/目录 | 用途 |
|-----------|------|
| `Agent-Tool-Use-MI/docs/architecture.md` | 完整系统架构设计 |
| `Agent-Tool-Use-MI/tasks/when2call_adapter.py` | When2Call 数据适配器 |
| `Agent-Tool-Use-MI/tasks/bfcl_adapter.py` | BFCL 数据适配器 |
| `Agent-Tool-Use-MI/sae/train_sae.py` | SAE 训练主脚本 |
| `Agent-Tool-Use-MI/run/cache_activations.py` | 激活流式处理 |
| `When2Call/data/train/when2call_train_sft.jsonl` | When2Call 主训练集 |
| `gorilla-main/berkeley-function-call-leaderboard/` | BFCL 数据集 |
| `Papers/when2call.pdf` | When2Call 论文 |
| `Papers/BFCL.pdf` | BFCL 论文 |
| `Papers/tau2bench.pdf` | Tau²-Bench 论文 |
| `Papers/vitabench.pdf` | VitaBench 论文 |
| `Papers/deepplanning.pdf` | Deep Planning 论文 |
