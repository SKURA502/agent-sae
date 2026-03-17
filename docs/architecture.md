# Tool-use Mechanistic Basis 代码架构设计

## 一、研究目标

验证三个核心假设：
1. **Tool-call Gating 假设**：LLM 内部存在稀疏 latent features 作为工具调用的门控信号
2. **Evidence Accumulation 假设**：门控是随"信息缺口感知"连续积累的，非单点触发
3. **Causal Controllability 假设**：对门控 features 的干预可显著改变 tool-call 行为

---

## 二、技术栈

| 模块          | 技术选型                                          |
| ------------- | ------------------------------------------------- |
| Agent Runtime | Python + Pydantic（工具参数校验）                 |
| 模型推理      | HuggingFace Transformers / vLLM                   |
| 激活抓取/干预 | dictionary_learning 库                            |
| SAE 可视化    | SAEDashboard                                      |
| SAE 训练      | TopK SAE（两阶段训练：预训练语料 + Tool-use）     |
| 目标模型      | **Qwen3.5-4B**（主）/ **Qwen3.5-9B**（扩展对比） |
| 预训练语料    | OpenWebText2（50-100M tokens）                    |

> **注**：仅使用 Qwen3.5 系列，不引入 Llama/Gemma，以降低跨模型工程成本。

---

## 三、目录结构

```
Agent-Tool-Use-MI/
├── configs/                    # 配置文件
│   ├── model_config.yaml       # 模型路径、层数、hook点
│   ├── sae_config.yaml         # SAE超参（两阶段训练配置）
│   └── task_config.yaml        # 数据集路径、采样参数
│
├── controller/                 # Agent控制器
│   ├── __init__.py
│   ├── agent_loop.py           # 最小 agent 闭环（prompt→LLM→action→env→obs）
│   ├── tool_schema.py          # 工具定义（Pydantic schemas）
│   ├── output_parser.py        # JSON输出解析、decision分类
│   └── sandbox_tools/          # 沙盒工具实现
│       ├── search.py           # 固定文档库检索
│       ├── calculator.py       # 表达式计算
│       ├── lookup.py           # 键值查询
│       └── tool_utils.py       # 噪声注入（p_fail, p_empty, p_corrupt）
│
├── tasks/                      # 数据集适配器
│   ├── __init__.py
│   ├── base_adapter.py         # 统一接口：输入→prompt，输出→decision_label
│   ├── when2call_adapter.py    # When2Call 数据集适配（⚠️ 待修复，见下方说明）
│   ├── bfcl_adapter.py         # BFCL 数据集适配
│   └── synthetic_generator.py  # 合成任务生成（可控难度）
│
├── run/                        # 实验运行脚本
│   ├── generate_rollouts.py    # 生成 episode（流式激活，不保存到磁盘）
│   ├── cache_activations.py    # 流式激活处理工具（运行时推理）
│   └── rollout_logger.py       # 结构化日志（JSONL格式）
│
├── sae/                        # SAE 训练与分析
│   ├── __init__.py
│   ├── train_sae.py            # SAE 训练主脚本（支持两阶段训练）
│   ├── sae_model.py            # TopK SAE 模型定义
│   ├── pretrain_data.py        # 预训练语料数据加载（OpenWebText2）
│   ├── feature_extraction.py   # SAE 特征提取
│   └── feature_scoring.py      # 特征与 decision 的相关性打分
│
├── analysis/                   # 机制分析与干预实验
│   ├── __init__.py
│   ├── correlation_analysis.py # feature→decision 互信息/AUROC
│   ├── linear_probe.py         # 少量 features 线性预测 tool_call
│   ├── steering.py             # 放大 gate feature → 增加 tool-call
│   └── visualization.py        # 核心图表生成
│
├── data/                       # 数据存储（相对于项目根目录）
│   ├── raw/                    # 原始数据集
│   ├── processed/              # 处理后的统一格式
│   └── rollouts/               # 生成的 episode 日志（JSONL）
│
├── outputs/                    # 实验输出
│   ├── sae_checkpoints/        # SAE 模型权重
│   │   ├── stage1/             # Stage 1 检查点（预训练语料）
│   │   └── stage2/             # Stage 2 检查点（Tool-use）
│   ├── analysis_results/       # 分析结果（CSV/JSON）
│   └── figures/                # 生成的图表
│
├── scripts/                    # 便捷脚本
│   ├── run_pipeline.sh         # 完整流水线脚本
│   └── run_steering.sh         # steering 实验脚本
│
├── requirements.txt
└── docs/                       # 项目文档
    ├── architecture.md         # 本文件
    ├── assessment.md           # 项目评估（数据、代码、可行性）
    ├── mission.md              # 任务清单与时间规划
    ├── question.md             # 研究问题与数据集分析
    └── ideaA.md                # 研究方向探索笔记
```

---

## 四、核心模块设计

### 4.1 Agent Controller

```
输入: prompt + tool_schema
  ↓
LLM 推理 (内存中缓存 hidden states)
  ↓
output_parser: 解析输出 → decision ∈ {CALL, NO_CALL}
  ↓
如果 CALL → sandbox_tools 执行 → 返回 tool_result
  ↓
循环直到终止条件
  ↓
流式传输激活到 SAE 训练（不保存到磁盘）
```

**关键设计**：
- 严格 JSON 输出格式（避免决策边界噪声）
- 工具名随机化（避免 label shortcut）
- 支持噪声注入（`p_fail`, `p_empty`, `p_corrupt`）
- **流式激活处理**：激活在内存中处理，不写入磁盘
- **Qwen3.5 thinking 模式**：推理时须关闭 thinking（`thinking_budget=0`），防止 `<think>...</think>` 内容污染 decision boundary

### 4.2 两阶段 SAE 训练

SAE 训练分为两个阶段：

#### Stage 1: 通用预训练语料（OpenWebText2）

**目的**：学习通用的语言表征特征，建立良好的字典初始化。

| 参数     | 推荐值                                |
| -------- | ------------------------------------- |
| 数据集   | OpenWebText2 (Skylion007/openwebtext) |
| Token 数 | 50-100M                               |
| 序列长度 | 1024                                  |
| 采样位置 | 全部                                  |
| 学习率   | 1e-4                                  |

**运行方式**：
```bash
python -m sae.train_sae stage1 \
    --model Qwen/Qwen3.5-4B \
    --layers $(python -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('Qwen/Qwen3.5-4B'); n=c.num_hidden_layers; print(int(n*3/4), int(n*5/6))") \
    --target-tokens 50000000 \
    --output-dir ./outputs/sae_checkpoints
```

> **层号说明**：不硬编码，用 `model.config.num_hidden_layers` 动态计算模型 3/4 和 5/6 位置。

#### Stage 2: Tool-use 数据（H1 + H3 特征提取）

**目的**：在 Stage 1 基础上微调，学习 tool-use 决策相关的特征，供 H1（gate 存在性）和 H3（因果干预）实验使用。

| 参数     | 推荐值                                                          |
| -------- | --------------------------------------------------------------- |
| 初始化   | Stage 1 检查点                                                  |
| 数据     | When2Call **Pref** 数据的 action boundary 激活（⚠️ 非 SFT 数据） |
| 样本平衡 | CALL : NO_CALL ≈ 1:1                                            |
| 学习率   | **1e-5**（保守，防止过拟合，6K 样本规模下比 5e-5 更稳）         |
| 训练轮数 | 1（配合早停）                                                   |
| 有效样本 | ~6K–10K（见数据说明）                                           |

**When2Call 数据格式说明**（重要）：

| 文件 | 样本量 | 实际标签 | 能否用于 Stage 2 |
|------|--------|---------|-----------------|
| `when2call_train_sft.jsonl` | 15,000 | **全部 NO_CALL** | ❌ 不用 |
| `when2call_train_pref.jsonl` | 9,000 | CALL 3,000 / NO_CALL 6,000 | ✅ 主要来源 |

- CALL 标签：`chosen_response.content` 包含 `<TOOLCALL>[...]</TOOLCALL>`
- NO_CALL 标签：`chosen_response.content` 为纯文本回复
- 激活提取位置：`chosen_response` 序列末尾（决策边界）的残差流

**⚠️ Adapter bug**：`tasks/when2call_adapter.py` 当前代码查找 `should_call` 字段（数据中不存在），导致所有样本被标为 `UNCERTAIN`。Stage 2 训练前必须先修复该适配器，改为解析 `chosen_response` 中的 `<TOOLCALL>` 标记。

**运行方式**：
```bash
python -m run.cache_activations train \
    --model Qwen/Qwen3.5-4B \
    --dataset when2call_pref \
    --layers <3/4_layer> <5/6_layer> \
    --stage1-dir ./outputs/sae_checkpoints/stage1 \
    --output-dir ./outputs/sae_checkpoints/stage2 \
    --balance \
    --lr 1e-5
```

#### H2 专用数据：Evidence Accumulation 轨迹

H2（门控随 step 积累）需要多步序列数据，与 Stage 2 特征提取数据分离。详见 [question.md](question.md) 中的方案讨论，待后续确定。当前优先级：H1 → H3 → H2。

### 4.3 流式激活处理（代替磁盘缓存）

**设计原则**：不保存 hidden states 到磁盘，使用运行时推理 + 流式处理。

**磁盘空间节省**：
- 传统方式：50k episodes × 10 steps × 20 tokens × hidden_dim × 4 bytes ≈ **160GB+**
- 流式方式：0GB（激活不落盘）

**架构**：
```
LLM 推理 → Hook 获取激活 → 内存缓冲区 → SAE 训练
                              ↓
                        缓冲区满时 yield
                              ↓
                        SAE 在线更新
```

**核心类**：
- `ActivationStreamer`：从 LLM 流式提取激活
- `PretrainActivationBuffer`：预训练激活缓冲区
- `ActivationBuffer`：Tool-use 激活缓冲区（支持平衡采样）
- `TwoStageTrainer`：两阶段训练管理器

### 4.4 SAE 训练配置

| 参数                        | Qwen3.5-4B 推荐值                                        |
| --------------------------- | -------------------------------------------------------- |
| dictionary_size             | hidden_size × 8（需确认 Qwen3.5-4B 的 hidden_size）      |
| target_sparsity (TopK 的 K) | hidden_size / 32                                         |
| 训练层                      | `int(n_layers * 3/4)` 和 `int(n_layers * 5/6)`，动态计算 |
| SAE batch_size              | 4096                                                     |
| 缓冲区大小                  | 8192                                                     |

> 确认 Qwen3.5-4B hidden_size：`python -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained('Qwen/Qwen3.5-4B').hidden_size)"`

### 4.5 分析流水线

```
Step 1: 相关性分析（H1）
  - 计算每个 feature f 的 E[f|CALL] - E[f|NO_CALL]
  - 计算 AUROC，筛选 top-K features
  - 目标：少量 features（K=10-50）AUROC > 0.75

Step 2: 可预测性验证（H1）
  - 用 top-K features 训练 logistic regression
  - 目标：小 K (10-50) 达到高 AUC（> 0.80）

Step 3: 语义验证
  - top features 的 activation maximization
  - 区分"信息缺口 gate" vs "格式触发器"

Step 4: 因果干预（H3）
  - Steering: h := h + α × decoder_vector(feature_i)
  - Ablation: h := h - α × decoder_vector(feature_i)
  - 测量 flip rate（目标 > 20%）、质量保持度（perplexity 变化 < 20%）

Step 5: Evidence Accumulation（H2，后续）
  - gate feature 强度随 agent step 演化
  - 数据来源待定（见 question.md）
```

---

## 五、数据集设计

| 数据集           | 用途                                   | 核心指标                  | 实际可用样本              |
| ---------------- | -------------------------------------- | ------------------------- | ------------------------- |
| **OpenWebText2** | Stage 1 预训练（50-100M tokens）       | reconstruction loss       | 充足                      |
| **When2Call Pref** | Stage 2 主实验（H1+H3）              | call/no-call AUROC        | 3K CALL + 6K NO_CALL      |
| **BFCL Irrelevance** | 补充 NO_CALL（1,124 条）           | 决策边界纯净度            | 240 + 884 = 1,124         |
| **BFCL**         | 外部有效性验证（H1 泛化性）            | function calling accuracy | 评测集，不用于训练        |
| **多步轨迹 TBD** | H2 Evidence Accumulation              | gate feature 时序演化     | 待定（见 question.md）    |

统一输入输出接口：
- 输入：`(messages, tools)` — 匹配 When2Call pref 格式
- 输出：`decision ∈ {CALL, NO_CALL}`，CALL 则附带解析的 `tool_name`、`args`

---

## 六、关键实验指标

### 6.1 行为指标
- `tool_call_rate`：工具调用频率
- `tool_call_accuracy`：调用时机正确率（相对于 When2Call chosen_response 标签）
- `task_success_rate`：任务完成率（H2 轨迹实验）
- `step_count`：完成任务所需步数

### 6.2 机制指标
- `feature_separability`：top features 的 AUROC（H1 核心指标）
- `linear_probe_auc`：少量 features 预测 decision 的 AUC（H1）
- `decision_flip_rate`：干预后决策翻转比例（H3 核心指标）
- `quality_preservation`：干预后语言质量保持度（H3）

---

## 七、核心输出（论文图表）

| 图表      | 对应假设 | 内容                                                        | 优先级  |
| --------- | -------- | ----------------------------------------------------------- | ------- |
| **Fig 1** | H1       | top features 的 AUROC 分布 + 小 K 线性模型 AUC 曲线         | 最高    |
| **Fig 2** | H3       | flip rate vs steering 强度 α，附带质量/正确率 tradeoff      | 最高    |
| **Fig 3** | H2       | gate feature 强度随 agent step 演化（成功 vs 失败 episode） | 次高    |

> **优先级说明**：H1+H3 数据来源清晰，先做出来。H2 轨迹数据方案待定，放到 Fig 3。最终 teaser figure 使用 Fig 3 的叙事性展示（gate feature 积累 → 触发调用），当 H2 数据方案确定后填入。

---

## 八、快速开始

### 8.1 环境准备

```bash
conda create -n agent_sae python=3.11
conda activate agent_sae
pip install -r requirements.txt
```

### 8.2 模型参数确认（首次必做）

```bash
python -c "
from transformers import AutoConfig
c = AutoConfig.from_pretrained('Qwen/Qwen3.5-4B')
n = c.num_hidden_layers
h = c.hidden_size
print(f'Layers: {n}')
print(f'Hidden size: {h}')
print(f'Layer 3/4: {int(n*3/4)}')
print(f'Layer 5/6: {int(n*5/6)}')
print(f'SAE dict size (x8): {h*8}')
print(f'TopK K (dim/32): {h//32}')
"
```

### 8.3 完整训练流程

```bash
# Stage 1: 预训练语料 SAE 训练
python -m sae.train_sae stage1 \
    --model Qwen/Qwen3.5-4B \
    --layers <3/4_layer> <5/6_layer> \
    --target-tokens 50000000 \
    --output-dir ./outputs/sae_checkpoints

# Stage 2: Tool-use Pref 数据流式训练（确认 when2call_adapter.py 已修复）
python -m run.cache_activations train \
    --model Qwen/Qwen3.5-4B \
    --dataset when2call_pref \
    --layers <3/4_layer> <5/6_layer> \
    --stage1-dir ./outputs/sae_checkpoints/stage1 \
    --output-dir ./outputs/sae_checkpoints/stage2 \
    --balance \
    --lr 1e-5

# 分析与可视化
python -m analysis.correlation_analysis \
    --sae-path ./outputs/sae_checkpoints/stage2/stage2_layer<N>_final.pt
```

---

## 九、已知问题与注意事项

| 问题 | 严重性 | 说明 |
|------|--------|------|
| `when2call_adapter.py` 标签提取错误 | 🔴 阻断 | 查找不存在的 `should_call` 字段；需改为解析 `chosen_response` 中的 `<TOOLCALL>` |
| Stage 2 训练层号未确定 | 🟡 待定 | 需先运行上方脚本确认 Qwen3.5-4B 的层数 |
| Qwen3.5 thinking 模式干扰 | 🟡 待验证 | 推理时建议关闭 thinking（`thinking_budget=0`） |
| Stage 2 数据量偏小（6K） | 🟡 可接受 | 两阶段训练缓解，配合 lr=1e-5 和早停 |
| H2 数据方案未定 | 🟢 后续 | 见 [question.md](question.md) |
