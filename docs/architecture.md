# Tool-use Mechanistic Basis 代码架构设计

## 一、研究目标

验证三个核心假设：
1. **Tool-call Gating 假设**：LLM 内部存在稀疏 latent features 作为工具调用的门控信号
2. **Evidence Accumulation 假设**：门控是随"信息缺口感知"连续积累的，非单点触发
3. **Causal Controllability 假设**：对门控 features 的干预可显著改变 tool-call 行为

---

## 二、技术栈

| 模块          | 技术选型                                      |
| ------------- | --------------------------------------------- |
| Agent Runtime | Python + Pydantic（工具参数校验）             |
| 模型推理      | HuggingFace Transformers / vLLM               |
| 激活抓取/干预 | dictionary_learning 库                        |
| SAE 可视化    | SAEDashboard                                  |
| SAE 训练      | TopK SAE（两阶段训练：预训练语料 + Tool-use） |
| 目标模型      | Llama-3-8B, Qwen-3-8B/14B, Gemma-3-4B/12B     |
| 预训练语料    | OpenWebText2（50-100M tokens）                |

---

## 三、目录结构

```
agent_tool_use/
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
│   ├── when2call_adapter.py    # When2Call 数据集适配
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
├── data/                       # 数据存储
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
├── requirements.txt            # 依赖列表
└── README.md                   # 项目说明
```

---

## 四、核心模块设计

### 4.1 Agent Controller

```
输入: prompt + tool_schema
  ↓
LLM 推理 (内存中缓存 hidden states)
  ↓
output_parser: 解析 JSON → decision ∈ {CALL, NO_CALL}
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
    --model meta-llama/Llama-3-8B-Instruct \
    --layers 24 27 \
    --target-tokens 50000000 \
    --output-dir ./outputs/sae_checkpoints
```

#### Stage 2: Tool-use 数据

**目的**：在 Stage 1 基础上微调，学习 tool-use 决策相关的特征。

| 参数     | 推荐值                                       |
| -------- | -------------------------------------------- |
| 初始化   | Stage 1 检查点                               |
| 数据     | When2Call / BFCL 任务的 action boundary 激活 |
| 样本平衡 | CALL : NO_CALL ≈ 1:1                         |
| 学习率   | 5e-5（比 Stage 1 小）                        |
| 训练轮数 | 1                                            |

**运行方式**：
```bash
python -m run.cache_activations train \
    --model meta-llama/Llama-3-8B-Instruct \
    --dataset when2call \
    --layers 24 27 \
    --stage1-dir ./outputs/sae_checkpoints/stage1 \
    --output-dir ./outputs/sae_checkpoints/stage2 \
    --balance
```

### 4.3 流式激活处理（代替磁盘缓存）

**设计原则**：不保存 hidden states 到磁盘，使用运行时推理 + 流式处理。

**磁盘空间节省**：
- 传统方式：50k episodes × 10 steps × 20 tokens × 4096 dim × 4 bytes ≈ **160GB**
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

| 参数                        | 推荐值                                   |
| --------------------------- | ---------------------------------------- |
| dictionary_size             | hidden_size × 8 (e.g., 4096 × 8 = 32768) |
| target_sparsity (TopK 的 K) | hidden_size / 32 (e.g., 4096 / 32 = 128) |
| 训练层                      | 模型 3/4 或 5/6 位置的残差流             |
| SAE batch_size              | 4096                                     |
| 缓冲区大小                  | 8192                                     |

### 4.5 分析流水线

```
Step 1: 相关性分析
  - 计算每个 feature f 的 E[f|CALL] - E[f|NO_CALL]
  - 计算 AUROC，筛选 top-K features

Step 2: 可预测性验证
  - 用 top-K features 训练 logistic regression
  - 目标：小 K (10-50) 达到高 AUC

Step 3: 语义验证
  - top features 的 activation maximization
  - 区分"信息缺口 gate" vs "格式触发器"

Step 4: 因果干预
  - Steering: h := h + α × decoder_vector(feature_i)
  - Ablation: h := h - α × decoder_vector(feature_i)
  - 测量 flip rate、质量保持度
```

---

## 五、数据集设计

| 数据集           | 用途                             | 核心指标                  |
| ---------------- | -------------------------------- | ------------------------- |
| **OpenWebText2** | Stage 1 预训练（50-100M tokens） | reconstruction loss       |
| **When2Call**    | Stage 2 主实验（门控机制最干净） | call/no-call accuracy     |
| **BFCL**         | 外部有效性（格式/选择）          | function calling accuracy |
| **API-Bank**     | 可执行验证（端到端）             | task success rate         |

统一输入输出接口：
- 输入：`(instruction, context, tool_schema)`
- 输出：`decision ∈ {CALL, NO_CALL}`, 如果 CALL 则附带 `tool_name`, `args`

---

## 六、关键实验指标

### 6.1 行为指标
- `tool_call_rate`：工具调用频率
- `tool_call_accuracy`：调用时机正确率
- `task_success_rate`：任务完成率
- `step_count`：完成任务所需步数

### 6.2 机制指标
- `feature_separability`：top features 的 AUROC
- `linear_probe_auc`：少量 features 预测 decision 的 AUC
- `decision_flip_rate`：干预后决策翻转比例
- `quality_preservation`：干预后语言质量保持度

---

## 七、核心输出（论文图表）

### Teaser Figure（Introduction）

**Fig 2 → 放在 Introduction 作为 teaser**，搭配 Fig 3 的 mini inset。

理由：Fig 2 展示 gate feature 随 agent 步骤的动态演化，天然具有叙事性（"信号积累 → 触发调用"），不需要 SAE/AUROC 背景知识即可理解。Evidence Accumulation 假设是本文最有 novelty 的贡献，Fig 2 直接传达这一 insight。

**Teaser 设计**：
- **主图**：多条 episode 轨迹叠加，成功 episode（gate feature 逐步上升 → 触发 tool call）vs 失败 episode（信号始终低迷）
- **Inset panel**：Fig 3 的核心结果（steering 后 flip rate 显著上升），形成"观测 + 因果"的 one-two punch

### 主实验图表

| 图表      | 对应假设                   | 内容                                                        | 设计要点                                                                         |
| --------- | -------------------------- | ----------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **Fig 1** | H1: Tool-call Gating       | top features 的 AUROC 分布 + 小 K 线性模型 AUC 曲线         | 左：AUROC 柱状图（top-50 features）；右：AUC vs K 曲线，标注 K=10/20/50 时的 AUC |
| **Fig 2** | H2: Evidence Accumulation  | gate feature 强度随 agent step 演化（成功 vs 失败 episode） | 多轨迹叠加 + 均值粗线，竖虚线标注 tool call 触发时刻                             |
| **Fig 3** | H3: Causal Controllability | flip rate vs steering 强度 α，附带质量/正确率 tradeoff      | X 轴 α，左 Y 轴 flip rate，右 Y 轴 perplexity/accuracy；标注 sweet spot          |

---

## 八、快速开始

### 8.1 环境准备

```bash
# 创建环境
conda create -n agent_sae python=3.11
conda activate agent_sae

# 安装依赖
pip install -r requirements.txt
```

### 8.2 完整训练流程

```bash
# Stage 1: 预训练语料 SAE 训练（约 2-4 小时，50M tokens）
python -m sae.train_sae stage1 \
    --model meta-llama/Llama-3-8B-Instruct \
    --layers 24 27 \
    --target-tokens 50000000

# Stage 2: Tool-use 数据流式训练
python -m run.cache_activations train \
    --model meta-llama/Llama-3-8B-Instruct \
    --dataset when2call \
    --layers 24 27 \
    --stage1-dir ./outputs/sae_checkpoints/stage1 \
    --balance

# 分析与可视化
python -m analysis.correlation_analysis \
    --sae-path ./outputs/sae_checkpoints/stage2/stage2_layer24_final.pt
```
