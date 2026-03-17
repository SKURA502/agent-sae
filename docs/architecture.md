# 代码架构与实验设计

## 目录结构

```
Agent-Tool-Use-MI/
├── configs/            # 模型、SAE、任务配置（YAML）
├── controller/         # 最小 agent 循环、工具 schema、沙盒工具
│   └── sandbox_tools/  # search、calculator、lookup（含噪声注入）
├── tasks/              # 数据集适配器：When2Call、BFCL、合成生成器
├── run/                # Rollout 生成、流式激活处理
├── sae/                # 两阶段 SAE 训练、特征提取与打分
├── analysis/           # 特征-决策相关性、线性探针、steering、可视化
├── data/               # 原始与处理后数据集、rollout 日志
└── outputs/            # SAE 检查点（stage1/stage2）、分析结果、图表
```

---

## 核心模块设计

### Agent Controller

- 输入：`(messages, tool_schema)` → 强制 JSON 输出 → 解析 `decision ∈ {CALL, NO_CALL}`
- 每个 episode 随机化工具名，防止模型对工具名 token 产生 shortcut 学习
- 噪声注入开关：`p_fail`、`p_empty`、`p_corrupt`（用于 H2 鲁棒性实验）
- 推理时关闭 Qwen3.5 thinking 模式（`thinking_budget=0`），保持决策边界干净

### 流式激活处理

激活在内存缓冲区内处理，不写入磁盘。50K 个 episode 可节省约 160GB 磁盘空间。架构：

```
LLM 推理 → 目标层 Hook → 内存缓冲区 → SAE 在线更新
```

Hook 点：action boundary 处的残差流（模型输出 `CALL`/`NO_CALL` 分支前的最后几个 token）。

### 两阶段 SAE 训练

| 阶段 | 数据 | 规模 | 目的 |
|------|------|------|------|
| Stage 1 | OpenWebText2（无标签） | 50–100M tokens | 学习通用语言表征 |
| Stage 2 | When2Call Pref（有标签） | ~6K 平衡样本 | 专项学习工具调用决策特征 |

- 训练层：`int(n_layers × 3/4)` 和 `int(n_layers × 5/6)`，由模型配置动态计算
- 字典大小：`hidden_size × 8`；TopK 的 k = `hidden_size / 32`
- Stage 2 使用 lr=1e-5 配合早停，防止在小规模有标签集上过拟合

### 分析流水线

1. **相关性分析**（H1）：计算每个 feature 的 `E[f|CALL] − E[f|NO_CALL]` 和 AUROC，筛选 top-K
2. **线性探针**（H1）：用 top-K SAE 激活训练逻辑回归；目标 K=50 时 AUC > 0.80
3. **语义验证**：对 top features 做 activation maximization，区分认知缺口门控与格式触发器
4. **因果 steering**（H3）：`h := h ± α × decoder_vector(feature_i)`；测量翻转率和 perplexity 变化
5. **轨迹分析**（H2）：追踪 sandbox rollout 中门控 feature 强度随 agent step 的演化

---

## 数据集策略

本项目有三类功能不同的数据集角色，不可混用：

| 角色 | 目的 | 标签要求 | 规模 |
|------|------|---------|------|
| **SAE 训练** | 学习残差流表征 | 无（Stage 1）/ CALL/NO_CALL（Stage 2） | 50M+ tokens / ~6K 样本 |
| **特征发现** | 通过 CALL/NO_CALL 差值定位门控 features | 严格二元标签对 | ~8K 对 |
| **实验评测** | 验证 H1/H2/H3 | 因假设而异 | 测试集规模 |

### SAE 训练数据集

- **Stage 1** — OpenWebText2（50–100M tokens，无标签）：在预训练 token 量级上建立通用字典
- **Stage 2 可选：域适应** — 无标签工具调用语料（When2Call SFT 15K + BFCL 完整语料 + Tau² 轨迹），目标 ~5–10M tokens：无需标签即可将表征空间向工具调用场景靠拢，弥合 Stage 1 与有标签微调之间的 token 量级差距
- **Stage 2 特征专项** — When2Call Pref 有标签（3K CALL + 3K NO_CALL，1:1 平衡，~6K 总计）：提取 action boundary 处的残差流激活，是学习门控 features 的关键数据

### 特征发现数据集

用于计算 `E[f|CALL] − E[f|NO_CALL]`、AUROC 及线性探针：

- **When2Call Pref**：3K CALL + 3K NO_CALL（从 9K Pref 集 1:1 采样）
- **BFCL Irrelevance**：1,124 条额外 NO_CALL 样本（`BFCL_v4_irrelevance.json` + `BFCL_v4_live_irrelevance.json`）

注意：`when2call_train_sft.jsonl`（15K）全为 NO_CALL，不可用于特征发现。

### 实验评测数据集

| 假设 | 数据集 | 格式 |
|------|--------|------|
| **H1** — 门控可分离性 | When2Call test MCQ（3,652）+ BFCL Live（泛化性验证） | 静态有标签 |
| **H2** — 证据积累 | 通过 `agent_loop.py` 生成的 sandbox rollouts | 动态，100–500 个多步 episode |
| **H3** — 因果 steering | When2Call test MCQ（3,652） | 静态有标签；测量干预下的翻转率 |

H2 的 rollout 生成依赖 agent loop 完整跑通。episode 需包含工具调用成功与失败两种情况，以对比门控 feature 的轨迹差异。

---

## 关键指标

| 指标 | 对应假设 | 目标 |
|------|---------|------|
| Top-feature AUROC | H1 | > 0.75 |
| 线性探针 AUC（K=50） | H1 | > 0.80 |
| 决策翻转率 | H3 | > 20% |
| 语言质量保持（Δperplexity） | H3 | < 20% |
| 门控 feature 强度跨 step 趋势 | H2 | CALL 前单调上升；成功返回后被抑制 |

---

## 论文图表

| 图 | 对应假设 | 内容 |
|----|---------|------|
| **Fig 1** | H1 | top features 的 AUROC 分布；AUC vs K（K=10/20/50）曲线 |
| **Fig 2** | H3 | 翻转率 vs steering 强度 α；质量-准确率 tradeoff |
| **Fig 3** | H2 | 门控 feature 强度随 agent step 演化（成功 vs 失败 episode） |
