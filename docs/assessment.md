# Tool-use Mechanistic Interpretability Project 评估报告

> 基于对 `question.md`、`Agent-Tool-Use-MI/docs/`、When2Call 数据文件及 BFCL 数据文件的实际检验。
> 评估时间：2026-03-17

---

## 一、关键发现（需立即处理）

### 🔴 严重数据问题：When2Call 标签分布与代码均存在错误

在实际读取数据文件后，发现以下与 `question.md` 描述**不符**的事实：

| 数据文件 | 样本量 | 实际标签分布 |
|----------|--------|-------------|
| `when2call_train_sft.jsonl` | 15,000 | **全部 NO_CALL**（CALL = 0） |
| `when2call_train_pref.jsonl` | 9,000 | CALL = 3,000 / NO_CALL = 6,000 |

**解析**：
- SFT 数据的助手回复全是直接回答或拒绝回复，**不含任何 `<TOOLCALL>` 标记**。
- Preference 数据的 `chosen_response` 字段用 `<TOOLCALL>[...]</TOOLCALL>` 区分 CALL，CALL 样本仅 3,000 条。
- `question.md` 中 "CALL:NO_CALL 平衡采样，有效样本约 12K × 2 = 24K" 的判断**完全不成立**。

**实际可用 CALL 样本**：最多 3,000（pref 数据的 chosen）。1:1 平衡后 Stage 2 实际训练规模约为 **6,000**，不是 24,000。

### 🔴 适配器代码与数据格式不匹配

`Agent-Tool-Use-MI/tasks/when2call_adapter.py` 中：

```python
should_call = raw_sample.get("should_call", raw_sample.get("label", None))
```

实际数据中**不存在** `should_call` 或 `label` 字段。数据格式为：
```
{"tools": [...], "messages": [{"role": "user", ...}, {"role": "assistant", "content": "..."}]}
```
（SFT）或：
```
{"tools": [...], "messages": [{"role": "user", ...}], "chosen_response": {...}, "rejected_response": {...}}
```
（Pref）

该适配器目前会将所有样本标记为 `UNCERTAIN`，无法正确提取标签。**Stage 2 训练依赖该适配器，必须先修复。**

### 🟡 架构文档中模型名称不一致

- `question.md`："目标模型：Qwen3.5-4B/9B"
- `architecture.md` 运行命令：`--model meta-llama/Llama-3-8B-Instruct`
- `mission.md`："下载模型：Llama-3-8B / Qwen3-8B / Gemma3-4B"

SAE 层选择（Layer 24, 27）对应 32 层 Llama-3-8B，**不适用于 Qwen3-4B（28 层）或 9B（36 层）**。需要为每个目标模型单独确认激活提取层。

---

## 二、回答 question.md 中的四个具体问题

### 问题 1：Stage 2 数据集组合策略

**结论：方案 A（仅 When2Call）是正确方向，但需使用正确数据子集。**

现有情况下三个方案的实际可行性：

| 方案 | 实际有效 CALL 样本 | 实际有效 NO_CALL 样本 | 1:1 后规模 | 可行性 |
|------|-------------------|----------------------|------------|--------|
| A：仅 When2Call (Pref) | 3,000 | 3,000（从 6K 采样） | ~6K | ✅ 可行，但规模偏小 |
| B：When2Call + BFCL | 3,000 + 2,000+ CALL | 3,000 + 1,124 NO_CALL | ~8K | ✅ 有增益，但标签噪声增加 |
| C：When2Call + Tau²轨迹 | 3,000 | 3,000 | 6K + 轨迹片段 | ⚠️ 后处理成本高，优先级低 |

**推荐方案（修订版 A+B）**：
1. 从 `when2call_train_pref.jsonl` 中提取 3,000 CALL + 3,000 NO_CALL 样本（1:1）
2. 将 BFCL `BFCL_v4_irrelevance.json`（240）+ `BFCL_v4_live_irrelevance.json`（884）合并补充 NO_CALL（共约 1,124 条）
3. 将 BFCL `BFCL_v4_simple_python.json`（400）+ `BFCL_v4_live_simple.json`（258）等补充 CALL
4. **不要使用** `when2call_train_sft.jsonl`（全是 NO_CALL，且格式不适合提取 CALL/NO_CALL 对比）

总有效 Stage 2 数据约 **8,000–10,000 样本**。

### 问题 2：BFCL NO_CALL 标签提取

**实际数量**（已核实）：

| 文件 | 样本量 | 说明 |
|------|--------|------|
| `BFCL_v4_irrelevance.json` | 240 | 干净的 NO_CALL |
| `BFCL_v4_live_irrelevance.json` | 884 | 真实用户提问 NO_CALL |
| **合计** | **1,124** | 可直接使用，无需清洗 |

BFCL v4 Irrelevance 样本质量较高（测试集设计），但总量有限，仅能作为补充，不能替代 When2Call Pref 作为主数据。BFCL CALL 样本（Simple/Multiple/Parallel 等）**均为评测集设计**，没有 CALL/NO_CALL 的配对标签，引入会增加决策边界噪声。

**建议**：Irrelevance 数据与 When2Call pref 的 CALL 样本配对（1:1），BFCL CALL 类别谨慎使用。

### 问题 3：多步轨迹对 Evidence Accumulation 验证的必要性

**是必要的，但 Fig 2 设计需要调整。**

`architecture.md` 将 Fig 2 定义为"gate feature 随 agent step 演化的主图"并放在 Introduction 作为 teaser。然而：

- When2Call 是**单轮决策数据**，没有 step 序列
- Tau²-Bench 的多步轨迹**没有逐步 CALL/NO_CALL 标注**
- VitaBench 样本量太少（400 任务）

**实用替代方案**：
1. **伪多步序列**：将 When2Call 的多个 pref 样本串联，构造"证据逐步给出 → 模型逐步倾向 CALL"的情境。这是一种受控实验，叙事可行但需注意不是真实 agent 轨迹。
2. **利用 sandbox 工具生成真实轨迹**：用 `agent_loop.py` + sandbox tools 跑 When2Call 题目，记录多步决策序列。这是最干净的做法，但需要 agent 闭环先跑通。
3. **降低 Fig 2 地位**：将 Evidence Accumulation 从"首要贡献"降为"扩展实验"，主论文先聚焦 H1（gate 存在性）和 H3（因果干预）。

**优先建议**：先跑 H1+H3，用方案 1（伪多步序列）做 H2 的 pilot 结果，后续再用方案 2 强化。

### 问题 4：24K（实为 6K）样本对 SAE Stage 2 的充分性

**结论：6K 样本对 TopK SAE 的 Stage 2 微调是不足但可接受的起点，需策略性处理。**

参考 SAE 训练规模文献：

| 项目 | 训练激活数 | SAE 字典大小 | 场景 |
|------|-----------|-------------|------|
| Anthropic Scaling SAEs (2024) | 数亿 tokens | 32K–1M | 通用语言理解 |
| Gemma Scope (2024) | 数十亿 tokens | 16K–1M | 通用 |
| 本项目 Stage 2 | 约 6K 样本 × ~20 tokens = **~120K tokens** | 32,768 | Tool-use 微调 |

120K tokens 对于从头训练 SAE 远远不够，但两阶段训练设计部分缓解了这个问题：

- Stage 1 用 50M tokens 的 OpenWebText2 建立通用字典初始化 ✅
- Stage 2 只是在此基础上做**领域微调**，类似 fine-tuning
- 类似的 "SAE fine-tuning" 工作表明少量领域数据（10K–50K 激活）也能有效偏移 feature 分布

**风险与建议**：
- 如果 Stage 2 过拟合（training loss 下降但 CALL/NO_CALL AUROC 不稳定），考虑**更小学习率**（1e-5 而非 5e-5）或**早停**
- 考虑只更新 decoder 权重，冻结 encoder（减少参数，防过拟合）
- Ablation：对比 Stage 1-only SAE vs Stage 2 SAE 的 AUROC，量化 Stage 2 的贡献

---

## 三、架构与代码质量评估

### 代码框架完整性

| 模块 | 状态 | 关键问题 |
|------|------|---------|
| `tasks/when2call_adapter.py` | ⚠️ 存在 bug | 字段名错误（`should_call` 不存在）；需改为解析 `chosen_response` 中的 `<TOOLCALL>` |
| `tasks/bfcl_adapter.py` | 未验证 | 需确认是否适配 v4 格式 |
| `sae/train_sae.py` | ✅ 基本合理 | 流式 prefetch 设计好；bfloat16 节省显存 |
| `sae/sae_model.py` | 未读，待验证 | - |
| `analysis/correlation_analysis.py` | 未读，待验证 | - |
| `analysis/steering.py` | 未读，待验证 | - |
| `run/cache_activations.py` | 未读，待验证 | - |

### 架构设计亮点

1. **流式激活处理**：不落盘设计节省约 160GB 存储，工程上合理
2. **两阶段训练**：Stage 1 通用预训练 + Stage 2 领域微调，是比直接在 tool-use 数据上从头训练更稳健的方法
3. **预分配缓冲区 `_PendingBuffer`**：避免频繁 `torch.cat`，内存效率好
4. **后台预取 `_prefetch_generator`**：推理/训练可并行，减少 GPU 空闲时间

### 架构设计风险

1. **hardcoded 层号**：`--layers 24 27` 在文档中为常量，切换到 Qwen/Gemma 时会出错
2. **SFT 数据被误用**：当前架构文档提到 "When2Call / BFCL 任务的 action boundary 激活" 但未区分 SFT（全 NO_CALL）与 pref（含 CALL）格式差异
3. **action boundary 的实际定义**：`W_pre`（输出 TOOLCALL 前 N token）在 pref 数据中可以定义，但在 SFT 数据中由于全是 NO_CALL，boundary 意义不同
4. **`<TOOLCALL>` 标记解析**：需要确认 Qwen3.5 在推理时使用的是哪种 tool call 格式（OpenAI schema vs `<TOOLCALL>` 文本标记），两者不同

---

## 四、三个核心假设的可验证性评估

### H1：Tool-call Gating（门控存在性）

**可验证性：高** ✅

- 数据：When2Call pref 的 CALL/NO_CALL 对是干净的二元标签
- 方法：AUROC + 线性探针已实现于 `analysis/correlation_analysis.py` 和 `analysis/linear_probe.py`
- 风险：CALL 样本仅 3,000，特征估计可能不稳定，建议用 bootstrap CI 量化不确定性

**预期结果范围**：如果存在 gate feature，AUROC 应 > 0.75（少量高 AUROC feature）；如果 < 0.65，可能需要换层或增大 dictionary_size。

### H2：Evidence Accumulation（证据积累）

**可验证性：中等** ⚠️

- 数据：When2Call 单轮，无法直接验证跨 step 演化
- 替代设计：用伪多步序列或 sandbox 生成的真实轨迹
- 这是三个假设中**实验设计最弱**的一个，建议论文中降级为 exploratory finding

### H3：Causal Controllability（因果可控性）

**可验证性：中等** ⚠️

- 方法已实现于 `analysis/steering.py`
- 关键挑战：Qwen3.5 的 thinking 模式（`<think>...</think>`）可能干扰 tool-call decision 测量
- 建议：使用 non-thinking 模式（`--thinking_budget 0`）确保决策边界干净
- flip rate 目标：在不破坏语言质量（perplexity 变化 < 20%）的前提下 flip rate > 20%

---

## 五、综合评分与优先行动项

### 整体评分

| 维度 | 评分 | 说明 |
|------|------|------|
| 研究方向新颖性 | 8/10 | MI × Agent tool-use 交叉，空白明显 |
| 实验设计完整性 | 6/10 | 三假设体系合理，但 H2 数据支撑弱 |
| 代码框架质量 | 7/10 | 流式设计好，但 adapter 有严重 bug |
| 数据准备充分性 | 4/10 | 标签分布问题严重，远少于预期 |
| 执行可行性 | 6/10 | 核心假设能验证，但 6K 数据规模有限 |

### 优先行动项（按紧迫性排序）

**必须立即处理：**

1. **修复 `when2call_adapter.py`**：改为从 `when2call_train_pref.jsonl` 的 `chosen_response` 字段提取标签（`<TOOLCALL>` → CALL，否则 → NO_CALL），废弃对 SFT 数据的直接使用
2. **确认目标模型**：统一为 Qwen3.5-4B/9B，更新 `architecture.md` 中的命令行，并对该模型做层 sweep（推荐从 layer 16/21 开始，对应 4B 的 3/4 和 5/6 位置）
3. **端到端数据管线验证**：在 100 个 pref 样本上跑完从 "适配器加载 → 激活提取 → SAE 更新" 的完整流程，确认无 bug 再扩大规模

**短期（1 周内）：**

4. **数据增补方案**：考虑用 `synthetic_generator.py` 合成额外的 CALL/NO_CALL 对，或从 BFCL Irrelevance（1,124 条）补充 NO_CALL，将有效训练集从 6K 扩大到 10K+
5. **快速验证 H1**：仅用 pref 数据中 500 CALL + 500 NO_CALL，训练 SAE Stage 2，计算 top feature AUROC。如果 AUROC > 0.7，说明假设方向正确，继续全面展开

**中期（2–4 周）：**

6. **H3 因果干预实验**：在 H1 验证后，用 `steering.py` 跑 flip rate vs α 曲线（先单特征，再多特征联合）
7. **H2 替代方案**：用 sandbox 生成 50–100 个真实多步 agent 轨迹，追踪 gate feature 随 step 演化（可作为论文 Fig 2 的素材）
8. **模型一致性**：确认 Qwen3.5 是否需要关闭 thinking 模式，确保 decision boundary 干净

---

## 六、关于 Stage 2 数据集的最终方案建议

综合以上分析，Stage 2 推荐方案如下：

**数据来源**：
- 主体：`when2call_train_pref.jsonl` 中 CALL = 3,000 + NO_CALL = 3,000（从 6K 中随机采样，1:1 平衡）
- 补充 NO_CALL：BFCL `BFCL_v4_irrelevance.json`（240）+ `BFCL_v4_live_irrelevance.json`（884）= 1,124 条
- 可选合成：`synthetic_generator.py` 额外生成 2,000 CALL/NO_CALL 对（控制难度梯度）

**数据格式**：
- CALL 标签：`chosen_response.content` 包含 `<TOOLCALL>` 的样本
- NO_CALL 标签：`chosen_response.content` 为纯文本回复的样本
- 激活提取位置：`chosen_response` 序列末尾（决策边界）的残差流

**不使用**：
- `when2call_train_sft.jsonl`：全为 NO_CALL，格式不适合提取 action boundary 对比激活
- BFCL Simple/Multiple/Parallel 系列：无 NO_CALL 配对，引入不需要的多任务噪声
- Tau²-Bench / VitaBench：标注粒度不足，后处理成本过高

**预期有效 Stage 2 数据规模**：约 8,000–10,000 样本（含合成部分），对应约 160K–200K tokens 激活。在两阶段训练框架下，这对特征微调是可接受的起点，建议配合早停和较小学习率（1e-5）使用。

---

*本评估基于对实际数据文件的统计分析和代码检查，主要关注数据质量、代码正确性及实验设计可行性。研究方向本身的学术价值评估见 `Agent-Tool-Use-MI/docs/assessment.md`。*
