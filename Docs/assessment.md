# Tool-use Mechanistic Basis 项目评估

## 一、Novelty（新颖性）评估

### ✅ 核心亮点：方向正确，交叉点空白大

这个项目的核心新颖性在于 **将 SAE-based MI 技术系统性地应用于 Agent 的工具调用决策机制**。根据最新文献调研（截至 2026 年 2 月）：

| 维度 | 现有工作 | 本项目的增量 |
|------|----------|-------------|
| SAE 用于 LLM 解释 | Anthropic (2024), Google (Gemma Scope), SAELens 等 | ✅ 已有成熟方法论 |
| Tool-calling 评测 | BFCL, When2Call (NAACL 2025), API-Bank | ✅ 可直接复用 |
| MI 用于 Agent 行为 | 几乎空白 ⚠️ | 🆕 **核心创新点** |
| SAE feature → decision-level 因果 | 零星工作（偏 token-level） | 🆕 **从 token 提升到 policy** |

**最强卖点**：目前 MI 社区主要聚焦于理解 LLM 的"知识表征"和"推理电路"，而 **Agent 场景下的决策机制（尤其是 tool-call gating）** 几乎无人系统做过。这是一个明确的空白。

### ⚠️ Novelty 风险

1. **"发现和解释一些 feature"本身不够新**：SAE 找 feature 已经是标准操作。novelty 必须体现在 **decision-level 因果证据**（假设 3），而非仅仅 "我们发现了 gate feature"
2. **方向 A（tool-use gating）相对窄**：只关注 "call vs no-call" 的二元决策，可能被 reviewer 认为场景过于简单。需要用 evidence accumulation（假设 2）和动态门控扩展来弥补
3. **竞争窗口正在缩小**：SAE + Agent 的交叉方向关注度在 2025 下半年明显上升（Automated Interpretability Agent、MechSci 等）。**需要尽快出结果**

### 📊 综合评分：**Novelty 7/10**

方向很好，但需要实验结果足够"硬"（特别是因果干预部分）才能拉开差距。

---

## 二、Feasibility（可行性）评估

### ✅ 有利因素

1. **代码框架完整**：已搭建 Agent Loop、SAE 模型、两阶段训练、分析流水线等核心模块（约 5000+ 行代码）
2. **数据集可得**：When2Call（HuggingFace 公开）、BFCL（GitHub）、API-Bank（GitHub）
3. **模型选择合理**：Llama-3-8B / Qwen-3 / Gemma-3 均为开源可 hook 模型
4. **流式激活设计节省存储**：避免 160GB+ 的激活落盘问题
5. **两阶段 SAE 训练**：Stage 1 通用语料预训练 + Stage 2 tool-use 微调，方法论上合理

### ⚠️ 可行性风险（按严重程度排序）

#### 🔴 高风险

| # | 风险 | 影响 | 建议 |
|---|------|------|------|
| 1 | **Gate feature 不存在或不稀疏** | 整个假设链崩塌 | 先跑小规模验证（1k episodes），确认 AUROC > 0.7 再全面展开 |
| 2 | **因果干预 flip rate 太低** | 论文最硬的部分没有结果 | 准备 fallback：即使 flip 不大，可以转向"gate 是连续信号而非开关"叙事 |
| 3 | **算力不足** | 8B 模型 + 50k episodes + SAE 训练 需要大量 GPU 时间 | 估算：Llama-3-8B inference ≈ 16GB VRAM，50k episodes（单卡 A100）约 3-5 天 |

#### 🟡 中风险

| # | 风险 | 影响 | 建议 |
|---|------|------|------|
| 4 | **JSON 输出格式不稳定** | Decision boundary 被噪声污染 | 使用 structured generation（如 outlines / vLLM JSON mode）强制 JSON |
| 5 | **When2Call 数据集规模/质量** | 训练数据不足或覆盖面窄 | 补充 synthetic generator；BFCL 做外部验证 |
| 6 | **feature 是格式触发器而非语义 gate** | 发现的 "gate" 只是 JSON 格式特征 | 必须做 activation maximization + 语义验证来区分 |
| 7 | **跨模型泛化性** | Llama 上的发现在 Qwen/Gemma 上不成立 | 至少做 2 个模型才有说服力 |

#### 🟢 低风险

| # | 风险 | 影响 | 建议 |
|---|------|------|------|
| 8 | 工具名 label shortcut | feature 只是学到工具名 | 已在设计中考虑（工具名随机化） |
| 9 | SAE 不收敛 | 需要调参迭代 | TopK SAE 通常比 L1 更稳定，两阶段训练也有帮助 |

### 📊 综合评分：**Feasibility 6.5/10**

框架到位，但**核心假设是否成立取决于实验**——这正是研究的本质。主要顾虑在算力和结果不确定性。

---

## 三、执行过程中可能遇到的具体问题

### 1. 🔧 工程问题

**（a）模型加载与 Hook 兼容性**
- 不同模型（Llama / Qwen / Gemma）的层结构和残差流位置**不统一**
- `ActivationCache` 中的 hook 注册代码假设了特定模型结构，切换模型时大概率需要修改
- **建议**：用 `model.config` 动态获取层数和隐藏维度，而非硬编码

**（b）流式训练的内存管理**
- `ActivationBuffer` 在内存中积累激活，如果 buffer_size 设太大或模型 hidden_dim 很高，可能 OOM
- 估算：buffer_size=8192, hidden_dim=4096, float32 → 约 128MB/layer
- 多层同时缓存时需注意内存压力

**（c）JSON 输出解析失败**
- 开源模型（尤其未微调版本）生成 JSON 的能力参差不齐
- `output_parser.py` 需要鲁棒的 fallback 机制（正则匹配、部分解析等）
- Base model（未 tool-tuned）几乎一定会在 JSON 格式上频繁出错

### 2. 📊 实验设计问题

**（a）CALL vs NO_CALL 样本平衡**
- When2Call 原始数据的 CALL:NO_CALL 比例可能不是 1:1
- 如果不平衡，SAE 学到的 feature 可能偏向多数类
- 代码中的 `ActivationBuffer.balance=True` 是对的，但需要确认在 Stage 1 预训练时不应该做平衡

**（b）"Action boundary window" 定义的敏感性**
- W_pre=20 tokens、W_post=10 tokens 的窗口大小是经验值
- 不同的窗口大小可能显著影响结果
- **建议**：作为 ablation 实验的一个维度

**（c）层选择**
- 默认选 Layer 24, 27（基于 32 层 Llama-3-8B 的 3/4 和 5/6 位置）
- 但 Qwen-3-8B 和 Gemma-3-4B 层数不同，hook 层需要相应调整
- 最好做一个 layer sweep 实验来确认最优层

### 3. 📈 结果不如预期时的应对

**（a）如果 gate feature 可分离性低（AUROC < 0.65）**
- 可能原因：决策信息分散在很多 feature 上（非稀疏）
- 应对：增大 SAE dictionary size、尝试不同层、或换成 phase-wise SAE

**（b）如果因果干预效果微弱**
- 可能原因：tool-call 决策是多个 feature 联合决定的
- 应对：改做 **多 feature 联合 steering**，而非单 feature 干预
- 或者转向"连续积累"叙事：展示 feature 强度与 P(CALL) 的连续相关性

**（c）如果 base vs tool-tuned 没有差异**
- 可能原因：base model 就已经有 gate feature（来自预训练 JSON 数据）
- 这其实也是一个有趣的 finding，可以卖成 "tool-call gating 是 emergent 的"

### 4. ⏱️ 时间线风险

根据 `mission.md` 估算的 5-8 周偏乐观：

| 阶段 | 估算 | 实际风险 |
|------|------|----------|
| 环境 + 数据准备 | 3-5 天 | ⚠️ 模型下载可能很慢；When2Call 数据格式需要适配 |
| Agent 闭环调通 | 5-7 天 | ⚠️ JSON 输出不稳定可能消耗大量 debug 时间 |
| Rollout 生成 | 3-5 天 | ⚠️ 取决于 GPU 数量和模型推理速度 |
| SAE 两阶段训练 | 5-7 天 | ⚠️ Stage 1 需要 50M tokens，可能需要多次调参 |
| 分析与干预 | 5-10 天 | ⚠️ 如果结果不理想需要反复迭代 |
| **总计** | **3-5 周**（理想） | **6-10 周**（现实） |

---

## 四、总结与建议

### 🎯 值得做吗？

**值得，但需要策略性执行。** 方向的新颖性和时效性都很好，关键在于：

1. **尽早验证核心假设**：用 1k episodes + 1 层 SAE 做快速验证，确认 gate feature 存在性
2. **把因果干预做扎实**：这是论文的 "evidence grade" 决定因素
3. **准备多种叙事**：如果二元 gating 不明显，转向连续积累叙事
4. **控制范围**：先做 1 个模型 + 1 个数据集出核心结果，再扩展
5. **时间紧迫**：这个交叉方向的竞争窗口在缩小，建议 **3 个月内** 完成初稿

### 🏆 最终投稿建议

- **目标会议**：ICML / NeurIPS / ICLR（MI workshop 作为备选）
- **论文卖点不要是** "我们发现了几个 feature"
- **论文卖点应该是** "我们首次为 Agent 工具调用决策提供了 mechanistic 解释，并通过 feature-level causal intervention 证明了这些内部表征的因果效力"
