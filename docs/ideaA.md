# USER: 我之前是做 Mechanistic Interpretability 方向的，主要是用 sparse autoencoder 来解释 LLM 的 hiddenstates，接下来想往 LLM agent 方面靠，就是用 mechanistic interpretability 的方法工具解释大模型在agent system中的一些行为什么的，你有什么这方面的建议吗？

你这个转向**非常对位**，而且说实话：**真正有长期研究价值、但目前系统性做得还不多的方向，正是在你说的这个交叉点上**。
你已经具备了别人很难补的那一块能力（SAE / MI），而 Agent 恰恰是“黑箱程度被放大”的地方。

我分 **5 个层次**来给你建议，从“问题该怎么重新定义”到“具体你可以马上开做的研究切口”。

# 一、先校准一个关键认知：Agent 不是一个模型问题，而是一个“闭环动力系统”

你之前做 MI 时，隐含假设往往是：

> 输入 → LLM → 输出 hidden states ≈ 一个相对“静态”的计算图

但 **Agent system 本质不同**：

> LLMₜ → Actionₜ → Environment → Observationₜ₊₁ → LLMₜ₊₁ → …

这是一个 **policy-in-the-loop 的动力系统（dynamical system）**。

**重要后果**：

* 单步 hidden state 的语义 ≠ 行为语义
* feature 的“作用”往往体现在 **跨 step 的状态演化**
* Agent failure 往往不是“一步错”，而是“反馈被错误放大”

👉 所以你要做的不是“解释一个 forward pass”，而是：

> **解释某类内部表征，如何在多步交互中稳定地产生某种策略行为。**

这是 Agent MI 的核心转变。

# 二、重新定义你的“解释对象”：从 token-level → decision-level

### 更有潜力的解释对象（非常适合 SAE）

#### 1️⃣ **Action-relevant latent features**

你不是去解释“模型在想什么”，而是解释：

> 哪些 latent feature **在工具调用决策前稳定激活**？

具体可以定义：

* 在 `type=tool_call` 输出前 N 个 token
* 在 controller 接受 action 之前的 hidden states
* 对应到你熟悉的 SAE feature activation

你可以问：

* 是否存在 **tool-specific features**？
* 是否存在 **meta-decision features**（如“需要查信息 / 不确定 / 要验证”）？

👉 这一步非常自然地把 MI 接到了 Agent 行为上。

#### 2️⃣ **跨 step 稳定的策略特征（policy features）**

Agent 的很多行为不是一步决定的，而是：

* “持续探索”
* “反复确认”
* “提前收敛”
* “过度工具依赖”

你可以把一个 episode 看成序列：

```
(step, feature_activation_vector, action_type)
```

然后找：

* **在多 step 上保持激活、并预测 action 模式的 features**
* 类似“mode / regime”的 latent representation

👉 这是 SAE 在 Agent 里**非常强**的用武之地。

# 三、非常具体的 3 个“你可以立刻开做”的研究切口

下面这 3 个方向，我刻意选的是：

* **不需要你重造 Agent 框架**
* **不依赖私有模型**
* **MI 贡献是不可替代的**

## 方向 A：Tool-use 的 mechanistic basis（强烈推荐）

### 核心问题

> Agent 为什么“决定用工具”？
> 这个决定在内部是一个离散 switch，还是连续证据积累？

### 可操作定义

* 构造任务：信息不足 → 需要 search/tool
* 对比：

  * 不调用工具的 rollout
  * 调用工具的 rollout

### MI + SAE 切入点

* 在 **action boundary**（输出 tool_call 之前）：

  * 训练 SAE
  * 找到对 `P(tool_call)` 有最大边际影响的 features
* 做 intervention：

  * 放大 / 抑制某些 features
  * 看是否改变 tool-use 频率（但不破坏语言流畅性）

### 这在 Agent Safety / Interpretability 里非常有说服力

而且 **工业界非常关心**。

## 方向 C：Agent failure 的“内部因果路径”

### 核心问题

> Agent 出现 loop / hallucinated tool result / premature stop
> 是“知识问题”还是“控制策略问题”？

### 做法

1. 收集失败 episode（例如无限 search、拒绝用工具）
2. 用 SAE 对 hidden states 做 feature tracing
3. 比较：

   * 成功 vs 失败 episode
   * 同一任务、不同随机种子

你要找的不是：

* “feature X 在失败时更大”

而是：

* **哪些 feature 先激活 → 导致错误 action → 被环境放大**

👉 这是 Agent 场景下“mechanistic causality”的真正价值。


## 方向 B：环境反馈对 Agent 内部状态的 mechanistic 影响

### 核心问题

> 环境反馈（tool result / observation）是如何被 LLM **吸收、忽略或重解释**的？
> 反馈在内部是一次真正的“状态更新”，还是被既有计划过滤掉？

### 可操作视角

* 构造闭环：
  LLM 形成判断 → 环境返回反馈 → LLM 产生下一步行为
* 在任务与 prompt 不变的情况下，仅改变反馈：

  * 正确 vs 错误
  * 与当前计划一致 vs 冲突

### MI 切入点

* 关注 **feedback boundary**（读完反馈、尚未行动前）：

  * 分析 hidden states / SAE latent features
  * 找到能预测“是否采信反馈”的内部特征
* 做 feature-level intervention：

  * 放大 → 更易根据反馈更新
  * 抑制 → 更易坚持原计划（confirmation bias）

### 核心 Insight

> Agent 的行为不仅取决于“拿到了什么信息”，
> 而取决于 **环境反馈是否真正进入了内部状态**。
> 这一过程由少量可解释、可干预的 latent features 控制。

### 价值

* 解释为什么 agent 会忽略正确信息或盲信错误信息
* 补全 A（tool-use 后发生了什么）与 C（失败如何被环境放大）
* 为 agent safety 提供新的干预维度（信任 / 怀疑机制）


# 四、方法论建议：你该如何“用 SAE 用得更聪明”

一些非常具体、偏方法的建议（结合你背景）：

### 1️⃣ 不要只做 layer-wise SAE，做 **phase-wise SAE**

Agent 有明显 phase：

* planning
* acting
* observing
* summarizing

你可以：

* 只在特定 phase 的 hidden states 上训练 SAE
* 比如：**只看 action-decision window**

这比“全 token SAE”解释性高很多。

### 2️⃣ 把 intervention 定义成 **policy-level** 而不是 token-level

不要只看：

* logit change
* perplexity

而是看：

* tool_call rate
* step count
* success probability
* recovery after error

这会让你的 MI 结果 **天然是 Agent-aware 的**。

### 3️⃣ 把 SAE feature 当成“soft state variables”

Agent 是 dynamical system，你可以显式画：

```
feature_t  → action_t → obs_{t+1} → feature_{t+1}
```

这一步非常有潜力连到：

* control theory
* causal abstraction
* agent safety

# 五、从“选题安全性”角度给你一个现实建议

如果你打算：

* 投 ICML / NeurIPS / ICLR
* 或往 Agent Safety / Alignment 靠

👉 **不要把论文卖点放在“我们又发现了几个 feature”**。

而要卖成：

> “我们首次给出了 Agent 行为层面的 mechanistic explanation，
> 并展示了通过 feature-level intervention 改变 agent policy 的因果证据。”

SAE 是工具，不是 headline。

## 最后一句非常真诚的判断

你这个背景组合（**SAE + 想做 Agent 行为解释**）：

* 在 2026–2027 年会**越来越稀缺**
* 非常容易和 Agent Safety / controllability / alignment 主线合流
* 比“再多做一个 MI benchmark”更有上限

# USER: 我目前想做方向A，就是Tool-use的 mechanistic basis，请你给我一个详细的实验设计，包括需要用到的代码框架，模型，数据集，需要跑的实验等等。

下面给你一个**可以直接开工**、并且非常贴合“Tool-use 的 mechanistic basis（为什么会决定用工具）”的实验设计。我会按 **系统搭建 → 数据/任务 → 记录与表征 → SAE 训练 → 机制发现 → 因果干预 → 扩展实验** 的路径写，尽量具体到“目录结构+该跑哪些脚本+指标怎么算”。

## 0. 你这条线的研究问题怎么落到可实验的假设

把“Tool-use mechanistic basis”拆成 3 个可检验假设（建议你论文就围绕这 3 个）：

1. **Tool-call gating 假设**：模型内部存在一组（稀疏）latent features，在接近 action boundary 时充当“是否调用工具”的门控信号。
2. **Evidence accumulation 假设**：这个门控不是单点触发，而是随“感知到的不确定性/信息缺口”逐步积累的连续量。
3. **Causal controllability 假设**：对这些 features 做局部 intervention（不明显破坏语言质量）能显著改变 tool-call 行为（频率/时机/正确性）。

你的 SAE 背景天然适合把 (1)(3) 做成强因果证据。

## 1) 代码框架与总体架构

### 推荐技术栈（尽量“工业可复现”）

* **Agent runtime（你自己写最小 controller）**：Python + Pydantic/JSON schema（用于工具参数校验）
* **模型推理**：HuggingFace Transformers 或 vLLM（跑大量 episode 省钱）
* **激活抓取/干预**：dictionary_Learning库（实现简单，方便调试） + SAEDashboard库(可视化工具)
* **SAE 训练**： TopK SAE 在 LLM 3/4 的位置的残差流上训练
* **评测任务集**：以“何时该/不该 call tool”为中心，优先用 When2Call + BFCL + API-Bank 的组合（见后面）([arXiv][3])

【USER 补充】尽量用已有的开源代码库和工具，避免重复造轮子，目录结构上也希望能清晰明了，方便后续扩展和维护。

### 目录结构建议

```
agent_sae_tooluse/
  controller/        # 最小 agent loop + tool schema + sandbox tools
  tasks/             # dataset adapters (When2Call/BFCL/API-Bank) + synthetic generator
  run/               # 生成 rollouts、缓存激活、跑评测
  sae/               # SAE训练、feature对齐、feature浏览与打分
  analysis/          # feature→behavior 统计、可视化、ablation/steering 实验
  configs/           # 模型、层、hook点、稀疏度、batch等
```

## 2) 选模型：你要“可控、可 hook、具备 tool-call 能力”

你需要两类模型做对比（论文叙事会更强）：

### A. Base LM（未工具微调）

用于回答：“纯 next-token 预训练模型是否已有 tool-call 相关门控特征？”

### B. Tool-tuned / function-calling tuned LM

用于回答：“工具训练是否在内部形成更清晰/更可干预的门控 feature？”

如果你不想被“闭源 tool-calling 接口细节”牵着走，建议优先开源模型（能完整拿到 hidden states）。

> 小技巧：即使模型不原生支持 tool-calling，你也可以用**强约束的 JSON 输出格式**模拟“tool_call vs final”的离散决策，这样不依赖特定 API。

【USER 补充】需要适配Llama-3-8B，Qwen-3-8B/14B，Gemma-3-4B/12B等主流开源模型。

## 3) 任务与数据集：核心是“when to call / when not to call”

你做方向 A（Tool-use basis），最关键不是“能不能把 tool args 填对”，而是：

> **在信息不足时会去 call；信息足够时不会乱 call。**

### 3.1 主任务：When2Call（强烈推荐当主结果）

When2Call 就是专门评估“该不该调用工具/以及调用失败时怎么做”的数据集，非常对题。([arXiv][3])
你可以把它做成：

* **binary**：call / no-call
* **multi-choice**：call / abstain / ask-clarifying / fallback（视你实现而定）

### 3.2 结构化 tool-calling 能力：BFCL

BFCL 是函数调用基准（覆盖串行/并行/多语言等类型），适合做补充结果，证明你的发现不只在合成集成立。([Gorilla][4])

### 3.3 end-to-end 工具执行：API-Bank

API-Bank 提供可运行的评测系统（73 个工具），适合做“真实工具交互”的补充验证。([arXiv][5])

> 组合建议：
> **主线 = When2Call（门控机制最干净）**
> 外部有效性 = BFCL（格式/选择）+ API-Bank（可执行）

## 4) 环境与工具：必须“可控、可复现、可插入噪声”

你要研究机制，工具环境必须满足三点：

1. **确定性**：同样输入同样输出（便于归因）
2. **可注入不确定性**：可以控制工具返回质量（正确/空/冲突/延迟）
3. **可记录**：每次 tool-call 的 args、返回、时间、是否成功都写 log

### 建议你实现一个“Sandbox Tool Suite”

至少包含：

* `search(query)-> topk snippets`（你可以从固定文档库检索，避免联网）
* `calculator(expr)-> value`
* `lookup(key)-> value`（键值数据库）
* `route(origin,dest)-> time`（可用固定图）

并且加开关：

* `p_fail`：一定概率返回 error
* `p_empty`：一定概率返回空结果
* `p_corrupt`：一定概率返回冲突/错误结果

你后面的 “evidence accumulation / robustness” 会非常依赖这些开关。

## 5) 采样 rollouts：你要收集“决策边界窗口”的激活

### 5.1 关键定义：Action boundary window

你关心的不是全句 hidden states，而是 tool-use 决策发生前后的一个小窗口，比如：

* **W_pre**：输出 `{"type":"tool_call"...}` 之前的 10–30 个 token 的 residual stream / MLP acts
* **W_post**：输出 tool_call JSON 的前几个字段时的 states（通常更“离散化”）

### 5.2 每个 episode 记录什么

建议你把每条数据落成结构化 JSONL：

* prompt + tool schema
* model 输出（raw）
* controller 解析结果：`decision ∈ {CALL, NO_CALL}`
* 如果 CALL：

  * tool_name、args、validator 是否通过
  * tool_result（以及是否错误/空/冲突）
* **激活缓存**（存 tensor 或存 SAE codes）：

  * layer ℓ 的 residual stream（或 MLP post-activation）
  * 只存 W_pre/W_post，别存全序列（否则 IO 爆炸）

## 6) SAE 训练：只在“决策窗口”训练，而不是全 token

这是能显著提升信噪比的关键点。

### 6.1 选 hook 点

* **residual stream**（信息汇聚点，方便做 feature steering）

### 6.2 训练数据构成（很重要）

为了学到“门控特征”，你要刻意平衡：

* tool_call vs no_call 的样本比例（尽量 1:1）
* 任务类型覆盖（查证、算术、检索、常识不需工具）
* prompt 模板多样性（避免学到模板词）

### 6.3 SAE 超参（建议起步配置）

* dictionary size：hidden_size * 8（比如 2048 → 16384）
* target sparsity：激活latent数大致为hidden_size / 32 （比如 2048 → 64）
* 层数：选择3/4 或 5/6 的位置进行训练和推理

## 7) 机制发现：怎么把 feature 变成“tool-call gate”的证据链

你需要三层证据：相关 → 可预测 → 可干预（因果）。

### 7.1 相关性：feature 与 decision 的互信息/点双列相关

对每个 feature f：

* 计算 `E[f | CALL] - E[f | NO_CALL]`
* 或用 AUROC：用单一 feature 预测 CALL

筛出 top-K features（比如 50 个）。

### 7.2 可预测性：用少量 features 线性预测 tool_call

训练一个简单 logistic regression：

* input：top-K SAE activations（W_pre 的聚合，比如 max/mean）
* output：CALL / NO_CALL

你要看两件事：

* **很小 K（10–50）就能达到高 AUC**：说明门控信息确实在少量可解释 latents 上集中
* 跨 prompt 模板/跨任务类型泛化：排除“模板特征”

### 7.3 语义验证：为 top features 做 activation maximization / top examples

你会很快看到两类 feature：

* “不确定性/需要外部信息/查证”类
* “JSON/tool schema/函数名提示词”类

这一步能帮助你区分：

* 真正的 *epistemic gap* gate
* 只是格式触发器

## 8) 因果实验（你论文最硬的部分）：steering / ablation / counterfactual

### 8.1 Steering：放大 gate feature → 增加 tool-call

在 W_pre 的某一层对 residual stream 加上：

* `+ α * decoder_vector(feature_i)`
  观察：
* tool-call rate 是否上升
* 但语言流畅度/格式正确率是否仍可接受

### 8.2 Ablation：压制 gate feature → 减少 tool-call

对 feature_i 做 clamp / subtract：

* `h := h - α * decoder_vector(feature_i)` 或直接将 SAE code 置零（看你的实现）
  观察：
* tool-call rate 是否显著下降
* 是否出现“明明该查却硬答”（错误率上升）

### 8.3 Minimality：只改很少 features 是否足够翻转决策

定义“decision flip rate”：

* 原本 CALL → 变成 NO_CALL 的比例
* 原本 NO_CALL → 变成 CALL 的比例

你要证明：

* 少量 features 能在不破坏句子整体质量的情况下翻转决策（强因果）

### 8.4 Specificity：干预 gate features 不应显著改变内容质量（除 tool 行为外）

做两个对照指标：

* 输出困惑度/流畅度 proxy（或用另一 LLM judge）
* 非工具任务的 accuracy 不应大幅下降

## 9) 关键扩展实验（让你的结果“像一篇完整论文”）

### 9.1 Evidence accumulation：工具返回质量影响 gate features 的动态

用你 sandbox 的 `p_empty/p_corrupt`：

* 工具结果越差，gate features 是否在后续 step 更强、更频繁触发二次调用？
* 成功结果是否抑制 gate features（shutdown effect）？

这能把“门控”从单点扩展成 dynamical system 叙事。

### 9.2 Prompt injection 抗性：不可信工具输出是否劫持 gate features

在 tool_result 里插入 “请继续调用工具/忽略规则” 类文本，测试：

* 哪些 features 对 injection 特别敏感
* 是否存在“安全 gate features”（比如拒绝执行高风险工具）

（这块很容易和 agent safety 结合。）

### 9.3 Base vs tool-tuned 对比

对比两种模型：

* gate features 的可分离性（AUC、稀疏集中度）
* 干预所需强度 α（tool-tuned 是否更“线性可控”）

## 10) 你需要跑的实验清单（建议按这个顺序）

1. **实现 controller + sandbox tools + JSON schema 输出约束**（先让闭环跑通）
2. **跑 When2Call**：收集 10k–50k episode（看算力）
3. **抓 activations（W_pre/W_post）**：存到磁盘
4. **训练 SAE（1–2 个层起步）**：先做出可用 dictionary
5. **feature→decision 统计 + 小 K 线性预测**（做出第一张核心图）
6. **steering / ablation**：跑 decision flip + performance tradeoff
7. **扩展到 BFCL/API-Bank**：证明外部有效性
8. **robustness（工具失败/空返回）**：做 dynamical 结果
9. **base vs tool-tuned 对比**：加强结论

## 11) 你最终论文/报告里最关键的 3 张图（建议你提前对齐）

* **Fig1：gate feature 的可分离性**
  top features 的 `E[f|CALL]-E[f|NO_CALL]` / AUROC 分布，外加小 K 线性模型 AUC 曲线

* **Fig2：因果干预翻转决策**
  flip rate vs α，附带质量/正确率的 tradeoff 曲线

* **Fig3：动态门控**（工具失败/空结果）
  feature 强度随 step 演化曲线（成功 vs 失败）

## 12) 小但很关键的工程注意事项

* **严格 JSON 输出**：不然你的 decision boundary 会被“格式噪声”污染。OpenAI 文档里也强调 JSON mode/函数调用时的约束与边界情况（虽然你跑开源模型也建议同样做）。([OpenAI平台][6])
* **避免把工具名当作 label shortcut**：prompt 里别总写“你可以使用 search 工具”，要随机化提示方式
* **把 “工具调用” 与 “工具选择” 分开**：先做 call vs no-call（最干净），再扩展到选择哪个工具

如果你愿意，我可以在你上面这个设计基础上继续给你更“落地”的两样东西（你选一个我就直接写到能跑的程度）：

1. **最小可运行代码骨架**（controller + sandbox tools + rollout logger + TransformerLens hook 示例，按你目录结构给出）
2. **When2Call/BFCL/API-Bank 的 adapter 设计**（输入输出如何统一成你的 JSON 决策接口、如何打分）

[3]: https://arxiv.org/pdf/2504.18851?utm_source=chatgpt.com "arXiv:2504.18851v1 [cs.CL] 26 Apr 2025"
[4]: https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html?utm_source=chatgpt.com "Berkeley Function Calling Leaderboard"
[5]: https://arxiv.org/abs/2304.08244?utm_source=chatgpt.com "API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs"
[6]: https://platform.openai.com/docs/guides/function-calling?utm_source=chatgpt.com "Function calling | OpenAI API"
