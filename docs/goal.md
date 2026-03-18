# 工具调用机制可解释性

## 研究问题

> LLM 内部的哪些表征决定了是否调用工具？

## 三个核心假设

**H1 — 工具调用门控**：残差流中的一组稀疏 latent features 在 action boundary 处充当工具调用决策的门控信号。

**H2 — 证据积累**：门控随上下文中感知到的信息缺口持续积累，而非单点触发。

**H3 — 因果可控性**：对门控 features 的定向激活干预能可靠地改变工具调用行为，且不显著损害语言质量。

## 方法

- **模型**：Qwen3.5-4B、Qwen3.5-9B
- **表征工具**：TopK 稀疏自编码器（SAE），作用于模型约 3/4 和 5/6 深度处的残差流
- **SAE 训练**：两阶段——Stage 1 通用文本（OpenWebText2，~100M tokens），Stage 2 工具调用域微调（When2Call ~25K 样本）
- **特征发现**：计算 CALL/NO_CALL 条件期望差值和 AUROC，筛选 top-K 门控 features
- **语义验证**：对 top features 做 activation maximization，区分认知缺口门控与格式触发器
- **因果证据**：对 top 门控 features 做激活 steering / ablation，测量决策翻转率
