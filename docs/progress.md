# 项目进展

## 当前状态（2026-03-17）

### 代码
- [x] 架构设计完成：controller、SAE、analysis 模块均已搭建框架
- [x] 数据分析完成：标签分布已核实，adapter bug 已定位
- [ ] `tasks/when2call_adapter.py` 存在阻断性 bug——代码读取数据中不存在的 `should_call` 字段，需改为从 `chosen_response` 解析 `<TOOLCALL>` 标签
- [ ] `tasks/bfcl_adapter.py` 未针对 v4 格式验证
- [ ] 端到端流水线未测试

### 数据
- [x] When2Call Pref 已确认：3K CALL + 6K NO_CALL（共 9K）；SFT 分割全为 NO_CALL，不可用
- [x] BFCL Irrelevance 已确认：1,124 条 NO_CALL 样本
- [ ] Stage 2 域适应用的无标签工具调用语料未整理（可选，推荐）
- [ ] Sandbox rollout 未生成（H2 必需）

---

## 阻断项（必须在任何实验前解决）

1. **修复 `when2call_adapter.py`**：从 `chosen_response` 解析 `<TOOLCALL>` 标签，替代读取 `should_call` 字段
2. **确认 Qwen3.5-4B 架构**：运行模型配置检查，获取 hidden_size、num_layers，计算 SAE 目标层
3. **冒烟测试**：100 个 Pref 样本 → 激活提取 → 一个 SAE 更新 batch，全程无报错

---

## 分阶段后续计划

### 第一阶段——解除阻断（立即）
- 修复 `when2call_adapter.py`
- 运行 100 样本冒烟测试

### 第二阶段——H1 快速验证（1–2 周）
- Stage 1 SAE 训练（OpenWebText2，50M tokens）
- Stage 2 小规模验证：500 CALL + 500 NO_CALL
- 计算 top-feature AUROC；通过门槛 > 0.65 后再全面展开

### 第三阶段——H1 + H3 完整实验
- Stage 2 SAE 全量训练（6K 平衡 When2Call Pref 样本）
- 相关性分析 + 线性探针 → Fig 1
- Steering / ablation 实验 → Fig 2
- BFCL 泛化性验证

### 第四阶段——H2 证据积累
- 实现 `agent_loop.py` 的 sandbox rollout 收集
- 生成 100–500 个多步 episode（调整 `p_fail`、`p_empty`、`p_corrupt`）
- 追踪门控 feature 强度随 step 演化 → Fig 3

### 第五阶段——扩展与收尾
- Qwen3.5-9B 规模对比
- Ablation：仅 Stage 1 SAE vs Stage 2 SAE 的效果对比
- 论文写作

---

## 实验优先级：H1 → H3 → H2

H1 和 H3 只依赖静态有标签数据（When2Call Pref + test MCQ），adapter bug 修复后即可推进。H2 需要 agent loop 完整跑通后才能生成 sandbox rollout。

---

## 待决问题

- **Stage 2 域适应**：在有标签的 6K 微调之前，是否值得增加一步无标签工具调用语料（When2Call SFT + BFCL 完整语料，~5–10M tokens）？此举可将 SAE 表征空间向工具调用场景靠拢，缩小与 Stage 1 的 token 量级差距。待 H1 快速验证结果出来后再决定。
- **H2 rollout 设计**：每个 episode 需要多少步，`p_fail` 设置为多少，才能得到干净的积累信号？待 sandbox 跑通后确定。
