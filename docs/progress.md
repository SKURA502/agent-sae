# 项目进展

## 当前状态（2026-03-18）

### 代码
- [x] 架构设计完成：controller、SAE、analysis 模块均已搭建框架
- [x] 数据分析完成：标签分布已核实
- [x] `tasks/when2call_adapter.py`：从 `chosen_response` 解析 `<TOOLCALL>` 标签（pref）；SFT 分割兜底 NO_CALL；MCQ `request_for_info` 返回 UNCERTAIN 供调用方过滤
- [x] `tasks/__init__.py`：移除 BFCL 导出，BFCL adapter 标记为弃用
- [ ] 端到端流水线未测试

### 数据
- [x] When2Call Pref 已确认：3K CALL + 6K NO_CALL（共 9K），用于 Stage 2 训练 + 特征发现
- [x] When2Call SFT 已确认：15K，全为 NO_CALL，用于 Stage 2 训练（SAE 无监督，不需标签平衡）
- [x] When2Call MCQ 三类标签已确认：1,295 tool_call / 1,295 cannot_answer / 1,062 request_for_info——H1/H3 使用时过滤 UNCERTAIN，取 2,590 条二类子集；整个 test_mcq 文件排除出训练
- [x] tau2-bench 角色已澄清：无现成轨迹，不用于 SAE 训练；H2 rollout 生成时作 prompt 种子
- [x] BFCL 已移除出流水线
- [ ] Sandbox rollout 未生成（H2 必需）

---

## 阻断项（必须在任何实验前解决）

1. **确认 Qwen3.5-4B 架构**：运行模型配置检查，获取 hidden_size、num_layers，计算 SAE 目标层
2. **冒烟测试**：100 个 Pref 样本 → 激活提取 → 一个 SAE 更新 batch，全程无报错

---

## 分阶段后续计划

### 第一阶段——解除阻断（立即）
- 运行 100 样本冒烟测试

### 第二阶段——H1 快速验证（1–2 周）
- Stage 1 SAE 训练（OpenWebText2，50M tokens）
- Stage 2 小规模验证：从 Pref 中取 500 CALL + 500 NO_CALL
- 计算 top-feature AUROC；通过门槛 > 0.65 后再全面展开

### 第三阶段——H1 + H3 完整实验
- Stage 2 SAE 全量训练（When2Call Pref 9K + SFT 15K）
- 特征发现：Pref 全量，`E[f|CALL] − E[f|NO_CALL]` 均值差异筛选
- 相关性分析 + 线性探针 → Fig 1
- Steering / ablation 实验 → Fig 2

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

H1 和 H3 只依赖静态有标签数据（When2Call Pref + test MCQ），冒烟测试通过后即可推进。H2 需要 agent loop 完整跑通后才能生成 sandbox rollout。

---

## 待决问题

- **H2 rollout 设计**：每个 episode 需要多少步，`p_fail` 设置为多少，才能得到干净的积累信号？待 sandbox 跑通后确定。
- **request_for_info 分析**：MCQ 中的 1,062 条 request_for_info 样本的 SAE feature 激活分布如何？是否介于 CALL 和 NO_CALL 之间？可作为 H1 的补充发现。
