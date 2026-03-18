# 项目进展

## 当前状态（2026-03-18）

### 代码
- [x] `run/when2call_adapter.py`：从 `chosen_response` 解析 `<TOOLCALL>` 标签（pref）；SFT 分割兜底 NO_CALL；MCQ `request_for_info` 返回 UNCERTAIN 供调用方过滤
- [x] SAE 两阶段训练搭建完成
- [x] **H1 特征发现**：`analysis/feature_discovery.py`——mean_diff + per-feature AUROC，输出 top-K JSON + 分布直方图（Fig 1a）
- [x] **H1 线性探针**：`analysis/linear_probe.py`——LogisticRegression 5-fold CV，AUC vs K 曲线 + ROC 曲线（Fig 1b）
- [x] **H3 Steering**：`analysis/steering.py`——`SteeringHook` 干预目标层激活，翻转率 + Δperplexity，Fig 2
- [x] **H2 Controller**：`controller/sandbox_tools.py`（search/calculator/lookup，随机化名称，噪声注入）+ `controller/agent_loop.py`（最小 agent 循环，per-step 激活收集，smoke test 通过）
- [x] **H2 Rollout 基础设施**：`run/rollout_logger.py` + `run/generate_rollouts.py`——tau2-bench `reason_for_call` 作 prompt 种子，批量生成 + per-step 激活保存
- [x] **H2 轨迹分析**：`analysis/trajectory_analysis.py`——CALL vs NO_CALL episode 的 feature 强度轨迹，Fig 3
- [x] `scripts/run_pipeline.sh` 补全 Step 3b–7（特征发现 → 线性探针 → steering → H2 rollout + 轨迹分析）
- [ ] 端到端流水线未测试（待 SAE 训练完成后）
- [x] `README.md` 已补全

### 数据
- [x] When2Call Pref 已确认：3K CALL + 6K NO_CALL（共 9K），用于 Stage 2 训练 + 特征发现
- [x] When2Call SFT 已确认：15K，全为 NO_CALL，用于 Stage 2 训练（SAE 无监督，不需标签平衡）
- [x] When2Call MCQ 三类标签已确认：1,295 tool_call / 1,295 cannot_answer / 1,062 request_for_info——H1/H3 使用时过滤 UNCERTAIN，取 2,590 条二类子集；整个 test_mcq 文件排除出训练
- [x] tau2-bench 角色已澄清：无现成轨迹，不用于 SAE 训练；H2 rollout 生成时取 `reason_for_call` 作 prompt 种子
- [ ] Sandbox rollout 未生成（H2 必需，待 SAE 训练完成后运行 `RUN_H2=1 scripts/run_pipeline.sh`）

---

## 分阶段后续计划

### 第一阶段——SAE训练
- Stage 1 SAE 预训练（OpenWebText2，50M tokens）
- Stage 2 SAE tool-call 特征增强训练（When2Call Pref 9K + SFT 15K）

### 第二阶段——H1 + H3 完整实验 ✅ 脚本就绪
- [x] 特征发现脚本：`analysis/feature_discovery.py`
- [x] 线性探针脚本：`analysis/linear_probe.py`
- [x] Steering 实验脚本：`analysis/steering.py`
- [ ] 运行实验（待 SAE 训练完成）

### 第三阶段——H2 证据积累 ✅ 脚本就绪
- [x] `controller/agent_loop.py` sandbox rollout 收集
- [x] `run/generate_rollouts.py`：tau2-bench 任务作 prompt 种子，支持 p_fail/p_empty/p_corrupt
- [x] `analysis/trajectory_analysis.py`：门控 feature 强度 vs step 可视化
- [ ] 运行实验（待 SAE 训练完成，`RUN_H2=1 scripts/run_pipeline.sh`）

### 第四阶段——扩展与收尾
- Qwen3.5-9B 规模对比
- Ablation：仅 Stage 1 SAE vs Stage 2 SAE 的效果对比
- 论文写作

---

## 实验优先级：H1 → H3 → H2

H1 和 H3 只依赖静态有标签数据（When2Call Pref + test MCQ）。H2 需要 agent loop 完整跑通后才能生成 sandbox rollout。

---

## 待决问题

- **H2 rollout 设计**：每个 episode 需要多少步，`p_fail` 设置为多少，才能得到干净的积累信号？待 sandbox 跑通后确定。
- **request_for_info 分析**：MCQ 中的 1,062 条 request_for_info 样本的 SAE feature 激活分布如何？是否介于 CALL 和 NO_CALL 之间？可作为 H1 的补充发现。
