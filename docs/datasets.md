# 数据集决策文档

## 实际规模（已核实）

| 文件 | 规模 | 标签分布 |
|------|------|---------|
| `when2call_train_pref.jsonl` | 9,000 | 3K CALL + 6K NO_CALL（`chosen_response.content` 含 `<TOOLCALL>`） |
| `when2call_train_sft.jsonl` | 15,000 | 全部 NO_CALL |
| `when2call_test_mcq.jsonl` | 3,652 | 1,295 tool_call / 1,295 cannot_answer / 1,062 request_for_info |

**tau2-bench / vitabench**：仅含任务描述 + 评估标准（`tasks.json`），无预存轨迹，不能直接作为训练或测试语料。

---

## 各用途决策表

| 用途 | 数据集 | 规模 | 关键说明 |
|------|-------|------|---------|
| Stage 2 训练 | When2Call Pref + SFT | ~24K 条 | 排除整个 `when2call_test_mcq.jsonl`（含 request_for_info） |
| 特征发现 | When2Call Pref（全量） | 9K | 分类别计算激活差异，无需人工平衡采样 |
| H1 主测 | When2Call MCQ 二类子集（tool_call vs cannot_answer） | 2,590 | 排除 request_for_info 1,062 条 |
| H2 rollout 生成 | 合成 sandbox rollout；tau2-bench `tasks.json` 作 prompt 种子 | 100–500 episodes | tau2-bench 提供业务语境，不是激活数据 |
| H3 steering | When2Call MCQ 二类子集（同 H1） | 2,590 | 排除 request_for_info，保持翻转率定义干净 |

---

## 关键约束

**1. 训练/测试严格隔离**：`when2call_test_mcq.jsonl` 整个文件（3,652 条，含 request_for_info）均不参与 Stage 2 训练和特征发现，确保 H1/H3 评测在完全未见数据上进行。

**2. Stage 2 类别不平衡**：`train_sft` 15K 全为 NO_CALL，加上 `train_pref` 中 6K NO_CALL，总计约 21K NO_CALL vs 3K CALL（7:1）。SAE 为无监督自编码器，不平衡不影响训练收敛；特征发现阶段分类别计算条件期望，差值方向不受影响。

**3. MCQ 三类标签**：`request_for_info`（1,062 条）语义介于 CALL 和 NO_CALL 之间，整体排除出训练和评测。H1/H3 使用完美平衡的 tool_call vs cannot_answer 二类子集（1,295 vs 1,295）；request_for_info 可在论文中作为 out-of-distribution 探针单独分析。

**4. tau2-bench 的角色澄清**：
- **不适用**：SAE 训练域适应（需要运行模型生成轨迹后才能使用，是 Stage 4 产物）
- **适用**：H2 rollout 生成的 prompt 种子（retail/telecom 场景提供真实业务语境，优于纯合成 prompt）
