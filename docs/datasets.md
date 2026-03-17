# 数据集决策文档

## 实际规模（已核实）

| 文件 | 规模 | 标签分布 |
|------|------|---------|
| `when2call_train_pref.jsonl` | 9,000 | 3K CALL + 6K NO_CALL（`chosen_response.content` 含 `<TOOLCALL>`） |
| `when2call_train_sft.jsonl` | 15,000 | 全部 NO_CALL |
| `when2call_test_mcq.jsonl` | 3,652 | 1,295 tool_call / 1,295 cannot_answer / 1,062 request_for_info |
| `BFCL_v4_irrelevance.json` | 240 | 全部 NO_CALL |
| `BFCL_v4_live_irrelevance.json` | 884 | 全部 NO_CALL |
| `BFCL_v4_live_relevance.json` | 16 | 全部 CALL（样本量不足，不可单独作泛化测试集） |
| `BFCL_v4_simple_python.json` | 400 | 全部 CALL（always-call 场景） |
| `BFCL_v4_live_simple.json` | 258 | 全部 CALL（always-call 场景） |

**tau2-bench / vitabench**：仅含任务描述 + 评估标准（`tasks.json`），无预存轨迹，不能直接作为训练或测试语料。

---

## 各用途决策表

| 用途 | 数据集 | 规模 | 关键说明 |
|------|-------|------|---------|
| Stage 2 域适应（可选） | When2Call SFT + BFCL simple/parallel/multiple | ~15K+ 条 | tau2-bench 无现成轨迹，不可用 |
| Stage 2 特征专项 | When2Call Pref 平衡采样 | 3K+3K | 从 9K pref 集 1:1 采样 |
| 特征发现 | When2Call Pref 3K+3K + BFCL Irrel 1,124 + BFCL Simple ~240 | ~6.6K | BFCL Simple 补充分布外 CALL 样本 |
| H1 主测 | When2Call MCQ 二类子集（tool_call vs cannot_answer） | 2,590 | 排除 request_for_info 1,062 条 |
| H1 泛化验证 | BFCL Irrel 1,124（NO_CALL）+ BFCL Simple ~240（CALL） | ~1.4K | 修复 live_relevance 仅 16 条的缺陷 |
| H2 rollout 生成 | 合成 sandbox rollout；tau2-bench `tasks.json` 作 prompt 种子 | 100–500 episodes | tau2-bench 提供业务语境，不是激活数据 |
| H3 steering | When2Call MCQ 二类子集（同 H1） | 2,590 | 排除 request_for_info，保持翻转率定义干净 |

---

## 关键约束

**1. MCQ 三类标签**：`request_for_info`（1,062 条）语义上介于 CALL 和 NO_CALL 之间——模型知道信息不足但未决定调用工具。排除后得到完美平衡的 1,295 vs 1,295 二类子集。这 1,062 条可在论文中作为 out-of-distribution 探针单独分析，反而是有意义的结果。

**2. BFCL 的先天局限**：BFCL 大多数类别（simple/parallel/multiple）是 always-call 场景，无 NO_CALL 对照。只有 irrelevance 类覆盖了 NO_CALL 决策。因此 BFCL 泛化测试只能是补充性的，不能作为主评测集。When2Call 是目前唯一系统性标注 CALL/NO_CALL 决策边界的公开数据集。

**3. tau2-bench 的角色澄清**：
- **不适用**：SAE 训练域适应（需要运行模型生成轨迹后才能使用，是 Stage 4 产物）
- **适用**：H2 rollout 生成的 prompt 种子（retail/telecom 场景提供真实业务语境，优于纯合成 prompt）
