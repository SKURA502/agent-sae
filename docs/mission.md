# Tool-use Mechanistic Basis 任务清单

## 阶段一：环境搭建

### 1.1 安装 Miniconda
```bash
# macOS
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh
```

### 1.2 创建虚拟环境
```bash
conda create -n agent_sae python=3.11
conda activate agent_sae
```

### 1.3 安装核心依赖
```bash
pip install torch transformers accelerate vllm
pip install pydantic datasets huggingface_hub
pip install dictionary_learning sae_dashboard
pip install scikit-learn pandas matplotlib seaborn
pip install jupyter wandb tqdm
```

---

## 阶段二：模型下载

### 2.1 目标模型（Qwen3.5 系列）

```bash
# 配置 HuggingFace token
huggingface-cli login

# 主模型：Qwen3.5-4B（起步，显存需求低）
huggingface-cli download Qwen/Qwen3.5-4B --local-dir ./models/qwen3.5-4b

# 扩展对比：Qwen3.5-9B（有更多 GPU 时可选）
huggingface-cli download Qwen/Qwen3.5-9B --local-dir ./models/qwen3.5-9b
```

### 2.2 确认模型参数

```bash
python -c "
from transformers import AutoConfig
for model_id in ['Qwen/Qwen3.5-4B', 'Qwen/Qwen3.5-9B']:
    c = AutoConfig.from_pretrained(model_id)
    n = c.num_hidden_layers
    h = c.hidden_size
    print(f'{model_id}: layers={n}, hidden={h}, '
          f'layer_3/4={int(n*3/4)}, layer_5/6={int(n*5/6)}, '
          f'sae_dict={h*8}, topk_k={h//32}')
"
```

- [ ] Qwen3.5-4B 下载并加载验证
- [ ] Qwen3.5-9B 下载（可选，等 4B 结果稳定后）
- [ ] Hook 点验证（可正确提取残差流激活）

---

## 阶段三：数据集准备

### 3.1 When2Call（Stage 2 主数据集）

**⚠️ 重要**：仅使用 **Pref** 数据，不使用 SFT 数据。

| 文件 | 样本量 | 实际标签 | 用途 |
|------|--------|---------|------|
| `When2Call/data/train/when2call_train_sft.jsonl` | 15,000 | 全部 NO_CALL | ❌ 不用于 Stage 2 |
| `When2Call/data/train/when2call_train_pref.jsonl` | 9,000 | CALL 3K / NO_CALL 6K | ✅ Stage 2 主数据 |
| `When2Call/data/test/when2call_test_mcq.jsonl` | 3,652 | 有标签 | ✅ 评测 |

- [ ] 确认 `when2call_train_pref.jsonl` 格式（`chosen_response` 字段存在且包含 `<TOOLCALL>` 标记）
- [ ] 修复 `tasks/when2call_adapter.py`（从 `chosen_response` 解析 CALL/NO_CALL，不再查找 `should_call`）

### 3.2 BFCL（辅助 NO_CALL 数据 + 外部评测）

```
gorilla-main/berkeley-function-call-leaderboard/bfcl_eval/data/
├── BFCL_v4_irrelevance.json        # 240 条 NO_CALL
├── BFCL_v4_live_irrelevance.json   # 884 条 NO_CALL（共 1,124 可用）
├── BFCL_v4_live_simple.json        # 用于外部 H1 泛化性评测
└── ...
```

- [ ] 确认 BFCL v4 格式并更新 `tasks/bfcl_adapter.py`

### 3.3 H2 轨迹数据（待定）

多步轨迹数据方案尚未确定，见 [question.md](question.md)。当前优先级：H1 → H3 → H2。

---

## 阶段四：代码框架修复与搭建

### 4.1 必须修复（阻断项）

| 序号 | 模块 | 问题 | 优先级 |
|------|------|------|--------|
| 1 | `tasks/when2call_adapter.py` | `should_call` 字段不存在，需改为解析 `chosen_response` 中的 `<TOOLCALL>` | 🔴 必须 |
| 2 | `configs/model_config.yaml` | 更新模型名称为 Qwen3.5-4B，层号动态计算 | 🔴 必须 |

### 4.2 核心模块实现顺序

| 序号 | 模块 | 文件 | 优先级 |
|------|------|------|--------|
| 1 | 配置文件 | `configs/*.yaml` | ⭐⭐⭐ |
| 2 | 工具 Schema | `controller/tool_schema.py` | ⭐⭐⭐ |
| 3 | 沙盒工具 | `controller/sandbox_tools/*.py` | ⭐⭐⭐ |
| 4 | Agent 循环 | `controller/agent_loop.py` | ⭐⭐⭐ |
| 5 | 输出解析 | `controller/output_parser.py` | ⭐⭐⭐ |
| 6 | 数据适配（修复）| `tasks/when2call_adapter.py` | ⭐⭐⭐ |
| 7 | Rollout 生成 | `run/generate_rollouts.py` | ⭐⭐ |
| 8 | 激活缓存 | `run/cache_activations.py` | ⭐⭐ |
| 9 | SAE 训练 | `sae/train_sae.py` | ⭐⭐ |
| 10 | 分析脚本 | `analysis/*.py` | ⭐ |

---

## 阶段五：实验执行

### 5.1 快速验证（先做，成本低）

在全面展开前，用小规模数据验证核心假设可行性：

| 步骤 | 操作 | 通过标准 |
|------|------|---------|
| 1 | 修复 adapter + 加载 100 个 pref 样本 | 无报错，CALL/NO_CALL 各约 50 条 |
| 2 | 跑通激活提取 → SAE 更新一个 batch | 无 OOM，loss 下降 |
| 3 | 500 CALL + 500 NO_CALL 训练 Stage 2 SAE | top feature AUROC > 0.65 |
| 4 | 简单 steering 实验（α 扫描） | 可观测到 flip rate 变化趋势 |

### 5.2 正式实验清单（按顺序执行）

| 序号 | 实验 | 输出 | 对应假设 |
|------|------|------|---------|
| 1 | Stage 1 SAE 训练（OpenWebText2, 50M tokens） | `outputs/sae_checkpoints/stage1/` | 基础 |
| 2 | Stage 2 SAE 训练（When2Call Pref, 1:1 平衡） | `outputs/sae_checkpoints/stage2/` | H1+H3 |
| 3 | 相关性分析：feature → decision AUROC | Fig 1 数据 | H1 |
| 4 | 线性探针：小 K 预测 AUC | AUC vs K 曲线 | H1 |
| 5 | 语义验证：activation maximization | 区分 gate vs 格式特征 | H1 |
| 6 | 因果干预：flip rate vs α | Fig 2 数据 | H3 |
| 7 | 外部验证：BFCL 上的 H1 泛化性 | 泛化结果 | H1 |
| 8 | Evidence Accumulation 实验（数据方案待定）| Fig 3 数据 | H2 |
| 9 | 扩展：Qwen3.5-9B 对比 | 对比表格 | 补充 |

### 5.3 SAE 训练超参

```yaml
# 基于 Qwen3.5-4B（需运行阶段二的参数确认脚本后填入）
model: Qwen/Qwen3.5-4B
hidden_size: <待填写>
num_layers: <待填写>

sae:
  dictionary_size: <hidden_size * 8>
  target_sparsity: <hidden_size / 32>   # TopK 的 K 值
  layers: [<int(n*3/4)>, <int(n*5/6)>]  # 动态计算
  batch_size: 4096
  buffer_size: 8192

stage1:
  learning_rate: 1e-4
  target_tokens: 50_000_000

stage2:
  learning_rate: 1e-5     # 保守，防 6K 样本过拟合
  num_epochs: 1
  early_stopping: true
  balance: true            # CALL:NO_CALL = 1:1
```

---

## 阶段六：结果整理

### 6.1 核心图表

- [ ] **Fig 1**（H1）：Gate feature 可分离性
  - top-50 features 的 AUROC 分布直方图
  - AUC vs K 曲线（K=10/20/50 时标注）

- [ ] **Fig 2**（H3）：因果干预效果
  - flip rate vs α 曲线
  - 质量保持度 vs α 曲线
  - steering vs ablation 对比

- [ ] **Fig 3**（H2）：动态门控（数据方案待定）
  - gate feature 强度随 step 演化曲线
  - 成功 episode vs 失败 episode 对比

### 6.2 核心表格

- [ ] 数据集统计表（When2Call Pref 实际分布、BFCL Irrelevance 统计）
- [ ] 模型对比表（Qwen3.5-4B vs 9B 的 gate feature 可分离性）
- [ ] Ablation 表（不同层、不同 K、Stage 1-only vs Stage 2 的效果对比）

---

## 检查清单

### 环境
- [ ] Miniconda 安装完成
- [ ] 虚拟环境创建并激活
- [ ] 所有依赖包安装完成
- [ ] GPU 环境可用（CUDA 验证）

### 模型
- [ ] Qwen3.5-4B 下载完成
- [ ] 模型加载验证通过
- [ ] 层数和 hidden_size 确认（运行参数确认脚本）
- [ ] Hook 点验证（可正确提取残差流激活）
- [ ] thinking 模式关闭验证

### 数据
- [ ] `when2call_train_pref.jsonl` 格式确认
- [ ] `when2call_adapter.py` 修复完成
- [ ] BFCL Irrelevance 数据格式确认
- [ ] 端到端数据管线验证（100 样本小测试）

### 代码
- [ ] Agent 闭环跑通
- [ ] Stage 1 SAE 训练收敛
- [ ] Stage 2 SAE 训练收敛（快速验证通过）
- [ ] 分析脚本可运行

### 分析
- [ ] H1 快速验证通过（AUROC > 0.65）
- [ ] H3 flip rate 有明显变化趋势
- [ ] H2 数据方案确定（见 question.md）

---

## 常见问题

### Q1: 模型显存不足
- Qwen3.5-4B 推理约需 10-12GB VRAM（bfloat16）
- 使用 `accelerate` 进行模型并行
- 使用 vLLM 提高推理效率
- 减小 batch size 或使用梯度检查点

### Q2: SAE Stage 2 过拟合（loss 下降但 AUROC 不稳定）
- 降低学习率（从 1e-5 → 5e-6）
- 增大 Stage 2 数据量（加入 BFCL Irrelevance 或合成数据）
- 冻结 encoder，只更新 decoder

### Q3: Stage 2 AUROC 过低（< 0.65）
- 尝试不同层（做 layer sweep）
- 增大 dictionary_size
- 检查 adapter 是否正确提取 CALL/NO_CALL 标签

### Q4: 干预效果不明显
- 增大干预强度 α
- 尝试多个 features 联合 steering
- 确认 thinking 模式已关闭

---

## 时间规划

| 阶段 | 预估时间 | 里程碑 |
|------|----------|--------|
| 环境搭建 | 1-2 天 | 环境可用，模型参数确认 |
| 数据修复与验证 | 2-3 天 | Adapter 修复，100 样本端到端通过 |
| Stage 1 SAE | 3-5 天 | SAE 收敛（50M tokens） |
| Stage 2 SAE + 快速验证 | 2-3 天 | AUROC > 0.65 通过 |
| H1 完整实验 | 3-5 天 | Fig 1 数据完成 |
| H3 干预实验 | 3-5 天 | Fig 2 数据完成 |
| H2 轨迹实验 | 5-7 天 | Fig 3 数据完成（方案待定） |
| 扩展 + 论文写作 | 7-14 天 | 初稿完成 |

**总计：约 6-10 周**（取决于 H2 数据方案和 GPU 数量）
