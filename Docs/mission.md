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

### 2.1 申请模型访问权限
- [ ] 申请 Llama-3-8B 访问权限（Meta 官网或 HuggingFace）
- [ ] 确认 Qwen-3 和 Gemma-3 的访问权限

### 2.2 下载模型
```bash
# 配置 HuggingFace token
huggingface-cli login

# 下载模型（选择其一作为起步）
huggingface-cli download meta-llama/Llama-3-8B-Instruct --local-dir ./models/llama3-8b
huggingface-cli download Qwen/Qwen3-8B --local-dir ./models/qwen3-8b
huggingface-cli download google/gemma-3-4b-it --local-dir ./models/gemma3-4b
```

---

## 阶段三：数据集下载

### 3.1 When2Call（主数据集）
```bash
# 从 HuggingFace 下载
huggingface-cli download <when2call-dataset-path> --local-dir ./data/raw/when2call
```
> 备注：检查 arXiv:2504.18851 论文获取官方数据集链接

### 3.2 BFCL（Berkeley Function Calling Leaderboard）
```bash
git clone https://github.com/ShishirPatil/gorilla.git
cp -r gorilla/berkeley-function-call-leaderboard/data ./data/raw/bfcl
```

### 3.3 API-Bank
```bash
git clone https://github.com/AlibabaResearch/DAMO-ConvAI.git
cp -r DAMO-ConvAI/api-bank ./data/raw/apibank
```

---

## 阶段四：代码框架搭建

### 4.1 创建目录结构
```bash
mkdir -p agent_sae_tooluse/{configs,controller/sandbox_tools,tasks,run,sae,analysis}
mkdir -p agent_sae_tooluse/{data/{raw,processed,rollouts,activations},outputs/{sae_checkpoints,analysis_results,figures},scripts}
```

### 4.2 核心模块实现顺序

| 序号 | 模块 | 文件 | 优先级 |
|------|------|------|--------|
| 1 | 配置文件 | `configs/*.yaml` | ⭐⭐⭐ |
| 2 | 工具 Schema | `controller/tool_schema.py` | ⭐⭐⭐ |
| 3 | 沙盒工具 | `controller/sandbox_tools/*.py` | ⭐⭐⭐ |
| 4 | Agent 循环 | `controller/agent_loop.py` | ⭐⭐⭐ |
| 5 | 输出解析 | `controller/output_parser.py` | ⭐⭐⭐ |
| 6 | 数据适配 | `tasks/when2call_adapter.py` | ⭐⭐ |
| 7 | Rollout 生成 | `run/generate_rollouts.py` | ⭐⭐ |
| 8 | 激活缓存 | `run/cache_activations.py` | ⭐⭐ |
| 9 | SAE 训练 | `sae/train_sae.py` | ⭐⭐ |
| 10 | 分析脚本 | `analysis/*.py` | ⭐ |

---

## 阶段五：实验执行

### 5.1 实验清单（按顺序执行）

| 序号 | 实验 | 输出 | 预估时间 |
|------|------|------|----------|
| 1 | 闭环验证：controller + sandbox tools 跑通 | 成功运行日志 | 1-2 天 |
| 2 | 生成 rollouts：When2Call 10k-50k episodes | `data/rollouts/*.jsonl` | 1-3 天 |
| 3 | 缓存激活：W_pre/W_post 提取 | `data/activations/*.pt` | 1 天 |
| 4 | SAE 训练：1-2 层起步 | `outputs/sae_checkpoints/` | 1-2 天 |
| 5 | 相关性分析：feature→decision 统计 | Fig 1 数据 | 1 天 |
| 6 | 线性探测：小 K 预测 AUC | AUC 曲线 | 半天 |
| 7 | 因果干预：steering/ablation | Fig 2 数据 | 2-3 天 |
| 8 | 扩展验证：BFCL + API-Bank | 泛化结果 | 2 天 |
| 9 | 动态分析：工具失败/空返回 | Fig 3 数据 | 1-2 天 |
| 10 | 模型对比：base vs tool-tuned | 对比表格 | 2-3 天 |

### 5.2 SAE 训练超参

```yaml
# configs/sae_config.yaml
dictionary_size: 32768  # hidden_size × 8
target_sparsity: 128    # hidden_size / 32
layers: [18, 24]        # 3/4 和 5/6 位置（以32层模型为例）
batch_size: 4096
learning_rate: 1e-4
num_epochs: 10
```

---

## 阶段六：结果整理

### 6.1 核心图表

- [ ] **Fig 1**：Gate feature 可分离性
  - top features 的 `E[f|CALL] - E[f|NO_CALL]` 分布
  - AUROC 分布直方图
  - 小 K 线性模型 AUC vs K 曲线

- [ ] **Fig 2**：因果干预效果
  - flip rate vs α 曲线
  - 质量保持度 vs α 曲线
  - steering vs ablation 对比

- [ ] **Fig 3**：动态门控
  - feature 强度随 step 演化曲线
  - 成功 episode vs 失败 episode 对比
  - 工具返回质量对后续 gate 的影响

### 6.2 核心表格

- [ ] 数据集统计表（When2Call/BFCL/API-Bank 样本量、CALL/NO_CALL 比例）
- [ ] 模型对比表（Llama/Qwen/Gemma 的 gate feature 可分离性、干预效果）
- [ ] Ablation 表（不同层、不同 K 的效果对比）

---

## 检查清单

### 环境
- [ ] Miniconda 安装完成
- [ ] 虚拟环境创建并激活
- [ ] 所有依赖包安装完成
- [ ] GPU 环境可用（CUDA 验证）

### 模型
- [ ] 至少一个目标模型下载完成
- [ ] 模型加载验证通过
- [ ] Hook 点验证（可正确提取激活）

### 数据
- [ ] When2Call 数据集下载并解析
- [ ] 数据适配器编写完成
- [ ] 统一格式验证通过

### 代码
- [ ] Agent 闭环跑通
- [ ] Rollout 生成正常
- [ ] 激活缓存正常
- [ ] SAE 训练收敛

### 分析
- [ ] 相关性分析完成
- [ ] 线性探测 AUC 达标
- [ ] 干预实验完成
- [ ] 核心图表生成

---

## 常见问题

### Q1: 模型显存不足
- 使用 `accelerate` 进行模型并行
- 使用 vLLM 提高推理效率
- 减小 batch size 或使用梯度检查点

### Q2: SAE 训练不收敛
- 检查数据平衡（CALL:NO_CALL ≈ 1:1）
- 调整 sparsity 约束强度
- 尝试不同层位置

### Q3: 干预效果不明显
- 增大干预强度 α
- 尝试组合多个 features
- 检查 feature 是否真正与 decision 相关

---

## 时间规划

| 阶段 | 预估时间 | 里程碑 |
|------|----------|--------|
| 环境搭建 | 1-2 天 | 环境可用 |
| 模型/数据准备 | 2-3 天 | 数据加载正常 |
| 代码框架 | 5-7 天 | Agent 闭环跑通 |
| Rollout 生成 | 3-5 天 | 50k episodes |
| SAE 训练 | 3-5 天 | SAE 收敛 |
| 机制分析 | 5-7 天 | 核心图表完成 |
| 扩展实验 | 5-7 天 | 完整结果 |
| 论文写作 | 7-14 天 | 初稿完成 |

**总计：约 5-8 周**
