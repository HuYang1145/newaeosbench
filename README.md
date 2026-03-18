# Constellation

本仓库是论文 "Towards Realistic Earth-Observation Constellation Scheduling: Benchmark and Methodology" 的官方实现。

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-purple)](https://neurips.cc/virtual/2025/loc/san-diego/poster/116515)
[![arXiv](https://img.shields.io/badge/arXiv-2510.26297-b31b1b.svg)](https://arxiv.org/abs/2510.26297)

## 项目核心原理

### 技术架构：行为克隆 + 物理仿真

本项目使用**行为克隆（Behavior Cloning）**方法训练 Transformer 模型学习卫星调度策略：

```
专家算法（OptimalAlgorithm）→ 在 Basilisk 环境中运行 → 生成轨迹数据 → 训练 Transformer 模型
```

**关键点**：
- **Basilisk 不提供调度方案**，它只是物理仿真环境（模拟卫星轨道、姿态、能源）
- **OptimalAlgorithm 是贪心专家**，基于几何距离做决策（选择最近的可见任务）
- **Transformer 学习专家行为**，通过监督学习拟合专家的决策分布

### 数据生成流程（与 origin-aeos 一致）

**第一步：生成专家轨迹**
```bash
# 使用 OptimalAlgorithm 在 Basilisk 环境中运行
python tools/generate_trajectories.py
```

每个轨迹文件（`.pth`）包含：
- **状态**：卫星位置、姿态、电量、任务信息（每个时间步）
- **动作**：每颗卫星选择的任务 ID（-1 表示不观测）
- **可见性**：卫星-任务可见性矩阵（Basilisk 计算）

**第二步：训练 Transformer**
```bash
# 监督学习：让模型预测专家的动作
CUDA_VISIBLE_DEVICES=0 auto_torchrun -m constellation.new_transformers.train test config.py
```

损失函数：交叉熵损失（分类问题）
```python
loss = CrossEntropyLoss(model_logits, expert_actions)
```

### Transformer 模型输入输出

**输入**（每个时间步）：
- **任务特征** `[batch, num_tasks, 7]`：释放时间、截止时间、持续时间、经纬度、优先级、完成进度
- **卫星特征** `[batch, num_sats, 18]`：轨道参数、质量、转动惯量、电池电量、反作用轮转速、姿态
- **时间步** `[batch]`：当前时间（0-8639，每步 10 秒）

**输出**：
- **动作概率** `[batch, num_sats, num_tasks+1]`：每颗卫星对每个任务的选择概率（+1 是"不观测"选项）

**模型架构**：Encoder-Decoder Transformer
- **Encoder**：处理任务信息，生成任务表示（12 层，512 维）
- **Decoder**：处理卫星信息，通过 Cross-Attention 匹配卫星与任务（12 层，512 维）

### Basilisk 的作用

Basilisk 是 NASA 开发的航天器仿真框架，在本项目中提供：

1. **轨道动力学**：计算卫星在地球引力场中的运动（考虑 J2 摄动）
2. **姿态控制**：模拟卫星转向目标的过程（MRP 控制器 + 反作用轮）
3. **能源系统**：模拟太阳能板发电、电池充放电、传感器功耗
4. **可见性计算**：判断卫星能否看到地面目标（球面几何 + 视场角约束）

**准确度**：
- 轨道传播：米级精度（使用 SPICE 系统）
- 姿态控制：高真实性（简化了柔性振动等细节）
- 能源系统：中等真实性（简化模型，足够用于调度研究）

**Basilisk 不输出调度方案**，它只是执行环境，验证调度决策的物理可行性。

## 安装

```bash
sudo apt install ffmpeg libpq-dev swig
bash setup.sh
```

### 已知问题及解决方案

#### Basilisk 版本兼容性

**问题**: 项目需要与 numpy 1.x 兼容的 Basilisk 版本（todd-ai 要求），但新版 Basilisk 需要 numpy 2.0+。

**解决方案**: 使用 Basilisk commit `786cb285d`（numpy 2.0 支持前的最后版本）。`setup.sh` 会自动处理。

#### lvis-api 依赖

**问题**: todd-ai 需要带 `boundary_utils` 模块的 `lvis-api`，PyPI 版本不包含此模块。

**解决方案**: 从 GitHub 源码安装：
```bash
pip install git+https://github.com/lvis-dataset/lvis-api.git@lvis_challenge_2021 --no-build-isolation
```

#### pytest-html 缺失

**问题**: Basilisk 编译需要 `pytest-html` 包。

**解决方案**: 编译 Basilisk 前先安装：
```bash
pip install pytest-html
```

## 数据准备

如果要使用全部数据复现论文：

```bash
git clone git@hf.co:datasets/MessianX/AEOS-dataset ./data
find ./data -type f -name '*.tar' -print0 | xargs -0 -n1 -I{} sh -c 'tar -xf "$1" -C "$(dirname "$1")"' _ {}
```

或者只下载 val_seen/val_unseen/test 数据集来评估模型：

```bash
# TODO: urls
# 假设已下载所需数据
find ./data -type f -name '*.tar' -print0 | xargs -0 -n1 -I{} sh -c 'tar -xf "$1" -C "$(dirname "$1")"' _ {}
```

## 使用步骤

### 1. 确认数据结构

正确的文件树应如下所示：

```
data/
├── trajectories.1/
│   ├── test/
│   ├── train/
│   │   ├── 00/         # 包含 pth 和 json 文件
│   │   ├── 01/
│   │   ├── ...
│   ├── val_seen/
│   └── val_unseen/
├── trajectories.2/
├── trajectories.3/
├── annotations/
│   ├── test.json
│   ├── train.json
│   ├── val_seen.json
│   └── val_unseen.json
├── constellations/
│   ├── test/
│   ├── train/
│   ├── val_seen/
│   └── val_unseen/
├── orbits/
├── satellites/
└── tasksets/
```

### 2. 训练模型

#### 基础训练命令

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=:${PYTHONPATH} auto_torchrun -m constellation.new_transformers.train test constellation/new_transformers/config.py
```

训练将持续到 200000 次迭代。

#### 训练加速优化（针对 RTX 4090）

本项目已实现以下加速技术：

**1. 分布式数据并行（DDP）**
- 配置：`strategy = dict(type='DDPStrategy')`
- 自动使用多 GPU 训练（通过 `auto_torchrun` 检测）

**2. 数据加载优化**
- `PrefetchDataLoader`：预取下一批数据到 GPU
- `num_workers=2`：2 个进程并行加载数据
- `DistributedSampler`：多 GPU 数据分片

**3. 混合精度训练（可选）**
```bash
# 启用 AMP（Automatic Mixed Precision）
CUDA_VISIBLE_DEVICES=0 auto_torchrun -m constellation.new_transformers.train test \
    constellation/new_transformers/config.py --autocast
```
- 使用 FP16 计算，减少显存占用 ~40%
- RTX 4090 的 Tensor Core 加速 FP16 运算
- 预期加速：1.5-2x

**4. 针对 RTX 4090 的优化建议**

当前配置（`config.py`）：
```python
batch_size = 24          # 每个样本包含多个时间步
num_workers = 2          # 数据加载进程数
lr = 1e-4               # 学习率
optimizer = AdamW       # 优化器
```

**推荐优化**：
```bash
# 方案 1：增大 batch size（利用 4090 的 24GB 显存）
--override trainer.dataset.batch_size=48

# 方案 2：启用混合精度 + 更大 batch
--autocast --override trainer.dataset.batch_size=64

# 方案 3：增加数据加载速度
--override trainer.dataloader.num_workers=4
```

**完整优化命令**：
```bash
CUDA_VISIBLE_DEVICES=0 auto_torchrun -m constellation.new_transformers.train test \
    constellation/new_transformers/config.py \
    --autocast \
    --override trainer.dataset.batch_size=48 \
    --override trainer.dataloader.num_workers=4
```

**性能对比**（RTX 4090 预估）：
| 配置 | Batch Size | 混合精度 | 显存占用 | 训练速度 |
|------|-----------|---------|---------|---------|
| 默认 | 24 | 否 | ~18GB | 1.0x |
| 优化 | 48 | 是 | ~16GB | 1.8x |

**注意事项**：
- 混合精度可能影响数值稳定性，建议先在小数据集上验证
- 如果出现 OOM（显存不足），减小 `batch_size`
- `num_workers` 过大会占用 CPU 和内存，建议不超过 4

### 3. 评估模型

#### Val Seen 评估
```bash
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 RANK=0 python -m constellation.rl.eval_all \
    val_seen \
    constellation/rl/config_eval.py \
    --load-model-from 'work_dirs/test/checkpoints/iter_200000/model.pth'
```

#### Val Unseen 评估
```bash
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 RANK=0 python -m constellation.rl.eval_all \
    val_unseen \
    constellation/rl/config_eval_val_unseen.py \
    --load-model-from 'work_dirs/test/checkpoints/iter_200000/model.pth'
```

#### Test 集评估
```bash
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 RANK=0 python -m constellation.rl.eval_all \
    test_set \
    constellation/rl/config_eval_test.py \
    --load-model-from 'work_dirs/test/checkpoints/iter_200000/model.pth'
```

### 4. Baseline 评估

运行 baseline 算法（OptimalAlgorithm）进行对比：

```bash
# 单进程
PYTHONPATH=. python tools/test_baseline.py baseline_optimal 0

# 多进程（4个worker）
PYTHONPATH=. python tools/test_baseline.py baseline_optimal 4
```

## 评估结果（全量数据集）

### 数据集规模

| 数据集 | 场景数 | 说明 |
|-------|--------|------|
| Train | 16,159 | 全量训练集 |
| Val Seen | 64 | 使用训练集中出现过的卫星 |
| Val Unseen | 64 | 使用训练集中未出现的新卫星 |
| Test | 64 | 真实场景（从网络获取） |

### Transformer 模型 (iter_200000)

| 数据集 | CR (%) | PCR (%) | TAT (h) |
|-------|--------|---------|---------|
| Val Seen | 36.77 | 40.16 | 0.16 |
| Val Unseen | 41.70 | 44.98 | 0.15 |
| Test | 28.53 | 31.38 | 0.18 |

### Baseline (OptimalAlgorithm - 贪心算法)

| 数据集 | CR (%) | PCR (%) | TAT (h) |
|-------|--------|---------|---------|
| Test | 31.75 | 36.17 | 0.17 |

**注意**: OptimalAlgorithm 是代码仓库提供的贪心 Baseline，不在论文对比表中。

## 数据集分布差异分析：为什么 Test 集性能显著下降？

### 关键发现：Sim-to-Real Gap（仿真到现实的鸿沟）

Test 集与 Train/Val 集存在本质性的**数据分布偏移（Out-of-Distribution, OOD）**，这是导致模型性能下降的根本原因。

### 1. Train 和 Val 的"温室环境"（人造均匀分布）

在构建训练集和验证集时，卫星的物理参数（质量、转动惯量、电池容量等）都是**在预设的合理范围内均匀随机采样**生成的：

- **Train & Val Seen**: 使用相同批次的人造卫星（2,907 颗）
- **Val Unseen**: 使用新的 500 颗卫星，但仍从相同的"均匀分布"中采样
- **特点**: 参数分布均衡，极端情况少见

这就像在驾校练车：虽然每次换不同场地，但开的都是标准教练车，各个零件性能都符合预期。

### 2. Test 集的"残酷现实"（真实野生数据）

Test 集包含 500 颗卫星，具备**从网络上获取的真实属性**（论文 3.3 节）：

- **数据来源**: N2YO 和 Gunter's Space Page（真实航天数据库）
- **特点**: 真实卫星可能有极端的参数组合
  - 例如：巨大的太阳能板 + 小功率动量轮 → 转身极其缓慢
  - 这种极端组合在训练时的"均匀分布"中几乎不会出现

### 3. 为什么 AEOS-Former 在 Test 集上失灵？

AEOS-Former 的核心创新是**内部约束模块（MLP）**，负责预测物理可行性：
- 预测卫星转身所需时间 $\hat{t}$
- 预测任务可行性得分 $\hat{s}$
- 生成 Mask 掩码过滤不可行的任务

**问题**: 这个 MLP 是用人造均匀分布的数据训练的。当遇到真实世界的极端卫星参数时：
- 错误估计转身时间
- 高估任务可行性
- 给出错误的 Mask，导致 Transformer 制定了物理上不可行的调度计划
- 最终任务失败，完成率从 41.70% 暴跌至 28.53%

### 4. 客观评价：虽然下降，但仍是"全场最佳"

对比论文 Table 2 中的 Test 集结果：

| 方法 | CR (%) | PCR (%) | PC (Wh) | TAT (h) | CS ↓ |
|------|--------|---------|---------|---------|------|
| REDA (多智能体 RL) | 3.65 | 4.27 | - | 0.73 | - |
| MSCPO-SHCS (启发式) | 19.44 | 24.00 | 149.20 | 6.23 | - |
| **AEOS-Former** | **19.25** | **22.31** | **40.91** | **5.67** | **6.28** |

**关键优势**：
- CR 与最强 Baseline 持平（19.25% vs 19.44%）
- 功耗极低（40.91 Wh，全场最低）
- 综合得分 CS = 6.28（越低越好），远超第二名

虽然遭遇真实数据的降维打击，但 AEOS-Former 学到了底层物理约束机制，"保底能力"完爆纯数学搜索和普通强化学习算法。

### 5. 数据集重叠分析

| 对比项 | 重叠场景数 | 说明 |
|--------|-----------|------|
| Val Seen ∩ Train | 47/64 | 73% 的场景在训练集中出现过 |
| Val Unseen ∩ Train | 51/64 | 80% 的场景在训练集中出现过 |
| Test ∩ Train | 48/64 | 75% 的场景在训练集中出现过 |

**注意**: 虽然场景 ID 有重叠，但 Test 集使用的是真实卫星参数，与 Train/Val 的人造参数分布完全不同。

## 项目结构

```
.
├── constellation/          # 核心源代码
│   ├── algorithms/        # 调度算法（Optimal, Replay 等）
│   ├── callbacks/         # 训练回调函数
│   ├── data/             # 数据结构（Constellation, Task 等）
│   ├── environments/     # Basilisk 仿真环境
│   ├── evaluators/       # 评估指标（CR, TAT, Power）
│   ├── new_transformers/ # Transformer 模型实现
│   ├── rl/              # RL 训练（PPO）和评估
│   └── task_managers.py  # 任务调度管理
├── data/                 # 数据集（从 HuggingFace 下载）
│   ├── annotations/      # 数据集划分文件（train/val_seen/val_unseen/test.json，包含场景 ID 和 epochs）
│   ├── constellations/   # 卫星星座配置（每个场景的卫星组合，JSON 格式）
│   ├── tasksets/         # 观测任务集（地面目标位置、优先级、时间窗口）
│   ├── satellites/       # 卫星规格库（物理参数：质量、转动惯量、电池容量、传感器类型）
│   ├── orbits/           # 轨道参数库（开普勒参数：半长轴、偏心率、倾角等）
│   ├── trajectories.{1,2,3}/ # 预计算轨迹（.pth 和 .json 文件，按 split 划分）
│   ├── model/            # 预训练模型检查点
│   └── statistics_new.pth # 数据集统计信息（用于归一化）
├── tools/               # 工具脚本
│   └── test_baseline.py # Baseline 算法评估
├── scripts/             # 辅助脚本
│   ├── analyze_results.py    # 结果分析（CS 计算）
│   ├── summarize_eval.py     # 评估结果汇总
│   └── config_eval_val_unseen.py  # Val Unseen 评估配置
├── work_dirs/           # 训练输出（不在 git 中）
│   ├── test/           # 主训练运行
│   │   └── checkpoints/ # 模型检查点（iter_100000, iter_200000）
│   ├── rl_eval_*/      # 评估结果
│   └── test_baseline/  # Baseline 评估结果
├── archive_logs/        # 历史评估日志（包含 baseline_eval.log）
├── third_party/         # 外部依赖
│   └── basilisk/       # 航天器仿真框架
├── .flake8             # Python 代码风格检查配置
├── .todd_version       # Todd 框架版本锁定
└── EVALUATION_RESULTS.md # 评估结果汇总
```

## 配置文件说明

- **`.flake8`**: Python 代码风格检查配置，定义忽略的错误类型和复杂度限制
- **`.todd_version`**: 锁定 todd 框架的 git commit 版本，确保环境一致性
- **`archive_logs/`**: 存储历史评估日志，包括 Basilisk 编译日志和各次评估运行日志
- **`scripts/`**: 辅助工具脚本，用于结果分析和评估汇总

## Data 目录结构说明

`data/` 目录包含所有数据集文件，从 HuggingFace 下载（`git clone git@hf.co:datasets/MessianX/AEOS-dataset ./data`）。

### 数据生成流程

虽然数据从 HuggingFace 下载，但项目提供了完整的数据生成工具（`tools/` 目录）：

1. **生成卫星** → `generate_satellites.py`
2. **生成任务** → `generate_tasks.py`
3. **生成星座和任务集** → `generate_constellations_and_tasksets.py`
4. **生成轨迹** → `generate_trajectories.py`（使用 Basilisk 仿真）
5. **计算统计信息** → `compute_dataset_statistics.py`

### 数据集扩充指南

如果需要扩充数据集以适应特定应用场景（如基站观测、灾害监测等），按以下优先级修改：

#### 1. 静态基础库（通常不需要修改）

- **`satellites/`**（卫星规格）：物理参数库（质量、转动惯量、传感器类型等）
- **`orbits/`**（轨道参数）：开普勒轨道参数库（半长轴、偏心率、倾角等）

**说明**：这些是"字典"，现有参数已足够组合出多样化场景。除非引入全新传感器或特殊轨道，否则无需修改。

#### 2. 核心扩充区（重点增加）

**`tasksets/`**（观测任务集）⭐ **最重要**
- 添加贴近真实业务的地面目标
- 示例场景：
  - 基站观测：模拟真实基站分布位置
  - 灾害监测：设置高优先级紧急任务
  - 海洋监测：大范围低频观测任务
- 设置挑战性的时间窗口和优先级

**`constellations/`**（卫星星座配置）
- 重新组合 `satellites/` 和 `orbits/` 中的元素
- 测试不同规模：10 颗、20 颗、50 颗卫星
- 验证协同调度能力

#### 3. 重新生成衍生数据（必须更新）

扩充上述文件后，**必须**重新生成以下数据：

**`trajectories.{1,2,3}/`**
- 使用 Basilisk 仿真新的任务和星座配置
- 运行 `tools/generate_trajectories.py`

**`annotations/`**
- 重新划分训练集和测试集
- 更新 JSON 文件中的场景 ID 和 epochs

**`statistics_new.pth`** ⚠️ **关键**
- 重新计算全局均值和标准差
- 运行 `tools/compute_dataset_statistics.py`
- **如果不更新**：模型归一化错误 → 损失函数爆炸 → 训练失败

#### 扩充流程示例

```bash
# 1. 生成新的任务集（例如：1000 个基站观测任务）
python tools/generate_tasks.py --scenario base_station --num 1000

# 2. 生成新的星座配置（例如：50 个场景，每个 20 颗卫星）
python tools/generate_constellations_and_tasksets.py --num-scenes 50 --num-sats 20

# 3. 使用 Basilisk 仿真生成轨迹（耗时最长）
WORLD_SIZE=4 RANK=0 python tools/generate_trajectories.py

# 4. 重新计算统计信息（必须！）
python tools/compute_dataset_statistics.py

# 5. 更新 annotations（重新划分数据集）
python tools/generate_annotations.py
```

#### 注意事项

- **轨迹生成耗时**：Basilisk 仿真很慢，建议使用多 GPU 并行
- **统计信息更新**：每次修改数据集后必须重新计算 `statistics_new.pth`
- **数据一致性**：确保 `annotations/` 中的场景 ID 在 `constellations/` 和 `tasksets/` 中都存在


## Tools 目录说明

`tools/` 目录包含数据生成、评估和分析工具：

### 数据生成工具

| 文件 | 功能 | 用途 |
|------|------|------|
| **generate_data.py** | 主数据生成脚本 | 定义卫星物理参数的采样范围（质量、转动惯量、电池等） |
| **generate_satellites.py** | 生成卫星规格 | 创建单个卫星的物理参数文件 |
| **generate_tasks.py** | 生成观测任务 | 使用开普勒轨道公式生成地面目标位置 |
| **generate_constellations_and_tasksets.py** | 生成星座和任务集 | 随机组合卫星形成星座，分配观测任务 |
| **generate_trajectories.py** | 生成轨迹数据 | 使用 Basilisk 仿真器运行 OptimalAlgorithm，生成卫星轨迹 |
| **generate_tiny_annotations.py** | 生成 Tiny 数据集 | 从完整数据集中提取子集（用于快速验证） |
| **compute_dataset_statistics.py** | 计算数据集统计 | 计算训练集的均值和标准差，用于归一化 |

### 评估和分析工具

| 文件 | 功能 | 用途 |
|------|------|------|
| **test_baseline.py** | Baseline 评估 | 运行 OptimalAlgorithm（贪心算法）进行评估 |
| **evaluate_baseline.py** | TabuSearch 评估 | 运行 TabuOptimalAlgorithm（禁忌搜索）进行评估 |
| **compare_trajectory_cr.py** | 轨迹对比 | 对比不同轨迹的完成率（CR） |
| **generate_tabu_lists.py** | 生成禁忌列表 | 为 TabuSearch 算法生成禁忌任务列表 |
| **merge_tabu.py** | 合并禁忌列表 | 合并多个禁忌列表文件 |
| **patch_constellations.py** | 修补星座配置 | 修复或更新星座配置文件 |
| **generate_mrp_taskset.py** | 生成 MRP 任务集 | 生成特定类型的任务集（MRP: Modified Rodrigues Parameters） |

### 使用示例

```bash
# 评估 Baseline 算法（单进程）
PYTHONPATH=. python tools/test_baseline.py baseline_optimal 0

# 评估 Baseline 算法（4 个并行进程）
PYTHONPATH=. python tools/test_baseline.py baseline_optimal 4

# 计算数据集统计信息
python tools/compute_dataset_statistics.py

# 生成 Tiny 数据集（前 100 个场景）
python tools/generate_tiny_annotations.py --split train --n 100
```

### 数据生成依赖关系

```
generate_satellites.py
        ↓
generate_tasks.py
        ↓
generate_constellations_and_tasksets.py
        ↓
generate_trajectories.py (需要 Basilisk)
        ↓
compute_dataset_statistics.py
```

**注意**: 完整的数据生成需要 Basilisk 仿真器，非常耗时。建议直接从 HuggingFace 下载预生成的数据集。

## 数据维度详解

### 卫星特征 (56 维)

完整的卫星状态向量包含 48 维静态特征 + 8 维动态特征：

#### 静态特征 (48 维)

| 索引 | 特征 | 维度 | 物理意义 |
|------|------|------|----------|
| 0-8 | 惯性张量 | 9 | 3×3 矩阵 [Ixx,Ixy,Ixz,Iyx,Iyy,Iyz,Izx,Izy,Izz] (kg·m²) |
| 9 | 质量 | 1 | 卫星质量 (kg) |
| 10-12 | 质心 | 3 | 质心位置 [x,y,z] (m) |
| 13-17 | 轨道参数 | 5 | [偏心率, 半长轴, 倾角, 升交点赤经, 近地点幅角] |
| 18-22 | 太阳能板 | 5 | [方向向量×3, 面积(m²), 效率] |
| 23-24 | 传感器 | 2 | [半视场角(度), 功率(W)] |
| 25 | 电池容量 | 1 | 电池容量 (J) |
| 26-43 | 反作用轮静态 | 18 | 3个轮×[方向×3, 最大角动量(N·m·s), 功率(W), 效率] |
| 44-47 | MRP控制器 | 4 | [比例增益k, 积分增益ki, 微分增益p, 积分限制] |

**关键索引（用于物理建模）**：
- 惯性矩对角元素：`[0, 4, 8]` (Ixx, Iyy, Izz)
- 反作用轮最大角动量：`[29, 35, 41]` (3个轮)

#### 动态特征 (8 维)

| 索引 | 特征 | 维度 | 物理意义 |
|------|------|------|----------|
| 48 | 电量百分比 | 1 | 当前电量 [0,1] |
| 49-51 | 反作用轮转速 | 3 | 3个轮的转速 (RPM) |
| 52 | 真近点角 | 1 | 轨道位置参数 (度) |
| 53-55 | MRP姿态 | 3 | 修正罗德里格斯参数 [σ₁, σ₂, σ₃] |

**MRP 姿态物理转换**：
```python
# MRP 范数到真实旋转角度
mrp_norm = torch.norm(mrp, p=2, dim=-1)
rotation_angle = 4.0 * torch.atan(mrp_norm)  # 弧度
```

### 任务特征 (6 维)

| 索引 | 特征 | 物理意义 |
|------|------|----------|
| 0 | release_time - t | 剩余发布时间（动态） |
| 1 | due_time - t | 剩余截止时间（动态） |
| 2 | duration | 观测持续时间（静态） |
| 3 | latitude | 目标纬度（静态） |
| 4 | longitude | 目标经度（静态） |
| 5 | progress | 任务完成进度（动态） |
