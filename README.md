# Constellation 项目说明

本仓库是论文 **“Towards Realistic Earth-Observation Constellation Scheduling: Benchmark and Methodology”** 的官方实现，用于研究真实约束下的地球观测星座调度问题。

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-purple)](https://neurips.cc/virtual/2025/loc/san-diego/poster/116515)
[![arXiv](https://img.shields.io/badge/arXiv-2510.26297-b31b1b.svg)](https://arxiv.org/abs/2510.26297)

## 文档导航

当前项目文档按用途分工如下：

| 文档 | 用途 |
|---|---|
| `README.md` | 项目入口说明，覆盖环境、数据、训练、评估和当前校准口径。 |
| `TODO.md` | 当前目标和短期任务板，只保留下一阶段工作方向。 |
| `docs/实验复现报告.md` | 当前复现实验结论、论文对齐关系和汇报口径。 |
| `docs/aeos_former_shape_flow.md` | AEOS-Former 的输入输出、张量形状和模块流图。 |
| `docs/AEOSFormer_Encoder_解析.md` | AEOS、AEOS-Bench、AEOS-Former 架构和指标体系解释稿。 |
| `docs/constellation_code_structure.md` | `constellation/` 代码结构、训练流程和评估调用链。 |
| `docs/new_transformers_dataset_model.md` | `Dataset`、`JointDataset`、`Model`、`JointModel` 的职责和数据流。 |
| `docs/paper_references_map.md` | AEOS-Bench / AEOS-Former 论文参考文献脉络。 |

## 环境

本项目在当前机器上应使用已有的 `aeos` conda 环境。运行项目命令时优先使用：

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" PYTHONPATH=":${PYTHONPATH:-}" <command>
```

或直接使用：

```bash
/home/hy/miniconda3/envs/aeos/bin/python
```

不要依赖裸 `python` 或系统默认环境。

原始安装命令如下，仅在需要重新配置环境时参考：

```bash
sudo apt install ffmpeg libpq-dev
bash setup.sh
```

## Slurm 集群使用规则

当前实验室集群使用 `Slurm` 管理计算资源，使用 `NFS` 暴露统一数据路径。以后在本仓库中运行命令时，先按任务类型判断是否需要经过 Slurm。

### 1. 日常文件和代码操作

下面这些命令不属于训练任务，通常不需要申请 Slurm 资源，和以前一样直接在当前终端运行：

```bash
pwd
ls
rg "keyword"
sed -n '1,120p' README.md
git diff
tail -f work_dirs/eval_logs/xxx.log
```

这类操作只是在登录节点上查看文件、修改代码、检查日志或做很轻量的文本搜索，不占用 GPU，也不会长时间占用大量 CPU。

### 2. GPU 训练、GPU 评估和长时间仿真

只要任务会占用 GPU，或者会长时间运行，例如模型训练、正式评估、模型滚动生成、大规模 Basilisk 仿真，就应该通过 Slurm 提交。

短时间调试用 `srun`，例如只确认环境、路径和 GPU 是否能启动：

```bash
srun -p groupA --nodes=1 --gres=gpu:1 --cpus-per-task=8 --mem=32G bash
```

进入 `srun` 分配的 shell 后，再运行本项目命令：

```bash
cd /home/hy/data/newaeosbench
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" PYTHONPATH=":${PYTHONPATH:-}" \
python -m constellation.new_transformers.train <run_name> <config.py>
```

正式训练或正式评估用 `sbatch`，不要直接在登录节点或计算节点上裸跑长任务：

```bash
sbatch -p groupA job.sh
```

如果确实需要固定节点，再加 `-w server-11` 这类节点限制：

```bash
sbatch -p groupA -w server-11 job.sh
```

固定节点会减少调度选择，可能增加排队时间；只有数据、调试或硬件原因明确时才使用。

### 3. 不用 GPU 但长时间占 CPU 的任务

轻量 CPU 命令可以直接运行，例如小规模语法检查、小测试或查看配置：

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" PYTHONPATH=":${PYTHONPATH:-}" \
python -m py_compile constellation/new_transformers/train.py
```

如果是长时间 CPU 任务，例如大规模数据生成、轨迹处理、批量统计、96 并行评估或长时间仿真，也应该用 `sbatch`，但不要申请 GPU：

```bash
#!/bin/bash
#SBATCH --job-name=aeos_cpu_task
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=work_dirs/eval_logs/%x-%j.out
#SBATCH --error=work_dirs/eval_logs/%x-%j.err

cd /home/hy/data/newaeosbench

env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" PYTHONPATH=":${PYTHONPATH:-}" \
python your_cpu_script.py
```

注意：CPU 任务脚本里不要写 `--gres=gpu:1`，否则会额外占用 GPU 资源。

### 4. 本集群常用分区和资源概况

当前说明中的分区规则如下：

```text
groupA: server-9, server-10, server-11, server-12
groupB: server-9, server-10
```

当前 `sinfo` 可见的资源摘要如下，具体可用状态以提交时 `sinfo` 输出为准：

```text
server-9   groupA/groupB  128 CPU  510000 MB  gpu:4
server-10  groupA/groupB  144 CPU  510000 MB  gpu:4
server-11  groupA         152 CPU  510000 MB  gpu:4
server-12  groupA         152 CPU  510000 MB  gpu:4
```

在 `server-10` 本机检查到的硬件是：

```text
CPU: Intel Xeon Platinum 8352V, 144 logical CPUs
GPU: 4 x NVIDIA GeForce RTX 4090, each 24564 MiB
Memory: about 503 GiB
```

其它节点的详细 CPU/GPU 型号需要通过 Slurm 任务或管理员权限在对应节点上查询。

### 5. 权限检查和常用命令

检查自己属于哪些组：

```bash
id
groups
```

查看节点和队列：

```bash
sinfo
squeue -u $USER
```

查看公平份额和优先级：

```bash
sshare -u $USER -l
sprio -u $USER -l
```

查看等待原因：

```bash
squeue -u $USER -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R"
```

如果 `srun` 或 `sbatch` 报错：

```text
User's group not permitted to use this partition
```

说明当前账号还没有被 Slurm 允许使用该分区。此时不是代码问题，也不是 `aeos` 环境问题，需要联系管理员检查账号是否已经加入对应 Linux 用户组、Slurm account/association，或是否允许使用 `groupA`/`groupB` 分区。

给管理员的最小说明可以写成：

```text
我的账号运行 sinfo 能看到 groupA/groupB 节点，
但是 srun -p groupA 会报：
User's group not permitted to use this partition

请帮我检查该账号是否已经加入 groupA 对应的用户组或 Slurm association。
```

## AEOS-Former 架构速览

AEOS-Former 不是把时间当作序列轴的普通时间序列 Transformer。当前实现更适合理解为：

```text
任务序列 Encoder + 卫星序列 Decoder + 卫星-任务约束配对模块
```

三个核心模块如下：

| 模块 | 序列或配对对象 | 作用 |
|---|---|---|
| Encoder | 任务序列，长度为当前候选任务数 `nt` | 编码任务发布时间、截止时间、持续时间、传感器类型、任务进度等信息，并用任务掩码屏蔽未发布、已过期或已完成任务。 |
| TimeModel / ICM | 每个卫星-任务二元组，形状近似为 `[batch, ns, nt]` | 预测卫星执行某任务的可行性和预计持续时间，作为软约束引导 Decoder 的交叉注意力。最终物理合法性仍由 Basilisk 硬约束验证。 |
| Decoder | 卫星序列，长度为当前卫星数 `ns` | 先让卫星之间通过自注意力交换状态，再以卫星为查询向量、任务为键/值向量做交叉注意力，输出每颗卫星选择空动作或某个任务的未归一化得分。 |

训练数据由 `Dataset` 或 `JointDataset` 从多个文件组合得到：

```text
data/annotations/*.json
data/trajectories.<epoch>/<split>/<id>.pth
data/constellations/<split>/<id>.json
data/tasksets/<split>/<id>.json
data/statistics_new.pth
        |
        v
`Batch` 或 `JointBatch`
        |
        v
`Model` 或 `JointModel`
```

其中：

- `Dataset + Model` 主要训练动作分配损失 `L_a`，更接近早期 CE-only 动作模型。
- `JointDataset + JointModel` 同时训练 `L_a`、可行性损失 `L_s` 和时间回归损失 `L_t`，是当前论文式联合训练主线。
- 当前论文式配置中三项权重均为 1，即：

```text
loss = L_s + L_t + L_a
```

更细的张量形状和数据流见 `docs/aeos_former_shape_flow.md` 与 `docs/new_transformers_dataset_model.md`。

## 数据目录

如果需要使用完整数据复现论文实验，可下载数据集到 `data/`：

```bash
git clone git@hf.co:datasets/MessianX/AEOS-dataset ./data
find ./data -type f -name '*.tar' -print0 | xargs -0 -n1 -I{} sh -c 'tar -xf "$1" -C "$(dirname "$1")"' _ {}
```

如果只需要评估自己的模型，可以只准备 `val_seen`、`val_unseen` 和 `test` 相关数据。

项目期望的数据结构大致如下：

```text
data/
├── annotations/
│   ├── train.json
│   ├── val_seen.json
│   ├── val_unseen.json
│   └── test.json
├── constellations/
│   ├── train/
│   ├── val_seen/
│   ├── val_unseen/
│   └── test/
├── tasksets/
│   ├── train/
│   ├── val_seen/
│   ├── val_unseen/
│   └── test/
├── trajectories.1/
│   ├── train/
│   ├── val_seen/
│   ├── val_unseen/
│   └── test/
├── trajectories.2/
├── trajectories.3/
├── orbits/
├── satellites/
└── statistics_new.pth
```

## 完整数据生成与训练数据组织流程

这一节说明从“生成卫星”到“模型训练样本”的完整逻辑链条。注意：当前正式复现实验通常直接使用已有 `data/`，不一定需要从零重新生成。

### 1. 生成筛选卫星用的 MRP 任务集

入口：

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" PYTHONPATH=":${PYTHONPATH:-}" \
python tools/generate_mrp_taskset.py
```

输出：

```text
data/tasksets/mrp.json
```

含义：

- 这不是最终训练场景的任务集。
- 它是筛选候选卫星时使用的共享 MRP 任务集。
- 后续 `tools/generate_satellites.py` 会用这个任务集测试候选单星是否足够好。

### 2. 生成并筛选候选卫星池

入口：

```bash
bash tools/generate_satellites.sh
```

内部主要调用：

```text
tools/generate_mrp_taskset.py
tools/generate_satellites.py
```

输出：

```text
data/satellites/train/*.json
data/satellites/val_seen -> data/satellites/train
data/satellites/val_unseen/*.json
data/satellites/test/*.json
```

逻辑：

```text
随机生成一颗候选卫星
        |
        v
把它放进一个单星 constellation
        |
        v
在 data/tasksets/mrp.json 上用 Basilisk + OptimalAlgorithm 仿真
        |
        v
计算完成率 CR
        |
        v
如果 CR 超过阈值，则保存到 data/satellites/<split>/
```

这里生成的是“候选卫星池”，不是最终训练场景。

### 3. 生成正式场景：constellation 和 taskset

入口：

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" PYTHONPATH=":${PYTHONPATH:-}" \
python tools/generate_constellations_and_tasksets.py
```

输出：

```text
data/constellations/<split>/<xx>/<id>.json
data/tasksets/<split>/<xx>/<id>.json
```

含义：

- `constellation` 是一个场景里的星座配置。
  - 包括这个场景有哪些卫星。
  - 包括每颗卫星的硬件参数、轨道参数、初始状态。
- `taskset` 是同一个场景里的任务集合。
  - 包括任务发布时间、截止时间、观测持续时间、地面坐标、传感器类型。

二者合起来才是一个静态仿真场景：

```text
静态场景 = constellation + taskset
```

当前生成流程已经加入任务点位可观测性筛选。`TaskSet.sample()` 仍然负责随机生成候选任务，但 `tools/generate_constellations_and_tasksets.py` 会在写入正式 `taskset` 前，用对应星座做快速物理几何判断，筛掉没有连续观测窗口的任务点位，并重新补齐任务数量。这里不启动完整 Basilisk 仿真，不检查姿态控制、能耗或传感器开关动态。

筛选标准是：候选任务在自己的 `release_time <= t <= due_time` 时间窗内，至少存在一段连续可观测时间，长度不小于该任务的 `duration`。这样可以避免大量“无论模型如何调度都无法完成”的随机点位进入正式训练和评估，从而让 `CR/PCR/WCR` 更接近模型调度能力本身。

### 4. 为什么不把 constellation 和 taskset 合并成一个文件

可以合并，但这个项目选择分开是合理的，因为二者语义不同：

```text
constellation = 卫星系统是什么
taskset       = 这次要观测什么任务
trajectory    = 这个场景里实际怎么调度
```

分开的好处：

- 同一个星座可以搭配不同任务集。
- 同一个任务集也可以用于不同星座实验。
- 同一个 `constellation + taskset` 可以生成多轮不同轨迹，例如 `trajectories.1`、`trajectories.2`、`trajectories.3`。
- Stage-2/Stage-3 只需要替换轨迹和标注文件，不需要复制一整份静态场景。
- `Dataset` 可以清楚地区分静态特征和动态轨迹特征。

因此训练样本不是一个单独文件，而是由多个文件组合得到：

```text
星座文件 + 任务集文件 + 轨迹文件 + 标注文件
```

### 5. 生成专家轨迹

入口：

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" PYTHONPATH=":${PYTHONPATH:-}" \
WORLD_SIZE=1 RANK=0 python tools/generate_trajectories.py
```

输出：

```text
data/trajectories/<split>/<xx>/<id>.pth
data/trajectories/<split>/<xx>/<id>.json
```

如果整理成带轮次的数据目录，训练代码通常读取：

```text
data/trajectories.1/
data/trajectories.2/
data/trajectories.3/
```

轨迹文件 `.pth` 保存的是动态调度过程，典型内容包括：

```text
constellation.sensor_enabled  每秒每颗卫星传感器是否开启
constellation.data            每秒每颗卫星动态状态
taskset.progress              每秒每个任务完成进度
actions.task_id               每秒每颗卫星选择的任务 ID
is_visible                    每秒每颗卫星对每个任务是否可见
```

指标文件 `.json` 保存该轨迹的评估结果，例如：

```text
CR
PCR
WCR
TAT
PC
```

### 6. Stage-2/Stage-3：模型滚动生成候选轨迹

论文式专家迭代不是只训练一次，而是会让当前模型在场景上滚动生成，得到新的候选轨迹。

入口工具：

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" PYTHONPATH=":${PYTHONPATH:-}" \
WORLD_SIZE=1 RANK=0 python tools/rollout_model_trajectories.py \
  <checkpoint.pth> \
  data/trajectories.N \
  --split train \
  --device cuda:0
```

输出：

```text
data/trajectories.N/train/<xx>/<id>.pth
data/trajectories.N/train/<xx>/<id>.json
```

这里的 `N` 表示第几轮轨迹池，例如：

```text
trajectories.1 = 初始专家轨迹
trajectories.2 = Stage-2 候选/筛选轨迹
trajectories.3 = Stage-3 候选/筛选轨迹
```

### 7. 用 tau_e 筛选轨迹并生成标注文件

入口工具：

```bash
env PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}" PYTHONPATH=":${PYTHONPATH:-}" \
python tools/build_tau_e_annotation.py \
  data/annotations/train.json \
  data/trajectories.N \
  data/annotations/train_filtered.json \
  --split train \
  --candidate-epoch N \
  --tau-e 4.5
```

作用：

```text
读取基础标注文件
        |
        v
读取 data/trajectories.N 中的候选轨迹指标
        |
        v
根据 CS <= tau_e 判断是否接受候选轨迹
        |
        v
输出新的标注文件
```

标注文件通常是：

```json
{
  "ids": [0, 1, 4, 5],
  "epochs": [3, 1, 1, 2]
}
```

含义：

```text
id=0 读取 data/trajectories.3/<split>/00/00000.pth
id=1 读取 data/trajectories.1/<split>/00/00001.pth
id=4 读取 data/trajectories.1/<split>/00/00004.pth
id=5 读取 data/trajectories.2/<split>/00/00005.pth
```

也就是说，标注文件决定“训练时用哪些场景，以及每个场景采用哪一轮轨迹”。

### 8. 进入 Dataset：把文件变成训练张量

入口代码：

```text
constellation/new_transformers/dataset.py
```

读取链条：

```text
data/annotations/*.json
        |
        v
拿到 id 和 epoch
        |
        v
data/trajectories.<epoch>/<split>/<id>.pth
data/constellations/<split>/<id>.json
data/tasksets/<split>/<id>.json
data/statistics_new.pth
        |
        v
Batch 或 JointBatch
```

`Dataset` 会做：

- 读取星座静态特征。
- 读取卫星动态轨迹特征。
- 读取任务静态特征。
- 读取任务进度。
- 生成任务有效性掩码。
- 读取专家动作 `actions.task_id` 作为监督标签。
- 对特征做归一化。

`JointDataset` 还会额外构造约束模块训练样本：

```text
constraint_time_steps
constraint_constellation_data
constraint_tasks_data
constraint_durations
```

这些用于训练：

```text
L_s：可行性损失
L_t：时间回归损失
```

### 9. 进入 Model：计算预测和损失

入口代码：

```text
constellation/new_transformers/model.py
```

普通动作模型：

```text
Dataset
        |
        v
Model
        |
        v
L_a = 动作分类损失
```

论文式联合模型：

```text
JointDataset
        |
        v
JointModel
        |
        v
L_a + L_s + L_t
```

其中：

```text
L_a = 任务分配动作损失
L_s = 可行性预测损失
L_t = 时间/持续时间预测损失
```

## 训练模型

旧版 README 中的默认训练命令是：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=:${PYTHONPATH} auto_torchrun -m constellation.new_transformers.train test constellation/new_transformers/config.py
```

这个命令会训练到 `200000` 次迭代，但它使用的是：

```text
constellation/new_transformers/config.py
```

需要注意：这个命令更接近旧动作模型训练，主要优化动作分类损失 `ce_loss`，不是严格论文式 `L_a + L_s + L_t` 联合训练。

当前论文式 200k 主线更应优先看：

```bash
bash scripts/run_paper_joint_200k_managed.sh
```

对应配置：

```text
constellation/new_transformers/config_paper_stage1_200k.py
constellation/new_transformers/config_paper_stage2_200k.py
constellation/new_transformers/config_paper_stage3_200k.py
```

训练调用链：

```text
scripts/run_paper_joint_200k_managed.sh
        |
        v
auto_torchrun -m constellation.new_transformers.train
        |
        v
constellation/new_transformers/train.py
        |
        v
config_paper_stage*_200k.py
        |
        v
JointDataset
        |
        v
JointModel
        |
        v
work_dirs/paper_joint_stage*_200k/checkpoints/
```

## Stage-2 专家迭代

Stage-2 不是在同一个固定标注文件上普通多训练几轮，而是仿真驱动的专家迭代：

```text
当前模型滚动生成
        |
        v
生成候选轨迹
        |
        v
Basilisk 仿真评估轨迹
        |
        v
保留 CS <= tau_e 的高质量轨迹
        |
        v
合并进训练标注文件
        |
        v
继续训练模型
```

相关工具：

- `tools/rollout_model_trajectories.py`
- `tools/build_tau_e_annotation.py`
- `tools/wrap_time_model_checkpoint.py`

`CS` 是整条轨迹级别的评估分数，不是每个训练步的直接损失。它主要用于 Stage-2/Stage-3 的轨迹筛选。

## 评估模型

基本评估命令：

```bash
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 RANK=0 python -m constellation.rl.eval_all \
    work_dir_name \
    constellation/rl/config_eval.py \
    --load-model-from 'work_dirs/test/checkpoints/iter_200000/model.pth'
```

正式复现评估建议优先使用 `scripts/` 中的托管脚本，例如：

```bash
bash scripts/run_stage3_200k_96core_eval_managed.sh
```

评估调用链：

```text
scripts/run_stage*_96core_eval_managed.sh
        |
        v
constellation/rl/eval_all.py
        |
        v
ControllerEnvironment
        |
        v
Controller + BasiliskEnvironment + TaskManager
        |
        v
Policy 加载 AEOS-Former 检查点
        |
        v
Evaluators 输出 CR/PCR/WCR/TAT/PC
        |
        v
tools/summarize_eval.py 统一汇总 CS_paper 与各单项指标
```

## 当前校准评估口径

当前论文复现统一使用下面的 paper-aligned 口径比较模型。

| 指标 | 当前口径 |
|---|---|
| CR | 数据划分内场景级 CR 平均，再乘 100 写入表格 |
| PCR | 数据划分内场景级 PCR 平均，再乘 100 写入表格 |
| WCR | 数据划分内场景级 WCR 平均，再乘 100 写入表格 |
| PC | 优先使用 `PC_Wh`；如果只有 `PC`，则使用 `PC_Wh = PC / 3600` |
| CS | 使用能够精确复现论文 Table 2 的 `CS_paper` |

统一公式：

```text
TAT_100s = TAT_s / 100
CS_paper = (0.6*CR + 0.2*PCR + 0.2*WCR)^(-1) + TAT_s/700 + PC_Wh/100
```

注意：

- CR、PCR、WCR 在公式中使用 0 到 1 的比例值，不使用百分数。
- `CS_no_TAT` 仅保留为历史辅助指标，不用于最终模型排序。
- 每次实验同时记录 `TAT_s`、`TAT_100s` 和全部单项指标。

其中 `TAT_s/700` 与 `TAT_100s/7` 完全等价。

## 场景生成可观测性筛选

### 修改内容

`tools/generate_constellations_and_tasksets.py` 已经加入任务点位可观测性筛选。原始脚本已备份为：

```text
tools/generate_constellations_and_tasksets.py.bak_20260622
```

新的生成逻辑不再把 `TaskSet.sample()` 随机生成的任务直接写入正式 `data/tasksets/`。流程变为：

```text
采样 constellation
        |
        v
过采样候选 taskset
        |
        v
按当前 constellation 扫描每个任务是否存在连续可观测窗口
        |
        v
只保留可观测任务，数量不足则继续补采样
        |
        v
重新编号任务 id，并写入 data/tasksets/
```

筛选标准是：

- 候选任务必须在自己的 `release_time <= t <= due_time` 时间窗内存在连续可观测片段。
- 连续可观测片段长度必须不小于该任务的 `duration`。
- 可观测性判断包含卫星与任务点的几何可达性、最大偏离星下点角约束和 `sensor_type` 匹配。
- 筛选不要求卫星初始姿态已经对准目标，也不把初始传感器开关状态当作任务无解原因。
- 当前实现使用快速两体轨道传播和地球自转近似做物理几何判断，不启动完整 Basilisk 仿真；Basilisk 仍用于后续真实评估和轨迹生成。

这次修改解决的问题是：旧版随机任务点位里可能存在大量物理上不可观测的任务。这样的任务无论模型如何调度都无法完成，会直接压低 `CR/PCR/WCR`，但这不属于模型预测能力差，而是场景生成阶段给了无解任务。筛选后，正式训练和评估中的失败更接近“模型是否调度成功”，而不是“任务本身是否无解”。

### 重新跑实验时的数据处理

如果要正式采用这版筛选逻辑，至少需要重新生成相关实验会读取的 `tasksets`。如果只拿旧检查点重新评估，则重建评估划分即可：

- `data/tasksets/val_seen`
- `data/tasksets/val_unseen`
- `data/tasksets/test`

如果要重新训练，再重建：

- `data/tasksets/train`

建议优先复用原来的：

- `data/satellites`
- `data/constellations`
- `data/orbits`

原因是这次主要修正任务点位是否可观测。如果同时重新生成卫星池或星座，后续结果变化就混入了星座分布变化，不利于判断筛选逻辑本身带来的影响。

如果只是拿已有检查点在筛选版 `tasksets` 上重新评估，不需要重新生成：

- `data/annotations`
- `data/trajectories.*`

原因是正式评估会根据 `data/annotations/{split}.json` 选择场景 id，再读取当前 `data/constellations` 和 `data/tasksets` 运行模型；旧检查点评估不依赖旧专家轨迹。

如果要重新训练模型，则需要重新生成：

- 基于筛选版 `tasksets` 的 `data/trajectories.*`
- 指向新轨迹池的训练 `data/annotations/*.json`

必须重新评估或重新整理的数据包括：

- 基于旧 `tasksets` 跑出的 `work_dirs/` 评估日志、指标汇总和报告表格

不一定需要重新生成的数据包括：

- 已有模型检查点。旧检查点可以直接拿来在筛选版 `tasksets` 上重新评估，用来判断“同一模型在过滤无解任务后 CR/PCR/WCR 是否变化”。
- 卫星池、星座和轨道数据。除非实验目标改成重新采样整个场景分布，否则不建议一起重建。

老数据不要直接删除。只做旧检查点重新评估时，建议至少归档旧 `tasksets` 和旧评估输出，例如：

```bash
mv data/tasksets data/tasksets_unfiltered_20260622
mv work_dirs/rl_eval_paper_joint_stage3_200k_96core_val_seen_managed \
  work_dirs/rl_eval_paper_joint_stage3_200k_96core_val_seen_unfiltered_20260622
```

如果要重新训练，才需要另外归档并重建训练标注文件和轨迹池，例如：

```bash
mv data/annotations data/annotations_unfiltered_20260622
mv data/trajectories.1 data/trajectories.1_unfiltered_20260622
```

如果要保留多轮专家迭代数据，也应分别归档 `data/trajectories.2`、`data/trajectories.3`、`data/trajectories.4` 等目录。归档后再运行新的场景生成脚本，让 `data/tasksets/` 重新生成筛选版任务集。

注意：当前生成脚本遇到已经存在的 `taskset` 文件会跳过，不会覆盖。因此如果要生成筛选版任务集，必须先把旧 `data/tasksets/` 移走或改名；不要在旧目录原地混合新旧任务集。

推荐实验顺序：

1. 归档旧 `data/tasksets` 和旧评估日志；如果要重新训练，再归档旧标注文件和旧轨迹。
2. 保留原 `data/satellites`、`data/constellations`、`data/orbits`。
3. 重新运行 `tools/generate_constellations_and_tasksets.py` 生成筛选版 `tasksets`。
4. 先用已有检查点在筛选版 test/val 划分上重新评估，得到新的 `CR/PCR/WCR/PC_Wh/CS_no_TAT`。
5. 如果要重新训练，再基于筛选版 `tasksets` 重新生成标注文件和轨迹，然后启动新一轮训练。

如果当前目标只是“旧检查点重新评估”，第 1 步只需要归档旧 `data/tasksets` 和旧评估输出；标注文件和轨迹可以保留不动。

报告和表格里必须明确区分：

- `unfiltered`（未过滤）：旧随机任务集口径，可能包含大量物理无解任务。
- `observable-filtered`（可观测性过滤）：新筛选任务集口径，任务点位至少存在连续可观测机会。

这两个口径下的 `CR/PCR/WCR` 不能直接混写到同一张结论表里，除非表格明确标注了场景生成口径。
