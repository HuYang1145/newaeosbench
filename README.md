# Constellation 项目说明

本仓库是论文 **“Towards Realistic Earth-Observation Constellation
Scheduling: Benchmark and Methodology”** 的官方实现，用于研究带真实物理约束的
地球观测星座调度。

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-purple)](https://neurips.cc/virtual/2025/loc/san-diego/poster/116515)
[![arXiv](https://img.shields.io/badge/arXiv-2510.26297-b31b1b.svg)](https://arxiv.org/abs/2510.26297)

## 文档导航

| 文档 | 用途 |
|---|---|
| `README.md` | 环境、数据、训练和评估入口 |
| `TODO.md` | 当前唯一主线和短期实验任务 |
| `改进日志.md` | 已完成改进、负结果和下一步依据 |
| `docs/实验复现报告.md` | 论文对齐结果和汇报口径 |
| `docs/aeos_former_shape_flow.md` | AEOS-Former 张量与模块流 |
| `docs/constellation_code_structure.md` | 代码结构和调用链 |
| `docs/new_transformers_dataset_model.md` | Dataset、Model 和损失关系 |

## 环境

只使用已有的 `aeos` conda 环境：

```bash
export PATH="/home/hy/miniconda3/envs/aeos/bin:${PATH}"
export PYTHONPATH="/home/hy/data/newaeosbench:${PYTHONPATH:-}"
```

Python 可执行文件：

```text
/home/hy/miniconda3/envs/aeos/bin/python
```

不要依赖裸 `python`、系统 Python，也不要擅自重建环境。

## 计算资源

当前机器属于 Slurm 集群。文件查看、代码修改和小测试可直接运行；GPU 训练、正式
评估、大规模 Basilisk 仿真和长时间 CPU 任务应通过 `srun` 或 `sbatch` 申请资源。

短时间调试：

```bash
srun -p groupA --nodes=1 --gres=gpu:1 --cpus-per-task=8 --mem=32G bash
```

正式任务：

```bash
sbatch -p groupA job.sh
```

进入已分配的计算节点后，可以再使用 `tmux` 托管长任务。包装脚本、Slurm job、
日志和输出路径必须记录在 `TODO.md`。

## 模型概览

AEOS-Former 可以简化理解为：

```text
任务 Encoder + 卫星 Decoder + TimeModel/ICM
```

| 模块 | 作用 |
|---|---|
| Encoder | 编码当前候选任务及其时间窗、持续时间、类型和进度 |
| Decoder | 编码卫星状态并输出每颗卫星的任务 logits |
| TimeModel / ICM | 预测卫星—任务可行性和预计持续时间，作为神经网络约束 |

论文式联合模型使用：

```text
loss = L_a + L_s + L_t
```

- `L_a`：任务分配动作损失。
- `L_s`：可行性预测损失。
- `L_t`：时间或持续时间回归损失。

模型负责近似调度规律，不能在训练 batch 或推理循环中为每个候选调用完整 Basilisk。
Basilisk 只用于离线轨迹、监督标签和正式评估。

## 数据

完整数据可下载到 `data/`：

```bash
git clone git@hf.co:datasets/MessianX/AEOS-dataset ./data
find ./data -type f -name '*.tar' -print0 | \
  xargs -0 -n1 -I{} sh -c 'tar -xf "$1" -C "$(dirname "$1")"' _ {}
```

主要结构：

```text
data/
├── annotations/       # id 与 trajectory epoch 路由
├── constellations/    # 每个场景的星座配置
├── tasksets/          # 每个场景的地面任务
├── trajectories.N/    # 动作、进度、可见性和动态状态
├── satellites/        # 候选卫星池
├── orbits/
└── statistics_new.pth
```

一个训练样本由 annotation 指定并组合读取：

```text
annotation
  -> trajectories.<epoch>/<split>/<id>.pth
  -> constellations/<split>/<id>.json
  -> tasksets/<split>/<id>.json
  -> Batch / JointBatch
```

主要数据工具：

| 工具 | 作用 |
|---|---|
| `tools/generate_mrp_taskset.py` | 生成卫星筛选用 MRP 任务集 |
| `tools/generate_satellites.sh` | 生成候选卫星池 |
| `tools/generate_constellations_and_tasksets.py` | 采样星座和随机 taskset |
| `tools/generate_trajectories.py` | 生成专家轨迹 |
| `tools/rollout_model_trajectories.py` | 用当前模型生成候选轨迹 |
| `tools/build_tau_e_annotation.py` | 按 `tau_e` 选择轨迹并生成 annotation |

当前 `tools/generate_constellations_and_tasksets.py` 直接使用 `TaskSet.sample()`，没有
启用任务点可观测性过滤。历史过滤实验只保留为诊断材料，不属于当前模型改进主线，
也不能与正式未过滤 benchmark 混写。

当前 `data/satellites/val_seen` 仍指向旧机器上的绝对路径。已有场景评估不一定受
影响，但重新生成 Val Seen 前必须先修复该符号链接，详见 `改进日志.md`。

## 训练

当前论文式三阶段 200k 联合训练入口：

```bash
bash scripts/run_paper_joint_200k_managed.sh
```

对应配置：

```text
constellation/new_transformers/config_paper_stage1_200k.py
constellation/new_transformers/config_paper_stage2_200k.py
constellation/new_transformers/config_paper_stage3_200k.py
```

主要调用链：

```text
config -> JointDataset -> JointModel -> L_a + L_s + L_t -> checkpoint
```

Stage2/Stage3 会使用模型 rollout、Basilisk 评估和 `tau_e` annotation 继续训练。
现有轨迹和 annotation 已确认是有效实验资产，不因 TAT 公式修正而重建。

## 评估

单次调试命令：

```bash
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 RANK=0 \
/home/hy/miniconda3/envs/aeos/bin/python -m constellation.rl.eval_all \
  <run_name> constellation/rl/config_eval.py \
  --load-model-from <checkpoint.pth>
```

在已分配计算节点内，正式 Stage3-200k 评估可参考：

```bash
bash scripts/run_stage3_200k_96core_eval_managed.sh
```

正式评估链路：

```text
Policy -> Controller -> BasiliskEnvironment -> TaskManager -> Evaluators
```

论文划分为 train 16,218 条轨迹，Val Seen、Val Unseen、Test 各 64 场。正式复现
优先使用论文的 96 并行环境；资源不足时必须记录实际 `world_size`。

## 统一指标

最终目标是降低 paper-aligned `CS_paper`，同时完整报告
`CR/PCR/WCR/TAT_s/PC_Wh`：

```text
Q = 0.6*CR + 0.2*PCR + 0.2*WCR
TAT_100s = TAT_s / 100
CS_paper = Q^(-1) + TAT_s/700 + PC_Wh/100
```

- `CR/PCR/WCR` 在公式中使用 0 到 1 的比例值。
- `PC_Wh = PC / 3600`。
- `CS_paper` 越低越好。
- `CS_no_TAT` 仅保留为历史辅助指标。

## 当前改进主线

Stage3-200k 在 Val Seen/Unseen 共 128 场的诊断显示，约 `47.12%` 的有效卫星选择
属于重复冗余，约 `85.99%` 的重复事件没有带来任务进度。当前主线是在现有
Transformer 后增加轻量卫星—任务二部图分配头，先用现有轨迹监督训练，再决定是否
使用 PPO。

具体结果、验收门槛和回滚点见 `TODO.md` 与 `改进日志.md`。
