# CLAUDE.md

NeurIPS 2025 论文 "Towards Realistic Earth-Observation Constellation Scheduling" 的官方实现。基于 Transformer 的地球观测卫星星座调度模型，搭配 Basilisk 物理模拟器。

## 环境

所有命令从仓库根目录运行，使用 `aeos` conda 环境：

```bash
/home/hy/miniconda3/envs/aeos/bin/python  # 直接使用完整路径
# 或通过 PYTHONPATH 时：
PYTHONPATH=:${PYTHONPATH} /home/hy/miniconda3/envs/aeos/bin/auto_torchrun ...
```

项目依赖 `todd`（北航内部框架）提供注册器、PyConfig、DDP 和 checkpoint 管理。

## 常用命令

```bash
# 训练动作模型（Stage-1 有监督）
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=:${PYTHONPATH} /home/hy/miniconda3/envs/aeos/bin/auto_torchrun -m constellation.new_transformers.train <name> constellation/new_transformers/config.py

# 训练时间模型
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=:${PYTHONPATH} /home/hy/miniconda3/envs/aeos/bin/auto_torchrun -m constellation.new_transformers.train <name> constellation/new_transformers/config_timemodel.py

# 完整流程（Stage1 + Stage2 专家迭代）
RUN_NAME=<name> TAU_E=4.5 bash scripts/train_paper_stage1_stage2.sh

# 评估
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 RANK=0 /home/hy/miniconda3/envs/aeos/bin/python -m constellation.rl.eval_all <name> constellation/rl/config_eval.py --load-model-from '<ckpt>'

# PPO 强化学习（实验性）
CUDA_VISIBLE_DEVICES=0 /home/hy/miniconda3/envs/aeos/bin/auto_torchrun -m constellation.rl.train <name> constellation/rl/config.py --load-model-from <ckpt>

# Stage-2: rollout + tau_e 过滤
WORLD_SIZE=1 RANK=0 /home/hy/miniconda3/envs/aeos/bin/python tools/rollout_model_trajectories.py <ckpt.pth> data/trajectories.N --split train --device cuda:0
/home/hy/miniconda3/envs/aeos/bin/python tools/build_tau_e_annotation.py data/annotations/train.json data/trajectories.N data/annotations/train_filtered.json --split train --candidate-epoch N --tau-e 4.5

# 基线评估与统计量
/home/hy/miniconda3/envs/aeos/bin/python tools/test_baseline.py <work_dir> <split>
/home/hy/miniconda3/envs/aeos/bin/python tools/compute_dataset_statistics.py
```

添加 `--load-model-from <path>` 可加载预训练权重。评估配置通过 `split`（val_seen/val_unseen/test）控制。

## 架构（`constellation/`）

| 模块 | 功能 |
|------|------|
| `data/` | 核心数据类型：Constellation, TaskSet, Actions, SensorType |
| `environments/` | Basilisk C++ 物理仿真后端（`third_party/basilisk`） |
| `task_managers.py` | 任务状态机：unreleased → ongoing → succeeded/failed |
| `controller.py` | 主仿真循环，协调环境/任务管理器/算法/回调 |
| `algorithms/` | 调度算法：最优求解、禁忌搜索、回放、神经网络包装器 |
| `evaluators/` | 评估指标：完成率、周转时间、功耗 |
| `new_transformers/` | AEOSFormer 模型：Encoder + Decoder + TimeModel |
| `rl/` | PPO 强化学习：环境封装、actor-critic、训练与批量评估 |

框架依赖 `todd`：PyConfig（配置）、Registry（可插拔组件）、IterBasedTrainer + DDP（分布式训练）、Memo（状态共享）、回调（Checkpoint/TensorBoard/LRSchedule）。

## 数据目录

```
data/
├── annotations/     # scene_id -> 专家动作列表
├── constellations/  # 卫星配置（split/XX/YYYY.json）
├── tasksets/        # 任务集
├── trajectories/    # 专家轨迹（.pth + .json）
├── trajectories.N/  # Stage-2 候选轨迹池
├── orbits/          # 轨道根数
├── satellites/      # 卫星硬件规格
└── statistics_new.pth  # 归一化统计量
```

## Stage-2 专家迭代

1. `tools/rollout_model_trajectories.py` — 运行模型，保存候选轨迹
2. `tools/build_tau_e_annotation.py` — 按 tau_e 过滤，生成新标注
3. 替换 `data/annotations/train.json`，在上一 checkpoint 基础上继续训练

`tools/wrap_time_model_checkpoint.py` — 将独立 TimeModel checkpoint 包装为主模型所需的嵌套键格式。
