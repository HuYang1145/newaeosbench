# constellation 代码结构与训练流程说明

本文档用于快速理清 `constellation/` 代码包和 `tools/`、`scripts/` 在实验中的关系。

## 1. 实验入口应该从哪里开始

正式实验优先运行 `scripts/` 里的 `.sh` 文件，而不是直接手动拼很长的 Python 命令。

`.sh` 文件是 shell script，也就是写在文件里的 Bash 命令。执行方式通常是：

```bash
bash scripts/run_paper_joint_200k_managed.sh
```

这些脚本的作用是把一整套实验命令固定下来，包括：

- 使用 `aeos` 虚拟环境：`/home/hy/miniconda3/envs/aeos/bin`
- 设置 `PYTHONPATH`
- 选择训练或评估配置
- 指定 checkpoint 路径
- 写日志到 `work_dirs/eval_logs/`
- 写结果到 `work_dirs/`
- 调用 `tools/` 中的底层工具

所以可以这样理解：

```text
scripts/*.sh = 实验流程入口
tools/*.py   = 被流程调用的工具
constellation/ = 真正的项目代码和模型代码
```

## 2. 根本代码流程

项目主流程可以概括为：

```text
data/annotations/*.json
data/trajectories.N/*.pth
data/constellations/*.json
data/tasksets/*.json
        |
        v
constellation/new_transformers/dataset.py
        |
        v
constellation/new_transformers/model.py
        |
        v
constellation/new_transformers/train.py
        |
        v
work_dirs/<run_name>/checkpoints/
```

评估和 rollout 流程是：

```text
checkpoint
        |
        v
constellation/rl/eval_all.py 或 tools/rollout_model_trajectories.py
        |
        v
constellation/controller.py
        |
        v
constellation/environments/basilisk/
        |
        v
constellation/evaluators/
        |
        v
work_dirs/ 或 data/trajectories.N/
```

## 3. constellation 子目录职责

### `constellation/data/`

定义项目的基础数据结构。

关键文件：

- `actions.py`
  - 定义 `Action` 和 `Actions`。
  - 一颗卫星每一步的动作包含两件事：是否开关传感器、指向哪个目标位置。

- `tasksets.py`
  - 定义 `Task` 和 `TaskSet`。
  - 任务包含发布时间、截止时间、持续观测时长、坐标、传感器类型。
  - 提供 `to_tensor()`，训练和评估会把任务转成张量。

- `constellations.py`
  - 定义卫星、传感器、电池、反作用轮、星座等结构。
  - 提供 `static_to_tensor()` 和 `dynamic_to_tensor()`。
  - 静态特征包括传感器、轨道、硬件参数；动态特征包括电量、姿态、反作用轮速度等。

- `orbits.py`
  - 定义轨道数据结构。

- `coordinates.py`
  - 定义坐标类型。

### `constellation/environments/`

定义仿真环境。

关键文件：

- `base.py`
  - 定义 `BaseEnvironment` 抽象接口。
  - 核心方法包括 `get_constellation()`、`take_actions()`、`step()`、`is_visible()`、`get_earth_rotation()`。

- `timer.py`
  - 维护当前仿真时间。

- `basilisk/basilisk_environment.py`
  - 真正调用 Basilisk 物理仿真的环境。
  - 负责初始化地球、太阳、卫星、任务地面点。
  - `is_visible()` 判断每颗卫星对每个任务是否可见，并检查传感器状态和传感器类型。
  - `take_actions()` 根据动作控制传感器开关和卫星姿态。
  - `step()` 推进 Basilisk 仿真。

### `constellation/task_managers.py`

维护任务状态。

任务状态逻辑是：

```text
未发布 -> 进行中 -> 成功
              |
              -> 超过截止时间后失败
```

`TaskManager.record(is_visible)` 会根据当前可见性更新任务进度：

- 如果任务正在进行，并且至少被一颗卫星可见，则进度增加。
- 进度达到任务 duration 后，任务标记为成功。

### `constellation/controller.py`

仿真主循环。

`Controller.run()` 每个时间步做：

1. 调用算法 `algorithm.step(...)`，得到每颗卫星的动作。
2. 调用 `environment.is_visible(...)`，得到可见性矩阵。
3. 调用 `task_manager.record(...)`，更新任务进度。
4. 调用 `environment.take_actions(...)`，执行卫星动作。
5. 调用 callbacks 的 `after_step()`，记录指标或轨迹。
6. 时间推进一秒，Basilisk 环境推进一步。

### `constellation/algorithms/`

传统算法和专家算法。

- `base.py`
  - 定义所有算法的接口：`prepare()` 和 `step()`。

- `optimal.py`
  - `OptimalAlgorithm`。
  - 根据当前任务和卫星位置做几何约束判断，选择距离最近且满足观测约束的任务。
  - 主要用于生成专家轨迹。

- `replay.py`
  - `ReplayAlgorithm`。
  - 从已有轨迹里读取动作并回放。

### `constellation/callbacks/`

回调系统。

`Controller` 不直接知道所有指标和日志逻辑，而是在不同时间点调用 callbacks：

- `before_run()`
- `before_step()`
- `after_step()`
- `after_run()`

`ComposedCallback` 把多个 callback 组合在一起。

### `constellation/evaluators/`

指标计算。

- `completion_rate.py`
  - 计算 `CR`、`PCR`、`WCR`、`WPCR`。

- `turn_around_time.py`
  - 计算 `TAT`。
  - 当前本地 TAT 与论文尺度不一致，因此正式比较时暂时不要把 TAT 当成最终判断依据。

- `power_usage.py`
  - 计算 `PC`。
  - 后续汇总时通常换算为 `PC_Wh`。

### `constellation/loggers/`

轨迹记录。

- `trajectory.py`
  - `TrajectoryLogger` 每一步保存卫星状态、任务进度、动作、可见性。
  - 输出 `.pth` 轨迹文件。

- `forbid_tasks.py`
  - 记录不应继续分配的任务，用于旧的 tabu/专家轨迹流程。

### `constellation/new_transformers/`

AEOS-Former 训练主线。

关键文件：

- `train.py`
  - 训练入口。
  - 读取 config，构造 `todd` 的 trainer，然后运行训练。

- `dataset.py`
  - `Dataset` 读取动作模型训练样本。
  - `JointDataset` 读取联合训练样本。
  - 数据来自 annotation 和 trajectory：

```text
data/annotations/*.json
        |
        v
annotation 中的 id 和 epoch
        |
        v
data/trajectories.<epoch>/<split>/<id>.pth
```

- `model.py`
  - `Model` 是动作预测模型，只优化 action 分类损失 `ce_loss`。
  - `JointModel` 是当前论文式主线，联合优化：
    - `L_s`：feasibility loss，可行性预测损失。
    - `L_t`：time loss，控制时间预测损失。
    - `L_a`：assignment loss，任务分配动作损失。
  - 当前论文式配置中三项权重都是 1。

- `time_model.py`
  - `TimeModel` 预测卫星-任务组合是否可行，以及预计持续时间。
  - `JointModel` 内部会调用它作为约束模块。

- `config.py`
  - 旧动作模型训练配置。
  - 使用 `ConstellationDatasetRegistry.Dataset` 和 `ConstellationModelRegistry.Model`。
  - 主要是 CE/action loss，不是严格论文联合训练。

- `config_paper_stage1_200k.py`
  - 当前 200k 论文式 Stage 1 配置。
  - 使用 `JointDataset` 和 `JointModel`。
  - annotation 是 `train_epoch1_existing.json`。

- `config_paper_stage2_200k.py`
  - 当前 200k 论文式 Stage 2 配置。
  - annotation 是 `train_paper_stage2_tau_e_existing.json`。

- `config_paper_stage3_200k.py`
  - 当前 200k 论文式 Stage 3 配置。
  - annotation 是 `train_paper_stage3_tau_e_existing.json`。

### `constellation/rl/`

评估和强化学习外壳。

当前正式评估主要看：

- `eval_all.py`
  - 批量评估入口。
  - 根据 split 遍历 annotation 中的场景。
  - 创建 `EvalEnvironment`。
  - 加载 checkpoint。
  - 输出每个场景的指标 JSON。

- `controller_environment.py`
  - 把 `Controller + BasiliskEnvironment + TaskManager` 包装成 Gym 环境。
  - 每一步把模型动作转成 `Actions`，再调用 `Controller.step()`。

- `policy.py`
  - 把 `new_transformers.Model` 接到 Stable-Baselines3 的 `Policy` 外壳里。
  - 评估时虽然外层叫 PPO，但 actor 实际加载的是你的 AEOS-Former checkpoint。

- `config_eval.py`
  - 评估配置，默认 `world_size=1`、`split='val_seen'`。
  - 正式脚本会用 `--override` 改成 `world_size=96` 和指定 split。

## 4. 训练模型用到了哪些 tools

严格说，模型训练本身主要用 `constellation/new_transformers/`，不直接依赖很多 `tools/`。

当前训练脚本主要分两类。

### 4.1 直接训练 200k 联合模型

入口：

```bash
bash scripts/run_paper_joint_200k_managed.sh
```

调用链：

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
config_paper_stage1_200k.py
config_paper_stage2_200k.py
config_paper_stage3_200k.py
        |
        v
JointDataset
        |
        v
JointModel
        |
        v
work_dirs/paper_joint_stage*/checkpoints/
```

这个流程通常不直接调用 `tools/`。

但是它依赖 `tools/` 之前生成或维护过的数据资产，例如：

- `data/trajectories.N/`
- `data/annotations/train_paper_stage2_tau_e_existing.json`
- `data/annotations/train_paper_stage3_tau_e_existing.json`

这些 Stage-2/Stage-3 annotation 通常来自 rollout 和 tau_e 筛选。

### 4.2 Stage-2/Stage-3 自探索训练

入口示例：

```bash
bash scripts/start_stage2_round4_parallel.sh
```

调用链：

```text
scripts/start_stage2_round4_parallel.sh
        |
        | 1. rollout
        v
tools/rollout_model_trajectories.py
        |
        v
data/trajectories.4/
        |
        | 2. tau_e 筛选
        v
tools/build_tau_e_annotation.py
        |
        v
data/annotations/train_<run_name>.json
        |
        | 3. 继续训练
        v
auto_torchrun -m constellation.new_transformers.train
        |
        v
constellation/new_transformers/train.py
        |
        v
constellation/new_transformers/config.py 或对应 config
```

这里实际用到的 `tools` 是：

- `rollout_model_trajectories.py`
  - 用当前模型生成候选轨迹。

- `build_tau_e_annotation.py`
  - 用 `tau_e` 筛选候选轨迹，生成新 annotation。

如果是旧的 `train_paper_stage1_stage2.sh`，还会用：

- `wrap_time_model_checkpoint.py`
  - 把单独 TimeModel checkpoint 包装成主模型可加载格式。

## 5. 训练模型用到了哪些 constellation 文件

训练入口：

```text
constellation/new_transformers/train.py
```

配置文件：

```text
constellation/new_transformers/config.py
constellation/new_transformers/config_paper_stage1_200k.py
constellation/new_transformers/config_paper_stage2_200k.py
constellation/new_transformers/config_paper_stage3_200k.py
```

数据读取：

```text
constellation/new_transformers/dataset.py
constellation/data/constellations.py
constellation/data/tasksets.py
constellation/data/actions.py
```

模型：

```text
constellation/new_transformers/model.py
constellation/new_transformers/time_model.py
constellation/new_transformers/constants.py
```

注册器：

```text
constellation/new_transformers/registries.py
```

训练时的数据流：

```text
annotation id/epoch
        |
        v
trajectory .pth
        |
        v
Dataset/JointDataset
        |
        v
Batch/JointBatch
        |
        v
Model/JointModel.forward()
        |
        v
loss
        |
        v
todd IterBasedTrainer
        |
        v
optimizer + checkpoint
```

## 6. JointModel 具体在训练什么

`JointModel.forward()` 做三件事。

第一，动作预测：

```text
输入当前时间、卫星状态、任务状态
        |
        v
Transformer
        |
        v
每颗卫星对每个任务的 logits
        |
        v
CrossEntropyLoss
        |
        v
L_a
```

第二，可行性预测：

```text
constraint_time_steps
constraint_constellation_data
constraint_tasks_data
        |
        v
TimeModel._predict()
        |
        v
pred_masks
        |
        v
BCEWithLogitsLoss
        |
        v
L_s
```

第三，控制时间预测：

```text
TimeModel._predict()
        |
        v
pred_durations
        |
        v
MSELoss
        |
        v
L_t
```

总损失：

```text
loss = L_s + L_t + L_a
```

当前 `config_paper_stage*_200k.py` 里三项权重都是 1。

## 7. 评估模型用到了哪些代码

入口：

```bash
bash scripts/run_stage3_200k_96core_eval_managed.sh
```

调用链：

```text
scripts/run_stage3_200k_96core_eval_managed.sh
        |
        v
python -m constellation.rl.eval_all
        |
        v
constellation/rl/eval_all.py
        |
        v
EvalEnvironment / ControllerEnvironment
        |
        v
Controller
        |
        v
BasiliskEnvironment
        |
        v
TaskManager
        |
        v
Policy 加载 AEOS-Former checkpoint
        |
        v
Evaluators
        |
        v
work_dirs/rl_eval_*/<split>/*.json
        |
        v
tools/summarize_no_tat_eval.py
```

核心含义：

- `Policy` 负责把当前仿真观测输入 AEOS-Former。
- AEOS-Former 输出每颗卫星选择哪个任务。
- `ControllerEnvironment` 把任务编号转成 `Action`。
- `Controller` 推进仿真。
- `Evaluator` 在仿真结束后写出指标。

## 8. tools/legacy

`tools/legacy/` 里是旧流程或不再作为当前主入口的工具：

- `compare_trajectory_cr.py`
- `evaluate_baseline.py`
- `generate_data.py`
- `generate_tasks.py`
- `merge_tabu.py`

这些文件保留是为了历史参考。当前正式复现实验不应优先从这些文件开始。
