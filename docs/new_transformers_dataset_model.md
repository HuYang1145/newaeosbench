# new_transformers 中 Dataset 和 Model 的作用

本目录是 AEOS-Former 训练主线。训练时，`dataset.py` 负责把磁盘上的场景和轨迹文件整理成模型输入，`model.py` 负责根据这些输入计算预测和训练损失。

## 1. 训练数据从哪里来

训练不是直接读取一个单独的大表，而是把多个文件组合起来：

```text
data/annotations/*.json
data/trajectories.N/<split>/<id>.pth
data/constellations/<split>/<id>.json
data/tasksets/<split>/<id>.json
data/statistics_new.pth
```

这些文件各自的含义不同：

- `annotations/*.json`
  - 训练索引文件。
  - 告诉 Dataset：这次训练使用哪些场景 ID，以及每个场景应该读取哪个 `trajectories.N`。

- `trajectories.N/*.pth`
  - 某个算法或某轮模型 rollout 生成的轨迹。
  - 包含每秒卫星动态状态、任务进度、专家动作、可见性矩阵。

- `constellations/*.json`
  - 静态星座配置。
  - 描述这个场景有哪些卫星、卫星硬件参数、轨道参数、初始状态。

- `tasksets/*.json`
  - 静态任务集合。
  - 描述这个场景有哪些观测任务、发布时间、截止时间、观测时长、地面坐标、传感器类型。

- `statistics_new.pth`
  - 特征归一化统计量。
  - Dataset 会用它对卫星特征和任务特征做标准化。

## 2. annotation 为什么有 ids 和 epochs

当前 annotation 通常是一个字典：

```json
{
  "ids": [0, 1, 4, 5],
  "epochs": [3, 1, 1, 2]
}
```

含义是：

```text
id=0 读 data/trajectories.3/<split>/00/00000.pth
id=1 读 data/trajectories.1/<split>/00/00001.pth
id=4 读 data/trajectories.1/<split>/00/00004.pth
id=5 读 data/trajectories.2/<split>/00/00005.pth
```

这样设计是为了支持 Stage-2/Stage-3 专家迭代：同一个场景可以有多轮轨迹，annotation 决定当前训练采用哪一轮轨迹。

## 3. `Dataset` 做了什么

`Dataset` 是普通动作模型训练用的数据集。

它的核心入口是：

```python
Dataset.__getitem__(index)
```

每取一个训练样本，会执行以下步骤。

### 3.1 读取 annotation

```python
id_ = self._annotations["ids"][index]
best_epoch_ = self._annotations["epochs"][index]
```

`id_` 表示场景编号，`best_epoch_` 表示读取哪一轮 trajectory。

### 3.2 读取 trajectory

```python
data/trajectories.<best_epoch_>/<split>/<id>.pth
```

trajectory 中包含：

```text
constellation.sensor_enabled: 每秒每颗卫星传感器是否开启
constellation.data: 每秒每颗卫星动态状态
taskset.progress: 每秒每个任务完成进度
actions.task_id: 每秒每颗卫星选择的专家任务 ID
is_visible: 每秒每颗卫星对每个任务是否可见
```

### 3.3 读取 constellation

```python
data/constellations/<split>/<id>.json
```

`Dataset._load_constellation()` 会读取卫星静态信息，并与 trajectory 里的卫星动态信息拼接：

```text
卫星输入特征 = 卫星静态特征 + 卫星动态特征
```

静态特征来自 JSON，例如卫星质量、惯量、轨道参数、传感器参数、电池容量、反作用轮参数等。

动态特征来自 trajectory，例如电量百分比、反作用轮速度、真近点角、姿态等。

### 3.4 读取 taskset

```python
data/tasksets/<split>/<id>.json
```

`Dataset._load_tasks()` 会读取任务静态信息，并与 trajectory 里的任务进度拼接：

```text
任务输入特征 = 任务静态特征 + 任务动态进度
```

任务静态特征包括：

```text
release_time
due_time
duration
coordinate.x
coordinate.y
```

在具体时间步 `t`，代码会把 `release_time` 和 `due_time` 改成相对当前时间：

```text
release_time -= t
due_time -= t
```

这样模型看到的是“距离发布还有多久”和“距离截止还有多久”，而不是绝对时间。

### 3.5 生成 mask

Dataset 会判断每个任务在每个时间步是否有效：

```text
已经发布：release_time <= 0
还没过期：due_time >= 0
还没完成：progress < duration
```

无效任务会被 mask 掉，避免模型在训练时选择不应该选择的任务。

### 3.6 读取专家动作标签

```python
actions_task_id = trajectory["actions"]["task_id"][indices]
```

这是模型要学习的标签。含义是：在这个时间步，专家算法或上一轮轨迹中，每颗卫星选择了哪个任务。

`-1` 表示该卫星不选择任务。进入模型损失时会整体加 1，把 `-1` 变成第 0 类，也就是空任务/null task。

### 3.7 输出 Batch

最终 `Dataset` 输出 `Batch`：

```text
time_steps
constellation_sensor_type
constellation_sensor_enabled
constellation_data
constellation_mask
tasks_sensor_type
tasks_data
tasks_mask
actions_task_id
```

这些就是 `Model.forward()` 的输入和监督标签。

## 4. `JointDataset` 做了什么

`JointDataset` 继承 `Dataset`，除了输出普通动作训练数据，还额外构造约束模块训练数据。

它多输出：

```text
constraint_time_steps
constraint_constellation_data
constraint_tasks_data
constraint_durations
```

这些数据用于训练 `TimeModel`，对应论文里的：

```text
L_s: feasibility loss，可行性预测损失
L_t: time loss，时间/持续时间预测损失
```

### 4.1 正样本

如果一颗卫星持续指向某个任务，并且连续可见，说明这个卫星-任务组合在这段时间内是可行的。

这类样本会被放进正样本，`constraint_durations >= 0`。

### 4.2 负样本

如果动作发生切换，或者轨迹显示不可持续，说明这个组合不能稳定完成当前任务。

这类样本会被放进负样本，`constraint_durations < 0`。

### 4.3 为什么 JointDataset 更符合论文训练

普通 `Dataset` 只训练动作：

```text
输入状态 -> 预测任务分配 -> L_a
```

`JointDataset` 同时训练动作和约束模块：

```text
输入状态 -> 预测任务分配 -> L_a
卫星-任务组合 -> 预测可行性 -> L_s
卫星-任务组合 -> 预测持续时间 -> L_t
```

所以当前论文式主线应该使用：

```text
JointDataset + JointModel
```

而不是只使用：

```text
Dataset + Model
```

## 5. `model.py` 和 `dataset.py` 的关系

`dataset.py` 不训练模型，它只负责把文件变成张量。

`model.py` 才真正计算神经网络输出和损失。

调用关系是：

```text
Dataset / JointDataset
        |
        v
Batch / JointBatch
        |
        v
Model.forward() / JointModel.forward()
        |
        v
loss
        |
        v
todd trainer 反向传播和更新参数
```

## 6. `Model` 做了什么

`Model` 是动作预测模型。

它调用内部 `Transformer`：

```text
任务特征 -> Encoder
卫星特征 -> Decoder
卫星特征 cross-attend 任务特征
        |
        v
每颗卫星对每个任务的 logits
```

然后计算动作分类损失：

```text
ce_loss = CrossEntropy(logits, actions_task_id)
```

也就是只训练：

```text
L_a
```

## 7. `JointModel` 做了什么

`JointModel` 是论文式联合训练模型。

它包含三项损失：

```text
L_a: assignment loss，任务分配动作损失
L_s: feasibility loss，可行性预测损失
L_t: time loss，持续时间预测损失
```

代码中总损失是：

```text
loss = feasibility_loss_weight * L_s
     + time_loss_weight * L_t
     + assignment_loss_weight * L_a
```

当前论文式配置中三个权重都是 1。

## 8. 最重要的一句话

`Dataset` 和 `JointDataset` 解决的是“训练样本怎么从文件变成模型输入”；`Model` 和 `JointModel` 解决的是“模型如何根据这些输入预测动作、可行性和时间，并计算损失”。
