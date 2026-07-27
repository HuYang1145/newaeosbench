# 完成率课程场景设计

## 1. 目标与边界

下一阶段候选主指标定义为：

```text
Q_completion = 0.8 * CR + 0.2 * PCR
```

该指标只用于后续课程学习研究，不追溯修改当前 job `4276`、后续依赖链或任何历史
结果。现行实验仍按论文指标
`Q_paper = 0.6*CR + 0.2*PCR + 0.2*WCR` 完成验收；正式报告继续完整记录
`CR/PCR/WCR/TAT_s/PC_Wh/CS_paper`。

本阶段只生成新的静态星座与任务集 JSON，不生成离线 trajectory，不启动 PPO，
也不运行 Val/Test。课程训练仍使用 Event V2 Actor-Critic 和 live Basilisk：

1. 若当前 V2-Large Gate 通过，从其 selected checkpoint 开始；
2. 若 Gate 失败，回退到已验证有效的 V2-2 selected checkpoint；
3. 依次进行 600 秒、1800 秒、现有 3600 秒场景的 PPO 训练；
4. 3600 秒阶段复用已有正式 train 场景，不生成新的 3600 秒数据。

## 2. 数据隔离

课程数据写入独立 split，禁止覆盖正式 benchmark：

```text
data/constellations/curriculum_600/
data/tasksets/curriculum_600/
data/constellations/curriculum_1800/
data/tasksets/curriculum_1800/
```

每个 split 生成 128 个静态场景：

- scene `0–119`：课程训练；
- scene `120–127`：课程 held-out checkpoint 选择；
- smoke 只使用训练 scene `0`，不额外消耗 held-out 场景；
- `curriculum_600` 固定种子为 `3407`，`curriculum_1800` 固定种子为
  `3408`；元数据记录种子、参数和文件数量。

输出目录必须原先不存在或为空。生成器不得覆盖单个已有 JSON；发生中断时保留现场并
报告，不自动删除实验状态。

## 3. 卫星与任务采样

两个课程阶段都复用现有 `data/satellites/train` 合格卫星池。该卫星池当前包含
2,063 个通过 MRP Basilisk 筛选的卫星资产。每次调用 `Constellation.sample()` 时，
从池中抽取控制与硬件基底，并重新采样轨道、初始真近点角、传感器、电池等场景属性。

课程规模如下：

| Split | Horizon | 卫星数 | 任务数 | 场景数 |
|---|---:|---:|---:|---:|
| `curriculum_600` | 600 s | 1–5 | 10–50 | 128 |
| `curriculum_1800` | 1800 s | 5–15 | 25–150 | 128 |

每个任务沿用正式 benchmark 的基本分布：

- `duration`：`15–60 s`；
- `release_time`：均匀采样于 `[0, horizon - 3*duration]`；
- `due_time`：均匀采样于 `[release_time + 3*duration, horizon]`；
- 纬度：`[-90, 90]`；经度：`[-180, 180]`；
- `sensor_type = VISIBLE`；
- 不做可观测性过滤，以免混入已经退出主线的 filtered-taskset 协议。

短场景必须重新生成 taskset，不能截断已有 3600 秒 taskset。否则会把尚未 release
的任务保留在分母中，破坏 CR/PCR 的语义。

## 4. 实现方案

新增独立工具 `tools/generate_curriculum_scenes.py`，不修改
`Task.sample()`、`TaskSet.sample()` 或正式
`tools/generate_constellations_and_tasksets.py`。

工具参数至少包含：

```text
--split
--horizon
--num-scenes
--satellite-min
--satellite-max
--task-min
--task-max
--seed
```

生成器执行以下步骤：

1. 加载 `data/satellites/train`；
2. 根据固定种子采样星座；
3. 使用 horizon-aware 任务采样器创建 taskset；
4. 先写入同一文件系统的临时目录；
5. 完成全量审计后原子移动到正式课程 split；
6. 写入不参与模型输入的
   `work_dirs/curriculum_scenes/<split>/metadata.json`，记录生成参数和完整性结果。

课程 split 继续使用现有
`BasiliskSceneBackend.from_scene_id(split=..., scene_id=...)` 读取，因此不修改在线
Basilisk 热路径。

## 5. 验证与验收

单元测试覆盖：

- 固定种子可复现；
- 两个 horizon 的卫星数、任务数范围正确；
- 所有任务满足 `0 <= release_time < due_time <= horizon`；
- 所有任务满足 `due_time - release_time >= 3*duration`；
- 经纬度、duration、sensor type 合法；
- 非空目标目录拒绝覆盖；
- 生成的 JSON 可由 `Constellation.load()` 和 `TaskSet.load()` 读取。

正式生成后审计：

- 两个 split 各有 128 个 constellation 和 128 个 taskset；
- scene ID 严格为 `0–127`，无缺口、无额外文件；
- held-out IDs 不进入训练 scene 列表；
- 各随机抽取一个场景完成 Basilisk backend 初始化；
- 不触碰 `data/constellations/train`、`data/tasksets/train`、Val 或 Test。

生成与审计成功后，向《改进日志.md》追加“下一阶段候选路线”，并在 `TODO.md`
记录实际目录、场景数量、种子、验证命令和结果。记录必须明确“静态课程场景已生成”
不等于“课程 PPO 已训练或完成率已提高”。

## 6. 后续训练顺序

课程 PPO 属于本数据生成任务之后的独立实施阶段：

```text
通过当前 Gate 的最佳 V2 checkpoint
  -> curriculum_600 scenes 0–119
  -> curriculum_600 scenes 120–127 选择 checkpoint
  -> curriculum_1800 scenes 0–119
  -> curriculum_1800 scenes 120–127 选择 checkpoint
  -> 现有未用于本轮调参的 3600 秒 train scenes
  -> 正式 Val Seen/Unseen 8+8
  -> 完整 Val 64+64
  -> Test 一次
```

600/1800 秒阶段只使用课程 held-out，不扫描官方 Val。每阶段按
`Q_completion` 选择 checkpoint，同时要求 CR、PCR 均不低于该阶段起点。WCR、
TAT、PC 和 CS 继续记录，但不参与课程阶段的 reward 或 checkpoint 排序。

少量事件对齐的 Basilisk 规划标签属于课程 PPO 之后的独立候选研究。只有课程训练在
完整 3600 秒验证上仍出现明确瓶颈，才重新设计规划标签；不得把历史 M3 未通过的
固定 180/300 秒偏好直接复用为监督。
