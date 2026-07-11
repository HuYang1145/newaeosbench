# TimeModel 可行性校准设计

## 目标

在不重新训练 Stage3-200k 模型的首轮实验中，把现有 TimeModel feasibility
score 从隐式交叉注意力偏置和临时 `AEOS_TAU_S` 环境变量，升级为可配置、
可测试、可校准、可复现的推理门控机制。

首轮实验使用 Val Seen 和 Val Unseen 选择阈值，锁定阈值后只运行一次 Test，
比较 baseline 与门控模型的 `CR/PCR/WCR/PC_Wh`。不得在 Test 上扫描阈值。

## 不可违反的架构边界

AEOS-Former / Transformer / TimeModel 的任务是学习并替代昂贵的在线物理预测。
本设计禁止在训练数据热路径或推理决策循环中为每个卫星—任务候选调用 Basilisk、
完整轨道传播或重型几何预测。

Basilisk 的使用范围限定为：

- 离线生成轨迹和监督标签；
- 离线校准 TimeModel；
- 离线挖掘 hard negatives；
- 正式评估模型输出的物理可执行性。

正式推理只消费 TimeModel 已经输出的 feasibility score，通过轻量阈值门控任务
logits。后续连续窗口、预计开始时间和姿态机动时间若进入模型，必须来自离线标签、
预计算特征或当前已有轻量状态，不能通过在线 Basilisk 求解得到。

## 当前实现与问题

当前 `Transformer.forward()` 已执行以下流程：

1. `TimeModel.predict()` 输出每个卫星—任务对的 `feasibility_logits`；
2. logits 经 `_time_projection` 后作为 Decoder 交叉注意力的软偏置；
3. 如果设置环境变量 `AEOS_TAU_S`，代码把 sigmoid 概率不高于阈值的任务
   logits 置为负无穷。

当前机制存在以下问题：

- 阈值来自进程环境，评估配置和输出 JSON 不记录它，难以复现；
- 缺少阈值范围验证和独立单元测试；
- 旧扫描脚本使用已不存在的 Stage3-30k checkpoint 和 retry CSV；
- 旧扫描只覆盖极少场景，无法支持正式结论；
- 离线结果只记录动作 accuracy 和有限任务比例，没有 precision、recall、FPR、
  FNR、Brier score 或 ECE；
- TimeModel 当前监督来自模型/专家实际选择任务上的连续可见片段，不等价于所有
  卫星—任务候选的纯几何可达性，报告中必须写清这一标签边界。

旧的 2 场景结果只能作为历史提示：阈值 0.001 和 0.01 与 baseline 相同，0.05
降低了完成率。这不是正式阈值结论。

## 总体方案

采用“离线校准缩小候选 → 小规模 Basilisk smoke → 完整 validation → 单次 Test”
四阶段流程。

```text
Stage3-200k checkpoint + validation trajectories
                    |
                    v
离线 TimeModel 校准和 hard-negative 统计
                    |
                    v
选出少量 threshold 候选
                    |
                    v
少量 Val 场景 Basilisk smoke
                    |
                    v
Val Seen 64 + Val Unseen 64 正式评估
                    |
                    v
锁定 threshold
                    |
                    v
Test 64 单次正式评估
```

## 组件设计

### 1. 可行性门控函数

在 `constellation/new_transformers/` 中提供一个职责单一的门控函数：

- 输入任务 logits、feasibility logits 和可选 threshold；
- `threshold=None` 时原样返回任务 logits，保证 baseline 输出不变；
- threshold 必须在闭区间 `[0, 1]`；
- feasibility probability 小于或等于 threshold 的任务被置为负无穷；
- 空动作 logit 不参与门控；
- 如果某颗卫星的所有任务都被过滤，确定性推理自然选择空动作；
- 不调用 Basilisk 或任何几何传播函数。

门控逻辑从 `Transformer.forward()` 中分离，便于无模型权重的单元测试。

### 2. 配置化 threshold

`Model/JointModel/Transformer` 接受：

```text
feasibility_threshold: float | None
```

正式新流程通过模型/评估参数传入 threshold，不再依赖 `AEOS_TAU_S`。评估入口应
把 threshold 写入运行日志与汇总元数据，确保结果可追溯。

旧 `AEOS_TAU_S` 环境变量分支和以下旧脚本删除：

- `scripts/run_tau_s_scan_val_seen2.sh`
- `scripts/run_tau_s_scan_val_seen8.sh`

删除只覆盖这套已经确认失效的 threshold 扫描入口；其他脚本必须逐一核对引用后再
决定，不把“没用”扩展成无边界清理。

### 3. 离线校准工具

新增命令行工具，输入 checkpoint、split、annotation、设备、最大场景数、batch
大小、threshold 列表和输出路径。

工具复用 `JointDataset` 的轨迹读取、归一化和时间片段语义，避免另写一套不同标签
定义。首轮 ground truth 只覆盖实际被专家或模型选择的卫星—任务对：

- positive：动作保持且形成训练定义中的连续可见片段；
- negative：动作切换或没有形成连续可见片段；
- hard negative：TimeModel 高置信度判断可行，但对应选择片段从未形成真实可见。

输出 JSON 至 `work_dirs/timemodel_calibration/`，至少包含：

- checkpoint、split、annotation、scene ids 和样本数；
- positive/negative support；
- confusion matrix；
- precision、recall、specificity、FPR、FNR、F1；
- Brier score；
- ECE 及每个 calibration bin 的 count、confidence、accuracy；
- 每个 threshold 的保留比例和上述分类指标；
- hard-negative 数量及可选样本索引。

该工具不运行 Basilisk，只读取已经由 Basilisk 生成的离线 `is_visible` 标签。

### 4. 阈值候选选择

离线校准只用于排除明显危险的阈值，不能替代正式调度评估。候选应包含：

- baseline：`None`；
- 一个高 recall 阈值，尽量避免过滤真实可行任务；
- 一个 F1 或 Youden 指标候选；
- 至多一个更保守候选。

候选总数控制在三项以内，避免重复进行大量 Basilisk 评估。

### 5. 小规模 smoke 与完整 validation

先在固定少量 Val Seen/Val Unseen 场景上确认：

- threshold 确实改变有限任务比例和动作；
- baseline 与未设置 threshold 的旧模型输出一致；
- 不出现 NaN、全负无穷分布或子进程崩溃；
- 输出目录、日志和 threshold 元数据正确。

smoke 通过后，对 baseline 和剩余候选运行：

- Val Seen：正式 annotation 的 64 场景；
- Val Unseen：正式 annotation 的 64 场景；
- 如果资源允许，`environment.world_size=96`；
- 所有长任务使用 Slurm 或托管脚本；
- 日志写入 `work_dirs/eval_logs/`；
- 汇总写入 `work_dirs/eval_summaries/`。

阈值选择规则：首先最大化 Val Seen 与 Val Unseen 的平均 CR；若 CR 实质相同，依次
比较 PCR、WCR，最后选择 `PC_Wh` 更低的候选。所有候选仍须完整报告功耗。

### 6. 单次 Test

validation 结束后把阈值写入带时间戳或明确名称的选择结果 JSON。随后只对 baseline
和锁定阈值运行一次 Test 64 场景正式评估。

Test 不参与 threshold 修改。如果 Test 没有提升，应报告负结果并回到 validation
分析，不能再次根据 Test 结果选择新阈值。

### 7. hard-negative 输出

首轮工具只导出 hard-negative 索引和统计，不在本次无训练实验中修改 TimeModel
权重。输出放在 `work_dirs/timemodel_calibration/hard_negatives/`，记录：

- split、scene id、trajectory epoch；
- time step、satellite id、task id；
- feasibility probability；
- 实际连续可见标签和片段长度；
- checkpoint 与 threshold。

后续重新训练 TimeModel 时，再为 hard negatives 设计独立数据权重和消融实验。

## 明确延期的内容

以下内容不与首轮 threshold 实验混在同一个因果比较中：

- 修改 TimeModel 输入维度；
- 连续可见窗口、预计开始时间、姿态机动时间的新特征；
- 重新训练 TimeModel 或 JointModel；
- 在线调用 Basilisk 或完整物理几何预测；
- 在 Test 上搜索 threshold。

若首轮校准证明确有收益，再单独设计“离线物理监督增强”实验。任何新物理特征必须
先证明不会显著降低训练吞吐或推理效率。

## 测试与错误处理

单元测试至少覆盖：

- `threshold=None` 不改变 logits；
- threshold 边界 0、1 和等于阈值；
- 非法 threshold 抛出清楚错误；
- 被过滤任务变成负无穷，空动作不变；
- 所有任务被过滤后仍能选择空动作；
- `use_constraint_module=False` 时不能启用 feasibility threshold；
- baseline checkpoint 加载不因新增配置字段破坏；
- precision、recall、FPR、FNR、Brier、ECE 的手工小样本结果正确；
- hard-negative 导出不包含 positive 样本。

校准工具遇到缺失 checkpoint、annotation、trajectory 或空标签集合时应直接失败并
报告具体路径，不生成看似完整但无效的结果文件。

## 文件范围

预计修改：

- `AGENTS.md`：记录禁止在线物理预测的架构边界；
- `constellation/new_transformers/model.py`：接入配置化门控，删除环境变量逻辑；
- `constellation/rl/eval_all.py` 或相关配置入口：显式传递并记录 threshold；
- `tools/`：新增离线校准和 hard-negative 导出工具；
- `tests/`：新增门控与指标测试；
- `scripts/timemodel_calibration/`：新增 smoke、validation 和单次 Test 托管脚本；
- `TODO.md`：记录实验阶段、命令、日志和输出。

预计删除：

- `scripts/run_tau_s_scan_val_seen2.sh`；
- `scripts/run_tau_s_scan_val_seen8.sh`。

## 验收标准

- baseline 不设置 threshold 时，模型 logits 和动作与修改前一致；
- calibration JSON 包含完整指标、样本支持数和可追溯元数据；
- threshold 不依赖进程环境变量；
- 旧失效扫描脚本删除且仓库中无引用；
- 训练和推理热路径不新增 Basilisk 或完整几何预测；
- Val Seen/Val Unseen 完成严格 threshold 选择；
- Test 只使用锁定 threshold 运行一次；
- 最终报告同时给出 CR、PCR、WCR、PC_Wh 和负面结果；
- 每个实现提交都有定向测试，能够从基线 commit `ed87932` 恢复。
