# P0 因果历史状态与 Temporal Adapter 设计

## 状态

- 日期：2026-07-16
- 基线：Stage3-200k `JointModel`
- 设计状态：已确认，等待用户审阅
- 实施状态：尚未修改模型代码，尚未启动训练

## 目标

在不推翻 Stage3-200k 主干、不引入在线 Basilisk 或轨道传播的前提下，让 Actor
显式获得每颗卫星的上一任务、决策前连续执行时长和近期切换历史，并通过零初始化的
`temporal residual adapter` 学习有物理结果的持续决策。

第一轮成功标准不是单独提高离线 top-1 accuracy，也不是机械降低切换次数，而是：

- 减少几乎无观测收益的一秒非空动作；
- 提高被选卫星—任务边的直接可见和进度转化；
- 保持或改善 `CR/PCR/WCR/TAT_s/PC_Wh/CS_paper`；
- 不显著增加重复冗余和推理耗时；
- 保持旧 checkpoint 的可加载性和功能关闭时的精确兼容性。

## 非目标

本设计第一轮不做以下工作：

- 不把动作改成强制 `1/5/15/30 s` commitment 宏动作；
- 不使用 pairwise 候选比较 loss；
- 不生成大规模 stay/switch 反事实轨迹；
- 不设置固定最短执行时长；
- 不使用 hard mask 或任务绝对唯一 owner；
- 不解冻全部约 9,000 万参数的 Transformer；
- 不进入 PPO、DPO 或端到端强化学习；
- 不使用 Test 调参。

已有的“多时间尺度宏动作边际价值策略”保留为第二阶段候选。本设计先回答更基础且
可隔离的问题：只补齐因果历史状态，是否足以改善当前 Actor 的时间持续性。

## 当前问题与证据

### 1. Actor 缺少显式历史状态

当前在线 observation 只包含时间、卫星物理状态、传感器状态、任务状态和 mask；
训练 `Batch/JointBatch` 也没有上一任务、连续执行时长和近期切换历史。Decoder 虽可从
姿态和传感器开关间接推断部分历史，但无法无损知道“当前候选是否就是上一任务”。

专家 `OptimalAlgorithm` 显式保存 `previous_assignment`，只要旧任务仍可执行就继续
保持；Stage3 Actor 没有等价的离散状态。现有轨迹审计又显示大量一秒短脉冲，因此
“状态表达不完整”是 P0 问题，而不是优先扩大模型容量。

### 2. 动作 CE 与物理结果目标错位

Stage3 的动作损失 `L_a` 对每颗卫星、每个抽样时间点做专家动作 CE。它能教模型模仿
专家当时选了什么，但不直接监督后续是否可见、是否产生进度、是否完成或是否产生
无收益切换。专家本身也存在局部贪心和重复分配，因此纯 CE 的监督上限受标签行为
限制。

### 3. 单步动作与 3,600 秒最终结果不能直接绑定

完整 `CS_paper` 属于整个调度 episode。一次动作之后还有数千次决策，后续行为会
淹没当前动作的局部影响。第一轮不再把单步动作直接绑定到完整 3,600 秒最终分数，
而是使用旧轨迹中可观测的短期事实结果。

### 4. 当前轨迹存在决策前/动作后对齐风险

`Controller.step()` 的顺序是先读取当前可见性和任务进度，再执行本时刻动作，随后
`TrajectoryLogger.after_step()` 保存传感器开关和卫星动态状态。因此轨迹中的
`sensor_enabled[t]` 已受 `action[t]` 影响，不能不加处理地作为预测同一
`action[t]` 的决策前输入。

P0 新数据路径必须显式区分：

- 输入只能来自决策前状态和 `actions[:t]`；
- 事实结果标签可以读取 `t+1` 及后续窗口；
- 未来结果绝不能进入 Actor 输入。

## 方案比较

### 方案 A：零初始化 Temporal Residual Adapter（采用）

保留 Stage3 Encoder、Decoder、TimeModel 和原 logits，在它们之后增加一个小型
历史感知边模块。模块消费卫星特征、任务特征、原始 logit 和因果历史特征，输出有界
`delta_logit` 与 pointwise outcome 预测。

优点：旧 checkpoint 兼容、变量隔离、可以只训练小模块、容易消融和回滚。缺点：
Actor 最终仍按卫星分解动作，不能单独解决全部跨卫星协调问题。

### 方案 B：把历史直接拼入 Decoder 卫星 token

把连续时长和切换统计拼接到现有 56 维卫星输入，再修改 Decoder input projector。

优点：历史信息能进入全部 Decoder block。缺点：改变输入维度、归一化统计和旧权重
形状；一旦结果变化，很难区分来自输入重构还是历史本身。第一轮不采用。

### 方案 C：联合输出任务与 commitment 宏动作

模型同时选择 `task_id` 和承诺秒数，控制器在承诺到期前维持任务。

优点：动作语义直接表达持续性。缺点：同时改变数据、模型输出、训练标签和在线控制，
容易出现过度坚持和安全中断错误。只有方案 A 证明历史信息有效但软残差不足时，才
进入该方案。

## 总体架构

```text
当前卫星/任务状态 ───────────────→ Stage3 Transformer ─→ 原始 logits
上一任务、连续时长、近期切换历史 ─→ Temporal Adapter ──→ delta logits
卫星/任务特征与事实结果标签 ─────→ Outcome Head ───────→ 局部结果预测

new_logits = stage3_logits + alpha * tanh(delta_logits)
```

`alpha` 是有界残差尺度。adapter 最后一层零初始化，因此在新模块未训练时必须满足：

```text
new_logits == stage3_logits
```

功能关闭时不创建任何行为修正，旧 checkpoint 的 logits 和动作必须与原实现逐位一致。

## 因果历史状态定义

### 1. 上一任务

在线环境内部保存每颗卫星上一时刻实际执行的全局 `task_id`。生成当前 observation
时，将该全局 ID 映射到当前 `valid_tasks` 的相对索引。

不得把全局 task id 当连续数值或普通 embedding 使用，因为编号没有物理大小意义，
且 ongoing task 列表会动态变化。模型最终使用的是边关系：

```text
previous_task_match[b, satellite, task]
```

若当前候选任务等于该卫星上一任务，则为 1，否则为 0。

同时提供两个卫星级标志：

- `previous_was_idle`：上一动作是否为空；
- `previous_task_available`：上一非空任务当前是否仍在 `valid_tasks` 中。

任务完成、到期、失败或关闭后，`previous_task_available=False`，Actor 不得被迫继续。

### 2. 决策前连续执行时长

在时间 `t` 做决策时：

```text
previous_task[t] = action[t - 1]
run_length_before_decision[t] =
    从 t - 1 向前连续等于 previous_task[t] 的长度
```

该值只允许由 `actions[:t]` 计算。不得使用从 `t` 向未来统计得到的完整动作片段长度，
否则会泄漏模型之后是否继续的答案。

模型输入使用：

```text
normalized_run_length = log1p(min(run_length, 300)) / log1p(300)
```

截断只控制数值范围，不代表在线强制最多保持 300 秒。

### 3. 近期切换历史

第一轮使用：

- `switch_count_30`：决策前最多 30 秒内动作发生变化的次数；
- `switch_count_60`：决策前最多 60 秒内动作发生变化的次数。

变化包括 `task -> idle`、`idle -> task` 和 `task A -> task B`。输入分别除以窗口长度
归一化。若后续审计证明三类切换需要区分，再增加分类计数；第一轮不扩展特征。

### 4. 时间语义

时间 `t` 的 Actor 输入只能消费：

```text
state_before_action_t
actions[0:t]
```

时间 `t` 的 outcome 标签可以消费：

```text
outcome[t+1:t+H]
```

任何修改 `actions[t:]` 的测试都不得改变时间 `t` 的历史输入。

## 离线训练数据流

### 1. 决策前状态重建

对 `t>0`，使用轨迹 `t-1` 保存的卫星动作后状态作为 `action[t]` 的决策前卫星状态；
任务 progress 使用与 Controller 顺序一致的当前决策状态。必须通过 replay 对齐测试
验证这一选择和在线 observation 一致，不能仅凭索引假设。

`t=0` 缺少动作前传感器状态。第一轮直接排除 `t=0` 动作监督样本，不补造状态。若
后续必须使用，再从场景 JSON 和环境 reset 状态显式重建。

### 2. 历史特征构造

新增纯函数模块统一构造：

- 上一全局任务；
- 上一任务在当前候选中的相对索引；
- `previous_task_match`；
- `run_length_before_decision`；
- `switch_count_30/60`；
- idle 和 available 标志。

Dataset、在线 Environment、轨迹 replay 和测试必须调用同一个定义或共享同一套纯
逻辑，禁止各自维护不同版本。

### 3. 事实结果标签

复用现有 `multi_horizon_edge_labels.py`，只为实际执行的非空卫星—任务边生成标签：

- 下一步直接可见；
- 下一步任务进度；
- 下一步完成；
- `5/15/30 s` 内可见、进度和完成；
- 第一次可见、第一次进度和完成的等待时间；
- 重复选择数和重复但本卫星下一步不可见。

连续执行满窗口且无事件可记为负样本；事件已发生可记为正样本；窗口结束前提前切换
且无事件时标为 censored，对应 loss mask 为 0，不能伪造负标签。

对于刚切换的新任务，`5/15/30 s` 可能短于姿态机动时间。额外记录最长 300 秒内的
`time_to_first_visible/progress`，但在再次切换、任务完成或到期时提前结束；该数据
用于事件时间监督，不把 300 秒窗口变成在线 commitment。

## 在线状态维护

`Environment` 为每个并行环境维护轻量历史状态：

- `last_global_task_ids[num_satellites]`；
- `run_lengths[num_satellites]`；
- 最近最多 61 个实际 assignment，用于 30/60 秒切换计数。

状态在每次 `_take_actions()` 中更新，包括 `_skip_idle()` 产生的每个空闲秒；否则跳过
空闲区间后历史将与真实执行时间不一致。

更新顺序固定为：

1. 用更新前历史生成当前 observation；
2. Actor 选择当前相对任务索引；
3. 在 `valid_tasks` 尚未变化前映射为全局 task id；
4. 执行动作；
5. 记录本秒实际全局 assignment 并更新 run length/switch history；
6. 生成下一时刻 observation。

reset 时全部上一任务置为 `-1`、run length 和切换数置为 0。任务失效只影响
`previous_task_available`，不篡改已经发生的历史记录。

## 模型接口

新增历史张量保持独立，不修改 `SATELLITE_DIM=56` 和现有归一化统计：

```text
previous_task_index:       b x ns
previous_task_available:   b x ns
previous_was_idle:         b x ns
run_length:                b x ns x 1
switch_count_30:           b x ns x 1
switch_count_60:           b x ns x 1
```

模型根据 `previous_task_index` 和当前 `nt` 构造：

```text
previous_task_match:       b x ns x nt
```

adapter 对每条有效边组合：

- Decoder 卫星特征；
- Encoder 任务特征；
- 原始 Actor task logit；
- `previous_task_match`；
- run length；
- 30/60 秒切换计数；
- previous idle/available 标志。

空动作使用独立的历史残差分支，至少消费 `previous_was_idle`、run length 和切换计数，
避免任务边修正后无意把所有卫星推向非空动作。

## Loss 设计

第一轮不使用 pairwise preference。总损失为：

```text
L_total = L_CE
        + lambda_visible * L_visible
        + lambda_progress * L_progress
        + lambda_completion * L_completion
        + lambda_event_time * L_event_time
```

- `L_CE`：保留原专家动作监督和 Stage3 行为锚点；
- `L_visible`：带 observed mask 的 BCE；
- `L_progress`：带 observed mask 的 BCE；
- `L_completion`：带 observed mask 的 BCE；
- `L_event_time`：只对真实观察到事件的样本做归一化 Smooth L1。

每个 horizon 单独报告正样本数、负样本数和 censored 数。若类别极不平衡，使用按训练
集计数确定的正类权重；不以普通 accuracy 作为主要验收指标。

第一轮 outcome head 和 logit residual 共享边表示，但 residual 最后一层独立零初始化。
先证明 outcome 可预测，再允许 residual 影响动作；不能因为 outcome loss 下降就直接
宣称 Actor 已改进。

## 训练策略

### 阶段 P0-A：兼容性与数据对齐

- 加载 Stage3-200k checkpoint；
- 新功能关闭时验证 logits 和动作逐位一致；
- 新功能开启但新头零初始化时验证 logits 仍一致；
- 使用真实轨迹做 Dataset/在线 replay 历史特征一致性检查；
- 验证未来动作扰动不会改变当前历史输入；
- 审计事实标签 observed/censored 覆盖率。

本阶段不训练模型。

### 阶段 P0-B：冻结主干训练新模块

冻结：

- Encoder；
- Decoder；
- TimeModel；
- 原 task logits 参数；
- 已有 assignment head（若未启用则保持关闭）。

只训练：

- history projection；
- temporal edge adapter；
- null-action temporal adapter；
- outcome heads。

从 Stage3-200k 权重开始，使用现有训练轨迹，不重新生成专家轨迹。按 scene 严格划分
train/val，使用验证集早停；具体 iteration 数由 observed 标签量和收敛曲线决定，不
预先固定为完整 200k。

### 阶段 P0-C：低风险行为接入

只有 P0-B 的 outcome head 在未见 scene 上优于简单基线，才允许逐步增加有界残差
尺度 `alpha`。若预测有效但 residual 对动作几乎无影响，先检查梯度、logit margin 和
残差尺度；仍不足时才低学习率解冻 Decoder 最后一个 block。

不在第一轮解冻 Encoder 或全部 Decoder，不进入 PPO。

### 阶段 P0-D：正式 Basilisk 验真

按同场景协议依次运行小规模 Val、8+8 Val、完整 64+64 Val。只有完整 Val 通过门槛，
才考虑 Test；Test 不参与超参数选择。

## 预期行为变化

预期但不预先承诺的变化包括：

- `task -> idle -> same task` 一秒脉冲减少；
- 有效上一任务的连续执行长度增加；
- 非空动作下一步直接可见率提高；
- “选过但从未由所选卫星看到”的任务比例下降；
- 无收益传感器开关减少，功耗可能下降；
- CR/PCR/WCR 保持或提高；
- 推理延迟仅有小幅增加。

P0 不保证解决跨卫星全局协调。多颗卫星可能同时坚持同一个旧任务，因此必须同时监控
重复冗余和合理接力；若时间持续性通过而全局重复仍高，再进入 P1 联合分配。

## 风险与修复策略

| 风险 | 可观察症状 | 修复策略 |
|---|---|---|
| 未来信息泄漏 | 离线指标异常高、正式评估无收益 | 输入只读 `actions[:t]`；加入未来动作扰动不变量测试；t=0 不补造状态。 |
| 轨迹状态时间错位 | replay 特征与在线 observation 不一致 | 用同场景逐步 replay 对齐；明确使用决策前帧；不沿用同索引假设。 |
| 全局/相对 task id 错位 | previous match 指向错误候选 | 环境保存全局 ID，生成 observation 时映射到当前 `valid_tasks`；无匹配显式标 unavailable。 |
| `_skip_idle()` 丢失历史秒数 | 长空闲后 run length/switch count 不真实 | 每次 `_take_actions()` 都更新历史，包括自动跳过的空闲秒。 |
| 模型过度坚持旧任务 | 切换下降但错过新任务、CR 下降 | 只用有界软残差；提供任务 available、due/progress；不做硬 commitment。 |
| 多星重复被锁死 | 多颗卫星长期坚持同一任务 | 监控重复率；P0 不硬 stay；必要时 P1 增加软竞争信息。 |
| adapter 被忽略 | outcome 可预测但 top-1 几乎不变 | 检查梯度与 logit margin，逐步调残差尺度，最后才解冻 Decoder 最后一层。 |
| Stage3 能力被破坏 | top-k 覆盖或完成率下降 | 零初始化、冻结主干、保留 CE、小学习率、逐层解冻。 |
| observed/censored 失衡 | loss 只来自少量长片段 | 按 horizon 报告覆盖；masked loss；不足时停止，而不是把 censored 当负样本。 |
| 类别极不平衡 | 高 accuracy 但 PR-AUC 很差 | 使用训练集正类权重，报告 PR-AUC、Brier、ECE 和混淆指标。 |
| 短窗口误判姿态机动 | 新任务在 30 秒内不可见被判坏 | 对新 switch 监督 time-to-first-event，最长 300 秒；不把短窗口阴性直接当长期无效。 |
| 推理开销过高 | 每步延迟显著增加 | 小 hidden width、向量化边计算、必要时只对 Actor top-k 计算 outcome；不调用在线物理预测。 |

## 测试设计

### 单元测试

- prefix run length 只依赖过去动作；
- 未来动作变化不影响当前历史特征；
- 30/60 秒切换计数边界正确；
- idle、任务完成、任务消失和重新发布的语义明确；
- 全局 task id 正确映射到当前候选相对索引；
- `_skip_idle()` 每秒更新历史；
- censored 标签不参与负样本 loss；
- outcome 各分支 mask 和归一化正确；
- residual 零初始化时输出严格为 0；
- 非法历史形状、越界索引和 task/logit 不一致时立即报错。

### 集成测试

- 旧 checkpoint 在功能关闭时 logits 与原实现逐位一致；
- 新功能开启、未训练时 logits 仍一致；
- 同一真实轨迹的离线历史特征与在线 replay 完全一致；
- 一条真实 `JointBatch` 完成 forward、backward 和 optimizer step；
- 冻结训练时仅新模块参数发生变化；
- evaluation 使用新 observation 后能完整运行若干步；
- 在线推理路径不调用 Basilisk 可见性预测、完整轨道传播或重型几何模块；
- 单线程交错基准比较 Stage3 与新模型推理耗时。

## 验收门槛

### 工程门槛

以下条件必须全部满足：

1. 功能关闭时旧 checkpoint logits 与动作逐位一致；
2. 离线 Dataset 与在线 replay 历史特征逐项一致；
3. 未来动作扰动不改变当前输入；
4. t=0、idle skip、任务失效和 episode reset 均有测试；
5. 真实样本训练步 loss/梯度有限，且仅新参数更新；
6. 在线推理无 Basilisk/轨道传播热路径调用。

### 离线监督门槛

1. outcome head 在严格 scene-level 未见验证集上优于常数、正样本率和 Actor logit
   等简单基线；
2. 主要分类指标使用 PR-AUC、Brier、ECE、precision/recall，不使用普通 accuracy
   单独验收；
3. 历史特征打乱后验证性能应显著下降，证明模型确实使用了历史；
4. 每个 horizon 报告 observed/censored 覆盖，覆盖不足时停止进入行为接入；
5. 不能通过永远预测 stay、空闲或不可见获得表面通过。

“显著优于”使用按 scene bootstrap 的 95% 置信区间判断，改善区间必须不跨 0，避免
另行拍定不稳定的绝对阈值。

### 正式行为门槛

第一轮建议门槛：

- 一秒非空片段率相对 Stage3 至少下降 20%；
- 非空动作下一步直接可见率相对提高至少 10%；
- `CR/PCR/WCR` 任一项不得下降超过 0.5 个百分点；
- Val Seen 和 Val Unseen 的 `CS_paper` 均不得恶化；
- 重复冗余率不得因多星共同坚持旧任务而显著上升；
- 推理耗时增加不超过 5%；
- 完整 Val 使用逐场 paired comparison，并报告均值、中位数和 bootstrap 置信区间。

行为门槛中的 20%、10%、0.5 个百分点和 5% 是本轮预注册筛选阈值。若需修改，必须
在查看对应正式 Val 结果前记录原因，不能看到结果后调整门槛。

## 文件与模块边界

计划中的实现边界如下，具体逐文件步骤在用户审阅本设计后另写 implementation plan：

- 新增纯历史状态与映射模块，供 Dataset、Environment 和 replay 共用；
- 修改 `dataset.py` 的 `Batch/JointBatch` 与因果状态构造；
- 修改 `rl/environment.py` 的 observation space、历史状态维护和 reset/step；
- 修改 `rl/policy.py` 的 `Observation/Batch/FeatureExtractor` 字段传递；
- 新增独立 temporal adapter/outcome head 模块；
- 通过显式配置开关接入 `model.py`，旧路径默认不变；
- 新增独立训练配置、评估脚本和定向测试；
- 复用 `multi_horizon_edge_labels.py`，不复制标签语义；
- 不覆盖 P3.0/P3.1、宏动作或既有评估输出。

## 回滚与实验隔离

- 从实施前基线创建独立 `codex/<topic>` 分支；
- 目标文件已有未提交改动时先建立用户确认的 checkpoint 或本地备份；
- 历史状态、adapter、训练接入和评估脚本拆成职责单一的小提交；
- 新 checkpoint、日志和评估目录使用独立后缀，不覆盖 Stage3 或 P3.x 资产；
- 旧 checkpoint 在功能关闭时始终可直接评估；
- 方向失败时使用 `git revert <commit>`，不使用破坏性 reset；
- 不修改或替换现有轨迹、tasksets、模型权重和标注路由。

## 决策结论

P0 第一轮采用“因果历史状态 + 零初始化 temporal residual adapter + pointwise 事实
结果辅助监督”。它需要从 Stage3-200k 继续训练新模块，但不需要从头训练主干，也不
需要重新生成专家轨迹。

只有 P0 在严格离线和正式 Val 门槛上证明有效，才继续解冻 Decoder、引入宏动作
commitment 或研究跨卫星联合分配；否则保留负结果，回到状态对齐、标签覆盖和候选
分布诊断，不盲目扩大模型。
