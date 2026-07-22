# Transformer V2 待办事项

## 当前状态

当前唯一活动主线是新建 **V2 事件级联合 Transformer Actor-Critic**。V2 与现有
Stage3 `JointModel` 并行存在，不覆盖 Stage3 checkpoint、M2/M3 结果或正式评估
输出。

设计已经逐节确认，正式文档为：

`docs/superpowers/specs/2026-07-21-event-joint-transformer-v2-design.md`

当前尚未编写 V2 模型代码，也没有启动 V2 训练。下一步是在用户审阅正式设计后编写
实施计划，再建立独立分支 `codex/event-joint-transformer-v2`。

## 目标

### 第一阶段：优先提高完成率

```text
Q = 0.6*CR + 0.2*PCR + 0.2*WCR
```

- 第一阶段只优化完成质量 `Q`。
- `TAT_s/PC_Wh/CS_paper` 仍完整记录，但不进入第一阶段 reward，也不作为否决项。
- Val Seen 和 Val Unseen 的 `Q` 均须比同场景 Stage3 baseline 至少提高 `0.5` 个
  百分点。
- 两个 split 上任一 `CR/PCR/WCR` 均不得下降。

### 第二阶段：再优化综合指标

第一阶段完整 Val 通过后，才从已通过 checkpoint 继续加入 TAT 和功耗代价，并把
终点目标切换为：

```text
-CS_paper
CS_paper = Q^(-1) + TAT_s/700 + PC_Wh/100
```

第二阶段不得以完成率下降换取表面 `CS_paper` 改善。

## 已确认架构

- [x] 复用 Stage3 的卫星、任务和关系表征权重作为 warm start。
- [x] 不复用旧的逐卫星独立动作头；重写事件状态层、自回归联合 Decoder、Actor、
  termination policy 和 centralized Critic。
- [x] 第一阶段冻结 Stage3 特征骨干；Val 8+8 通过后只解冻最后 1–2 层，学习率约为
  新模块的 `0.1x`。
- [x] Actor 和 Critic 都不读取 `is_visible`、未来轨迹或在线物理预测，也不在
  forward 中调用 Basilisk。
- [x] Critic 只比 Actor 多聚合完整星座的轻量联合状态；训练完成后可从推理路径移除。

## 显式状态

V2 必须显式使用：

- 上一全局任务和当前任务；
- 连续执行时长和最小承诺剩余时间；
- 最近 30/60 秒切换次数；
- 上次重规划时间和 termination 原因；
- 任务 release/due 剩余时间、要求观测时长和当前进度；
- 当前任务 owner 数及已锁定 owner；
- 当前事件类型和距离上次事件经过的物理时间。

上一任务通过 satellite-task edge 的 `is_previous_task/is_current_task` 关系表达，不使用
跨场景固定 task-id embedding。

## 事件与动作

### 承诺和 termination

- [x] 有效且仍处在最小承诺期内的任务硬锁定。
- [x] 任务完成、失败、到期、离开 ongoing 集合或新任务发布时立即复核。
- [x] 没有外部事件时，每 5 秒做一次 termination 安全复核。
- [x] 物理失效导致的强制中断不计入策略 log-prob；主动 termination 只能发生在
  最小承诺结束后。

### 最小承诺

- [x] 使用带物理掩码的 `{1,5,15,30,60}s` 离散动作。
- [x] 档位表示不可主动终止的最低锁定时间，不是最长执行时间。
- [x] 第一版仅当任务剩余要求观测时长不超过 1 秒时允许 `1s`；其他新任务至少锁定
  5 秒。

### 自回归联合分配

- [x] 待重规划卫星按事件紧迫度确定性排序，不固定按卫星 id，也不训练额外 ordering
  policy。
- [x] 每分配一颗卫星，就把已选任务、owner 数、承诺档位和边际协作状态写回上下文。
- [x] 后续卫星基于已经形成的部分联合动作继续选择，联合 log-prob 为各条件动作
  log-prob 之和。

### 软任务容量

- [x] 每个任务默认容量为 1。
- [x] 第二、第三个 owner 必须具有正的预测边际协作价值。
- [x] 同一任务安全上限为 3 个 owner，第四个 owner 永久 mask。
- [x] deterministic Val/Test 中新增 owner 相对最佳非重复动作的预测边际必须为正。
- [x] 不使用已经证明损害完成率的硬容量 1。

## Basilisk 和 event transition

每个在线环境只执行当前策略的一条真实 Basilisk 轨迹，不为每个候选额外生成反事实
分支。Basilisk 逐秒推进物理仿真，但完整 V2 Actor 只在事件点运行。

PPO 样本以事件为单位保存：

```text
(state,
 joint_action,
 behavior_log_prob,
 value,
 reward,
 delta_t,
 next_state,
 done,
 action_order,
 action_masks,
 owner_state,
 policy_version)
```

- [x] termination、task 和 minimum commitment 的 log-prob 全部进入联合概率。
- [x] learner 必须使用相同的卫星顺序、mask 和 owner 状态重放行为概率。
- [x] `delta_t` 进入 Critic 和 time-aware GAE，不能把 1 秒与 60 秒事件当成相同距离。

## Reward

使用现有 Evaluator 的 CR/PCR/WCR 任务权重定义 `omega_i`，令任务进度比例为：

```text
p_i(s) = clamp(progress_i / required_duration_i, 0, 1)
Phi(s) = sum_i omega_i * p_i(s)
```

第一阶段事件 reward：

```text
r_e = Phi(s_e+1) - Phi(s_e)
r_terminal += Q_final - Phi(s_terminal)
```

必须逐轨迹验证：

```text
sum(event_reward) = Q_final
```

- [x] 第一阶段使用 `gamma=1`，不因事件持续更久而折扣最终完成质量。
- [x] time-aware GAE 使用物理 `delta_t` 调整 eligibility。
- [x] 未完成任务的临时部分进度收益在终点被精确收回。
- [x] 第二阶段同样要求 `sum(event_reward) = -CS_paper`。

## 训练阶段

### V2-0：离线 warm start

- [ ] 最长约 4 小时。
- [ ] 加载 Stage3 表征权重。
- [ ] 在旧轨迹事件状态上蒸馏 Stage3 候选任务基础 logits。
- [ ] 从连续动作片段初始化 termination 和 minimum commitment。
- [ ] 使用旧轨迹最终 `Q` 的 event return 预训练 centralized Critic。
- [ ] 只验证不是随机策略，不以离线 loss 宣布性能提升。

### V2-1：同步 PPO 正确性

- [ ] 最长约 4 小时，使用少量 train scene 和并行环境。
- [ ] 验证 reward 精确重建。
- [ ] 验证联合 log-prob、mask、顺序和 owner 状态重放一致。
- [ ] 验证 Stage3 冻结参数逐值不变。
- [ ] 验证数值有限、事件时间严格推进、承诺必然终止。
- [ ] 验证 checkpoint、RNG 和第一批恢复动作可复现。
- [ ] 本阶段不要求完成率提高。

### V2-2：同步 PPO 收益

- [ ] 建议最多 12–16 小时。
- [ ] checkpoint 只根据固定 held-out train scenes 的 `Q` 和训练稳定性选取。
- [ ] 运行一个完整 3,600 秒 train scene smoke。
- [ ] 只运行一次 Val Seen/Unseen 8+8。
- [ ] 未通过完成率门槛则停止，不使用 APPO 掩盖同步策略问题。

### V2-3：APPO 扩展

- [ ] 只在同步 PPO 8+8 通过后启动。
- [ ] 解冻 Stage3 最后 1–2 层，使用约新模块 `0.1x` 学习率。
- [ ] 使用全部可申请 GPU、最多 120 个异步 Basilisk actor。
- [ ] 使用剩余约 24–28 小时预算。
- [ ] 保存 behavior log-prob 和 policy version。
- [ ] 使用 importance ratio、PPO clipping 和 policy-lag 上限；过旧样本直接丢弃。

## 资源边界

- [x] 通过 Slurm 申请当前节点全部可用 GPU。
- [x] 最多 120 个并行 Basilisk 环境。
- [x] 单次正式在线训练最长 48 小时。
- [x] 资源不足时只降低环境并行数，不改变 reward、动作定义或验收门槛。
- [x] 不在登录节点直接运行大规模仿真或训练。

## 验证顺序

```text
合成环境
-> 单场 3,600 秒 smoke
-> Val Seen 8 + Val Unseen 8
-> Val Seen 64 + Val Unseen 64
-> Test 一次
```

- [x] 任一级失败都停止扩大。
- [x] Test 只用于最终报告，不根据 Test 调 reward、模型或超参数。
- [x] 不通过降低门槛、挑选 scene 或重复扫描官方 Val 修饰结果。

## 稳定性与停止规则

以下任一情况必须停止更新、输出审计并回退到更新前 checkpoint：

- reward、return、advantage、log-prob 或 value 出现 NaN/Inf；
- behavior log-prob 与 learner 重放不一致；
- action order、mask、owner state 或 schema fingerprint 不一致；
- invalid-action count 非零；
- KL 超过预注册上限；
- 连续 minibatch 梯度异常；
- idle、单一任务或单一 commitment 档位持续塌缩；
- 事件时间不前进、事件率异常或承诺无法结束；
- APPO 样本超过 policy-lag 上限。

GPU OOM 只允许降低 learner minibatch 后重试一次；不得修改环境数、reward、动作定义
或验收门槛掩盖资源问题。

## Checkpoint 与回滚

独立保留：

```text
Stage3 baseline
V2 offline warm start
同步 PPO 最佳 checkpoint
APPO 最佳 checkpoint
```

checkpoint 必须保存模型、optimizer/scheduler、AMP、policy version、schema
fingerprint、normalizer、RNG、物理秒数、episode/event 数和 Encoder 解冻状态。

## 下一步

- [ ] 用户审阅并最终批准 V2 正式设计文档。
- [ ] 使用 writing-plans 编写分阶段实施计划。
- [ ] 创建独立分支 `codex/event-joint-transformer-v2`。
- [ ] 按测试驱动方式实现 V2-0，不直接启动正式 PPO/APPO。
- [ ] 合成环境和真实单场 smoke 通过后，才申请正式训练资源。
