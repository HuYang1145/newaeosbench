# V2 事件级联合 Transformer Actor-Critic 设计

> 状态：设计已在对话中逐节确认，等待用户审阅正式文档。
>
> 设计日期：2026-07-21
>
> 目标分支：`codex/event-joint-transformer-v2`

## 1. 目标与成功定义

V2 并行新建一个面向事件级星座联合调度的 Transformer Actor-Critic。它复用
Stage3-200k 已学习的卫星和任务表征，但不复用旧的逐卫星独立动作头，不覆盖现有
`JointModel`、checkpoint 或正式评估结果。

V2 的长期目标仍是降低论文口径 `CS_paper`：

```text
Q = 0.6*CR + 0.2*PCR + 0.2*WCR
CS_paper = Q^(-1) + TAT_s/700 + PC_Wh/100
```

训练分成两个目标严格对齐的阶段：

1. 第一阶段只优化完成质量 `Q`。`TAT_s/PC_Wh/CS_paper` 完整记录，但不进入
   reward，也不作为否决项。
2. 只有第一阶段完成率通过正式 Val 门槛后，第二阶段才从已通过 checkpoint 继续
   优化精确 `-CS_paper`。

第一阶段正式成功门槛为：

- Val Seen 和 Val Unseen 的 `Q` 都比同场景 Stage3 baseline 至少提高 `0.5` 个
  百分点；
- 两个 split 上任一 `CR/PCR/WCR` 都不得下降；
- 训练 loss、Critic accuracy、局部 reward 或单场结果不能替代上述门槛；
- Test 只在完整 Val 通过后运行一次，不用于调参。

## 2. 设计动机

现有实验已经把性能问题拆成四个层面：

1. **显式时间状态不足**：旧 Actor 没有无损使用上一任务、连续执行时长和近期切换
   历史。模型轨迹含大量几乎无观测收益的一秒短脉冲。
2. **监督与结果错位**：逐时刻动作 CE 只模仿专家动作，而专家本身含约
   `42%–45%` 重复冗余；事实 outcome 标签又不能判断未执行候选谁更好。
3. **长期信用分配不稳定**：把单步动作绑定到 3,600 秒最终结果会被后续动作淹没；
   M3 的 180/300 秒候选方向一致率也只有 `58.33%`。
4. **最终解码仍按卫星分解**：历史诊断中的重复冗余选择率约为 `47.12%`，但硬容量
   1 又显著损害完成质量，说明需要可学习的软容量联合决策。

V2 不假设增加 Transformer 层数就能自动修复这些问题。它同时重构状态、动作空间、
联合解码、事件 reward 和在线训练闭环。

## 3. 已选择路线与未选择路线

### 3.1 已选择：Stage3 表征热启动的事件级 V2

- 复用 Stage3 checkpoint 的任务、卫星及关系表征；
- 新建事件历史层、自回归联合 Decoder、termination policy、minimum commitment
  policy 和 centralized value Critic；
- 先离线 warm start，再进行同步 PPO，最后在门槛通过后切换 APPO；
- 接受较长 Basilisk 在线训练，以真实性能提升为第一目标。

### 3.2 未选择：继续修补旧 JointModel

该路线改动较小，但难以摆脱逐秒决策、旧动作 CE 和独立 `argmax` 的结构边界。历史
Temporal Adapter、二部图残差头和后处理实验都说明，仅在旧输出后追加小模块不能稳定
改变闭环行为。

### 3.3 未选择：完全从零训练

完全从零训练结构最自由，但会丢失 Stage3 已有的场景表示能力，显著增加 Basilisk
样本需求和 48 小时预算内的失败风险。V2 只重写决策与价值模块，不重新学习已经可复用
的底层卫星/任务表示。

## 4. 总体架构

```text
轻量在线状态
  -> Stage3 特征骨干
  -> 事件与历史状态层
     -> termination policy
     -> 自回归联合 Actor Decoder
     -> centralized state-value Critic
  -> Event Runtime
  -> Basilisk 黑盒环境
  -> event reward / next state
  -> PPO 或 APPO 更新
```

### 4.1 Stage3 特征骨干

骨干加载 Stage3-200k 的卫星、任务和关系表征权重，只负责产生 satellite tokens、
task tokens 和 satellite-task edge features。旧动作 logits 可用于 warm start 蒸馏，
但旧动作头不参与 V2 正式联合决策。

第一阶段冻结全部 Stage3 骨干参数。同步 PPO 的 8+8 Val 通过后，才允许解冻最后
1–2 层，并使用约新模块 `0.1x` 的学习率。完整 Val 通过前不全量解冻。

### 4.2 事件与历史状态层

V2 显式输入以下可部署状态：

- 每颗卫星上一全局任务、当前任务和当前承诺状态；
- 连续执行时长、最小承诺剩余时间、上次重规划时间；
- 最近 30/60 秒切换次数和上次 termination 原因；
- 卫星传感器类型、开关状态和已有动态特征；
- 任务 release/due 剩余时间、要求观测时长、当前进度和传感器类型；
- 当前任务 owner 数、已锁定 owner 及其连续执行状态；
- 本次事件类型和自上次事件经过的物理时间。

上一任务不使用跨场景固定 task-id embedding。它通过 satellite-task edge 上的
`is_previous_task/is_current_task` 关系和当前候选映射表达，避免不同场景 task id
没有共享语义的问题。

### 4.3 Termination policy

仍处在最小承诺期内的有效任务硬锁定，Actor 不得主动终止。超过最小承诺后：

- 任务完成、失败、到期、离开 ongoing 集合或新任务发布等外部事件立即复核；
- 没有外部事件时每 5 秒做一次安全复核；
- termination 输出 `keep/terminate` Bernoulli 动作及其 log-probability；
- 物理失效导致的强制中断不是策略动作，不计入 Actor log-probability；
- 正式推理中只有主动 termination 才进入策略行为统计。

### 4.4 自回归联合 Actor Decoder

Actor 只处理自然到期、强制中断、idle 唤醒或主动 termination 后需要重规划的卫星。
有效且仍锁定的承诺作为固定上下文保留。

待重规划卫星按以下确定性紧迫度排序：

1. 强制中断或当前任务已失效；
2. 当前兼容候选中最小的任务 deadline slack；
3. 等待重规划的物理秒数，等待更久者优先；
4. satellite id 仅作为最终稳定 tie-break。

每处理一颗卫星，Decoder 都把已选择任务、owner 数、承诺档位和边际协作状态写回
自回归上下文。后续卫星的条件概率因此依赖前面已经形成的联合分配，而不是并行独立
取最大值。

## 5. 动作空间与软任务容量

### 5.1 新任务动作

每个需要重规划的卫星输出：

```text
task categorical + minimum commitment categorical
```

task categorical 包括显式 idle。idle 永远合法；非空任务才预测 minimum
commitment。联合动作 log-probability 是所有 termination、task 和 commitment
条件 log-probability 的和。

### 5.2 最小承诺

非空任务使用 `{1,5,15,30,60}s` 离散最低锁定时间。它不是最长执行时间；锁定结束后
termination policy 可以继续保持任务。

第一版中 `1s` 只在任务按照当前进度计算的剩余要求观测时长不超过 1 秒时合法。
其他新任务至少锁定 5 秒，避免重新产生无约束一秒短脉冲。

### 5.3 软任务容量

- 每个任务默认容量为 1；
- 第二和第三个 owner 使用独立的条件边际协作分数；
- 第四个 owner 被物理 mask 永久屏蔽；
- 自回归任务分数由基础任务价值与当前 owner rank 的边际协作价值共同组成；
- deterministic Val/Test 中，新增 owner 相对最佳非重复动作的预测边际必须为正；
- PPO 训练保留随机探索概率，但所有行为 log-prob、owner 状态和 mask 必须随 transition
  保存，不能在 learner 端使用不同容量状态重算。

该设计允许有限冗余、接力和不确定性覆盖，同时禁止无界热门任务争抢。它不使用硬容量
1，也不把“降低重复率”本身当作 reward。

## 6. Actor、Critic 与 Basilisk 边界

### 6.1 Actor

Actor 只读取可部署的轻量状态，不读取 `is_visible`、未来轨迹、完整轨道传播结果或
在线物理预测。Actor forward 不调用 Basilisk。

### 6.2 Centralized Critic

Critic 是 PPO/APPO 的 state-value 网络，不是 M3 的候选偏好裁判模型。它读取与
Actor 同源的轻量状态，但可以聚合完整星座、所有任务、锁定承诺和当前 owner 状态，
输出事件状态标量 `V(s)`。

Critic 不使用 Actor 部署时不可得的 privileged simulator state。训练完成后 Critic
可从正式推理路径移除。

### 6.3 Basilisk

Basilisk 只作为黑盒环境：接收一次真实联合动作，逐秒执行到下一个事件，返回下一
状态和 reward 所需事实结果。每个并行环境只运行当前策略的一条轨迹，不为每个候选
动作额外生成反事实分支。

不可对 Basilisk 反向传播不构成 PPO 障碍。梯度只通过保存的动作 log-probability、
value 和 entropy 返回 V2 参数。

## 7. Event transition 与概率重放

每个 PPO 样本为：

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

其中 `delta_t` 是到下一个事件经过的真实物理秒数。两次事件之间 Basilisk 继续逐秒
推进，但不重复调用完整 Actor。

联合行为概率必须按原顺序重放：

```text
log pi(a|s)
  = sum(log p(termination_i | prefix_i))
  + sum(log p(task_j | prefix_j))
  + sum(log p(commitment_j | task_j, prefix_j))
```

learner 重算时必须复用行为侧保存的物理 mask、卫星顺序和 owner 状态。任何一项不
一致都使样本无效。

## 8. 第一阶段 completion reward

### 8.1 完成质量权重

使用现有 Evaluator 的 CR/PCR/WCR 任务权重，为每个任务导出非负权重 `omega_i`，
使终局完成指标可以写成：

```text
Q_final = sum_i omega_i * completed_i
```

任务当前进度比例为：

```text
p_i(s) = clamp(progress_i / required_duration_i, 0, 1)
```

完成质量 potential 定义为：

```text
Phi(s) = sum_i omega_i * p_i(s)
```

它只依赖 TaskManager 已提供的当前进度，不使用未来可见性。

### 8.2 事件 reward 与终点校正

非终止事件 reward：

```text
r_e = Phi(s_{e+1}) - Phi(s_e)
```

终局额外加入：

```text
r_terminal += Q_final - Phi(s_terminal)
```

因此每条完整轨迹必须满足：

```text
sum_e r_e = Q_final
```

部分进度为策略提供早期信号，但未完成任务的临时收益会在终点被精确收回。第一阶段
不加入 TAT、功耗或切换惩罚；这些指标只记录，不参与梯度。

### 8.3 半马尔可夫 GAE

第一阶段最终目标只关心 3,600 秒结束时的完成质量，因此使用 `gamma=1`，不因动作
持续更久而折扣完成价值。TD residual 为：

```text
delta_e = r_e + V(s_{e+1}) - V(s_e)
```

GAE 使用物理时间一致的 eligibility：

```text
lambda_e = 0.95 ** (delta_t_e / 5)
```

终局 `V=0`。`delta_t` 同时作为 Critic 输入和审计字段，1 秒事件与 60 秒事件不能
被当作相同时间距离。

## 9. 第二阶段 CS_paper reward

第二阶段只能从第一阶段完整 Val 已通过的 checkpoint 启动。它继续使用完成进度
potential，同时把事件期间新增的 TAT 和传感器功耗代价纳入 dense reward，并在终局
加入精确校正，使整条轨迹满足：

```text
sum_e r_e = -CS_paper
```

第二阶段不回头修改第一阶段完成率结论。若 `CS_paper` 改善伴随任何完成率跌破第一
阶段已通过门槛，回退到第一阶段 checkpoint，而不是接受质量退化。

## 10. PPO/APPO 目标

PPO 优化目标由 clipped policy surrogate、value loss 和 entropy bonus 组成：

```text
L = L_policy_clip + c_v * L_value - c_e * entropy
```

完成率不是可直接反向传播的分类 loss。`Q_final` 通过 event reward、return 和
advantage 进入 policy gradient；因此 PPO loss 下降不能替代 Basilisk Val 完成率。

同步 PPO 与 APPO 共用：

- 同一 V2 模型与动作空间；
- 同一 event transition schema；
- 同一 reward 和 time-aware GAE；
- 同一 checkpoint 格式；
- 同一正式评估协议。

APPO 只增加异步 actor/learner、importance ratio、policy version 和 policy-lag
控制，不改变任务定义。

## 11. 四阶段训练流程

### V2-0：离线 warm start

最长约 4 小时：

- 加载 Stage3 表征骨干；
- 在旧轨迹事件状态上蒸馏 Stage3 候选任务基础 logits；
- 从连续动作片段初始化 termination 和 minimum commitment；
- 使用旧轨迹最终 `Q` 的 event return 预训练 centralized value Critic；
- 自回归 owner 和边际协作参数保持独立，不用专家重复动作定义全局最优容量。

该阶段只避免随机策略，不以离线 loss 宣布性能提升。

### V2-1：同步 PPO 正确性

最长约 4 小时，使用少量 train scene 和并行环境。必须验证：

- reward 精确重建；
- 联合 log-prob、mask、顺序和 owner 状态重放一致；
- 冻结参数逐值不变；
- PPO ratio、KL、entropy、value、advantage 和 gradient 全部有限；
- checkpoint 恢复后 RNG 和第一批动作可复现；
- 事件时间严格推进，承诺能够终止，不存在无限事件循环。

本阶段不要求完成率提高。

### V2-2：同步 PPO 收益

建议最多 12–16 小时。checkpoint 只根据固定 held-out train scenes 的 `Q` 和训练
稳定性选取，不反复使用官方 Val 选模型。

候选依次运行单场 3,600 秒 smoke 和一次 Val 8+8。未通过第一阶段完成率门槛时停止，
不得使用 APPO 扩样掩盖同步策略问题。

### V2-3：APPO 扩展

只在同步 PPO 8+8 通过后启动：

- 解冻 Stage3 骨干最后 1–2 层，学习率约为新模块的 `0.1x`；
- 使用全部可申请 GPU；
- 最多运行 120 个异步 Basilisk actor；
- 使用剩余约 24–28 小时训练预算；
- transition 保存 behavior log-prob 与 policy version；
- learner 使用 importance ratio、PPO clipping 和 policy-lag 上限；
- 超过预注册 policy-lag 的样本直接丢弃。

APPO 结束后必须重新通过 8+8，才进入完整 64+64 Val。

## 12. 资源与 Slurm 边界

- 正式训练通过 Slurm 申请当前节点全部可用 GPU；
- Basilisk 并行环境最多 120 个；
- 单次正式在线训练最长 48 小时；
- 资源不足时只降低并行环境数，不改变 reward、动作定义或验收门槛；
- 不在登录节点直接运行大规模仿真或训练；
- 训练、actor、learner、评估和恢复日志使用独立路径；
- Slurm preemption 后只能从 schema fingerprint 一致的 checkpoint 恢复。

## 13. 动作合法性与事件循环保护

### 13.1 物理合法 mask

- 完成、失败、到期或不在 ongoing 集合的任务不可选；
- 传感器类型不兼容的卫星—任务边不可选；
- 仍在最小承诺期内的卫星不可主动重规划；
- 已有 3 个 owner 的任务不可增加 owner；
- 不满足剩余观测时长条件时屏蔽 `1s` 承诺；
- idle 永远合法。

mask 只使用在线轻量状态，不调用 Basilisk 预测。未被 mask 的 logits 出现 NaN/Inf
时立即停止更新，不能用有限值替换静默掩盖模型错误。

### 13.2 事件时间

- 同一物理时刻的外部事件合并为一次联合决策；
- `next_event_time` 必须严格大于当前时间；
- 没有外部事件时，下次安全复核不超过 5 秒；
- 物理任务失效可立即强制中断；
- Actor 主动 termination 只能发生在最小承诺结束后；
- 每个承诺记录开始、结束、实际时长和终止原因；
- 事件率超过预注册上限时输出 trace 并停止，不继续消耗正式预算。

## 14. 数值稳定与停止规则

每次更新记录：

```text
policy loss, value loss, entropy, approx KL, clip fraction,
gradient norm, advantage mean/std, value explained variance,
invalid-action count, policy version lag
```

立即停止或回退条件：

- reward、return、advantage、log-prob 或 value 出现 NaN/Inf；
- behavior log-prob 与 learner 重放不一致；
- action order、mask、owner state 或 schema fingerprint 不一致；
- KL 超过预注册上限时停止当前 epoch，并回退到更新前 checkpoint；
- 连续 minibatch 梯度异常；
- idle、单一任务或单一 commitment 档位持续塌缩；
- invalid-action count 非零；
- 事件死循环或事件率异常；
- APPO 样本超过 policy-lag 上限。

GPU OOM 只允许降低 learner minibatch 后重试一次，不得修改环境数、reward、动作定义
或门槛来掩盖资源问题。再次 OOM 则停止该 job。

## 15. Checkpoint、指纹与回滚

每个 checkpoint 保存：

- Actor、Critic 和 Stage3 骨干参数；
- optimizer、scheduler 和 AMP 状态；
- policy version；
- observation、action、reward 和 transition schema fingerprint；
- normalizer；
- Python、NumPy、PyTorch 和环境 RNG 状态；
- 已消费物理秒、环境 episode 数和 event transition 数；
- 当前训练阶段与 Encoder 解冻状态。

独立保留四类恢复点：

```text
Stage3 baseline
V2 offline warm start
同步 PPO 最佳 checkpoint
APPO 最佳 checkpoint
```

阶段失败时回到上一已通过恢复点。V2 使用独立模型名、配置、checkpoint、日志和评估
目录，不覆盖 Stage3、M2/M3 产物或现有正式 Val/Test 输出。

## 16. 测试与验证

### 16.1 单元测试

- 事件紧迫度排序和 tie-break；
- 自回归联合 log-prob；
- owner 容量 1/2/3 和边际协作条件；
- minimum commitment mask；
- termination 合法时机；
- reward telescoping 与独立重建；
- time-aware GAE；
- APPO stale sample 丢弃；
- checkpoint schema fingerprint。

### 16.2 合成环境

在不运行 Basilisk 的小型已知最优环境中验证：

- 同步 PPO 能学会联合分配；
- termination 与最小承诺能学会已知最优策略；
- APPO policy lag 控制有效；
- 保存/恢复后动作序列一致。

### 16.3 真实 Basilisk smoke

- 固定 train scene 和随机种子；
- 完整运行 3,600 秒；
- 无非法动作、无事件死循环；
- Actor/Critic forward 不触发额外 Basilisk；
- Stage3 冻结参数逐值不变；
- reward 可以从原始日志独立重建。

### 16.4 正式行为验收

```text
单场 smoke
-> Val Seen 8 + Val Unseen 8
-> Val Seen 64 + Val Unseen 64
-> Test 一次
```

任一级失败都停止扩大，不通过降低门槛、挑选 scene、扫描官方 Val 或使用 Test 修饰
结果。

## 17. 非目标

V2 第一版明确不做：

- 在 Actor/Critic forward 中调用 Basilisk、轨道传播或重型几何预测；
- 使用 `is_visible` 或未来轨迹作为网络输入；
- 完全从零训练卫星/任务表征；
- 为每个在线候选额外运行反事实分支；
- 硬容量 1；
- 每秒运行完整联合 Actor；
- 用训练 loss 下降替代正式完成率；
- 在完整 Val 通过前运行 Test；
- 覆盖或删除 Stage3、M2/M3 和历史负实验资产。

## 18. 交付与实施边界

本设计批准后才编写实施计划。实施必须在独立分支
`codex/event-joint-transformer-v2` 上分阶段进行，并为每一阶段保留可运行测试、
恢复点、Slurm 包装和结果审计。

设计批准不等于授权立即占用 48 小时正式资源。正式 PPO/APPO 训练只能在合成环境、
真实单场 smoke 和同步 PPO 正确性门槛通过后启动。
