# Transformer V2 待办事项

## 当前状态

当前唯一活动主线是新建 **V2 事件级联合 Transformer Actor-Critic**。V2 与现有
Stage3 `JointModel` 并行存在，不覆盖 Stage3 checkpoint、M2/M3 结果或正式评估
输出。

设计和 V2-0 foundation 已经完成，正式设计与实施计划为：

`docs/superpowers/specs/2026-07-21-event-joint-transformer-v2-design.md`

`docs/superpowers/plans/2026-07-21-event-joint-transformer-v2-foundation.md`

当前工作分支为 `codex/offline-critic-ranking`。V2-0 离线 warm start、V2-1 同步
PPO 正确性、V2-2 同步 PPO 收益训练和唯一一次 Basilisk Val Seen/Unseen 8+8 门槛
均已完成。V2-2 selected checkpoint 在两个 split 的 `Q` 分别提高 `2.79` 和
`3.18` 个百分点，且 `CR/PCR/WCR` 全部上升，已经获得第一份正式完成率提高证据。
当前自动进入 V2-3 APPO 扩展。

离线验收设计和实施记录为：

`docs/superpowers/specs/2026-07-22-event-v2-unseen-offline-acceptance-design.md`

`docs/superpowers/plans/2026-07-22-event-v2-unseen-offline-acceptance.md`

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

现有 `CompletionRateEvaluator` 中 `PCR` 包含未完成任务的终点部分进度，因此精确
终点不能写成只依赖 `completed_i` 的加权和。V2 直接重建：

```text
Q_final = 0.6*mean(completed_i)
        + 0.2*mean(progress_ratio_i)
        + 0.2*sum(duration_i*completed_i)/sum(duration_i)
```

dense potential 使用 `omega_i = 0.8/N + 0.2*duration_i/sum(duration)`，令任务进度
比例为：

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
- [x] 未完成任务的 CR/WCR 代理进度在终点被收回，同时保留 Evaluator 中真实 PCR
  的 `0.2` 部分进度权重。
- [x] 第二阶段同样要求 `sum(event_reward) = -CS_paper`。

## 训练阶段

### V2-0：离线 warm start

- [x] V2-0 模型、事件数据、loss、checkpoint 与 Slurm 代码完成。
- [x] 兼容缺少后来新增 `_duration_head` 的 legacy Stage3-200k checkpoint；其他缺键
  继续严格拒绝。
- [x] 在旧轨迹事件状态上蒸馏 Stage3 候选任务基础 logits。
- [x] 从连续动作片段初始化 termination 和 minimum commitment。
- [x] 使用从当前 potential 到精确终点 `Q` 的 event return 预训练 centralized
  Critic。
- [x] owner marginal head 不使用专家重复 owner 作为正监督；旧专家超过 3 个 owner
  的状态只饱和记录为 3。
- [x] 一次真实 CPU forward/backward/optimizer/checkpoint preflight 通过。
- [x] 通过 Slurm 完成正式 10k GPU warm start；job `915` 用时 `01:06:44`、
  `COMPLETED 0:0`，保存 1k–10k 共 10 个恢复点。
- [x] 对 warm-start checkpoint 完成未见轨迹离线验收；job `965` 使用固定 seed
  `3407`、同一 Stage3-200k backbone、同一批 64 场 `val_unseen` 事实事件，
  `COMPLETED 0:0`，用时 `00:22:55`。
- [x] 离线验收严格通过：加权 `total` 从 `4.797582` 降至 `0.751693`
  （下降 `84.33%`）；`task_distillation` 从 `2.662716` 降至 `0.257852`，
  `termination` 从 `0.618425` 降至 `0.00001289`，`commitment` 从 `1.495417`
  降至 `0.491663`，`value` 从 `0.0210251` 降至 `0.00216482`，四个分量均严格下降。
- [x] 四类事实 support 分别为 `26073/510846/18815/26931`；64 个 scene id 与
  annotation 原顺序逐值一致，全部数值有限，checkpoint/schema/config 指纹匹配，
  未调用 Basilisk、未读取 Test。
- [x] 显存探针改用固定最坏 shape 的 scene index `36`；batch `512` probe 峰值
  reserved `89.61%`，正式 64 场峰值 allocated `21.999 GB`、reserved `24.142 GB`
  （`95.61%`），使用 BF16、SDPA、`inference_mode`、pinned transfer 和
  `expandable_segments`。首次 job `947` 因外部作业中途抢占显存而 OOM，不作为模型
  结果；修复探针和 allocator 后 job `965` 成功。
- [x] 本验收只证明 10k warm start 明显优于随机 V2 初始化，不能写成完成率提高；下一
  步仍必须进入 V2-1 同步 PPO 正确性阶段。

### V2-1：同步 PPO 正确性

- [x] 首次正式 Slurm smoke job `1018` 已完成两级 preflight，但 4 场正式训练在首个
  update 被正确拒绝：BF16 行为采样使用 `batch=1`，learner 曾把同 shape 事件合并为
  batch，联合 log-prob 最大误差 `0.15719557`。未放宽 `1e-6` 门槛；已改为行为采样、
  校验和 learner 全部逐事件 `batch=1`，并增加回归测试。
- [x] 修复后的正式 Slurm smoke 首次提交为 job `1027`，因 `server-10` 的 4 张 GPU
  均被 job `1016_4`–`1016_7` 占用而保持排队，未开始执行；现按实测内存峰值把申请量
  安全下调到 96 GiB，并重提为 job `1029`。job `1029` 的合成和 60 秒真实 BF16
  GPU preflight 均通过，其中真实 preflight 为 12 个事件、52 物理秒，
  `logprob_replay_max_error=0`、`reward_reconstruction_max_error=0`、冻结参数变化数
  为 0、checkpoint 第一动作可复现；正式首个 update 在后续 PPO epoch 触碰 KL
  `0.03` 上限时被旧逻辑整体回滚。
- [x] 已将 KL 门槛修正为严格 early-stop：后续 epoch 触碰上限时只回滚该 epoch，
  保留此前满足上限的 epoch；若第一个 epoch 即超限仍拒绝整个 update，最终全 rollout
  KL 仍必须不超过 `0.03`，没有放宽阈值。job `2016` 进一步证明 64 事件首轮内部会
  出现高方差局部 minibatch KL；现已改为立即停止后续 minibatch，并以完整 rollout
  KL 决定保留或回滚当前部分 epoch，仍不允许完整 KL 超限。修复后的正式作业为
  job `2023`。job `2023` 进一步确认局部停止后的完整 rollout KL 也超限，因此回滚
  正确；根因是 64 事件 rollout 配 `minibatch_events=4` 会在首轮执行 16 次优化。
  job `2028` 测试时先保持学习率 `3e-5` 和 KL `0.03` 不变，将 minibatch 提高到
  16、每轮降为 4 次优化，同时提高 GPU 利用率；实测完整 KL 为 `0.038289208`，
  4 个 minibatch 全部执行，说明 batch
  调整有效但 `3e-5` 仍略激进；依据实测将学习率保守校准为 `2e-5`，KL 上限和其他
  PPO 定义不变。
  配置为 `server-10/local-10`，1 GPU、24 CPU、96 GiB、上限 `04:00:00`；依次运行
  合成 CPU preflight、scene 0 的 60 秒 BF16 GPU
  preflight、4 个 train scene 的 3,600 秒同步 PPO。日志：
  学习率校准后的重试为 job `2029`，日志：
  `work_dirs/eval_logs/event_v2_sync_ppo_2029.log`；输出：
  `work_dirs/event_joint_transformer_v2/v2_1_sync_ppo/`。
- [x] job `2029` 在 35 分 24 秒内稳定完成预设的 64 updates、4,096 events，
  `logprob_replay_max_error=0`、冻结参数变化数为 0、数值全部有限、checkpoint 第一
  动作可复现；但四场仅推进到 `2309/2232/2194/3280` 秒，未达到 3,600 秒，因此
  `accepted=false` 和 Slurm exit `2:0` 是完整性门槛，不是训练崩溃。已增加
  cosine scheduler 到达第 64 步后固定在 `eta_min` 的回归测试，准备从
  `checkpoint_update_000064.pth` 续跑至最多 104 updates。续跑已提交为 job
  `2190`，日志：`work_dirs/eval_logs/event_v2_sync_ppo_resume_2190.log`。
- [x] job `2190` 已 `COMPLETED 0:0`，用时 `00:26:07`；最终在 update `101`
  完成四个 3,600 秒 train scene，共 `6,423` events、`14,333` 计入 reward 的物理秒。
  最终 checkpoint：`checkpoint_update_000101.pth`。
- [x] 总运行时间小于 4 小时，只使用 4 个 train scene 和 1 张 GPU。
- [x] reward 精确重建，最大误差 `1.4551915228366852e-11`。
- [x] 联合 log-prob、mask、顺序和 owner 状态重放一致，最大误差 `0`。
- [x] Stage3 冻结参数变化数为 `0`。
- [x] 数值全部有限，事件时间违规、无效动作和未终止承诺计数均为 `0`。
- [x] checkpoint、RNG 和第一批恢复动作可复现。
- [x] 本阶段只通过同步 PPO 正确性门槛，不宣称完成率提高；下一步进入 V2-2。

### V2-2：同步 PPO 收益

- [x] 已新增 V2-1 → V2-2 安全 bootstrap：只继承 model/optimizer，不继承旧场景
  runtime、计数器或 RNG；V2-2 checkpoint 使用独立 stage/config/scene 指纹并保持
  精确恢复。相关 V2 回归共 `135 passed`。
- [x] 真实 BF16 bootstrap smoke job `2194` 已 `COMPLETED 0:0`，scene 4 运行
  120 秒、1 update、15 events；`accepted=true`、reward/log-prob 最大误差均为
  `0`、冻结参数变化数为 `0`、checkpoint 第一动作可复现。日志：
  `work_dirs/eval_logs/event_v2_2_smoke_2194.log`；输出：
  `work_dirs/event_joint_transformer_v2/v2_2_sync_ppo/smoke_2194/`。
- [x] 正式全预算同步训练 job `2203` 已 `COMPLETED 0:0`，用时 `10:29:57`：
  `local-10`、当前全部 3 张可用 GPU、96 CPU、160 GiB。四个 replica 使用独立
  seed，分别训练 scenes `4–51`、`52–99`、`100–147`、`148–195`；固定 held-out
  train scenes 为 `196–203`，训练期间未访问 Val/Test。四个 replica 均为
  `accepted=true`、48/48 scenes 完成、数值有限、reward 重建误差小于
  `5e-10`、log-prob 重放误差为 `0`、冻结参数变化为 `0`、checkpoint 第一动作
  可复现；最终 updates 分别为 `1046/950/924/914`。父日志：
  `work_dirs/eval_logs/event_v2_2_full_2203.log`；replica 日志位于
  `work_dirs/event_joint_transformer_v2/v2_2_sync_ppo/replica_{0,1,2,3}/train_2203.log`。
- [x] 运行时间满足最多 12–16 小时预算。
- [x] 已实现固定 held-out train checkpoint selector；完整 scene 196 确定性评估
  smoke job `2211` 已 `COMPLETED 0:0`，用时 `00:01:17`。V2-1 的
  `CR/PCR/WCR/Q=0.08676/0.09813/0.08974/0.08963`，V2-2 replica 0 为
  `0.12329/0.14185/0.12643/0.12763`，四项均上升且 reward 重建误差小于
  `3e-11`。首次 job `2208` 因 GPU 0 被 Slurm 外部进程占满而 OOM，已改为申请
  完整设备可见性并自动排除显存占用超过 4 GiB 的物理卡，未改变模型、场景或评估
  协议。
- [x] 五个最终 checkpoint × 固定 held-out train scenes `196–203` 的正式确定性
  比较 job `2212` 已 `COMPLETED 0:0`，用时 `00:28:45`。V2-2 replica 0 以
  `Q=0.311805` 入选，相对 V2-1 的 `Q=0.287824` 提高 `2.398` 个百分点；
  `CR/PCR/WCR` 分别提高 `2.519/1.921/2.511` 个百分点。选择结果：
  `work_dirs/event_joint_transformer_v2/v2_2_heldout/heldout_2212/selection.json`。
- [x] selected checkpoint 的完整 train scene 204 smoke job `2213` 已
  `COMPLETED 0:0`，用时 `00:04:08`；`Q=0.498340`、reward 重建误差
  `2.91e-11`、数值有限。
- [x] 唯一一次 Val Seen/Unseen 8+8 的原 job `2214` 在评估中途收到外部
  `SIGTERM`，无 Python 异常、无 OOM、未生成 gate，不能作为模型结果。4 张 GPU
  空闲后按同一 checkpoint、场景和协议原样重跑 job `2225`，已
  `COMPLETED 0:0`，用时 `00:36:10`。
- [x] Val Seen：`Q 0.405210 -> 0.433100`，提高 `2.789` 个百分点；
  `CR/PCR/WCR` 分别提高 `2.863/2.457/2.899` 个百分点。
- [x] Val Unseen：`Q 0.260397 -> 0.292165`，提高 `3.177` 个百分点；
  `CR/PCR/WCR` 分别提高 `3.237/3.213/2.962` 个百分点。
- [x] 两个 split 均超过预注册的 `0.5` 个百分点门槛，且任一
  `CR/PCR/WCR` 均未下降；`gate.json` 的 `passed=true`。允许进入 V2-3。

### V2-3：APPO 扩展

- [x] 只在同步 PPO 8+8 通过后启动。
- [x] 按
  `docs/superpowers/plans/2026-07-24-event-v2-appo-expansion.md`
  完成 V2-3 policy-lag、learner、共享 policy、actor 进程、checkpoint/恢复和 Slurm
  入口；V2 全量相关回归 `168 passed`，合成 APPO `accepted=true`。
- [x] 解冻 Stage3 Encoder/Decoder 最后各 1 层；新模块学习率 `1e-6`，解冻层
  学习率 `1e-7`，其余 Stage3 参数继续逐值审计。
- [x] 保存 behavior log-prob 和 policy version；learner 使用 importance ratio、
  PPO clipping 和 `max_policy_lag=2`，过旧事件直接丢弃。
- [x] 首次真实 smoke job `2226` 在 spawn 前发现旧 Transformer 内部局部
  `lambda` 不可 pickle；共享方式已从完整 Module 改为仅共享 CPU 参数 tensors。
  第二次 job `2227` 完成 13 次更新后暴露 actor 过早退出导致最后队列 tensor 的
  resource sharer 消失；已增加 learner acknowledgement，两个失败均未改模型或门槛。
- [x] 修复后的完整 train scene 205 smoke job `2228` 已 `COMPLETED 0:0`，
  用时 `00:07:46`、`accepted=true`：14 updates、1,671 accepted events、
  96 stale events 被按预注册上限丢弃，reward 重建误差 `0`、actor log-prob
  重放误差 `0`、冻结参数变化 `0`、checkpoint 第一动作可复现。
- [x] 正式 V2-3 job `2229` 已 `COMPLETED 0:0`，用时 `07:29:34`：
  固定 train scenes `205–324` 共 120 场全部完成，832 updates、
  106,422 accepted events、57,026 stale events 按 `max_policy_lag=2` 丢弃；
  `accepted=true`、reward 重建最大误差 `<5e-10`、actor log-prob 重放误差 `0`、
  冻结参数变化 `0`、checkpoint 第一动作可复现。最终 checkpoint：
  `work_dirs/event_joint_transformer_v2/v2_3_appo/full_2229/checkpoint_update_000832.pth`。
  日志：
  `work_dirs/eval_logs/event_v2_appo_full_2229.log`；输出：
  `work_dirs/event_joint_transformer_v2/v2_3_appo/full_2229/`。
- [x] V2-2 与 V2-3 的同场景 Val Seen/Unseen 8+8 对照 job `3170` 已完成；
  比较脚本因 `passed=false` 按设计返回退出码 `2`，不是仿真或加载故障。
  Val Seen 的 `Q 0.433100 -> 0.402062`，下降 `3.104` 个百分点，
  `CR/PCR/WCR` 分别下降 `3.083/3.015/3.255` 个百分点；Val Unseen 的
  `Q 0.292165 -> 0.288264`，下降 `0.390` 个百分点，`CR/PCR/WCR`
  分别下降 `0.233/1.102/0.150` 个百分点。两个 split 均未通过门槛，因此停止
  Val 64+64，不运行 Test，保留 V2-2 replica 0 为当前最佳 checkpoint。日志：
  `work_dirs/eval_logs/event_v2_appo_val8_3170.log`；输出：
  `work_dirs/event_joint_transformer_v2/v2_3_val8/val_3170/`。
- [x] 为排查最终小批量 update 是否单独损坏模型，使用唯一保留下来的早期
  `checkpoint_latest.pth`（V2-3 update 800）在未访问官方 Val/Test 的固定
  train-heldout scenes `196–203` 上完成诊断 job `3173`，用时 `00:28:11`。
  相对 V2-2 replica 0，`Q 0.311805 -> 0.308587`，下降 `0.322` 个百分点；
  `CR/PCR/WCR` 分别下降 `0.436/0.170/0.131` 个百分点。因此 update 800
  仍未超过 V2-2，最终 update 832 不是唯一问题。update `100–700` 已被
  `checkpoint_latest.pth` 覆盖，无法追溯，后续训练必须独立保留周期 checkpoint。
  输出：
  `work_dirs/event_joint_transformer_v2/v2_3_checkpoint_diagnostic/u800_3173/`。

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

V2-0 单步 preflight checkpoint 已验证上述字段，实际大小约 `367 MiB`。正式阶段应按
1k 间隔独立保留恢复点，并预留 optimizer/多 checkpoint 的额外空间。

## 下一步

- [x] 用户审阅并最终批准 V2 正式设计文档。
- [x] 使用 writing-plans 编写 V2-0 foundation 实施计划。
- [x] 创建独立分支 `codex/event-joint-transformer-v2`。
- [x] 按测试驱动方式实现 V2-0 foundation，未直接启动 PPO/APPO。
- [x] 提交并完成 V2-0 正式 10k GPU warm start。
  - 2026-07-22 首次提交的 Slurm job `895` 因单 GRES 固定绑定到已被外部进程占满的
    物理 GPU 0，在模型搬入 CUDA 时 OOM 退出；模型训练尚未开始。
  - 最小 Slurm 诊断确认申请两张 GRES 后可在分配范围 `0,1` 内单独使用空闲 GPU 1；
    job `905` 随后暴露 `delta_t.expand()` 的 pinned-memory 重叠问题，已由 commit
    `8be19de` 修复。
  - job `906` 的 CUDA preflight 通过，但正式 batch 8 暴露 FP16 梯度溢出；FP32/BF16
    对照均连续两步有限，正式配置已由 commit `be639a1` 切换到 BF16。
  - 正式 10k 重试 job `915` 已完成（`COMPLETED 0:0`，用时 `01:06:44`）；最终
    checkpoint 为 `checkpoint_step_010000.pth`。
  - 最终 checkpoint 的 447 个模型 tensor 和 126 个 optimizer 浮点 tensor 均有限；
    1k→5k 间 397 个冻结骨干 tensor 逐值不变，50 个 V2 tensor 中 42 个已更新。
  - 训练日志的前 1k/后 1k 窗口平均 total loss 为 `1.3368/0.5957`；该结果只证明
    离线 warm start 收敛，不代表 CR/PCR/WCR 已提高。
  - 日志：`work_dirs/eval_logs/event_v2_warm_start_915.log`。
  - checkpoint 目录：`work_dirs/event_joint_transformer_v2/v2_0_warm_start/`。
- [x] 对 10k checkpoint 完成固定 `val_unseen` 轨迹离线验收。
- [x] 已写同步 PPO/Event Runtime 实施计划，并通过 CPU 合成闭环与 scene 0 的 10 秒
  真实 smoke；job `1018` 在正式 BF16 首个 update 暴露并阻断了批量形状导致的
  log-prob 重放误差，修复重试通过后才进入 V2-2，不直接进入 Val 8+8。

## 追加主线：V2-2 大规模同步 PPO

目标是在当前最佳
`work_dirs/event_joint_transformer_v2/v2_2_sync_ppo/replica_0/checkpoint_update_001046.pth`
基础上扩大同步 PPO，而不是继续使用当前 APPO 配置。训练期间所有采样器必须使用同一
policy version；每轮收集结束后统一更新并广播新权重，不产生 stale rollout，也不丢弃
过期事件。

### 冻结与训练边界

- [x] 保持 Stage3 Transformer Encoder、Decoder、TimeModel 和旧约束模块冻结，
  不再解冻尾层，避免破坏 V2-2 已验证有效的旧表征。
- [x] 继续训练约 `1.67M` 个 V2 参数：卫星/任务 edge projection、
  `EventStateEncoder`、自回归联合 Actor 和 centralized Critic。Actor 包括
  termination、idle、task value、owner marginal、commitment 和 prefix update；
  Critic 学习事件状态价值。冻结骨干不等于只训练 loss，也不等于模型不再学习。
- [x] 启动前逐项记录总参数、可训练参数及其名称；训练后要求冻结参数逐值变化数为
  `0`。

### 大规模同步采样与训练

- [x] 在现有 V2-2 同步 PPO 基础上实现单一 policy 的多采样器同步轮次：最多使用
  4 张 GPU、96–120 个 CPU Basilisk 环境；所有采样器完成固定 event chunk 后进入
  barrier，learner 聚合完整 batch 后更新一次。
- [x] 固定 train scenes `205–324`，不访问 Test；从 V2-2 replica 0 只继承模型和
  optimizer 的兼容状态，不继承旧 runtime、计数器或 RNG。
- [x] 第一轮保持 V2-2 已验证的 `clip_ratio=0.2`、`max_kl=0.03`、
  `gamma=1.0`、time-aware GAE 和精确终点 Q 校正；不同时修改 reward、动作定义和
  Transformer 解冻范围。
- [ ] 至少运行 2 个独立 seed。资源不足时减少并行环境数，不把异步采样或放宽
  policy-lag 作为替代。
- [ ] 先完成合成同步 barrier 测试和单场 3,600 秒 smoke；只有 reward 重建误差、
  behavior log-prob 重放误差、invalid action、冻结参数审计和 checkpoint 恢复全部
  通过，才提交正式 120 场训练。
  - 2026-07-25：合成 preflight 已通过，`accepted=true`、2 updates、16 events、
    stale rollout 为 `0`，reward/log-prob 最大误差均为 `0`，冻结参数变化数为
    `0`；输出位于
    `work_dirs/event_joint_transformer_v2/v2_2_large_sync_ppo/preflight_65b970b/`。
  - 真实 3,600 秒 smoke 已提交 Slurm job `3254`；日志为
    `work_dirs/eval_logs/event_v2_large_sync_smoke_3254.log`，输出目录为
    `work_dirs/event_joint_transformer_v2/v2_2_large_sync_ppo/smoke_3254/`。

### Checkpoint 与选择

- [x] 每 `100` 次 update 永久保存独立 checkpoint，禁止只覆盖
  `checkpoint_latest.pth`；同时保存 optimizer/scheduler、AMP、RNG、policy version、
  场景进度、事件数和物理秒数。
- [ ] 只使用固定 train-heldout scenes `196–203` 对不同 seed 和周期 checkpoint
  排序，选择指标仍为 `Q=0.6CR+0.2PCR+0.2WCR`；不得反复扫描官方 Val 选择训练轮次。
- [ ] 将官方 Val Seen/Unseen scenes `0–7` 视为已经使用过的历史诊断集。新模型只在
  checkpoint 锁定后使用尚未访问的 scenes `8–15` 做一次 8+8 gate。

### 成功门槛与资源预算

- [ ] 相对当前最佳 V2-2，新的 Val Seen/Unseen 8+8 必须同时满足：
  `Q` 各提高至少 `0.005`，且任一 `CR/PCR/WCR` 不下降；否则停止 Val 64+64 和
  Test，继续保留当前 V2-2 checkpoint。
- [ ] 8+8 通过后才运行完整 Val Seen/Unseen；报告中把已用于历史诊断的 scenes
  `0–7`、本轮 gate scenes `8–15` 和其余场景分开列出，避免把调参场景伪装成完全
  未见验证。
- [ ] 完整 Val 通过后只运行一次 Test；第一阶段仍以完成率 Q 为目标，TAT、功耗和
  `CS_paper` 只记录，不参与 checkpoint 选择。
- [ ] 正式训练使用 Slurm 和 BF16，同时运行 2 个独立 seed；每个 seed 使用 2 张
  GPU、60 个活跃 Basilisk 环境，合计 4 张 GPU、120 个活跃环境，整项任务申请的
  CPU 核心总数不得超过 120。
- [ ] 不设置人为训练时长上限；若 Slurm 分区存在单次作业时限，则在同步 barrier
  边界原子保存并自动从 checkpoint 续跑。每 `100` 次 update 永久保留一个恢复点，
  同时维护 `checkpoint_latest.pth`，但不得用 latest 覆盖周期 checkpoint。
