# M3 事件点受控候选比较设计

## 1. 目标与成功定义

M3 在冻结 Stage3 Actor 的前提下，为事件决策点生成受控局部反事实标签，学习
“保持上一任务、切换到 Actor 候选以及承诺多长时间”之间的相对优劣。M3 不直接
进入 PPO/APPO，不在线调用 Basilisk，也不使用 Test 调参。

M3 的“完成”指以下研究闭环已经结束：

1. 事件级受控分支协议、标签审计、scene-level 交叉验证和可选闭环接入均有可复现
   代码与记录；
2. 如果任一预注册门槛失败，停止后续扩大并记录“M3 完成但未通过”；
3. 只有标签和 Critic 同时通过，才允许进行单场闭环 smoke；只有 smoke 通过，才允许
   8+8 Val；M3 不运行 Test。

最终目标仍是降低 `CS_paper`，同时完整报告
`CR/PCR/WCR/TAT_s/PC_Wh`。Critic accuracy、训练 loss 或局部 cost 不能替代
Basilisk 闭环指标。

## 2. 为什么不能复用旧 P3.1

历史 P3.1 只覆盖一个时刻的一秒动作，并用 300 秒 prefix cost 产生标签。8 场 pilot
中 33 个源 pair 只有 18 个通过 margin，300/600 秒偏好一致率为 `45.45%`，Graph-Q
四折合并准确率为 `0.4125`，只有 `1/4` fold 通过。

M3 不扩大这套一秒标签，而是同时控制“任务选择”和“承诺时长”。当前动作在承诺
结束、任务不可用或任务完成后，两个分支都恢复为同一个冻结 deterministic Stage3
Actor。这样比较的是事件动作，而不是一个容易被下一秒立即覆盖的脉冲。

## 3. 受控事件分支协议

### 3.1 决策状态

第一轮复用现有 Stage3 greedy train 轨迹选择事件点，不重新采样随机参考策略。候选点
必须满足：

- 决策前至少有一秒历史；
- 上一任务与当前 Actor top-1 不同，或上一任务仍 ongoing、可形成有意义的 stay；
- stay 和 switch 涉及的非空任务在决策时刻均已 release、未 due、未完成；
- 距离场景结束至少 300 秒；
- 按 300 秒时间桶分层，避免样本只来自开场。

每个分支从零确定性重放到同一决策时刻。覆盖动作前必须比较完整
`decision_state_signature` 和 `decision_context`；不一致的 group 直接作废。

### 3.2 候选集合

每个事件点最多保留两个互异任务动作：

1. `stay`：上一秒实际任务；
2. `switch`：Stage3 logits 中排名最高且不同于 `stay` 的候选，可以是 idle。

非空任务分别组合 `1/5/15/30/60 s` 五档承诺；idle 只允许 1 秒，避免用长空闲
获得表面低功耗。一个 group 最多 10 个分支。第一轮不增加 Actor top-2/top-3，
避免将任务选择、时长和候选覆盖三个变量同时扩大。

### 3.3 承诺执行

新增事件承诺包装器，只修改目标卫星：

- 从 `decision_time` 开始保持指定任务，最多持续 `commitment_seconds`；
- 其他卫星每秒继续使用同一个冻结 Stage3 Actor；
- 任务完成、失败、到期或离开 ongoing 集合时立即结束承诺；
- 承诺结束后目标卫星也恢复 Stage3 Actor；
- 记录请求时长、实际时长、中断原因、目标任务和原始 Actor 动作；
- Basilisk 只在离线分支生成中运行，不进入 Critic forward 或正式在线热路径。

## 4. 原始结果与稳健偏好标签

每个分支只运行一次最长 300 秒，并从同一运行提取 `60/180/300 s` 前缀，避免重复
仿真。保存以下原始分量：

- 完成任务数、完成任务总 duration；
- 归一化 partial progress gain；
- 目标卫星及全星 direct-visible seconds；
- 传感器 `PC_Wh`；
- switch 数、一秒片段数、重复卫星秒；
- 首次可见、首次进度、完成时间及 observed/censored 状态；
- 每个前缀的 `CR/PCR/WCR/TAT_s/local_PC_Wh/prefix_cost`。

偏好不重新发明一套任意加权 reward。对同一状态下两个候选，使用论文口径
`prefix_cost` 的差值，但必须同时满足：

1. 180 秒和 300 秒均可比较；
2. 两个窗口的优劣方向一致；
3. 300 秒绝对 margin 不小于 `0.01`；
4. 两个候选不是相同任务与相同时长；
5. 质量更差但仅靠关闭传感器降低功耗的候选不能通过质量保护条件：其
   `CR/PCR/WCR` 三项不得同时低于对手。

60 秒只作早期响应诊断，不参与主标签。方向反转、小 margin、窗口未观测完整和状态
不一致均保留为审计记录，但不进入 Critic loss。

## 5. 事件候选 Critic

M3 新建轻量 `EventCandidateCritic`，不复用缺少 duration 输入的旧 Graph-Q。每个候选
只消费在线已有信息：

- 目标卫星静态/动态特征；
- 候选任务特征，idle 使用显式 idle 标志；
- 有限化后的 Stage3 candidate logit；
- previous-task match、run length、30/60 秒 switch count；
- 承诺时长的对数归一化表示；
- 当前任务进度比例、release/due 剩余时间和传感器类型兼容性。

Critic 输出候选标量 cost。训练使用同 group 内 pairwise logistic loss，margin 只用于
样本权重上限，不把 300 秒绝对 cost 当作精确回归目标。所有划分按 scene 完成，禁止
同一 scene 的不同事件进入 train 和 validation 两侧。

比较三个无训练 baseline：

1. Stage3 candidate logit；
2. always-stay；
3. M2 continue/duration 规则在可映射候选上的选择。

## 6. 分阶段门槛

### M3-A：协议 smoke

在 train scene 0 的一个中段事件点生成全部候选，要求：

- 所有分支状态签名一致；
- 请求承诺与实际承诺记录正确；
- 任务失效能提前中断；
- 60/180/300 前缀来自同一次最长 rollout；
- 重复运行 summary 可确定性复现。

### M3-B：8 场标签 pilot

通过 Slurm CPU 资源生成 8 个 train scene、每场最多 2 个事件点。只有同时满足以下
条件才训练 Critic：

- 至少 6 个 scene 产生有效 group；
- 至少 32 个稳定 pair；
- 180/300 可比较 pair 的方向一致率不低于 `0.70`；
- 被选为更优的 duration 至少覆盖 3 个档位；
- 任一 duration 占比不超过 `85%`。

未通过时停止，不靠降低 margin 或挑选场景修饰结果。

### M3-C：scene-level Critic 验收

使用 4-fold scene-level 交叉验证。通过条件为：

- 合并 pairwise accuracy 不低于 `0.60`；
- 相对最强 baseline 提升至少 `0.05`；
- mean top-1 regret 不高于最强 baseline；
- 至少 `3/4` fold 分别通过；
- stay、switch 和 duration 子组均不能完全塌缩为单一输出。

未通过则停止在离线 Critic，不接入 Actor。

### M3-D：条件式闭环验收

只有 M3-B/M3-C 通过才实现有界 rerank：在事件点枚举同一 stay/switch-duration
候选集，Critic margin 达到离线校准阈值时才覆盖 Stage3，否则回退 Stage3/M1。

先运行 train scene 0 完整 3,600 秒 smoke。与同场景 Stage3 比较时：

- `CR/PCR/WCR` 任一项下降超过 `0.5` 个百分点即失败；
- `CS_paper` 必须改善；
- `PC_Wh`、任务一秒承诺率、duration 分布和模型调用次数必须完整报告。

单场通过后才运行 Val Seen/Unseen 各 8 场；8+8 仍执行同样质量保护，并要求两个
split 的平均 `CS_paper` 改善。M3 不运行 64+64 或 Test。

## 7. 文件边界

预计新增：

- `constellation/new_transformers/event_candidate.py`：事件候选、稳健标签和审计；
- `constellation/new_transformers/event_candidate_critic.py`：候选编码、pairwise loss
  与指标；
- `tools/generate_event_candidate_branches_m3.py`：单场受控事件分支；
- `tools/generate_event_candidate_dataset_m3.py`：多场景 Slurm/CPU 数据入口；
- `tools/train_event_candidate_critic_m3.py`：scene-fold 训练与门槛汇总；
- `scripts/run_event_candidate_m3_pilot_slurm.sh`：M3-A/M3-B/M3-C 包装；
- 对应 `tests/test_event_candidate*.py`。

预计小范围修改：

- `constellation/new_transformers/local_action_branch.py`：增加多秒受控承诺，保留现有
  单步 API 兼容性；
- `tools/rollout_model_trajectories.py`：仅在 M3 离线门槛通过后增加显式关闭默认的
  Critic rerank；
- `TODO.md` 与 `改进日志.md`：记录 Slurm job、数据量、门槛和停止原因。

## 8. 风险与停止规则

- **标签仍不稳定**：停止 M3，不增加场景、不调整窗口挑结果；下一步重新定义事件
  终止型 reward，而不是进入 PPO。
- **类别塌缩**：检查候选生成覆盖和 pair 过滤，不能只靠 class weight 隐藏数据问题。
- **仿真成本过高**：第一轮固定 stay + 一个 switch，每场最多两个事件；不扩 top-k。
- **Critic 离线好、闭环差**：按单场门槛停止，记录为新的离线—在线分布偏移证据。
- **任务完成导致承诺提前结束**：这是物理事件，不补齐到请求时长；保存实际时长和
  observed mask。
- **共享 GPU/CPU 资源冲突**：长任务通过 Slurm，代码 smoke 使用小规模 CPU；不在
  登录节点直接运行大规模 Basilisk。
- **原有工作树改动**：不暂存或修改 `.claude/settings.json`、`CLAUDE.md`、现有备份
  文件和 Basilisk AutoTeX 目录。
