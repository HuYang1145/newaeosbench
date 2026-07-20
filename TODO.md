# 待办事项

## 当前目标

最终目标是降低正式评估中的 `CS_paper`，同时完整报告
`CR/PCR/WCR/TAT_s/PC_Wh`。这里的“准确率”指调度完成率，不是训练动作 top-1
accuracy。

统一公式：

```text
Q = 0.6*CR + 0.2*PCR + 0.2*WCR
CS_paper = Q^(-1) + TAT_s/700 + PC_Wh/100
```

## 当前路线：事件式 Actor + 局部监督 + APPO/PPO

从 2026-07-20 起，活动路线统一使用 `M0–M5`：

```text
M0 结果、分支和基线收束
→ M1 无训练事件式 Actor
→ M2 终止、持续时间与事实结果监督
→ M3 事件决策点的受控局部候选比较
→ M4 事件级 APPO/PPO
→ M5 软容量星座级联合分配
```

旧的 `P0/P0.1/P1/P2/P3.x` 只作为已经运行实验、commit、脚本和产物目录的历史
编号保留，不再使用新的 `P` 编号扩展活动路线。

## M0：结果与恢复点收束

- [x] 当前工作分支保持为 `codex/offline-critic-ranking`。
- [x] 修改前基线为 `71dc76d`。
- [x] 局部受控 rollout 与 Graph-Q 工作通过 47 项测试，并保存为 checkpoint
  commit `0a760ee`。
- [x] M0/M1 正式实施计划保存为
  `docs/superpowers/plans/2026-07-20-event-actor-m0-m1.md`，commit 为 `f50760b`。
- [x] 合入 `codex/p0-causal-history-adapter` 的因果历史、Temporal Adapter、
  训练/评估脚本和测试；保留 Stage3 关闭新模块时的兼容路径。
- [x] Temporal Adapter 10k 已完成 8+8 Val，但 Val Seen/Unseen 的
  `CS_paper` 分别从 `4.2255/4.1632` 恶化到 `4.2994/4.2372`，停止 64+64
  Val 和 Test。
- [x] 历史 P3.1 局部 Graph-Q pilot 已回收：33 个源样本中只有 18 个 pair
  通过 margin，来自 5 个 scene；300/600 秒偏好一致率只有 `45.45%`，
  Graph-Q 合并准确率 `0.4125`，只有 `1/4` fold 通过。
- [x] 历史 P3.1 决策为 `stop_before_actor_or_reranking`：不扩大到 512 场，
  不接入 Actor，不进入 PPO。

## M1：无训练事件式 Actor

目标：Basilisk 仍按 1 秒推进，但冻结 Stage3 Actor 只在事件点接受新的非空任务
决策；第一轮只比较固定 `1/5/15/30/60 s` 承诺，不训练新模型。

- [x] 新增每星 `EventAssignmentState`，使用全局 `task_id` 维护当前任务、剩余
  承诺、开始时间和中断原因。
- [x] 新增 `EventActorRuntime`；承诺有效时不调用 planner，只在事件发生时重规划
  对应卫星。
- [x] 默认 idle 承诺 1 秒以保持新任务响应；另保留可控的多秒 idle 消融，
  `ongoing taskset` 变化时立即唤醒 idle 卫星。任务完成、到期、失败或离开
  ongoing 集合时立即中断。
- [x] 接入 `tools/rollout_model_trajectories.py`，新增 `--event-actor`、
  `--event-commitment-seconds` 和 `--event-idle-commitment-seconds`；默认关闭，
  关闭时逐秒 Stage3 行为保持不变。
- [x] 输出 `model_call_count`、每档承诺数量、任务一秒承诺率、任务平均承诺时长和
  各类中断计数。
- [x] 完成 train scene 0 的完整 3,600 秒 CPU 协议 smoke，并准备正式 8+8 Val
  的 Slurm 包装；未使用 Test。
- [x] M1 标记为“机制实现完成、性能未通过”：任务 5 秒、idle 1 秒时，非空任务
  一秒承诺率降为 `0%`，但联合 Actor 仍调用模型 `3,580` 次，接近逐秒路径，
  且 `CS_paper 3.6409 → 3.7189`，不满足进入正式 Val 的收益门槛。

单场机制 smoke（仅用于诊断，不作为泛化结论）：

| 方案 | CR/% | PCR/% | WCR/% | TAT_s | PC_Wh | CS_paper | model calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage3 逐秒 baseline | 64.44 | 68.62 | 64.76 | 425.33 | 150.29 | 3.6409 | 未单独记录 |
| task 5 s / idle 1 s | 65.56 | 67.89 | 69.59 | 343.58 | 173.18 | 3.7189 | 3,580 |
| task 5 s / idle 5 s，taskset 唤醒 | 56.67 | 61.87 | 58.60 | 381.49 | 153.85 | 3.8048 | 3,056 |

结论：事件承诺已经真正作用于任务动作，但 42 星联合 Transformer 只要任一 idle
卫星需要重规划，就仍会做一次全星前向；因此响应式配置没有实质减少全局前向。
盲目延长 idle 虽相对响应式配置减少 `14.6%` 模型调用，
却明显漏掉可用时机。M2 应训练终止/持续时间/事实结果头，并将“是否触发全局
Actor 前向”与“哪些卫星接受新动作”进一步解耦，而不是继续手工放大固定时长。

## M2：终止、持续时间与事实结果监督

目标：不重新生成反事实轨迹，先复用旧 Stage3 轨迹监督事件决策需要的
`continue/stop`、`1/5/15/30/60 s` 持续时间和短窗口事实结果。第一轮严格冻结
Stage3 Actor，并令 `temporal_residual_scale=0`，因此训练不会改变现有任务 logits。

- [x] 正式设计和实施计划已保存为
  `docs/superpowers/specs/2026-07-20-event-supervision-m2-design.md` 与
  `docs/superpowers/plans/2026-07-20-event-supervision-m2.md`。
- [x] 从旧轨迹构造 `continue/stop` 和保守 duration 标签：提前切换或短轨迹末端
  作为 censored，不伪装成负样本；idle 不参与 duration loss。
- [x] `JointDataset` 已返回事件标签，Temporal Adapter 已新增 continue head、
  5 档 duration head，并复用可见、进度、完成和事件时间 outcome heads。
- [x] M2 配置冻结 91,381,765 个 Stage3 参数，只训练 101,795 个
  Temporal Adapter 参数；动作、feasibility、time 和 assignment loss 均关闭。
- [x] 对 Stage3 annotation 前 256 场完成标签审计，共 15,402,489 条非空执行边。
  continue 占 `99.4442%`；duration 仅 `0.1352%` censored，已观测 duration 中
  `1/5/15/30/60 s` 分别占 `2.21%/5.40%/7.77%/14.45%/70.16%`。配置使用审计
  推导的类别权重，避免多数类主导损失。
- [x] 真实 train scene 0、batch size 2 的 forward/backward/AdamW step 通过：
  总 loss 为 `9.3418`，冻结参数全部无梯度且逐值不变，event head 确实更新；
  64 项 M2/Temporal 回归测试通过。
- [x] 首次 Slurm job `582` 在首个迭代前按预期暴露资源问题并退出：四张 4090
  均被 Slurm 外的 `VLLM::Worker_TP0–TP3` 各占约 21.5 GiB，原继承的
  `batch_size=48` 还需分配 1.45 GiB，GPU 0 当时仅余 1.37 GiB。没有生成 M2
  checkpoint，也不是标签或模型数值失败。
- [ ] 通过 Slurm 在 `local-10` 运行 10k event-head 训练，并记录 job、日志和
  checkpoint；资源受限重试使用 `batch_size=8` 和
  `constraint_batch_size=8`。训练 loss 下降只证明可拟合，不能当作调度性能提升。
- [ ] 训练完成后先审计未见 scene 的 continue/duration/outcome 指标，再把学到的
  终止/持续时间接入 M1 runtime 做小规模同场景 Val；在此之前不运行 Test。

重要边界：continue/duration 是历史专家实际行为标签，outcome 是执行后的事实标签；
它们都不是“同一状态下哪个候选更好”的反事实标签。因此 M2 可以学习何时持续和
实际结果，但不能单独解决候选优劣和专家重复冗余；候选比较仍属于 M3。

## 当前基线

| 模型 | Split | CR/% | PCR/% | WCR/% | PC_Wh |
|---|---|---:|---:|---:|---:|
| Stage2-200k | Val Seen | 36.81 | 39.69 | 36.81 | 88.92 |
| Stage3-200k | Val Seen | 37.50 | 40.79 | 37.29 | 95.15 |
| Stage2-200k | Val Unseen | 41.80 | 45.19 | 41.77 | 104.65 |
| Stage3-200k | Val Unseen | 42.82 | 46.25 | 42.78 | 112.43 |
| Stage2-200k | Test | 21.24 | 23.40 | 20.85 | 43.85 |
| Stage3-200k | Test | 23.28 | 26.02 | 23.02 | 52.11 |

Stage3-200k 完成率较高，但功耗也更高；Val 与 Test 仍有明显泛化差距。

## 已完成诊断

- [x] 统一 `TAT_s/700` 与 `CS_paper`，并增加论文 Table 2 回归测试。
- [x] 完成 TimeModel hard mask 和 bounded soft penalty 的完整 Val 验证。
  收益极小或不稳定，该推理后处理方向停止，不运行 Test。
- [x] 使用 Stage3-200k 在 Val Seen/Unseen 各 64 场统计重复分配、连续接力和
  top-5 任务覆盖。

协调诊断结果：

| 指标 | Val Seen | Val Unseen | 合并 128 场 |
|---|---:|---:|---:|
| 重复冗余选择率 | 46.16% | 47.96% | 47.12% |
| 重复事件产生进度 | 14.39% | 13.70% | 14.01% |
| 重复但没有进度 | 85.61% | 86.30% | 85.99% |
| 合理连续接力 | 11.97% | 10.84% | 11.36% |
| 未完成任务从未进入 top-5 | 9.35% | 3.37% | 6.60% |
| 进入 top-5 但从未被选择 | 33.43% | 28.76% | 31.28% |

结论：主要问题是星座级重复争抢，次要问题是候选任务最终覆盖不足；合理接力不能
解释大部分重复选择。

## 历史实验 P0：轻量二部图联合分配

目标：保留现有 Transformer 的候选表示能力，在其后增加轻量
`bipartite graph assignment head`，减少多颗卫星同时争抢同一任务。

- [x] 从稳定提交 `25718c8` 创建 `codex/bipartite-assignment-head` 分支。
- [x] 复用 Encoder 任务特征、Decoder 卫星特征和现有 logits，定义卫星—任务边。
- [x] 使用残差式分配头；关闭模块时不增加参数，开启后的零初始化也精确复现 baseline。
- [x] 配置第一阶段训练：冻结原模型，只用现有轨迹监督训练 35,105 个图头参数。
- [x] 保留动作 CE，并增加 bounded 重复冲突和专家任务覆盖辅助损失。
- [x] 使用软损失保留有限重复能力，不做绝对一对一硬约束。
- [x] 完成 52 项相关测试、真实 forward/backward/step 和 100 次 CPU 延迟基准。
- [x] 在 `server-10` 使用 4 张 RTX 4090 完成 10k iter 第一阶段训练。
- [x] 完成 Val Seen/Unseen 各 8 场筛选；重复冗余率未达门槛，停止各 64 场验证。
- [ ] 完整 Val 有稳定收益后，只运行一次 Test；在此之前不使用 Test 调参。

第一轮验收门槛：

- 重复冗余率相对 `47.12%` 至少下降 25%。
- Val 小样本 CR/PCR/WCR 任一项下降不超过 0.5 个百分点。
- `PC_Wh`、`TAT_s` 和 `CS_paper` 不出现明显退化。
- 新模块不能调用在线 Basilisk、完整轨道传播或重型几何预测。

第一轮 8+8 场结果：

| Split | CR/% | PCR/% | WCR/% | TAT_s | PC_Wh | CS_paper | 重复冗余率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Val Seen | 41.47 | 45.26 | 41.04 | 505.82 | 110.78 | 4.2032 | 48.81% |
| Val Unseen | 44.39 | 47.56 | 45.12 | 554.26 | 115.55 | 4.1612 | 52.34% |

结论：相对同场景 Stage3-200k baseline，完成率小幅提高，`CS_paper` 基本持平；
但重复冗余率没有下降到目标值 `35.34%`，核心目标失败，因此不运行完整 Val。

根因证据：训练轨迹专家动作本身约有 `44.55%` 重复；当前 CE 会继续模仿这些重复。
在 4 个真实训练样本上，图头仅改变约 `0.20%` 的 top-1 动作，soft collision 从
`0.069053` 降到 `0.068995`，硬重复率只从 `39.04%` 降到 `38.97%`。

## 历史实验 P0.1：无训练全局 owner 分配

- [x] 回到原始 Stage3-200k checkpoint，不加载 P0 图头，不使用专家动作 CE。
- [x] 使用 Transformer logits 做 Hungarian 全局匹配；每颗卫星最多一个任务、每个
  任务每步最多一个 owner，其余卫星选择次优任务或空动作。
- [x] 使用全局 `task_id` 记录上一时刻 owner，并加 `0.25` continuation bonus；任务
  列表重排后不会把相对编号误认成同一任务。
- [x] 未接入在线 Basilisk、轨道传播或几何预测；51 星、301 任务的分配耗时约
  `0.264 ms/step`。
- [x] 43 项相关测试通过；完成 Val Seen/Unseen 各 2 个同场景筛选。
- [x] 2+2 未通过指标门槛，因此停止 8+8、完整 Val 和 Test。

2+2 同场景对照：

| Split | 方案 | CR/% | PCR/% | WCR/% | TAT_s | PC_Wh | CS_paper | 重复率 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Val Seen | Stage3 baseline | 35.26 | 39.06 | 35.08 | 575.32 | 68.50 | 4.2858 | 36.41% |
| Val Seen | P0.1 owner | 28.68 | 33.44 | 26.25 | 503.06 | 58.21 | 4.7318 | 0.00% |
| Val Unseen | Stage3 baseline | 25.93 | 28.34 | 26.24 | 584.52 | 89.98 | 5.5124 | 28.11% |
| Val Unseen | P0.1 owner | 23.89 | 26.95 | 23.96 | 596.78 | 81.97 | 5.7513 | 0.00% |

结论：严格一任务一 owner 确实消除了同一步冲突，也降低了功耗，但同时删除了过多
仍有用的并行选择，导致质量指标和 `CS_paper` 明显变差。下一轮不能继续使用绝对
唯一约束，应研究只压制低价值重复、允许少量高价值协作的自适应容量；仍不恢复专家
动作 CE。

## 历史 P2/P3 探索

下一轮优先方向：

- [x] 从现有轨迹构造 `(s,a,r,s')`，训练 state-only baseline 与
  action-conditioned Critic；1,024 场诊断未通过前半时段排序增益门槛，因此 Actor
  保持冻结，不训练 Advantage adapter。
- [x] 已从现有转移构造 dense reward：任务质量增量、完成时延和传感器功耗组成局部
  奖励，终点校正确保累计奖励严格等于 `-CS_paper`。
- [x] 已完成 1,024 场、每场 8/32 个转移的 dense Critic 对照；全时段和前半时段
  Spearman 增益均低于 `0.05`，停止纯单轨迹离线更新，Actor 继续冻结。
- [x] 已为少量相同场景生成多个“模型候选轨迹”（不是专家轨迹），
  先建立同场景偏好对，避免把场景难度误当成动作质量。
- [x] 首轮 smoke：使用 Stage3-200k checkpoint，对 train 前 2 个场景各生成
  `1 greedy + 3 seeded top-k` 候选。命令为
  `bash scripts/run_same_scene_candidate_smoke.sh`；输出位于
  `work_dirs/same_scene_candidates_stage3_200k_smoke/`，单候选日志位于其
  `logs/`，汇总为 `summary.json`。两场均形成 6 个偏好对，其中一场
  最佳采样候选比贪心降低 `0.0938 CS_paper`。
- [x] 已扩大到 train 前 16 个场景：
  `LIMIT=16 SCENE_WORKERS=4 OUTPUT_ROOT=work_dirs/same_scene_candidates_stage3_200k_16 bash scripts/run_same_scene_candidate_smoke.sh`。共 4 种候选、每种 4 个 scene worker，
  本机 CPU 用时约 46 分钟。`16/16` 场都有 4 条不同动作序列，共
  96 个偏好对；`11/16` 场的采样候选优于贪心，平均最佳改进
  `0.1987 CS_paper`。
- [x] 已对 16 场做严格 4-fold scene 验证。`hidden=64, epochs=200` 的平均
  pairwise accuracy 由 baseline `0.6250` 提高到 `0.6771`，增益 `+0.0521`；
  但只有 `2/4` fold 通过门槛。小网络/少轮数扫描更差，Actor 继续冻结。
- [x] 已完成 64 场扩展实验，仍保持每场 `1 greedy + 3 seeded top-k`：
  `LIMIT=64 SCENE_WORKERS=4 NUM_THREADS=6 OUTPUT_ROOT=work_dirs/same_scene_candidates_stage3_200k_64 bash scripts/run_same_scene_candidate_smoke.sh`。
  共 256 条轨迹、384 个偏好对；`49/64` 场存在优于 greedy 的采样候选。但严格
  4-fold 裁判模型（Critic）平均准确率仅 `0.5938`，相对 baseline 增益 `+0.0260`，
  `0/4` fold 通过，Actor 继续冻结。
- [x] P0 误差审计完成：过滤小 margin 不能修复排序；裁判模型 top-1 只在
  `22/64` 场选中真实最优。去掉 scene 86 离群点后，所选候选平均比 greedy 高
  `0.0175 CS_paper`，当前轨迹级 MLP 停止进入 Actor 更新。
- [x] P1 第一分歧点数据完成：384 对均有动作分歧，363 对的可重建决策前状态全部
  一致；排除 21 对 `t=0` 初始传感器状态缺失和 54 对 margin `<0.05` 后，得到
  311 个 Graph-Q 可用偏好样本。首次分歧中位数为第 14 步，通常只改变 1 颗卫星。
- [x] P2 已完成 256 个独立 train 场景的轻量 Graph-Q 裁判模型实验：共生成
  1,024 条轨迹和 1,536 个候选对，得到 1,286 个可用第一分歧点样本。Graph-Q
  合并排序准确率为 `0.5210`，相对 baseline `0.4914` 仅提升 `+0.0295`，
  `0/4` fold 通过。
  平均 regret 从 `0.5211` 恶化到 `0.6522`，所选候选平均比 greedy 高
  `0.0880 CS_paper`。结果见
  `work_dirs/first_divergence_graph_q_256/summary.json`。
- [x] P2 按门槛停止：不使用该 Graph-Q 更新 Actor，不训练 Advantage/DPO adapter，
  不运行 Val/Test，也不进入 PPO。64 场扩大到 256 场后准确率仍约 52%，说明继续
  增加同类场景或只扩大裁判网络不能解决“第一分歧动作与 3,600 步最终回报之间的
  长期归因噪声”。
- [x] 完成动作持续性和专家轨迹对照：Stage3 greedy 的一秒片段占片段数
  `95.68%`，而 256 条 `OptimalAlgorithm` 专家样本只有 `0.36%`；同场景 48 场
  对照分别为 `95.42%` 和 `0.30%`。一秒抖动主要是模型行为，不是专家常态。
- [x] 完成一秒短脉冲因果审计：`89.18%` 为 `空闲 -> 任务 -> 空闲`，全部打开
  传感器并计入功耗，但只有 `0.0238%` 在下一步由该卫星真正看到目标。它不等于
  空闲，绝大多数却是“有代价、无直接观测收益”的动作。
- [x] 确认专家本身仍有联合分配缺陷：同场景专家重复冗余边为 `42.53%`，模型为
  `46.78%`。持续性问题主要由模型放大，跨卫星冲突则部分继承自独立贪心专家；
  两者必须分开改进。
- [x] P3.0 受控局部 rollout smoke 已完成，不改 Actor：两个分支只改变当前动作，
  后续统一使用冻结的 deterministic greedy Actor。train scene 0 的第 8 秒短脉冲
  已完成 `H=180/300/600 s` 对照；相同 H=180 重跑结果逐字节一致，三个窗口的
  stay/switch 决策前状态哈希也一致，15 项相关测试通过。结果见
  `work_dirs/local_action_branch_p30_smoke_cpu/summary.json`。
- [x] P3.1 多场景受控标签与四折 pilot 已完成：以 `H=300 s` 为主窗口、
  `H=180/600 s` 做一致性对照；33 个源样本中只有 18 个有效 pair，来自 5 个
  scene，300/600 秒偏好一致率为 `45.45%`。
- [x] P3.2 已实现局部多目标 Graph-Q 裁判：输入仅包含在线可得的上一任务、连续
  执行时长、近 30/60 秒切换、任务进度/剩余时限、Actor logits 和卫星—任务图；
  输出候选代价及完成、进度、功耗、切换、一秒片段、重复观测六个分量。
  `is_visible` 与 Basilisk 结果只作为离线监督，不进入在线推理。
- [x] P3.3 四折训练已完成但未通过：Graph-Q 合并 pairwise accuracy
  `0.4125`，汇总基线为 `0.3000`，只有 `1/4` fold 通过，未达到 `0.60` 和
  `3/4` fold 门槛。
- [x] P3.4 按门槛取消：不进行 top-k 重排序，不扩大到 512 场，不运行
  Val/Test，不训练 Advantage/DPO adapter，也不进入 PPO。

## 实验记录要求

每个实验必须记录：

```text
实验目的：
工作分支和基线 commit：
checkpoint 与训练 annotation：
关键配置和 world_size：
Val Seen  CR / PCR / WCR / TAT_s / PC_Wh / CS_paper：
Val Unseen CR / PCR / WCR / TAT_s / PC_Wh / CS_paper：
重复率 / 合理接力率 / top-k 覆盖率：
相对 Stage3-200k 的收益与代价：
结论与下一步：
```

不允许只报告训练 loss、单个最好场景或单一完成率，也不能把功耗明显上升后的完成率
增益写成“全面提升”。

## 当前托管任务

- Temporal Adapter P0-B 10k 已于 2026-07-20 00:42 EDT 完成：分支
  `codex/p0-causal-history-adapter`，训练代码提交 `f70a8b0`，共完成 10,000 iter。
- P0-B 加载 Stage3-200k checkpoint：
  `work_dirs/paper_joint_stage3_200k/checkpoints/iter_200000/model.pth`；训练日志：
  `work_dirs/eval_logs/temporal_adapter_p0_10k_train.log`；输出目录：
  `work_dirs/temporal_adapter_p0_10k/`；最终 checkpoint：
  `work_dirs/temporal_adapter_p0_10k/checkpoints/iter_10000/model.pth`。
- 已通过 Slurm 运行 Val Seen/Unseen 各 8 场：包装脚本
  `scripts/eval_temporal_adapter_p0_8_slurm.sh`，账户 `lab_team`，申请
  `1 GPU / 24 CPU / 96G / 2h`；不再直接占用本机 GPU。
- Slurm job `493` 已在 `local-10/server-10` 完成，`ExitCode=0:0`，耗时
  `00:35:36`；日志：
  `work_dirs/eval_logs/temporal_adapter_p0_eval8_slurm_493.log`。
- job `492` 因 Slurm spool 中的脚本无法按 `BASH_SOURCE` 找到代码目录而退出；
  已在提交 `7bb993c` 中改为优先使用 `SLURM_SUBMIT_DIR`，job `493` 已正常加载
  Temporal Adapter checkpoint 并完成评估。
- 8+8 汇总：`work_dirs/eval_summaries/temporal_adapter_p0_10k_val8.json`。
- 同场景对照中，Val Seen/Unseen 的 `CS_paper` 分别由 Stage3 baseline 的
  `4.2255/4.1632` 恶化到 `4.2994/4.2372`；两个 split 均未通过，不运行
  64+64 Val 或 Test。
- P0 第一阶段训练已在 `server-10` 直接完成，不经过 Slurm；`groupA/groupB` 权限
  不影响本机实验。
- 训练日志：`work_dirs/assignment_head_p0_c020_cov010_10k/`；最终 checkpoint：
  `work_dirs/assignment_head_p0_c020_cov010_10k/checkpoints/iter_10000/model.pth`。
- 8+8 Val 汇总：`work_dirs/eval_summaries/assignment_head_p0_10k_val8.json`。
- 8+8 协调诊断：`work_dirs/rl_eval_assignment_head_p0_10k_*_8/coordination_diagnostics.json`。
- P0.1 2+2 汇总：`work_dirs/eval_summaries/owner_assignment_p01_b025_val2.json`。
- P0.1 2+2 协调诊断：
  `work_dirs/rl_eval_owner_assignment_p01_b025_val_*_2/coordination_diagnostics.json`。
- 最近完成：`aeos_stage3_coordination_diag64`，Stage3-200k、Val Seen/Unseen
  各 64 场、`world_size=16`、`top-k=5`。
- 日志：`work_dirs/eval_logs/stage3_200k_coordination_top5_*.log`。
- 结果：`work_dirs/rl_eval_stage3_200k_coordination_top5_*/coordination_diagnostics.json`。

## 已知工程问题

- `data/satellites/val_seen` 仍指向旧机器的绝对路径；重新生成 Val Seen 前必须修复。
- `third_party/basilisk` 下的 AutoTeX 是测试生成状态，不纳入正式提交。

## 参考文档

- `README.md`：项目入口。
- `改进日志.md`：完整实验过程、负结果和回滚点。
- `docs/实验复现报告.md`：论文对齐基线与汇报口径。
- `docs/aeos_former_shape_flow.md`：模型张量流。
- `third_party/gnn_references/README.md`：GNN 调度参考实现说明。
