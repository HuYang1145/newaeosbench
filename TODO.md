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

## P0：轻量二部图联合分配

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

## P0.1：无训练全局 owner 分配

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

## 后续方向

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
- [ ] 下一条策略进化路线待确认：不直接用通用小语言模型替换 Graph-Q。优先比较
  能读取轨迹前缀的轻量 Decision Transformer、直接偏好更新原 Actor，以及
  IQL/AWR 等保守离线强化学习；新实验必须先定义小样本验收门槛和回滚点。

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

- 计划启动 Temporal Adapter P0-B 10k：分支
  `codex/p0-causal-history-adapter`，基线 `d5859a3`，使用本机 GPU 0-3，
  `tmux=aeos_temporal_p0_10k`，命令
  `bash scripts/train_temporal_adapter_p0_10k.sh`。
- P0-B 加载 Stage3-200k checkpoint：
  `work_dirs/paper_joint_stage3_200k/checkpoints/iter_200000/model.pth`；训练日志：
  `work_dirs/eval_logs/temporal_adapter_p0_10k_train.log`；输出目录：
  `work_dirs/temporal_adapter_p0_10k/`；预期最终 checkpoint：
  `work_dirs/temporal_adapter_p0_10k/checkpoints/iter_10000/model.pth`。
- 当前没有正在运行的训练或评估任务。
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
