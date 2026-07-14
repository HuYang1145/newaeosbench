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
- [ ] 不生成新专家轨迹，先从现有转移构造可归因到单步动作的 dense reward：任务
  进度增量、任务完成事件和传感器功耗代价；验证它与最终 `CS_paper` 的排序方向一致。
- [ ] 只有局部奖励 Critic 明显超过 state-only baseline 后，才使用 Advantage 加权
  训练小型 adapter；继续不使用等权专家动作 CE。
- [ ] adapter 离线验收通过后，仅运行 Val Seen/Unseen 各 2 场 Basilisk 验真；未通过
  时不扩大 Val，不运行 Test，也不进入 PPO。
- [ ] 如果局部奖励仍无法提供动作级信号，则停止纯单轨迹离线更新，改为为少量同一
  场景保存多个模型候选轨迹，再研究偏好学习或 PPO。

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
