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

- [ ] 从当前稳定提交创建 `codex/bipartite-assignment-head` 分支。
- [ ] 复用 Encoder 任务特征、Decoder 卫星特征和现有 logits，定义卫星—任务边。
- [ ] 使用残差式分配头，使旧 checkpoint 在关闭新模块时能够恢复原 baseline。
- [ ] 第一阶段冻结或低学习率微调 Transformer，只用现有轨迹监督训练分配头。
- [ ] 保留原动作损失，并增加重复冲突和候选覆盖辅助损失。
- [ ] 对连续观测和跨卫星接力样本允许有限重复，不做绝对一对一硬约束。
- [ ] 增加单元测试、真实 forward/backward/step smoke test 和推理耗时基准。
- [ ] 先评估 Val Seen/Unseen 各 8 场，通过后再跑各 64 场。
- [ ] 完整 Val 有稳定收益后，只运行一次 Test；在此之前不使用 Test 调参。

第一轮验收门槛：

- 重复冗余率相对 `47.12%` 至少下降 25%。
- Val 小样本 CR/PCR/WCR 任一项下降不超过 0.5 个百分点。
- `PC_Wh`、`TAT_s` 和 `CS_paper` 不出现明显退化。
- 新模块不能调用在线 Basilisk、完整轨道传播或重型几何预测。

## 后续方向

以下方向暂缓，不与 P0 同时启动：

- 图分配头在完整 Val 有效后，再用 PPO 小规模微调 `CS_paper`。
- 对失败状态做 DAgger 或新一轮专家迭代。
- 改进 one-hot CE、soft target、focal loss 和 success-weighted imitation。
- 增强连续观测、课程学习、难度采样和轻量物理相对特征。
- 若图结构仍暴露可行性问题，再用 hard negatives 重训 TimeModel。

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

- 当前没有正在运行的托管任务。
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
