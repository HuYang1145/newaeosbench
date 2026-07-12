# 待办事项

## 当前唯一主线

> 以正式评估协议下的 **paper-aligned 综合得分 `CS_paper` 越低越好**
> 为最终目标，同时完整报告 `CR/PCR/WCR/TAT_s/PC_Wh`。不再把单独
> Test CR 作为唯一优化目标。

这里的“准确率”默认指调度完成率，不是训练过程中的动作分类 top-1
accuracy。后续文档、实验和汇报必须优先使用 `CR`、`PCR`、`WCR` 的正式名称。

benchmark 难度分层、物理可观测性统计和失败原因分析继续保留，但它们只是定位
问题的诊断手段。最终验收要看任务完成质量、周转时间和功耗的综合折中。

### TAT 与 CS 统一口径

官方代码的 `TurnAroundTimeEvaluator` 输出原始秒数，仿真上限为 3600 秒，
因此论文表格中 `TAT/h=7.50` 不可能是真实小时。将表中 TAT 解释为
`TAT_s / 100` 可以精确复现论文 Table 2 的 CS。当前统一使用：

```text
Q = 0.6*CR + 0.2*PCR + 0.2*WCR
TAT_100s = TAT_s / 100
CS_paper = Q^(-1) + TAT_100s/7 + PC_Wh/100
         = Q^(-1) + TAT_s/700 + PC_Wh/100
```

`CR/PCR/WCR` 使用 0 到 1 的比例值。`CS_no_TAT` 仅保留为历史辅助指标，
不再作为最终模型排序依据。

## 当前正式基线

当前主结果来自论文式 `JointModel` 200k 联合训练：

| 模型 | Split | CR/% | PCR/% | WCR/% | PC_Wh |
|---|---|---:|---:|---:|---:|
| Stage2-200k | Val Seen | 36.81 | 39.69 | 36.81 | 88.92 |
| Stage3-200k | Val Seen | 37.50 | 40.79 | 37.29 | 95.15 |
| Stage2-200k | Val Unseen | 41.80 | 45.19 | 41.77 | 104.65 |
| Stage3-200k | Val Unseen | 42.82 | 46.25 | 42.78 | 112.43 |
| Stage2-200k | Test | 21.24 | 23.40 | 20.85 | 43.85 |
| Stage3-200k | Test | 23.28 | 26.02 | 23.02 | 52.11 |

当前判断：

- Val Seen 的 CR 约为 37%，Val Unseen 约为 43%，但 Test 只有约 23%。
- Stage3-200k 的完成率高于 Stage2-200k，但功耗也更高。
- 第一优先级是缩小 Test 泛化差距，而不是继续只抬高 Val Seen 或 Val Unseen。
- 早期 Tiny / CE-only 结果只证明流程跑通，不与论文式联合训练结果直接混合比较。
- 后续报告增加 `TAT_s`、`TAT_100s` 和 `CS_paper`，并保留各单项指标避免
  综合分数掩盖退化。

## 当前失败诊断

历史 observable-filtered Stage3 Test 诊断中共有 9,433 个失败任务：

| 失败原因 | 占比 |
|---|---:|
| 选过，但 Basilisk 仿真中始终没有真正可见 | 47.41% |
| 模型从未选择 | 44.39% |
| 已有进度，但连续观测时长不够 | 8.21% |

这组数据不是当前正式 benchmark 结果，只作为问题定位证据。它说明下一阶段应优先
解决“预测可行性与 Basilisk 不一致”和“任务覆盖不足”，而不是直接扩大模型规模。

## P0：建立固定、可解释的评估基线

- [ ] 固定 Stage2-200k、Stage3-200k checkpoint 和正式 64 场景
  Val Seen、Val Unseen、Test annotation。
- [ ] 对每次实验统一输出
  `CR/PCR/WCR/WPCR/TAT_s/TAT_100s/PC_Wh/CS_paper`。
- [ ] 分开记录训练动作 top-1 accuracy 与 Basilisk 调度完成率，禁止混称 accuracy。
- [ ] 在当前正式 tasksets 上重新统计三类失败：`never_selected`、
  `selected_never_visible`、`insufficient_continuous_progress`。
- [ ] 增加每个时间步的重复任务分配率、空动作比例、可行任务覆盖率和任务切换率。
- [ ] 后续消融先在 Val Seen / Val Unseen 比较 `CS_paper`和全部单项，
  锁定方案后只运行一次 Test。

验收标准：同一 checkpoint 重复评估时使用相同 split、场景数、并行度、指标公式和
汇总方式，结果能够追溯到具体 checkpoint、配置和输出目录。

## P1：校准 TimeModel 与 Basilisk 可行性

目标：减少“模型选过任务，但真实仿真始终不可见”的失败。

- [ ] 单独评估 `TimeModel` 的 precision、recall、FPR、FNR 和概率校准曲线。
- [ ] 使用真实轨迹中的 `is_visible` 对比不同 feasibility threshold。
- [ ] 统计模型高置信度选择但 Basilisk 从未可见的卫星—任务对，构建 hard negative。
- [ ] 先做不重新训练的推理消融：对高置信度不可行任务使用 soft penalty 或 hard mask。
- [ ] 检查传感器类型、姿态机动时间、电量、反作用轮状态和连续可见窗口是否被正确
  反映到可行性预测。
- [ ] 比较仅调整 threshold、重新训练 TimeModel、联合微调 `JointModel` 三种方案。

验收标准：不仅报告 TimeModel 分类指标，还必须验证 Test `CR/PCR/WCR` 是否提高，
并确认 `PC_Wh` 和空动作比例没有异常恶化。

当前进度：已实现可配置的 `feasibility_threshold` 和离线校准工具，并移除
旧 `AEOS_TAU_S` 环境变量入口。hard threshold 扫描和完整 64+64 场验证已完成。

`0.1/0.2/0.3` 会严重过滤。`0.03` 在完整 Val Seen / Val Unseen 上小幅降低
`CR/WCR`，但降低了 `TAT_s/PC_Wh`。按统一后的 `CS_paper`，Val Seen 约从
`4.3596` 降到 `4.3549`，Val Unseen 约从 `4.1654` 降到 `4.1434`。
这是很小的综合改善，hard mask 不再继续扫描；下一步优先测试更温和的 soft penalty。

## P1：从逐卫星分类改进为星座级联合分配

目标：减少不同卫星之间的任务冲突、重复争抢和全局资源分配不合理。

- [ ] 统计当前确定性推理中多颗卫星同时选择同一任务的频率及其结果。
- [ ] 基于现有 logits 实现不重新训练的 top-k 联合分配消融。
- [ ] 比较 Hungarian matching、顺序选择加动态 mask、允许有限重复分配三种策略。
- [ ] 对正在观测且接近完成的任务设置合理的持续分配优先级。
- [ ] 若推理后处理有效，再设计星座级 assignment loss，而不是立即重写整个 Decoder。

验收标准：记录重复分配率、任务覆盖率、任务切换率以及最终 Test CR 的变化，不能只
报告动作分类 accuracy。

## P1：改进动作监督和损失定义

目标：避免把单条专家轨迹中的单个 `task_id` 当作唯一正确动作。

- [ ] 统计空动作、热门任务和长尾任务的标签分布。
- [ ] 判断同一状态是否存在多个物理可行、最终可成功的替代动作。
- [ ] 比较 one-hot CE、带类别权重 CE、focal loss 和 soft target。
- [ ] 尝试 success-weighted imitation，使高质量轨迹和最终成功动作权重更高。
- [ ] 对模型高置信度但导致失败的动作加入 hard-negative 或 ranking 约束。
- [ ] 分别扫描 `L_a/L_s/L_t` 权重，不再默认三项固定为 1 就是最优设置。

验收标准：同时报告 `L_a/L_s/L_t`、动作 top-1 accuracy、TimeModel 指标和正式调度
指标，确认离线 loss 改善能够转化为 Test CR 改善。

## P2：针对失败状态进行 DAgger / 专家迭代

目标：缓解训练只看到专家状态、推理却不断进入自身错误状态的 exposure bias。

- [ ] 使用 Stage3 checkpoint 在训练场景上进行模型 rollout。
- [ ] 提取 `never_selected`、`selected_never_visible` 和观测中途切换的关键状态。
- [ ] 在这些状态上调用专家算法生成纠正动作或更优候选动作。
- [ ] 将纠正样本按失败类型加入新一轮 annotation，不只按整条轨迹 `tau_e` 筛选。
- [ ] 对 hard-but-solvable 场景提高采样权重，同时保留原始专家数据防止遗忘。
- [ ] 用小规模 Stage4 消融确认有效后，再启动正式长训练。

验收标准：明确记录新增状态数量、各失败类型比例、annotation 来源和 epoch 路由，
并在相同 Test 协议下与 Stage3-200k 比较。

## P2：增强连续观测和短时规划

目标：减少“已有进度但没有连续观测到 duration”的失败和无意义任务切换。

- [ ] 显式输入任务剩余观测时长、当前连续观测长度和距离截止时间的 slack。
- [ ] 统计成功任务与失败任务的平均切换次数。
- [ ] 对接近完成的任务增加 completion bonus 或 continuation bias。
- [ ] 测试最小承诺时长和切换惩罚，并保留失去可见性时的退出机制。
- [ ] 对 top-k 动作做轻量短时 rollout 或 beam search，避免只看当前一步 logits。

验收标准：重点观察 `insufficient_continuous_progress` 占比、任务切换率、CR 和
`PC_Wh`，防止通过长期锁定不可行任务造成反效果。

## P2：训练课程与难度采样

目标：提高 Test 泛化，而不是通过过滤正式评估任务制造更高分数。

- [ ] 按 `任务数 / 卫星数`、可行任务比例、时间窗紧迫度建立场景难度标签。
- [ ] 先在普通场景学习基本策略，再逐步混入困难和极难场景。
- [ ] 对“物理可解但模型失败”的场景增加采样权重。
- [ ] 保证 batch 内各难度层比例稳定，避免异常拥挤场景长期主导梯度。
- [ ] 正式 Val/Test 保持原 annotation 和任务定义，不用过滤后的简单数据代替。

验收标准：按难度层分别报告指标，同时保留原始正式 split 的总体结果。

## P3：补充直接的物理相对特征

目标：为卫星—任务配对提供更明确的物理归纳偏置，而不是只扩大网络宽度和深度。

- [ ] 候选特征包括离轴角、下一可见窗口开始时间、预计窗口长度、任务 slack、
  剩余 duration、姿态机动角、预计机动时间、电量余量和反作用轮余量。
- [ ] 优先验证特征是否能从当前状态快速计算，并保证训练与评估计算一致。
- [ ] 每次只加入一组特征做消融，避免无法判断收益来源。
- [ ] 只有在数据、损失、推理和物理特征消融完成后，再考虑扩大模型或更换架构。

## 推荐执行顺序

1. 完成 `TAT_s -> TAT_100s -> CS_paper` 的统一汇总工具和回归测试。
2. 不重新训练，将 hard mask 改为轻量 soft penalty，只在 Val 上扫少量惩罚强度。
3. 如果 soft penalty 稳定降低 `CS_paper`，再进行完整 Val 并锁定配置。
4. 若轻量后处理收益有限，使用 hard negatives 重新校准/训练 TimeModel。
5. 星座级联合分配作为第二主线，优先解决重复分配和覆盖不足。
6. DAgger、课程学习、额外物理特征和更大模型暂缓，不同时开多条主线。

## 实验记录要求

每个实验必须记录：

```text
实验目的：
基线 checkpoint：
训练 annotation：
关键配置差异：
评估 split / 场景数 / world_size：
Val Seen  CR / PCR / WCR / TAT_s / PC_Wh / CS_paper：
Val Unseen CR / PCR / WCR / TAT_s / PC_Wh / CS_paper：
Test      CR / PCR / WCR / TAT_s / PC_Wh / CS_paper：
三类失败原因占比：
相对 Stage3-200k 的收益与代价：
结论与下一步：
```

不允许只报告最优单个场景、只报告训练 loss、只报告 Val Unseen，或把功耗明显上升
后的完成率增益写成“全面提升”。

## 当前托管任务

- 当前没有正在运行的托管训练或评估任务。
- 最近完成：`aeos_timemodel_threshold003_full_val`，已完成 `threshold=0.03`
  的 Val Seen / Val Unseen 各 64 场评估，结果见上文 TimeModel 进度。
- 最近完成：`aeos_timemodel_valscan8_fix`，`0.1 / 0.2 / 0.3` 均显著降低
  Val Seen / Val Unseen 完成率，不进入完整 64 场评估。
- 旧 `aeos_timemodel_valscan8` 已完成，但因 `actor_model_kwargs` 构建顺序错误，
  阈值未进入 Actor，仅 baseline 可作为参考，三组阈值结果作废。
- 最近完成：`aeos_timemodel_calib64`，输出位于
  `work_dirs/timemodel_calibration/{val_seen,val_unseen}_stage3_200k_64.json`。
- 下一次长任务启动前，必须在这里记录 Slurm job、命令或包装脚本、日志路径、
  checkpoint 来源和预期输出目录。

## 参考文档

- `README.md`：环境、数据、训练和评估入口。
- `docs/实验复现报告.md`：当前正式基线和论文对齐结论。
- `docs/constellation_code_structure.md`：代码、训练和评估调用链。
- `docs/new_transformers_dataset_model.md`：训练数据与模型损失关系。
- `docs/aeos_former_shape_flow.md`：Encoder、TimeModel、Decoder 张量流。
- `docs/observable_filtered_stage3_eval_summary.md`：历史可观测性过滤实验，仅作诊断参考。
