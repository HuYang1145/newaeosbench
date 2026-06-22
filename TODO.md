# TODO

## 当前目标

当前项目目标从“继续堆叠旧复现实验记录”调整为：

> 在保持论文复现口径清楚、实验记录可追溯的前提下，优先修正场景生成阶段的任务点位有效性问题，减少随机生成但物理上不可观测的无解任务。

这里的“任务成功率”优先对应 `CR`、`PCR`、`WCR` 等任务完成相关指标。当前需要先区分两类失败：

- **场景无解导致的失败**：随机生成的观测点位在给定星座、时间窗、传感器和仿真约束下，没有任何卫星能够有效拍摄。这类任务不应简单归因于模型预测失败。
- **模型调度导致的失败**：任务本身存在可观测机会，但模型没有及时分配、没有连续观测到完成，或引发能耗/传感器约束问题。

下一阶段的核心目标是：在生成正式 `constellation + taskset` 场景时增加可观测性筛选，先判断随机任务点位是否至少存在可行观测机会，再进入训练、评估或论文对比。这样才能让 `CR/PCR/WCR` 更准确地反映调度模型能力，而不是被大量物理无解点位压低。

## 当前统一口径

当前 `TODO.md` 服从 `docs/实验复现报告.md` 中的主结论：

- 论文式 `JointModel` 200k 联合训练是当前正式对齐 AEOS-Former 的主线。
- 早期 Tiny / CE-only 实验只作为流程跑通记录，不作为最终论文对齐依据。
- `Stage2-200k` 更适合作为功耗受控的综合折中候选。
- `Stage3-200k` 更适合作为完成率优先候选。
- 当前本地 `TAT` 与论文表格口径仍未完全统一，因此临时比较优先使用 `CS_no_TAT`。

当前临时指标为：

```text
CS_no_TAT = (0.6*CR + 0.2*PCR + 0.2*WCR)^(-1) + PC_Wh/100
```

注意：

- `CR`、`PCR`、`WCR` 在公式中使用 0 到 1 的比例值，不使用百分数。
- `PC` 优先使用 `PC_Wh`；如果原始输出只有 `PC`，再按 `PC_Wh = PC / 3600` 换算。
- `CS_no_TAT` 只是当前排查和模型排序用的临时指标，不等同于论文最终 `CS`。

## 与实验复现报告的对齐结论

当前需要避免的过时表述：

- 不再把旧 200k CE-only 模型当作严格论文复现模型。
- 不再把 Val Unseen 高于 Val Seen 简单写成“泛化能力已充分验证”。
- 不再只看完成率而忽略 `PC_Wh` 上升。
- 不再使用含义不清的本地 `TAT` 去判断是否已经完全对齐论文。

当前应采用的表述：

- 本地论文式联合训练已经获得较高任务完成率。
- 任务完成率提升伴随功耗上升，属于完成率和能耗之间的折中。
- 当前低 `CR/PCR/WCR` 中有相当一部分可能来自场景生成阶段的不可观测任务，而不是模型本身不会调度。
- 后续目标应先在场景生成阶段筛掉物理上不可观测的任务点位，再讨论模型如何继续提高任务成功率。

## 后续任务

1. 检查当前场景生成流程，重点看 `tools/generate_constellations_and_tasksets.py` 以及相关 taskset/constellation 生成逻辑。
2. 设计任务点位可观测性筛选：对随机生成的候选任务，判断在对应星座和时间窗内是否存在至少一次有效观测机会。
3. 统计现有评估结果中不可观测任务占比，优先区分 `never_visible`、`energy_or_sensor_blocked`、`assigned_but_not_completed` 等失败原因。
4. 生成新的筛选版场景或任务集，并记录筛选前后任务数量、可观测任务比例和 split 分布。
5. 用筛选版场景重新评估模型时，必须同时列出 `CR`、`PCR`、`WCR`、`PC_Wh` 和 `CS_no_TAT`，并明确说明这是“可观测性过滤后”的评估口径。
6. 新实验结果必须记录到报告或独立 summary 中，避免再次把长期过程记录堆进 `TODO.md`。

## 当前托管任务

- `taskset_filter_full_eval_4x_20260622_1236_r0` 到 `taskset_filter_full_eval_4x_20260622_1236_r3`：已完成 4 路并行重建完整评估 split 的筛选版 `tasksets`，包括 `val_seen=500`、`val_unseen=500`、`test=1000`。旧任务集已归档到 `data/tasksets_unfiltered_20260622_122858`，新任务集输出到 `data/tasksets`，日志为 `work_dirs/taskset_filtering_logs/taskset_filter_full_eval_4x_20260622_1236_r*.log`。相关脚本集中放在 `scripts/taskset_filtering/`。
- `stage3_observable_filtered_eval_20260622`：已完成最新 `paper_joint_stage3_200k` checkpoint 的筛选版 `tasksets` 评估，脚本为 `scripts/eval_observable_filtered/run_stage3_200k_96core_eval.sh`，输出目录为 `work_dirs/rl_eval_paper_joint_stage3_200k_96core_*_observable_filtered/`，汇总文件为 `work_dirs/eval_summaries/paper_joint_stage3_200k_no_tat_96core_observable_filtered.json`，结果摘要见 `docs/observable_filtered_stage3_eval_summary.md`。

## 关键文档

- `README.md`：项目入口、环境、数据、训练和评估说明。
- `docs/实验复现报告.md`：当前复现实验结论和汇报口径。
- `docs/aeos_former_shape_flow.md`：AEOS-Former 输入输出和张量形状流图。
- `docs/AEOSFormer_Encoder_解析.md`：AEOS-Former 架构解释稿。
- `docs/constellation_code_structure.md`：`constellation/` 代码结构和训练流程说明。
- `docs/new_transformers_dataset_model.md`：`Dataset`、`JointDataset`、`Model`、`JointModel` 的关系说明。
