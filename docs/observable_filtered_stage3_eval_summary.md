# Stage3-200k 可观测性过滤后评估摘要

## 评估设置

- 模型：`work_dirs/paper_joint_stage3_200k/checkpoints/iter_200000/model.pth`
- 评估脚本：`scripts/eval_observable_filtered/run_stage3_200k_96core_eval.sh`
- 新任务集：`data/tasksets`，已用快速物理几何筛选重建 `val_seen=500`、`val_unseen=500`、`test=1000`
- 旧任务集归档：`data/tasksets_unfiltered_20260622_122858`
- 并行环境：`environment.world_size=96`
- 新结果：`work_dirs/eval_summaries/paper_joint_stage3_200k_no_tat_96core_observable_filtered.json`
- 对照结果：`work_dirs/eval_summaries/paper_joint_stage3_200k_no_tat_96core_managed.json`

## 指标对比

| Split | 口径 | CR (%) | PCR (%) | WCR (%) | PC_Wh | CS_no_TAT |
|---|---|---:|---:|---:|---:|---:|
| val_seen | unfiltered | 37.4989 | 40.7907 | 37.2899 | 95.1519 | 3.5751 |
| val_seen | observable-filtered | 38.1761 | 41.3861 | 37.9531 | 94.0937 | 3.5200 |
| val_seen | delta | +0.6773 | +0.5954 | +0.6632 | -1.0582 | -0.0551 |
| val_unseen | unfiltered | 42.8164 | 46.2528 | 42.7816 | 112.4270 | 3.4233 |
| val_unseen | observable-filtered | 42.5057 | 46.0574 | 42.4415 | 111.6739 | 3.4314 |
| val_unseen | delta | -0.3107 | -0.1954 | -0.3402 | -0.7531 | +0.0081 |
| test | unfiltered | 23.2786 | 26.0220 | 23.0216 | 52.1138 | 4.7271 |
| test | observable-filtered | 23.0693 | 25.6117 | 22.5694 | 51.9190 | 4.7785 |
| test | delta | -0.2093 | -0.4103 | -0.4522 | -0.1947 | +0.0514 |

## 初步结论

这次可观测性过滤后，`val_seen` 的 `CR/PCR/WCR` 小幅上升，`PC_Wh` 下降，`CS_no_TAT` 变好；但 `val_unseen` 和 `test` 的完成率没有同步上升，反而略低，功耗也略降。

因此，当前结果不能简单写成“过滤无解任务后完成率全面提升”。更稳妥的解释是：这次筛选改变了任务点分布，并且保留了原场景任务数量；它消除了几何不可观测点，但也可能让剩余任务更集中于真实可竞争的观测窗口。对 Stage3-200k 模型而言，筛选后 `val_seen` 改善明显一些，`val_unseen/test` 变化很小且方向不一致。

后续如果要判断“原低完成率中有多少来自无解点”，应额外统计旧 `tasksets_unfiltered_20260622_122858` 中 `never_visible` 任务占比，并与新 `tasksets` 的可观测比例并列报告，而不能只依赖模型重评估后的 `CR/PCR/WCR` 差值。
