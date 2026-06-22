# TODO

## 当前目标

当前项目目标从“继续堆叠旧复现实验记录”调整为：

> 在保持论文复现口径清楚、实验记录可追溯的前提下，重点提高 AEOS-Former 在正式评估场景中的任务成功率。

这里的“任务成功率”优先对应 `CR`、`PCR`、`WCR` 等任务完成相关指标。后续写报告或答辩时，应同时说明功耗代价，避免把“完成率提高”表述成“所有指标全面更优”。

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
- 后续目标是在这个基础上继续提高任务成功率，并保持评估口径、功耗代价和论文对齐关系可解释。

## 后续任务

1. 围绕“提高任务成功率”设计下一轮实验目标。
2. 在正式实验前明确对应 split、checkpoint、评估并行数和指标口径。
3. 新实验结果必须记录到报告或独立 summary 中，避免再次把长期过程记录堆进 `TODO.md`。
4. 如需比较新旧模型，必须同时列出 `CR`、`PCR`、`WCR`、`PC_Wh` 和 `CS_no_TAT`。

## 关键文档

- `README.md`：项目入口、环境、数据、训练和评估说明。
- `docs/实验复现报告.md`：当前复现实验结论和汇报口径。
- `docs/aeos_former_shape_flow.md`：AEOS-Former 输入输出和张量形状流图。
- `docs/AEOSFormer_Encoder_解析.md`：AEOS-Former 架构解释稿。
- `docs/constellation_code_structure.md`：`constellation/` 代码结构和训练流程说明。
- `docs/new_transformers_dataset_model.md`：`Dataset`、`JointDataset`、`Model`、`JointModel` 的关系说明。
