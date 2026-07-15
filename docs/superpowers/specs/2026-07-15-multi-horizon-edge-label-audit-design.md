# 多时间尺度边际价值标签审计设计

## 目标

使用现有 Stage3 greedy 轨迹构造卫星—任务边的短期物理结果标签，验证
`5/15/30 s` 内的可见、进度、完成和无进度重复是否具有稳定信号。阶段 A 不修改
Stage3 Actor，不训练价值头，也不生成新的 Basilisk 轨迹。

## 样本与时间对齐

每个样本是 `(scene, time, satellite, executed_task)`，只统计非空动作。即时结果采用
`action[t] -> outcome[t+1]`。多时间尺度结果只沿该卫星连续执行同一任务的动作段观察：

- 窗口内出现事件则记为正样本；
- 连续执行满窗口仍无事件则记为负样本；
- 未等满窗口就切换且尚无事件则记为 censored，不作为负样本。

轨迹中的 task id 必须能直接索引 `is_visible` 和 task progress；否则拒绝该轨迹。

## 标签

即时标签包括下一步直接可见、下一步任务进度、下一步完成、重复选择和重复但本卫星
无直接可见。对 `5/15/30 s` 分别统计：完整可观察数、censored 数、窗口内首次可见、
窗口内首次进度、窗口内完成，以及首次事件等待时间。

任务完成由 `progress >= task.duration` 判断，duration 从对应 taskset JSON 读取。
重复标签不把所有多星选择都判错；本阶段只报告保守的“重复且本卫星下一步不可见”。

## 输出与门槛

CLI 对多个轨迹累计原始计数，再由总计数计算比例，禁止直接平均场景比例。输出 JSON
包含配置、逐场摘要和合并摘要，并按 `one_second_run`、`duplicate`、`other` 分层。

阶段 A 的作用是判断数据是否值得进入价值头训练。若 5/15/30 秒的大部分样本都被
censor，或物理正标签极少，则停止并转向短窗口反事实生成；不得把行为持续性标签直接
当成策略收益。

## 文件边界

- `constellation/new_transformers/multi_horizon_edge_labels.py`：纯标签和汇总逻辑；
- `tools/audit_multi_horizon_edge_labels.py`：文件发现、taskset 加载和 JSON 输出；
- `tests/test_multi_horizon_edge_labels.py`：因果对齐、censor、重复和聚合测试；
- `work_dirs/multi_horizon_edge_label_audit_stage3_16.json`：首轮真实审计结果。

