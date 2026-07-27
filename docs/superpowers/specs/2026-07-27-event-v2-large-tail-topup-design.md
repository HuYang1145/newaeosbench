# Event V2-Large 尾部补采设计

## 目标

修复 V2-2-Large 在部分 actor 先完成场景时产生小于 64 events 的非终止 batch 并退出的问题，保证两个 seed 都完整训练 scenes `205–324`，随后按既定 heldout、Val Gate、完整 Val 和 Test 协议自动执行。

## 根因

训练轮开始时按活跃 actor 数计算每个 actor 的目标事件数。尾部有 actor 在轮内完成全部分配场景时，它无法提供完整目标事件；其余 actor 尚未全部完成，因此聚合事件数可能小于 `min_update_events=64`。旧代码只允许“至少 64 events”或“所有 actor 同时完成”，所以 seed 5408 在 58 events、seed 5409 在 61 events 时触发保护性异常。

## 设计

1. 一个 learner update 可以由一个或多个连续严格同步采样轮组成，但所有采样轮必须使用同一个 `policy_version`。
2. 第一采样轮继续使用原有目标：`max(events_per_actor_round, ceil(min_update_events / active_actors))`。
3. 若部分 actor 完成且累计 events 不足 64，立即移除已完成 actor，让剩余 actor 按 `ceil(缺口 / 剩余 actor 数)` 补采；模型在补采期间不更新、不发布新 policy。
4. 聚合器校验 policy version、连续 round、actor 集合只缩不增、transition 不重复，并累计 events、物理秒和重放误差。
5. 累计 events 达到 64 后只执行一次 PPO update。若所有 actor 已完成且最终累计仍不足 64，则保持现有协议，记录并跳过最后残余 batch。
6. 只有 PPO update 完成或最终残余 batch 已明确跳过时才允许保存 checkpoint。异常发生在待补采状态时保留上一个安全 checkpoint，避免 actor 状态已前进但 events 未用于训练。

## 恢复与后续作业

- 两个 seed 均从永久 checkpoint `checkpoint_update_001600.pth` 恢复，重新生成故障附近的 events。
- 不删除已有 checkpoint、日志或 metrics；一次性通过 Slurm 环境变量指定恢复点，后续若遇到 6 小时时限则继续使用新的 `checkpoint_latest.pth`。
- 训练成功后严格串联 heldout checkpoint selection、Val Seen/Unseen 8+8 Gate、完整 Val 64+64 和唯一一次 Test。任一阶段失败，后续 `afterok` 作业不启动。

## 验证

- 单元测试先复现 58 events 加 7 个存活 actor 的场景，证明旧实现不能形成 update batch。
- 验证补采后使用同一 policy version、事件无重复且累计达到至少 64。
- 验证全部 actor 结束时小残余 batch 仍不会触发 update。
- 运行同步协议、checkpoint、训练入口和 Slurm 包装脚本回归测试，并执行 shell 语法检查。
