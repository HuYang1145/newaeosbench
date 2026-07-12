# GNN 调度参考实现

本目录只保存外部研究参考，不直接参与 AEOS-Former 的训练或推理。

## Wheatley

- 路径：`third_party/gnn_references/wheatley`
- 上游：`https://github.com/jolibrain/wheatley`
- 当前固定版本：`7eca9dd4e77d7337f4b670b3ccb8235b3b707e60`
- 许可证：Apache-2.0

Wheatley 是 GNN + PPO 的通用 JSSP/RCPSP 调度框架，也是卫星 GNN 论文作者所参考的
基础框架。它不是《Earth Observation Satellite Scheduling with Graph Neural
Networks》的完整卫星实现，不能直接替换 AEOS-Former。

## 论文

- `docs/papers/earth_observation_satellite_scheduling_with_gnns_ewrl2024.pdf`：
  arXiv v1，对应 EWRL 2024 原始方法。
- `docs/papers/earth_observation_satellite_scheduling_with_gnns.pdf`：
  当前 arXiv 版本，标题和方法已包含后续更新。

后续只借鉴图表示、消息传递和 PPO 训练结构；卫星—任务候选边不得依赖在线
Basilisk 或重型物理预测。
