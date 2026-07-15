# Multi-Horizon Edge Label Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从现有 Stage3 greedy 轨迹生成 `5/15/30 s` 卫星—任务边标签并输出可复现审计 JSON。

**Architecture:** 标签库只消费动作、可见性、任务进度和任务 duration，并使用连续同任务动作段处理 censor。CLI 负责发现轨迹、匹配 taskset JSON、累计计数和写出结果，不修改 Actor 或调用 Basilisk。

**Tech Stack:** Python 3.11、PyTorch、pytest、现有 `aeos` Conda 环境。

---

### Task 1: 因果标签与 censor 规则

**Files:**
- Create: `tests/test_multi_horizon_edge_labels.py`
- Create: `constellation/new_transformers/multi_horizon_edge_labels.py`

- [ ] 先写测试：验证 `action[t] -> visible[t+1]`、连续执行满窗口的负样本、提前切换的 censored 样本和窗口内正样本。
- [ ] 运行 `/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_multi_horizon_edge_labels.py -q`，确认因模块不存在而失败。
- [ ] 实现 `summarize_trajectory_edge_labels()`，返回只含可加总整数的摘要。
- [ ] 重跑测试，确认通过。

### Task 2: 重复与完成标签

**Files:**
- Modify: `tests/test_multi_horizon_edge_labels.py`
- Modify: `constellation/new_transformers/multi_horizon_edge_labels.py`

- [ ] 先写测试：同一步两颗卫星选择同一任务时统计 duplicate edge，并区分本卫星下一步是否直接可见。
- [ ] 写完成事件测试：`progress[t] < duration` 且窗口内 `progress >= duration`。
- [ ] 运行测试确认失败，再实现最小逻辑并确认通过。

### Task 3: 加权聚合与 CLI

**Files:**
- Modify: `tests/test_multi_horizon_edge_labels.py`
- Create: `tools/audit_multi_horizon_edge_labels.py`

- [ ] 先写 `aggregate_edge_label_summaries()` 的失败测试，确保比例由总计数重算。
- [ ] 实现聚合函数并确认测试通过。
- [ ] 实现 CLI：参数为 trajectory root、taskset root、limit、horizons 和 output；按相对路径匹配 taskset JSON。
- [ ] 运行 `py_compile`、CLI `--help` 和完整定向测试。

### Task 4: 真实 16 场审计

**Files:**
- Create: `work_dirs/multi_horizon_edge_label_audit_stage3_16.json`

- [ ] 使用 `same_scene_candidates_stage3_200k_256/candidate_000_greedy` 前 16 场运行审计。
- [ ] 核对 scene count、原始计数、比例范围与输出配置。
- [ ] 运行 `git diff --check` 和相关回归测试，记录结果与阶段 A 结论。

