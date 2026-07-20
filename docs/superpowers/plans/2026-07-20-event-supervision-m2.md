# M2 Event Supervision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 使用现有轨迹为事件式 Actor 增加因果对齐的 continue、duration 与短窗口
事实结果监督，并完成冻结 Stage3 主干的真实样本训练 smoke。

**Architecture:** 复用 Temporal Adapter 的因果历史和卫星—任务边隐藏特征，新增
continue/duration heads；Stage3 logits residual 固定为零。标签来自专家动作连续段
和既有 censored outcome builder，idle 保持 1 秒。

**Tech Stack:** Python 3.11、PyTorch、pytest、Todd trainer、Slurm、现有
Stage3-200k checkpoint 与轨迹。

---

### Task 1：事件 continue/duration 标签

**Files:**
- Modify: `constellation/new_transformers/multi_horizon_edge_labels.py`
- Modify: `tests/test_temporal_outcomes.py`

- [x] **Step 1: 写失败测试**

覆盖 `1/5/15/30/60` 向下分桶、下一秒 continue、idle mask，以及延伸到轨迹末尾
且小于 60 秒的 duration censor。

- [x] **Step 2: 验证 RED**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q \
  tests/test_temporal_outcomes.py -k event_supervision
```

Expected: FAIL，原因是 `build_event_supervision()` 尚不存在。

- [x] **Step 3: 实现 `EventSupervisionTensors`**

新增 `valid/continue_next/duration_index/duration_observed/
remaining_run_lengths`，承诺集合固定为 `(1, 5, 15, 30, 60)`。

- [x] **Step 4: 验证 GREEN 并提交**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q \
  tests/test_temporal_outcomes.py
```

Expected: PASS。

### Task 2：Dataset 传递事件标签

**Files:**
- Modify: `constellation/new_transformers/dataset.py`
- Modify: `tests/test_temporal_dataset.py`
- Modify: `tests/test_temporal_model.py`

- [x] **Step 1: 写失败测试**

断言 `TemporalBatch` 在采样时间上给出正确的
`event_continue/event_duration_index/event_duration_observed`，并确认历史输入仍只
来自 `actions[:t]`。

- [x] **Step 2: 验证 RED**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q \
  tests/test_temporal_dataset.py
```

Expected: FAIL，原因是 `TemporalBatch` 尚无事件字段。

- [x] **Step 3: 最小接入 Dataset**

在 `_build_temporal_batch()` 中调用 `build_event_supervision()`；任务裁剪只映射
输入任务索引，不改变逐卫星事件标签。

- [x] **Step 4: 验证 GREEN 并提交**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q \
  tests/test_temporal_dataset.py tests/test_temporal_model.py
```

Expected: PASS。

### Task 3：continue/duration heads 与 masked loss

**Files:**
- Modify: `constellation/new_transformers/temporal_adapter.py`
- Modify: `tests/test_temporal_adapter.py`

- [x] **Step 1: 写失败测试**

覆盖输出形状、按已执行任务 gather、全 idle/全 censored 返回可反传零，以及非法
duration index 拒绝。

- [x] **Step 2: 验证 RED**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q \
  tests/test_temporal_adapter.py -k event
```

Expected: FAIL，原因是事件 heads/loss 不存在。

- [x] **Step 3: 实现 heads 与 loss**

从 `edge_hidden` 输出 `continue_logits` 和五档 `duration_logits`；损失只在真实执行
非空边和 observed duration 上计算。

- [x] **Step 4: 验证 GREEN 并提交**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q \
  tests/test_temporal_adapter.py
```

Expected: PASS。

### Task 4：JointModel 冻结训练接入

**Files:**
- Modify: `constellation/new_transformers/model.py`
- Create: `constellation/new_transformers/config_event_heads_m2.py`
- Modify: `tests/test_temporal_model.py`

- [x] **Step 1: 写失败测试**

断言 M2 总 loss 包含 continue/duration，Stage3 主干无梯度，residual scale 为零时
logits 精确兼容，配置只训练 Temporal Adapter。

- [x] **Step 2: 验证 RED**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q \
  tests/test_temporal_model.py -k m2
```

Expected: FAIL，原因是 JointModel 尚无 M2 loss 参数和配置。

- [x] **Step 3: 接入损失和配置**

配置 horizons 为 `(5, 15, 30, 60)`、residual scale 为 `0`、主干冻结、动作/TimeModel
loss 权重为 `0`，M2 六类监督权重初始为 `1`。

- [x] **Step 4: 验证 GREEN 并提交**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q \
  tests/test_temporal_model.py
```

Expected: PASS。

### Task 5：标签审计与真实样本训练 smoke

**Files:**
- Create: `tools/audit_event_supervision_m2.py`
- Create: `tests/test_audit_event_supervision_m2.py`
- Create: `scripts/train_event_heads_m2_slurm.sh`
- Create: `tests/test_event_heads_m2_scripts.py`

- [x] **Step 1: TDD 实现审计汇总**

输出 continue 正负数、五档 duration 数量、duration censor 比例，以及既有 outcome
覆盖。工具只读取轨迹与 taskset，不运行 Basilisk。

- [x] **Step 2: 对少量真实 Stage3 annotation 运行审计**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python \
  tools/audit_event_supervision_m2.py \
  --annotation-file data/annotations/train_paper_stage3_tau_e_existing.json \
  --split train --limit 16 \
  --output work_dirs/event_supervision_m2_preflight/label_audit_16.json
```

Expected: 输出非空且五档计数之和等于 observed duration 数量。

- [x] **Step 3: 完成真实 Dataset 单 batch forward/backward/step**

使用 Stage3-200k checkpoint，在 CPU 或空闲 GPU 上确认 loss 有限，只有新模块参数
变化。

- [x] **Step 4: 准备 Slurm 包装**

脚本使用 `aeos` 环境、`local-10/lab_team`、独立日志和
`work_dirs/event_heads_m2_10k`，正式运行前检查 checkpoint。

- [x] **Step 5: 回归并提交**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q \
  tests/test_temporal_outcomes.py \
  tests/test_temporal_dataset.py \
  tests/test_temporal_adapter.py \
  tests/test_temporal_model.py \
  tests/test_event_action.py \
  tests/test_event_policy.py
bash -n scripts/train_event_heads_m2_slurm.sh
git diff --check
```

Expected: 全部通过。

### Task 6：记录 M2-A 结果并决定训练

**Files:**
- Modify: `TODO.md`
- Modify: `改进日志.md`

- [ ] **Step 1: 记录标签分布和 smoke**

明确区分行为标签与事实结果标签，记录 checkpoint、annotation、样本数、loss 和冻结
参数检查。

- [ ] **Step 2: 应用训练门槛**

只有标签非退化、真实 batch 可训练且资源可用时才提交 Slurm；否则停止在 M2-A，
先修复标签。

- [ ] **Step 3: 文档与回归检查**

Run:

```bash
git diff --check -- TODO.md 改进日志.md
```

Expected: 无空白错误，未把 M2-A 写成性能提升。
