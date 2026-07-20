# 事件式 Actor M0/M1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** 收束现有 Temporal Adapter 与局部 Graph-Q 实验，在保留 Stage3-200k
兼容路径的前提下，实现无需重新训练的事件式 Actor，使非空任务动作持续到任务失效
或固定承诺到期。

**Architecture:** Basilisk 仍以 1 秒物理步长推进；新增纯 Python/PyTorch 的
`EventAssignmentState` 和 `EventActorRuntime`，只在任务失效、承诺到期、任务集
变化或初始状态接受新的局部动作。第一轮使用固定的 `1/5/15/30/60 s` 承诺做因果
消融，不引入 duration head、PPO、在线候选 Basilisk 或硬 owner 分配。真实 smoke
确认 42 星联合 Actor 的全局前向次数不会随局部任务承诺等比例下降，因此“局部动作
更新”和“全局模型前向”作为两个指标分别验收。

**Tech Stack:** Python 3.11、PyTorch、pytest、现有
`Controller + BasiliskEnvironment + TaskManager + Evaluators`。

---

## 文件结构

- `TODO.md`：把活动研究路线切换为 M0–M5，历史 P 编号只作为实验索引保留。
- `改进日志.md`：记录 Temporal Adapter 和 P3.1 局部 Graph-Q 的最终结果，以及
  M0/M1 的设计、实现和 smoke 结论。
- `constellation/new_transformers/event_action.py`：定义事件承诺值对象和每星状态机。
- `constellation/new_transformers/event_policy.py`：只在事件发生时调用 planner，
  维护每星重规划掩码。
- `tools/rollout_model_trajectories.py`：接入冻结 Stage3 的固定承诺事件式 Actor，
  输出事件行为统计。
- `tests/test_event_action.py`：状态机的到期、失效、空闲和单星重规划测试。
- `tests/test_event_policy.py`：planner 调用次数和部分重规划测试。
- `tests/test_rollout_model_candidates.py`：事件式 rollout、基线兼容和统计测试。
- `scripts/run_event_actor_m1_smoke.sh`：单场 CPU 协议 smoke。
- `scripts/run_event_actor_m1_val_slurm.sh`：后续正式 8+8/64+64 Val 的 Slurm 包装器。

### Task 1：保存 M0 修改前恢复点

**Files:**
- Stage: `TODO.md`
- Stage: `改进日志.md`
- Stage: `constellation/new_transformers/local_action_branch.py`
- Stage: `constellation/new_transformers/local_graph_q_critic.py`
- Stage: `tools/generate_local_action_branches.py`
- Stage: `tools/generate_local_graph_q_dataset.py`
- Stage: `tools/train_local_graph_q_critic.py`
- Stage: `tools/rollout_model_trajectories.py`
- Stage: `tests/test_*local*`

- [x] **Step 1: 记录当前分支、HEAD 和未提交边界**

Run:

```bash
git status --short --branch
git rev-parse HEAD
```

Expected: branch 为 `codex/offline-critic-ranking`，基线 HEAD 为 `71dc76d`。

- [x] **Step 2: 验证现有局部分支与 Graph-Q 代码**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q \
  tests/test_rollout_model_candidates.py \
  tests/test_generate_local_action_branches_tool.py \
  tests/test_generate_local_graph_q_dataset.py \
  tests/test_local_action_branch.py \
  tests/test_local_graph_q_critic.py \
  tests/test_train_local_graph_q_critic.py \
  tests/test_multi_horizon_edge_labels.py
```

Expected: `47 passed`。

- [x] **Step 3: 创建只包含研究代码的 checkpoint commit**

不得暂存 `.claude/settings.json`、`CLAUDE.md`、`dataset.py.bak_*` 或 Basilisk
AutoTeX。

Expected commit:

```text
0a760ee exp: checkpoint controlled local graph q pilot
```

### Task 2：合入 Temporal Adapter 因果历史能力

**Files:**
- Merge: `codex/p0-causal-history-adapter`
- Preserve: `constellation/new_transformers/temporal_history.py`
- Preserve: `constellation/rl/environment.py`
- Preserve: `constellation/rl/controller_environment.py`
- Preserve: `constellation/rl/policy.py`
- Resolve: `TODO.md`
- Resolve: `改进日志.md`

- [x] **Step 1: 合并经过验证的 Temporal Adapter 分支**

Run:

```bash
git merge --no-ff codex/p0-causal-history-adapter
```

Expected: 代码冲突只允许出现在双方都记录实验状态的文档；不得删除当前 Graph-Q
文件或 Temporal History 测试。

- [x] **Step 2: 解决文档冲突并保留两条实验历史**

保留以下事实：

```text
Temporal Adapter 10k 已训练，8+8 Val 的两个 CS_paper 都比同场景 Stage3 差约 0.074。
P3.1 局部 Graph-Q pilot 只有 18 个有效 pair、5 个 scene、1/4 fold 通过。
300/600 秒偏好一致率为 0.4545，decision=stop_before_actor_or_reranking。
```

- [x] **Step 3: 运行合并回归**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q \
  tests/test_temporal_history.py \
  tests/test_temporal_policy.py \
  tests/test_temporal_adapter.py \
  tests/test_temporal_model.py \
  tests/test_local_action_branch.py \
  tests/test_local_graph_q_critic.py
```

Expected: 全部通过。

### Task 3：把活动路线从 P 切换为 M

**Files:**
- Modify: `TODO.md`
- Modify: `改进日志.md`

- [x] **Step 1: 在 TODO 顶部声明编号语义**

写入：

```markdown
## 当前路线：事件式 Actor + 局部监督 + APPO/PPO

从 2026-07-20 起，活动路线统一使用 M0–M5。旧的 P0/P0.1/P2/P3.x 只作为已经运行
实验及产物目录的历史编号保留，不再继续扩展。
```

- [x] **Step 2: 把已完成和停止项写入 M0**

M0 必须记录 checkpoint、Temporal Adapter 结果、局部 Graph-Q 结果和分支合并状态。

- [x] **Step 3: 写入 M1 验收清单**

M1 包含：

```text
1. Stage3 关闭事件模式时动作完全兼容。
2. 事件模式只在初始、承诺到期、任务失效或 taskset 唤醒时重规划。
3. Basilisk 仍每秒推进。
4. 支持 1/5/15/30/60 秒固定承诺。
5. idle 默认只承诺 1 秒；多秒 idle 仅用于带 taskset 唤醒的消融。
6. 先完成单场 smoke，再申请 Slurm 运行同场景 Val。
```

- [x] **Step 4: 文档检查**

Run:

```bash
git diff --check -- TODO.md 改进日志.md
```

Expected: 无空白错误，活动待办中不再使用新 P 编号。

### Task 4：事件承诺状态机

**Files:**
- Create: `tests/test_event_action.py`
- Create: `constellation/new_transformers/event_action.py`

- [x] **Step 1: 写失败测试**

核心测试：

```python
def test_event_state_counts_down_and_replans_at_expiry() -> None:
    state = EventAssignmentState.empty(num_satellites=2)
    state.start([
        EventDecision(task_id=3, commitment_seconds=5),
        EventDecision(task_id=-1, commitment_seconds=1),
    ], start_time=10)

    assert state.advance(time=11, ongoing_task_ids={3}) == [False, True]
    assert state.advance(time=15, ongoing_task_ids={3}) == [True, True]
```

还要覆盖：

```text
任务失效提前中断
只替换需要重规划的卫星
idle 默认 1 秒，多秒消融只允许在 taskset 变化时立即唤醒
不支持的承诺时长被拒绝
时间倒退被拒绝
```

- [x] **Step 2: 验证 RED**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q \
  tests/test_event_action.py
```

Expected: FAIL，原因是 `event_action` 尚不存在。

- [x] **Step 3: 实现最小状态机**

公开 API：

```python
from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence

import torch

ALLOWED_EVENT_COMMITMENTS = (1, 5, 15, 30, 60)

@dataclasses.dataclass(frozen=True)
class EventDecision:
    task_id: int
    commitment_seconds: int

    def __post_init__(self) -> None:
        if self.commitment_seconds not in ALLOWED_EVENT_COMMITMENTS:
            raise ValueError("unsupported event commitment")
        if self.task_id < -1:
            raise ValueError("task_id must be -1 or non-negative")
@dataclasses.dataclass
class EventAssignmentState:
    task_ids: torch.Tensor
    remaining_seconds: torch.Tensor
    start_times: torch.Tensor
    last_update_times: torch.Tensor
    interruption_reasons: list[str | None]

    @classmethod
    def empty(cls, num_satellites: int) -> "EventAssignmentState":
        if num_satellites <= 0:
            raise ValueError("num_satellites must be positive")
        return cls(
            task_ids=torch.full((num_satellites,), -1, dtype=torch.long),
            remaining_seconds=torch.zeros(num_satellites, dtype=torch.long),
            start_times=torch.full((num_satellites,), -1, dtype=torch.long),
            last_update_times=torch.full(
                (num_satellites,), -1, dtype=torch.long
            ),
            interruption_reasons=[None] * num_satellites,
        )

    def assignment(self) -> list[int]:
        return [int(value) for value in self.task_ids.tolist()]

    def start(
        self,
        decisions: Sequence[EventDecision],
        *,
        start_time: int,
    ) -> None:
        if len(decisions) != self.task_ids.numel():
            raise ValueError("one decision is required per satellite")
        for satellite_index, decision in enumerate(decisions):
            self.replace(
                satellite_index,
                decision,
                start_time=start_time,
            )

    def replace(
        self,
        satellite_index: int,
        decision: EventDecision,
        *,
        start_time: int,
    ) -> None:
        if not 0 <= satellite_index < self.task_ids.numel():
            raise IndexError("satellite_index is out of range")
        self.task_ids[satellite_index] = decision.task_id
        self.remaining_seconds[
            satellite_index
        ] = decision.commitment_seconds
        self.start_times[satellite_index] = start_time
        self.last_update_times[satellite_index] = start_time
        self.interruption_reasons[satellite_index] = None

    def advance(
        self,
        *,
        time: int,
        ongoing_task_ids: Iterable[int],
    ) -> list[bool]:
        ongoing = {int(task_id) for task_id in ongoing_task_ids}
        replans: list[bool] = []
        for satellite_index, task_id in enumerate(self.assignment()):
            last_update = int(self.last_update_times[satellite_index])
            if last_update < 0:
                replans.append(True)
                continue
            if time < last_update:
                raise ValueError("time must be monotonic")
            elapsed = time - last_update
            remaining = max(
                int(self.remaining_seconds[satellite_index]) - elapsed,
                0,
            )
            self.remaining_seconds[satellite_index] = remaining
            self.last_update_times[satellite_index] = time
            if task_id >= 0 and task_id not in ongoing:
                self.remaining_seconds[satellite_index] = 0
                self.interruption_reasons[
                    satellite_index
                ] = "task_unavailable"
                replans.append(True)
            elif remaining == 0:
                self.interruption_reasons[satellite_index] = "expired"
                replans.append(True)
            else:
                replans.append(False)
        return replans
```

- [x] **Step 4: 验证 GREEN**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q \
  tests/test_event_action.py
```

Expected: PASS。

- [x] **Step 5: 提交状态机**

```bash
git add constellation/new_transformers/event_action.py tests/test_event_action.py
git commit -m "feat: add event assignment state machine"
```

### Task 5：事件式 Actor runtime

**Files:**
- Create: `tests/test_event_policy.py`
- Create: `constellation/new_transformers/event_policy.py`

- [x] **Step 1: 写 planner 调用次数失败测试**

```python
def test_runtime_skips_planner_before_event() -> None:
    runtime = EventActorRuntime(num_satellites=1)
    calls = 0

    def planner(active, previous):
        nonlocal calls
        calls += 1
        return [EventDecision(7, 5)]

    assert runtime.update(time=0, ongoing_task_ids={7}, planner=planner) == [7]
    assert runtime.update(time=1, ongoing_task_ids={7}, planner=planner) == [7]
    assert calls == 1
```

还要覆盖任务失效只重规划对应卫星，仍有效的其他卫星保持剩余承诺。

- [x] **Step 2: 验证 RED**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q \
  tests/test_event_policy.py
```

Expected: FAIL，原因是 `EventActorRuntime` 尚不存在。

- [x] **Step 3: 实现最小 runtime**

```python
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import torch

from .event_action import EventAssignmentState, EventDecision

class EventActorRuntime:
    def __init__(self, *, num_satellites: int) -> None:
        self.state = EventAssignmentState.empty(num_satellites)
        self.replan_count = 0

    def update(
        self,
        *,
        time: int,
        ongoing_task_ids: Iterable[int],
        planner: Callable[
            [torch.Tensor, torch.Tensor],
            Sequence[EventDecision],
        ],
    ) -> list[int]:
        replans = self.state.advance(
            time=time,
            ongoing_task_ids=ongoing_task_ids,
        )
        if not any(replans):
            return self.state.assignment()
        active = ~torch.tensor(replans, dtype=torch.bool)
        decisions = list(planner(active, self.state.task_ids.clone()))
        if len(decisions) != self.state.task_ids.numel():
            raise ValueError("planner must return one decision per satellite")
        for satellite_index, should_replan in enumerate(replans):
            if not should_replan:
                continue
            self.state.replace(
                satellite_index,
                decisions[satellite_index],
                start_time=time,
            )
            self.replan_count += 1
        return self.state.assignment()
```

planner 仍返回每颗卫星一个 decision，但 runtime 只替换 `replan=True` 的卫星。

- [x] **Step 4: 验证 GREEN 并提交**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q \
  tests/test_event_action.py tests/test_event_policy.py
```

Expected: PASS。

Commit:

```bash
git add constellation/new_transformers/event_policy.py tests/test_event_policy.py
git commit -m "feat: add event actor runtime"
```

### Task 6：冻结 Stage3 的固定承诺 rollout

**Files:**
- Modify: `tests/test_rollout_model_candidates.py`
- Modify: `tools/rollout_model_trajectories.py`

- [x] **Step 1: 写失败测试**

覆盖：

```text
--event-actor 必须显式提供合法 commitment。
不启用 event actor 时继续逐秒调用 Stage3。
启用后，5 秒承诺期间只调用一次模型。
任务从 ongoing 集合消失时立即重规划。
idle 默认使用 1 秒承诺；多秒 idle 在 taskset 变化时立即重规划。
输出 task_one_second_commitment_rate、task_mean_commitment_seconds、
model_call_count 和 interruption reason。
```

- [x] **Step 2: 验证 RED**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q \
  tests/test_rollout_model_candidates.py
```

Expected: 新增事件式测试失败，旧测试仍通过。

- [x] **Step 3: 最小接入**

新增参数：

```text
--event-actor
--event-commitment-seconds {1,5,15,30,60}
--event-idle-commitment-seconds {1,5,15,30,60}
```

新增 `EventGreedyModelAlgorithm`：

```text
planner 被调用时执行一次现有 Stage3 greedy；
非空 task 使用固定 commitment；
idle 默认使用 1 秒，多秒 idle 仅作带 taskset 唤醒的消融；
每个物理秒仍返回当前缓存 assignment 给 Controller；
任务失效时 runtime 提前触发 planner；
不调用在线 Basilisk 候选搜索。
```

- [x] **Step 4: 验证 GREEN**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q \
  tests/test_rollout_model_candidates.py \
  tests/test_event_action.py \
  tests/test_event_policy.py
```

Expected: PASS。

- [x] **Step 5: 提交 rollout 接入**

```bash
git add tools/rollout_model_trajectories.py \
  tests/test_rollout_model_candidates.py
git commit -m "feat: run stage3 actor on scheduling events"
```

### Task 7：M1 smoke 与 Slurm 包装

**Files:**
- Create: `scripts/run_event_actor_m1_smoke.sh`
- Create: `scripts/run_event_actor_m1_val_slurm.sh`
- Modify: `TODO.md`
- Modify: `改进日志.md`

- [x] **Step 1: 写脚本静态测试**

测试脚本必须：

```text
使用 /home/hy/miniconda3/envs/aeos/bin/python
使用 Stage3-200k checkpoint
输出到独立 event_actor_m1_* 目录
正式 Val 通过 Slurm
不使用 Test
```

- [x] **Step 2: 运行单场 CPU smoke**

以 train scene 0 依次运行 baseline 1 秒与 event 5 秒，确认：

```text
完整 3600 秒场景能结束；
任务承诺期间不覆盖该卫星动作；
分别统计全局 model_call_count 和局部 satellite_replan_count；
任务失效没有崩溃；
metrics 和 macro/event behavior 均落盘。
```

- [x] **Step 3: 运行完整定向回归**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q \
  tests/test_event_action.py \
  tests/test_event_policy.py \
  tests/test_rollout_model_candidates.py \
  tests/test_temporal_history.py \
  tests/test_temporal_policy.py \
  tests/test_local_action_branch.py \
  tests/test_local_graph_q_critic.py
```

Expected: 全部通过。

- [x] **Step 4: 记录结果并提交**

```bash
git add TODO.md 改进日志.md scripts/run_event_actor_m1_*.sh
git commit -m "docs: record event actor m1 smoke"
```

## M1 停止门槛

本计划只宣布“事件式执行机制实现成功”，不提前宣布模型性能提升。进入 8+8 Val
需要同时满足：

```text
关闭事件模式时兼容 Stage3；
单场真实 smoke 无错误；
5 秒模式显著减少非空任务的一秒承诺；
没有完成/失效任务被继续引用；
新增路径不调用在线候选 Basilisk；
定向测试全部通过。
```

真实 smoke 发现：42 星联合 Actor 只要任一响应式 idle 卫星需要规划，就会执行
一次全局 Transformer 前向，所以局部事件承诺不能保证 `model_call_count` 同比例
下降。task 5 秒 / idle 1 秒的任务一秒承诺率为 `0%`，但模型调用仅降至 `3,580`；
task 5 秒 / idle 5 秒虽降至 `3,056`，却使 `CS_paper` 恶化到 `3.8048`。因此 M1
机制完成，但不进入 8+8 Val，M2 需要学习终止与持续时间，而不是继续手工延长 idle。

进入 M2 或正式完整 Val 还需要后续同场景实验确认：

```text
一秒非空片段率相对下降至少 20%；
CR/PCR/WCR 任一项下降不超过 0.5 个百分点；
Val Seen/Unseen 的 CS_paper 不恶化；
重复率、TAT_s、PC_Wh 和推理耗时完整报告。
```
