# P0 Causal History Temporal Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变 `SATELLITE_DIM=56`、不引入在线物理仿真、并保持旧 checkpoint 兼容的前提下，为 Stage3 Actor 增加因果历史输入、事实结果辅助监督和零初始化 Temporal Adapter。

**Architecture:** 使用一个纯函数 `temporal_history.py` 统一离线轨迹与在线环境的历史语义；`Dataset/JointDataset` 追加历史和 outcome 张量；`TemporalAdapter` 消费 Transformer 已有卫星/任务隐状态并输出有界残差与 outcome logits。功能关闭时完全走旧路径，功能开启但新头未训练时残差严格为零。

**Tech Stack:** Python 3.11、PyTorch、Gymnasium、Stable-Baselines3、pytest、todd runner/config。

---

## 文件结构

- Create: `constellation/new_transformers/temporal_history.py`：唯一的因果历史定义、前缀构造和在线状态机。
- Create: `constellation/new_transformers/temporal_adapter.py`：历史张量校验、边/空动作残差、outcome heads、masked loss。
- Modify: `constellation/new_transformers/multi_horizon_edge_labels.py`：公开批量事实结果张量，不复制现有标签语义。
- Modify: `constellation/new_transformers/dataset.py`：过滤无决策前状态/无下一结果的时间点，追加历史和 outcome 字段。
- Modify: `constellation/new_transformers/model.py`：接入可选 adapter、冻结主干模式和辅助 loss。
- Modify: `constellation/rl/environment.py`：扩展 observation schema、padding 与历史状态更新。
- Modify: `constellation/rl/controller_environment.py`：正式评估路径维护全局任务 ID 历史，包括 `_skip_idle()`。
- Modify: `constellation/rl/policy.py`：把 observation 历史字段转换为模型 batch。
- Modify: `constellation/rl/eval_all.py`：增加 Temporal Adapter CLI/config metadata。
- Create: `constellation/new_transformers/config_temporal_adapter_p0.py`：冻结主干的小规模 P0-B 训练配置。
- Create: `tests/test_temporal_history.py`：因果前缀、映射、reset/idle 的单元测试。
- Create: `tests/test_temporal_outcomes.py`：事实结果张量和 censor mask 测试。
- Create: `tests/test_temporal_adapter.py`：零残差、shape、mask、梯度和 masked loss 测试。
- Create: `tests/test_temporal_dataset.py`：Dataset 时间对齐、task pruning 和 future-invariance 测试。
- Create: `tests/test_temporal_model.py`：旧模型兼容、冻结范围和真实训练步测试。
- Create: `tests/test_temporal_policy.py`：在线 observation 到 Actor 输入的字段一致性测试。

## Task 1: 统一因果历史状态

**Files:**
- Create: `constellation/new_transformers/temporal_history.py`
- Create: `tests/test_temporal_history.py`

- [ ] **Step 1: 写 prefix run length 与 future-invariance 失败测试**

```python
def test_prefix_history_uses_only_actions_before_decision() -> None:
    actions = torch.tensor([[-1], [4], [4], [7], [7]])
    first = build_prefix_history(actions, time_steps=torch.tensor([3]))
    changed_future = actions.clone()
    changed_future[3:] = torch.tensor([[9], [9]])
    second = build_prefix_history(changed_future, time_steps=torch.tensor([3]))
    assert first.previous_global_task_ids.tolist() == [[4]]
    assert first.run_lengths.tolist() == [[2]]
    assert first == second
```

- [ ] **Step 2: 运行测试并确认因模块缺失而失败**

Run: `/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q tests/test_temporal_history.py`

Expected: collection fails with `ModuleNotFoundError: constellation.new_transformers.temporal_history`。

- [ ] **Step 3: 实现不可变历史张量与前缀构造**

```python
@dataclasses.dataclass(frozen=True)
class TemporalHistory:
    previous_global_task_ids: torch.Tensor
    previous_task_indices: torch.Tensor
    previous_task_available: torch.Tensor
    previous_was_idle: torch.Tensor
    run_lengths: torch.Tensor
    switch_count_30: torch.Tensor
    switch_count_60: torch.Tensor

def build_prefix_history(
    actions: torch.Tensor,
    time_steps: torch.Tensor,
    *,
    candidate_global_task_ids: torch.Tensor | None = None,
    candidate_mask: torch.Tensor | None = None,
) -> TemporalHistory:
    # 对每个 t 仅遍历 actions[:t]；t=0 返回 idle/zero。
    # previous_task_indices 为当前候选相对索引，不可用时为 -1。
```

实现必须校验 `(time, satellites)`、time 越界、候选 shape 和重复候选 ID；run length 在决策前累计，switch count 使用最多 30/60 个相邻动作转移。

- [ ] **Step 4: 写全局/相对 ID 映射与 idle 边界失败测试**

```python
def test_previous_task_mapping_marks_disappeared_task_unavailable() -> None:
    mapped, available = map_previous_tasks(
        torch.tensor([[8, -1]]),
        torch.tensor([[3, 8, 10]]),
        torch.tensor([[True, False, True]]),
    )
    assert mapped.tolist() == [[-1, -1]]
    assert available.tolist() == [[False, False]]
```

- [ ] **Step 5: 实现在线 `CausalAssignmentHistory` 状态机**

```python
class CausalAssignmentHistory:
    def __init__(self, num_satellites: int) -> None:
        self.reset(num_satellites)

    def reset(self, num_satellites: int | None = None) -> None:
        if num_satellites is not None:
            if num_satellites <= 0:
                raise ValueError('num_satellites must be positive')
            self._num_satellites = num_satellites
        self._assignments: deque[torch.Tensor] = deque(maxlen=61)

    def snapshot(
        self,
        candidate_global_task_ids: Sequence[int],
    ) -> TemporalHistory:
        actions = (
            torch.stack(tuple(self._assignments))
            if self._assignments else
            torch.empty((0, self._num_satellites), dtype=torch.long)
        )
        return build_prefix_history(
            actions,
            time_steps=torch.tensor([actions.shape[0]]),
            candidate_global_task_ids=torch.tensor(
                [list(candidate_global_task_ids)], dtype=torch.long
            ),
        )

    def record(self, global_task_ids: Sequence[int]) -> None:
        assignment = torch.as_tensor(global_task_ids, dtype=torch.long)
        if assignment.shape != (self._num_satellites,):
            raise ValueError('assignment must contain one task per satellite')
        self._assignments.append(assignment.clone())
```

`record()` 每调用一次代表真实执行一秒，包括全 idle；`snapshot()` 不修改状态；最近历史最多保留 61 个 assignment。

- [ ] **Step 6: 运行单元测试并提交**

Run: `/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q tests/test_temporal_history.py`

Expected: all tests pass。

Commit: `feat: add causal assignment history state`

## Task 2: 公开事实 outcome 张量与 censor 语义

**Files:**
- Modify: `constellation/new_transformers/multi_horizon_edge_labels.py`
- Create: `tests/test_temporal_outcomes.py`

- [ ] **Step 1: 写批量结果与提前切换 censor 的失败测试**

```python
def test_batched_outcomes_censor_switch_without_event() -> None:
    result = build_batched_edge_outcomes(
        actions=torch.tensor([[0], [0], [-1], [-1]]),
        is_visible=torch.zeros(4, 1, 1, dtype=torch.bool),
        progress=torch.zeros(4, 1),
        task_durations=torch.tensor([3]),
        horizons=(1, 3),
    )
    assert result.horizons[3].visible_observed[0, 0].item() is False
    assert result.horizons[1].visible_observed[0, 0].item() is True
```

- [ ] **Step 2: 运行并确认公开 API 缺失**

Run: `/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q tests/test_temporal_outcomes.py`

Expected: import fails for `build_batched_edge_outcomes`。

- [ ] **Step 3: 将现有 `_batched_outcomes` 包装为类型化公开 API**

```python
@dataclasses.dataclass(frozen=True)
class HorizonOutcomeTensors:
    visible: torch.Tensor
    visible_observed: torch.Tensor
    progress: torch.Tensor
    progress_observed: torch.Tensor
    completion: torch.Tensor
    completion_observed: torch.Tensor
    time_to_first_visible: torch.Tensor
    time_to_first_progress: torch.Tensor
    time_to_completion: torch.Tensor

@dataclasses.dataclass(frozen=True)
class BatchedEdgeOutcomes:
    valid: torch.Tensor
    visible_next: torch.Tensor
    progress_next: torch.Tensor
    completed_next: torch.Tensor
    horizons: dict[int, HorizonOutcomeTensors]
```

`summarize_trajectory_edge_labels()` 改为调用同一公开构造函数；不得保留第二套标签公式。

- [ ] **Step 4: 加入最长 300 秒事件时间测试**

测试刚切换边在第 45 秒首次可见时：5/15/30 可按窗口语义观测，300 秒事件时间为 45；若第 20 秒切换且无事件，300 秒 observed mask 为 false。

- [ ] **Step 5: 运行新旧标签测试并提交**

Run: `/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q tests/test_multi_horizon_edge_labels.py tests/test_temporal_outcomes.py`

Expected: all tests pass。

Commit: `feat: expose censored temporal outcome tensors`

## Task 3: Dataset 追加历史与事实标签

**Files:**
- Modify: `constellation/new_transformers/dataset.py`
- Create: `tests/test_temporal_dataset.py`
- Modify: `tests/test_joint_dataset_io.py`

- [ ] **Step 1: 写 t=0/t=last 排除与未来动作不变量失败测试**

```python
def test_temporal_dataset_samples_only_decisions_with_past_and_next() -> None:
    batch = build_fake_temporal_dataset(actions=[[-1], [2], [2], [-1]])[0]
    assert min(batch.time_steps) >= 1
    assert max(batch.time_steps) <= 2

def test_temporal_dataset_history_ignores_actions_at_and_after_t() -> None:
    first = build_at_time(actions=[[-1], [2], [2]], time=1)
    second = build_at_time(actions=[[-1], [9], [9]], time=1)
    torch.testing.assert_close(first.run_lengths, second.run_lengths)

def test_temporal_dataset_uses_previous_saved_satellite_state() -> None:
    batch = build_at_time(
        actions=[[-1], [2], [2]],
        time=1,
        satellite_rows=[10., 20., 30.],
    )
    assert batch.constellation_data[0, 0, -1].item() == 10.
```

- [ ] **Step 2: 运行并确认 `Batch` 不含历史字段**

Run: `/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q tests/test_temporal_dataset.py`

Expected: assertion/attribute failure for `previous_task_indices`。

- [ ] **Step 3: 扩展 `Batch/JointBatch` 尾部字段并保持旧构造兼容**

追加字段：

```python
previous_task_indices: torch.Tensor | None
previous_task_available: torch.Tensor | None
previous_was_idle: torch.Tensor | None
run_lengths: torch.Tensor | None
switch_count_30: torch.Tensor | None
switch_count_60: torch.Tensor | None
outcome_valid: torch.Tensor | None
visible_next: torch.Tensor | None
progress_next: torch.Tensor | None
completed_next: torch.Tensor | None
outcome_horizons: dict[int, HorizonOutcomeTensors] | None
```

为尾部字段设置 `None` defaults，使现有测试和旧调用仍可用；Dataset 实际返回非 `None` 张量。

- [ ] **Step 4: 在 `_build_batch()` 中使用统一历史和 outcome API**

有效采样时间改为 `tasks_mask.any(-1) & (t > 0) & (t < T - 1)`。任务状态继续使用当前决策时间 `t`，卫星动作后状态改取轨迹保存的 `t-1` 行，避免把 `action[t]` 已改变的传感器开关/姿态喂回同一动作。先以全局 taskset ID 构造历史/outcome，再在移除 never-valid tasks 后同步映射 `actions_task_id` 与 `previous_task_indices`；失效上一任务必须为 `-1/False`。

- [ ] **Step 5: 更新 JointDataset 返回顺序与 I/O 回归测试**

`JointBatch` 使用命名参数追加 temporal 字段；`test_joint_dataset_io.py` 继续断言轨迹、任务和卫星静态数据各只加载一次。

- [ ] **Step 6: 运行 Dataset 测试并提交**

Run: `/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q tests/test_joint_dataset_io.py tests/test_temporal_dataset.py`

Expected: all tests pass。

Commit: `feat: add causal history labels to transformer dataset`

## Task 4: Temporal Adapter 与 masked outcome loss

**Files:**
- Create: `constellation/new_transformers/temporal_adapter.py`
- Create: `tests/test_temporal_adapter.py`

- [ ] **Step 1: 写零初始化精确 no-op 失败测试**

```python
def test_temporal_adapter_starts_as_exact_noop() -> None:
    adapter = TemporalAdapter(satellite_width=8, task_width=8, hidden_width=16)
    result = adapter(
        satellite_features=torch.randn(1, 2, 8),
        task_features=torch.randn(1, 3, 8),
        null_logits=torch.randn(1, 2),
        task_logits=torch.randn(1, 2, 3),
        satellite_mask=torch.ones(1, 2, dtype=torch.bool),
        task_mask=torch.ones(1, 3, dtype=torch.bool),
        history=TemporalHistoryTensors(
            previous_task_indices=torch.tensor([[1, -1]]),
            previous_task_available=torch.tensor([[True, False]]),
            previous_was_idle=torch.tensor([[False, True]]),
            run_lengths=torch.tensor([[3., 2.]]),
            switch_count_30=torch.tensor([[1., 0.]]),
            switch_count_60=torch.tensor([[2., 0.]]),
        ),
    )
    torch.testing.assert_close(result.null_delta, torch.zeros_like(result.null_delta), rtol=0, atol=0)
    torch.testing.assert_close(result.task_delta, torch.zeros_like(result.task_delta), rtol=0, atol=0)
```

- [ ] **Step 2: 运行并确认模块缺失**

Run: `/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q tests/test_temporal_adapter.py`

Expected: collection fails with missing module。

- [ ] **Step 3: 实现边表示、任务残差和空动作残差**

```python
@dataclasses.dataclass(frozen=True)
class TemporalAdapterOutput:
    null_delta: torch.Tensor
    task_delta: torch.Tensor
    outcome_logits: dict[str, torch.Tensor]

class TemporalAdapter(nn.Module):
    def forward(
        self,
        satellite_features: torch.Tensor,
        task_features: torch.Tensor,
        null_logits: torch.Tensor,
        task_logits: torch.Tensor,
        satellite_mask: torch.Tensor,
        task_mask: torch.Tensor,
        history: TemporalHistoryTensors,
    ) -> TemporalAdapterOutput:
        history.validate(
            batch_size=task_logits.shape[0],
            num_satellites=task_logits.shape[1],
            num_tasks=task_logits.shape[2],
        )
        edge_features = self._build_edge_features(
            satellite_features,
            task_features,
            task_logits,
            history,
        )
        edge_hidden = self.edge_mlp(edge_features)
        task_delta = self.task_residual(edge_hidden).squeeze(-1)
        task_delta = task_delta.masked_fill(
            ~(satellite_mask.unsqueeze(-1) & task_mask.unsqueeze(1)), 0.
        )
        null_hidden = self.null_mlp(
            self._build_null_features(
                satellite_features, null_logits, history
            )
        )
        null_delta = self.null_residual(null_hidden).squeeze(-1)
        null_delta = null_delta.masked_fill(~satellite_mask, 0.)
        outcome_logits = {
            name: head(edge_hidden).squeeze(-1)
            for name, head in self.outcome_heads.items()
        }
        return TemporalAdapterOutput(
            null_delta=null_delta,
            task_delta=task_delta,
            outcome_logits=outcome_logits,
        )
```

输入包含 `previous_task_match`、`log1p(min(run_length,300))/log1p(300)`、归一化 switch counts、idle/available。残差最后线性层权重和 bias 显式置零，task padding delta 必须为零。

- [ ] **Step 4: 写 mask loss 失败测试并实现 `temporal_outcome_loss()`**

```python
losses = temporal_outcome_loss(
    logits={'visible_5': torch.tensor([10., -10.])},
    targets={'visible_5': torch.tensor([1., 1.])},
    observed={'visible_5': torch.tensor([True, False])},
)
```

断言第二个 censored 样本不改变 loss；全 mask false 时返回与 logits 同 device/dtype 的可反传零标量。事件时间只在 `time > 0` 时使用 `SmoothL1`，除以对应 horizon 归一化。

- [ ] **Step 5: 写 shape/非法索引/梯度测试并提交**

Run: `/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q tests/test_temporal_adapter.py`

Expected: all tests pass。

Commit: `feat: add zero initialized temporal adapter`

## Task 5: JointModel 接入、冻结和 loss

**Files:**
- Modify: `constellation/new_transformers/model.py`
- Create: `tests/test_temporal_model.py`

- [ ] **Step 1: 写关闭功能与开启零初始化的 logits 等价失败测试**

```python
baseline = Model(**tiny_kwargs).eval()
temporal = Model(**tiny_kwargs, use_temporal_adapter=True).eval()
incompatible = temporal.load_state_dict(baseline.state_dict(), strict=False)
torch.testing.assert_close(
    temporal.predict(*inputs, temporal_history=history),
    baseline.predict(*inputs),
    rtol=0,
    atol=0,
)
assert all('_temporal_adapter.' in key for key in incompatible.missing_keys)
```

- [ ] **Step 2: 运行并确认构造参数/API 缺失**

Run: `/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q tests/test_temporal_model.py`

Expected: failure for unsupported `temporal_history` or missing adapter。

- [ ] **Step 3: 在 Transformer 内部接入 adapter**

新增参数：

```python
use_temporal_adapter: bool = False
temporal_adapter_hidden_width: int = 64
temporal_residual_scale: float = 1.0
freeze_temporal_backbone: bool = False
```

Decoder 返回的 `satellite_features/hidden_states` 直接供 adapter 使用；修正顺序为 feasibility -> assignment（若启用）-> temporal bounded residual。只有 `use_temporal_adapter=True` 才要求历史张量。

- [ ] **Step 4: 实现 P0-B 冻结范围**

`freeze_temporal_backbone=True` 时先 `self.requires_grad_(False)`，再只打开 `_temporal_adapter`。若未开启 adapter 立即 `ValueError`。测试 trainable names 只包含 adapter 参数。

- [ ] **Step 5: JointModel 聚合 CE 与 outcome loss**

新增权重：`temporal_visible_loss_weight`、`temporal_progress_loss_weight`、`temporal_completion_loss_weight`、`temporal_event_time_loss_weight`。`forward()` 将 Batch 历史传给 `predict()`，将 outcome losses 写入 memo/TensorBoard；保留原 `la/ls/lt/assignment` 口径。

- [ ] **Step 6: 真实小 batch backward/step 验证仅新参数更新**

使用真实 `JointDataset` 一个样本、tiny temporal model、SGD 一步；断言 loss/grad 有限、adapter 至少一个参数变化、主干逐位不变。

- [ ] **Step 7: 运行模型相关回归并提交**

Run: `/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q tests/test_temporal_model.py tests/test_timemodel_only_training.py tests/test_bipartite_assignment.py`

Expected: all tests pass。

Commit: `feat: integrate temporal adapter into joint model`

## Task 6: 正式在线观察与 Policy 字段传递

**Files:**
- Modify: `constellation/rl/environment.py`
- Modify: `constellation/rl/controller_environment.py`
- Modify: `constellation/rl/policy.py`
- Create: `tests/test_temporal_policy.py`

- [ ] **Step 1: 写 padding、FeatureExtractor 和 reset 失败测试**

测试 observation 包含固定大小的：

```python
previous_task_index       # MAX_NUM_SATELLITES, int, -1 表示无匹配
previous_task_available   # MAX_NUM_SATELLITES, binary
previous_was_idle         # MAX_NUM_SATELLITES, binary
run_length                # MAX_NUM_SATELLITES, float
switch_count_30           # MAX_NUM_SATELLITES, float
switch_count_60           # MAX_NUM_SATELLITES, float
```

`FeatureExtractor` 裁剪为实际卫星数，保留 `-1`，并构造与 Dataset 相同字段名的 policy `Batch`。

- [ ] **Step 2: 运行并确认 observation schema 缺字段**

Run: `/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q tests/test_temporal_policy.py`

Expected: missing history key failure。

- [ ] **Step 3: 扩展 `Observation/null_observation/Padding/observation_space`**

离散索引空间使用 `spaces.Box(low=-1, high=MAX_NUM_TASKS-1, dtype=np.int32)`；连续计数使用 `np.float32`。Padding 对历史卫星字段与 constellation 字段使用同一卫星长度。

- [ ] **Step 4: 在两个环境中维护同一历史状态机**

reset 后创建 `CausalAssignmentHistory(num_satellites)`；`_get_observation()` 先取当前 ongoing/valid task 的全局索引 `flags.nonzero()` 再 snapshot；`_take_actions()` 在候选列表变化前把相对索引映射为全局 ID，并在真实一步执行后调用 `record()`。`_skip_idle()` 每秒经过同一 `_take_actions()`，因此 idle 秒不会丢失。

- [ ] **Step 5: Policy 仅在 adapter 开启时传历史**

`ActorCritic.forward_actor()` 使用命名参数调用 `predict()`；旧 Actor 配置仍可接收扩展 observation，但 `use_temporal_adapter=False` 时不会改变 logits。

- [ ] **Step 6: 运行 policy/history 测试并提交**

Run: `/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q tests/test_temporal_history.py tests/test_temporal_policy.py`

Expected: all tests pass。

Commit: `feat: pass causal history through online policy`

## Task 7: P0-B 配置与评估开关

**Files:**
- Create: `constellation/new_transformers/config_temporal_adapter_p0.py`
- Modify: `constellation/rl/eval_all.py`
- Modify: `tests/test_temporal_model.py`

- [ ] **Step 1: 写配置与 CLI metadata 失败测试**

断言训练配置：10k pilot、Stage3 annotation、`use_temporal_adapter=True`、`freeze_temporal_backbone=True`、旧 loss 保留 CE 锚点、checkpoint 间隔 1k；评估 metadata 记录 adapter hidden width 和 residual scale。

- [ ] **Step 2: 运行并确认配置文件/参数缺失**

Run: `/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q tests/test_temporal_model.py`

Expected: missing config or CLI option failure。

- [ ] **Step 3: 新建独立训练配置**

从 `config_paper_stage3_200k.py` deepcopy trainer/validator；启用 temporal adapter、冻结主干，optimizer `lr=5e-4`，不覆盖 Stage3/P3.x 配置。

- [ ] **Step 4: 扩展正式评估入口**

新增 `--use-temporal-adapter`、`--temporal-adapter-hidden-width`、`--temporal-residual-scale`；通过 `build_policy_kwargs()` 传入 Actor，并写入 `eval_metadata.json`。默认关闭，旧命令行为不变。

- [ ] **Step 5: 运行 config/CLI 和相关全回归并提交**

Run: `/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q tests/test_temporal_model.py tests/test_temporal_policy.py tests/test_rollout_model_candidates.py`

Expected: all tests pass。

Commit: `feat: add temporal adapter training and evaluation switches`

## Task 8: 完整工程验收

**Files:**
- Verify only; no new production behavior.

- [x] **Step 1: 运行 P0 定向测试集**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q \
  tests/test_temporal_history.py \
  tests/test_temporal_outcomes.py \
  tests/test_temporal_dataset.py \
  tests/test_temporal_adapter.py \
  tests/test_temporal_model.py \
  tests/test_temporal_policy.py
```

Expected: all tests pass。

- [x] **Step 2: 运行受影响旧回归**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest -q \
  tests/test_joint_dataset_io.py \
  tests/test_multi_horizon_edge_labels.py \
  tests/test_timemodel_duration_scale.py \
  tests/test_timemodel_feasibility.py \
  tests/test_timemodel_only_training.py \
  tests/test_bipartite_assignment.py
```

Expected: all tests pass。

- [x] **Step 3: 静态检查与 diff 检查**

Run: `/home/hy/miniconda3/envs/aeos/bin/python -m py_compile constellation/new_transformers/temporal_history.py constellation/new_transformers/temporal_adapter.py constellation/new_transformers/multi_horizon_edge_labels.py constellation/new_transformers/dataset.py constellation/new_transformers/model.py constellation/rl/environment.py constellation/rl/controller_environment.py constellation/rl/policy.py constellation/rl/eval_all.py`

Run: `git diff --check`

Expected: both exit 0。

- [x] **Step 4: 运行真实 Dataset + forward/backward/optimizer smoke**

从 `train_paper_stage3_tau_e_existing.json` 读取一个真实样本，使用 tiny 模型或显存允许时 Stage3 checkpoint；断言：loss 有限、adapter grad 有限、一步后仅 adapter 参数改变。该步骤不启动正式 10k 训练。

- [x] **Step 5: 记录未执行的实验门槛**

工程实现完成不等于模型表现通过。明确记录 P0-B 训练、离线 PR-AUC/Brier/ECE、8+8 Val、64+64 Val、推理开销和 `CS_paper` 仍需后续实验，不编造结果。

2026-07-16 工程验收记录：

- `tests/` 全量回归为 174 passed；P0 定向测试和受影响旧回归均包含在内；
- 生产文件 `py_compile` 与 `git diff --check` 均退出 0；
- 真实 Stage3 annotation 样本完成 forward、backward、optimizer step，loss 和梯度有限，且仅 adapter 参数更新；
- Stage3 200k checkpoint 按正式 `strict=False` 加载协议验证：关闭 adapter 与启用零初始化 adapter 的 logits 逐位一致；
- CUDA 热路径已移除 history value validation 引起的 DtoH 同步。无外部负载时的初步交错基准为约 5.46% 开销，尚未达到预注册的 5% 行为门槛；需在 GPU 无外部作业时复测，必要时继续优化；
- 尚未执行正式 10k P0-B 训练、离线 PR-AUC/Brier/ECE、历史打乱消融、8+8/64+64 Val、行为指标和 `CS_paper` 对比，因此当前只确认工程管线成立，不宣称模型表现已经提升。

- [x] **Step 6: 最终提交**

Commit: `test: verify causal history temporal adapter pipeline`
