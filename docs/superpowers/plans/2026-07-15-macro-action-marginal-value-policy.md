# Macro-Action Marginal Value Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把每秒 `task_id` greedy 决策改造成可训练、可评估的 `(task_id, commitment)` 宏动作策略，并用 1/5/15/30 秒受控反事实分支监督多时间尺度边际价值头。

**Architecture:** 保留现有 Transformer Encoder/Decoder 作为状态和候选表示，新增独立的边际价值头与持续时间选择头；离线使用 Basilisk 生成同状态、不同任务与持续时间的短分支，在线只维护宏动作承诺状态并消费神经网络输出。实现先完成纯逻辑和模型闭环，再接入真实分支生成、训练和正式评估。

**Tech Stack:** Python 3.11、PyTorch、TODD runner、Basilisk、pytest；所有命令使用 `/home/hy/miniconda3/envs/aeos/bin/python`。

---

## 文件结构

- Create `constellation/new_transformers/macro_action.py`：宏动作状态、承诺倒计时和安全中断。
- Create `constellation/new_transformers/macro_action_branch.py`：离线连续强制任务的算法包装器。
- Create `constellation/new_transformers/marginal_value_head.py`：多时间尺度结果头、持续时间头和损失。
- Create `constellation/new_transformers/macro_action_dataset.py`：反事实 JSON 到训练 batch 的严格映射。
- Modify `constellation/new_transformers/model.py`：通过显式配置启用新头并暴露结构化预测。
- Create `constellation/new_transformers/macro_policy.py`：联合选择任务与 commitment 的纯张量策略。
- Modify `tools/rollout_model_trajectories.py`：增加新 checkpoint 的宏动作 greedy 算法，旧路径不变。
- Create `tools/generate_macro_action_branches.py`：生成 1/5/15/30 秒短反事实分支。
- Create `tools/train_marginal_value_head.py`：冻结主干的 B1 训练入口。
- Create `tools/evaluate_macro_action_policy.py`：正式 Basilisk 评估入口和行为指标。
- Create `tests/test_macro_action.py`、`tests/test_macro_action_branch.py`、`tests/test_marginal_value_head.py`、`tests/test_macro_action_dataset.py`、`tests/test_macro_policy.py`。

### Task 1: 宏动作承诺状态机

**Files:**
- Create: `constellation/new_transformers/macro_action.py`
- Test: `tests/test_macro_action.py`

- [ ] **Step 1: 写失败测试**

```python
from constellation.new_transformers.macro_action import (
    CommitmentDecision,
    CommitmentState,
)


def test_commitment_counts_down_and_replans_at_expiry():
    state = CommitmentState.empty(num_satellites=2)
    state.start([
        CommitmentDecision(task_id=3, commitment_seconds=5),
        CommitmentDecision(task_id=-1, commitment_seconds=1),
    ], start_time=10)
    assert state.assignment() == [3, -1]
    assert state.advance(time=11, ongoing_task_ids={3}) == [False, True]
    assert state.remaining_seconds.tolist() == [4, 0]


def test_closed_task_interrupts_before_expiry():
    state = CommitmentState.empty(num_satellites=1)
    state.start([
        CommitmentDecision(task_id=3, commitment_seconds=30),
    ], start_time=10)
    assert state.advance(time=11, ongoing_task_ids=set()) == [True]
    assert state.interruption_reasons == ['task_unavailable']
```

- [ ] **Step 2: 运行测试并确认失败**

Run:
`/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_macro_action.py -q`

Expected: FAIL，模块 `macro_action` 不存在。

- [ ] **Step 3: 实现状态机**

```python
@dataclasses.dataclass(frozen=True)
class CommitmentDecision:
    task_id: int
    commitment_seconds: int


@dataclasses.dataclass
class CommitmentState:
    task_ids: torch.Tensor
    remaining_seconds: torch.Tensor
    start_times: torch.Tensor
    last_update_times: torch.Tensor
    interruption_reasons: list[str | None]

    @classmethod
    def empty(cls, num_satellites: int) -> 'CommitmentState':
        if num_satellites <= 0:
            raise ValueError('num_satellites must be positive')
        return cls(
            task_ids=torch.full((num_satellites,), -1, dtype=torch.long),
            remaining_seconds=torch.zeros(num_satellites, dtype=torch.long),
            start_times=torch.full((num_satellites,), -1, dtype=torch.long),
            last_update_times=torch.full(
                (num_satellites,), -1, dtype=torch.long
            ),
            interruption_reasons=[None] * num_satellites,
        )

    def start(
        self,
        decisions: Sequence[CommitmentDecision],
        *,
        start_time: int,
    ) -> None:
        if len(decisions) != self.task_ids.numel():
            raise ValueError('one decision is required per satellite')
        for index, decision in enumerate(decisions):
            duration = 1 if decision.task_id < 0 else decision.commitment_seconds
            if duration not in {1, 5, 15, 30}:
                raise ValueError('unsupported commitment')
            self.task_ids[index] = decision.task_id
            self.remaining_seconds[index] = duration
            self.start_times[index] = start_time
            self.last_update_times[index] = start_time
            self.interruption_reasons[index] = None

    def advance(
        self,
        *,
        time: int,
        ongoing_task_ids: set[int],
    ) -> list[bool]:
        replan = []
        for index, task_id in enumerate(self.task_ids.tolist()):
            elapsed = max(time - int(self.last_update_times[index]), 0)
            self.remaining_seconds[index] = max(
                int(self.remaining_seconds[index]) - elapsed, 0
            )
            self.last_update_times[index] = time
            unavailable = task_id >= 0 and task_id not in ongoing_task_ids
            expired = int(self.remaining_seconds[index]) == 0
            if unavailable:
                self.remaining_seconds[index] = 0
                self.interruption_reasons[index] = 'task_unavailable'
            elif expired:
                self.interruption_reasons[index] = 'expired'
            replan.append(unavailable or expired)
        return replan

    def assignment(self) -> list[int]:
        return self.task_ids.tolist()
```

约束：commitment 只能是 1/5/15/30；空动作强制为 1 秒；任务不再 ongoing 时立即释放；不得访问 Basilisk。

- [ ] **Step 4: 运行测试并确认通过**

Run:
`/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_macro_action.py -q`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add constellation/new_transformers/macro_action.py tests/test_macro_action.py
git commit -m "feat: add macro-action commitment state"
```

### Task 2: 连续强制宏动作分支

**Files:**
- Create: `constellation/new_transformers/macro_action_branch.py`
- Test: `tests/test_macro_action_branch.py`

- [ ] **Step 1: 写失败测试**

```python
def test_forced_window_overrides_exactly_five_steps():
    wrapper = ForcedMacroActionAlgorithm(
        base_algorithm=FakeAlgorithm([7, 8, 9, 10, 11, 12]),
        decision_time=20,
        satellite_index=0,
        forced_task_id=3,
        commitment_seconds=5,
    )
    assignments = [wrapper.resolve_assignment(time, [value])
                   for time, value in zip(range(20, 26), [7, 8, 9, 10, 11, 12])]
    assert assignments == [[3], [3], [3], [3], [3], [12]]


def test_forced_window_stops_when_task_disappears():
    wrapper = ForcedMacroActionAlgorithm(
        base_algorithm=FakeAlgorithm([7, 8, 9]),
        decision_time=20,
        satellite_index=0,
        forced_task_id=3,
        commitment_seconds=5,
    )
    assignment = wrapper.resolve_assignment(
        22, [8], ongoing_task_ids=set()
    )
    assert assignment == [8]
    assert wrapper.invalidated_at == 22
```

- [ ] **Step 2: 运行测试并确认失败**

Run:
`/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_macro_action_branch.py -q`

Expected: FAIL，模块不存在。

- [ ] **Step 3: 实现连续覆盖包装器**

```python
class ForcedMacroActionAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        *,
        base_algorithm: BaseAlgorithm,
        decision_time: int,
        satellite_index: int,
        forced_task_id: int,
        commitment_seconds: int,
    ) -> None:
        if commitment_seconds not in {1, 5, 15, 30}:
            raise ValueError('unsupported commitment')
        self.base_algorithm = base_algorithm
        self.decision_time = decision_time
        self.satellite_index = satellite_index
        self.forced_task_id = forced_task_id
        self.commitment_seconds = commitment_seconds
        self.invalidated_at = None

    def is_forced_time(self, time: int) -> bool:
        return self.decision_time <= time < (
            self.decision_time + self.commitment_seconds
        )

    def step(self, taskset, constellation, earth_rotation):
        actions, assignment = self.base_algorithm.step(
            taskset, constellation, earth_rotation
        )
        ongoing = set(taskset.ids.tolist())
        if self.is_forced_time(self._timer.time):
            if self.forced_task_id >= 0 and self.forced_task_id not in ongoing:
                self.invalidated_at = self._timer.time
            else:
                assignment[self.satellite_index] = self.forced_task_id
                actions[self.satellite_index] = self._forced_action(
                    taskset, constellation, self.forced_task_id
                )
        return actions, assignment
```

同时保存 `decision_state_signature`、实际覆盖秒数、`invalidated_at` 和中断原因。

- [ ] **Step 4: 运行定向测试**

Run:
`/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_macro_action_branch.py -q`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add constellation/new_transformers/macro_action_branch.py tests/test_macro_action_branch.py
git commit -m "feat: force multi-second counterfactual actions"
```

### Task 3: 多时间尺度边际价值头

**Files:**
- Create: `constellation/new_transformers/marginal_value_head.py`
- Test: `tests/test_marginal_value_head.py`

- [ ] **Step 1: 写失败测试**

```python
def test_head_emits_interpretable_horizon_outputs():
    head = MarginalValueHead(width=16, hidden_width=8, horizons=(1, 5, 15, 30))
    output = head(
        satellite_features=torch.randn(2, 3, 16),
        task_features=torch.randn(2, 4, 16),
        task_logits=torch.randn(2, 3, 4),
    )
    assert output.visible_logits.shape == (2, 3, 4, 4)
    assert output.progress_gain.shape == (2, 3, 4, 4)
    assert output.completion_logits.shape == (2, 3, 4, 4)
    assert output.redundancy_logits.shape == (2, 3, 4, 4)
    assert output.duration_values.shape == (2, 3, 4, 4)


def test_censored_targets_do_not_change_loss():
    first = marginal_value_loss(prediction, targets, observed_mask)
    modified = dataclasses.replace(targets, visible=1 - targets.visible)
    second = marginal_value_loss(prediction, modified, torch.zeros_like(observed_mask))
    assert second.visible.item() == 0.0
```

- [ ] **Step 2: 运行测试并确认失败**

Run:
`/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_marginal_value_head.py -q`

Expected: FAIL。

- [ ] **Step 3: 实现模型和 mask loss**

```python
class MarginalValueOutput(NamedTuple):
    visible_logits: torch.Tensor
    progress_gain: torch.Tensor
    completion_logits: torch.Tensor
    redundancy_logits: torch.Tensor
    duration_values: torch.Tensor


class MarginalValueHead(nn.Module):
    def __init__(self, *, width: int, hidden_width: int,
                 horizons: Sequence[int]) -> None:
        super().__init__()
        self.horizons = tuple(horizons)
        output_width = len(self.horizons)
        self.edge_encoder = nn.Sequential(
            nn.Linear(width * 3 + 1, hidden_width),
            nn.LayerNorm(hidden_width),
            nn.GELU(),
        )
        self.visible = nn.Linear(hidden_width, output_width)
        self.progress = nn.Linear(hidden_width, output_width)
        self.completion = nn.Linear(hidden_width, output_width)
        self.redundancy = nn.Linear(hidden_width, output_width)
        self.value = nn.Linear(hidden_width, output_width)

    def forward(self, satellite_features, task_features,
                task_logits) -> MarginalValueOutput:
        satellites = satellite_features.unsqueeze(2).expand(
            -1, -1, task_features.shape[1], -1
        )
        tasks = task_features.unsqueeze(1).expand(
            -1, satellite_features.shape[1], -1, -1
        )
        edge = self.edge_encoder(torch.cat(
            (satellites, tasks, satellites * tasks, task_logits.unsqueeze(-1)),
            dim=-1,
        ))
        return MarginalValueOutput(
            self.visible(edge), F.softplus(self.progress(edge)),
            self.completion(edge), self.redundancy(edge), self.value(edge)
        )


def marginal_value_loss(
    prediction: MarginalValueOutput,
    targets: MarginalValueTargets,
    masks: MarginalValueMasks,
) -> MarginalValueLoss:
    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return values[mask].mean() if mask.any() else values.new_zeros(())
    return MarginalValueLoss(
        visible=masked_mean(
            F.binary_cross_entropy_with_logits(
                prediction.visible_logits, targets.visible.float(),
                reduction='none'
            ), masks.visible
        ),
        progress=masked_mean(
            F.huber_loss(prediction.progress_gain, targets.progress_gain,
                         reduction='none'), masks.progress
        ),
        completion=masked_mean(
            F.binary_cross_entropy_with_logits(
                prediction.completion_logits, targets.completed.float(),
                reduction='none'
            ), masks.completion
        ),
        redundancy=masked_mean(
            F.binary_cross_entropy_with_logits(
                prediction.redundancy_logits, targets.redundant.float(),
                reduction='none'
            ), masks.redundancy
        ),
        ranking=pairwise_logistic_loss(
            prediction.duration_values, targets.better_horizon,
            targets.worse_horizon, masks.ranking
        ),
    )
```

边表示使用 `[satellite, task, satellite*task, actor_logit]`；BCE 监督可见、完成和冗余，Huber 监督进度，pairwise logistic 监督 `duration_values`。

- [ ] **Step 4: 运行 forward/backward 测试**

Run:
`/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_marginal_value_head.py -q`

Expected: PASS，所有可训练参数获得有限梯度。

- [ ] **Step 5: 提交**

```bash
git add constellation/new_transformers/marginal_value_head.py tests/test_marginal_value_head.py
git commit -m "feat: add multi-horizon marginal value head"
```

### Task 4: 反事实训练数据接口

**Files:**
- Create: `constellation/new_transformers/macro_action_dataset.py`
- Test: `tests/test_macro_action_dataset.py`

- [ ] **Step 1: 写失败测试**

```python
def test_branch_record_maps_horizons_and_censor_masks():
    sample = sample_from_branch_record(FIXTURE)
    assert sample.horizons.tolist() == [1, 5, 15, 30]
    assert sample.visible_observed.tolist() == [True, True, False, False]
    assert sample.progress_gain[1].item() == pytest.approx(2.0)


def test_mismatched_state_signatures_are_rejected():
    broken = copy.deepcopy(FIXTURE)
    broken['branches']['hold_5']['decision_state_signature'] = 'other'
    with pytest.raises(ValueError, match='same decision state'):
        samples_from_macro_summary(broken)
```

- [ ] **Step 2: 运行测试并确认失败**

Run:
`/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_macro_action_dataset.py -q`

Expected: FAIL。

- [ ] **Step 3: 实现严格数据映射和 padding collate**

```python
class MacroActionSample(NamedTuple):
    satellite_features: torch.Tensor
    task_features: torch.Tensor
    actor_logits: torch.Tensor
    satellite_index: int
    task_index: int
    horizons: torch.Tensor
    visible: torch.Tensor
    visible_observed: torch.Tensor
    progress_gain: torch.Tensor
    progress_observed: torch.Tensor
    completed: torch.Tensor
    completion_observed: torch.Tensor
    redundant: torch.Tensor
    redundancy_observed: torch.Tensor
    pairwise_better_horizon: int
    pairwise_worse_horizon: int


def samples_from_macro_summary(
    payload: Mapping[str, Any],
) -> list[MacroActionSample]:
    signatures = {
        branch['decision_state_signature']
        for record in payload['records']
        for branch in record['branches'].values()
    }
    if len(signatures) != 1:
        raise ValueError('branches must share the same decision state')
    return [
        sample_from_branch(record, branch_name, branch, payload['horizons'])
        for record in payload['records']
        for branch_name, branch in record['branches'].items()
    ]


def collate_macro_action_samples(
    samples: Sequence[MacroActionSample],
) -> MacroActionBatch:
    if not samples:
        raise ValueError('at least one sample is required')
    return pad_and_stack_macro_samples(samples)
```

缺失窗口只能产生 `observed=False`，不得填成零标签；任务 ID 必须通过记录的 `ongoing_task_ids` 显式映射。

- [ ] **Step 4: 运行测试**

Run:
`/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_macro_action_dataset.py -q`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add constellation/new_transformers/macro_action_dataset.py tests/test_macro_action_dataset.py
git commit -m "feat: load macro-action counterfactual labels"
```

### Task 5: 模型结构化预测与旧 checkpoint 兼容

**Files:**
- Modify: `constellation/new_transformers/model.py`
- Test: `tests/test_marginal_value_head.py`

- [ ] **Step 1: 写失败兼容性测试**

```python
def test_disabled_head_preserves_legacy_logits():
    legacy = Model(use_marginal_value_head=False).eval()
    candidate = Model(use_marginal_value_head=False).eval()
    candidate.load_state_dict(legacy.state_dict())
    assert torch.equal(legacy.predict(*inputs), candidate.predict(*inputs))


def test_enabled_head_requires_explicit_structured_prediction():
    model = Model(use_marginal_value_head=True)
    output = model.predict_with_marginal_values(*inputs)
    assert output.task_logits.shape[:2] == output.duration_values.shape[:2]
```

- [ ] **Step 2: 运行测试并确认失败**

Run:
`/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_marginal_value_head.py -q`

Expected: FAIL，`predict_with_marginal_values` 不存在。

- [ ] **Step 3: 修改 Transformer/Model 输出接口**

新增 `TransformerFeatures(task_features, satellite_features, null_logits, task_logits)`；原 `predict()` 保持返回相同 logits；新 `predict_with_marginal_values()` 只有配置启用且 checkpoint 含新头权重时可调用。加载旧 checkpoint 时若开关为真但缺少头权重，抛出明确错误。

- [ ] **Step 4: 验证旧路径与真实训练步**

Run:
`/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_marginal_value_head.py tests/test_rollout_model_candidates.py -q`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add constellation/new_transformers/model.py tests/test_marginal_value_head.py
git commit -m "feat: expose marginal-value model predictions"
```

### Task 6: 联合任务与持续时间选择

**Files:**
- Create: `constellation/new_transformers/macro_policy.py`
- Test: `tests/test_macro_policy.py`

- [ ] **Step 1: 写失败测试**

```python
def test_policy_selects_best_task_and_duration():
    decision = select_macro_actions(
        task_logits=torch.tensor([[[0.0, 2.0, 1.0]]]),
        duration_values=torch.tensor([[[[0., 0., 0., 0.],
                                        [0., 1., 4., 2.]]]]),
        ongoing_task_ids=torch.tensor([7, 8]),
        previous_task_ids=torch.tensor([-1]),
        active_commitments=torch.tensor([False]),
    )
    assert decision[0].task_id == 8
    assert decision[0].commitment_seconds == 15


def test_active_commitment_is_not_replanned_on_logit_noise():
    decision = select_macro_actions(
        task_logits=torch.tensor([[[0.0, -5.0]]]),
        duration_values=torch.zeros(1, 1, 1, 4),
        ongoing_task_ids=torch.tensor([7]),
        active_commitments=torch.tensor([True]),
        previous_task_ids=torch.tensor([7]),
    )
    assert decision[0].task_id == 7
```

- [ ] **Step 2: 运行测试并确认失败**

Run:
`/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_macro_policy.py -q`

Expected: FAIL。

- [ ] **Step 3: 实现纯张量选择器**

```python
def select_macro_actions(
    *,
    task_logits: torch.Tensor,
    duration_values: torch.Tensor,
    ongoing_task_ids: torch.Tensor,
    previous_task_ids: torch.Tensor,
    active_commitments: torch.Tensor,
    horizons: Sequence[int] = (1, 5, 15, 30),
    redundancy_logits: torch.Tensor | None = None,
    redundancy_penalty: float = 0.0,
) -> list[CommitmentDecision]:
    task_scores = task_logits[:, :, 1:].clone()
    if redundancy_logits is not None and redundancy_penalty:
        task_scores -= (
            redundancy_penalty * redundancy_logits.sigmoid()[:, :, :, 0]
        )
    selected_tasks = task_scores.argmax(-1)
    decisions = []
    for satellite in range(task_scores.shape[1]):
        if bool(active_commitments[satellite]):
            decisions.append(CommitmentDecision(
                int(previous_task_ids[satellite]), 1
            ))
            continue
        task_index = int(selected_tasks[0, satellite])
        horizon_index = int(
            duration_values[0, satellite, task_index].argmax()
        )
        decisions.append(CommitmentDecision(
            int(ongoing_task_ids[task_index]), int(horizons[horizon_index])
        ))
    return decisions
```

先选任务，再只读取所选边的持续时间值；null 固定 1 秒；有效承诺形成硬约束；重复惩罚仅在显式配置时启用。

- [ ] **Step 4: 运行测试**

Run:
`/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_macro_policy.py -q`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add constellation/new_transformers/macro_policy.py tests/test_macro_policy.py
git commit -m "feat: select joint task-duration macro actions"
```

### Task 7: 生成与训练工具

**Files:**
- Create: `tools/generate_macro_action_branches.py`
- Create: `tools/train_marginal_value_head.py`
- Test: `tests/test_generate_macro_action_branches_tool.py`
- Test: `tests/test_train_marginal_value_head_tool.py`

- [ ] **Step 1: 写失败 CLI 与分支规格测试**

```python
def test_branch_specs_cover_all_commitments():
    specs = build_macro_branch_specs(actor_task_id=7, previous_task_id=3,
                                     top_k_task_ids=[8])
    assert {(item.task_id, item.commitment_seconds) for item in specs} >= {
        (7, 1), (7, 5), (7, 15), (7, 30),
        (3, 5), (3, 15), (8, 5), (8, 15),
    }
```

- [ ] **Step 2: 运行测试并确认失败**

Run:
`/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_generate_macro_action_branches_tool.py tests/test_train_marginal_value_head_tool.py -q`

Expected: FAIL。

- [ ] **Step 3: 实现生成工具**

CLI 参数必须包含 checkpoint、reference trajectory、split、scene id、decision count、horizons、top-k、device、output root 和 overwrite。每条记录保存同状态签名、模型输入、分支原始指标、observed mask、实际持续秒数和中断原因。

- [ ] **Step 4: 实现 B1 训练工具**

训练入口按 scene id 划分 train/val；冻结 Transformer 主干；报告各 horizon BCE/MAE、pairwise accuracy、常数 baseline 和 Actor-logit baseline；保存 checkpoint、config 和 summary JSON。

- [ ] **Step 5: 运行 CLI/测试/py_compile**

Run:
`/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_generate_macro_action_branches_tool.py tests/test_train_marginal_value_head_tool.py -q`

Run:
`/home/hy/miniconda3/envs/aeos/bin/python -m py_compile tools/generate_macro_action_branches.py tools/train_marginal_value_head.py`

Expected: PASS。

- [ ] **Step 6: 提交**

```bash
git add tools/generate_macro_action_branches.py tools/train_marginal_value_head.py tests/test_generate_macro_action_branches_tool.py tests/test_train_marginal_value_head_tool.py
git commit -m "feat: generate and train macro-action labels"
```

### Task 8: 在线宏动作算法与正式评估

**Files:**
- Modify: `tools/rollout_model_trajectories.py`
- Create: `tools/evaluate_macro_action_policy.py`
- Test: `tests/test_rollout_model_candidates.py`

- [ ] **Step 1: 写失败集成测试**

```python
def test_macro_algorithm_reuses_assignment_until_commitment_expires():
    model = FakeMacroModel(
        assignments=[[7], [8]], commitments=[[5], [1]]
    )
    algorithm = MacroGreedyModelAlgorithm(model=model)
    first = algorithm.step(taskset, constellation, earth_rotation)[1]
    second = algorithm.step(taskset, constellation, earth_rotation)[1]
    assert first == second
    assert algorithm.model_call_count == 1
```

- [ ] **Step 2: 运行测试并确认失败**

Run:
`/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_rollout_model_candidates.py -q`

Expected: FAIL，宏动作算法不存在。

- [ ] **Step 3: 实现 `MacroGreedyModelAlgorithm`**

复用 `GreedyModelAlgorithm._build_inputs()`，只在需要重规划的卫星存在时调用结构化模型输出；每秒仍返回 Basilisk `Actions`，但未到期卫星保持承诺任务。记录每颗卫星的宏动作开始、持续、结束和中断原因。

- [ ] **Step 4: 实现评估包装器**

输出正式 evaluator 指标和行为指标：一秒动作率、平均/中位持续时间、各 commitment 使用率、重复冗余率、合理接力率、top-k 覆盖率、推理总耗时和每决策耗时。

- [ ] **Step 5: 运行回归测试**

Run:
`/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_rollout_model_candidates.py tests/test_macro_action.py tests/test_macro_policy.py -q`

Expected: PASS，旧 greedy 测试不变。

- [ ] **Step 6: 提交**

```bash
git add tools/rollout_model_trajectories.py tools/evaluate_macro_action_policy.py tests/test_rollout_model_candidates.py
git commit -m "feat: execute and evaluate macro-action policy"
```

### Task 9: 真实纵向闭环与阶段验收

**Files:**
- Create: `scripts/run_macro_action_b1_pilot_2.sh`
- Create: `work_dirs/macro_action_b1_pilot_2/`（实验输出，不提交）
- Modify: `TODO.md`（仅在独立工作树中，无重叠时）
- Modify: `改进日志.md`（仅记录实际结果，不预写结论）

- [ ] **Step 1: 在 2 个训练场景生成反事实分支**

Run:
`bash scripts/run_macro_action_b1_pilot_2.sh generate`

Expected: 每个场景至少包含 1/5/15/30 秒同状态分支；状态签名一致；实际覆盖时长和中断原因可追溯。

- [ ] **Step 2: 训练 B1 头并验证排序**

Run:
`bash scripts/run_macro_action_b1_pilot_2.sh train`

Expected: 完成真实 forward/backward/optimizer step；生成 checkpoint 与 summary；报告而不伪造验证集指标。

- [ ] **Step 3: 运行 2 场景宏动作 rollout**

Run:
`bash scripts/run_macro_action_b1_pilot_2.sh evaluate`

Expected: 新策略实际产生大于 1 秒的宏动作；在线路径无 Basilisk 预测调用；旧 Stage3 和新策略使用相同场景与 evaluator。

- [ ] **Step 4: 决定是否扩展 16 场景**

扩展门槛：四种持续时间结果可区分；结果头优于常数 baseline；宏动作推理无任务 ID 错位；一秒动作率下降且 `CR/PCR/WCR/TAT_s/PC_Wh` 无明显协议错误。满足后再生成 16 场景数据，否则先修复数据或控制逻辑。

- [ ] **Step 5: 验证与记录**

Run:
`/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_macro_action.py tests/test_macro_action_branch.py tests/test_marginal_value_head.py tests/test_macro_action_dataset.py tests/test_macro_policy.py tests/test_rollout_model_candidates.py -q`

Run: `git diff --check`

在 `改进日志.md` 记录真实命令、checkpoint、输出目录、场景数、所有指标、失败项和是否进入 16 场景阶段。

- [ ] **Step 6: 提交实验包装与日志**

```bash
git add scripts/run_macro_action_b1_pilot_2.sh TODO.md 改进日志.md
git commit -m "exp: validate macro-action B1 pilot"
```

## 完成定义

- 所有新增与相关回归测试通过；
- 旧 checkpoint 在开关关闭时行为不变；
- 至少 2 个真实场景完成反事实生成、B1 训练和宏动作评估；
- 输出包含模型精度、行为持续性和论文指标，而不是只报告 loss；
- 只有满足 Task 9 门槛才启动 16 场景和完整 Val。
