# Event Joint Transformer V2 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现可独立测试、可从 Stage3-200k 热启动的事件级联合 Transformer V2 基础模型，并完成 V2-0 离线 warm start 所需的数据、损失、配置、checkpoint 和 Slurm 入口。

**Architecture:** 保留现有 `JointModel` 不变，新建 `event_v2` 包。`Stage3FeatureBackbone` 复用旧 Transformer 的任务 Encoder 与卫星 Decoder 产生 task/satellite/edge token；`EventStateEncoder` 注入上一任务、当前任务、承诺、切换和事件时间等显式状态；`AutoregressiveJointActor` 按确定性紧迫度顺序依次产生 termination、task 和 minimum commitment；`CentralizedValueCritic` 只聚合同源轻量状态。V2-0 只使用旧轨迹上的事实标签做离线 warm start，不把专家重复动作当作软容量全局最优标签。

**Tech Stack:** Python 3.11、PyTorch、todd 配置/Runner、pytest、Slurm；所有 Python 命令使用 `/home/hy/miniconda3/envs/aeos/bin/python`。

---

## 实施边界

- 本计划只交付 V2-0 foundation，不实现 Basilisk Event Runtime、同步 PPO 或 APPO。
- 新代码位于 `constellation/new_transformers/event_v2/`，不修改旧 `JointModel` 的行为。
- Actor/Critic 输入不得含 `is_visible`、未来状态或在线 Basilisk 预测。
- 允许从 Stage3 checkpoint 复制 `_time_embedding`、`_sensor_type_embedding`、`_encoder`、`_decoder` 和 `_time_model`；旧任务 logits 只作蒸馏 teacher。
- V2 新模块包括事件状态层、termination head、自回归联合 task head、minimum commitment head、owner-rank marginal head 和 centralized Critic。
- 第一阶段 commitments 固定为 `(1, 5, 15, 30, 60)`；非 idle 的 `1s` 仅在剩余要求观测时长不超过 1 秒时合法。
- owner hard cap 为 3；owner rank 2/3 需要正的 marginal collaboration score 才能在 deterministic 模式超过最佳非重复动作。

## Task 1: 建立 V2 tensor schema、事件排序和物理 mask

**Files:**

- Create: `constellation/new_transformers/event_v2/__init__.py`
- Create: `constellation/new_transformers/event_v2/state.py`
- Test: `tests/test_event_v2_state.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_event_v2_state.py` 覆盖：

```python
import pytest
import torch

from constellation.new_transformers.event_v2.state import (
    COMMITMENT_SECONDS,
    EventStateTensors,
    build_commitment_mask,
    build_replan_order,
)


def _state() -> EventStateTensors:
    return EventStateTensors(
        previous_task_indices=torch.tensor([[1, -1, 0]]),
        current_task_indices=torch.tensor([[1, -1, 0]]),
        minimum_commitment_remaining=torch.tensor([[0., 0., 5.]]),
        run_lengths=torch.tensor([[8., 9., 3.]]),
        seconds_since_replan=torch.tensor([[2., 12., 4.]]),
        switch_count_30=torch.tensor([[0., 1., 2.]]),
        switch_count_60=torch.tensor([[1., 2., 3.]]),
        termination_reason=torch.tensor([[0, 0, 1]]),
        event_type=torch.tensor([[0, 0, 1]]),
        delta_t=torch.tensor([[5., 5., 1.]]),
        replan_mask=torch.tensor([[True, True, True]]),
        forced_interrupt_mask=torch.tensor([[False, False, True]]),
        can_terminate_mask=torch.tensor([[True, False, False]]),
        compatible_deadline_slack=torch.tensor([[20., 5., 100.]]),
        task_remaining_required_seconds=torch.tensor([[1., 4., 30.]]),
        task_owner_count=torch.tensor([[0, 1, 3]]),
        task_locked_owner_count=torch.tensor([[0, 1, 1]]),
    )


def test_replan_order_uses_interrupt_slack_wait_and_id() -> None:
    assert build_replan_order(_state())[0].tolist() == [2, 1, 0]


def test_commitment_mask_reserves_one_second_for_nearly_complete_task() -> None:
    mask = build_commitment_mask(
        remaining_required_seconds=torch.tensor([[1., 4.]]),
        task_selected=torch.tensor([[True, True]]),
    )
    assert COMMITMENT_SECONDS == (1, 5, 15, 30, 60)
    assert mask.tolist() == [[[True] * 5, [False, True, True, True, True]]]


def test_event_state_rejects_owner_count_above_three() -> None:
    state = _state()
    with pytest.raises(ValueError, match='owner'):
        state._replace(task_owner_count=torch.tensor([[0, 1, 4]])).validate()
```

- [ ] **Step 2: 运行测试并确认因模块不存在而失败**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_event_v2_state.py -q
```

Expected: `ModuleNotFoundError: constellation.new_transformers.event_v2`。

- [ ] **Step 3: 实现 schema 与纯函数**

`state.py` 固定公开常量 `COMMITMENT_SECONDS = (1, 5, 15, 30, 60)`、
`MAX_TASK_OWNERS = 3`。`EventStateTensors` 使用上面 `_state()` 展示的 18 个同名字段，
并提供 `validate() -> None`。纯函数签名为
`build_replan_order(state: EventStateTensors) -> list[torch.Tensor]` 和
`build_commitment_mask(remaining_required_seconds: torch.Tensor,
task_selected: torch.Tensor) -> torch.Tensor`。

`validate()` 必须检查 batch/satellite/task 轴、dtype、finite、非负时间、owner 范围和 `locked_owner_count <= owner_count`。排序 key 精确为：forced interrupt 降序、deadline slack 升序、wait seconds 降序、satellite id 升序。`__init__.py` 只导出稳定公开接口。

- [ ] **Step 4: 运行单测**

Run:

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_event_v2_state.py -q
```

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add constellation/new_transformers/event_v2/__init__.py constellation/new_transformers/event_v2/state.py tests/test_event_v2_state.py
git commit -m "feat: add event v2 state schema"
```

## Task 2: 实现完成质量 potential、终点校正和 time-aware GAE

**Files:**

- Create: `constellation/new_transformers/event_v2/reward.py`
- Test: `tests/test_event_v2_reward.py`

- [ ] **Step 1: 写失败测试**

测试必须用人工可核算数字证明：

```python
def test_completion_reward_telescopes_to_exact_q() -> None:
    weights = torch.tensor([0.3, 0.7])
    progress = [
        torch.tensor([0., 0.]),
        torch.tensor([5., 0.]),
        torch.tensor([10., 5.]),
    ]
    duration = torch.tensor([10., 10.])
    rewards = build_completion_event_rewards(
        progress=progress,
        required_duration=duration,
        task_weights=weights,
        completed=torch.tensor([True, False]),
    )
    torch.testing.assert_close(sum(rewards), torch.tensor(0.3))


def test_time_aware_gae_uses_physical_delta_t() -> None:
    result = time_aware_gae(
        rewards=torch.tensor([1., 2.]),
        values=torch.tensor([0.5, 0.25]),
        next_values=torch.tensor([0.25, 0.]),
        delta_t=torch.tensor([5., 10.]),
        done=torch.tensor([False, True]),
        lambda_base=0.95,
        reference_seconds=5.,
    )
    expected_last = torch.tensor(1.75)
    expected_first = torch.tensor(0.75) + 0.95 * expected_last
    torch.testing.assert_close(result.advantages, torch.stack((expected_first, expected_last)))
```

同时测试非法 duration/weights、未完成临时 progress 在 terminal correction 被收回、`done=True` 后不 bootstrap、所有输出 finite。

- [ ] **Step 2: 运行失败测试**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_event_v2_reward.py -q
```

Expected: import 失败。

- [ ] **Step 3: 实现纯 reward API**

`GAEOutput` 包含 `advantages` 和 `returns` 两个 tensor。公开函数固定为
`completion_potential(progress, required_duration, task_weights)`、
`terminal_completion_quality(completed, task_weights)`、
`build_completion_event_rewards(progress, required_duration, task_weights,
completed)` 和 `time_aware_gae(rewards, values, next_values, delta_t, done,
lambda_base=0.95, reference_seconds=5.0)`；类型分别按测试输入推导，前三个返回 tensor、
tensor、tensor list，最后一个返回 `GAEOutput`。

实现中 `gamma=1`；`lambda_e = lambda_base ** (delta_t/reference_seconds)`；terminal reward 添加 `Q_final - Phi_terminal`。不接受负权重或非正 required duration。

- [ ] **Step 4: 运行测试并提交**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_event_v2_reward.py -q
git add constellation/new_transformers/event_v2/reward.py tests/test_event_v2_reward.py constellation/new_transformers/event_v2/__init__.py
git commit -m "feat: add event completion reward and gae"
```

## Task 3: 固定联合动作与 transition schema fingerprint

**Files:**

- Create: `constellation/new_transformers/event_v2/transition.py`
- Test: `tests/test_event_v2_transition.py`

- [ ] **Step 1: 写失败测试**

覆盖 `JointEventAction`、`ActionTrace`、`EventTransition` 的 shape 校验、CPU 序列化和稳定 fingerprint。两个字段顺序或 dtype 不同的 schema fingerprint 必须不同；相同 schema 跨进程重建必须相同。

- [ ] **Step 2: 实现 schema**

定义 `TRANSITION_SCHEMA_VERSION = 1`。`JointEventAction` 依次包含 `terminate`、
`task_indices`、`commitment_indices`；`ActionTrace` 依次包含 `action_order`、
`termination_mask`、`task_masks`、`commitment_masks`、`owner_state`；冻结 dataclass
`EventTransition` 依次包含设计文档规定的 state/action/log-prob/value/reward/delta_t/
next_state/done/trace/policy_version，并提供 `validate()`。公开函数
`transition_schema_fingerprint() -> str` 返回 64 字符小写 SHA-256。

fingerprint 使用 version、字段名、嵌套顺序和语义 dtype 生成 canonical JSON 后做 SHA-256；不得依赖 Python `hash()`。

- [ ] **Step 3: 测试并提交**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_event_v2_transition.py -q
git add constellation/new_transformers/event_v2/transition.py tests/test_event_v2_transition.py constellation/new_transformers/event_v2/__init__.py
git commit -m "feat: define event v2 transition schema"
```

## Task 4: 提取可热启动的 Stage3 token backbone

**Files:**

- Create: `constellation/new_transformers/event_v2/backbone.py`
- Test: `tests/test_event_v2_backbone.py`

- [ ] **Step 1: 写失败测试**

构造现有 tiny `Model`，将其 `state_dict()` 加载到 V2 backbone，验证：

- task tokens 等于旧 `_encoder` 输出；
- satellite tokens 等于旧 `_decoder` 返回的第三项；
- edge logits 等于旧 decoder 的 task logits；
- `freeze()` 后全部 Stage3 checkpoint 参数 `requires_grad=False`，V2 新建的 edge
  projection 保持可训练；
- forward 参数列表不包含 `is_visible`；
- checkpoint 仅允许 V2 新头缺键，不允许 Stage3 backbone unexpected/missing key。

- [ ] **Step 2: 实现 backbone**

`Stage3BackboneOutput` 固定包含 `task_tokens`、`satellite_tokens`、`edge_features`、
`teacher_null_logits`、`teacher_task_logits`、`feasibility_logits`。`Stage3FeatureBackbone`
提供构造、`freeze()`、`unfreeze_last_layers(encoder_layers, decoder_layers)`、
`load_stage3_state_dict(state_dict)` 和与现有 `Transformer.forward()` 前八个参数完全一致的
`forward()`；返回 `Stage3BackboneOutput`。

`edge_features` 使用 `satellite_projection(satellite_tokens)[:, :, None, :] + task_projection(task_tokens)[:, None, :, :]`，而不是调用 Basilisk/TimeModel 生成未来特征。
两个 projection 属于 V2 新模块，不能随 Stage3 checkpoint 参数冻结。teacher logits
保持旧 Stage3 计算路径，供 V2-0 蒸馏。

- [ ] **Step 3: 运行测试并提交**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_event_v2_backbone.py -q
git add constellation/new_transformers/event_v2/backbone.py tests/test_event_v2_backbone.py constellation/new_transformers/event_v2/__init__.py
git commit -m "feat: expose stage3 tokens for event v2"
```

## Task 5: 实现事件状态编码与 centralized Critic

**Files:**

- Create: `constellation/new_transformers/event_v2/critic.py`
- Test: `tests/test_event_v2_critic.py`

- [ ] **Step 1: 写失败测试**

验证关系特征不依赖跨场景 task id embedding：交换 task token 和相应 previous/current index 后输出保持置换等价；mask 的卫星/任务不影响 value；Critic 输出 `[batch]`；全零有效 token 被拒绝；冻结 backbone 时反向传播只更新 V2 状态层和 Critic。

- [ ] **Step 2: 实现**

`EventStateEncoding` 包含 `satellite_tokens`、`task_tokens`、`edge_tokens`。
`EventStateEncoder.forward(backbone, state, satellite_mask, task_mask)` 返回该结构；
`CentralizedValueCritic.forward(encoding, satellite_mask, task_mask)` 返回 `[batch]` value。

`EventStateEncoder` 将 normalized run/commit/wait/switch/reason/event/delta_t 投影到 satellite token；将 owner/locked-owner/remaining-duration 投影到 task token；上一任务和当前任务只通过 edge 上的 boolean relation 投影。Critic 用 masked mean + masked max 聚合卫星与任务 token，再输出标量。

- [ ] **Step 3: 测试并提交**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_event_v2_critic.py -q
git add constellation/new_transformers/event_v2/critic.py tests/test_event_v2_critic.py constellation/new_transformers/event_v2/__init__.py
git commit -m "feat: add event state encoder and value critic"
```

## Task 6: 实现 termination 与自回归联合 Actor

**Files:**

- Create: `constellation/new_transformers/event_v2/actor.py`
- Test: `tests/test_event_v2_actor.py`

- [ ] **Step 1: 写失败测试**

测试覆盖：

1. 最小承诺未结束的卫星不产生主动 termination log-prob；强制物理中断也不计策略 log-prob。
2. action order 与 `build_replan_order()` 一致。
3. 第二颗卫星的 task logits 随第一颗已选 owner 状态变化。
4. owner count 3 的任务永久 mask。
5. deterministic 模式下 owner rank 2/3 的 marginal score 不为正时退回最佳非重复任务或 idle。
6. `1s` commitment mask 符合 Task 1。
7. `evaluate_actions()` 使用保存的 order/masks/owner_state 后与 `sample_actions()` 的 joint log-prob 逐值一致。
8. 任意合法 batch 的 task、commitment、termination log-prob 和 entropy 均 finite。

- [ ] **Step 2: 实现公开 Actor API**

`ActorOutput` 包含 action/log_prob/entropy/trace，`ActionEvaluation` 包含 log_prob/
entropy。`AutoregressiveJointActor` 提供 `sample_actions(encoding, state,
satellite_mask, task_mask, deterministic)` 和 `evaluate_actions(encoding, state,
satellite_mask, task_mask, action, trace)`，分别返回上述两个结构。

task categorical 的 index 0 为 idle，1..N 为 task。每个自回归 prefix 更新 `owner_count`、已选 task embedding、commitment embedding 和 owner-rank marginal context。`trace.task_masks`、`trace.commitment_masks`、`trace.owner_state` 必须保存每个 action-order 位置的行为侧状态；learner 不得重新推断这些物理 mask。

- [ ] **Step 3: 测试并提交**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_event_v2_actor.py -q
git add constellation/new_transformers/event_v2/actor.py tests/test_event_v2_actor.py constellation/new_transformers/event_v2/__init__.py
git commit -m "feat: add autoregressive event actor"
```

## Task 7: 组装 `EventJointActorCritic` 与 Stage3 加载边界

**Files:**

- Create: `constellation/new_transformers/event_v2/model.py`
- Modify: `constellation/new_transformers/event_v2/__init__.py`
- Test: `tests/test_event_v2_model.py`

- [ ] **Step 1: 写失败测试**

测试 tiny forward 返回 action/log-prob/entropy/value；`evaluate_actions()` 精确重放；`freeze_backbone=True` 时 optimizer step 后 Stage3 参数逐值不变；输入签名无 `is_visible`；保存/加载后 deterministic action/value 一致；`unfreeze_last_layers(1, 1)` 只解冻最后一层 Encoder/Decoder 和 V2 模块。

- [ ] **Step 2: 实现模型**

`EventActorCriticOutput` 包含 `actor` 与 `value`。`EventJointActorCritic` 构造参数包含
`event_width=256`、`freeze_backbone=True` 及现有 Stage3 模型参数；提供 `act()`、
`evaluate_actions()`、`load_stage3_checkpoint(path)`、
`parameter_groups(new_module_lr, backbone_lr_scale=0.1)`。`act/evaluate_actions` 的状态与
mask 参数与 Actor 相同，并额外接收 Stage3 的八个原始输入 tensor。

checkpoint loader 同时兼容裸 state dict 与 todd `model.pth`，剥离 `module.` 前缀，并给出明确 missing/unexpected backbone key 错误。V2 模型不注册为旧 `JointModel` 的替代类型；后续单独 runner 使用独立 import/type。

- [ ] **Step 3: 测试并提交**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_event_v2_model.py -q
git add constellation/new_transformers/event_v2/model.py constellation/new_transformers/event_v2/__init__.py tests/test_event_v2_model.py
git commit -m "feat: assemble event joint actor critic"
```

## Task 8: 构建 V2-0 旧轨迹事件数据与离线损失

**Files:**

- Create: `constellation/new_transformers/event_v2/offline.py`
- Create: `constellation/new_transformers/event_v2/dataset.py`
- Test: `tests/test_event_v2_offline.py`
- Test: `tests/test_event_v2_dataset.py`

- [ ] **Step 1: 写失败测试**

使用 8 秒人工轨迹验证事件压缩：连续相同任务只形成一个 segment；task 失效/切换/idle 形成新事件；termination label 只来自实际 segment 边界；minimum commitment label 取不超过实际 segment 长度的最大合法档位；剩余任务时长为 1 秒时允许 1 秒档。验证 owner marginal loss 不读取专家 owner 2/3 作为正标签。

离线 loss 测试验证：teacher task KL、termination BCE、commitment CE、value Huber 都 finite；被 mask 项不贡献损失；总 loss 权重可配置；value target 为从事件点到终局的 completion event return，而不是 3,600 秒单动作标签。

- [ ] **Step 2: 实现数据 API**

`OfflineEventTargets` 包含 termination/termination_observed/task_indices/task_observed/
commitment_indices/commitment_observed/value_returns；`OfflineEventBatch` 包含
`stage3_batch`、`event_state`、`targets`。实现
`compress_expert_actions_to_events(actions, task_valid, progress, durations) -> list[int]`
和返回 `OfflineEventBatch` 的 `EventV2OfflineDataset`。

数据仍从现有 `trajectories.N` 和 taskset 加载，不生成反事实 rollout，不把 `is_visible` 放入输出 batch。事实 `is_visible` 只可在离线构造 segment 是否发生有效观测的审计统计中使用，不能进入 model forward。

- [ ] **Step 3: 实现 loss**

`OfflineLosses` 包含 total/task_distillation/termination/commitment/value。
`event_v2_offline_loss(model, batch, task_weight=1.0, termination_weight=1.0,
commitment_weight=1.0, value_weight=1.0)` 返回该结构。

teacher task distribution 来自 frozen Stage3 logits；soft capacity marginal head 在 V2-0 不使用专家重复 owner 监督，仅通过零均值小初始化和正则保持可训练。

- [ ] **Step 4: 测试并提交**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_event_v2_dataset.py tests/test_event_v2_offline.py -q
git add constellation/new_transformers/event_v2/dataset.py constellation/new_transformers/event_v2/offline.py tests/test_event_v2_dataset.py tests/test_event_v2_offline.py constellation/new_transformers/event_v2/__init__.py
git commit -m "feat: add event v2 offline warm start data"
```

## Task 9: 添加 V2-0 trainer、配置、checkpoint 审计和 Slurm 包装

**Files:**

- Create: `tools/train_event_v2_warm_start.py`
- Create: `constellation/new_transformers/config_event_v2_warm_start.py`
- Create: `scripts/train_event_v2_warm_start_slurm.sh`
- Test: `tests/test_event_v2_warm_start.py`
- Test: `tests/test_event_v2_scripts.py`

- [ ] **Step 1: 写失败测试**

测试配置明确：Stage3 checkpoint、annotation、冻结 backbone、4 小时时限、独立输出目录、optimizer 仅含 trainable 参数。checkpoint round-trip 保存：model、optimizer、scheduler、AMP、policy version=0、schema fingerprint、normalizer、Python/NumPy/PyTorch RNG、processed physical seconds、episodes、events、stage=`V2-0`、unfreeze state。Slurm 脚本必须申请 GPU、`local-10`、不超过 4 小时，使用 `aeos` 环境并在 checkpoint 缺失时失败。

- [ ] **Step 2: 实现 standalone trainer**

CLI 固定为：

```bash
/home/hy/miniconda3/envs/aeos/bin/python tools/train_event_v2_warm_start.py \
  --config constellation/new_transformers/config_event_v2_warm_start.py \
  --stage3-checkpoint work_dirs/paper_joint_stage3_200k/checkpoints/iter_200000/model.pth \
  --output work_dirs/event_joint_transformer_v2/v2_0_warm_start
```

trainer 启动时打印 schema fingerprint、trainable/frozen parameter count 和数据指纹；每个 checkpoint 原子写入临时文件后 rename；恢复时先校验 schema 和 config fingerprint。默认不启动正式训练，只提供可执行入口。

- [ ] **Step 3: 实现配置与 Slurm 脚本**

配置固定：

```python
stage = 'V2-0'
max_hours = 4
seed = 3407
annotation_file = 'train_paper_stage3_tau_e_existing.json'
stage3_checkpoint = 'work_dirs/paper_joint_stage3_200k/checkpoints/iter_200000/model.pth'
output_dir = 'work_dirs/event_joint_transformer_v2/v2_0_warm_start'
model = dict(event_width=256, freeze_backbone=True, use_constraint_module=True, use_sdpa=True)
optimizer = dict(lr=3e-4, betas=(0.9, 0.98), weight_decay=1e-4)
loss_weights = dict(task=1.0, termination=1.0, commitment=1.0, value=1.0)
```

Slurm 使用 `#SBATCH --partition=local-10`、`#SBATCH --time=04:00:00`、`#SBATCH --gres=gpu:1`，先运行单 batch preflight，再进入 trainer；日志写入 `work_dirs/eval_logs/event_v2_warm_start_%j.log`。

- [ ] **Step 4: 测试并提交**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_event_v2_warm_start.py tests/test_event_v2_scripts.py -q
git add tools/train_event_v2_warm_start.py constellation/new_transformers/config_event_v2_warm_start.py scripts/train_event_v2_warm_start_slurm.sh tests/test_event_v2_warm_start.py tests/test_event_v2_scripts.py
git commit -m "feat: add event v2 warm start runner"
```

## Task 10: Foundation 全量验证与交付审计

**Files:**

- Modify: `TODO.md`
- Modify: `改进日志.md`
- Create: `docs/event_v2_foundation_verification.md`

- [ ] **Step 1: 运行 V2 全部单测**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest \
  tests/test_event_v2_state.py \
  tests/test_event_v2_reward.py \
  tests/test_event_v2_transition.py \
  tests/test_event_v2_backbone.py \
  tests/test_event_v2_critic.py \
  tests/test_event_v2_actor.py \
  tests/test_event_v2_model.py \
  tests/test_event_v2_dataset.py \
  tests/test_event_v2_offline.py \
  tests/test_event_v2_warm_start.py \
  tests/test_event_v2_scripts.py -q
```

Expected: 全部 PASS。

- [ ] **Step 2: 运行相关旧回归**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest \
  tests/test_event_action.py \
  tests/test_event_policy.py \
  tests/test_temporal_model.py \
  tests/test_bipartite_assignment.py -q
```

Expected: 全部 PASS，证明 V2 未改变旧模型。

- [ ] **Step 3: 运行静态与差异检查**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m compileall -q constellation/new_transformers/event_v2 tools/train_event_v2_warm_start.py
git diff --check
git status --short
```

- [ ] **Step 4: 单 batch CPU preflight**

```bash
/home/hy/miniconda3/envs/aeos/bin/python tools/train_event_v2_warm_start.py \
  --config constellation/new_transformers/config_event_v2_warm_start.py \
  --stage3-checkpoint work_dirs/paper_joint_stage3_200k/checkpoints/iter_200000/model.pth \
  --output /tmp/event_v2_warm_start_preflight \
  --max-steps 1 \
  --device cpu
```

Expected: loss/grad finite、Stage3 frozen 参数 hash 前后一致、生成可恢复 checkpoint。该命令不声明性能提升。

- [ ] **Step 5: 更新项目状态文档**

`TODO.md` 仅把 V2-0 foundation 的代码/测试/preflight 标为完成；V2-0 正式 GPU warm start、同步 PPO、APPO 和正式 Val 仍保持未完成。`改进日志.md` 记录实现边界和 preflight 事实，不写未运行的指标。

- [ ] **Step 6: 提交文档并请求代码审查**

```bash
git add TODO.md 改进日志.md docs/event_v2_foundation_verification.md
git commit -m "docs: record event v2 foundation verification"
```

随后使用 `requesting-code-review` skill，重点审查：Basilisk/未来信息泄漏、联合 log-prob 重放、mask 一致性、Stage3 冻结、owner soft capacity 和 checkpoint 恢复。

## Foundation 完成定义

只有同时满足以下条件才可进入同步 PPO 实施计划：

- 上述 V2 与旧回归测试全部通过；
- 单 batch preflight 的全部数值 finite；
- Stage3 冻结参数逐值不变；
- sampled action 与 replayed log-prob 精确一致；
- reward telescoping 精确成立；
- checkpoint 恢复后 deterministic action/value 一致；
- Actor/Critic forward 无 Basilisk、`is_visible` 或未来预测输入；
- V2-0 入口具备独立 checkpoint/log/output，不覆盖 Stage3/M2/M3。

V2-0 正式 GPU 训练结束只说明 warm start 可用，不能说明完成率提高。真正性能结论必须等待后续同步 PPO 通过固定 train smoke、Val 8+8 和 Val 64+64。
