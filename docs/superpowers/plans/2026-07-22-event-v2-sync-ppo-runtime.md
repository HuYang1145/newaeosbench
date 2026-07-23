# Event V2 Synchronous PPO Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建成可审计的事件式 Basilisk Runtime 与同步 PPO 训练闭环，并通过 V2-1 的合成环境、真实单场 3,600 秒、概率重放、冻结参数和恢复复现验收。

**Architecture:** 保留 V2-0 已完成的 Stage3 Encoder、事件联合 Actor、centralized Critic、动作 trace 和 reward/GAE；新增纯事件状态机、Basilisk 单场适配、同步 rollout buffer、PPO learner 与独立 checkpoint。每个环境只执行一条真实 Basilisk 轨迹；Basilisk 逐秒推进，模型仅在外部事件、承诺结束后可主动 termination 的复核点和每 5 秒安全复核点运行。

**Tech Stack:** Python 3.11、PyTorch、Basilisk、pytest、Slurm `local-10`、BF16 AMP；统一使用 `/home/hy/miniconda3/envs/aeos/bin/python`。

---

## 固定边界与文件结构

本计划只实现 V2-1：Stage3 全冻结、不实现 APPO、不访问 Val/Test、不改变 V2-0 动作或 reward。旧 `constellation/rl` 只可参考场景加载和单秒调用顺序，不复用其独立 `MultiCategorical` policy、逐秒 PPO step 或旧 reward。

- `event_v2/observation.py`：完整 policy 输入及 batch/device 操作。
- `event_v2/runtime_state.py`：不依赖 Basilisk 的承诺、事件、历史和 task-id 状态机。
- `event_v2/basilisk_runtime.py`：单场真实仿真、任务账本、观测和精确 reward。
- `event_v2/rollout.py`：同步收集、完整行为 trace 保存和重放。
- `event_v2/ppo.py`：time-aware GAE、clipped PPO 和停止守卫。
- `event_v2/checkpoint.py`：模型、优化器、RNG、环境游标和恢复动作复现。
- `config_event_v2_sync_ppo.py`、`train_event_v2_sync_ppo.py`、Slurm 包装：正式 V2-1 入口。

## Task 1: 完整 policy observation

**Files:**
- Create: `constellation/new_transformers/event_v2/observation.py`
- Modify: `constellation/new_transformers/event_v2/__init__.py`
- Test: `tests/test_event_v2_observation.py`

- [x] **Step 1: 写失败测试固定字段、形状和 device 搬运**

```python
def test_event_policy_observation_validates_and_moves_named_tensors():
    observation = make_observation(batch=2, satellites=3, tasks=4)
    observation.validate()
    assert (observation.batch_size, observation.num_satellites, observation.num_tasks) == (2, 3, 4)
    moved = observation.to(torch.device('cpu'))
    assert moved.event_state.replan_mask.shape == (2, 3)


def test_event_policy_observation_rejects_task_mask_shape_mismatch():
    observation = make_observation(batch=1, satellites=2, tasks=3)
    with pytest.raises(ValueError, match='task mask'):
        observation._replace(tasks_mask=torch.ones(1, 4, dtype=torch.bool)).validate()
```

- [x] **Step 2: 运行测试，确认因模块不存在而失败**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_event_v2_observation.py -q
```

Expected: collection error 指向 `event_v2.observation`。

- [x] **Step 3: 实现完整观测类型**

```python
class EventPolicyObservation(NamedTuple):
    time_steps: torch.Tensor
    constellation_sensor_type: torch.Tensor
    constellation_sensor_enabled: torch.Tensor
    constellation_data: torch.Tensor
    constellation_mask: torch.Tensor
    tasks_sensor_type: torch.Tensor
    tasks_data: torch.Tensor
    tasks_mask: torch.Tensor
    event_state: EventStateTensors
```

在该类型上完整实现以下 API：`batch_size` 返回 `time_steps.shape[0]`，`num_satellites` 返回 `constellation_mask.shape[1]`，`num_tasks` 返回 `tasks_mask.shape[1]`；`validate()` 严格检查 `time_steps=(B,)`、卫星输入 `(B,S,...)`、任务输入 `(B,T,...)`、bool mask 和 `EventStateTensors` 的 `(B,S)/(B,T)` 一致性；`to(device, non_blocking=False)` 递归移动普通 tensor 和 `EventStateTensors`，并返回新的 NamedTuple。实现 `stack_event_observations()`，逐字段 `torch.cat(dim=0)`，只合并相同 S/T 且单项 batch=1 的观测，避免静默 padding 错配。不要修改 `EventTransition`，确保 V2-0 schema fingerprint 不变。

- [x] **Step 4: 验证；提交因当前 Git 索引权限暂缓**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_event_v2_observation.py tests/test_event_v2_model.py tests/test_event_v2_transition.py -q
git add constellation/new_transformers/event_v2/observation.py constellation/new_transformers/event_v2/__init__.py tests/test_event_v2_observation.py
git commit -m "feat: add complete event v2 policy observations"
```

Expected: 全部 PASS，schema fingerprint 不变。

## Task 2: 纯事件状态机与承诺生命周期

**Files:**
- Create: `constellation/new_transformers/event_v2/runtime_state.py`
- Modify: `constellation/new_transformers/event_v2/__init__.py`
- Test: `tests/test_event_v2_runtime_state.py`

- [x] **Step 1: 写失败测试覆盖硬锁、外部事件和 task-id 重排**

```python
def test_locked_assignment_survives_review_until_commitment_expires():
    machine = committed_machine(seconds=15)
    events = [machine.advance_one_second(visible_snapshot(t)) for t in range(1, 16)]
    assert events[4].safety_review and not events[4].state.replan_mask[0, 0]
    assert events[-1].state.replan_mask[0, 0]
    assert events[-1].state.can_terminate_mask[0, 0]


def test_external_close_forces_replan_without_policy_termination():
    machine = committed_machine(seconds=30)
    event = machine.advance_one_second(closed_snapshot(time_step=1))
    assert event.state.forced_interrupt_mask[0, 0]
    assert not event.state.can_terminate_mask[0, 0]


def test_global_assignment_maps_to_current_relative_task_index():
    machine = machine_assigning_global_task(12)
    event = machine.advance_one_second(snapshot(time_step=1, ongoing_global_ids=(7, 12, 19)))
    assert event.state.current_task_indices.tolist() == [[1]]
```

- [x] **Step 2: 运行测试，确认因模块不存在而失败**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_event_v2_runtime_state.py -q
```

- [x] **Step 3: 实现状态机和稳定枚举**

```python
class RuntimeSnapshot(NamedTuple):
    time_step: int
    ongoing_global_task_ids: tuple[int, ...]
    task_progress: torch.Tensor
    task_required_duration: torch.Tensor
    visible: torch.Tensor
    released_global_task_ids: tuple[int, ...]
    closed_global_task_ids: tuple[int, ...]


class RuntimeEvent(NamedTuple):
    requires_policy: bool
    safety_review: bool
    state: EventStateTensors
```

实现 `EventRuntimeState.apply_joint_action(action, ongoing_global_task_ids)` 与 `advance_one_second(snapshot) -> RuntimeEvent`。前者把 action 的相对 task id 转为 global id，并用 `COMMITMENT_SECONDS[commitment_index]` 写入硬承诺；后者每次只接受 `snapshot.time_step == previous + 1`。内部保存 global task id，输出前才映射为当前 ongoing 相对索引。优先级固定为任务关闭/失效/新发布等外部事件，其次承诺到期后的 termination 复核，再其次 5 秒安全复核。硬承诺内只允许物理强制中断；同步更新 run length、30/60 秒 switch 队列、owner/locked-owner 和 termination reason。

- [x] **Step 4: 覆盖 idle、单任务、空候选、60 秒档位并验证；提交因 Git 权限暂缓**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_event_v2_runtime_state.py tests/test_event_v2_state.py tests/test_event_v2_actor.py -q
git add constellation/new_transformers/event_v2/runtime_state.py constellation/new_transformers/event_v2/__init__.py tests/test_event_v2_runtime_state.py
git commit -m "feat: add event v2 commitment state machine"
```

Expected: 全部 PASS；相邻时间严格增加，所有 commitment 最终归零。

## Task 3: Basilisk 单场 Event Runtime

**Files:**
- Create: `constellation/new_transformers/event_v2/basilisk_runtime.py`
- Test: `tests/test_event_v2_basilisk_runtime.py`

- [x] **Step 1: 用 fake backend 写逐秒推进与 telescoping reward 失败测试**

```python
def test_runtime_advances_one_second_until_next_policy_event():
    backend = FakeBackend(next_external_event_at=7)
    runtime = BasiliskEventRuntime(backend=backend, statistics=fake_statistics())
    runtime.reset()
    result = runtime.step(action_assigning_task_zero_for_five_seconds())
    assert result.delta_t == 5
    assert backend.step_times == [1, 2, 3, 4, 5]


def test_trajectory_rewards_equal_exact_terminal_quality():
    runtime = deterministic_fake_scene(progress_by_second=[0, 1, 2, 2, 3])
    rewards = run_to_done(runtime)
    assert sum(rewards) == pytest.approx(runtime.final_quality, abs=1e-6)


def test_no_candidate_counterfactual_simulators_are_created():
    runtime = counting_runtime()
    runtime.step(sample_joint_action())
    assert runtime.backend.num_scene_instances == 1
```

- [x] **Step 2: 运行测试，确认失败**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_event_v2_basilisk_runtime.py -q
```

- [x] **Step 3: 实现 backend protocol 与 runtime**

```python
class RuntimeStep(NamedTuple):
    observation: EventPolicyObservation
    reward: float
    delta_t: int
    done: bool
    final_quality: float | None
    invalid_action_count: int
```

实现 `EventPhysicsBackend` protocol 的四个明确接口：只读 `time_step`、`snapshot() -> RuntimeSnapshot`、`apply_assignments(global_task_ids)`、`step_one_second()`。`BasiliskSceneBackend.from_scene_id(split, scene_id)` 从固定数据路径创建且只创建一个 `BasiliskEnvironment`/`TaskManager`。`BasiliskEventRuntime.reset() -> EventPolicyObservation` 初始化状态机；`step(action) -> RuntimeStep` 先应用一次 action，再循环调用单秒 backend，遇到下一 policy event 或终局立即返回。真实单秒顺序固定为：visibility 写入 `TaskManager.record` → 当前联合 assignment 转 `Actions` → `take_actions` → `timer.step` → `environment.step` → 下一 snapshot。观测标准化读取 `STATISTICS_PATH`。reward 使用 `Phi(next)-Phi(current)`，终局加 `Q_final-Phi_terminal`。不得调用旧 RL reward 或为候选复制 Basilisk。

- [x] **Step 4: 验证 fake backend 和真实 scene 8 reset；提交因 Git 权限暂缓**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_event_v2_basilisk_runtime.py -q
/home/hy/miniconda3/envs/aeos/bin/python -c "from constellation.new_transformers.event_v2.basilisk_runtime import BasiliskSceneBackend; b=BasiliskSceneBackend.from_scene_id(split='train', scene_id=0); print(b.time_step)"
git add constellation/new_transformers/event_v2/basilisk_runtime.py tests/test_event_v2_basilisk_runtime.py
git commit -m "feat: connect event v2 runtime to Basilisk"
```

Expected: pytest PASS，构造 smoke 输出 `0` 且不启动长循环。

## Task 4: 同步 rollout 与行为概率重放

**Files:**
- Create: `constellation/new_transformers/event_v2/rollout.py`
- Test: `tests/test_event_v2_rollout.py`

- [x] **Step 1: 写失败测试要求采样与 learner 重放一致**

```python
def test_rollout_replays_joint_behavior_probability_exactly():
    model, runtime = seeded_model_and_fake_runtime(seed=3407)
    batch = collect_synchronous_rollout(model, [runtime], target_events=8, policy_version=0, device=torch.device('cpu'))
    replay = evaluate_rollout_actions(model, batch)
    torch.testing.assert_close(replay.log_prob, batch.behavior_log_prob, atol=1e-6, rtol=1e-6)


def test_rollout_rejects_nonpositive_time_or_invalid_action():
    with pytest.raises(RuntimeError, match='delta_t'):
        collect_from_scripted_runtime(delta_t=0)
    with pytest.raises(RuntimeError, match='invalid action'):
        collect_from_scripted_runtime(invalid_action_count=1)
```

- [x] **Step 2: 运行测试，确认失败**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_event_v2_rollout.py -q
```

- [x] **Step 3: 实现完整存储和同步 collector**

```python
class StoredEventStep(NamedTuple):
    observation: EventPolicyObservation
    action: JointEventAction
    trace: ActionTrace
    behavior_log_prob: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor
    delta_t: torch.Tensor
    next_observation: EventPolicyObservation
    done: torch.Tensor
    policy_version: int
```

实现 `collect_synchronous_rollout(model, runtimes, target_events, policy_version, device) -> list[StoredEventStep]`：在 `torch.inference_mode()` 下按 runtime 轮询，调用 `model.act()`、保存 Actor 输出、执行 runtime step，直到累计事件达到目标；终局 runtime 先计 episode 再 reset。实现 `evaluate_rollout_actions(model, steps)`：按相同 S/T bucket stack observation/action/trace，并调用 `model.evaluate_actions()`。不同场景不强行 pad 成同一 task shape；必须保存完整 observation、`action_order`、三类 mask、owner state 和 policy version。收集时拒绝非有限数、`delta_t<=0` 和任何 invalid action。

- [x] **Step 4: 验证；提交因 Git 权限暂缓**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_event_v2_rollout.py tests/test_event_v2_actor.py tests/test_event_v2_transition.py -q
git add constellation/new_transformers/event_v2/rollout.py tests/test_event_v2_rollout.py
git commit -m "feat: collect replayable synchronous event rollouts"
```

Expected: 最大 log-prob 绝对差 `<=1e-6`，trace 逐值相同。

## Task 5: 同步 PPO learner 与停止守卫

**Files:**
- Create: `constellation/new_transformers/event_v2/ppo.py`
- Test: `tests/test_event_v2_ppo.py`

- [x] **Step 1: 写 clipped loss、冻结参数和异常回滚失败测试**

```python
def test_clipped_objective_uses_joint_event_probability():
    out = clipped_ppo_objective(torch.log(torch.tensor([1.3, .7])), torch.zeros(2), torch.tensor([1., -1.]), clip_ratio=.2)
    assert out.ratio.tolist() == pytest.approx([1.3, .7])
    assert out.policy_loss.item() == pytest.approx(-0.2)


def test_update_changes_new_heads_but_not_stage3():
    trainer = seeded_trainer_with_rollout()
    frozen_before = clone_frozen_parameters(trainer.model)
    trainable_before = clone_trainable_parameters(trainer.model)
    trainer.update()
    assert_parameters_equal(frozen_before, clone_frozen_parameters(trainer.model))
    assert any_parameter_changed(trainable_before, clone_trainable_parameters(trainer.model))


def test_bad_update_restores_preupdate_state():
    trainer = seeded_trainer_with_rollout(kl_limit=1e-4)
    before = copy.deepcopy(trainer.model.state_dict())
    with pytest.raises(PPOUpdateRejected):
        trainer.update(force_excessive_kl=True)
    assert_state_dict_equal(before, trainer.model.state_dict())
```

- [x] **Step 2: 运行测试，确认失败**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_event_v2_ppo.py -q
```

- [x] **Step 3: 实现 PPO 公式与原子 update**

```python
ratio = torch.exp(new_log_prob - behavior_log_prob)
unclipped = ratio * advantages
clipped = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantages
policy_loss = -torch.minimum(unclipped, clipped).mean()
value_loss = 0.5 * (new_value - returns).square().mean()
loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy.mean()
```

优势必须调用已有 `time_aware_gae`，参数 `gamma=1`、`lambda_base=0.95`、`reference_seconds=5`。每个 PPO epoch 用行为 trace 调 `model.evaluate_actions()`，不得重采样。update 前复制可训练 state；NaN/Inf、replay 超差、KL 超限或梯度异常时恢复。Stage3 `requires_grad=False` 且 update 后逐 tensor 完全相等。记录 ratio、clip fraction、KL、entropy、value loss、grad norm 与动作分布。

- [x] **Step 4: 验证；提交因 Git 权限暂缓**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_event_v2_ppo.py tests/test_event_v2_reward.py tests/test_event_v2_model.py -q
git add constellation/new_transformers/event_v2/ppo.py tests/test_event_v2_ppo.py
git commit -m "feat: add guarded synchronous event PPO updates"
```

## Task 6: checkpoint 与第一批恢复动作复现

**Files:**
- Create: `constellation/new_transformers/event_v2/checkpoint.py`
- Modify: `tools/train_event_v2_warm_start.py`
- Test: `tests/test_event_v2_checkpoint.py`
- Test: `tests/test_event_v2_warm_start.py`

- [x] **Step 1: 写 checkpoint round-trip 失败测试**

```python
def test_checkpoint_restores_rng_runtime_cursor_and_first_actions(tmp_path):
    trainer = seeded_trainer(seed=3407)
    save_checkpoint_atomic(tmp_path / 'v2_1.pth', build_sync_ppo_checkpoint(trainer))
    expected = trainer.sample_first_batch_actions()
    restored = fresh_trainer(seed=999)
    load_sync_ppo_checkpoint(tmp_path / 'v2_1.pth', restored)
    assert_joint_actions_equal(expected, restored.sample_first_batch_actions())


def test_checkpoint_rejects_stage_schema_config_or_scene_mismatch(tmp_path):
    path = write_corrupt_checkpoint(tmp_path, field='scene_ids')
    with pytest.raises(ValueError, match='scene'):
        load_for_test(path)
```

- [x] **Step 2: 运行测试，确认失败**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_event_v2_checkpoint.py -q
```

- [x] **Step 3: 实现 V2-1 checkpoint schema 与 Basilisk 确定性回放恢复**

```python
checkpoint = {
    'checkpoint_version': 1,
    'stage': 'V2-1',
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'amp_scaler': scaler.state_dict(),
    'policy_version': policy_version,
    'transition_schema_fingerprint': transition_schema_fingerprint(),
    'config_fingerprint': config_fingerprint(config),
    'normalizer': normalizer,
    'rng_state': capture_rng_state(),
    'runtime_states': [runtime.state_dict() for runtime in runtimes],
    'scene_ids': scene_ids,
    'processed_physical_seconds': physical_seconds,
    'episodes': episodes,
    'events': events,
    'updates': updates,
    'unfreeze_state': {'backbone_is_frozen': True},
}
```

共享 RNG/config/atomic-save helper 移到 `checkpoint.py`，V2-0 trainer 改为导入它，保持旧 API 和旧 checkpoint 可加载。runtime state 保存 time、global assignment、commitment、历史窗口与 scene cursor。

- [x] **Step 4: 验证 V2-1、V2-0 与真实 scene 8 物理恢复；提交因 Git 权限暂缓**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_event_v2_checkpoint.py tests/test_event_v2_warm_start.py -q
git add constellation/new_transformers/event_v2/checkpoint.py tools/train_event_v2_warm_start.py tests/test_event_v2_checkpoint.py tests/test_event_v2_warm_start.py
git commit -m "feat: make event PPO checkpoints exactly resumable"
```

## Task 7: 配置、训练 CLI、审计摘要和 Slurm 包装

**Files:**
- Create: `constellation/new_transformers/config_event_v2_sync_ppo.py`
- Create: `tools/train_event_v2_sync_ppo.py`
- Create: `scripts/train_event_v2_sync_ppo_slurm.sh`
- Test: `tests/test_event_v2_sync_ppo_scripts.py`

- [x] **Step 1: 写配置和脚本失败测试**

```python
def test_config_is_train_only_and_keeps_stage3_frozen():
    config = runpy.run_path('constellation/new_transformers/config_event_v2_sync_ppo.py')
    assert config['stage'] == 'V2-1'
    assert config['split'] == 'train'
    assert config['freeze_backbone'] is True
    assert config['max_hours'] == 4
    assert config['safety_review_seconds'] == 5
    assert config['gamma'] == 1.0


def test_slurm_wrapper_uses_local_10_aeos_and_3600_seconds():
    script = Path('scripts/train_event_v2_sync_ppo_slurm.sh').read_text()
    assert '#SBATCH --partition=local-10' in script
    assert '#SBATCH --time=04:00:00' in script
    assert '/home/hy/miniconda3/envs/aeos/bin/python' in script
    assert '--max-time-step 3600' in script
```

- [x] **Step 2: 运行测试，确认文件不存在而失败**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_event_v2_sync_ppo_scripts.py -q
```

- [x] **Step 3: 实现预注册配置和 CLI**

```python
stage = 'V2-1'
split = 'train'
seed = 3407
scene_ids = (0, 1, 2, 3)
max_time_step = 3600
max_hours = 4
safety_review_seconds = 5
rollout_events_per_update = 256
ppo_epochs = 4
minibatch_events = 64
gamma = 1.0
lambda_base = 0.95
reference_seconds = 5.0
clip_ratio = 0.2
value_coefficient = 0.5
entropy_coefficient = 0.01
max_grad_norm = 1.0
max_kl = 0.03
logprob_replay_atol = 1e-6
freeze_backbone = True
amp = True
amp_dtype = 'bfloat16'
```

CLI 支持 `--synthetic-preflight`、`--scene-ids`、`--max-time-step`、`--max-updates`、`--resume`、`--device`、`--output`。顺序固定为：加载 V2-0 10k → synthetic preflight → 真实短 preflight → 少量 train scenes 同步 rollout/update → 保存 checkpoint 与 `summary.json`。

摘要至少记录 reward reconstruction/replay 最大误差、冻结 tensor 变化数、finite/invalid/time/commitment 计数、恢复动作一致性、episode/event/physical seconds、PPO ratio/KL/clip/entropy/value/grad、动作分布、显存峰值和 `accepted`。

- [x] **Step 4: 实现 Slurm 包装并验证；提交因 Git 权限暂缓**

脚本使用 `local-10`、最长 `04:00:00`、`aeos` Python，输出到 `work_dirs/event_joint_transformer_v2/v2_1_sync_ppo`，日志为 `work_dirs/eval_logs/event_v2_sync_ppo_%j.log`，读取 V2-0 `checkpoint_step_010000.pth`。先跑 synthetic preflight，再跑真实 3,600 秒。

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_event_v2_sync_ppo_scripts.py -q
bash -n scripts/train_event_v2_sync_ppo_slurm.sh
git add constellation/new_transformers/config_event_v2_sync_ppo.py tools/train_event_v2_sync_ppo.py scripts/train_event_v2_sync_ppo_slurm.sh tests/test_event_v2_sync_ppo_scripts.py
git commit -m "feat: add audited V2-1 synchronous PPO runner"
```

## Task 8: CPU 回归与合成闭环验收

**Files:**
- Modify only if a scoped V2-1 defect is exposed.

- [x] **Step 1: 运行全部 V2 单元测试（124 passed）**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_event_v2_*.py -q
```

- [x] **Step 2: 运行 CPU synthetic preflight（另加真实 scene 0 的 10 秒 CPU smoke）**

```bash
/home/hy/miniconda3/envs/aeos/bin/python tools/train_event_v2_sync_ppo.py --config constellation/new_transformers/config_event_v2_sync_ppo.py --synthetic-preflight --device cpu --max-updates 2 --output /tmp/event_v2_sync_ppo_preflight
```

Expected: `accepted=true`；reward/replay error `<=1e-6`；frozen changed、invalid、time、commitment violation 均为 0；恢复动作一致。

- [x] **Step 3: 运行静态检查**

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m compileall -q constellation/new_transformers/event_v2 tools/train_event_v2_sync_ppo.py
git diff --check
```

Expected: 退出码均为 0。若修复缺陷，单独提交 `fix: satisfy V2-1 synthetic acceptance`；无修复则不创建空提交。

## Task 9: Slurm 真实 3,600 秒 smoke 与证据入账

**Files:**
- Modify: `TODO.md`
- Modify: `改进日志.md`（只追加 V2-1 事实，不覆盖既有 M3 改动）
- Generated, do not commit: `work_dirs/event_joint_transformer_v2/v2_1_sync_ppo/**`
- Generated, do not commit: `work_dirs/eval_logs/event_v2_sync_ppo_<job>.log`

- [ ] **Step 1: 检查资源并提交**

```bash
nvidia-smi
sbatch --test-only scripts/train_event_v2_sync_ppo_slurm.sh
JOB_ID="$(sbatch --parsable scripts/train_event_v2_sync_ppo_slurm.sh)"
printf '%s\n' "${JOB_ID}"
```

Expected: Slurm 返回 job id。立即在 `TODO.md` 记录 job、命令、日志、checkpoint 和输出目录，状态写“运行中”，不提前勾选验收。

- [ ] **Step 2: 等待并核对三方证据**

```bash
sacct -j "${JOB_ID}" --format=JobID,State,ExitCode,Elapsed,MaxRSS,AllocTRES -P
sed -n '1,260p' "work_dirs/eval_logs/event_v2_sync_ppo_${JOB_ID}.log"
/home/hy/miniconda3/envs/aeos/bin/python -m json.tool work_dirs/event_joint_transformer_v2/v2_1_sync_ppo/summary.json
```

Expected: `COMPLETED|0:0` 且所有 V2-1 correctness gate 通过。本阶段不要求 Q/CR/PCR/WCR 提升。

- [ ] **Step 3: 严格执行失败边界**

NaN/Inf、概率重放、冻结参数、invalid action、时间、承诺或恢复复现任一失败：停在 V2-1，写最小复现测试并修复，不进入 V2-2。仅 OOM 可把 learner minibatch 降低一次后按原 reward/动作/环境数重试。

- [ ] **Step 4: 更新 TODO/改进日志并最终验证**

只勾选实际通过项；记录 job id、elapsed、scene ids、events、physical seconds、最大 reward/replay error、冻结变化数、恢复一致性、日志与 checkpoint。同步修正 TODO 底部已经完成但仍未勾选的 V2-0 unseen 验收旧项。

```bash
/home/hy/miniconda3/envs/aeos/bin/python -m pytest tests/test_event_v2_*.py -q
git diff --check
git diff --cached --name-only
```

暂存范围必须排除当前工作树已有 M3 文件与其他用户改动；随后提交：

```bash
git commit -m "feat: complete V2-1 synchronous PPO correctness stage"
```

## V2-1 完成后的唯一下一步

真实 3,600 秒 smoke 全部门槛通过后，才另写并执行 V2-2 同步 PPO 收益计划。V2-2 才扩大到 12–16 小时、按固定 held-out train scenes 的 Q 选 checkpoint，并只运行一次 Val Seen/Unseen 8+8。因此“全量数据训练”不是当前下一步；在 correctness 尚未证明前不会启动。
