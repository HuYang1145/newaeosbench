import pytest
import torch

from constellation.new_transformers.event_v2.basilisk_runtime import RuntimeStep
from constellation.new_transformers.event_v2.model import EventJointActorCritic
from constellation.new_transformers.event_v2.observation import (
    EventPolicyObservation,
)
from constellation.new_transformers.event_v2.rollout import (
    SynchronousRuntimeSlot,
    collect_synchronous_rollout,
    replay_rollout_log_probs,
)
from constellation.new_transformers.event_v2.state import EventStateTensors


def _model() -> EventJointActorCritic:
    return EventJointActorCritic(
        event_width=8,
        sensor_type_embedding_dim=4,
        tasks_data_embedding_dim=4,
        encoder_width=8,
        encoder_depth=1,
        encoder_num_heads=2,
        sensor_enabled_embedding_dim=4,
        constellation_data_embedding_dim=4,
        decoder_width=8,
        decoder_depth=1,
        decoder_num_heads=2,
        use_constraint_module=False,
        use_sdpa=False,
    ).eval()


def _state(time_step: int) -> EventStateTensors:
    return EventStateTensors(
        previous_task_indices=torch.tensor([[-1, -1]]),
        current_task_indices=torch.tensor([[-1, -1]]),
        minimum_commitment_remaining=torch.zeros(1, 2),
        run_lengths=torch.full((1, 2), float(time_step)),
        seconds_since_replan=torch.full((1, 2), float(time_step)),
        switch_count_30=torch.zeros(1, 2),
        switch_count_60=torch.zeros(1, 2),
        termination_reason=torch.zeros(1, 2, dtype=torch.long),
        event_type=torch.full((1, 2), 3, dtype=torch.long),
        delta_t=torch.full((1, 2), 5.),
        replan_mask=torch.ones(1, 2, dtype=torch.bool),
        forced_interrupt_mask=torch.zeros(1, 2, dtype=torch.bool),
        can_terminate_mask=torch.zeros(1, 2, dtype=torch.bool),
        compatible_deadline_slack=torch.tensor([[10., 20.]]),
        task_remaining_required_seconds=torch.tensor([[10., 30., 60.]]),
        task_owner_count=torch.zeros(1, 3, dtype=torch.long),
        task_locked_owner_count=torch.zeros(1, 3, dtype=torch.long),
    )


def _observation(time_step: int) -> EventPolicyObservation:
    observation = EventPolicyObservation(
        time_steps=torch.tensor([time_step]),
        constellation_sensor_type=torch.zeros(1, 2, dtype=torch.long),
        constellation_sensor_enabled=torch.ones(1, 2, dtype=torch.long),
        constellation_data=torch.zeros(1, 2, 56),
        constellation_mask=torch.ones(1, 2, dtype=torch.bool),
        tasks_sensor_type=torch.zeros(1, 3, dtype=torch.long),
        tasks_data=torch.zeros(1, 3, 6),
        tasks_mask=torch.ones(1, 3, dtype=torch.bool),
        event_state=_state(time_step),
    )
    observation.validate()
    return observation


class ScriptedRuntime:
    def __init__(
        self,
        *,
        num_events: int = 8,
        delta_t: int = 5,
        invalid_action_count: int = 0,
    ) -> None:
        self._num_events = num_events
        self._delta_t = delta_t
        self._invalid_action_count = invalid_action_count
        self._events = 0

    def reset(self) -> EventPolicyObservation:
        return _observation(0)

    def step(self, action) -> RuntimeStep:
        del action
        self._events += 1
        done = self._events >= self._num_events
        next_observation = None if done else _observation(
            self._events * max(self._delta_t, 1),
        )
        return RuntimeStep(
            observation=next_observation,
            reward=float(self._events) / 100,
            delta_t=self._delta_t,
            done=done,
            final_quality=(0.5 if done else None),
            invalid_action_count=self._invalid_action_count,
        )


def test_rollout_replays_joint_behavior_probability_exactly() -> None:
    torch.manual_seed(3407)
    model = _model()
    runtime = ScriptedRuntime()
    slot = SynchronousRuntimeSlot(
        environment_index=0,
        episode_id=0,
        observation=runtime.reset(),
        runtime=runtime,
    )

    steps = collect_synchronous_rollout(
        model,
        [slot],
        target_events=6,
        policy_version=3,
        device=torch.device('cpu'),
    )
    replay = replay_rollout_log_probs(
        model,
        steps,
        device=torch.device('cpu'),
    )
    behavior = torch.stack([step.behavior_log_prob for step in steps])

    assert len(steps) == 6
    torch.testing.assert_close(replay, behavior, atol=1e-6, rtol=1e-6)
    assert all(step.policy_version == 3 for step in steps)
    assert [step.event_index for step in steps] == list(range(6))


def test_bfloat16_replay_keeps_behavior_batch_shape(monkeypatch) -> None:
    torch.manual_seed(3407)
    model = _model()
    runtime = ScriptedRuntime()
    slot = SynchronousRuntimeSlot(
        environment_index=0,
        episode_id=0,
        observation=runtime.reset(),
        runtime=runtime,
    )

    steps = collect_synchronous_rollout(
        model,
        [slot],
        target_events=6,
        policy_version=0,
        device=torch.device('cpu'),
        amp_enabled=True,
        amp_dtype=torch.bfloat16,
    )
    replay_batch_sizes: list[int] = []
    evaluate_actions = model.evaluate_actions

    def record_batch_size(*args, **kwargs):
        replay_batch_sizes.append(int(args[0].shape[0]))
        return evaluate_actions(*args, **kwargs)

    monkeypatch.setattr(model, 'evaluate_actions', record_batch_size)
    replay = replay_rollout_log_probs(
        model,
        steps,
        device=torch.device('cpu'),
        amp_enabled=True,
        amp_dtype=torch.bfloat16,
    )

    behavior = torch.stack([step.behavior_log_prob for step in steps])
    assert replay_batch_sizes == [1] * len(steps)
    torch.testing.assert_close(replay, behavior, atol=1e-6, rtol=1e-6)


def test_rollout_keeps_terminal_transition_and_zero_bootstrap() -> None:
    torch.manual_seed(11)
    model = _model()
    runtime = ScriptedRuntime(num_events=2)
    slot = SynchronousRuntimeSlot(
        environment_index=4,
        episode_id=9,
        observation=runtime.reset(),
        runtime=runtime,
    )

    steps = collect_synchronous_rollout(
        model,
        [slot],
        target_events=8,
        policy_version=0,
        device=torch.device('cpu'),
    )

    assert len(steps) == 2
    assert steps[-1].done.item()
    assert steps[-1].next_observation is None
    assert steps[-1].next_value.item() == 0
    assert slot.finished


@pytest.mark.parametrize(
    ('delta_t', 'invalid_action_count', 'message'),
    [
        (0, 0, 'delta_t'),
        (5, 1, 'invalid action'),
    ],
)
def test_rollout_rejects_invalid_runtime_results(
    delta_t: int,
    invalid_action_count: int,
    message: str,
) -> None:
    model = _model()
    runtime = ScriptedRuntime(
        delta_t=delta_t,
        invalid_action_count=invalid_action_count,
    )
    slot = SynchronousRuntimeSlot(
        environment_index=0,
        episode_id=0,
        observation=runtime.reset(),
        runtime=runtime,
    )

    with pytest.raises(RuntimeError, match=message):
        collect_synchronous_rollout(
            model,
            [slot],
            target_events=1,
            policy_version=0,
            device=torch.device('cpu'),
        )


def test_rollout_rejects_empty_or_finished_slot_sets() -> None:
    model = _model()

    with pytest.raises(ValueError, match='active runtime'):
        collect_synchronous_rollout(
            model,
            [],
            target_events=1,
            policy_version=0,
            device=torch.device('cpu'),
        )
