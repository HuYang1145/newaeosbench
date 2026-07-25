from collections.abc import Sequence
from types import SimpleNamespace

import pytest
import torch

from constellation.new_transformers.dataset import Statistics
from constellation.new_transformers.event_v2.basilisk_runtime import (
    BasiliskEventRuntime,
    BasiliskSceneBackend,
    CompletionSnapshot,
    load_runtime_statistics,
)
from constellation.new_transformers.event_v2.observation import (
    EventPolicyObservation,
)
from constellation.new_transformers.event_v2.runtime_state import RuntimeSnapshot
from constellation.new_transformers.event_v2.state import EventStateTensors
from constellation.new_transformers.event_v2.transition import JointEventAction


class FakeBackend:
    num_scene_instances = 1

    def __init__(
        self,
        *,
        progress_by_second: Sequence[float],
        required_duration: float,
    ) -> None:
        self.num_satellites = 1
        self.time_step = 0
        self.step_times: list[int] = []
        self._progress = tuple(progress_by_second)
        self._required_duration = required_duration
        self._assignments = (-1,)

    @property
    def done(self) -> bool:
        return self.time_step >= len(self._progress) - 1

    def snapshot(self) -> RuntimeSnapshot:
        return RuntimeSnapshot(
            time_step=self.time_step,
            ongoing_global_task_ids=(12,),
            task_progress=torch.tensor([self._progress[self.time_step]]),
            task_required_duration=torch.tensor([self._required_duration]),
            task_deadline_slack=torch.tensor([
                float(len(self._progress) - self.time_step),
            ]),
            task_compatibility=torch.ones(1, 1, dtype=torch.bool),
            assignment_valid=torch.ones(1, dtype=torch.bool),
            released_global_task_ids=(),
            closed_global_task_ids=(),
        )

    def completion_snapshot(self) -> CompletionSnapshot:
        progress = torch.tensor([self._progress[self.time_step]])
        required = torch.tensor([self._required_duration])
        return CompletionSnapshot(
            progress=progress,
            required_duration=required,
            completed=progress >= required,
        )

    def apply_assignments(self, global_task_ids: Sequence[int]) -> None:
        self._assignments = tuple(global_task_ids)

    def step_one_second(self) -> None:
        if self.done:
            raise RuntimeError('fake backend is already done')
        self.time_step += 1
        self.step_times.append(self.time_step)

    def build_observation(
        self,
        state: EventStateTensors,
        statistics: Statistics,
    ) -> EventPolicyObservation:
        del statistics
        return EventPolicyObservation(
            time_steps=torch.tensor([self.time_step]),
            constellation_sensor_type=torch.zeros(1, 1, dtype=torch.long),
            constellation_sensor_enabled=torch.ones(1, 1, dtype=torch.long),
            constellation_data=torch.zeros(1, 1, 56),
            constellation_mask=torch.ones(1, 1, dtype=torch.bool),
            tasks_sensor_type=torch.zeros(1, 1, dtype=torch.long),
            tasks_data=torch.zeros(1, 1, 6),
            tasks_mask=torch.ones(1, 1, dtype=torch.bool),
            event_state=state,
        )


class GapBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__(
            progress_by_second=[0., 0., 0., 0., 0.],
            required_duration=30.,
        )

    def snapshot(self) -> RuntimeSnapshot:
        if self.time_step == 0:
            ongoing = (12,)
            released = ()
            closed = ()
        elif self.time_step < 3:
            ongoing = ()
            released = ()
            closed = ((12,) if self.time_step == 1 else ())
        else:
            ongoing = (13,)
            released = ((13,) if self.time_step == 3 else ())
            closed = ()
        num_tasks = len(ongoing)
        return RuntimeSnapshot(
            time_step=self.time_step,
            ongoing_global_task_ids=ongoing,
            task_progress=torch.zeros(num_tasks),
            task_required_duration=torch.full((num_tasks,), 30.),
            task_deadline_slack=torch.full((num_tasks,), 10.),
            task_compatibility=torch.ones(1, num_tasks, dtype=torch.bool),
            assignment_valid=torch.tensor([
                self._assignments[0] < 0 or self._assignments[0] in ongoing,
            ]),
            released_global_task_ids=released,
            closed_global_task_ids=closed,
        )


def _statistics() -> Statistics:
    return Statistics(
        constellation_mean=torch.zeros(56),
        constellation_std=torch.ones(56),
        taskset_mean=torch.zeros(6),
        taskset_std=torch.ones(6),
    )


def _assignment(commitment_index: int = 1) -> JointEventAction:
    return JointEventAction(
        terminate=torch.tensor([[False]]),
        task_indices=torch.tensor([[0]]),
        commitment_indices=torch.tensor([[commitment_index]]),
    )


def _keep() -> JointEventAction:
    return JointEventAction(
        terminate=torch.tensor([[False]]),
        task_indices=torch.tensor([[-1]]),
        commitment_indices=torch.tensor([[-1]]),
    )


def test_runtime_advances_one_second_until_next_policy_event() -> None:
    backend = FakeBackend(
        progress_by_second=[0.] * 8,
        required_duration=30.,
    )
    runtime = BasiliskEventRuntime(backend=backend, statistics=_statistics())
    runtime.reset()

    result = runtime.step(_assignment(commitment_index=1))

    assert result.delta_t == 5
    assert backend.step_times == [1, 2, 3, 4, 5]
    assert result.observation is not None
    assert result.observation.time_steps.item() == 5


def test_trajectory_rewards_equal_exact_terminal_quality() -> None:
    backend = FakeBackend(
        progress_by_second=[0., 1., 1., 1., 1., 2., 3.],
        required_duration=3.,
    )
    runtime = BasiliskEventRuntime(backend=backend, statistics=_statistics())
    runtime.reset()

    first = runtime.step(_assignment(commitment_index=1))
    terminal = runtime.step(_keep())

    assert terminal.done
    assert terminal.observation is None
    assert terminal.final_quality == pytest.approx(1.)
    assert first.reward + terminal.reward == pytest.approx(1., abs=1e-6)
    assert runtime.reward_reconstruction_error == pytest.approx(0., abs=1e-6)


def test_runtime_never_creates_candidate_counterfactual_backends() -> None:
    backend = FakeBackend(
        progress_by_second=[0.] * 7,
        required_duration=30.,
    )
    runtime = BasiliskEventRuntime(backend=backend, statistics=_statistics())
    runtime.reset()

    runtime.step(_assignment(commitment_index=1))

    assert runtime.backend is backend
    assert backend.num_scene_instances == 1


def test_runtime_skips_mid_scene_task_gap_without_calling_policy_on_empty_set() -> None:
    backend = GapBackend()
    runtime = BasiliskEventRuntime(backend=backend, statistics=_statistics())
    runtime.reset()

    result = runtime.step(_assignment(commitment_index=1))

    assert result.delta_t == 3
    assert not result.done
    assert result.observation is not None
    assert result.observation.time_steps.item() == 3
    assert result.observation.num_tasks == 1
    assert runtime.current_global_task_ids == (-1,)


def test_runtime_rejects_step_after_terminal() -> None:
    backend = FakeBackend(
        progress_by_second=[0., 1.],
        required_duration=1.,
    )
    runtime = BasiliskEventRuntime(backend=backend, statistics=_statistics())
    runtime.reset()
    runtime.step(_assignment(commitment_index=0))

    with pytest.raises(RuntimeError, match='finished'):
        runtime.step(_keep())


def test_real_runtime_state_round_trip_replays_physics_to_same_observation() -> None:
    statistics = load_runtime_statistics()
    runtime = BasiliskEventRuntime(
        backend=BasiliskSceneBackend.from_scene_id(
            split='train',
            scene_id=8,
            max_time_step=2,
        ),
        statistics=statistics,
    )
    expected = runtime.reset()

    restored = BasiliskEventRuntime.from_state_dict(
        runtime.state_dict(),
        statistics=statistics,
    )
    actual = restored.current_observation

    assert restored.backend.time_step == runtime.backend.time_step == 1
    assert restored.current_global_task_ids == runtime.current_global_task_ids
    for field in expected._fields[:-1]:
        torch.testing.assert_close(getattr(actual, field), getattr(expected, field))
    for field in expected.event_state._fields:
        torch.testing.assert_close(
            getattr(actual.event_state, field),
            getattr(expected.event_state, field),
        )


def test_basilisk_backend_operational_metrics_match_tat_and_power_units() -> None:
    backend = BasiliskSceneBackend.__new__(BasiliskSceneBackend)
    backend._task_manager = SimpleNamespace(
        succeeded_flags=torch.tensor([True, True, False]),
    )
    backend._taskset = SimpleNamespace(
        release_times=torch.tensor([0.0, 10.0, 20.0]),
    )
    backend._completion_time = torch.tensor([10.0, 30.0, float('inf')])
    backend._working_time_steps = torch.tensor([10.0, 5.0])
    backend._sensor_power = torch.tensor([20.0, 40.0])

    metrics = backend.operational_metrics()

    assert metrics['TAT_s'] == pytest.approx(15.0)
    assert metrics['PC_Wh'] == pytest.approx(
        (10 * 20 + 5 * 40) / 3600,
    )
