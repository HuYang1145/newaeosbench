from types import SimpleNamespace

import pytest
import torch

from constellation.algorithms.base import BaseAlgorithm
from constellation.data import Action, Actions, Coordinate, Task, TaskSet
from constellation.data.constellations import SensorType
from constellation.new_transformers.local_action_branch import (
    ControlledActionAlgorithm,
    LocalWindowCallback,
    find_stay_switch_decisions,
    is_decision_replayable,
    summarize_prefix_paper_metrics,
    summarize_local_window,
)


class _StubAlgorithm(BaseAlgorithm):

    def prepare(self, environment, task_manager) -> None:
        self.prepared = (environment, task_manager)

    def step(self, taskset, constellation, earth_rotation):
        del taskset, earth_rotation
        return (
            Actions(Action() for _ in constellation.sort()),
            [-1, 5],
        )


class _StubRankedAlgorithm(_StubAlgorithm):

    def step(self, taskset, constellation, earth_rotation):
        result = super().step(taskset, constellation, earth_rotation)
        self.last_logits = torch.tensor([[[0.0, 1.0, 3.0], [3.0, 2.0, 1.0]]])
        self.last_task_ids = torch.tensor([3, 5])
        return result


class _FakeConstellation:

    def __init__(self) -> None:
        self._satellites = [
            SimpleNamespace(sensor=SimpleNamespace(enabled=False)),
            SimpleNamespace(sensor=SimpleNamespace(enabled=True)),
        ]

    def __len__(self) -> int:
        return len(self._satellites)

    def sort(self):
        return self._satellites

    def dynamic_to_tensor(self):
        return torch.tensor([False, True]), torch.zeros((2, 8))

    def static_to_tensor(self):
        return torch.tensor([1, 1]), torch.zeros((2, 4))


class _FakeController:
    pass


def _tasks() -> TaskSet:
    return TaskSet([
        Task(
            id_=3,
            release_time=0,
            due_time=100,
            duration=15,
            coordinate=Coordinate(10.0, 20.0),
            sensor_type=SensorType.VISIBLE,
        ),
        Task(
            id_=5,
            release_time=0,
            due_time=100,
            duration=15,
            coordinate=Coordinate(30.0, 40.0),
            sensor_type=SensorType.VISIBLE,
        ),
    ])


def test_find_stay_switch_decisions_selects_one_second_pulses() -> None:
    actions = torch.tensor([
        [-1, 5],
        [3, 5],
        [-1, 7],
        [-1, 7],
        [4, 7],
        [-1, 7],
    ])

    decisions = find_stay_switch_decisions(
        actions,
        max_decisions=2,
        latest_decision_time=4,
    )

    assert [decision.to_dict() for decision in decisions] == [
        {
            'decision_time': 1,
            'satellite_index': 0,
            'stay_task_id': -1,
            'switch_task_id': 3,
            'pattern': 'idle_task_idle',
        },
        {
            'decision_time': 4,
            'satellite_index': 0,
            'stay_task_id': -1,
            'switch_task_id': 4,
            'pattern': 'idle_task_idle',
        },
    ]


def test_decision_replayability_rejects_already_completed_task() -> None:
    decision = find_stay_switch_decisions(
        torch.tensor([
            [3],
            [3],
            [5],
            [3],
        ]),
        max_decisions=1,
        latest_decision_time=2,
    )[0]
    progress = torch.tensor([
        [0, 0],
        [15, 0],
        [0, 0],
        [0, 0],
    ],
                            dtype=torch.uint8)

    assert not is_decision_replayable(
        decision,
        taskset=_tasks(),
        reference_progress=progress,
    )

    progress.zero_()
    assert is_decision_replayable(
        decision,
        taskset=_tasks(),
        reference_progress=progress,
    )


def test_controlled_algorithm_overrides_only_one_step_and_satellite() -> None:
    timer = SimpleNamespace(time=5)
    base = _StubAlgorithm(timer=timer)
    algorithm = ControlledActionAlgorithm(
        timer=timer,
        base_algorithm=base,
        decision_time=5,
        satellite_index=0,
        forced_task_id=3,
    )
    constellation = _FakeConstellation()
    task_manager = SimpleNamespace(progress=torch.tensor([1, 2]))
    algorithm.prepare(
        environment=SimpleNamespace(),
        task_manager=task_manager,
    )

    actions, assignment = algorithm.step(
        _tasks(),
        constellation,
        torch.eye(3),
    )

    assert assignment == [3, 5]
    assert actions[0].toggle is True
    assert actions[0].target_location == Coordinate(10.0, 20.0)
    assert actions[1] == Action()
    assert algorithm.override_applied
    assert algorithm.original_task_id == -1
    assert algorithm.original_assignment == [-1, 5]
    assert len(algorithm.decision_state_signature) == 64

    timer.time = 6
    actions, assignment = algorithm.step(
        _tasks(),
        constellation,
        torch.eye(3),
    )
    assert assignment == [-1, 5]
    assert actions == Actions([Action(), Action()])


def test_controlled_algorithm_rejects_task_that_is_not_ongoing() -> None:
    timer = SimpleNamespace(time=5)
    algorithm = ControlledActionAlgorithm(
        timer=timer,
        base_algorithm=_StubAlgorithm(timer=timer),
        decision_time=5,
        satellite_index=0,
        forced_task_id=99,
    )
    algorithm.prepare(
        environment=SimpleNamespace(),
        task_manager=SimpleNamespace(
            progress=torch.tensor([0, 0]),
            taskset=_tasks(),
        ),
    )

    with pytest.raises(ValueError, match='forced task 99 is not ongoing'):
        algorithm.step(_tasks(), _FakeConstellation(), torch.eye(3))


def test_controlled_algorithm_can_force_a_ranked_actor_candidate() -> None:
    timer = SimpleNamespace(time=5)
    algorithm = ControlledActionAlgorithm(
        timer=timer,
        base_algorithm=_StubRankedAlgorithm(timer=timer),
        decision_time=5,
        satellite_index=0,
        forced_candidate_rank=1,
    )
    algorithm.prepare(
        environment=SimpleNamespace(),
        task_manager=SimpleNamespace(
            progress=torch.tensor([0, 0]),
            taskset=_tasks(),
        ),
    )

    timer.time = 4
    algorithm.step(_tasks(), _FakeConstellation(), torch.eye(3))
    timer.time = 5

    _, assignment = algorithm.step(
        _tasks(),
        _FakeConstellation(),
        torch.eye(3),
    )

    assert assignment == [3, 5]
    assert algorithm.applied_task_id == 3
    context = algorithm.decision_context
    assert context['previous_assignment'] == [-1, 5]
    assert context['run_lengths'] == [1, 1]
    assert context['switch_counts_30'] == [0, 0]
    assert context['switch_counts_60'] == [0, 0]
    assert context['ongoing_task_ids'] == [3, 5]
    assert context['actor_logits'][0] == [0.0, 1.0, 3.0]
    assert len(context['satellite_features']) == 2
    assert len(context['task_features']) == 2
    assert context['uses_is_visible_as_input'] is False


def test_summarize_local_window_keeps_raw_causal_components() -> None:
    summary = summarize_local_window(
        assignments=torch.tensor([
            [-1, 2],
            [3, 2],
            [-1, 2],
        ]),
        assignment_before=torch.tensor([-1, 2]),
        assignment_after=torch.tensor([-1, 2]),
        progress=torch.tensor([
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 0],
        ]),
        succeeded_before=torch.tensor([False, False]),
        succeeded_after=torch.tensor([True, False]),
        direct_visible=torch.tensor([
            [False, True],
            [False, True],
            [False, False],
        ]),
        durations=torch.tensor([2, 4]),
        sensor_power=torch.tensor([10.0, 20.0]),
        target_satellite_index=0,
    )

    assert summary['horizon'] == 3
    assert summary['completed_tasks'] == 1
    assert summary['completed_duration'] == 2
    assert summary['partial_progress_gain'] == pytest.approx(1.5)
    assert summary['working_satellite_seconds'] == 4
    assert summary['pc_wh'] == pytest.approx(70.0 / 3600.0)
    assert summary['switches'] == 2
    assert summary['target_satellite_switches'] == 2
    assert summary['one_second_runs'] == 1
    assert summary['target_satellite_one_second_runs'] == 1
    assert summary['direct_visible_satellite_seconds'] == 2
    assert summary['target_satellite_direct_visible_seconds'] == 0


def test_summarize_prefix_paper_metrics_matches_snapshot_formula() -> None:
    metrics = summarize_prefix_paper_metrics(
        max_progress=torch.tensor([10.0, 5.0]),
        succeeded=torch.tensor([True, False]),
        completion_time=torch.tensor([120.0, float('inf')]),
        release_times=torch.tensor([20.0, 0.0]),
        durations=torch.tensor([10.0, 10.0]),
        local_pc_wh=2.0,
    )

    # CR=0.5, PCR=0.75, WCR=0.5, quality=0.55.
    expected_cost = 1.0 / 0.55 + 100.0 / 700.0 + 2.0 / 100.0
    assert metrics['cr'] == pytest.approx(0.5)
    assert metrics['pcr'] == pytest.approx(0.75)
    assert metrics['wcr'] == pytest.approx(0.5)
    assert metrics['quality'] == pytest.approx(0.55)
    assert metrics['tat_s'] == pytest.approx(100.0)
    assert metrics['prefix_cost'] == pytest.approx(expected_cost)


def test_summarize_prefix_paper_metrics_marks_zero_quality_unrankable(
) -> None:
    metrics = summarize_prefix_paper_metrics(
        max_progress=torch.zeros(2),
        succeeded=torch.zeros(2, dtype=torch.bool),
        completion_time=torch.full((2, ), float('inf')),
        release_times=torch.zeros(2),
        durations=torch.ones(2),
        local_pc_wh=0.0,
    )

    assert metrics['quality'] == 0.0
    assert metrics['prefix_cost'] is None


def test_local_window_callback_aligns_action_with_next_step_visibility(
) -> None:
    timer = SimpleNamespace(time=0)
    taskset = TaskSet([
        Task(
            id_=2,
            release_time=0,
            due_time=100,
            duration=2,
            coordinate=Coordinate(0.0, 0.0),
            sensor_type=SensorType.VISIBLE,
        ),
        Task(
            id_=3,
            release_time=0,
            due_time=100,
            duration=2,
            coordinate=Coordinate(10.0, 10.0),
            sensor_type=SensorType.VISIBLE,
        ),
    ])
    task_manager = SimpleNamespace(
        taskset=taskset,
        progress=torch.zeros(2, dtype=torch.uint8),
        succeeded_flags=torch.zeros(2, dtype=torch.bool),
    )
    satellites = [
        SimpleNamespace(sensor=SimpleNamespace(power=10.0)),
        SimpleNamespace(sensor=SimpleNamespace(power=20.0)),
    ]
    environment = SimpleNamespace(
        timer=timer,
        get_constellation=lambda: SimpleNamespace(sort=lambda: satellites),
    )
    controller = _FakeController()
    controller.environment = environment
    controller.task_manager = task_manager
    controller.memo = {}
    callback = LocalWindowCallback(
        controller=controller,
        decision_time=1,
        horizon=2,
        target_satellite_index=0,
    )
    callback.before_run()

    frames = [
        (0, [-1, 2], [0, 0], [[False, False], [True, False]]),
        (1, [3, 2], [0, 0], [[False, False], [True, False]]),
        (2, [-1, 2], [0, 1], [[False, True], [True, False]]),
        (3, [-1, 2], [0, 0], [[False, False], [True, False]]),
    ]
    for time, assignment, progress, is_visible in frames:
        timer.time = time
        controller.memo['assignment'] = assignment
        controller.memo['is_visible'] = torch.tensor(is_visible)
        task_manager.progress = torch.tensor(progress, dtype=torch.uint8)
        callback.after_step()
    callback.after_run()

    summary = callback.summary
    assert summary['horizon'] == 2
    assert summary['target_satellite_direct_visible_seconds'] == 1
    assert summary['target_satellite_one_second_runs'] == 1
    assert summary['working_satellite_seconds'] == 3


def test_local_window_callback_reuses_one_trace_for_multiple_horizons(
) -> None:
    timer = SimpleNamespace(time=0)
    taskset = TaskSet([
        Task(
            id_=2,
            release_time=0,
            due_time=100,
            duration=2,
            coordinate=Coordinate(0.0, 0.0),
            sensor_type=SensorType.VISIBLE,
        ),
        Task(
            id_=3,
            release_time=0,
            due_time=100,
            duration=2,
            coordinate=Coordinate(10.0, 10.0),
            sensor_type=SensorType.VISIBLE,
        ),
    ])
    task_manager = SimpleNamespace(
        taskset=taskset,
        progress=torch.zeros(2, dtype=torch.uint8),
        succeeded_flags=torch.zeros(2, dtype=torch.bool),
    )
    satellites = [
        SimpleNamespace(sensor=SimpleNamespace(power=10.0)),
        SimpleNamespace(sensor=SimpleNamespace(power=20.0)),
    ]
    environment = SimpleNamespace(
        timer=timer,
        get_constellation=lambda: SimpleNamespace(sort=lambda: satellites),
    )
    controller = _FakeController()
    controller.environment = environment
    controller.task_manager = task_manager
    controller.memo = {}
    callback = LocalWindowCallback(
        controller=controller,
        decision_time=1,
        horizons=(1, 2),
        target_satellite_index=0,
    )
    callback.before_run()

    frames = [
        (0, [-1, 2], [0, 0], [False, False]),
        (1, [3, 2], [0, 0], [False, False]),
        (2, [-1, 2], [0, 1], [False, False]),
        (3, [-1, 2], [0, 0], [False, False]),
    ]
    for time, assignment, progress, succeeded in frames:
        timer.time = time
        controller.memo['assignment'] = assignment
        controller.memo['is_visible'] = torch.zeros((2, 2), dtype=torch.bool)
        task_manager.progress = torch.tensor(progress, dtype=torch.uint8)
        task_manager.succeeded_flags = torch.tensor(succeeded)
        callback.after_step()
    callback.after_run()

    assert callback.summaries[1]['horizon'] == 1
    assert callback.summaries[1]['working_satellite_seconds'] == 2
    assert callback.summaries[1]['prefix_metrics']['quality'] == pytest.approx(
        0.05,
    )
    assert callback.summaries[1]['prefix_metrics']['prefix_cost'] is not None
    assert callback.summaries[2]['horizon'] == 2
    assert callback.summaries[2]['working_satellite_seconds'] == 3
