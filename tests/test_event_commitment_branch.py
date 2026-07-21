from types import SimpleNamespace

from constellation.algorithms.base import BaseAlgorithm
from constellation.data import Action, Actions, Coordinate, Task, TaskSet
from constellation.data.constellations import SensorType
from constellation.new_transformers.local_action_branch import (
    ControlledCommitmentAlgorithm,
)
import torch


class _BaseAlgorithm(BaseAlgorithm):

    def prepare(self, environment, task_manager) -> None:
        self.prepared = (environment, task_manager)

    def step(self, taskset, constellation, earth_rotation):
        del earth_rotation
        self.last_task_ids = taskset.ids.clone()
        self.last_logits = torch.zeros(
            (1, len(constellation), len(taskset) + 1),
        )
        return Actions([Action(), Action()]), [-1, 5]


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


def _algorithm(timer, commitment_seconds: int = 5):
    full_taskset = _tasks()
    task_manager = SimpleNamespace(
        progress=torch.tensor([0, 0]),
        taskset=full_taskset,
    )
    algorithm = ControlledCommitmentAlgorithm(
        timer=timer,
        base_algorithm=_BaseAlgorithm(timer=timer),
        decision_time=5,
        satellite_index=0,
        forced_task_id=3,
        commitment_seconds=commitment_seconds,
    )
    algorithm.prepare(SimpleNamespace(), task_manager)
    return algorithm, full_taskset


def _step(
    algorithm: ControlledCommitmentAlgorithm,
    timer,
    time: int,
    taskset: TaskSet,
) -> list[int]:
    timer.time = time
    _, assignment = algorithm.step(
        taskset,
        _FakeConstellation(),
        torch.eye(3),
    )
    return assignment


def test_commitment_overrides_only_target_for_requested_seconds() -> None:
    timer = SimpleNamespace(time=0)
    algorithm, taskset = _algorithm(timer)

    assignments = [
        _step(algorithm, timer, time, taskset)
        for time in range(4, 12)
    ]

    assert [row[0] for row in assignments] == [
        -1, 3, 3, 3, 3, 3, -1, -1,
    ]
    assert all(row[1] == 5 for row in assignments)
    assert algorithm.actual_commitment_seconds == 5
    assert algorithm.interruption_reason == 'expired'
    assert algorithm.requested_commitment_seconds == 5
    assert algorithm.original_task_id == -1
    assert algorithm.applied_task_id == 3
    assert len(algorithm.decision_state_signature) == 64


def test_commitment_stops_when_task_leaves_ongoing_set() -> None:
    timer = SimpleNamespace(time=0)
    algorithm, full_taskset = _algorithm(timer)
    task_five_only = TaskSet([full_taskset[1]])

    assert _step(algorithm, timer, 5, full_taskset)[0] == 3
    assert _step(algorithm, timer, 6, full_taskset)[0] == 3
    assert _step(algorithm, timer, 7, task_five_only)[0] == -1

    assert algorithm.actual_commitment_seconds == 2
    assert algorithm.interruption_reason == 'task_unavailable'


def test_idle_commitment_is_allowed_for_one_second() -> None:
    timer = SimpleNamespace(time=0)
    taskset = _tasks()
    task_manager = SimpleNamespace(
        progress=torch.tensor([0, 0]),
        taskset=taskset,
    )
    algorithm = ControlledCommitmentAlgorithm(
        timer=timer,
        base_algorithm=_BaseAlgorithm(timer=timer),
        decision_time=5,
        satellite_index=0,
        forced_task_id=-1,
        commitment_seconds=1,
    )
    algorithm.prepare(SimpleNamespace(), task_manager)

    assert _step(algorithm, timer, 5, taskset)[0] == -1
    assert _step(algorithm, timer, 6, taskset)[0] == -1
    assert algorithm.actual_commitment_seconds == 1
    assert algorithm.interruption_reason == 'expired'
