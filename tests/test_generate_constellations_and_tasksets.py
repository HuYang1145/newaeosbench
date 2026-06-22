from constellation.data import Coordinate, SensorType, Task, TaskSet
from tools import generate_constellations_and_tasksets as generator
from tools.generate_constellations_and_tasksets import (
    has_contiguous_observation_window,
    renumber_taskset,
    sample_observable_taskset,
)


def make_task(
    *,
    id_: int = 7,
    release_time: int = 2,
    due_time: int = 8,
    duration: int = 3,
) -> Task:
    return Task(
        id_,
        release_time,
        due_time,
        duration,
        Coordinate(10.0, 20.0),
        SensorType.VISIBLE,
    )


def test_task_requires_contiguous_visibility_inside_time_window() -> None:
    task = make_task()

    assert has_contiguous_observation_window(
        task,
        [False, False, True, True, True, False, False, False, False],
    )
    assert not has_contiguous_observation_window(
        task,
        [False, False, True, True, False, True, False, False, False],
    )
    assert not has_contiguous_observation_window(
        task,
        [True, True, True, False, False, False, False, False, False],
    )


def test_renumber_taskset_preserves_fields_after_filtering() -> None:
    first = make_task(id_=42, release_time=1, due_time=9, duration=2)
    second = make_task(id_=99, release_time=3, due_time=11, duration=4)

    taskset = renumber_taskset(TaskSet([first, second]))

    assert [task.id_ for task in taskset] == [0, 1]
    assert [task.release_time for task in taskset] == [1, 3]
    assert [task.due_time for task in taskset] == [9, 11]
    assert [task.duration for task in taskset] == [2, 4]
    assert [task.coordinate for task in taskset] == [
        first.coordinate,
        second.coordinate,
    ]


def test_sample_observable_taskset_filters_and_refills(monkeypatch) -> None:
    def fake_scan(constellation, candidates, *, max_time_step):
        return [task.id_ % 2 == 0 for task in candidates]

    monkeypatch.setattr(
        generator,
        'scan_observable_task_flags',
        fake_scan,
    )

    taskset = sample_observable_taskset(
        object(),
        3,
        oversample_factor=2,
        max_rounds=1,
        max_time_step=10,
    )

    assert len(taskset) == 3
    assert [task.id_ for task in taskset] == [0, 1, 2]
