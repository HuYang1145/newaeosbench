import pytest

from constellation.new_transformers.event_action import EventDecision
from constellation.new_transformers.event_policy import EventActorRuntime


def test_runtime_skips_planner_before_event() -> None:
    runtime = EventActorRuntime(num_satellites=1)
    planner_calls = 0

    def planner(active, previous):
        nonlocal planner_calls
        planner_calls += 1
        assert active.tolist() == [False]
        return [EventDecision(7, 5)]

    assert runtime.update(
        time=0,
        ongoing_task_ids={7},
        planner=planner,
    ) == [7]
    assert runtime.update(
        time=1,
        ongoing_task_ids={7},
        planner=planner,
    ) == [7]

    assert planner_calls == 1
    assert runtime.replan_count == 1
    assert runtime.state.remaining_seconds.tolist() == [4]


def test_runtime_replans_at_commitment_expiry() -> None:
    runtime = EventActorRuntime(num_satellites=1)
    decisions = iter([
        [EventDecision(7, 5)],
        [EventDecision(8, 15)],
    ])

    def planner(active, previous):
        del active, previous
        return next(decisions)

    runtime.update(time=0, ongoing_task_ids={7, 8}, planner=planner)
    assert runtime.update(
        time=5,
        ongoing_task_ids={7, 8},
        planner=planner,
    ) == [8]
    assert runtime.replan_count == 2


def test_runtime_replans_only_satellite_with_invalidated_task() -> None:
    runtime = EventActorRuntime(num_satellites=2)

    def initial(active, previous):
        del active, previous
        return [EventDecision(7, 15), EventDecision(8, 15)]

    runtime.update(time=0, ongoing_task_ids={7, 8}, planner=initial)
    planner_active_masks: list[list[bool]] = []

    def replacement(active, previous):
        del previous
        planner_active_masks.append(active.tolist())
        return [EventDecision(9, 5), EventDecision(10, 5)]

    assignment = runtime.update(
        time=1,
        ongoing_task_ids={8, 9, 10},
        planner=replacement,
    )

    assert planner_active_masks == [[False, True]]
    assert assignment == [9, 8]
    assert runtime.state.remaining_seconds.tolist() == [5, 14]
    assert runtime.replan_count == 3


def test_runtime_rejects_incomplete_planner_output() -> None:
    runtime = EventActorRuntime(num_satellites=2)

    def planner(active, previous):
        del active, previous
        return [EventDecision(7, 5)]

    with pytest.raises(ValueError, match='one decision per satellite'):
        runtime.update(time=0, ongoing_task_ids={7}, planner=planner)
