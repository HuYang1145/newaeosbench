import pytest
import torch

from constellation.new_transformers.event_v2.runtime_state import (
    EventRuntimeState,
    EventType,
    RuntimeSnapshot,
    TerminationReason,
)
from constellation.new_transformers.event_v2.transition import JointEventAction


def _snapshot(
    time_step: int,
    *,
    ongoing_global_task_ids: tuple[int, ...] = (12,),
    released_global_task_ids: tuple[int, ...] = (),
    closed_global_task_ids: tuple[int, ...] = (),
    assignment_valid: torch.Tensor | None = None,
) -> RuntimeSnapshot:
    num_tasks = len(ongoing_global_task_ids)
    if assignment_valid is None:
        assignment_valid = torch.ones(1, dtype=torch.bool)
    return RuntimeSnapshot(
        time_step=time_step,
        ongoing_global_task_ids=ongoing_global_task_ids,
        task_progress=torch.zeros(num_tasks),
        task_required_duration=torch.full((num_tasks,), 30.),
        task_deadline_slack=torch.arange(
            num_tasks,
            dtype=torch.float32,
        ) + 10,
        task_compatibility=torch.ones(1, num_tasks, dtype=torch.bool),
        assignment_valid=assignment_valid,
        released_global_task_ids=released_global_task_ids,
        closed_global_task_ids=closed_global_task_ids,
    )


def _action(
    *,
    task_index: int,
    commitment_index: int,
    terminate: bool = False,
) -> JointEventAction:
    return JointEventAction(
        terminate=torch.tensor([[terminate]]),
        task_indices=torch.tensor([[task_index]]),
        commitment_indices=torch.tensor([[commitment_index]]),
    )


def _committed_machine(seconds_index: int) -> EventRuntimeState:
    machine = EventRuntimeState(num_satellites=1, safety_review_seconds=5)
    machine.initial_event(_snapshot(0))
    machine.apply_joint_action(
        _action(task_index=0, commitment_index=seconds_index),
        (12,),
    )
    return machine


def test_locked_assignment_survives_review_until_commitment_expires() -> None:
    machine = _committed_machine(seconds_index=2)  # 15 seconds

    events = [machine.advance_one_second(_snapshot(t)) for t in range(1, 16)]

    assert events[4].safety_review
    assert not events[4].requires_policy
    assert not events[4].state.replan_mask[0, 0]
    assert events[-1].requires_policy
    assert events[-1].state.event_type.item() == EventType.COMMITMENT_EXPIRED
    assert events[-1].state.can_terminate_mask[0, 0]
    assert events[-1].state.minimum_commitment_remaining.item() == 0


def test_external_close_forces_replan_without_policy_termination() -> None:
    machine = _committed_machine(seconds_index=3)  # 30 seconds

    event = machine.advance_one_second(_snapshot(
        1,
        ongoing_global_task_ids=(),
        closed_global_task_ids=(12,),
        assignment_valid=torch.tensor([False]),
    ))

    assert event.requires_policy
    assert event.state.forced_interrupt_mask[0, 0]
    assert event.state.replan_mask[0, 0]
    assert not event.state.can_terminate_mask[0, 0]
    assert event.state.termination_reason.item() == TerminationReason.FORCED


def test_global_assignment_maps_to_current_relative_task_index() -> None:
    machine = _committed_machine(seconds_index=1)  # 5 seconds

    event = machine.advance_one_second(_snapshot(
        1,
        ongoing_global_task_ids=(7, 12, 19),
    ))

    assert event.state.current_task_indices.tolist() == [[1]]
    assert event.state.previous_task_indices.tolist() == [[1]]


def test_keep_action_preserves_assignment_and_schedules_next_review() -> None:
    machine = _committed_machine(seconds_index=1)
    event = None
    for time_step in range(1, 6):
        event = machine.advance_one_second(_snapshot(time_step))
    assert event is not None and event.requires_policy
    assert event.state.can_terminate_mask[0, 0]

    machine.apply_joint_action(
        _action(task_index=-1, commitment_index=-1, terminate=False),
        (12,),
    )
    later = None
    for time_step in range(6, 11):
        later = machine.advance_one_second(_snapshot(time_step))

    assert machine.current_global_task_ids == (12,)
    assert later is not None and later.requires_policy
    assert later.state.event_type.item() == EventType.SAFETY_REVIEW
    assert later.state.can_terminate_mask[0, 0]


def test_owner_counts_include_locked_assignments() -> None:
    machine = EventRuntimeState(num_satellites=2, safety_review_seconds=5)
    initial = _snapshot(0)._replace(
        task_compatibility=torch.ones(2, 1, dtype=torch.bool),
        assignment_valid=torch.ones(2, dtype=torch.bool),
    )
    machine.initial_event(initial)
    machine.apply_joint_action(
        JointEventAction(
            terminate=torch.tensor([[False, False]]),
            task_indices=torch.tensor([[0, 0]]),
            commitment_indices=torch.tensor([[1, 2]]),
        ),
        (12,),
    )

    event = machine.advance_one_second(_snapshot(1)._replace(
        task_compatibility=torch.ones(2, 1, dtype=torch.bool),
        assignment_valid=torch.ones(2, dtype=torch.bool),
    ))

    assert event.state.task_owner_count.tolist() == [[2]]
    assert event.state.task_locked_owner_count.tolist() == [[2]]


def test_new_task_release_wakes_idle_satellite_immediately() -> None:
    machine = EventRuntimeState(num_satellites=1, safety_review_seconds=5)
    machine.initial_event(_snapshot(0, ongoing_global_task_ids=()))
    machine.apply_joint_action(
        _action(task_index=-1, commitment_index=-1),
        (),
    )

    event = machine.advance_one_second(_snapshot(
        1,
        ongoing_global_task_ids=(12,),
        released_global_task_ids=(12,),
    ))

    assert event.requires_policy
    assert event.state.replan_mask[0, 0]
    assert event.state.event_type.item() == EventType.EXTERNAL


def test_runtime_rejects_nonconsecutive_physical_time() -> None:
    machine = _committed_machine(seconds_index=1)

    with pytest.raises(ValueError, match='strictly advance by one second'):
        machine.advance_one_second(_snapshot(2))


def test_runtime_state_round_trip_preserves_next_event() -> None:
    machine = _committed_machine(seconds_index=2)
    for time_step in range(1, 4):
        machine.advance_one_second(_snapshot(time_step))

    restored = EventRuntimeState.from_state_dict(machine.state_dict())
    expected = machine.advance_one_second(_snapshot(4))
    actual = restored.advance_one_second(_snapshot(4))

    assert restored.current_global_task_ids == machine.current_global_task_ids
    assert actual.requires_policy == expected.requires_policy
    assert actual.safety_review == expected.safety_review
    for field in actual.state._fields:
        torch.testing.assert_close(
            getattr(actual.state, field),
            getattr(expected.state, field),
        )
