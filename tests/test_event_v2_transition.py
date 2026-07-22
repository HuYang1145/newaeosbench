import copy

import pytest
import torch

from constellation.new_transformers.event_v2.state import EventStateTensors
from constellation.new_transformers.event_v2.transition import (
    TRANSITION_SCHEMA_VERSION,
    ActionTrace,
    EventTransition,
    JointEventAction,
    transition_schema_definition,
    transition_schema_fingerprint,
)


def _state() -> EventStateTensors:
    return EventStateTensors(
        previous_task_indices=torch.tensor([[0, -1]]),
        current_task_indices=torch.tensor([[0, -1]]),
        minimum_commitment_remaining=torch.tensor([[0., 0.]]),
        run_lengths=torch.tensor([[5., 1.]]),
        seconds_since_replan=torch.tensor([[5., 10.]]),
        switch_count_30=torch.tensor([[0., 1.]]),
        switch_count_60=torch.tensor([[0., 2.]]),
        termination_reason=torch.tensor([[0, 1]]),
        event_type=torch.tensor([[0, 1]]),
        delta_t=torch.tensor([[5., 5.]]),
        replan_mask=torch.tensor([[True, True]]),
        forced_interrupt_mask=torch.tensor([[False, True]]),
        can_terminate_mask=torch.tensor([[True, False]]),
        compatible_deadline_slack=torch.tensor([[20., 5.]]),
        task_remaining_required_seconds=torch.tensor([[4., 30.]]),
        task_owner_count=torch.tensor([[1, 0]]),
        task_locked_owner_count=torch.tensor([[0, 0]]),
    )


def _transition() -> EventTransition:
    return EventTransition(
        state=_state(),
        joint_action=JointEventAction(
            terminate=torch.tensor([[True, False]]),
            task_indices=torch.tensor([[1, -1]]),
            commitment_indices=torch.tensor([[1, -1]]),
        ),
        behavior_log_prob=torch.tensor([-1.25]),
        value=torch.tensor([0.4]),
        reward=torch.tensor([0.1]),
        delta_t=torch.tensor([5.]),
        next_state=_state()._replace(delta_t=torch.tensor([[5., 5.]])),
        done=torch.tensor([False]),
        trace=ActionTrace(
            action_order=torch.tensor([[1, 0]]),
            termination_mask=torch.tensor([[True, False]]),
            task_masks=torch.tensor([[
                [True, True, True],
                [True, True, False],
            ]]),
            commitment_masks=torch.tensor([[
                [False, True, True, True, True],
                [False, False, False, False, False],
            ]]),
            owner_state=torch.tensor([[
                [1, 0],
                [1, 1],
            ]]),
        ),
        policy_version=3,
    )


def test_event_transition_accepts_consistent_schema() -> None:
    _transition().validate()


def test_event_transition_round_trips_through_torch_save(tmp_path) -> None:
    path = tmp_path / 'transition.pth'
    expected = _transition()

    torch.save(expected, path)
    actual = torch.load(path, weights_only=False)

    assert isinstance(actual, EventTransition)
    assert actual.policy_version == expected.policy_version
    torch.testing.assert_close(actual.behavior_log_prob, expected.behavior_log_prob)
    torch.testing.assert_close(
        actual.trace.owner_state,
        expected.trace.owner_state,
    )


def test_schema_fingerprint_is_stable_sha256() -> None:
    first = transition_schema_fingerprint()
    second = transition_schema_fingerprint()

    assert TRANSITION_SCHEMA_VERSION == 1
    assert first == second
    assert len(first) == 64
    assert set(first) <= set('0123456789abcdef')


def test_schema_fingerprint_changes_with_field_order_or_dtype() -> None:
    schema = transition_schema_definition()
    reordered = copy.deepcopy(schema)
    reordered['transition_fields'][0:2] = reversed(
        reordered['transition_fields'][0:2]
    )
    retyped = copy.deepcopy(schema)
    retyped['joint_action_fields'][0]['dtype'] = 'int64'

    assert transition_schema_fingerprint(reordered) != (
        transition_schema_fingerprint(schema)
    )
    assert transition_schema_fingerprint(retyped) != (
        transition_schema_fingerprint(schema)
    )


def test_event_transition_rejects_trace_task_mask_shape() -> None:
    transition = _transition()
    bad_trace = transition.trace._replace(
        task_masks=torch.ones(1, 2, 2, dtype=torch.bool),
    )

    with pytest.raises(ValueError, match='task_masks'):
        EventTransition(
            **{
                **transition.__dict__,
                'trace': bad_trace,
            },
        ).validate()


def test_event_transition_rejects_non_finite_behavior_log_prob() -> None:
    transition = _transition()

    with pytest.raises(ValueError, match='behavior_log_prob'):
        EventTransition(
            **{
                **transition.__dict__,
                'behavior_log_prob': torch.tensor([float('nan')]),
            },
        ).validate()


def test_event_transition_rejects_negative_policy_version() -> None:
    transition = _transition()

    with pytest.raises(ValueError, match='policy_version'):
        EventTransition(
            **{
                **transition.__dict__,
                'policy_version': -1,
            },
        ).validate()


def test_event_transition_rejects_duplicate_action_order() -> None:
    transition = _transition()
    bad_trace = transition.trace._replace(
        action_order=torch.tensor([[1, 1]]),
    )

    with pytest.raises(ValueError, match='action_order'):
        EventTransition(
            **{
                **transition.__dict__,
                'trace': bad_trace,
            },
        ).validate()
