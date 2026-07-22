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


def test_replan_order_uses_wait_then_satellite_id_as_tie_breaks() -> None:
    state = _state()._replace(
        forced_interrupt_mask=torch.zeros(1, 3, dtype=torch.bool),
        compatible_deadline_slack=torch.full((1, 3), 5.),
        seconds_since_replan=torch.tensor([[2., 12., 12.]]),
        replan_mask=torch.tensor([[True, True, True]]),
    )

    assert build_replan_order(state)[0].tolist() == [1, 2, 0]


def test_replan_order_excludes_satellites_that_do_not_need_planning() -> None:
    state = _state()._replace(
        replan_mask=torch.tensor([[True, False, True]]),
    )

    assert build_replan_order(state)[0].tolist() == [2, 0]


def test_commitment_mask_reserves_one_second_for_nearly_complete_task() -> None:
    mask = build_commitment_mask(
        remaining_required_seconds=torch.tensor([[1., 4.]]),
        task_selected=torch.tensor([[True, True]]),
    )

    assert COMMITMENT_SECONDS == (1, 5, 15, 30, 60)
    assert mask.tolist() == [
        [[True, True, True, True, True],
         [False, True, True, True, True]],
    ]


def test_commitment_mask_has_no_categories_for_idle() -> None:
    mask = build_commitment_mask(
        remaining_required_seconds=torch.tensor([[1., 10.]]),
        task_selected=torch.tensor([[False, True]]),
    )

    assert mask[0, 0].tolist() == [False] * 5
    assert mask[0, 1].tolist() == [False, True, True, True, True]


def test_event_state_accepts_consistent_shapes_and_values() -> None:
    _state().validate()


def test_event_state_rejects_owner_count_above_three() -> None:
    state = _state()._replace(
        task_owner_count=torch.tensor([[0, 1, 4]]),
    )

    with pytest.raises(ValueError, match='owner'):
        state.validate()


def test_event_state_rejects_locked_owner_count_above_owner_count() -> None:
    state = _state()._replace(
        task_locked_owner_count=torch.tensor([[0, 2, 1]]),
    )

    with pytest.raises(ValueError, match='locked owner'):
        state.validate()


def test_event_state_rejects_non_finite_time_values() -> None:
    state = _state()._replace(
        delta_t=torch.tensor([[5., float('nan'), 1.]]),
    )

    with pytest.raises(ValueError, match='finite'):
        state.validate()


def test_event_state_rejects_mismatched_satellite_shape() -> None:
    state = _state()._replace(run_lengths=torch.ones(1, 2))

    with pytest.raises(ValueError, match='satellite shape'):
        state.validate()


def test_commitment_mask_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match='same shape'):
        build_commitment_mask(
            remaining_required_seconds=torch.ones(1, 2),
            task_selected=torch.ones(1, 3, dtype=torch.bool),
        )
