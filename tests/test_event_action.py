import pytest
import torch

from constellation.new_transformers.event_action import (
    ALLOWED_EVENT_COMMITMENTS,
    EventAssignmentState,
    EventDecision,
    select_learned_event_commitments,
)


def test_event_state_counts_down_and_replans_at_expiry() -> None:
    state = EventAssignmentState.empty(num_satellites=2)
    state.start([
        EventDecision(task_id=3, commitment_seconds=5),
        EventDecision(task_id=-1, commitment_seconds=1),
    ], start_time=10)

    assert state.assignment() == [3, -1]
    assert state.advance(time=11, ongoing_task_ids={3}) == [False, True]
    assert state.remaining_seconds.tolist() == [4, 0]
    assert state.interruption_reasons == [None, 'expired']

    assert state.advance(time=15, ongoing_task_ids={3}) == [True, True]
    assert state.remaining_seconds.tolist() == [0, 0]
    assert state.interruption_reasons == ['expired', 'expired']


def test_event_state_interrupts_unavailable_task_before_expiry() -> None:
    state = EventAssignmentState.empty(num_satellites=1)
    state.start([
        EventDecision(task_id=3, commitment_seconds=60),
    ], start_time=10)

    assert state.advance(time=11, ongoing_task_ids=set()) == [True]
    assert state.remaining_seconds.tolist() == [0]
    assert state.interruption_reasons == ['task_unavailable']


def test_event_state_replaces_only_requested_satellite() -> None:
    state = EventAssignmentState.empty(num_satellites=2)
    state.start([
        EventDecision(task_id=3, commitment_seconds=5),
        EventDecision(task_id=4, commitment_seconds=15),
    ], start_time=10)
    assert state.advance(time=15, ongoing_task_ids={3, 4}) == [True, False]

    state.replace(
        0,
        EventDecision(task_id=5, commitment_seconds=30),
        start_time=15,
    )

    assert state.assignment() == [5, 4]
    assert state.remaining_seconds.tolist() == [30, 10]
    assert state.interruption_reasons == [None, None]


@pytest.mark.parametrize('duration', [0, 2, 6, 31, 61])
def test_event_decision_rejects_unsupported_duration(duration: int) -> None:
    assert duration not in ALLOWED_EVENT_COMMITMENTS
    with pytest.raises(ValueError, match='commitment_seconds'):
        EventDecision(task_id=3, commitment_seconds=duration)


def test_event_decision_allows_bounded_idle_commitment() -> None:
    decision = EventDecision(task_id=-1, commitment_seconds=5)

    assert decision.task_id == -1
    assert decision.commitment_seconds == 5


def test_event_state_rejects_time_regression() -> None:
    state = EventAssignmentState.empty(num_satellites=1)
    state.start([
        EventDecision(task_id=3, commitment_seconds=15),
    ], start_time=10)
    state.advance(time=12, ongoing_task_ids={3})

    with pytest.raises(ValueError, match='monotonic'):
        state.advance(time=11, ongoing_task_ids={3})


def test_event_state_requires_positive_satellite_count() -> None:
    with pytest.raises(ValueError, match='num_satellites'):
        EventAssignmentState.empty(num_satellites=0)


def test_learned_commitment_uses_selected_edge_and_continue_gate() -> None:
    continue_logits = torch.tensor([
        [-3.0, 3.0],
        [-3.0, 3.0],
        [3.0, 3.0],
    ])
    duration_logits = torch.zeros(3, 2, 5)
    duration_logits[0, 1, 3] = 5.0
    duration_logits[1, 0, 4] = 5.0

    selected = select_learned_event_commitments(
        relative_task_ids=torch.tensor([1, 0, -1]),
        continue_logits=continue_logits,
        duration_logits=duration_logits,
        continue_threshold=0.5,
    )

    assert selected.task_selected.tolist() == [True, True, False]
    assert selected.duration_proposals.tolist() == [30, 60, 1]
    assert selected.commitment_seconds.tolist() == [30, 1, 1]
    assert selected.continue_probabilities[:2].tolist() == pytest.approx([
        torch.sigmoid(torch.tensor(3.0)).item(),
        torch.sigmoid(torch.tensor(-3.0)).item(),
    ])


def test_learned_commitment_rejects_bad_shapes_and_threshold() -> None:
    with pytest.raises(ValueError, match='threshold'):
        select_learned_event_commitments(
            relative_task_ids=torch.tensor([0]),
            continue_logits=torch.zeros(1, 1),
            duration_logits=torch.zeros(1, 1, 5),
            continue_threshold=1.0,
        )
    with pytest.raises(ValueError, match='duration'):
        select_learned_event_commitments(
            relative_task_ids=torch.tensor([0]),
            continue_logits=torch.zeros(1, 1),
            duration_logits=torch.zeros(1, 1, 4),
            continue_threshold=0.5,
        )
