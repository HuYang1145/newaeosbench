import pytest

from constellation.new_transformers.event_action import (
    ALLOWED_EVENT_COMMITMENTS,
    EventAssignmentState,
    EventDecision,
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


def test_event_decision_requires_one_second_idle() -> None:
    with pytest.raises(ValueError, match='idle action'):
        EventDecision(task_id=-1, commitment_seconds=5)


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
