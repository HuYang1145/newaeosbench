import pytest

from tools.generate_event_candidate_branches_m3 import (
    build_group_preference_audits,
    resolve_switch_task_id,
    validate_common_decision_state,
)


def _branch(
    *,
    task_id: int,
    duration: int,
    cost180: float,
    cost300: float,
    signature: str = 'same-state',
    context: dict | None = None,
) -> dict:
    common_context = {
        'previous_assignment': [7],
        'ongoing_task_ids': [7, 8],
    } if context is None else context
    return {
        'decision_state_signature': signature,
        'decision_context': common_context,
        'applied_task_id': task_id,
        'requested_commitment_seconds': duration,
        'horizons': {
            '180': {
                'prefix_metrics': {
                    'prefix_cost': cost180,
                    'cr': 0.5,
                    'pcr': 0.5,
                    'wcr': 0.5,
                },
            },
            '300': {
                'prefix_metrics': {
                    'prefix_cost': cost300,
                    'cr': 0.5,
                    'pcr': 0.5,
                    'wcr': 0.5,
                },
            },
        },
    }


def test_resolve_switch_uses_highest_candidate_different_from_stay() -> None:
    task_id = resolve_switch_task_id(
        stay_task_id=7,
        actor_logits=[5.0, 9.0, 8.0],
        ongoing_task_ids=[7, 8],
    )

    assert task_id == 8


def test_resolve_switch_can_select_idle() -> None:
    task_id = resolve_switch_task_id(
        stay_task_id=7,
        actor_logits=[10.0, 9.0, 8.0],
        ongoing_task_ids=[7, 8],
    )

    assert task_id == -1


def test_resolve_switch_rejects_misaligned_logits() -> None:
    with pytest.raises(ValueError, match='do not match'):
        resolve_switch_task_id(
            stay_task_id=7,
            actor_logits=[1.0, 2.0],
            ongoing_task_ids=[7, 8],
        )


def test_common_decision_state_requires_identical_signatures() -> None:
    branches = {
        'a': _branch(task_id=7, duration=5, cost180=3.0, cost300=2.9),
        'b': _branch(
            task_id=8,
            duration=15,
            cost180=3.2,
            cost300=3.1,
            signature='different-state',
        ),
    }

    with pytest.raises(ValueError, match='same decision state'):
        validate_common_decision_state(branches)


def test_common_decision_state_requires_identical_contexts() -> None:
    branches = {
        'a': _branch(task_id=7, duration=5, cost180=3.0, cost300=2.9),
        'b': _branch(
            task_id=8,
            duration=15,
            cost180=3.2,
            cost300=3.1,
            context={
                'previous_assignment': [-1],
                'ongoing_task_ids': [7, 8],
            },
        ),
    }

    with pytest.raises(ValueError, match='same decision context'):
        validate_common_decision_state(branches)


def test_group_audits_keep_accepted_and_rejected_pairs() -> None:
    branches = {
        'stay_task7_d5': _branch(
            task_id=7,
            duration=5,
            cost180=3.0,
            cost300=2.8,
        ),
        'switch_task8_d15': _branch(
            task_id=8,
            duration=15,
            cost180=3.2,
            cost300=3.1,
        ),
        'switch_task8_d30': _branch(
            task_id=8,
            duration=30,
            cost180=2.9,
            cost300=3.3,
        ),
    }

    audits = build_group_preference_audits(branches, min_margin=0.01)

    assert len(audits) == 3
    reasons = {item['reason'] for item in audits}
    assert 'accepted' in reasons
    assert 'horizon_reversal' in reasons
    assert all('first_branch' in item for item in audits)
