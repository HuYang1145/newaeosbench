import pytest
import torch

from constellation.new_transformers.event_candidate import (
    audit_preference_pair,
    build_event_candidate_specs,
    find_event_decisions,
    summarize_preference_audits,
)


def _branch(
    *,
    task_id: int,
    duration: int,
    cost180: float | None,
    cost300: float | None,
    quality180: tuple[float, float, float] = (0.5, 0.5, 0.5),
    quality300: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> dict:
    def horizon(
        cost: float | None,
        quality: tuple[float, float, float],
    ) -> dict:
        return {
            'prefix_metrics': {
                'prefix_cost': cost,
                'cr': quality[0],
                'pcr': quality[1],
                'wcr': quality[2],
            },
        }

    return {
        'applied_task_id': task_id,
        'requested_commitment_seconds': duration,
        'horizons': {
            '180': horizon(cost180, quality180),
            '300': horizon(cost300, quality300),
        },
    }


def test_build_event_candidate_specs_limits_idle_to_one_second() -> None:
    specs = build_event_candidate_specs(
        stay_task_id=-1,
        switch_task_id=8,
    )

    assert [
        (item.name, item.task_id, item.commitment_seconds, item.action_kind)
        for item in specs
    ] == [
        ('stay_task-1_d1', -1, 1, 'stay'),
        ('switch_task8_d1', 8, 1, 'switch'),
        ('switch_task8_d5', 8, 5, 'switch'),
        ('switch_task8_d15', 8, 15, 'switch'),
        ('switch_task8_d30', 8, 30, 'switch'),
        ('switch_task8_d60', 8, 60, 'switch'),
    ]


def test_build_event_candidate_specs_rejects_duplicate_actions() -> None:
    with pytest.raises(ValueError, match='must differ'):
        build_event_candidate_specs(stay_task_id=8, switch_task_id=8)


def test_stable_preference_requires_180_300_agreement() -> None:
    branches = {
        'a': _branch(
            task_id=7,
            duration=15,
            cost180=3.0,
            cost300=2.8,
        ),
        'b': _branch(
            task_id=8,
            duration=30,
            cost180=3.2,
            cost300=3.1,
        ),
    }

    audit = audit_preference_pair(
        'a',
        'b',
        branches,
        min_margin=0.01,
    )

    assert audit.accepted
    assert audit.reason == 'accepted'
    assert audit.better_branch == 'a'
    assert audit.worse_branch == 'b'
    assert audit.margin_300 == pytest.approx(0.3)
    assert audit.direction_agrees is True


def test_stable_preference_censors_horizon_reversal() -> None:
    branches = {
        'a': _branch(
            task_id=7,
            duration=15,
            cost180=3.0,
            cost300=3.2,
        ),
        'b': _branch(
            task_id=8,
            duration=30,
            cost180=3.1,
            cost300=3.0,
        ),
    }

    audit = audit_preference_pair('a', 'b', branches)

    assert not audit.accepted
    assert audit.reason == 'horizon_reversal'
    assert audit.better_branch is None
    assert audit.direction_agrees is False


def test_stable_preference_rejects_small_300_second_margin() -> None:
    branches = {
        'a': _branch(
            task_id=7,
            duration=15,
            cost180=3.0,
            cost300=3.000,
        ),
        'b': _branch(
            task_id=8,
            duration=30,
            cost180=3.1,
            cost300=3.005,
        ),
    }

    audit = audit_preference_pair(
        'a',
        'b',
        branches,
        min_margin=0.01,
    )

    assert not audit.accepted
    assert audit.reason == 'small_margin'
    assert audit.direction_agrees is True


def test_stable_preference_applies_quality_protection() -> None:
    branches = {
        'low_power_only': _branch(
            task_id=-1,
            duration=1,
            cost180=3.0,
            cost300=2.8,
            quality180=(0.3, 0.3, 0.3),
            quality300=(0.3, 0.3, 0.3),
        ),
        'working': _branch(
            task_id=8,
            duration=30,
            cost180=3.2,
            cost300=3.1,
            quality180=(0.5, 0.5, 0.5),
            quality300=(0.5, 0.5, 0.5),
        ),
    }

    audit = audit_preference_pair(
        'low_power_only',
        'working',
        branches,
    )

    assert not audit.accepted
    assert audit.reason == 'quality_protection'


def test_find_event_decisions_spreads_transitions_across_time_bins() -> None:
    actions = torch.full((1000, 1), -1, dtype=torch.long)
    actions[10:20, 0] = 3
    actions[310:320, 0] = 4
    actions[610:620, 0] = 5

    decisions = find_event_decisions(
        actions,
        max_decisions=2,
        latest_decision_time=699,
        bin_seconds=300,
    )

    assert [item.decision_time for item in decisions] == [10, 610]
    assert [item.stay_task_id for item in decisions] == [-1, -1]
    assert [item.switch_task_id for item in decisions] == [3, 5]


def test_summarize_preference_audits_reports_gate_inputs() -> None:
    branches = {
        'a': _branch(
            task_id=7,
            duration=15,
            cost180=3.0,
            cost300=2.8,
        ),
        'b': _branch(
            task_id=8,
            duration=30,
            cost180=3.2,
            cost300=3.1,
        ),
        'c': _branch(
            task_id=8,
            duration=60,
            cost180=2.9,
            cost300=3.3,
        ),
    }
    accepted = audit_preference_pair('a', 'b', branches).to_dict()
    reversed_ = audit_preference_pair('a', 'c', branches).to_dict()

    summary = summarize_preference_audits([
        {
            'scene_id': 4,
            'branches': branches,
            'pair_audits': [accepted, reversed_],
        },
    ])

    assert summary['accepted_scene_count'] == 1
    assert summary['stable_pair_count'] == 1
    assert summary['comparable_pair_count'] == 2
    assert summary['horizon_agreement'] == pytest.approx(0.5)
    assert summary['winning_duration_counts'] == {'15': 1}
    assert summary['winning_duration_class_count'] == 1
    assert summary['max_winning_duration_fraction'] == pytest.approx(1.0)
