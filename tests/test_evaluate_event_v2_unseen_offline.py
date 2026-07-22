import math

import pytest

from tools.evaluate_event_v2_unseen_offline import (
    aggregate_weighted_losses,
    decide_acceptance,
)


LOSS_WEIGHTS = {
    'task_distillation': 1.0,
    'termination': 2.0,
    'commitment': 3.0,
    'value': 4.0,
}


def _scene(
    *,
    random_losses: tuple[float, float, float, float],
    trained_losses: tuple[float, float, float, float],
    supports: tuple[int, int, int, int],
) -> dict:
    names = tuple(LOSS_WEIGHTS)
    return {
        'random': dict(zip(names, random_losses)),
        'trained': dict(zip(names, trained_losses)),
        'supports': dict(zip(names, supports)),
    }


def _passing_metrics() -> dict:
    return aggregate_weighted_losses(
        [
            _scene(
                random_losses=(2.0, 2.0, 2.0, 2.0),
                trained_losses=(1.0, 1.0, 1.0, 1.0),
                supports=(1, 1, 1, 1),
            ),
        ],
        LOSS_WEIGHTS,
    )


def test_aggregate_weights_each_component_by_its_own_support() -> None:
    metrics = aggregate_weighted_losses(
        [
            _scene(
                random_losses=(1.0, 10.0, 100.0, 1000.0),
                trained_losses=(0.5, 8.0, 80.0, 800.0),
                supports=(1, 1, 1, 1),
            ),
            _scene(
                random_losses=(3.0, 30.0, 300.0, 3000.0),
                trained_losses=(1.5, 24.0, 240.0, 2400.0),
                supports=(3, 2, 4, 5),
            ),
        ],
        LOSS_WEIGHTS,
    )

    assert metrics['supports'] == {
        'task_distillation': 4,
        'termination': 3,
        'commitment': 5,
        'value': 6,
    }
    assert metrics['random']['task_distillation'] == pytest.approx(2.5)
    assert metrics['random']['termination'] == pytest.approx(70 / 3)
    assert metrics['random']['commitment'] == pytest.approx(260.0)
    assert metrics['random']['value'] == pytest.approx(8000 / 3)
    expected_total = sum(
        metrics['random'][name] * LOSS_WEIGHTS[name]
        for name in LOSS_WEIGHTS
    )
    assert metrics['random']['total'] == pytest.approx(expected_total)
    assert metrics['delta']['total'] == pytest.approx(
        metrics['trained']['total'] - metrics['random']['total']
    )
    assert metrics['relative_reduction']['task_distillation'] == (
        pytest.approx(0.5)
    )


def test_aggregate_rejects_invalid_support() -> None:
    scene = _scene(
        random_losses=(1.0, 1.0, 1.0, 1.0),
        trained_losses=(0.5, 0.5, 0.5, 0.5),
        supports=(-1, 1, 1, 1),
    )

    with pytest.raises(ValueError, match='support'):
        aggregate_weighted_losses([scene], LOSS_WEIGHTS)


def test_acceptance_requires_all_losses_to_strictly_decrease() -> None:
    metrics = _passing_metrics()
    accepted = decide_acceptance(
        metrics=metrics,
        scene_count=64,
        expected_scene_count=64,
        audit_passed=True,
    )
    assert accepted == {'accepted': True, 'reasons': []}

    metrics['trained']['commitment'] = metrics['random']['commitment']
    metrics['delta']['commitment'] = 0.0
    rejected = decide_acceptance(
        metrics=metrics,
        scene_count=64,
        expected_scene_count=64,
        audit_passed=True,
    )
    assert rejected['accepted'] is False
    assert any('commitment' in reason for reason in rejected['reasons'])


@pytest.mark.parametrize(
    ('scene_count', 'audit_passed', 'mutation', 'expected_reason'),
    [
        (63, True, None, 'scene count'),
        (64, False, None, 'audit'),
        (64, True, ('supports', 'termination', 0), 'support'),
        (64, True, ('random', 'value', math.inf), 'finite'),
    ],
)
def test_acceptance_rejects_incomplete_or_invalid_evidence(
    scene_count: int,
    audit_passed: bool,
    mutation: tuple[str, str, float] | None,
    expected_reason: str,
) -> None:
    metrics = _passing_metrics()
    if mutation is not None:
        section, name, value = mutation
        metrics[section][name] = value

    result = decide_acceptance(
        metrics=metrics,
        scene_count=scene_count,
        expected_scene_count=64,
        audit_passed=audit_passed,
    )

    assert result['accepted'] is False
    assert any(expected_reason in reason for reason in result['reasons'])
