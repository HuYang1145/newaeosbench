import pytest

from tools.compare_event_v2_val_gate import compare_val_gate


SCENE_IDS = tuple(range(8))


def _summary(
    label: str,
    stage: str,
    split: str,
    *,
    cr: float,
    pcr: float,
    wcr: float,
) -> dict:
    return {
        'label': label,
        'checkpoint': f'/checkpoints/{label}.pth',
        'stage': stage,
        'split': split,
        'scene_ids': list(SCENE_IDS),
        'max_time_step': 3600,
        'deterministic': True,
        'finite': True,
        'reward_reconstruction_max_error': 0.0,
        'aggregate': {
            'CR': cr,
            'PCR': pcr,
            'WCR': wcr,
            'Q': 0.6 * cr + 0.2 * pcr + 0.2 * wcr,
        },
    }


def test_val_gate_passes_only_when_both_splits_meet_registered_threshold() -> None:
    result = compare_val_gate(
        baseline_seen=_summary(
            'baseline_seen', 'V2-1', 'val_seen',
            cr=0.4, pcr=0.5, wcr=0.3,
        ),
        candidate_seen=_summary(
            'candidate_seen', 'V2-2', 'val_seen',
            cr=0.41, pcr=0.51, wcr=0.31,
        ),
        baseline_unseen=_summary(
            'baseline_unseen', 'V2-1', 'val_unseen',
            cr=0.3, pcr=0.4, wcr=0.2,
        ),
        candidate_unseen=_summary(
            'candidate_unseen', 'V2-2', 'val_unseen',
            cr=0.31, pcr=0.41, wcr=0.21,
        ),
        expected_scene_ids=SCENE_IDS,
        minimum_q_improvement=0.005,
    )

    assert result['passed'] is True
    assert result['splits']['val_seen']['delta']['Q'] == pytest.approx(0.01)
    assert result['splits']['val_unseen']['delta']['Q'] == pytest.approx(0.01)


def test_val_gate_rejects_metric_regression_even_when_q_improves() -> None:
    result = compare_val_gate(
        baseline_seen=_summary(
            'baseline_seen', 'V2-1', 'val_seen',
            cr=0.4, pcr=0.4, wcr=0.4,
        ),
        candidate_seen=_summary(
            'candidate_seen', 'V2-2', 'val_seen',
            cr=0.42, pcr=0.42, wcr=0.39,
        ),
        baseline_unseen=_summary(
            'baseline_unseen', 'V2-1', 'val_unseen',
            cr=0.3, pcr=0.3, wcr=0.3,
        ),
        candidate_unseen=_summary(
            'candidate_unseen', 'V2-2', 'val_unseen',
            cr=0.31, pcr=0.31, wcr=0.31,
        ),
        expected_scene_ids=SCENE_IDS,
        minimum_q_improvement=0.005,
    )

    assert result['passed'] is False
    assert result['splits']['val_seen']['metrics_non_decreasing'] is False


def test_val_gate_rejects_q_improvement_below_half_percentage_point() -> None:
    result = compare_val_gate(
        baseline_seen=_summary(
            'baseline_seen', 'V2-1', 'val_seen',
            cr=0.4, pcr=0.4, wcr=0.4,
        ),
        candidate_seen=_summary(
            'candidate_seen', 'V2-2', 'val_seen',
            cr=0.404, pcr=0.404, wcr=0.404,
        ),
        baseline_unseen=_summary(
            'baseline_unseen', 'V2-1', 'val_unseen',
            cr=0.3, pcr=0.3, wcr=0.3,
        ),
        candidate_unseen=_summary(
            'candidate_unseen', 'V2-2', 'val_unseen',
            cr=0.304, pcr=0.304, wcr=0.304,
        ),
        expected_scene_ids=SCENE_IDS,
        minimum_q_improvement=0.005,
    )

    assert result['passed'] is False
    assert result['splits']['val_seen']['q_threshold_met'] is False
