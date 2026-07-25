from __future__ import annotations

import pytest

from tools.compare_event_v2_full_val import compare_full_val


SCENE_IDS = tuple(range(64))


def _summary(
    stage: str,
    split: str,
    *,
    cr: float,
    pcr: float,
    wcr: float,
) -> dict:
    return {
        'label': f'{stage}_{split}',
        'checkpoint': f'/checkpoints/{stage}.pth',
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
            'TAT_s': 100.0,
            'PC_Wh': 2.0,
            'CS_paper': 3.0,
        },
    }


def test_full_val_requires_both_64_scene_splits_to_pass() -> None:
    result = compare_full_val(
        baseline_seen=_summary(
            'V2-2', 'val_seen',
            cr=0.40, pcr=0.50, wcr=0.30,
        ),
        candidate_seen=_summary(
            'V2-2-Large', 'val_seen',
            cr=0.41, pcr=0.51, wcr=0.31,
        ),
        baseline_unseen=_summary(
            'V2-2', 'val_unseen',
            cr=0.30, pcr=0.40, wcr=0.20,
        ),
        candidate_unseen=_summary(
            'V2-2-Large', 'val_unseen',
            cr=0.31, pcr=0.41, wcr=0.21,
        ),
        expected_scene_ids=SCENE_IDS,
    )

    assert result['passed'] is True
    assert result['splits']['val_seen']['delta']['Q'] == pytest.approx(
        0.01,
    )
    assert result['protocol']['scene_ids'] == list(SCENE_IDS)
    assert result['protocol']['candidate_stage'] == 'V2-2-Large'


def test_full_val_stops_test_when_any_metric_decreases() -> None:
    result = compare_full_val(
        baseline_seen=_summary(
            'V2-2', 'val_seen',
            cr=0.40, pcr=0.40, wcr=0.40,
        ),
        candidate_seen=_summary(
            'V2-2-Large', 'val_seen',
            cr=0.42, pcr=0.42, wcr=0.39,
        ),
        baseline_unseen=_summary(
            'V2-2', 'val_unseen',
            cr=0.30, pcr=0.30, wcr=0.30,
        ),
        candidate_unseen=_summary(
            'V2-2-Large', 'val_unseen',
            cr=0.31, pcr=0.31, wcr=0.31,
        ),
        expected_scene_ids=SCENE_IDS,
    )

    assert result['passed'] is False
    assert result['splits']['val_seen']['metrics_non_decreasing'] is False


def test_full_val_rejects_any_scene_set_other_than_zero_to_63() -> None:
    with pytest.raises(ValueError, match='0-63'):
        compare_full_val(
            baseline_seen=_summary('V2-2', 'val_seen', cr=.4, pcr=.4, wcr=.4),
            candidate_seen=_summary(
                'V2-2-Large', 'val_seen',
                cr=.41, pcr=.41, wcr=.41,
            ),
            baseline_unseen=_summary(
                'V2-2', 'val_unseen',
                cr=.3, pcr=.3, wcr=.3,
            ),
            candidate_unseen=_summary(
                'V2-2-Large', 'val_unseen',
                cr=.31, pcr=.31, wcr=.31,
            ),
            expected_scene_ids=tuple(range(1, 65)),
        )
