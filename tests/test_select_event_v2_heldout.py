import pytest

from tools.select_event_v2_heldout import select_heldout_summaries


SCENE_IDS = tuple(range(196, 204))


def _summary(
    label: str,
    stage: str,
    *,
    cr: float,
    pcr: float,
    wcr: float,
) -> dict:
    return {
        'label': label,
        'checkpoint': f'/checkpoints/{label}.pth',
        'stage': stage,
        'split': 'train',
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


def test_selector_chooses_best_v2_2_and_reports_baseline_delta() -> None:
    baseline = _summary(
        'v2_1',
        'V2-1',
        cr=0.4,
        pcr=0.5,
        wcr=0.3,
    )
    candidates = [
        _summary(
            f'v2_2_replica_{index}',
            'V2-2',
            cr=cr,
            pcr=pcr,
            wcr=wcr,
        )
        for index, (cr, pcr, wcr) in enumerate((
            (0.45, 0.55, 0.35),
            (0.50, 0.60, 0.40),
            (0.48, 0.58, 0.38),
            (0.46, 0.56, 0.36),
        ))
    ]

    selection = select_heldout_summaries(
        baseline=baseline,
        candidates=candidates,
        expected_scene_ids=SCENE_IDS,
    )

    assert selection['selected']['label'] == 'v2_2_replica_1'
    assert selection['selected']['checkpoint'].endswith(
        'v2_2_replica_1.pth',
    )
    assert selection['delta_vs_baseline'] == pytest.approx({
        'CR': 0.1,
        'PCR': 0.1,
        'WCR': 0.1,
        'Q': 0.1,
    })
    assert [row['label'] for row in selection['ranking']] == [
        'v2_2_replica_1',
        'v2_2_replica_2',
        'v2_2_replica_3',
        'v2_2_replica_0',
    ]


def test_selector_breaks_equal_q_ties_by_label() -> None:
    baseline = _summary(
        'v2_1',
        'V2-1',
        cr=0.4,
        pcr=0.4,
        wcr=0.4,
    )
    candidates = [
        _summary(label, 'V2-2', cr=0.5, pcr=0.5, wcr=0.5)
        for label in ('replica_b', 'replica_a', 'replica_c', 'replica_d')
    ]

    selection = select_heldout_summaries(
        baseline=baseline,
        candidates=candidates,
        expected_scene_ids=SCENE_IDS,
    )

    assert selection['selected']['label'] == 'replica_a'


@pytest.mark.parametrize(
    ('field', 'value', 'message'),
    [
        ('split', 'val_seen', 'train'),
        ('scene_ids', list(range(8)), 'scene'),
        ('deterministic', False, 'deterministic'),
        ('finite', False, 'finite'),
        ('reward_reconstruction_max_error', 1e-3, 'reward'),
    ],
)
def test_selector_rejects_incompatible_evaluation_protocol(
    field: str,
    value,
    message: str,
) -> None:
    baseline = _summary(
        'v2_1',
        'V2-1',
        cr=0.4,
        pcr=0.4,
        wcr=0.4,
    )
    baseline[field] = value
    candidates = [
        _summary(
            f'replica_{index}',
            'V2-2',
            cr=0.5,
            pcr=0.5,
            wcr=0.5,
        )
        for index in range(4)
    ]

    with pytest.raises(ValueError, match=message):
        select_heldout_summaries(
            baseline=baseline,
            candidates=candidates,
            expected_scene_ids=SCENE_IDS,
        )


def test_selector_rejects_quality_not_reconstructable_from_metrics() -> None:
    baseline = _summary(
        'v2_1',
        'V2-1',
        cr=0.4,
        pcr=0.4,
        wcr=0.4,
    )
    candidates = [
        _summary(
            f'replica_{index}',
            'V2-2',
            cr=0.5,
            pcr=0.5,
            wcr=0.5,
        )
        for index in range(4)
    ]
    candidates[0]['aggregate']['Q'] = 0.9

    with pytest.raises(ValueError, match='formula'):
        select_heldout_summaries(
            baseline=baseline,
            candidates=candidates,
            expected_scene_ids=SCENE_IDS,
        )
