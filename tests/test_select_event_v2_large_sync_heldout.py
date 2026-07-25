from __future__ import annotations

import json
import pathlib

import pytest
import torch

from tools.select_event_v2_large_sync_heldout import (
    select_large_sync_heldout_summaries,
    write_selection_artifacts,
)


SCENE_IDS = tuple(range(196, 204))


def _summary(
    label: str,
    stage: str,
    *,
    cr: float,
    pcr: float,
    wcr: float,
    update: int,
) -> dict:
    return {
        'label': label,
        'checkpoint': f'/checkpoints/{label}.pth',
        'stage': stage,
        'checkpoint_updates': update,
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


def _baseline() -> dict:
    return _summary(
        'v2_2_replica_0',
        'V2-2',
        cr=0.40,
        pcr=0.50,
        wcr=0.30,
        update=1046,
    )


def test_large_sync_selector_ranks_all_seed_checkpoints_by_heldout_q() -> None:
    candidates = [
        _summary(
            'seed_5408_update_000100',
            'V2-2-Large',
            cr=0.42,
            pcr=0.52,
            wcr=0.32,
            update=100,
        ),
        _summary(
            'seed_5409_update_000100',
            'V2-2-Large',
            cr=0.44,
            pcr=0.54,
            wcr=0.34,
            update=100,
        ),
        _summary(
            'seed_5408_update_000200',
            'V2-2-Large',
            cr=0.43,
            pcr=0.53,
            wcr=0.33,
            update=200,
        ),
    ]

    selection = select_large_sync_heldout_summaries(
        baseline=_baseline(),
        candidates=candidates,
        expected_scene_ids=SCENE_IDS,
    )

    assert selection['selected']['label'] == (
        'seed_5409_update_000100'
    )
    assert [row['label'] for row in selection['ranking']] == [
        'seed_5409_update_000100',
        'seed_5408_update_000200',
        'seed_5408_update_000100',
    ]
    assert selection['selected']['seed'] == 5409
    assert selection['selected']['update'] == 100
    assert selection['delta_vs_baseline'] == pytest.approx({
        'CR': 0.04,
        'PCR': 0.04,
        'WCR': 0.04,
        'Q': 0.04,
    })


def test_large_sync_selector_uses_registered_tie_break_order() -> None:
    candidates = [
        _summary(
            'seed_5409_update_000100',
            'V2-2-Large',
            cr=0.50,
            pcr=0.40,
            wcr=0.40,
            update=100,
        ),
        _summary(
            'seed_5408_update_000200',
            'V2-2-Large',
            cr=0.46,
            pcr=0.46,
            wcr=0.46,
            update=200,
        ),
    ]
    # 两者 Q 都是 0.46；第二个候选的最小单项提升更大。
    selection = select_large_sync_heldout_summaries(
        baseline=_summary(
            'v2_2_replica_0',
            'V2-2',
            cr=0.40,
            pcr=0.40,
            wcr=0.40,
            update=1046,
        ),
        candidates=candidates,
        expected_scene_ids=SCENE_IDS,
    )

    assert selection['selected']['label'] == (
        'seed_5408_update_000200'
    )

    equal_candidates = [
        _summary(
            'seed_5409_update_000100',
            'V2-2-Large',
            cr=0.45,
            pcr=0.45,
            wcr=0.45,
            update=100,
        ),
        _summary(
            'seed_5408_update_000100',
            'V2-2-Large',
            cr=0.45,
            pcr=0.45,
            wcr=0.45,
            update=100,
        ),
        _summary(
            'seed_5408_update_000200',
            'V2-2-Large',
            cr=0.45,
            pcr=0.45,
            wcr=0.45,
            update=200,
        ),
    ]
    selection = select_large_sync_heldout_summaries(
        baseline=_baseline(),
        candidates=equal_candidates,
        expected_scene_ids=SCENE_IDS,
    )

    assert selection['selected']['label'] == (
        'seed_5408_update_000100'
    )


@pytest.mark.parametrize(
    ('field', 'value', 'message'),
    [
        ('stage', 'V2-3', 'stage'),
        ('split', 'val_seen', 'train'),
        ('scene_ids', list(range(8)), 'scene'),
        ('checkpoint_updates', 999, 'update'),
        ('finite', False, 'finite'),
    ],
)
def test_large_sync_selector_rejects_protocol_drift(
    field: str,
    value,
    message: str,
) -> None:
    candidate = _summary(
        'seed_5408_update_000100',
        'V2-2-Large',
        cr=0.45,
        pcr=0.45,
        wcr=0.45,
        update=100,
    )
    candidate[field] = value

    with pytest.raises(ValueError, match=message):
        select_large_sync_heldout_summaries(
            baseline=_baseline(),
            candidates=[candidate, {
                **candidate,
                'label': 'seed_5409_update_000100',
                'checkpoint_updates': 100,
                field: (
                    'V2-2-Large'
                    if field == 'stage'
                    else candidate.get(field)
                ),
            }],
            expected_scene_ids=SCENE_IDS,
        )


def test_selection_artifacts_are_atomic_and_best_is_hard_linked(
    tmp_path: pathlib.Path,
) -> None:
    checkpoint = tmp_path / 'checkpoint_update_000100.pth'
    torch.save({'stage': 'V2-2-Large'}, checkpoint)
    selection = {
        'selected': {
            'checkpoint': str(checkpoint),
            'label': 'seed_5408_update_000100',
        },
    }
    output = tmp_path / 'selection.json'
    best = tmp_path / 'checkpoint_best.pth'

    write_selection_artifacts(
        selection,
        output=output,
        best_link=best,
    )

    assert json.loads(output.read_text()) == selection
    assert best.stat().st_ino == checkpoint.stat().st_ino
    assert not output.with_suffix('.json.tmp').exists()
