from __future__ import annotations

import pytest

from tools.merge_event_v2_eval_summaries import merge_eval_summaries


def _summary(scene_ids: tuple[int, ...], *, checkpoint: str = '/c.pth'):
    scenes = [
        {
            'scene_id': scene_id,
            'CR': 0.4 + scene_id / 100,
            'PCR': 0.5 + scene_id / 100,
            'WCR': 0.3 + scene_id / 100,
            'Q': 0.6 * (0.4 + scene_id / 100)
            + 0.2 * (0.5 + scene_id / 100)
            + 0.2 * (0.3 + scene_id / 100),
            'TAT_s': 100.0 + scene_id,
            'PC_Wh': 2.0 + scene_id / 10,
            'CS_paper': 3.0 + scene_id,
            'events': 10,
            'physical_seconds': 3600,
            'reward_reconstruction_error': 0.0,
        }
        for scene_id in scene_ids
    ]
    return {
        'label': 'candidate',
        'checkpoint': checkpoint,
        'stage': 'V2-2-Large',
        'checkpoint_updates': 100,
        'checkpoint_policy_version': 100,
        'checkpoint_train_scene_ids': list(range(205, 325)),
        'config_fingerprint': 'a' * 64,
        'split': 'val_seen',
        'scene_ids': list(scene_ids),
        'max_time_step': 3600,
        'deterministic': True,
        'amp_enabled': True,
        'amp_dtype': 'bfloat16',
        'scenes': scenes,
        'aggregate': {},
        'finite': True,
        'reward_reconstruction_max_error': 0.0,
    }


def test_merge_eval_summaries_reconstructs_macro_metrics_and_scene_order() -> None:
    merged = merge_eval_summaries(
        [_summary((2, 3)), _summary((0, 1))],
        expected_scene_ids=(0, 1, 2, 3),
    )

    assert merged['scene_ids'] == [0, 1, 2, 3]
    assert [row['scene_id'] for row in merged['scenes']] == [0, 1, 2, 3]
    assert merged['aggregate']['CR'] == pytest.approx(0.415)
    assert merged['aggregate']['PCR'] == pytest.approx(0.515)
    assert merged['aggregate']['WCR'] == pytest.approx(0.315)
    assert merged['aggregate']['Q'] == pytest.approx(
        0.6 * 0.415 + 0.2 * 0.515 + 0.2 * 0.315,
    )
    assert merged['aggregate']['TAT_s'] == pytest.approx(101.5)
    assert merged['aggregate']['PC_Wh'] == pytest.approx(2.15)
    assert merged['finite'] is True


@pytest.mark.parametrize(
    ('mutation', 'message'),
    [
        (lambda summary: summary.update(split='val_unseen'), 'split'),
        (lambda summary: summary.update(checkpoint='/different.pth'), 'checkpoint'),
        (lambda summary: summary.update(max_time_step=120), 'max time'),
        (lambda summary: summary.update(finite=False), 'finite'),
    ],
)
def test_merge_eval_summaries_rejects_protocol_drift(
    mutation,
    message: str,
) -> None:
    first = _summary((0, 1))
    second = _summary((2, 3))
    mutation(second)

    with pytest.raises(ValueError, match=message):
        merge_eval_summaries(
            [first, second],
            expected_scene_ids=(0, 1, 2, 3),
        )


def test_merge_eval_summaries_rejects_missing_or_duplicate_scenes() -> None:
    with pytest.raises(ValueError, match='scene'):
        merge_eval_summaries(
            [_summary((0, 1)), _summary((1, 2))],
            expected_scene_ids=(0, 1, 2, 3),
        )
