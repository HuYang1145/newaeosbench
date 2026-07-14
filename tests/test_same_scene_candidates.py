import json
from pathlib import Path

import pytest
import torch

from tools.summarize_same_scene_candidates import summarize_candidates


def _write_candidate(
    root: Path,
    *,
    candidate: str,
    scene_id: int,
    metrics: dict[str, float],
    actions: list[list[int]],
) -> None:
    directory = root / candidate / 'train' / f'{scene_id // 1000:02}'
    directory.mkdir(parents=True)
    (directory / f'{scene_id:05}.json').write_text(
        json.dumps(metrics),
        encoding='utf-8',
    )
    torch.save(
        {'actions': {'task_id': torch.tensor(actions)}},
        directory / f'{scene_id:05}.pth',
    )


def test_summary_builds_only_valid_distinct_same_scene_preferences(
    tmp_path: Path,
) -> None:
    common = dict(scene_id=7)
    _write_candidate(
        tmp_path,
        candidate='candidate_000_greedy',
        metrics={'CR': 0.5, 'PCR': 0.5, 'WCR': 0.5,
                 'TAT': 700.0, 'PC_Wh': 100.0},
        actions=[[-1, 0], [1, -1]],
        **common,
    )
    _write_candidate(
        tmp_path,
        candidate='candidate_001_sample',
        metrics={'CR': 0.6, 'PCR': 0.6, 'WCR': 0.6,
                 'TAT': 700.0, 'PC_Wh': 100.0},
        actions=[[0, -1], [1, -1]],
        **common,
    )
    # 动作完全相同的重复候选不能构成偏好对。
    _write_candidate(
        tmp_path,
        candidate='candidate_002_duplicate',
        metrics={'CR': 0.6, 'PCR': 0.6, 'WCR': 0.6,
                 'TAT': 700.0, 'PC_Wh': 100.0},
        actions=[[0, -1], [1, -1]],
        **common,
    )
    # 无成功任务会产生 NaN TAT，必须保留诊断但退出偏好训练。
    _write_candidate(
        tmp_path,
        candidate='candidate_003_invalid',
        metrics={'CR': 0.0, 'PCR': 0.0, 'WCR': 0.0,
                 'TAT': float('nan'), 'PC_Wh': 10.0},
        actions=[[-1, -1], [-1, -1]],
        **common,
    )

    summary = summarize_candidates(
        tmp_path,
        split='train',
        greedy_candidate='candidate_000_greedy',
    )

    assert summary['num_scenes'] == 1
    assert summary['num_scenes_with_action_diversity'] == 1
    assert summary['num_scenes_with_preference'] == 1
    assert summary['num_candidate_pairs'] == 1
    scene = summary['scenes'][0]
    assert scene['best_candidate'] == 'candidate_001_sample'
    assert scene['greedy_cost'] == pytest.approx(4.0)
    assert scene['best_cost'] == pytest.approx(1 / 0.6 + 2.0)
    assert scene['best_improvement_vs_greedy'] == pytest.approx(
        4.0 - (1 / 0.6 + 2.0),
    )
    invalid = next(
        item for item in scene['candidates']
        if item['candidate'] == 'candidate_003_invalid'
    )
    assert invalid['valid'] is False
    assert invalid['cost'] is None
    pair = summary['preference_pairs'][0]
    assert pair['scene_id'] == 7
    assert pair['better_candidate'] == 'candidate_001_sample'
    assert pair['worse_candidate'] == 'candidate_000_greedy'
    assert pair['cost_margin'] == pytest.approx(4.0 - (1 / 0.6 + 2.0))
