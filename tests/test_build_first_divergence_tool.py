import json
from pathlib import Path

import torch

from tools.build_first_divergence_preferences import build_divergence_dataset


def _write_trajectory(path: Path, actions: list[list[int]]) -> None:
    num_steps = len(actions)
    torch.save({
        'constellation': {
            'sensor_enabled': torch.ones(num_steps, 2, dtype=torch.long),
            'data': torch.zeros(num_steps, 2, 8),
        },
        'taskset': {
            'progress': torch.zeros(num_steps, 2, dtype=torch.uint8),
        },
        'actions': {'task_id': torch.tensor(actions)},
        'is_visible': torch.rand(num_steps, 2, 2) > 0.5,
    }, path)


def test_build_dataset_groups_same_scene_first_divergence(tmp_path: Path) -> None:
    better_path = tmp_path / 'better.pth'
    worse_path = tmp_path / 'worse.pth'
    _write_trajectory(better_path, [[0, -1], [1, -1], [1, 0]])
    _write_trajectory(worse_path, [[0, -1], [0, 1], [1, 0]])
    summary_path = tmp_path / 'summary.json'
    summary_path.write_text(json.dumps({
        'score_definition': 'lower is better',
        'preference_pairs': [{
            'scene_id': 3,
            'better_candidate': 'sample',
            'worse_candidate': 'greedy',
            'better_cost': 2.0,
            'worse_cost': 3.0,
            'cost_margin': 1.0,
            'better_trajectory_path': str(better_path),
            'worse_trajectory_path': str(worse_path),
        }],
    }), encoding='utf-8')

    result = build_divergence_dataset(summary_path, min_cost_margin=0.05)

    assert result['summary']['num_source_pairs'] == 1
    assert result['summary']['num_divergence_records'] == 1
    assert result['summary']['num_shared_state_records'] == 1
    assert result['summary']['num_unreconstructable_initial_state_records'] == 0
    assert result['summary']['num_reconstructable_state_mismatch_records'] == 0
    assert result['summary']['num_usable_records'] == 1
    assert result['records'][0]['divergence_index'] == 1
    assert result['records'][0]['usable_for_graph_q'] is True
