import json
from pathlib import Path

import torch

from tools import train_same_scene_preference_critic


def _trajectory() -> dict[str, object]:
    return {
        'constellation': {
            'sensor_enabled': torch.ones(5, 2, dtype=torch.long),
            'data': torch.zeros(5, 2, 8),
        },
        'taskset': {
            'progress': torch.tensor([
                [0, 0],
                [1, 0],
                [2, 0],
                [2, 1],
                [2, 1],
            ], dtype=torch.uint8),
        },
        'actions': {
            'task_id': torch.tensor([
                [0, -1],
                [0, -1],
                [-1, 1],
                [-1, 1],
                [-1, -1],
            ]),
        },
        'is_visible': torch.ones(5, 2, 2, dtype=torch.bool),
    }


def test_load_candidate_features_aggregates_early_transitions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trajectory_paths = [tmp_path / f'candidate_{index}.pth' for index in range(4)]
    summary = {
        'split': 'train',
        'scenes': [
            {
                'scene_id': scene_id,
                'candidates': [
                    {
                        'candidate': f'candidate_{candidate_id}',
                        'valid': True,
                        'cost': float(scene_id + candidate_id + 1),
                        'trajectory_path': str(
                            trajectory_paths[scene_id * 2 + candidate_id]
                        ),
                    }
                    for candidate_id in range(2)
                ],
            }
            for scene_id in range(2)
        ],
    }
    summary_path = tmp_path / 'summary.json'
    summary_path.write_text(json.dumps(summary), encoding='utf-8')
    monkeypatch.setattr(
        train_same_scene_preference_critic.torch,
        'load',
        lambda *args, **kwargs: _trajectory(),
    )
    monkeypatch.setattr(
        train_same_scene_preference_critic,
        'load_scene_context',
        lambda taskset_path, constellation_path: {
            'task_durations': torch.tensor([2.0, 1.0]),
            'task_static_data': torch.zeros(2, 5),
            'constellation_static_data': torch.zeros(2, 8),
            'task_sensor_type': torch.tensor([1, 2]),
            'constellation_sensor_type': torch.tensor([1, 2]),
        },
    )

    candidates, records = (
        train_same_scene_preference_critic.load_candidate_features(
            summary_path,
            tasksets_root=tmp_path / 'tasksets',
            constellations_root=tmp_path / 'constellations',
            samples_per_candidate=2,
        )
    )

    assert candidates.scene_ids.tolist() == [0, 0, 1, 1]
    assert candidates.cost.tolist() == [1.0, 2.0, 2.0, 3.0]
    assert candidates.state.shape[0] == 4
    assert candidates.action.shape[0] == 4
    assert candidates.state.shape[1] > 10
    assert candidates.action.shape[1] > 10
    assert [record['candidate'] for record in records] == [
        'candidate_0', 'candidate_1', 'candidate_0', 'candidate_1',
    ]


def test_scene_folds_cover_each_scene_exactly_once() -> None:
    candidates = train_same_scene_preference_critic.CandidateTensors(
        scene_ids=torch.arange(8).repeat_interleave(2),
        state=torch.zeros(16, 2),
        action=torch.zeros(16, 1),
        cost=torch.arange(16).float(),
    )
    validation_scenes = []
    for fold_index in range(4):
        train, val, train_ids, val_ids = (
            train_same_scene_preference_critic.split_candidates_by_scene(
                candidates,
                val_fraction=0.25,
                seed=3407,
                num_folds=4,
                fold_index=fold_index,
            )
        )
        assert set(train_ids).isdisjoint(val_ids)
        assert set(train.scene_ids.tolist()).isdisjoint(
            val.scene_ids.tolist(),
        )
        validation_scenes.extend(val_ids)

    assert sorted(validation_scenes) == list(range(8))
