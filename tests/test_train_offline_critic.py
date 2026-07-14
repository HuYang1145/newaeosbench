from pathlib import Path

import pytest
import torch

from constellation.new_transformers.offline_critic import TrajectoryRecord
from tools import train_offline_critic


def _trajectory() -> dict:
    return {
        'constellation': {
            'sensor_enabled': torch.ones(4, 2, dtype=torch.long),
            'data': torch.zeros(4, 2, 8),
        },
        'taskset': {
            'progress': torch.tensor([
                [0, 0],
                [1, 0],
                [2, 0],
                [2, 1],
            ], dtype=torch.uint8),
        },
        'actions': {
            'task_id': torch.tensor([
                [0, -1],
                [0, -1],
                [1, -1],
                [-1, -1],
            ]),
        },
        'is_visible': torch.ones(4, 2, 2, dtype=torch.bool),
    }


def test_select_records_is_seeded_and_scene_safe() -> None:
    records = [
        TrajectoryRecord(i, 1, Path(f'{i}.pth'), Path(f'{i}.json'), float(i))
        for i in range(20)
    ]
    records.append(
        TrajectoryRecord(3, 2, Path('3b.pth'), Path('3b.json'), 2.0),
    )

    first = train_offline_critic.select_records(records, limit=5, seed=9)
    second = train_offline_critic.select_records(records, limit=5, seed=9)

    assert first == second
    selected_ids = {record.scene_id for record in first}
    assert len(selected_ids) == 5
    assert all(
        sum(record.scene_id == scene_id for record in first)
        == sum(record.scene_id == scene_id for record in records)
        for scene_id in selected_ids
    )


def test_load_transition_dataset_uses_existing_trajectory(
    monkeypatch,
) -> None:
    record = TrajectoryRecord(
        scene_id=17,
        epoch=1,
        trajectory_path=Path('trajectory.pth'),
        metrics_path=Path('trajectory.json'),
        episode_cost=3.5,
    )
    monkeypatch.setattr(
        train_offline_critic.torch,
        'load',
        lambda *args, **kwargs: _trajectory(),
    )
    monkeypatch.setattr(
        train_offline_critic,
        'load_scene_context',
        lambda taskset_path, constellation_path: {
            'task_durations': torch.tensor([2.0, 1.0]),
            'task_release_times': torch.tensor([0.0, 0.0]),
            'satellite_sensor_power': torch.tensor([10.0, 20.0]),
            'task_static_data': torch.zeros(2, 5),
            'constellation_static_data': torch.zeros(2, 8),
            'task_sensor_type': torch.tensor([1, 2]),
            'constellation_sensor_type': torch.tensor([1, 2]),
        },
    )

    dataset = train_offline_critic.load_transition_dataset(
        [record],
        tasksets_root=Path('tasksets'),
        split='train',
        samples_per_trajectory=2,
    )

    assert dataset.trajectory_ids.tolist() == [0, 0]
    assert dataset.episode_cost.tolist() == [3.5, 3.5]
    assert dataset.done.tolist() == [False, True]
    assert dataset.action.shape[1] > 6


def test_load_transition_dataset_builds_dense_cost_to_go(
    monkeypatch,
) -> None:
    record = TrajectoryRecord(
        scene_id=17,
        epoch=1,
        trajectory_path=Path('trajectory.pth'),
        metrics_path=Path('trajectory.json'),
        episode_cost=3.5,
    )
    monkeypatch.setattr(
        train_offline_critic.torch,
        'load',
        lambda *args, **kwargs: _trajectory(),
    )
    monkeypatch.setattr(
        train_offline_critic,
        'load_scene_context',
        lambda taskset_path, constellation_path: {
            'task_durations': torch.tensor([2.0, 1.0]),
            'task_release_times': torch.tensor([0.0, 0.0]),
            'satellite_sensor_power': torch.tensor([10.0, 20.0]),
            'task_static_data': torch.zeros(2, 5),
            'constellation_static_data': torch.zeros(2, 8),
            'task_sensor_type': torch.tensor([1, 2]),
            'constellation_sensor_type': torch.tensor([1, 2]),
        },
    )

    dataset = train_offline_critic.load_transition_dataset(
        [record],
        tasksets_root=Path('tasksets'),
        split='train',
        samples_per_trajectory=3,
        reward_mode='dense',
    )

    assert dataset.reward.sum().item() == pytest.approx(-3.5)
    assert dataset.return_to_go[0].item() == pytest.approx(-3.5)
    assert dataset.reward[:-1].abs().sum().item() > 0
