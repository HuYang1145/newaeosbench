import json
from pathlib import Path

import torch
from tools import train_first_divergence_graph_q
from tools.train_first_divergence_graph_q import (
    _action_summary,
    build_graph_q_sample,
)

from constellation.new_transformers.graph_q_critic import GraphQSceneContext


def _trajectory() -> dict[str, object]:
    return {
        'constellation': {
            'sensor_enabled': torch.tensor([
                [1, 0],
                [0, 1],
                [1, 1],
            ]),
            'data': torch.arange(3 * 2 * 2).reshape(3, 2, 2).float(),
        },
        'taskset': {
            'progress': torch.tensor([
                [0, 0],
                [2, 0],
                [2, 1],
            ]),
        },
        'actions': {
            'task_id': torch.tensor([
                [10, -1],
                [10, 20],
                [-1, 20],
            ]),
        },
        # Graph-Q 数据构造不得读取这个离线物理标签。
        'is_visible': torch.ones(3, 2, 2, dtype=torch.bool),
    }


def test_build_graph_q_sample_uses_pre_action_state_and_exact_ids() -> None:
    context = GraphQSceneContext(
        task_ids=torch.tensor([10, 20]),
        task_durations=torch.tensor([4.0, 2.0]),
        task_static_data=torch.tensor([
            [0.0, 8.0, 4.0, 1.0, 2.0],
            [1.0, 9.0, 2.0, 3.0, 4.0],
        ]),
        task_sensor_type=torch.tensor([1, 2]),
        constellation_static_data=torch.zeros(2, 3),
        constellation_sensor_type=torch.tensor([1, 2]),
    )
    record = {
        'scene_id': 9,
        'divergence_index': 1,
        'sensor_enabled_source_index': 0,
        'better_action': [10, 20],
        'worse_action': [20, 10],
        'better_candidate': 'sample',
        'worse_candidate': 'greedy',
        'better_cost': 3.0,
        'worse_cost': 4.0,
        'cost_margin': 1.0,
    }

    sample = build_graph_q_sample(
        record=record,
        trajectory=_trajectory(),
        context=context,
    )

    assert sample.better_action.tolist() == [0, 1]
    assert sample.worse_action.tolist() == [1, 0]
    assert sample.previous_action.tolist() == [0, -1]
    assert sample.satellite_features[:, -2].tolist() == [1.0, 0.0]
    assert sample.task_features[:, -3].tolist() == [0.5, 0.0]
    assert sample.compatibility.tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_build_graph_q_sample_rejects_unknown_task_id() -> None:
    context = GraphQSceneContext(
        task_ids=torch.tensor([0]),
        task_durations=torch.tensor([1.0]),
        task_static_data=torch.zeros(1, 5),
        task_sensor_type=torch.tensor([1]),
        constellation_static_data=torch.zeros(1, 3),
        constellation_sensor_type=torch.tensor([1]),
    )
    record = {
        'scene_id': 10,
        'divergence_index': 1,
        'sensor_enabled_source_index': 0,
        'better_action': [99],
        'worse_action': [-1],
        'better_candidate': 'a',
        'worse_candidate': 'b',
        'better_cost': 1.0,
        'worse_cost': 2.0,
        'cost_margin': 1.0,
    }

    try:
        build_graph_q_sample(
            record=record,
            trajectory=_trajectory(),
            context=context,
        )
    except ValueError as error:
        assert 'unknown task id 99' in str(error)
    else:
        raise AssertionError('unknown task id must be rejected')


def test_action_summary_does_not_count_null_to_null_as_continuity() -> None:
    summary = _action_summary(
        torch.tensor([1, -1]),
        previous_action=torch.tensor([0, -1]),
        task_progress_ratio=torch.tensor([0.0, 0.5]),
        compatibility=torch.ones(2, 2),
    )

    assert summary[-1].item() == 0.0


def test_load_graph_q_samples_uses_only_usable_current_state_records(
    tmp_path: Path,
    monkeypatch,
) -> None:
    divergence_path = tmp_path / 'preferences.json'
    divergence_path.write_text(
        json.dumps({
            'input_contract': {
                'uses_current_state_only': True,
                'uses_is_visible_as_input': False,
                'basilisk_online_inference': False,
            },
            'records': [
                {
                    'scene_id': 9,
                    'divergence_index': 1,
                    'sensor_enabled_source_index': 0,
                    'better_action': [10, 20],
                    'worse_action': [20, 10],
                    'better_candidate': 'sample',
                    'worse_candidate': 'greedy',
                    'better_cost': 3.0,
                    'worse_cost': 4.0,
                    'cost_margin': 1.0,
                    'better_trajectory_path': 'sample.pth',
                    'usable_for_graph_q': True,
                },
                {
                    'scene_id': 10,
                    'usable_for_graph_q': False,
                },
            ],
        }),
        encoding='utf-8'
    )
    context = GraphQSceneContext(
        task_ids=torch.tensor([10, 20]),
        task_durations=torch.tensor([4.0, 2.0]),
        task_static_data=torch.zeros(2, 5),
        task_sensor_type=torch.tensor([1, 2]),
        constellation_static_data=torch.zeros(2, 3),
        constellation_sensor_type=torch.tensor([1, 2]),
    )
    loaded_paths = []
    monkeypatch.setattr(
        train_first_divergence_graph_q,
        'load_graph_q_scene_context',
        lambda **kwargs: context,
    )

    def fake_load(path, **kwargs):
        loaded_paths.append(str(path))
        return _trajectory()

    monkeypatch.setattr(
        train_first_divergence_graph_q.torch, 'load', fake_load
    )

    samples, metadata = train_first_divergence_graph_q.load_graph_q_samples(
        divergence_path,
        tasksets_root=tmp_path / 'tasksets',
        constellations_root=tmp_path / 'constellations',
        split='train',
    )

    assert len(samples) == 1
    assert metadata['num_usable_samples'] == 1
    assert loaded_paths == ['sample.pth']
