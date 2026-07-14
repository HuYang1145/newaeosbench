import pytest
import torch

from constellation.new_transformers.preference_divergence import (
    build_first_divergence_record,
    first_divergence_index,
)


def _trajectory(actions: list[list[int]], *, progress_offset: int = 0):
    num_steps = len(actions)
    progress = torch.tensor([
        [step + progress_offset, 0] for step in range(num_steps)
    ], dtype=torch.uint8)
    return {
        'constellation': {
            'sensor_enabled': torch.ones(num_steps, 2, dtype=torch.long),
            'data': torch.zeros(num_steps, 2, 8),
        },
        'taskset': {'progress': progress},
        'actions': {'task_id': torch.tensor(actions)},
        # P1 不允许把离线 is_visible 标签放进裁判模型输入。
        'is_visible': torch.zeros(num_steps, 2, 2, dtype=torch.bool),
    }


def test_first_divergence_index_finds_first_different_joint_action() -> None:
    left = torch.tensor([[0, -1], [1, -1], [1, 0]])
    right = torch.tensor([[0, -1], [0, 1], [1, 0]])

    assert first_divergence_index(left, right) == 1
    assert first_divergence_index(left, left.clone()) is None


def test_first_divergence_index_rejects_different_shapes() -> None:
    with pytest.raises(ValueError, match='same shape'):
        first_divergence_index(
            torch.zeros(2, 2, dtype=torch.long),
            torch.zeros(3, 2, dtype=torch.long),
        )


def test_record_keeps_exact_actions_and_checks_shared_state() -> None:
    record = build_first_divergence_record(
        scene_id=7,
        better_candidate='sample',
        worse_candidate='greedy',
        better_cost=3.0,
        worse_cost=4.0,
        better_trajectory_path='sample.pth',
        worse_trajectory_path='greedy.pth',
        better_trajectory=_trajectory([[0, -1], [1, -1], [1, 0]]),
        worse_trajectory=_trajectory([[0, -1], [0, 1], [1, 0]]),
    )

    assert record is not None
    assert record['divergence_index'] == 1
    assert record['cost_margin'] == 1.0
    assert record['shared_state_match'] is True
    assert record['better_action'] == [1, -1]
    assert record['worse_action'] == [0, 1]
    assert record['changed_satellites'] == 2
    assert record['better_action_summary']['duplicate_assignments'] == 0


def test_record_marks_state_mismatch_without_using_visibility() -> None:
    record = build_first_divergence_record(
        scene_id=8,
        better_candidate='a',
        worse_candidate='b',
        better_cost=1.0,
        worse_cost=2.0,
        better_trajectory_path='a.pth',
        worse_trajectory_path='b.pth',
        better_trajectory=_trajectory([[0, -1], [1, -1], [1, 0]]),
        worse_trajectory=_trajectory(
            [[0, -1], [0, 1], [1, 0]],
            progress_offset=1,
        ),
    )

    assert record is not None
    assert record['shared_state_match'] is False
    assert record['state_match']['task_progress'] is False


def test_record_uses_previous_logged_sensor_state() -> None:
    better = _trajectory([[0, -1], [1, -1], [1, 0]])
    worse = _trajectory([[0, -1], [0, 1], [1, 0]])
    better['constellation']['sensor_enabled'][1] = torch.tensor([1, 0])
    worse['constellation']['sensor_enabled'][1] = torch.tensor([0, 1])

    record = build_first_divergence_record(
        scene_id=9,
        better_candidate='a',
        worse_candidate='b',
        better_cost=1.0,
        worse_cost=2.0,
        better_trajectory_path='a.pth',
        worse_trajectory_path='b.pth',
        better_trajectory=better,
        worse_trajectory=worse,
    )

    assert record is not None
    assert record['shared_state_match'] is True
    assert record['sensor_enabled_source_index'] == 0


def test_divergence_at_zero_marks_initial_sensor_state_unavailable() -> None:
    record = build_first_divergence_record(
        scene_id=10,
        better_candidate='a',
        worse_candidate='b',
        better_cost=1.0,
        worse_cost=2.0,
        better_trajectory_path='a.pth',
        worse_trajectory_path='b.pth',
        better_trajectory=_trajectory([[0, -1], [1, -1]]),
        worse_trajectory=_trajectory([[-1, 0], [1, -1]]),
    )

    assert record is not None
    assert record['divergence_index'] == 0
    assert record['current_state_reconstructable'] is False
    assert record['shared_state_match'] is False
    assert record['sensor_enabled_source_index'] is None
