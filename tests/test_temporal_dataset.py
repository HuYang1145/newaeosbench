import types

import torch

from constellation.new_transformers.dataset import Dataset, TemporalBatch


def _fake_dataset(
    *,
    tasks_mask: torch.Tensor,
    horizons: tuple[int, ...] = (1, 3),
) -> Dataset:
    dataset = object.__new__(Dataset)
    dataset._batch_size = 99
    dataset._include_temporal_history = True
    dataset._temporal_horizons = horizons

    def load_tasks(self, taskset, id_):
        del id_
        time, tasks = taskset['progress'].shape
        data = torch.zeros(time, tasks, 6)
        data[..., 2] = 3
        return (
            torch.ones(time, tasks, dtype=torch.long),
            data,
            tasks_mask.clone(),
        )

    def load_constellation(self, constellation, id_, indices):
        del id_
        index = torch.tensor(indices, dtype=torch.long)
        satellites = constellation['data'].shape[1]
        return (
            torch.ones(len(indices), satellites, dtype=torch.long),
            constellation['sensor_enabled'][index],
            constellation['data'][index],
            torch.ones(len(indices), satellites, dtype=torch.bool),
        )

    dataset._load_tasks = types.MethodType(load_tasks, dataset)
    dataset._load_constellation = types.MethodType(load_constellation, dataset)
    return dataset


def _trajectory(
    actions: list[list[int]],
    *,
    num_tasks: int = 3,
) -> dict:
    action_tensor = torch.tensor(actions, dtype=torch.long)
    time, satellites = action_tensor.shape
    constellation_data = torch.zeros(time, satellites, 56)
    constellation_data[..., -1] = torch.arange(1, time + 1).view(time, 1) * 10
    return {
        'constellation': {
            'sensor_enabled': torch.arange(time).view(time, 1) % 2 == 1,
            'data': constellation_data,
        },
        'taskset': {
            'progress': torch.zeros(time, num_tasks),
        },
        'actions': {
            'task_id': action_tensor,
        },
        'is_visible': torch.zeros(
            time, satellites, num_tasks, dtype=torch.bool
        ),
    }


def test_temporal_dataset_uses_previous_saved_satellite_state() -> None:
    mask = torch.ones(5, 3, dtype=torch.bool)
    dataset = _fake_dataset(tasks_mask=mask)

    batch = dataset._build_batch(
        0,
        123,
        7,
        _trajectory([[-1], [1], [1], [-1], [-1]]),
    )

    assert batch.time_steps == [1, 2, 3]
    assert batch.constellation_data[:, 0, -1].tolist() == [10., 20., 30.]
    assert batch.constellation_sensor_enabled[:, 0].tolist() == [False, True, False]


def test_temporal_dataset_builds_history_and_outcome_for_sampled_actions() -> None:
    mask = torch.ones(5, 3, dtype=torch.bool)
    dataset = _fake_dataset(tasks_mask=mask)

    batch = dataset._build_batch(
        0,
        123,
        7,
        _trajectory([[-1], [1], [1], [-1], [-1]]),
    )

    assert isinstance(batch.temporal, TemporalBatch)
    assert batch.temporal.previous_task_indices[:, 0].tolist() == [-1, 1, 1]
    assert batch.temporal.previous_was_idle[:, 0].tolist() == [True, False, False]
    assert batch.temporal.run_lengths[:, 0].tolist() == [1, 1, 2]
    assert batch.temporal.outcome_valid[:, 0].tolist() == [True, True, False]
    assert batch.temporal.horizons.tolist() == [1, 3]
    assert batch.temporal.visible.shape == (3, 1, 2)
    assert batch.temporal.visible_observed.shape == (3, 1, 2)


def test_temporal_dataset_remaps_previous_task_after_task_pruning() -> None:
    mask = torch.tensor([
        [True, False, True],
        [True, False, True],
        [True, False, True],
        [True, False, True],
    ])
    dataset = _fake_dataset(tasks_mask=mask, horizons=(1,))

    batch = dataset._build_batch(
        0,
        123,
        7,
        _trajectory([[-1], [2], [2], [-1]]),
    )

    assert batch.tasks_data.shape[1] == 2
    assert batch.actions_task_id[:, 0].tolist() == [1, 1]
    assert batch.temporal is not None
    assert batch.temporal.previous_task_indices[:, 0].tolist() == [-1, 1]
    assert batch.temporal.previous_task_available[:, 0].tolist() == [False, True]


def test_temporal_dataset_history_ignores_actions_at_and_after_decision() -> None:
    mask = torch.zeros(4, 3, dtype=torch.bool)
    mask[1] = True
    dataset = _fake_dataset(tasks_mask=mask, horizons=(1,))

    first = dataset._build_batch(
        0,
        123,
        7,
        _trajectory([[-1], [1], [1], [-1]]),
    )
    second = dataset._build_batch(
        0,
        123,
        7,
        _trajectory([[-1], [2], [-1], [2]]),
    )

    assert first.temporal is not None
    assert second.temporal is not None
    torch.testing.assert_close(
        first.temporal.previous_task_indices,
        second.temporal.previous_task_indices,
    )
    torch.testing.assert_close(
        first.temporal.run_lengths,
        second.temporal.run_lengths,
    )
    torch.testing.assert_close(
        first.temporal.switch_count_60,
        second.temporal.switch_count_60,
    )


def test_legacy_dataset_path_keeps_same_time_state_and_no_temporal_batch() -> None:
    mask = torch.ones(4, 3, dtype=torch.bool)
    dataset = _fake_dataset(tasks_mask=mask, horizons=(1,))
    dataset._include_temporal_history = False

    batch = dataset._build_batch(
        0,
        123,
        7,
        _trajectory([[-1], [1], [1], [-1]]),
    )

    assert batch.time_steps == [0, 1, 2, 3]
    assert batch.constellation_data[:, 0, -1].tolist() == [10., 20., 30., 40.]
    assert batch.temporal is None
