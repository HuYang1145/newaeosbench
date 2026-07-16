import torch

from constellation.new_transformers.temporal_history import (
    CausalAssignmentHistory,
    build_prefix_history,
    map_previous_tasks,
)


def _assert_history_equal(first, second) -> None:
    for field in first.__dataclass_fields__:
        torch.testing.assert_close(
            getattr(first, field),
            getattr(second, field),
        )


def test_prefix_history_uses_only_actions_before_decision() -> None:
    actions = torch.tensor([[-1], [4], [4], [7], [7]])

    first = build_prefix_history(actions, torch.tensor([3]))
    changed_future = actions.clone()
    changed_future[3:] = torch.tensor([[9], [9]])
    second = build_prefix_history(changed_future, torch.tensor([3]))

    assert first.previous_global_task_ids.tolist() == [[4]]
    assert first.run_lengths.tolist() == [[2]]
    assert first.switch_count_30.tolist() == [[1]]
    _assert_history_equal(first, second)


def test_prefix_history_handles_initial_decision_without_fabricated_run() -> None:
    history = build_prefix_history(
        torch.tensor([[4, -1], [4, 7]]),
        torch.tensor([0]),
    )

    assert history.previous_global_task_ids.tolist() == [[-1, -1]]
    assert history.previous_was_idle.tolist() == [[True, True]]
    assert history.run_lengths.tolist() == [[0, 0]]
    assert history.switch_count_30.tolist() == [[0, 0]]
    assert history.switch_count_60.tolist() == [[0, 0]]


def test_switch_counts_use_only_recent_prefix_windows() -> None:
    actions = torch.tensor([
        [0], [1], [0], [1], [0], [1], [0], [1], [0], [1],
    ])

    history = build_prefix_history(
        actions,
        torch.tensor([4, 10]),
        switch_windows=(3, 5),
    )

    assert history.switch_count_30.tolist() == [[2], [2]]
    assert history.switch_count_60.tolist() == [[3], [4]]


def test_previous_task_mapping_marks_disappeared_and_idle_unavailable() -> None:
    mapped, available = map_previous_tasks(
        torch.tensor([[8, -1, 3]]),
        torch.tensor([[3, 8, 10]]),
        torch.tensor([[True, False, True]]),
    )

    assert mapped.tolist() == [[-1, -1, 0]]
    assert available.tolist() == [[False, False, True]]


def test_previous_task_mapping_rejects_duplicate_available_global_ids() -> None:
    try:
        map_previous_tasks(
            torch.tensor([[3]]),
            torch.tensor([[3, 3]]),
            torch.tensor([[True, True]]),
        )
    except ValueError as error:
        assert 'duplicate' in str(error)
    else:
        raise AssertionError('duplicate candidate IDs must be rejected')


def test_online_history_snapshot_and_reset_match_prefix_semantics() -> None:
    state = CausalAssignmentHistory(num_satellites=2)

    initial = state.snapshot([4, 7, 9])
    assert initial.previous_task_indices.tolist() == [[-1, -1]]
    assert initial.run_lengths.tolist() == [[0, 0]]

    state.record([4, -1])
    state.record([4, 7])
    current = state.snapshot([4, 7, 9])

    assert current.previous_global_task_ids.tolist() == [[4, 7]]
    assert current.previous_task_indices.tolist() == [[0, 1]]
    assert current.previous_task_available.tolist() == [[True, True]]
    assert current.previous_was_idle.tolist() == [[False, False]]
    assert current.run_lengths.tolist() == [[2, 1]]
    assert current.switch_count_30.tolist() == [[0, 1]]

    state.reset()
    reset = state.snapshot([4, 7, 9])
    assert reset.previous_global_task_ids.tolist() == [[-1, -1]]
    assert reset.run_lengths.tolist() == [[0, 0]]


def test_online_history_records_each_idle_second() -> None:
    state = CausalAssignmentHistory(num_satellites=1)
    state.record([5])
    state.record([-1])
    state.record([-1])

    history = state.snapshot([5])

    assert history.previous_was_idle.tolist() == [[True]]
    assert history.run_lengths.tolist() == [[2]]
    assert history.switch_count_30.tolist() == [[1]]


def test_history_validates_shapes_and_time_bounds() -> None:
    for actions, times, message in (
        (torch.zeros(2), torch.tensor([1]), 'actions'),
        (torch.zeros(2, 1), torch.tensor([[1]]), 'time_steps'),
        (torch.zeros(2, 1), torch.tensor([3]), 'range'),
    ):
        try:
            build_prefix_history(actions, times)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError('invalid history input must be rejected')
