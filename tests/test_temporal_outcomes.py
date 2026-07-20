import torch

from constellation.new_transformers import multi_horizon_edge_labels
from constellation.new_transformers.multi_horizon_edge_labels import (
    BatchedEdgeOutcomes,
    build_batched_edge_outcomes,
)


def _inputs(
    actions: list[list[int]],
    *,
    visible_events: tuple[tuple[int, int, int], ...] = (),
    progress: list[list[int]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    action_tensor = torch.tensor(actions, dtype=torch.long)
    time, satellites = action_tensor.shape
    visible = torch.zeros(time, satellites, 1, dtype=torch.bool)
    for event_time, satellite, task in visible_events:
        visible[event_time, satellite, task] = True
    if progress is None:
        progress = [[0] for _ in range(time)]
    return (
        action_tensor,
        visible,
        torch.tensor(progress),
        torch.tensor([3]),
    )


def test_batched_outcomes_censor_switch_without_event() -> None:
    actions, visible, progress, durations = _inputs(
        [[0], [0], [-1], [-1]],
    )

    result = build_batched_edge_outcomes(
        actions=actions,
        is_visible=visible,
        progress=progress,
        task_durations=durations,
        horizons=(1, 3),
    )

    assert isinstance(result, BatchedEdgeOutcomes)
    assert result.horizons[3].visible_observed[0, 0].item() is False
    assert result.horizons[3].progress_observed[0, 0].item() is False
    assert result.horizons[3].completion_observed[0, 0].item() is False
    assert result.horizons[1].visible_observed[0, 0].item() is True


def test_batched_outcomes_keep_positive_event_observed_before_switch() -> None:
    actions, visible, progress, durations = _inputs(
        [[0], [0], [-1], [-1]],
        visible_events=((1, 0, 0),),
    )

    result = build_batched_edge_outcomes(
        actions=actions,
        is_visible=visible,
        progress=progress,
        task_durations=durations,
        horizons=(3,),
    )

    horizon = result.horizons[3]
    assert horizon.visible[0, 0].item() is True
    assert horizon.visible_observed[0, 0].item() is True
    assert horizon.time_to_first_visible[0, 0].item() == 1


def test_batched_outcomes_record_event_time_beyond_short_horizons() -> None:
    actions, visible, progress, durations = _inputs(
        [[0] for _ in range(50)],
        visible_events=((45, 0, 0),),
    )

    result = build_batched_edge_outcomes(
        actions=actions,
        is_visible=visible,
        progress=progress,
        task_durations=durations,
        horizons=(5, 15, 30, 300),
    )

    assert result.horizons[30].visible[0, 0].item() is False
    assert result.horizons[30].visible_observed[0, 0].item() is True
    assert result.horizons[300].visible[0, 0].item() is True
    assert result.horizons[300].visible_observed[0, 0].item() is True
    assert result.horizons[300].time_to_first_visible[0, 0].item() == 45


def test_batched_outcomes_censor_long_event_time_after_early_switch() -> None:
    actions, visible, progress, durations = _inputs(
        [[0] for _ in range(20)] + [[-1] for _ in range(30)],
    )

    result = build_batched_edge_outcomes(
        actions=actions,
        is_visible=visible,
        progress=progress,
        task_durations=durations,
        horizons=(300,),
    )

    assert result.horizons[300].visible[0, 0].item() is False
    assert result.horizons[300].visible_observed[0, 0].item() is False


def test_batched_outcomes_validate_selected_task_ids() -> None:
    actions, visible, progress, durations = _inputs([[1], [1]])

    try:
        build_batched_edge_outcomes(
            actions=actions,
            is_visible=visible,
            progress=progress,
            task_durations=durations,
            horizons=(1,),
        )
    except ValueError as error:
        assert 'task ids' in str(error)
    else:
        raise AssertionError('out-of-range selected task IDs must fail')


def test_event_supervision_maps_remaining_runs_to_safe_buckets() -> None:
    time = 61
    actions = torch.empty(time, 6, dtype=torch.long)
    run_lengths = (1, 5, 15, 30, 60)
    for satellite, run_length in enumerate(run_lengths):
        actions[:run_length, satellite] = satellite
        actions[run_length:, satellite] = satellite + 10
    actions[:, -1] = -1

    result = multi_horizon_edge_labels.build_event_supervision(actions)

    assert result.valid[0].tolist() == [True] * 5 + [False]
    assert result.continue_next[0].tolist() == [
        False, True, True, True, True, False,
    ]
    assert result.duration_index[0].tolist() == [0, 1, 2, 3, 4, 0]
    assert result.duration_observed[0].tolist() == [True] * 5 + [False]
    assert result.remaining_run_lengths[0].tolist() == [
        1, 5, 15, 30, 60, 61,
    ]


def test_event_supervision_censors_short_run_at_trajectory_end() -> None:
    actions = torch.tensor([[3], [3], [3], [3], [3]])

    result = multi_horizon_edge_labels.build_event_supervision(actions)

    assert result.continue_next[:, 0].tolist() == [True] * 4
    assert result.duration_index[:, 0].tolist() == [1, 0, 0, 0]
    assert result.duration_observed[:, 0].tolist() == [False] * 4


def test_event_supervision_keeps_max_bucket_observed_at_end() -> None:
    actions = torch.zeros(61, 1, dtype=torch.long)

    result = multi_horizon_edge_labels.build_event_supervision(actions)

    assert result.duration_index[0, 0].item() == 4
    assert result.duration_observed[0, 0].item() is True
