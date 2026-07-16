import torch

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
