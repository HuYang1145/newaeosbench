from pathlib import Path

import torch

import constellation.new_transformers.multi_horizon_edge_labels as edge_labels

from constellation.new_transformers.multi_horizon_edge_labels import (
    aggregate_edge_label_summaries,
    label_executed_edge,
    summarize_trajectory_edge_labels,
)
from tools.audit_multi_horizon_edge_labels import taskset_path_for_trajectory


def _trajectory(
    actions: list[list[int]],
    *,
    visible_events: tuple[tuple[int, int, int], ...] = (),
    progress: list[list[int]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    action_tensor = torch.tensor(actions, dtype=torch.long)
    num_times, num_satellites = action_tensor.shape
    num_tasks = 1
    is_visible = torch.zeros(
        num_times, num_satellites, num_tasks, dtype=torch.bool
    )
    for time, satellite, task in visible_events:
        is_visible[time, satellite, task] = True
    if progress is None:
        progress = [[0] for _ in range(num_times)]
    return action_tensor, is_visible, torch.tensor(progress)


def test_edge_label_uses_next_step_visibility_and_censors_unknown_progress() -> None:
    actions, is_visible, progress = _trajectory(
        [[0], [-1], [-1]],
        visible_events=((1, 0, 0),),
    )

    label = label_executed_edge(
        actions=actions,
        is_visible=is_visible,
        progress=progress,
        task_durations=torch.tensor([3]),
        time=0,
        satellite=0,
        horizons=(5,),
    )

    assert label.visible_next is True
    assert label.run_length == 1
    assert label.horizons[5].visible is True
    assert label.horizons[5].visible_observed is True
    assert label.horizons[5].time_to_first_visible == 1
    assert label.horizons[5].progress_observed is False


def test_edge_label_marks_full_negative_window_as_observed() -> None:
    actions, is_visible, progress = _trajectory([[0]] * 7)

    label = label_executed_edge(
        actions=actions,
        is_visible=is_visible,
        progress=progress,
        task_durations=torch.tensor([3]),
        time=0,
        satellite=0,
        horizons=(5,),
    )

    outcome = label.horizons[5]
    assert outcome.visible is False
    assert outcome.visible_observed is True
    assert outcome.progress is False
    assert outcome.progress_observed is True
    assert outcome.completed is False
    assert outcome.completion_observed is True


def test_edge_label_censors_window_when_policy_switches_before_any_event() -> None:
    actions, is_visible, progress = _trajectory(
        [[0], [0], [-1], [-1], [-1], [-1]],
    )

    label = label_executed_edge(
        actions=actions,
        is_visible=is_visible,
        progress=progress,
        task_durations=torch.tensor([3]),
        time=0,
        satellite=0,
        horizons=(5,),
    )

    outcome = label.horizons[5]
    assert outcome.visible_observed is False
    assert outcome.progress_observed is False
    assert outcome.completion_observed is False


def test_edge_label_records_first_progress_and_completion_times() -> None:
    actions, is_visible, progress = _trajectory(
        [[0]] * 7,
        progress=[[0], [0], [1], [3], [3], [3], [3]],
    )

    label = label_executed_edge(
        actions=actions,
        is_visible=is_visible,
        progress=progress,
        task_durations=torch.tensor([3]),
        time=0,
        satellite=0,
        horizons=(5,),
    )

    outcome = label.horizons[5]
    assert outcome.progress is True
    assert outcome.time_to_first_progress == 2
    assert outcome.completed is True
    assert outcome.time_to_completion == 3


def test_edge_label_counts_duplicate_edges_per_satellite() -> None:
    actions, is_visible, progress = _trajectory(
        [[0, 0], [0, 0], [-1, -1]],
        visible_events=((1, 0, 0),),
    )

    first = label_executed_edge(
        actions=actions,
        is_visible=is_visible,
        progress=progress,
        task_durations=torch.tensor([3]),
        time=0,
        satellite=0,
        horizons=(1,),
    )
    second = label_executed_edge(
        actions=actions,
        is_visible=is_visible,
        progress=progress,
        task_durations=torch.tensor([3]),
        time=0,
        satellite=1,
        horizons=(1,),
    )

    assert first.duplicate_count == 2
    assert first.duplicate is True
    assert first.duplicate_no_visible_next is False
    assert second.duplicate_count == 2
    assert second.duplicate_no_visible_next is True


def test_trajectory_summary_stratifies_one_second_and_duplicate_edges() -> None:
    actions, is_visible, progress = _trajectory(
        [[0, 0], [-1, 0], [-1, -1]],
        visible_events=((1, 0, 0),),
    )

    summary = summarize_trajectory_edge_labels(
        actions=actions,
        is_visible=is_visible,
        progress=progress,
        task_durations=torch.tensor([3]),
        horizons=(1,),
    )

    assert summary['all']['edge_count'] == 3
    assert summary['all']['duplicate_edge_count'] == 2
    assert summary['strata']['one_second_run']['edge_count'] == 2
    assert summary['strata']['duplicate']['edge_count'] == 2


def test_aggregation_recomputes_rates_from_total_counts() -> None:
    first_actions, first_visible, first_progress = _trajectory(
        [[0], [-1]],
        visible_events=((1, 0, 0),),
    )
    second_actions, second_visible, second_progress = _trajectory(
        [[0] * 9, [-1] * 9],
    )
    first = summarize_trajectory_edge_labels(
        actions=first_actions,
        is_visible=first_visible,
        progress=first_progress,
        task_durations=torch.tensor([3]),
        horizons=(1,),
    )
    second = summarize_trajectory_edge_labels(
        actions=second_actions,
        is_visible=second_visible,
        progress=second_progress,
        task_durations=torch.tensor([3]),
        horizons=(1,),
    )

    combined = aggregate_edge_label_summaries([first, second])

    assert combined['scene_count'] == 2
    assert combined['all']['edge_count'] == 10
    assert combined['all']['rates']['visible_next_rate'] == 0.1


def test_taskset_path_preserves_split_and_scene_subdirectory() -> None:
    trajectory_root = Path('/runs/candidate_000_greedy')
    trajectory = trajectory_root / 'train/00/00007.pth'

    result = taskset_path_for_trajectory(
        trajectory,
        trajectory_root=trajectory_root,
        taskset_root=Path('/repo/data/tasksets'),
    )

    assert result == Path('/repo/data/tasksets/train/00/00007.json')


def test_trajectory_summary_does_not_relabel_every_edge(monkeypatch) -> None:
    actions, is_visible, progress = _trajectory([[0], [0], [-1]])

    def fail_per_edge(*args, **kwargs):
        raise AssertionError('trajectory summary must use the batched path')

    monkeypatch.setattr(edge_labels, 'label_executed_edge', fail_per_edge)

    summary = summarize_trajectory_edge_labels(
        actions=actions,
        is_visible=is_visible,
        progress=progress,
        task_durations=torch.tensor([3]),
        horizons=(1,),
    )

    assert summary['all']['edge_count'] == 2
