"""从既有轨迹构造多时间尺度卫星—任务边结果标签。"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import torch


@dataclasses.dataclass(frozen=True)
class HorizonEdgeOutcome:
    visible: bool
    visible_observed: bool
    progress: bool
    progress_observed: bool
    completed: bool
    completion_observed: bool
    time_to_first_visible: int | None
    time_to_first_progress: int | None
    time_to_completion: int | None


@dataclasses.dataclass(frozen=True)
class ExecutedEdgeLabel:
    time: int
    satellite: int
    task_id: int
    run_length: int
    duplicate_count: int
    duplicate: bool
    visible_next: bool
    progress_next: bool
    completed_next: bool
    duplicate_no_visible_next: bool
    horizons: dict[int, HorizonEdgeOutcome]


@dataclasses.dataclass(frozen=True)
class HorizonOutcomeTensors:
    """同一 horizon 下所有决策边的事实结果和 observed mask。"""

    visible: torch.Tensor
    visible_observed: torch.Tensor
    progress: torch.Tensor
    progress_observed: torch.Tensor
    completed: torch.Tensor
    completion_observed: torch.Tensor
    time_to_first_visible: torch.Tensor
    time_to_first_progress: torch.Tensor
    time_to_completion: torch.Tensor


@dataclasses.dataclass(frozen=True)
class BatchedEdgeOutcomes:
    """轨迹中 `0..T-2` 每颗卫星实际执行边的批量结果。"""

    valid: torch.Tensor
    run_lengths: torch.Tensor
    duplicate_count: torch.Tensor
    visible_next: torch.Tensor
    progress_next: torch.Tensor
    completed_next: torch.Tensor
    horizons: dict[int, HorizonOutcomeTensors]


@dataclasses.dataclass(frozen=True)
class EventSupervisionTensors:
    """轨迹中 `0..T-2` 每颗卫星的事件持续行为标签。"""

    valid: torch.Tensor
    continue_next: torch.Tensor
    duration_index: torch.Tensor
    duration_observed: torch.Tensor
    remaining_run_lengths: torch.Tensor


def _validate_inputs(
    actions: torch.Tensor,
    is_visible: torch.Tensor,
    progress: torch.Tensor,
    task_durations: torch.Tensor,
) -> None:
    if actions.ndim != 2:
        raise ValueError('actions must have shape (time, satellites)')
    if is_visible.ndim != 3:
        raise ValueError(
            'is_visible must have shape (time, satellites, tasks)'
        )
    if progress.ndim != 2:
        raise ValueError('progress must have shape (time, tasks)')
    if actions.shape != is_visible.shape[:2]:
        raise ValueError('actions and is_visible shapes do not align')
    if actions.shape[0] != progress.shape[0]:
        raise ValueError('actions and progress time dimensions do not align')
    if progress.shape[1] != is_visible.shape[2]:
        raise ValueError('progress and is_visible task dimensions do not align')
    if task_durations.shape != (progress.shape[1],):
        raise ValueError('task_durations must contain one value per task')


def _first_true_offset(values: torch.Tensor) -> int | None:
    indices = values.nonzero().flatten()
    return None if indices.numel() == 0 else int(indices[0]) + 1


def label_executed_edge(
    *,
    actions: torch.Tensor,
    is_visible: torch.Tensor,
    progress: torch.Tensor,
    task_durations: torch.Tensor,
    time: int,
    satellite: int,
    horizons: Sequence[int] = (5, 15, 30),
) -> ExecutedEdgeLabel:
    """标注一个真实执行的非空边，提前切换的未知结果保持 censored。"""
    _validate_inputs(actions, is_visible, progress, task_durations)
    if not 0 <= time < actions.shape[0] - 1:
        raise ValueError('time must have a next outcome step')
    if not 0 <= satellite < actions.shape[1]:
        raise ValueError('satellite index is out of range')
    normalized_horizons = tuple(int(value) for value in horizons)
    if not normalized_horizons or any(value <= 0 for value in normalized_horizons):
        raise ValueError('horizons must be positive')
    if len(set(normalized_horizons)) != len(normalized_horizons):
        raise ValueError('horizons must be unique')

    task_id = int(actions[time, satellite])
    if not 0 <= task_id < progress.shape[1]:
        raise ValueError('the selected edge must reference a valid task')

    run_end = time + 1
    while (
        run_end < actions.shape[0]
        and int(actions[run_end, satellite]) == task_id
    ):
        run_end += 1
    run_length = run_end - time
    duplicate_count = int((actions[time] == task_id).sum())
    baseline_progress = progress[time, task_id]
    duration = task_durations[task_id]
    visible_next = bool(is_visible[time + 1, satellite, task_id])
    progress_next = bool(progress[time + 1, task_id] > baseline_progress)
    completed_next = bool(
        baseline_progress < duration
        and progress[time + 1, task_id] >= duration
    )

    outcomes: dict[int, HorizonEdgeOutcome] = {}
    for horizon in normalized_horizons:
        outcome_end = min(time + horizon, run_end, actions.shape[0] - 1)
        full_window = time + horizon <= min(run_end, actions.shape[0] - 1)
        visible_values = is_visible[
            time + 1:outcome_end + 1,
            satellite,
            task_id,
        ]
        future_progress = progress[time + 1:outcome_end + 1, task_id]
        progress_values = future_progress > baseline_progress
        completion_values = (
            (baseline_progress < duration)
            & (future_progress >= duration)
        )
        time_to_visible = _first_true_offset(visible_values)
        time_to_progress = _first_true_offset(progress_values)
        time_to_completion = _first_true_offset(completion_values)
        outcomes[horizon] = HorizonEdgeOutcome(
            visible=time_to_visible is not None,
            visible_observed=time_to_visible is not None or full_window,
            progress=time_to_progress is not None,
            progress_observed=time_to_progress is not None or full_window,
            completed=time_to_completion is not None,
            completion_observed=(
                time_to_completion is not None or full_window
            ),
            time_to_first_visible=time_to_visible,
            time_to_first_progress=time_to_progress,
            time_to_completion=time_to_completion,
        )

    duplicate = duplicate_count > 1
    return ExecutedEdgeLabel(
        time=time,
        satellite=satellite,
        task_id=task_id,
        run_length=run_length,
        duplicate_count=duplicate_count,
        duplicate=duplicate,
        visible_next=visible_next,
        progress_next=progress_next,
        completed_next=completed_next,
        duplicate_no_visible_next=duplicate and not visible_next,
        horizons=outcomes,
    )


_EDGE_COUNT_KEYS = (
    'edge_count',
    'visible_next_count',
    'progress_next_count',
    'completed_next_count',
    'duplicate_edge_count',
    'duplicate_no_visible_next_count',
)

_HORIZON_COUNT_KEYS = (
    'visible_observed_count',
    'visible_positive_count',
    'progress_observed_count',
    'progress_positive_count',
    'completion_observed_count',
    'completion_positive_count',
    'time_to_first_visible_sum',
    'time_to_first_visible_count',
    'time_to_first_progress_sum',
    'time_to_first_progress_count',
    'time_to_completion_sum',
    'time_to_completion_count',
)


def _empty_counter(horizons: Sequence[int]) -> dict[str, Any]:
    counter: dict[str, Any] = {key: 0 for key in _EDGE_COUNT_KEYS}
    counter['horizons'] = {
        str(horizon): {key: 0 for key in _HORIZON_COUNT_KEYS}
        for horizon in horizons
    }
    return counter


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _with_rates(counter: dict[str, Any]) -> dict[str, Any]:
    output = {
        key: value
        for key, value in counter.items()
        if key not in {'rates', 'horizons'}
    }
    edge_count = int(counter['edge_count'])
    duplicate_count = int(counter['duplicate_edge_count'])
    output['rates'] = {
        'visible_next_rate': _ratio(
            int(counter['visible_next_count']), edge_count
        ),
        'progress_next_rate': _ratio(
            int(counter['progress_next_count']), edge_count
        ),
        'completed_next_rate': _ratio(
            int(counter['completed_next_count']), edge_count
        ),
        'duplicate_edge_rate': _ratio(duplicate_count, edge_count),
        'duplicate_no_visible_fraction': _ratio(
            int(counter['duplicate_no_visible_next_count']),
            duplicate_count,
        ),
    }
    output['horizons'] = {}
    for horizon, values in counter['horizons'].items():
        horizon_output = dict(values)
        horizon_output['rates'] = {
            'visible_positive_rate': _ratio(
                values['visible_positive_count'],
                values['visible_observed_count'],
            ),
            'visible_censored_rate': _ratio(
                edge_count - values['visible_observed_count'], edge_count
            ),
            'progress_positive_rate': _ratio(
                values['progress_positive_count'],
                values['progress_observed_count'],
            ),
            'progress_censored_rate': _ratio(
                edge_count - values['progress_observed_count'], edge_count
            ),
            'completion_positive_rate': _ratio(
                values['completion_positive_count'],
                values['completion_observed_count'],
            ),
            'completion_censored_rate': _ratio(
                edge_count - values['completion_observed_count'], edge_count
            ),
            'mean_time_to_first_visible': _ratio(
                values['time_to_first_visible_sum'],
                values['time_to_first_visible_count'],
            ),
            'mean_time_to_first_progress': _ratio(
                values['time_to_first_progress_sum'],
                values['time_to_first_progress_count'],
            ),
            'mean_time_to_completion': _ratio(
                values['time_to_completion_sum'],
                values['time_to_completion_count'],
            ),
        }
        output['horizons'][horizon] = horizon_output
    return output


def _run_lengths(actions: torch.Tensor) -> torch.Tensor:
    lengths = torch.ones_like(actions)
    for time in range(actions.shape[0] - 2, -1, -1):
        lengths[time] = torch.where(
            actions[time] == actions[time + 1],
            lengths[time + 1] + 1,
            1,
        )
    return lengths


def build_event_supervision(
    actions: torch.Tensor,
    commitments: Sequence[int] = (1, 5, 15, 30, 60),
) -> EventSupervisionTensors:
    """把真实非空动作连续段转换为保守的事件持续标签。"""
    if actions.ndim != 2:
        raise ValueError('actions must have shape (time, satellites)')
    if actions.shape[0] < 2:
        raise ValueError('actions must contain at least two time steps')
    normalized = tuple(int(value) for value in commitments)
    if (
        not normalized
        or normalized[0] != 1
        or any(value <= 0 for value in normalized)
        or any(a >= b for a, b in zip(normalized, normalized[1:]))
    ):
        raise ValueError(
            'commitments must be strictly increasing and start at one'
        )

    actions = actions.to(dtype=torch.long)
    selected = actions[:-1]
    valid = selected >= 0
    remaining = _run_lengths(actions)[:-1]
    commitment_tensor = torch.tensor(
        normalized,
        dtype=remaining.dtype,
        device=remaining.device,
    )
    duration_index = (
        remaining.unsqueeze(-1) >= commitment_tensor
    ).sum(-1).sub(1).clamp_min(0)
    duration_index = torch.where(
        valid,
        duration_index,
        torch.zeros_like(duration_index),
    )

    row = torch.arange(
        selected.shape[0],
        device=remaining.device,
    ).unsqueeze(-1)
    reaches_trajectory_end = row + remaining >= actions.shape[0]
    duration_observed = valid & (
        ~reaches_trajectory_end
        | (remaining >= normalized[-1])
    )
    continue_next = valid & (actions[1:] == selected)
    return EventSupervisionTensors(
        valid=valid,
        continue_next=continue_next,
        duration_index=duration_index,
        duration_observed=duration_observed,
        remaining_run_lengths=remaining,
    )


def build_batched_edge_outcomes(
    *,
    actions: torch.Tensor,
    is_visible: torch.Tensor,
    progress: torch.Tensor,
    task_durations: torch.Tensor,
    horizons: Sequence[int],
) -> BatchedEdgeOutcomes:
    """构造可直接用于训练的批量事实结果，未知窗口保持 censored。"""
    _validate_inputs(actions, is_visible, progress, task_durations)
    normalized_horizons = tuple(int(value) for value in horizons)
    if not normalized_horizons or any(value <= 0 for value in normalized_horizons):
        raise ValueError('horizons must be positive')
    if len(set(normalized_horizons)) != len(normalized_horizons):
        raise ValueError('horizons must be unique')
    selected_task_ids = actions[actions >= 0]
    if (
        selected_task_ids.numel()
        and int(selected_task_ids.max()) >= progress.shape[1]
    ):
        raise ValueError('action task ids do not align with trajectory tasks')

    num_times, num_satellites = actions.shape
    num_edges = num_times - 1
    selected = actions[:-1]
    task_indices = selected.clamp_min(0)
    valid = selected >= 0
    full_run_lengths = _run_lengths(actions)
    run_lengths = full_run_lengths[:-1]
    duplicate_count = (
        selected.unsqueeze(-1) == selected.unsqueeze(-2)
    ).sum(-1)
    visible_next = torch.gather(
        is_visible[1:], 2, task_indices.unsqueeze(-1)
    ).squeeze(-1) & valid
    baseline_progress = torch.gather(progress[:-1], 1, task_indices)
    next_progress = torch.gather(progress[1:], 1, task_indices)
    durations = task_durations[task_indices]
    progress_next = (next_progress > baseline_progress) & valid
    completed_next = (
        (baseline_progress < durations)
        & (next_progress >= durations)
        & valid
    )

    horizon_values = {}
    for horizon in normalized_horizons:
        first_visible = torch.zeros_like(selected)
        first_progress = torch.zeros_like(selected)
        first_completion = torch.zeros_like(selected)
        for offset in range(1, horizon + 1):
            available = num_times - offset
            if available <= 0:
                break
            edge_actions = actions[:available]
            edge_tasks = edge_actions.clamp_min(0)
            edge_valid = edge_actions >= 0
            continued = full_run_lengths[:available] >= offset
            eligible = edge_valid & continued
            visible = torch.gather(
                is_visible[offset:], 2, edge_tasks.unsqueeze(-1)
            ).squeeze(-1) & eligible
            base_progress = torch.gather(
                progress[:available], 1, edge_tasks
            )
            future_progress = torch.gather(
                progress[offset:], 1, edge_tasks
            )
            edge_durations = task_durations[edge_tasks]
            progressed = (future_progress > base_progress) & eligible
            completed = (
                (base_progress < edge_durations)
                & (future_progress >= edge_durations)
                & eligible
            )
            for target, event in (
                (first_visible, visible),
                (first_progress, progressed),
                (first_completion, completed),
            ):
                current = target[:available]
                target[:available] = torch.where(
                    (current == 0) & event,
                    offset,
                    current,
                )

        row = torch.arange(num_edges).unsqueeze(-1)
        full_window = (
            (run_lengths >= horizon)
            & (row + horizon < num_times)
            & valid
        )
        visible_positive = first_visible > 0
        progress_positive = first_progress > 0
        completion_positive = first_completion > 0
        horizon_values[horizon] = HorizonOutcomeTensors(
            visible=visible_positive,
            visible_observed=visible_positive | full_window,
            progress=progress_positive,
            progress_observed=progress_positive | full_window,
            completed=completion_positive,
            completion_observed=completion_positive | full_window,
            time_to_first_visible=first_visible,
            time_to_first_progress=first_progress,
            time_to_completion=first_completion,
        )
    return BatchedEdgeOutcomes(
        valid=valid,
        run_lengths=run_lengths,
        duplicate_count=duplicate_count,
        visible_next=visible_next,
        progress_next=progress_next,
        completed_next=completed_next,
        horizons=horizon_values,
    )


def _counter_from_batched(
    batched: BatchedEdgeOutcomes,
    *,
    mask: torch.Tensor,
    horizons: Sequence[int],
) -> dict[str, Any]:
    counter = _empty_counter(horizons)
    duplicate = batched.duplicate_count > 1
    counter['edge_count'] = int(mask.sum())
    counter['visible_next_count'] = int(
        (batched.visible_next & mask).sum()
    )
    counter['progress_next_count'] = int(
        (batched.progress_next & mask).sum()
    )
    counter['completed_next_count'] = int(
        (batched.completed_next & mask).sum()
    )
    counter['duplicate_edge_count'] = int((duplicate & mask).sum())
    counter['duplicate_no_visible_next_count'] = int(
        (duplicate & ~batched.visible_next & mask).sum()
    )
    for horizon in horizons:
        source = batched.horizons[horizon]
        target = counter['horizons'][str(horizon)]
        for outcome, observed_name, positive_name in (
            ('visible', 'visible_observed', 'visible'),
            ('progress', 'progress_observed', 'progress'),
            ('completion', 'completion_observed', 'completed'),
        ):
            target[f'{outcome}_observed_count'] = int(
                (getattr(source, observed_name) & mask).sum()
            )
            target[f'{outcome}_positive_count'] = int(
                (getattr(source, positive_name) & mask).sum()
            )
        for name in (
            'time_to_first_visible',
            'time_to_first_progress',
            'time_to_completion',
        ):
            values = getattr(source, name)
            positive = (values > 0) & mask
            target[f'{name}_sum'] = int(values[positive].sum())
            target[f'{name}_count'] = int(positive.sum())
    return counter


def summarize_trajectory_edge_labels(
    *,
    actions: torch.Tensor,
    is_visible: torch.Tensor,
    progress: torch.Tensor,
    task_durations: torch.Tensor,
    horizons: Sequence[int] = (5, 15, 30),
) -> dict[str, Any]:
    """汇总一条轨迹中的非空执行边及其短期物理结果。"""
    _validate_inputs(actions, is_visible, progress, task_durations)
    normalized_horizons = tuple(int(value) for value in horizons)
    selected = actions[actions >= 0]
    if selected.numel() and int(selected.max()) >= progress.shape[1]:
        raise ValueError('action task ids do not align with trajectory tasks')
    batched = build_batched_edge_outcomes(
        actions=actions,
        is_visible=is_visible,
        progress=progress,
        task_durations=task_durations,
        horizons=normalized_horizons,
    )
    valid = batched.valid
    one_second = valid & (batched.run_lengths == 1)
    duplicate = valid & (batched.duplicate_count > 1)
    masks = {
        'all': valid,
        'one_second_run': one_second,
        'duplicate': duplicate,
        'other': valid & ~one_second & ~duplicate,
    }
    counters = {
        name: _counter_from_batched(
            batched,
            mask=mask,
            horizons=normalized_horizons,
        )
        for name, mask in masks.items()
    }
    return {
        'horizons': list(normalized_horizons),
        'all': _with_rates(counters['all']),
        'strata': {
            name: _with_rates(counters[name])
            for name in ('one_second_run', 'duplicate', 'other')
        },
    }


def _add_counter(
    target: dict[str, Any],
    source: dict[str, Any],
) -> None:
    for key in _EDGE_COUNT_KEYS:
        target[key] += int(source[key])
    for horizon, values in target['horizons'].items():
        source_values = source['horizons'][horizon]
        for key in _HORIZON_COUNT_KEYS:
            values[key] += int(source_values[key])


def aggregate_edge_label_summaries(
    summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """按原始计数聚合场景，并重新计算所有比例。"""
    if not summaries:
        raise ValueError('at least one summary is required')
    horizons = tuple(int(value) for value in summaries[0]['horizons'])
    counters = {
        'all': _empty_counter(horizons),
        'one_second_run': _empty_counter(horizons),
        'duplicate': _empty_counter(horizons),
        'other': _empty_counter(horizons),
    }
    for summary in summaries:
        if tuple(int(value) for value in summary['horizons']) != horizons:
            raise ValueError('all summaries must use the same horizons')
        _add_counter(counters['all'], summary['all'])
        for name in ('one_second_run', 'duplicate', 'other'):
            _add_counter(counters[name], summary['strata'][name])
    return {
        'scene_count': len(summaries),
        'horizons': list(horizons),
        'all': _with_rates(counters['all']),
        'strata': {
            name: _with_rates(counters[name])
            for name in ('one_second_run', 'duplicate', 'other')
        },
    }
