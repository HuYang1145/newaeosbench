"""星座级任务分配诊断的纯数据统计工具。"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch


def _safe_ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def map_topk_task_ids(
    actor_logits: torch.Tensor,
    *,
    ongoing_task_ids: list[list[int]],
    num_satellites: list[int],
    top_k: int,
) -> list[list[list[int]]]:
    """将策略输出中的相对任务索引映射回场景内的原始 task id。"""
    if top_k <= 0:
        raise ValueError('top_k must be positive')
    if actor_logits.shape[0] != len(ongoing_task_ids):
        raise ValueError('batch size and ongoing_task_ids must match')
    if actor_logits.shape[0] != len(num_satellites):
        raise ValueError('batch size and num_satellites must match')

    result: list[list[list[int]]] = []
    for env_index, (task_ids, ns) in enumerate(zip(
        ongoing_task_ids,
        num_satellites,
    )):
        if not task_ids:
            result.append([[] for _ in range(ns)])
            continue
        k = min(top_k, len(task_ids))
        task_logits = actor_logits[env_index, :ns, 1:1 + len(task_ids)]
        relative_ids = task_logits.topk(k, dim=-1).indices.tolist()
        result.append([
            [task_ids[relative_id] for relative_id in satellite_ids]
            for satellite_ids in relative_ids
        ])
    return result


def build_step_diagnostics(
    *,
    time_step: int,
    action: Sequence[int],
    ongoing_task_ids: list[int],
    all_task_ids: list[int],
    progress_before: torch.Tensor,
    progress_after: torch.Tensor,
    is_visible: torch.Tensor,
) -> dict[str, object]:
    """把环境的相对动作转换为紧凑、可序列化的逐步诊断信息。"""
    if progress_before.shape != progress_after.shape:
        raise ValueError('progress tensors must have the same shape')
    if progress_after.numel() != len(all_task_ids):
        raise ValueError('progress and all_task_ids must match')
    if is_visible.shape != (len(action), len(all_task_ids)):
        raise ValueError('is_visible shape does not match actions and tasks')

    assignment = [
        ongoing_task_ids[relative_id]
        if 0 <= relative_id < len(ongoing_task_ids) else -1
        for relative_id in (int(value) - 1 for value in action)
    ]
    task_index = {task_id: index for index, task_id in enumerate(all_task_ids)}
    selected_visible = [
        False if task_id < 0 else bool(
            is_visible[satellite_id, task_index[task_id]].item(),
        )
        for satellite_id, task_id in enumerate(assignment)
    ]
    made_progress = progress_after > progress_before
    progress_made_task_ids = [
        task_id
        for task_id, changed in zip(all_task_ids, made_progress.tolist())
        if changed
    ]
    return dict(
        assignment=assignment,
        selected_visible=selected_visible,
        progress_made_task_ids=progress_made_task_ids,
        ongoing_task_ids=ongoing_task_ids,
        time_step=int(time_step),
    )


@dataclass
class _ProgressRun:
    last_step: int
    satellite_ids: set[int] = field(default_factory=set)
    had_duplicate: bool = False


class SceneRecorder:
    """逐步累计一个场景的重复分配、接力和 top-k 覆盖信息。"""

    def __init__(self, *, scene_id: int, top_k: int) -> None:
        if top_k <= 0:
            raise ValueError('top_k must be positive')
        self.scene_id = scene_id
        self.top_k = top_k
        self._recorded_steps = 0
        self._active_satellite_selections = 0
        self._duplicate_group_events = 0
        self._redundant_satellite_selections = 0
        self._duplicate_progress_events = 0
        self._duplicate_stalled_events = 0
        self._duplicate_event_counts: dict[int, int] = defaultdict(int)
        self._duplicate_tasks: set[int] = set()
        self._selected_tasks: set[int] = set()
        self._topk_tasks: set[int] = set()
        self._progress_runs: dict[int, _ProgressRun] = {}
        self._relay_supported_tasks: set[int] = set()
        self._time_step_stats: dict[int, Counter[str]] = defaultdict(Counter)

    def _finish_progress_run(self, task_id: int) -> None:
        run = self._progress_runs.pop(task_id)
        if run.had_duplicate and len(run.satellite_ids) > 1:
            self._relay_supported_tasks.add(task_id)

    def record_step(
        self,
        *,
        time_step: int,
        assignment: list[int],
        topk_task_ids: list[list[int]],
        selected_visible: list[bool],
        progress_made_task_ids: list[int],
    ) -> None:
        if len(assignment) != len(selected_visible):
            raise ValueError('assignment and selected_visible must match')
        if len(assignment) != len(topk_task_ids):
            raise ValueError('assignment and topk_task_ids must match')

        active = [task_id for task_id in assignment if task_id >= 0]
        self._active_satellite_selections += len(active)
        self._selected_tasks.update(active)
        self._topk_tasks.update(
            task_id
            for satellite_topk in topk_task_ids
            for task_id in satellite_topk
            if task_id >= 0
        )

        counts = Counter(active)
        duplicate_counts = {
            task_id: count for task_id, count in counts.items() if count >= 2
        }
        time_stats = self._time_step_stats[time_step]
        time_stats['active_satellite_selections'] += len(active)
        time_stats['duplicate_group_events'] += len(duplicate_counts)
        time_stats['redundant_satellite_selections'] += sum(
            count - 1 for count in duplicate_counts.values()
        )
        progress_made = set(progress_made_task_ids)
        for task_id, count in duplicate_counts.items():
            self._duplicate_group_events += 1
            self._redundant_satellite_selections += count - 1
            self._duplicate_event_counts[task_id] += 1
            self._duplicate_tasks.add(task_id)
            if task_id in progress_made:
                self._duplicate_progress_events += 1
            else:
                self._duplicate_stalled_events += 1

        for task_id in progress_made:
            visible_satellites = {
                satellite_id
                for satellite_id, (selected_task_id, visible) in enumerate(
                    zip(assignment, selected_visible),
                )
                if selected_task_id == task_id and visible
            }
            run = self._progress_runs.get(task_id)
            if run is None or run.last_step != time_step - 1:
                if run is not None:
                    self._finish_progress_run(task_id)
                run = _ProgressRun(last_step=time_step)
                self._progress_runs[task_id] = run
            run.last_step = time_step
            run.satellite_ids.update(visible_satellites)
            run.had_duplicate |= task_id in duplicate_counts

        self._recorded_steps += 1

    def finalize(
        self,
        *,
        succeeded_task_ids: Iterable[int],
        failed_task_ids: Iterable[int],
        open_task_ids: Iterable[int],
    ) -> dict[str, object]:
        for task_id in list(self._progress_runs):
            self._finish_progress_run(task_id)

        succeeded = set(succeeded_task_ids)
        failed = set(failed_task_ids)
        open_tasks = set(open_task_ids)
        unfinished = failed | open_tasks
        duplicate_succeeded = self._duplicate_tasks & succeeded
        duplicate_failed = self._duplicate_tasks & failed
        duplicate_open = self._duplicate_tasks & open_tasks
        never_topk = unfinished - self._topk_tasks
        topk_never_selected = (
            unfinished & self._topk_tasks - self._selected_tasks
        )

        return dict(
            scene_id=self.scene_id,
            top_k=self.top_k,
            time_steps=self._recorded_steps,
            active_satellite_selections=self._active_satellite_selections,
            duplicate_group_events=self._duplicate_group_events,
            redundant_satellite_selections=(
                self._redundant_satellite_selections
            ),
            duplicate_progress_events=self._duplicate_progress_events,
            duplicate_stalled_events=self._duplicate_stalled_events,
            duplicate_tasks=len(self._duplicate_tasks),
            duplicate_tasks_succeeded=len(duplicate_succeeded),
            duplicate_tasks_failed=len(duplicate_failed),
            duplicate_tasks_open=len(duplicate_open),
            duplicate_success_events=sum(
                self._duplicate_event_counts[task_id]
                for task_id in duplicate_succeeded
            ),
            duplicate_failed_events=sum(
                self._duplicate_event_counts[task_id]
                for task_id in duplicate_failed
            ),
            duplicate_open_events=sum(
                self._duplicate_event_counts[task_id]
                for task_id in duplicate_open
            ),
            relay_supported_tasks=len(self._relay_supported_tasks),
            unfinished_tasks=len(unfinished),
            unfinished_never_topk=len(never_topk),
            unfinished_never_topk_ids=sorted(never_topk),
            unfinished_ever_topk_never_selected=len(topk_never_selected),
            unfinished_ever_topk_never_selected_ids=sorted(
                topk_never_selected,
            ),
            time_step_stats={
                str(time_step): dict(stats)
                for time_step, stats in sorted(self._time_step_stats.items())
            },
        )


def summarize_scene_results(
    scene_results: list[dict[str, object]],
) -> dict[str, float | int]:
    """汇总多个场景，并给出用于判断主要瓶颈的比例。"""
    sum_keys = (
        'active_satellite_selections',
        'duplicate_group_events',
        'redundant_satellite_selections',
        'duplicate_progress_events',
        'duplicate_stalled_events',
        'duplicate_tasks',
        'duplicate_tasks_succeeded',
        'duplicate_tasks_failed',
        'duplicate_tasks_open',
        'relay_supported_tasks',
        'unfinished_tasks',
        'unfinished_never_topk',
        'unfinished_ever_topk_never_selected',
    )
    totals = {
        key: sum(int(result[key]) for result in scene_results)
        for key in sum_keys
    }
    duplicate_events = totals['duplicate_group_events']
    duplicate_tasks = totals['duplicate_tasks']
    unfinished_tasks = totals['unfinished_tasks']
    time_step_totals: dict[int, Counter[str]] = defaultdict(Counter)
    for result in scene_results:
        for time_step, stats in dict(result['time_step_stats']).items():
            time_step_totals[int(time_step)].update(stats)
    return dict(
        scene_count=len(scene_results),
        **totals,
        duplicate_selection_rate=_safe_ratio(
            totals['redundant_satellite_selections'],
            totals['active_satellite_selections'],
        ),
        duplicate_progress_rate=_safe_ratio(
            totals['duplicate_progress_events'],
            duplicate_events,
        ),
        duplicate_task_success_rate=_safe_ratio(
            totals['duplicate_tasks_succeeded'],
            duplicate_tasks,
        ),
        relay_support_rate=_safe_ratio(
            totals['relay_supported_tasks'],
            duplicate_tasks,
        ),
        unfinished_never_topk_rate=_safe_ratio(
            totals['unfinished_never_topk'],
            unfinished_tasks,
        ),
        unfinished_ever_topk_never_selected_rate=_safe_ratio(
            totals['unfinished_ever_topk_never_selected'],
            unfinished_tasks,
        ),
        time_step_stats={
            str(time_step): {
                **dict(stats),
                'duplicate_selection_rate': _safe_ratio(
                    stats['redundant_satellite_selections'],
                    stats['active_satellite_selections'],
                ),
            }
            for time_step, stats in sorted(time_step_totals.items())
        },
    )
