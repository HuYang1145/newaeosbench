"""Transformer 与在线环境共用的因果动作历史定义。"""

from __future__ import annotations

import dataclasses
from collections import deque
from collections.abc import Sequence

import torch


__all__ = [
    'TemporalHistory',
    'map_previous_tasks',
    'build_prefix_history',
    'CausalAssignmentHistory',
]


@dataclasses.dataclass(frozen=True)
class TemporalHistory:
    """时间 `t` 做决策前可见的每星历史张量。"""

    previous_global_task_ids: torch.Tensor
    previous_task_indices: torch.Tensor
    previous_task_available: torch.Tensor
    previous_was_idle: torch.Tensor
    run_lengths: torch.Tensor
    switch_count_30: torch.Tensor
    switch_count_60: torch.Tensor


def _validate_candidate_ids(
    candidate_global_task_ids: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> None:
    if candidate_global_task_ids.ndim != 2:
        raise ValueError(
            'candidate_global_task_ids must have shape (batch, tasks)'
        )
    if candidate_mask.shape != candidate_global_task_ids.shape:
        raise ValueError('candidate_mask shape does not match candidate IDs')
    if (candidate_global_task_ids[candidate_mask] < 0).any():
        raise ValueError('available candidate global task IDs must be non-negative')
    for row_ids, row_mask in zip(candidate_global_task_ids, candidate_mask):
        available_ids = row_ids[row_mask]
        if available_ids.unique().numel() != available_ids.numel():
            raise ValueError('duplicate available candidate global task IDs')


def map_previous_tasks(
    previous_global_task_ids: torch.Tensor,
    candidate_global_task_ids: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把上一全局任务 ID 映射到当前候选中的相对索引。"""
    if previous_global_task_ids.ndim != 2:
        raise ValueError(
            'previous_global_task_ids must have shape (batch, satellites)'
        )
    if candidate_global_task_ids.ndim == 1:
        candidate_global_task_ids = candidate_global_task_ids.unsqueeze(0)
    if candidate_global_task_ids.shape[0] != previous_global_task_ids.shape[0]:
        raise ValueError('candidate IDs and previous tasks must share a batch')
    if candidate_mask is None:
        candidate_mask = torch.ones_like(
            candidate_global_task_ids,
            dtype=torch.bool,
        )
    else:
        candidate_mask = candidate_mask.to(dtype=torch.bool)
    _validate_candidate_ids(candidate_global_task_ids, candidate_mask)

    matches = (
        previous_global_task_ids.unsqueeze(-1)
        == candidate_global_task_ids.unsqueeze(1)
    )
    matches &= candidate_mask.unsqueeze(1)
    matches &= previous_global_task_ids.unsqueeze(-1) >= 0
    available = matches.any(-1)
    indices = matches.to(dtype=torch.long).argmax(-1)
    indices = torch.where(available, indices, indices.new_full((), -1))
    return indices, available


def _prefix_run_lengths(prefix: torch.Tensor) -> torch.Tensor:
    if prefix.shape[0] == 0:
        return torch.zeros(prefix.shape[1], dtype=torch.long)
    previous = prefix[-1]
    lengths = torch.ones(prefix.shape[1], dtype=torch.long)
    active = torch.ones(prefix.shape[1], dtype=torch.bool)
    for row in prefix[:-1].flip(0):
        active &= row == previous
        lengths += active.to(dtype=torch.long)
    return lengths


def _prefix_switch_counts(prefix: torch.Tensor, window: int) -> torch.Tensor:
    if prefix.shape[0] < 2:
        return torch.zeros(prefix.shape[1], dtype=torch.long)
    recent = prefix[-window:]
    return (recent[1:] != recent[:-1]).sum(0)


def build_prefix_history(
    actions: torch.Tensor,
    time_steps: torch.Tensor,
    *,
    candidate_global_task_ids: torch.Tensor | None = None,
    candidate_mask: torch.Tensor | None = None,
    switch_windows: tuple[int, int] = (30, 60),
) -> TemporalHistory:
    """仅用 `actions[:t]` 构造一个或多个决策时间的历史。"""
    if actions.ndim != 2:
        raise ValueError('actions must have shape (time, satellites)')
    if time_steps.ndim != 1:
        raise ValueError('time_steps must have shape (batch,)')
    if any(window <= 0 for window in switch_windows):
        raise ValueError('switch windows must be positive')
    if (time_steps < 0).any() or (time_steps > actions.shape[0]).any():
        raise ValueError('time_steps are outside the valid prefix range')

    actions = actions.to(dtype=torch.long, device='cpu')
    time_steps = time_steps.to(dtype=torch.long, device='cpu')
    previous_rows: list[torch.Tensor] = []
    run_length_rows: list[torch.Tensor] = []
    switch_30_rows: list[torch.Tensor] = []
    switch_60_rows: list[torch.Tensor] = []
    for time_step in time_steps.tolist():
        prefix = actions[:time_step]
        if time_step == 0:
            previous = torch.full(
                (actions.shape[1],),
                -1,
                dtype=torch.long,
            )
        else:
            previous = prefix[-1].clone()
        previous_rows.append(previous)
        run_length_rows.append(_prefix_run_lengths(prefix))
        switch_30_rows.append(
            _prefix_switch_counts(prefix, switch_windows[0])
        )
        switch_60_rows.append(
            _prefix_switch_counts(prefix, switch_windows[1])
        )

    batch_size = time_steps.numel()
    num_satellites = actions.shape[1]
    if batch_size:
        previous_global_task_ids = torch.stack(previous_rows)
        run_lengths = torch.stack(run_length_rows)
        switch_count_30 = torch.stack(switch_30_rows)
        switch_count_60 = torch.stack(switch_60_rows)
    else:
        shape = (0, num_satellites)
        previous_global_task_ids = torch.empty(shape, dtype=torch.long)
        run_lengths = torch.empty(shape, dtype=torch.long)
        switch_count_30 = torch.empty(shape, dtype=torch.long)
        switch_count_60 = torch.empty(shape, dtype=torch.long)

    if candidate_global_task_ids is None:
        previous_task_indices = torch.full_like(
            previous_global_task_ids,
            -1,
        )
        previous_task_available = torch.zeros_like(
            previous_global_task_ids,
            dtype=torch.bool,
        )
    else:
        candidate_global_task_ids = candidate_global_task_ids.to(
            dtype=torch.long,
            device='cpu',
        )
        if candidate_global_task_ids.ndim == 1:
            candidate_global_task_ids = candidate_global_task_ids.unsqueeze(0)
        if candidate_mask is not None:
            candidate_mask = candidate_mask.to(dtype=torch.bool, device='cpu')
        previous_task_indices, previous_task_available = map_previous_tasks(
            previous_global_task_ids,
            candidate_global_task_ids,
            candidate_mask,
        )

    return TemporalHistory(
        previous_global_task_ids=previous_global_task_ids,
        previous_task_indices=previous_task_indices,
        previous_task_available=previous_task_available,
        previous_was_idle=previous_global_task_ids == -1,
        run_lengths=run_lengths,
        switch_count_30=switch_count_30,
        switch_count_60=switch_count_60,
    )


class CausalAssignmentHistory:
    """在线环境每执行一秒调用一次的轻量历史状态机。"""

    def __init__(self, num_satellites: int) -> None:
        self._num_satellites = 0
        self.reset(num_satellites)

    def reset(self, num_satellites: int | None = None) -> None:
        if num_satellites is not None:
            if num_satellites <= 0:
                raise ValueError('num_satellites must be positive')
            self._num_satellites = num_satellites
        if self._num_satellites <= 0:
            raise ValueError('num_satellites must be initialized')
        self._assignments: deque[torch.Tensor] = deque(maxlen=61)

    def snapshot(
        self,
        candidate_global_task_ids: Sequence[int],
    ) -> TemporalHistory:
        if self._assignments:
            actions = torch.stack(tuple(self._assignments))
        else:
            actions = torch.empty(
                (0, self._num_satellites),
                dtype=torch.long,
            )
        candidate_ids = torch.as_tensor(
            list(candidate_global_task_ids),
            dtype=torch.long,
        ).unsqueeze(0)
        return build_prefix_history(
            actions,
            torch.tensor([actions.shape[0]]),
            candidate_global_task_ids=candidate_ids,
        )

    def record(self, global_task_ids: Sequence[int]) -> None:
        assignment = torch.as_tensor(global_task_ids, dtype=torch.long)
        if assignment.shape != (self._num_satellites,):
            raise ValueError('assignment must contain one task per satellite')
        if (assignment < -1).any():
            raise ValueError('global task IDs must be -1 or non-negative')
        self._assignments.append(assignment.clone())
