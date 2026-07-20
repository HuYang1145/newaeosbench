"""事件式 Actor 的每星任务承诺状态。

该模块只维护任务 ID 和时间，不调用 Basilisk、轨道传播或模型推理。
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence
from typing import NamedTuple

import torch

ALLOWED_EVENT_COMMITMENTS = (1, 5, 15, 30, 60)

__all__ = [
    'ALLOWED_EVENT_COMMITMENTS',
    'EventAssignmentState',
    'EventDecision',
    'LearnedEventCommitments',
    'select_learned_event_commitments',
]


class LearnedEventCommitments(NamedTuple):
    commitment_seconds: torch.Tensor
    duration_proposals: torch.Tensor
    continue_probabilities: torch.Tensor
    task_selected: torch.Tensor


def select_learned_event_commitments(
    *,
    relative_task_ids: torch.Tensor,
    continue_logits: torch.Tensor,
    duration_logits: torch.Tensor,
    continue_threshold: float,
) -> LearnedEventCommitments:
    """为 Actor 已选中的任务读取 M2 终止和持续时间预测。"""
    if not 0 < continue_threshold < 1:
        raise ValueError('continue threshold must be in (0, 1)')
    if relative_task_ids.ndim != 1:
        raise ValueError('relative task ids must be one-dimensional')
    if continue_logits.ndim != 2:
        raise ValueError('continue logits must have shape satellites/tasks')
    if duration_logits.ndim != 3 or duration_logits.shape[-1] != len(
        ALLOWED_EVENT_COMMITMENTS
    ):
        raise ValueError(
            'duration logits must have five commitment classes'
        )
    if (
        continue_logits.shape != duration_logits.shape[:2]
        or continue_logits.shape[0] != relative_task_ids.numel()
    ):
        raise ValueError('event logits and selected tasks must align')
    task_selected = relative_task_ids >= 0
    selected = relative_task_ids[task_selected]
    if selected.numel() and int(selected.max()) >= continue_logits.shape[1]:
        raise ValueError('relative task id exceeds event logits')
    if (relative_task_ids < -1).any():
        raise ValueError('relative task ids must be -1 or non-negative')

    indices = relative_task_ids.clamp_min(0)
    row_indices = torch.arange(
        relative_task_ids.numel(),
        device=relative_task_ids.device,
    )
    selected_continue_logits = continue_logits[row_indices, indices]
    selected_duration_logits = duration_logits[row_indices, indices]
    continue_probabilities = selected_continue_logits.sigmoid()
    commitments = torch.tensor(
        ALLOWED_EVENT_COMMITMENTS,
        dtype=torch.long,
        device=duration_logits.device,
    )
    duration_proposals = commitments[
        selected_duration_logits.argmax(-1)
    ]
    ones = torch.ones_like(duration_proposals)
    duration_proposals = torch.where(
        task_selected,
        duration_proposals,
        ones,
    )
    commitment_seconds = torch.where(
        task_selected & (continue_probabilities >= continue_threshold),
        duration_proposals,
        ones,
    )
    continue_probabilities = torch.where(
        task_selected,
        continue_probabilities,
        torch.zeros_like(continue_probabilities),
    )
    return LearnedEventCommitments(
        commitment_seconds=commitment_seconds,
        duration_proposals=duration_proposals,
        continue_probabilities=continue_probabilities,
        task_selected=task_selected,
    )


@dataclasses.dataclass(frozen=True)
class EventDecision:
    """一颗卫星在下一事件前需要保持的任务。"""

    task_id: int
    commitment_seconds: int

    def __post_init__(self) -> None:
        if self.commitment_seconds not in ALLOWED_EVENT_COMMITMENTS:
            raise ValueError(
                'commitment_seconds must be one of '
                f'{ALLOWED_EVENT_COMMITMENTS}'
            )
        if self.task_id < -1:
            raise ValueError('task_id must be -1 or a non-negative id')


@dataclasses.dataclass
class EventAssignmentState:
    """维护同一星座中每颗卫星的事件承诺。"""

    task_ids: torch.Tensor
    remaining_seconds: torch.Tensor
    start_times: torch.Tensor
    last_update_times: torch.Tensor
    interruption_reasons: list[str | None]

    @classmethod
    def empty(cls, num_satellites: int) -> 'EventAssignmentState':
        if num_satellites <= 0:
            raise ValueError('num_satellites must be positive')
        return cls(
            task_ids=torch.full((num_satellites,), -1, dtype=torch.long),
            remaining_seconds=torch.zeros(
                num_satellites,
                dtype=torch.long,
            ),
            start_times=torch.full(
                (num_satellites,),
                -1,
                dtype=torch.long,
            ),
            last_update_times=torch.full(
                (num_satellites,),
                -1,
                dtype=torch.long,
            ),
            interruption_reasons=[None] * num_satellites,
        )

    @property
    def num_satellites(self) -> int:
        return int(self.task_ids.numel())

    def assignment(self) -> list[int]:
        return [int(value) for value in self.task_ids.tolist()]

    def start(
        self,
        decisions: Sequence[EventDecision],
        *,
        start_time: int,
    ) -> None:
        if len(decisions) != self.num_satellites:
            raise ValueError('one decision is required per satellite')
        for satellite_index, decision in enumerate(decisions):
            self.replace(
                satellite_index,
                decision,
                start_time=start_time,
            )

    def replace(
        self,
        satellite_index: int,
        decision: EventDecision,
        *,
        start_time: int,
    ) -> None:
        if not 0 <= satellite_index < self.num_satellites:
            raise IndexError('satellite_index is out of range')
        self.task_ids[satellite_index] = decision.task_id
        self.remaining_seconds[
            satellite_index
        ] = decision.commitment_seconds
        self.start_times[satellite_index] = start_time
        self.last_update_times[satellite_index] = start_time
        self.interruption_reasons[satellite_index] = None

    def advance(
        self,
        *,
        time: int,
        ongoing_task_ids: Iterable[int],
    ) -> list[bool]:
        """推进到当前物理时刻并返回每颗卫星是否需要重规划。"""
        ongoing = {int(task_id) for task_id in ongoing_task_ids}
        replans: list[bool] = []
        for satellite_index, task_id in enumerate(self.assignment()):
            last_update = int(self.last_update_times[satellite_index])
            if last_update < 0:
                replans.append(True)
                continue
            if time < last_update:
                raise ValueError('time must be monotonic')

            elapsed = time - last_update
            remaining = max(
                int(self.remaining_seconds[satellite_index]) - elapsed,
                0,
            )
            self.remaining_seconds[satellite_index] = remaining
            self.last_update_times[satellite_index] = time

            task_unavailable = task_id >= 0 and task_id not in ongoing
            if task_unavailable:
                self.remaining_seconds[satellite_index] = 0
                self.interruption_reasons[
                    satellite_index
                ] = 'task_unavailable'
                replans.append(True)
            elif remaining == 0:
                self.interruption_reasons[satellite_index] = 'expired'
                replans.append(True)
            else:
                replans.append(False)
        return replans
