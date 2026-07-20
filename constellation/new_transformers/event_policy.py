"""事件发生时才接受 Actor 新决策的运行时。"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import torch

from .event_action import EventAssignmentState, EventDecision

__all__ = ['EventActorRuntime']

EventPlanner = Callable[
    [torch.Tensor, torch.Tensor],
    Sequence[EventDecision],
]


class EventActorRuntime:
    """缓存每星任务，并只替换需要重规划的卫星。"""

    def __init__(self, *, num_satellites: int) -> None:
        self.state = EventAssignmentState.empty(num_satellites)
        self.replan_count = 0

    def update(
        self,
        *,
        time: int,
        ongoing_task_ids: Iterable[int],
        planner: EventPlanner,
        force_replan_mask: torch.Tensor | None = None,
    ) -> list[int]:
        replans = self.state.advance(
            time=time,
            ongoing_task_ids=ongoing_task_ids,
        )
        if force_replan_mask is not None:
            if force_replan_mask.shape != (self.state.num_satellites,):
                raise ValueError(
                    'force_replan_mask must contain every satellite'
                )
            if force_replan_mask.dtype != torch.bool:
                raise TypeError('force_replan_mask must be boolean')
            replans = [
                should_replan or bool(force)
                for should_replan, force in zip(
                    replans,
                    force_replan_mask.tolist(),
                )
            ]
        if not any(replans):
            return self.state.assignment()

        active_commitments = ~torch.tensor(replans, dtype=torch.bool)
        decisions = list(
            planner(
                active_commitments,
                self.state.task_ids.clone(),
            )
        )
        if len(decisions) != self.state.num_satellites:
            raise ValueError('planner must return one decision per satellite')

        for satellite_index, should_replan in enumerate(replans):
            if not should_replan:
                continue
            self.state.replace(
                satellite_index,
                decisions[satellite_index],
                start_time=time,
            )
            self.replan_count += 1
        return self.state.assignment()
