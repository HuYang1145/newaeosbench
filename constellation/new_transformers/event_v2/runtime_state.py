"""不依赖 Basilisk 的 V2 事件、承诺与任务映射状态机。"""

from collections.abc import Mapping, Sequence
from enum import IntEnum
from typing import Any, NamedTuple

import torch

from constellation.new_transformers.temporal_history import (
    CausalAssignmentHistory,
)

from .state import COMMITMENT_SECONDS, MAX_TASK_OWNERS, EventStateTensors
from .transition import JointEventAction


class EventType(IntEnum):
    INITIAL = 0
    EXTERNAL = 1
    COMMITMENT_EXPIRED = 2
    SAFETY_REVIEW = 3


class TerminationReason(IntEnum):
    NONE = 0
    FORCED = 1
    POLICY = 2
    TASK_CLOSED = 3


class RuntimeSnapshot(NamedTuple):
    """一个物理秒结束后的轻量事实状态。"""

    time_step: int
    ongoing_global_task_ids: tuple[int, ...]
    task_progress: torch.Tensor
    task_required_duration: torch.Tensor
    task_deadline_slack: torch.Tensor
    task_compatibility: torch.Tensor
    assignment_valid: torch.Tensor
    released_global_task_ids: tuple[int, ...]
    closed_global_task_ids: tuple[int, ...]

    def validate(self, num_satellites: int) -> None:
        if not isinstance(self.time_step, int) or self.time_step < 0:
            raise ValueError('snapshot time must be a non-negative integer')
        if (
            len(set(self.ongoing_global_task_ids))
            != len(self.ongoing_global_task_ids)
            or any(task_id < 0 for task_id in self.ongoing_global_task_ids)
        ):
            raise ValueError('ongoing global task IDs must be unique and non-negative')
        num_tasks = len(self.ongoing_global_task_ids)
        for name, value in (
            ('task progress', self.task_progress),
            ('task required duration', self.task_required_duration),
            ('task deadline slack', self.task_deadline_slack),
        ):
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != (num_tasks,)
                or not value.is_floating_point()
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f'{name} has an invalid shape or value')
        if (self.task_progress < 0).any():
            raise ValueError('task progress must be non-negative')
        if (self.task_required_duration <= 0).any():
            raise ValueError('task required duration must be positive')
        if (self.task_deadline_slack < 0).any():
            raise ValueError('task deadline slack must be non-negative')
        if (
            self.task_compatibility.shape != (num_satellites, num_tasks)
            or self.task_compatibility.dtype != torch.bool
        ):
            raise ValueError('task compatibility has an invalid shape or dtype')
        if (
            self.assignment_valid.shape != (num_satellites,)
            or self.assignment_valid.dtype != torch.bool
        ):
            raise ValueError('assignment validity has an invalid shape or dtype')
        for name, task_ids in (
            ('released', self.released_global_task_ids),
            ('closed', self.closed_global_task_ids),
        ):
            if len(set(task_ids)) != len(task_ids) or any(
                task_id < 0 for task_id in task_ids
            ):
                raise ValueError(f'{name} task IDs must be unique and non-negative')


class RuntimeEvent(NamedTuple):
    requires_policy: bool
    safety_review: bool
    state: EventStateTensors


class EventRuntimeState:
    """维护当前联合 assignment，并只在合法事件点开放策略动作。"""

    def __init__(
        self,
        *,
        num_satellites: int,
        safety_review_seconds: int = 5,
    ) -> None:
        if num_satellites <= 0:
            raise ValueError('num_satellites must be positive')
        if safety_review_seconds <= 0:
            raise ValueError('safety review interval must be positive')
        self._num_satellites = num_satellites
        self._safety_review_seconds = safety_review_seconds
        self._time_step: int | None = None
        self._last_policy_time = 0
        self._last_safety_review_time = 0
        self._current_global_task_ids = [-1] * num_satellites
        self._minimum_commitment_remaining = [0] * num_satellites
        self._last_replan_times = [0] * num_satellites
        self._termination_reasons = [
            int(TerminationReason.NONE)
        ] * num_satellites
        self._history = CausalAssignmentHistory(num_satellites)
        self._last_event_state: EventStateTensors | None = None

    @property
    def current_global_task_ids(self) -> tuple[int, ...]:
        return tuple(self._current_global_task_ids)

    @property
    def time_step(self) -> int:
        if self._time_step is None:
            raise RuntimeError('runtime state has not been initialized')
        return self._time_step

    @property
    def last_event_state(self) -> EventStateTensors:
        if self._last_event_state is None:
            raise RuntimeError('runtime has no policy event state')
        return self._last_event_state

    def _relative_current_indices(
        self,
        ongoing_global_task_ids: tuple[int, ...],
    ) -> torch.Tensor:
        relative = {
            task_id: index
            for index, task_id in enumerate(ongoing_global_task_ids)
        }
        return torch.tensor([[
            relative.get(task_id, -1)
            for task_id in self._current_global_task_ids
        ]], dtype=torch.long)

    def _owner_counts(
        self,
        ongoing_global_task_ids: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        relative = {
            task_id: index
            for index, task_id in enumerate(ongoing_global_task_ids)
        }
        owner_count = torch.zeros(len(ongoing_global_task_ids), dtype=torch.long)
        locked_owner_count = torch.zeros_like(owner_count)
        for task_id, remaining in zip(
            self._current_global_task_ids,
            self._minimum_commitment_remaining,
        ):
            task_index = relative.get(task_id)
            if task_index is None:
                continue
            owner_count[task_index] += 1
            if remaining > 0:
                locked_owner_count[task_index] += 1
        if (owner_count > MAX_TASK_OWNERS).any():
            raise RuntimeError('runtime assignment exceeds task owner capacity')
        return owner_count.unsqueeze(0), locked_owner_count.unsqueeze(0)

    def _compatible_deadline_slack(
        self,
        snapshot: RuntimeSnapshot,
    ) -> torch.Tensor:
        values = torch.full((self._num_satellites,), 1e6)
        for satellite_id in range(self._num_satellites):
            compatible = snapshot.task_compatibility[satellite_id]
            if compatible.any():
                values[satellite_id] = snapshot.task_deadline_slack[
                    compatible
                ].min()
        return values.unsqueeze(0)

    def _build_state(
        self,
        snapshot: RuntimeSnapshot,
        *,
        replan_mask: torch.Tensor,
        forced_interrupt_mask: torch.Tensor,
        can_terminate_mask: torch.Tensor,
        event_types: torch.Tensor,
        delta_t: int,
    ) -> EventStateTensors:
        history = self._history.snapshot(snapshot.ongoing_global_task_ids)
        owner_count, locked_owner_count = self._owner_counts(
            snapshot.ongoing_global_task_ids,
        )
        current_indices = self._relative_current_indices(
            snapshot.ongoing_global_task_ids,
        )
        satellite_shape = (1, self._num_satellites)
        state = EventStateTensors(
            previous_task_indices=history.previous_task_indices,
            current_task_indices=current_indices,
            minimum_commitment_remaining=torch.tensor([
                self._minimum_commitment_remaining
            ], dtype=torch.float32),
            run_lengths=history.run_lengths.to(torch.float32),
            seconds_since_replan=torch.tensor([[
                snapshot.time_step - time_step
                for time_step in self._last_replan_times
            ]], dtype=torch.float32),
            switch_count_30=history.switch_count_30.to(torch.float32),
            switch_count_60=history.switch_count_60.to(torch.float32),
            termination_reason=torch.tensor([
                self._termination_reasons
            ], dtype=torch.long),
            event_type=event_types.reshape(satellite_shape).to(torch.long),
            delta_t=torch.full(satellite_shape, float(delta_t)),
            replan_mask=replan_mask.reshape(satellite_shape),
            forced_interrupt_mask=forced_interrupt_mask.reshape(
                satellite_shape,
            ),
            can_terminate_mask=can_terminate_mask.reshape(satellite_shape),
            compatible_deadline_slack=self._compatible_deadline_slack(
                snapshot,
            ),
            task_remaining_required_seconds=(
                snapshot.task_required_duration - snapshot.task_progress
            ).clamp_min(0).unsqueeze(0),
            task_owner_count=owner_count,
            task_locked_owner_count=locked_owner_count,
        )
        state.validate()
        return state

    def initial_event(self, snapshot: RuntimeSnapshot) -> RuntimeEvent:
        if self._time_step is not None:
            raise RuntimeError('runtime state is already initialized')
        snapshot.validate(self._num_satellites)
        self._time_step = snapshot.time_step
        self._last_policy_time = snapshot.time_step
        self._last_safety_review_time = snapshot.time_step
        self._last_replan_times = [snapshot.time_step] * self._num_satellites
        replan_mask = torch.ones(self._num_satellites, dtype=torch.bool)
        state = self._build_state(
            snapshot,
            replan_mask=replan_mask,
            forced_interrupt_mask=torch.zeros_like(replan_mask),
            can_terminate_mask=torch.zeros_like(replan_mask),
            event_types=torch.full(
                (self._num_satellites,),
                int(EventType.INITIAL),
            ),
            delta_t=0,
        )
        self._last_event_state = state
        return RuntimeEvent(True, False, state)

    def apply_joint_action(
        self,
        action: JointEventAction,
        ongoing_global_task_ids: Sequence[int],
    ) -> None:
        if self._last_event_state is None or self._time_step is None:
            raise RuntimeError('a policy event is required before applying actions')
        ongoing = tuple(int(task_id) for task_id in ongoing_global_task_ids)
        num_tasks = len(ongoing)
        satellite_shape = (1, self._num_satellites)
        for name, value in action._asdict().items():
            if value.shape != satellite_shape:
                raise ValueError(f'{name} action has an invalid shape')
        if action.terminate.dtype != torch.bool:
            raise ValueError('termination action must use bool dtype')
        if (action.terminate & ~self._last_event_state.can_terminate_mask).any():
            raise ValueError('termination action is outside the policy mask')

        active = (
            self._last_event_state.replan_mask
            | self._last_event_state.forced_interrupt_mask
            | action.terminate
        )[0]
        next_assignments = list(self._current_global_task_ids)
        next_commitments = list(self._minimum_commitment_remaining)
        for satellite_id in range(self._num_satellites):
            task_index = int(action.task_indices[0, satellite_id].item())
            commitment_index = int(
                action.commitment_indices[0, satellite_id].item()
            )
            if not active[satellite_id]:
                if task_index != -1 or commitment_index != -1:
                    raise ValueError('inactive satellite received a task action')
                continue
            if task_index < -1 or task_index >= num_tasks:
                raise ValueError('task action is outside current ongoing tasks')
            if task_index == -1:
                if commitment_index != -1:
                    raise ValueError('idle action cannot have a commitment')
                next_assignments[satellite_id] = -1
                next_commitments[satellite_id] = 0
            else:
                if not 0 <= commitment_index < len(COMMITMENT_SECONDS):
                    raise ValueError('selected task needs a valid commitment')
                remaining = self._last_event_state.task_remaining_required_seconds[
                    0,
                    task_index,
                ]
                if commitment_index == 0 and remaining > 1:
                    raise ValueError('one-second commitment is physically masked')
                next_assignments[satellite_id] = ongoing[task_index]
                next_commitments[satellite_id] = COMMITMENT_SECONDS[
                    commitment_index
                ]
            self._last_replan_times[satellite_id] = self._time_step
            if self._last_event_state.forced_interrupt_mask[0, satellite_id]:
                self._termination_reasons[satellite_id] = int(
                    TerminationReason.FORCED
                )
            elif next_assignments[satellite_id] != self._current_global_task_ids[
                satellite_id
            ] or action.terminate[0, satellite_id]:
                self._termination_reasons[satellite_id] = int(
                    TerminationReason.POLICY
                )

        counts: dict[int, int] = {}
        for task_id in next_assignments:
            if task_id >= 0:
                counts[task_id] = counts.get(task_id, 0) + 1
        if any(count > MAX_TASK_OWNERS for count in counts.values()):
            raise ValueError('joint action exceeds task owner capacity')
        self._current_global_task_ids = next_assignments
        self._minimum_commitment_remaining = next_commitments

    def advance_one_second(self, snapshot: RuntimeSnapshot) -> RuntimeEvent:
        if self._time_step is None:
            raise RuntimeError('initial_event must be called first')
        snapshot.validate(self._num_satellites)
        if snapshot.time_step != self._time_step + 1:
            raise ValueError('physical time must strictly advance by one second')
        self._time_step = snapshot.time_step
        self._history.record(self._current_global_task_ids)

        commitment_before = torch.tensor(
            self._minimum_commitment_remaining,
            dtype=torch.long,
        )
        commitment_after = (commitment_before - 1).clamp_min(0)
        self._minimum_commitment_remaining = commitment_after.tolist()
        current = torch.tensor(self._current_global_task_ids, dtype=torch.long)
        ongoing = set(snapshot.ongoing_global_task_ids)
        still_ongoing = torch.tensor([
            task_id < 0 or task_id in ongoing
            for task_id in self._current_global_task_ids
        ], dtype=torch.bool)
        forced = (
            (current >= 0)
            & (~still_ongoing | ~snapshot.assignment_valid)
        )
        if forced.any():
            for satellite_id in forced.nonzero().flatten().tolist():
                self._termination_reasons[satellite_id] = int(
                    TerminationReason.FORCED
                )

        external = bool(
            snapshot.released_global_task_ids
            or snapshot.closed_global_task_ids
            or forced.any()
        )
        commitment_expired = (
            (current >= 0)
            & (commitment_before > 0)
            & (commitment_after == 0)
            & ~forced
        )
        safety_review = (
            snapshot.time_step - self._last_safety_review_time
            >= self._safety_review_seconds
        )
        if safety_review:
            self._last_safety_review_time = snapshot.time_step
        review = external or safety_review or bool(commitment_expired.any())

        idle = current < 0
        replan_mask = forced | (idle & review)
        can_terminate_mask = (
            (current >= 0)
            & (commitment_after == 0)
            & ~forced
            & review
        )
        event_types = torch.full(
            (self._num_satellites,),
            int(EventType.INITIAL),
            dtype=torch.long,
        )
        if safety_review:
            event_types.fill_(int(EventType.SAFETY_REVIEW))
        event_types[commitment_expired] = int(EventType.COMMITMENT_EXPIRED)
        if external:
            external_mask = replan_mask | can_terminate_mask
            event_types[external_mask] = int(EventType.EXTERNAL)

        requires_policy = bool(
            replan_mask.any()
            or forced.any()
            or can_terminate_mask.any()
        )
        delta_t = snapshot.time_step - self._last_policy_time
        state = self._build_state(
            snapshot,
            replan_mask=replan_mask,
            forced_interrupt_mask=forced,
            can_terminate_mask=can_terminate_mask,
            event_types=event_types,
            delta_t=delta_t,
        )
        if requires_policy:
            self._last_policy_time = snapshot.time_step
            self._last_event_state = state
        return RuntimeEvent(requires_policy, safety_review, state)

    def state_dict(self) -> dict[str, Any]:
        return {
            'version': 1,
            'num_satellites': self._num_satellites,
            'safety_review_seconds': self._safety_review_seconds,
            'time_step': self._time_step,
            'last_policy_time': self._last_policy_time,
            'last_safety_review_time': self._last_safety_review_time,
            'current_global_task_ids': tuple(self._current_global_task_ids),
            'minimum_commitment_remaining': tuple(
                self._minimum_commitment_remaining
            ),
            'last_replan_times': tuple(self._last_replan_times),
            'termination_reasons': tuple(self._termination_reasons),
            'history': self._history.state_dict(),
            'last_event_state': self._last_event_state,
        }

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, Any],
    ) -> 'EventRuntimeState':
        if state_dict.get('version') != 1:
            raise ValueError('runtime state checkpoint version does not match')
        num_satellites = state_dict.get('num_satellites')
        safety_review_seconds = state_dict.get('safety_review_seconds')
        if not isinstance(num_satellites, int) or num_satellites <= 0:
            raise ValueError('runtime checkpoint satellite count is invalid')
        if (
            not isinstance(safety_review_seconds, int)
            or safety_review_seconds <= 0
        ):
            raise ValueError('runtime checkpoint review interval is invalid')
        runtime = cls(
            num_satellites=num_satellites,
            safety_review_seconds=safety_review_seconds,
        )
        time_step = state_dict.get('time_step')
        if time_step is not None and (
            not isinstance(time_step, int) or time_step < 0
        ):
            raise ValueError('runtime checkpoint time is invalid')
        runtime._time_step = time_step
        for name, target in (
            ('last_policy_time', '_last_policy_time'),
            ('last_safety_review_time', '_last_safety_review_time'),
        ):
            value = state_dict.get(name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f'runtime checkpoint {name} is invalid')
            setattr(runtime, target, value)

        sequence_fields = (
            ('current_global_task_ids', '_current_global_task_ids', -1),
            (
                'minimum_commitment_remaining',
                '_minimum_commitment_remaining',
                0,
            ),
            ('last_replan_times', '_last_replan_times', 0),
            ('termination_reasons', '_termination_reasons', 0),
        )
        for source, target, minimum in sequence_fields:
            values = state_dict.get(source)
            if (
                not isinstance(values, (list, tuple))
                or len(values) != num_satellites
                or any(not isinstance(value, int) or value < minimum for value in values)
            ):
                raise ValueError(f'runtime checkpoint {source} is invalid')
            setattr(runtime, target, list(values))
        if any(
            value > max(COMMITMENT_SECONDS)
            for value in runtime._minimum_commitment_remaining
        ):
            raise ValueError('runtime checkpoint commitment exceeds maximum')
        if time_step is not None and any(
            value > time_step for value in runtime._last_replan_times
        ):
            raise ValueError('runtime checkpoint replan time is in the future')
        runtime._history.load_state_dict(state_dict.get('history', {}))
        last_event_state = state_dict.get('last_event_state')
        if last_event_state is not None:
            if not isinstance(last_event_state, EventStateTensors):
                raise ValueError('runtime checkpoint event state has invalid type')
            last_event_state.validate()
        runtime._last_event_state = last_event_state
        return runtime
