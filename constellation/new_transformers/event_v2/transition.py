"""V2 联合动作、行为 trace 与事件 transition 的稳定 schema。"""

import copy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, NamedTuple

import torch

from .state import (
    COMMITMENT_SECONDS,
    MAX_TASK_OWNERS,
    EventStateTensors,
)


TRANSITION_SCHEMA_VERSION = 1


class JointEventAction(NamedTuple):
    terminate: torch.Tensor
    task_indices: torch.Tensor
    commitment_indices: torch.Tensor


class ActionTrace(NamedTuple):
    action_order: torch.Tensor
    termination_mask: torch.Tensor
    task_masks: torch.Tensor
    commitment_masks: torch.Tensor
    owner_state: torch.Tensor


_SCHEMA_DEFINITION: dict[str, Any] = {
    'version': TRANSITION_SCHEMA_VERSION,
    'event_state_fields': [
        {'name': 'previous_task_indices', 'dtype': 'int64'},
        {'name': 'current_task_indices', 'dtype': 'int64'},
        {'name': 'minimum_commitment_remaining', 'dtype': 'float32'},
        {'name': 'run_lengths', 'dtype': 'float32'},
        {'name': 'seconds_since_replan', 'dtype': 'float32'},
        {'name': 'switch_count_30', 'dtype': 'float32'},
        {'name': 'switch_count_60', 'dtype': 'float32'},
        {'name': 'termination_reason', 'dtype': 'int64'},
        {'name': 'event_type', 'dtype': 'int64'},
        {'name': 'delta_t', 'dtype': 'float32'},
        {'name': 'replan_mask', 'dtype': 'bool'},
        {'name': 'forced_interrupt_mask', 'dtype': 'bool'},
        {'name': 'can_terminate_mask', 'dtype': 'bool'},
        {'name': 'compatible_deadline_slack', 'dtype': 'float32'},
        {'name': 'task_remaining_required_seconds', 'dtype': 'float32'},
        {'name': 'task_owner_count', 'dtype': 'int64'},
        {'name': 'task_locked_owner_count', 'dtype': 'int64'},
    ],
    'joint_action_fields': [
        {'name': 'terminate', 'dtype': 'bool'},
        {'name': 'task_indices', 'dtype': 'int64'},
        {'name': 'commitment_indices', 'dtype': 'int64'},
    ],
    'trace_fields': [
        {'name': 'action_order', 'dtype': 'int64'},
        {'name': 'termination_mask', 'dtype': 'bool'},
        {'name': 'task_masks', 'dtype': 'bool'},
        {'name': 'commitment_masks', 'dtype': 'bool'},
        {'name': 'owner_state', 'dtype': 'int64'},
    ],
    'transition_fields': [
        {'name': 'state', 'type': 'EventStateTensors'},
        {'name': 'joint_action', 'type': 'JointEventAction'},
        {'name': 'behavior_log_prob', 'dtype': 'float32'},
        {'name': 'value', 'dtype': 'float32'},
        {'name': 'reward', 'dtype': 'float32'},
        {'name': 'delta_t', 'dtype': 'float32'},
        {'name': 'next_state', 'type': 'EventStateTensors'},
        {'name': 'done', 'dtype': 'bool'},
        {'name': 'trace', 'type': 'ActionTrace'},
        {'name': 'policy_version', 'dtype': 'int64'},
    ],
}


def transition_schema_definition() -> dict[str, Any]:
    """返回可安全修改的 schema 副本，供 checkpoint 审计使用。"""

    return copy.deepcopy(_SCHEMA_DEFINITION)


def transition_schema_fingerprint(
    schema: dict[str, Any] | None = None,
) -> str:
    """对 canonical JSON 做 SHA-256，避免进程相关的 ``hash()``。"""

    if schema is None:
        schema = _SCHEMA_DEFINITION
    encoded = json.dumps(
        schema,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True,
    ).encode('ascii')
    return hashlib.sha256(encoded).hexdigest()


def _is_integer_tensor(value: torch.Tensor) -> bool:
    return value.dtype in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }


@dataclass(frozen=True)
class EventTransition:
    state: EventStateTensors
    joint_action: JointEventAction
    behavior_log_prob: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor
    delta_t: torch.Tensor
    next_state: EventStateTensors
    done: torch.Tensor
    trace: ActionTrace
    policy_version: int

    def validate(self) -> None:
        self.state.validate()
        self.next_state.validate()
        for name in self.state._fields:
            if getattr(self.state, name).shape != getattr(
                self.next_state,
                name,
            ).shape:
                raise ValueError(f'next_state {name} shape does not match')

        batch_size, num_satellites = self.state.replan_mask.shape
        num_tasks = self.state.task_owner_count.shape[1]
        satellite_shape = (batch_size, num_satellites)
        batch_shape = (batch_size,)

        action_tensors = self.joint_action._asdict()
        for name, value in action_tensors.items():
            if value.shape != satellite_shape:
                raise ValueError(f'joint action {name} has invalid shape')
        if self.joint_action.terminate.dtype != torch.bool:
            raise ValueError('joint action terminate must use bool dtype')
        for name in ('task_indices', 'commitment_indices'):
            if not _is_integer_tensor(action_tensors[name]):
                raise ValueError(f'joint action {name} must use integer dtype')
        if (
            (self.joint_action.task_indices < -1).any()
            or (self.joint_action.task_indices >= num_tasks).any()
        ):
            raise ValueError('joint action task_indices are out of range')
        commitment_indices = self.joint_action.commitment_indices
        if (
            (commitment_indices < -1).any()
            or (commitment_indices >= len(COMMITMENT_SECONDS)).any()
        ):
            raise ValueError('joint action commitment_indices are out of range')
        selected = self.joint_action.task_indices >= 0
        if (
            (selected & (commitment_indices < 0)).any()
            or (~selected & (commitment_indices != -1)).any()
        ):
            raise ValueError('commitment index must match selected task')

        scalar_tensors = {
            'behavior_log_prob': self.behavior_log_prob,
            'value': self.value,
            'reward': self.reward,
            'delta_t': self.delta_t,
            'done': self.done,
        }
        for name, value in scalar_tensors.items():
            if value.shape != batch_shape:
                raise ValueError(f'{name} must contain one value per scene')
        if self.done.dtype != torch.bool:
            raise ValueError('done must use bool dtype')
        for name in ('behavior_log_prob', 'value', 'reward', 'delta_t'):
            if not torch.isfinite(scalar_tensors[name]).all():
                raise ValueError(f'{name} must contain finite values')
        if (self.delta_t <= 0).any():
            raise ValueError('delta_t must be positive')

        expected_trace_shapes = {
            'action_order': satellite_shape,
            'termination_mask': satellite_shape,
            'task_masks': (batch_size, num_satellites, num_tasks + 1),
            'commitment_masks': (
                batch_size,
                num_satellites,
                len(COMMITMENT_SECONDS),
            ),
            'owner_state': (batch_size, num_satellites, num_tasks),
        }
        for name, expected in expected_trace_shapes.items():
            if getattr(self.trace, name).shape != expected:
                raise ValueError(f'{name} has invalid trace shape')
        for name in ('termination_mask', 'task_masks', 'commitment_masks'):
            if getattr(self.trace, name).dtype != torch.bool:
                raise ValueError(f'{name} must use bool dtype')
        for name in ('action_order', 'owner_state'):
            if not _is_integer_tensor(getattr(self.trace, name)):
                raise ValueError(f'{name} must use integer dtype')
        if (
            (self.trace.action_order < -1).any()
            or (self.trace.action_order >= num_satellites).any()
        ):
            raise ValueError('action_order contains an invalid satellite id')
        for order in self.trace.action_order:
            active = order[order >= 0]
            if active.unique().numel() != active.numel():
                raise ValueError('action_order contains duplicate satellites')
        if (
            (self.trace.owner_state < 0).any()
            or (self.trace.owner_state > MAX_TASK_OWNERS).any()
        ):
            raise ValueError('owner_state is outside the physical capacity')
        if not isinstance(self.policy_version, int) or self.policy_version < 0:
            raise ValueError('policy_version must be a non-negative integer')
