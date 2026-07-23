"""V2 在线策略的一次完整、可重放观测。"""

from collections.abc import Sequence
from typing import NamedTuple

import torch

from .state import EventStateTensors


class EventPolicyObservation(NamedTuple):
    """一次 Actor-Critic 调用所需的全部公开输入。"""

    time_steps: torch.Tensor
    constellation_sensor_type: torch.Tensor
    constellation_sensor_enabled: torch.Tensor
    constellation_data: torch.Tensor
    constellation_mask: torch.Tensor
    tasks_sensor_type: torch.Tensor
    tasks_data: torch.Tensor
    tasks_mask: torch.Tensor
    event_state: EventStateTensors

    @property
    def batch_size(self) -> int:
        return int(self.time_steps.shape[0])

    @property
    def num_satellites(self) -> int:
        return int(self.constellation_mask.shape[1])

    @property
    def num_tasks(self) -> int:
        return int(self.tasks_mask.shape[1])

    def model_args(self) -> tuple[torch.Tensor, ...]:
        """按 ``EventJointActorCritic`` 的位置参数顺序返回输入。"""

        return (
            self.time_steps,
            self.constellation_sensor_type,
            self.constellation_sensor_enabled,
            self.constellation_data,
            self.constellation_mask,
            self.tasks_sensor_type,
            self.tasks_data,
            self.tasks_mask,
        )

    def validate(self) -> None:
        if not isinstance(self.time_steps, torch.Tensor):
            raise ValueError('time steps must be a tensor')
        if self.time_steps.ndim != 1 or self.time_steps.shape[0] == 0:
            raise ValueError('time steps must contain a non-empty batch')
        if self.time_steps.is_floating_point():
            raise ValueError('time steps must use an integer dtype')

        batch_size = self.batch_size
        satellite_shape = self.constellation_mask.shape
        task_shape = self.tasks_mask.shape
        if (
            self.constellation_mask.ndim != 2
            or satellite_shape[0] != batch_size
            or satellite_shape[1] == 0
        ):
            raise ValueError('constellation mask has an invalid shape')
        if (
            self.tasks_mask.ndim != 2
            or task_shape[0] != batch_size
            or task_shape[1] == 0
        ):
            raise ValueError('task mask has an invalid shape')
        if self.constellation_mask.dtype != torch.bool:
            raise ValueError('constellation mask must use bool dtype')
        if self.tasks_mask.dtype != torch.bool:
            raise ValueError('task mask must use bool dtype')
        if not self.constellation_mask.any(dim=1).all():
            raise ValueError('each scene needs a valid satellite')
        if not self.tasks_mask.any(dim=1).all():
            raise ValueError('each scene needs a valid task')

        satellite_inputs = (
            ('constellation sensor type', self.constellation_sensor_type, 2),
            (
                'constellation sensor enabled',
                self.constellation_sensor_enabled,
                2,
            ),
            ('constellation data', self.constellation_data, 3),
        )
        for name, value, rank in satellite_inputs:
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != rank
                or value.shape[:2] != satellite_shape
            ):
                raise ValueError(f'{name} has an invalid shape')

        task_inputs = (
            ('task sensor type', self.tasks_sensor_type, 2),
            ('task data', self.tasks_data, 3),
        )
        for name, value, rank in task_inputs:
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != rank
                or value.shape[:2] != task_shape
            ):
                raise ValueError(f'task mask does not match {name}')

        for name, value in (
            ('constellation data', self.constellation_data),
            ('task data', self.tasks_data),
        ):
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f'{name} must be finite floating point')

        self.event_state.validate()
        if self.event_state.replan_mask.shape != satellite_shape:
            raise ValueError('event state satellite shape does not match observation')
        if self.event_state.task_owner_count.shape != task_shape:
            raise ValueError('event state task shape does not match observation')

    def to(
        self,
        device: torch.device,
        *,
        non_blocking: bool = False,
    ) -> 'EventPolicyObservation':
        def move(value: torch.Tensor) -> torch.Tensor:
            return value.to(device, non_blocking=non_blocking)

        event_state = EventStateTensors(*(
            move(getattr(self.event_state, name))
            for name in self.event_state._fields
        ))
        return EventPolicyObservation(
            *(move(value) for value in self.model_args()),
            event_state,
        )


def stack_event_observations(
    observations: Sequence[EventPolicyObservation],
) -> EventPolicyObservation:
    """合并 shape 相同的单场观测，禁止隐式 padding。"""

    if not observations:
        raise ValueError('at least one observation is required')
    for observation in observations:
        observation.validate()
        if observation.batch_size != 1:
            raise ValueError('each observation must contain exactly one scene')

    first = observations[0]
    if any(
        observation.num_satellites != first.num_satellites
        or observation.num_tasks != first.num_tasks
        for observation in observations[1:]
    ):
        raise ValueError('observations must share the same satellite and task shapes')

    model_fields = [
        torch.cat([observation[index] for observation in observations], dim=0)
        for index in range(8)
    ]
    event_state = EventStateTensors(*(
        torch.cat([
            getattr(observation.event_state, name)
            for observation in observations
        ], dim=0)
        for name in first.event_state._fields
    ))
    stacked = EventPolicyObservation(*model_fields, event_state)
    stacked.validate()
    return stacked
