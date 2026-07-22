"""V2 可部署事件状态、确定性规划顺序与动作物理掩码。"""

from typing import NamedTuple

import torch


COMMITMENT_SECONDS = (1, 5, 15, 30, 60)
MAX_TASK_OWNERS = 3


class EventStateTensors(NamedTuple):
    """一个 batch 的显式事件状态。

    前十五个字段以卫星为第二维，最后三个字段以任务为第二维。任务索引均为当前
    场景内的相对索引，idle 使用 ``-1``，因此没有跨场景 task-id embedding。
    """

    previous_task_indices: torch.Tensor
    current_task_indices: torch.Tensor
    minimum_commitment_remaining: torch.Tensor
    run_lengths: torch.Tensor
    seconds_since_replan: torch.Tensor
    switch_count_30: torch.Tensor
    switch_count_60: torch.Tensor
    termination_reason: torch.Tensor
    event_type: torch.Tensor
    delta_t: torch.Tensor
    replan_mask: torch.Tensor
    forced_interrupt_mask: torch.Tensor
    can_terminate_mask: torch.Tensor
    compatible_deadline_slack: torch.Tensor
    task_remaining_required_seconds: torch.Tensor
    task_owner_count: torch.Tensor
    task_locked_owner_count: torch.Tensor

    def validate(self) -> None:
        tensors = self._asdict()
        for name, value in tensors.items():
            if not isinstance(value, torch.Tensor) or value.ndim != 2:
                raise ValueError(f'{name} must be a rank-two tensor')

        satellite_shape = self.previous_task_indices.shape
        satellite_fields = (
            'previous_task_indices',
            'current_task_indices',
            'minimum_commitment_remaining',
            'run_lengths',
            'seconds_since_replan',
            'switch_count_30',
            'switch_count_60',
            'termination_reason',
            'event_type',
            'delta_t',
            'replan_mask',
            'forced_interrupt_mask',
            'can_terminate_mask',
            'compatible_deadline_slack',
        )
        for name in satellite_fields:
            if tensors[name].shape != satellite_shape:
                raise ValueError(
                    f'{name} must match the satellite shape {satellite_shape}'
                )

        task_shape = self.task_remaining_required_seconds.shape
        if task_shape[0] != satellite_shape[0]:
            raise ValueError('task and satellite tensors must share batch size')
        for name in ('task_owner_count', 'task_locked_owner_count'):
            if tensors[name].shape != task_shape:
                raise ValueError(
                    f'{name} must match the task shape {task_shape}'
                )

        boolean_fields = (
            'replan_mask',
            'forced_interrupt_mask',
            'can_terminate_mask',
        )
        for name in boolean_fields:
            if tensors[name].dtype != torch.bool:
                raise ValueError(f'{name} must use bool dtype')

        integer_fields = (
            'previous_task_indices',
            'current_task_indices',
            'termination_reason',
            'event_type',
            'task_owner_count',
            'task_locked_owner_count',
        )
        integer_dtypes = {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }
        for name in integer_fields:
            if tensors[name].dtype not in integer_dtypes:
                raise ValueError(f'{name} must use an integer dtype')

        num_tasks = task_shape[1]
        for name in ('previous_task_indices', 'current_task_indices'):
            value = tensors[name]
            if ((value < -1) | (value >= num_tasks)).any():
                raise ValueError(f'{name} contains an invalid task index')

        numeric_fields = tuple(
            name for name in tensors if name not in boolean_fields
        )
        for name in numeric_fields:
            value = tensors[name]
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f'{name} must contain finite values')

        nonnegative_fields = (
            'minimum_commitment_remaining',
            'run_lengths',
            'seconds_since_replan',
            'switch_count_30',
            'switch_count_60',
            'termination_reason',
            'event_type',
            'delta_t',
            'task_remaining_required_seconds',
            'task_owner_count',
            'task_locked_owner_count',
        )
        for name in nonnegative_fields:
            if (tensors[name] < 0).any():
                raise ValueError(f'{name} must be non-negative')

        if (self.task_owner_count > MAX_TASK_OWNERS).any():
            raise ValueError(
                f'task owner count cannot exceed {MAX_TASK_OWNERS}'
            )
        if (self.task_locked_owner_count > self.task_owner_count).any():
            raise ValueError('locked owner count cannot exceed owner count')


def build_replan_order(state: EventStateTensors) -> list[torch.Tensor]:
    """按强制中断、deadline、等待时长、卫星 id 生成稳定顺序。"""

    state.validate()
    orders: list[torch.Tensor] = []
    for batch_index in range(state.replan_mask.shape[0]):
        satellite_ids = state.replan_mask[batch_index].nonzero().flatten()
        ordered = sorted(
            satellite_ids.tolist(),
            key=lambda satellite_id: (
                -int(state.forced_interrupt_mask[
                    batch_index, satellite_id
                ].item()),
                float(state.compatible_deadline_slack[
                    batch_index, satellite_id
                ].item()),
                -float(state.seconds_since_replan[
                    batch_index, satellite_id
                ].item()),
                satellite_id,
            ),
        )
        orders.append(torch.tensor(
            ordered,
            dtype=torch.long,
            device=state.replan_mask.device,
        ))
    return orders


def build_commitment_mask(
    remaining_required_seconds: torch.Tensor,
    task_selected: torch.Tensor,
) -> torch.Tensor:
    """返回 minimum commitment 的合法类别。

    idle 不产生 commitment 动作。非空任务只有剩余要求观测时长不超过一秒时才开放
    ``1s`` 档，其余四档始终可选。
    """

    if remaining_required_seconds.shape != task_selected.shape:
        raise ValueError('remaining seconds and selected mask need same shape')
    if remaining_required_seconds.ndim == 0:
        raise ValueError('commitment inputs must include a batch dimension')
    if task_selected.dtype != torch.bool:
        raise ValueError('task_selected must use bool dtype')
    if (
        not torch.isfinite(remaining_required_seconds).all()
        or (remaining_required_seconds < 0).any()
    ):
        raise ValueError('remaining required seconds must be finite and non-negative')

    mask = task_selected.unsqueeze(-1).expand(
        *task_selected.shape,
        len(COMMITMENT_SECONDS),
    ).clone()
    mask[..., 0] &= remaining_required_seconds <= 1
    return mask
