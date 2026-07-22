"""从现有专家轨迹构造 V2-0 事件级事实监督。"""

import random
from typing import NamedTuple

import torch

from ..dataset import Batch, Dataset, TrajectoryData
from ..registries import ConstellationDatasetRegistry
from ..temporal_history import build_prefix_history
from .reward import (
    completion_potential,
    completion_task_weights,
    terminal_completion_quality,
)
from .state import (
    COMMITMENT_SECONDS,
    MAX_TASK_OWNERS,
    EventStateTensors,
)


class OfflineEventTargets(NamedTuple):
    termination: torch.Tensor
    termination_observed: torch.Tensor
    task_indices: torch.Tensor
    task_observed: torch.Tensor
    commitment_indices: torch.Tensor
    commitment_observed: torch.Tensor
    value_returns: torch.Tensor


class OfflineEventBatch(NamedTuple):
    stage3_batch: Batch
    event_state: EventStateTensors
    targets: OfflineEventTargets


def _validate_trajectory_tensors(
    actions: torch.Tensor,
    task_valid: torch.Tensor,
    progress: torch.Tensor,
    durations: torch.Tensor,
) -> None:
    if actions.ndim != 2 or task_valid.ndim != 2 or progress.ndim != 2:
        raise ValueError('trajectory tensors must be rank two')
    if task_valid.shape != progress.shape:
        raise ValueError('task validity and progress must share shape')
    if actions.shape[0] != progress.shape[0]:
        raise ValueError('actions and tasks must share the time axis')
    if durations.shape != (progress.shape[1],):
        raise ValueError('durations must contain one value per task')
    if task_valid.dtype != torch.bool:
        raise ValueError('task validity must use bool dtype')
    if (
        not torch.isfinite(progress).all()
        or not torch.isfinite(durations).all()
        or (durations <= 0).any()
    ):
        raise ValueError('progress and durations must be finite and valid')


def compress_expert_actions_to_events(
    *,
    actions: torch.Tensor,
    task_valid: torch.Tensor,
    progress: torch.Tensor,
    durations: torch.Tensor,
) -> list[int]:
    """保留策略切换、任务集合变化和任务完成边界，去掉逐秒重复状态。"""

    _validate_trajectory_tensors(actions, task_valid, progress, durations)
    if actions.shape[0] < 2:
        return []
    completed = progress >= durations.view(1, -1)
    events: list[int] = []
    for time_step in range(1, actions.shape[0]):
        if not task_valid[time_step].any():
            continue
        first_active = not task_valid[:time_step].any()
        action_changed = not torch.equal(
            actions[time_step],
            actions[time_step - 1],
        )
        validity_changed = not torch.equal(
            task_valid[time_step],
            task_valid[time_step - 1],
        )
        completion_changed = not torch.equal(
            completed[time_step],
            completed[time_step - 1],
        )
        if (
            first_active
            or time_step == 1
            or action_changed
            or validity_changed
            or completion_changed
        ):
            events.append(time_step)
    return events


def build_commitment_targets(
    *,
    actions: torch.Tensor,
    event_indices: list[int],
    task_remaining_required_seconds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从实际连续动作段提取不超过段长的最大合法最低承诺。"""

    if actions.ndim != 2:
        raise ValueError('actions must be time by satellite')
    if task_remaining_required_seconds.ndim != 2:
        raise ValueError('remaining duration must be time by task')
    if actions.shape[0] != task_remaining_required_seconds.shape[0]:
        raise ValueError('commitment inputs must share the time axis')
    num_tasks = task_remaining_required_seconds.shape[1]
    indices = torch.zeros(
        len(event_indices),
        actions.shape[1],
        dtype=torch.long,
        device=actions.device,
    )
    observed = torch.zeros_like(indices, dtype=torch.bool)
    for event_position, time_step in enumerate(event_indices):
        if not 0 <= time_step < actions.shape[0]:
            raise ValueError('event index is outside the trajectory')
        for satellite_id in range(actions.shape[1]):
            task_id = int(actions[time_step, satellite_id].item())
            if not 0 <= task_id < num_tasks:
                continue
            end = time_step + 1
            while (
                end < actions.shape[0]
                and actions[end, satellite_id] == task_id
            ):
                end += 1
            segment_length = end - time_step
            remaining = float(task_remaining_required_seconds[
                time_step,
                task_id,
            ].item())
            legal = [
                position
                for position, seconds in enumerate(COMMITMENT_SECONDS)
                if seconds <= segment_length
                and (seconds != 1 or remaining <= 1)
            ]
            if legal:
                indices[event_position, satellite_id] = legal[-1]
                observed[event_position, satellite_id] = True
    return indices, observed


def build_capped_owner_counts(
    task_indices: torch.Tensor,
    num_tasks: int,
) -> torch.Tensor:
    """统计当前 owner，并把旧专家的超容量状态饱和到 V2 上限。"""

    if task_indices.ndim != 2 or num_tasks <= 0:
        raise ValueError('owner count inputs are invalid')
    counts = torch.zeros(
        task_indices.shape[0],
        num_tasks,
        dtype=torch.long,
        device=task_indices.device,
    )
    for batch_index in range(task_indices.shape[0]):
        active = task_indices[batch_index]
        active = active[(active >= 0) & (active < num_tasks)]
        if active.numel():
            counts[batch_index].scatter_add_(
                0,
                active.long(),
                torch.ones_like(active, dtype=torch.long),
            )
    return counts.clamp_max(MAX_TASK_OWNERS)


@ConstellationDatasetRegistry.register_()
class EventV2OfflineDataset(Dataset):
    """一个轨迹样本返回一组事件状态；batch 内不携带 ``is_visible``。"""

    def _build_offline_batch(
        self,
        index: int,
        id_: int,
        best_epoch_: int,
        trajectory: TrajectoryData,
    ) -> OfflineEventBatch:
        actions = trajectory['actions']['task_id'].long()
        progress = trajectory['taskset']['progress'].float()
        (
            full_tasks_sensor_type,
            full_tasks_data,
            full_tasks_mask,
        ) = self._load_tasks(trajectory['taskset'], id_)
        durations = full_tasks_data[0, :, 2].float()
        all_event_indices = compress_expert_actions_to_events(
            actions=actions,
            task_valid=full_tasks_mask,
            progress=progress,
            durations=durations,
        )
        if not all_event_indices:
            raise RuntimeError(f'trajectory {id_} contains no trainable events')
        event_indices = all_event_indices
        if len(event_indices) > self._batch_size:
            event_indices = sorted(random.sample(
                event_indices,
                self._batch_size,
            ))
        state_indices = [time_step - 1 for time_step in event_indices]
        (
            constellation_sensor_type,
            constellation_sensor_enabled,
            constellation_data,
            constellation_mask,
        ) = self._load_constellation(
            trajectory['constellation'],
            id_,
            state_indices,
        )
        tasks_sensor_type = full_tasks_sensor_type[event_indices]
        tasks_data = full_tasks_data[event_indices]
        tasks_mask = full_tasks_mask[event_indices]
        if self.normalize:
            constellation_data = (
                constellation_data - self._statistics.constellation_mean
            ) / (self._statistics.constellation_std + 1e-6)
            tasks_data_normalized = (
                tasks_data - self._statistics.taskset_mean
            ) / (self._statistics.taskset_std + 1e-6)
        else:
            tasks_data_normalized = tasks_data

        stage3_batch = Batch(
            id_=index,
            annotation_id=id_,
            time_steps=event_indices,
            constellation_sensor_type=constellation_sensor_type - 1,
            constellation_sensor_enabled=constellation_sensor_enabled,
            constellation_data=constellation_data,
            constellation_mask=constellation_mask,
            tasks_sensor_type=tasks_sensor_type - 1,
            tasks_data=tasks_data_normalized,
            tasks_mask=tasks_mask,
            actions_task_id=actions[event_indices],
            temporal=None,
        )

        event_tensor = torch.tensor(event_indices, dtype=torch.long)
        candidate_ids = torch.arange(
            progress.shape[1],
            device=actions.device,
        ).repeat(len(event_indices), 1)
        history = build_prefix_history(
            actions,
            event_tensor,
            candidate_global_task_ids=candidate_ids,
            candidate_mask=tasks_mask,
        )
        previous_actions = actions[state_indices]
        current_actions = actions[event_indices]
        gathered_previous_valid = tasks_mask.gather(
            1,
            previous_actions.clamp_min(0),
        )
        forced_interrupt = (
            (previous_actions >= 0) & ~gathered_previous_valid
        )
        action_changed = current_actions != previous_actions
        replan_mask = action_changed | forced_interrupt
        can_terminate = (previous_actions >= 0) & ~forced_interrupt

        delta_by_event: dict[int, int] = {}
        previous_event = 0
        for event in all_event_indices:
            delta_by_event[event] = max(1, event - previous_event)
            previous_event = event
        delta_t = torch.tensor([
            delta_by_event[event] for event in event_indices
        ], dtype=torch.float).view(-1, 1).expand_as(history.run_lengths)

        event_types = torch.zeros_like(previous_actions)
        termination_reasons = torch.zeros_like(previous_actions)
        completed = progress >= durations.view(1, -1)
        for batch_index, time_step in enumerate(event_indices):
            if not torch.equal(
                full_tasks_mask[time_step],
                full_tasks_mask[time_step - 1],
            ):
                event_types[batch_index].fill_(1)
            elif not torch.equal(
                completed[time_step],
                completed[time_step - 1],
            ):
                event_types[batch_index].fill_(2)
            elif action_changed[batch_index].any():
                event_types[batch_index].fill_(3)
        termination_reasons = torch.where(
            forced_interrupt,
            termination_reasons.new_tensor(1),
            termination_reasons,
        )
        termination_reasons = torch.where(
            action_changed & ~forced_interrupt,
            termination_reasons.new_tensor(2),
            termination_reasons,
        )

        compatible = (
            constellation_sensor_type.unsqueeze(2)
            == tasks_sensor_type.unsqueeze(1)
        ) & tasks_mask.unsqueeze(1)
        deadline = full_tasks_data[event_indices, :, 1].float().unsqueeze(1)
        deadline = deadline.expand(-1, actions.shape[1], -1)
        compatible_deadline_slack = torch.where(
            compatible,
            deadline,
            deadline.new_full((), 3600.),
        ).min(dim=-1).values
        task_remaining = (
            durations.view(1, -1) - progress[event_indices]
        ).clamp_min(0)
        owner_count = build_capped_owner_counts(
            previous_actions,
            progress.shape[1],
        )
        event_state = EventStateTensors(
            previous_task_indices=previous_actions,
            current_task_indices=previous_actions,
            minimum_commitment_remaining=torch.zeros_like(
                history.run_lengths,
                dtype=torch.float,
            ),
            run_lengths=history.run_lengths.float(),
            seconds_since_replan=history.run_lengths.float(),
            switch_count_30=history.switch_count_30.float(),
            switch_count_60=history.switch_count_60.float(),
            termination_reason=termination_reasons,
            event_type=event_types,
            delta_t=delta_t,
            replan_mask=replan_mask,
            forced_interrupt_mask=forced_interrupt,
            can_terminate_mask=can_terminate,
            compatible_deadline_slack=compatible_deadline_slack,
            task_remaining_required_seconds=task_remaining,
            task_owner_count=owner_count,
            task_locked_owner_count=torch.zeros_like(owner_count),
        )
        event_state.validate()

        commitment_indices, commitment_observed = (
            build_commitment_targets(
                actions=actions,
                event_indices=event_indices,
                task_remaining_required_seconds=(
                    durations.view(1, -1) - progress
                ).clamp_min(0),
            )
        )
        commitment_observed &= replan_mask & (current_actions >= 0)
        terminal_progress = progress.max(dim=0).values
        terminal_completed = terminal_progress >= durations
        terminal_q = terminal_completion_quality(
            terminal_progress,
            durations,
            terminal_completed,
        )
        task_weights = completion_task_weights(durations)
        current_potential = completion_potential(
            progress[event_indices],
            durations.expand(len(event_indices), -1),
            task_weights.expand(len(event_indices), -1),
        )
        targets = OfflineEventTargets(
            termination=action_changed & can_terminate,
            termination_observed=can_terminate,
            task_indices=current_actions,
            task_observed=replan_mask,
            commitment_indices=commitment_indices,
            commitment_observed=commitment_observed,
            value_returns=terminal_q - current_potential,
        )
        del best_epoch_
        return OfflineEventBatch(stage3_batch, event_state, targets)

    def __getitem__(self, index: int) -> OfflineEventBatch:
        id_, best_epoch_, trajectory = self._load_trajectory(index)
        return self._build_offline_batch(
            index,
            id_,
            best_epoch_,
            trajectory,
        )
