"""数据集模块：将专家轨迹加工为 Transformer 可消费的 Batch 张量。

核心流程（Dataset.__getitem__）：
  1. 从 trajectories.N/ 加载 .pth 轨迹（卫星动态状态、任务进度、专家动作、可见性）
  2. 拼接静态特征（轨道根数/任务坐标）与动态特征（电池/进度等），构建任务和卫星的
     有效 mask（未释放/已过期/已完成的任务被 mask 掉）
  3. 从有效时间步中随机采样 batch_size 个决策时刻（避免爆显存，同时起到数据增强效果）
  4. 用预计算的均值/标准差归一化特征
  5. 输出 Batch（卫星传感器类型、卫星数据、卫星 mask、任务传感器类型、任务数据、
     任务 mask、专家动作标签）

JointDataset 在此基础上额外采样 TimeModel 所需的约束样本（正/负时间段），
用于联合训练可行性预测和控制时间预测。

本模块属于数据预处理，不包含任何网络层。对应 Transformer 论文中的数据准备阶段。
"""

__all__ = [
    'DynamicConstellationData',
    'DynamicTasksetData',
    'Actions',
    'TrajectoryData',
    'TemporalBatch',
    'Batch',
    'JointBatch',
    'Statistics',
    'Dataset',
    'JointDataset',
]

import bisect
from collections import UserList
import dataclasses
import random
from typing import NamedTuple, TypedDict, cast

import einops
import torch
from todd.patches.py_ import json_load
from todd.utils import NestedTensorCollectionUtils

from constellation import (
    ANNOTATIONS_ROOT,
    CONSTELLATIONS_ROOT,
    STATISTICS_PATH,
    TASKSETS_ROOT,
    TRAJECTORIES_ROOT,
    DATA_ROOT,
)
from constellation.data import Constellation, TaskSet

from .constants import TIME_SCALE
from .multi_horizon_edge_labels import (
    build_batched_edge_outcomes,
    build_event_supervision,
)
from .registries import ConstellationDatasetRegistry
from .temporal_history import build_prefix_history


class DynamicConstellationData(TypedDict):
    # shape: t x num_satellites
    # dtype: bool
    sensor_enabled: torch.Tensor

    # shape: t x num_satellites x satellite_dim (8)
    # dtype: float
    #
    # satellite_dim:
    #   - battery_percentage
    #   - reaction_wheels[0].speed
    #   - reaction_wheels[1].speed
    #   - reaction_wheels[2].speed
    #   - true_anomaly
    #   - attitude[0]
    #   - attitude[1]
    #   - attitude[2]
    data: torch.Tensor


class DynamicTasksetData(TypedDict):
    # shape: t x num_tasks
    # dtype: uint8
    progress: torch.Tensor


class Actions(TypedDict):
    # shape: t x num_satellites
    # dtype: int
    task_id: torch.Tensor


class TrajectoryData(TypedDict):
    constellation: DynamicConstellationData
    taskset: DynamicTasksetData
    actions: Actions
    # shape: t x num_satellites x num_tasks
    # dtype: bool
    is_visible: torch.Tensor


class TemporalBatch(NamedTuple):
    """P0 历史输入与真实执行边结果标签。"""

    previous_task_indices: torch.Tensor
    previous_task_available: torch.Tensor
    previous_was_idle: torch.Tensor
    run_lengths: torch.Tensor
    switch_count_30: torch.Tensor
    switch_count_60: torch.Tensor
    event_continue: torch.Tensor
    event_duration_index: torch.Tensor
    event_duration_observed: torch.Tensor
    outcome_valid: torch.Tensor
    visible_next: torch.Tensor
    progress_next: torch.Tensor
    completed_next: torch.Tensor
    horizons: torch.Tensor
    visible: torch.Tensor
    visible_observed: torch.Tensor
    progress: torch.Tensor
    progress_observed: torch.Tensor
    completed: torch.Tensor
    completion_observed: torch.Tensor
    time_to_first_visible: torch.Tensor
    time_to_first_progress: torch.Tensor
    time_to_completion: torch.Tensor


class Batch(NamedTuple):
    id_: int
    annotation_id: int
    time_steps: list[int]
    constellation_sensor_type: torch.Tensor
    constellation_sensor_enabled: torch.Tensor
    constellation_data: torch.Tensor
    constellation_mask: torch.Tensor
    tasks_sensor_type: torch.Tensor
    tasks_data: torch.Tensor
    tasks_mask: torch.Tensor
    actions_task_id: torch.Tensor  # TODO: rename
    temporal: TemporalBatch | None = None


class JointBatch(NamedTuple):
    id_: int
    annotation_id: int
    time_steps: list[int]
    constellation_sensor_type: torch.Tensor
    constellation_sensor_enabled: torch.Tensor
    constellation_data: torch.Tensor
    constellation_mask: torch.Tensor
    tasks_sensor_type: torch.Tensor
    tasks_data: torch.Tensor
    tasks_mask: torch.Tensor
    actions_task_id: torch.Tensor
    constraint_time_steps: torch.Tensor
    constraint_constellation_data: torch.Tensor
    constraint_tasks_data: torch.Tensor
    constraint_durations: torch.Tensor
    temporal: TemporalBatch | None = None


class Statistics(NamedTuple):
    constellation_mean: torch.Tensor
    constellation_std: torch.Tensor
    taskset_mean: torch.Tensor
    taskset_std: torch.Tensor


@dataclasses.dataclass(frozen=True)
class TimeSpan:
    start_time: int
    end_time: int
    satellite_id: int
    task_id: int

    @property
    def length(self) -> int:
        return self.end_time - self.start_time


class TimeSpans(UserList[TimeSpan]):

    def __init__(self) -> None:
        super().__init__()
        self._offsets: list[int] = []

    def append(self, item: TimeSpan) -> None:
        self._offsets.append(self.total_length)
        super().append(item)

    @property
    def total_length(self) -> int:
        return 0 if len(self) == 0 else self[-1].length + self._offsets[-1]

    def _to_data(
        self,
        index: int,
        *,
        with_duration: bool = True,
    ) -> tuple[int, int, int, int]:
        i = bisect.bisect(self._offsets, index) - 1
        time_span = self[i]
        time_step = time_span.start_time + index - self._offsets[i]

        if with_duration:
            duration = time_span.end_time - time_step
            if duration > 2 * TIME_SCALE:
                duration = -TIME_SCALE
        else:
            duration = -TIME_SCALE

        return time_step, duration, time_span.satellite_id, time_span.task_id

    def sample_data(self, n: int, **kwargs) -> torch.Tensor:
        if self.total_length > n:
            indices = random.sample(range(self.total_length), n)
        else:
            indices = list(range(self.total_length))
        return torch.tensor(
            [self._to_data(i, **kwargs) for i in indices],
            dtype=torch.int,
        )


@ConstellationDatasetRegistry.register_()
class Dataset(torch.utils.data.Dataset[Batch]):

    def __init__(
        self,
        *args,
        split: str,
        annotation_file: str | None = None,
        batch_size: int,
        normalize: bool = True,
        include_temporal_history: bool = False,
        temporal_horizons: tuple[int, ...] = (5, 15, 30, 300),
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._split = split

        if annotation_file is None:
            annotation_file = f'{split}.json'
        self._annotations: dict[str, list[int]] = json_load(
            str(ANNOTATIONS_ROOT / annotation_file),
        )

        self._batch_size = batch_size
        self._include_temporal_history = include_temporal_history
        self._temporal_horizons = tuple(int(h) for h in temporal_horizons)
        if include_temporal_history and (
            not self._temporal_horizons
            or any(h <= 0 for h in self._temporal_horizons)
            or len(set(self._temporal_horizons))
            != len(self._temporal_horizons)
        ):
            raise ValueError('temporal_horizons must be unique and positive')

        if normalize:
            self._statistics: Statistics = torch.load(
                STATISTICS_PATH,
                weights_only=False,
            )

        self._nested_tensor_collection_utils = NestedTensorCollectionUtils()

    @property
    def normalize(self) -> bool:
        return hasattr(self, '_statistics')

    def __len__(self) -> int:
        return len(self._annotations['ids'])

    def _load_constellation(
        self,
        constellation: DynamicConstellationData,
        id_: int,
        indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sensor_enabled = constellation['sensor_enabled'][indices]
        dynamic_data = constellation['data'][indices]

        constellation_path = (
            CONSTELLATIONS_ROOT / self._split / f'{id_ // 1000:02}'
            / f'{id_:05}.json'
        )
        sensor_type, static_data = Constellation.load(
            str(constellation_path),
        ).static_to_tensor()

        sensor_type = einops.repeat(
            sensor_type,
            'ns -> t ns',
            t=len(indices),
        )
        static_data = einops.repeat(
            static_data,
            'ns nd -> t ns nd',
            t=len(indices),
        )
        data = torch.cat([static_data, dynamic_data], -1)

        mask = torch.ones_like(sensor_type, dtype=torch.bool)

        return sensor_type, sensor_enabled, data, mask

    def _load_tasks(
        self,
        taskset: DynamicTasksetData,
        id_: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        progress = taskset['progress']
        t = progress.shape[0]

        taskset_path = (
            TASKSETS_ROOT / self._split / f'{id_ // 1000:02}'
            / f'{id_:05}.json'
        )
        sensor_type, static_data = TaskSet.load(str(taskset_path)).to_tensor()
        duration = static_data[..., 2]

        sensor_type = einops.repeat(sensor_type, 'nt -> t nt', t=t)
        static_data = einops.repeat(static_data, 'nt nd -> t nt nd', t=t)

        static_data = static_data.clone()  # for in-place modification
        time_steps = einops.rearrange(torch.arange(t), 't -> t 1')
        static_data[..., 0] -= time_steps
        static_data[..., 1] -= time_steps

        dynamic_data = einops.rearrange(progress, 't nt -> t nt 1')

        data = torch.cat([static_data, dynamic_data], -1)

        release_time_mask = static_data[..., 0] <= 0
        due_time_mask = static_data[..., 1] >= 0
        finished_mask = progress >= duration
        finished_mask, _ = finished_mask.cummax(0)
        mask = release_time_mask & due_time_mask
        mask[1:] &= ~finished_mask[:-1] # FIXME

        return sensor_type, data, mask

    def _load_actions(
        self,
        actions: Actions,
        indices: list[int],
    ) -> torch.Tensor:
        return actions['task_id'][indices]

    def _load_trajectory(self, index: int) -> tuple[int, int, TrajectoryData]:
        id_ = self._annotations['ids'][index]
        best_epoch_ = self._annotations['epochs'][index]

        trajectory: TrajectoryData = torch.load(
            DATA_ROOT
            / f'trajectories.{best_epoch_}'
            / self._split
            / f'{id_ // 1000:02}'
            / f'{id_:05}.pth',
        )
        return id_, best_epoch_, trajectory

    def _build_batch(
        self,
        index: int,
        id_: int,
        best_epoch_: int,
        trajectory: TrajectoryData,
        return_full_data: bool = False,
    ):
        tasks_sensor_type, tasks_data, tasks_mask = self._load_tasks(
            trajectory['taskset'],
            id_,
        )
        full_tasks_data = tasks_data
        task_durations = full_tasks_data[0, :, 2].clone()

        # a time step is valid iff any task is valid
        valid_time_steps = tasks_mask.any(-1)
        include_temporal_history = getattr(
            self,
            '_include_temporal_history',
            False,
        )
        if include_temporal_history:
            valid_time_steps = valid_time_steps.clone()
            valid_time_steps[0] = False
            valid_time_steps[-1] = False
        indices = valid_time_steps.nonzero().flatten().tolist()
        if len(indices) > self._batch_size:
            indices = random.sample(indices, self._batch_size)

        tasks_sensor_type = tasks_sensor_type[indices]
        tasks_data = tasks_data[indices]
        tasks_mask = tasks_mask[indices]
        temporal_candidate_mask = tasks_mask.clone()

        # TODO: rename, `actions_task_id` is ambiguous
        actions_task_id = self._load_actions(trajectory['actions'], indices)

        # remove the tasks that are never valid
        task_is_valid = tasks_mask.any(0)
        tasks_id_mapper = task_is_valid.cumsum(0) - 1
        if not task_is_valid.all():
            tasks_sensor_type = tasks_sensor_type[:, task_is_valid]
            tasks_data = tasks_data[:, task_is_valid]
            tasks_mask = tasks_mask[:, task_is_valid]
            actions_task_id = torch.where(
                actions_task_id == -1,
                actions_task_id,
                tasks_id_mapper[actions_task_id],
            )

        temporal = None
        if include_temporal_history:
            temporal = self._build_temporal_batch(
                trajectory=trajectory,
                indices=indices,
                candidate_mask=temporal_candidate_mask,
                task_durations=task_durations,
                tasks_id_mapper=tasks_id_mapper,
            )

        # ensure that `actions_task_id` is valid
        augmented_tasks_mask = torch.cat([
            tasks_mask.new_ones(len(indices), 1),
            tasks_mask,
        ], -1)
        if not augmented_tasks_mask.gather(-1, actions_task_id + 1).all():
            raise RuntimeError(
                f"Trajectory.{best_epoch_} {index} ({id_}) is invalid",
            )

        if return_full_data:
            time_indices = list(range(
                trajectory['constellation']['data'].shape[0],
            ))
            (
                full_constellation_sensor_type,
                full_constellation_sensor_enabled,
                full_constellation_data,
                full_constellation_mask,
            ) = self._load_constellation(
                trajectory['constellation'],
                id_,
                time_indices,
            )
            state_indices = (
                [time_step - 1 for time_step in indices]
                if include_temporal_history else indices
            )
            constellation_sensor_type = full_constellation_sensor_type[
                state_indices
            ]
            constellation_sensor_enabled = full_constellation_sensor_enabled[
                state_indices
            ]
            constellation_data = full_constellation_data[state_indices]
            constellation_mask = full_constellation_mask[state_indices]
        else:
            state_indices = (
                [time_step - 1 for time_step in indices]
                if include_temporal_history else indices
            )
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

        if self.normalize:
            if return_full_data:
                full_constellation_data = (
                    (
                        full_constellation_data
                        - self._statistics.constellation_mean
                    ) / (self._statistics.constellation_std + 1e-6)
                )
                full_tasks_data = (
                    (full_tasks_data - self._statistics.taskset_mean) /
                    (self._statistics.taskset_std + 1e-6)
                )
                constellation_data = full_constellation_data[state_indices]
                tasks_data = full_tasks_data[indices]
                if not task_is_valid.all():
                    tasks_data = tasks_data[:, task_is_valid]
            else:
                constellation_data = (
                    (constellation_data - self._statistics.constellation_mean) /
                    (self._statistics.constellation_std + 1e-6)
                )
                tasks_data = ((tasks_data - self._statistics.taskset_mean) /
                              (self._statistics.taskset_std + 1e-6))

        # NOTE: sensor type should be 0-indexed
        batch = Batch(
            index,
            id_,
            indices,
            constellation_sensor_type - 1,
            constellation_sensor_enabled,
            constellation_data,
            constellation_mask,
            tasks_sensor_type - 1,
            tasks_data,
            tasks_mask,
            actions_task_id,
            temporal,
        )

        if return_full_data:
            return batch, full_constellation_data, full_tasks_data
        return batch

    def _build_temporal_batch(
        self,
        *,
        trajectory: TrajectoryData,
        indices: list[int],
        candidate_mask: torch.Tensor,
        task_durations: torch.Tensor,
        tasks_id_mapper: torch.Tensor,
    ) -> TemporalBatch:
        actions = trajectory['actions']['task_id']
        num_tasks = trajectory['taskset']['progress'].shape[1]
        candidate_ids = torch.arange(num_tasks).repeat(len(indices), 1)
        history = build_prefix_history(
            actions,
            torch.tensor(indices, dtype=torch.long),
            candidate_global_task_ids=candidate_ids,
            candidate_mask=candidate_mask,
        )
        previous_task_indices = torch.where(
            history.previous_task_available,
            tasks_id_mapper[history.previous_task_indices.clamp_min(0)],
            history.previous_task_indices.new_full((), -1),
        )

        horizons = getattr(self, '_temporal_horizons', (5, 15, 30, 300))
        outcomes = build_batched_edge_outcomes(
            actions=actions,
            is_visible=trajectory['is_visible'],
            progress=trajectory['taskset']['progress'],
            task_durations=task_durations,
            horizons=horizons,
        )
        event_targets = build_event_supervision(actions)

        def stack_horizons(name: str) -> torch.Tensor:
            return torch.stack([
                getattr(outcomes.horizons[horizon], name)[indices]
                for horizon in horizons
            ], -1)

        return TemporalBatch(
            previous_task_indices=previous_task_indices,
            previous_task_available=history.previous_task_available,
            previous_was_idle=history.previous_was_idle,
            run_lengths=history.run_lengths,
            switch_count_30=history.switch_count_30,
            switch_count_60=history.switch_count_60,
            event_continue=event_targets.continue_next[indices],
            event_duration_index=event_targets.duration_index[indices],
            event_duration_observed=(
                event_targets.duration_observed[indices]
            ),
            outcome_valid=outcomes.valid[indices],
            visible_next=outcomes.visible_next[indices],
            progress_next=outcomes.progress_next[indices],
            completed_next=outcomes.completed_next[indices],
            horizons=torch.tensor(horizons, dtype=torch.long),
            visible=stack_horizons('visible'),
            visible_observed=stack_horizons('visible_observed'),
            progress=stack_horizons('progress'),
            progress_observed=stack_horizons('progress_observed'),
            completed=stack_horizons('completed'),
            completion_observed=stack_horizons('completion_observed'),
            time_to_first_visible=stack_horizons('time_to_first_visible'),
            time_to_first_progress=stack_horizons('time_to_first_progress'),
            time_to_completion=stack_horizons('time_to_completion'),
        )

    def __getitem__(self, index: int) -> Batch:
        id_, best_epoch_, trajectory = self._load_trajectory(index)
        return self._build_batch(index, id_, best_epoch_, trajectory)


@ConstellationDatasetRegistry.register_()
class JointDataset(Dataset):

    def __init__(
        self,
        *args,
        constraint_batch_size: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._constraint_batch_size = (
            self._batch_size if constraint_batch_size is None
            else constraint_batch_size
        )

    def _append_time_spans(
        self,
        positives: TimeSpans,
        negatives: TimeSpans,
        satellite_id: int,
        actions: torch.Tensor,
        action_changed: torch.Tensor,
        consecutive_visible: torch.Tensor,
    ) -> None:
        a = b = 0
        while True:
            b += 1
            if b >= actions.shape[0]:
                break

            if not action_changed[b] and not consecutive_visible[b]:
                continue

            time_span = TimeSpan(
                a,
                b,
                satellite_id,
                cast(int, actions[a].item()),
            )
            if action_changed[b]:
                negatives.append(time_span)
            else:
                positives.append(time_span)
                while b < actions.shape[0] and not action_changed[b]:
                    b += 1
            a = b

    def _parse_time_spans(
        self,
        actions: torch.Tensor,
        is_visible: torch.Tensor,
    ) -> tuple[TimeSpans, TimeSpans]:
        action_changed = torch.ones_like(actions, dtype=torch.bool)
        action_changed[1:] = actions[1:] != actions[:-1]

        is_visible = torch.gather(
            is_visible,
            -1,
            einops.rearrange(actions.clamp(0), 't ns -> t ns 1'),
        )
        is_visible = einops.rearrange(is_visible, 't ns 1 -> t ns')
        is_visible[actions == -1] = False

        consecutive_visible = torch.zeros_like(is_visible)
        consecutive_visible[:-2] = (
            is_visible[:-2] & is_visible[1:-1] & is_visible[2:]
            & ~action_changed[:-2] & ~action_changed[1:-1]
            & ~action_changed[2:]
        )

        positives = TimeSpans()
        negatives = TimeSpans()
        for satellite_id in range(actions.shape[1]):
            self._append_time_spans(
                positives,
                negatives,
                satellite_id,
                actions[:, satellite_id],
                action_changed[:, satellite_id],
                consecutive_visible[:, satellite_id],
            )

        return positives, negatives

    def __getitem__(self, index: int) -> JointBatch:
        id_, best_epoch_, trajectory = self._load_trajectory(index)
        batch, full_constellation_data, full_tasks_data = self._build_batch(
            index,
            id_,
            best_epoch_,
            trajectory,
            return_full_data=True,
        )

        positive_time_spans, negative_time_spans = self._parse_time_spans(
            trajectory['actions']['task_id'],
            trajectory['is_visible'],
        )
        n_positive = self._constraint_batch_size // 2
        n_negative = self._constraint_batch_size - n_positive
        positive_data = positive_time_spans.sample_data(n_positive)
        negative_data = negative_time_spans.sample_data(
            n_negative,
            with_duration=False,
        )
        constraint_data = torch.cat([positive_data, negative_data])
        (
            constraint_time_steps,
            constraint_durations,
            constraint_satellite_ids,
            constraint_task_ids,
        ) = constraint_data.unbind(-1)

        return JointBatch(
            id_=batch.id_,
            annotation_id=batch.annotation_id,
            time_steps=batch.time_steps,
            constellation_sensor_type=batch.constellation_sensor_type,
            constellation_sensor_enabled=batch.constellation_sensor_enabled,
            constellation_data=batch.constellation_data,
            constellation_mask=batch.constellation_mask,
            tasks_sensor_type=batch.tasks_sensor_type,
            tasks_data=batch.tasks_data,
            tasks_mask=batch.tasks_mask,
            actions_task_id=batch.actions_task_id,
            constraint_time_steps=constraint_time_steps,
            constraint_constellation_data=full_constellation_data[
                constraint_time_steps,
                constraint_satellite_ids,
            ],
            constraint_tasks_data=full_tasks_data[
                constraint_time_steps,
                constraint_task_ids,
            ],
            constraint_durations=constraint_durations.float() / TIME_SCALE,
            temporal=batch.temporal,
        )
