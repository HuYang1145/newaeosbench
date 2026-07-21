"""受控单步动作分支与局部窗口原始指标。

该模块不在在线推理中调用 Basilisk。它只为离线短窗口 rollout 提供两项能力：
在指定时刻覆盖一颗卫星的动作，以及把窗口结果汇总为不带主观权重的原始分量。
"""

__all__ = [
    'BranchDecision',
    'ControlledActionAlgorithm',
    'ControlledCommitmentAlgorithm',
    'LocalWindowCallback',
    'find_stay_switch_decisions',
    'is_decision_replayable',
    'summarize_local_window',
    'summarize_prefix_paper_metrics',
]

import dataclasses
import hashlib
from typing import Any

import torch

from ..algorithms.base import BaseAlgorithm
from ..callbacks import BaseCallback
from ..data import Action, Actions, Constellation, TaskSet
from ..environments import BaseEnvironment
from ..task_managers import TaskManager


@dataclasses.dataclass(frozen=True)
class BranchDecision:
    """同一状态下需要比较的 stay/switch 动作。"""

    decision_time: int
    satellite_index: int
    stay_task_id: int
    switch_task_id: int
    pattern: str

    def to_dict(self) -> dict[str, int | str]:
        return dataclasses.asdict(self)


def _pulse_pattern(previous: int, current: int, following: int) -> str:
    if previous == -1 and current >= 0 and following == -1:
        return 'idle_task_idle'
    if previous >= 0 and current == -1 and following == previous:
        return 'task_idle_task'
    if previous >= 0 and current >= 0 and following == previous:
        return 'task_other_task'
    return 'one_second_switch'


def find_stay_switch_decisions(
    actions: torch.Tensor,
    *,
    max_decisions: int,
    latest_decision_time: int,
) -> list[BranchDecision]:
    """从参考动作中按时间顺序提取一秒 stay/switch 决策点。"""

    if actions.ndim != 2:
        raise ValueError('actions must have shape (time, satellites)')
    if max_decisions <= 0:
        raise ValueError('max_decisions must be positive')

    decisions: list[BranchDecision] = []
    last_time = min(latest_decision_time, actions.shape[0] - 2)
    for time in range(1, last_time + 1):
        for satellite in range(actions.shape[1]):
            previous = int(actions[time - 1, satellite])
            current = int(actions[time, satellite])
            following = int(actions[time + 1, satellite])
            if current == previous or current == following:
                continue
            decisions.append(
                BranchDecision(
                    decision_time=time,
                    satellite_index=satellite,
                    stay_task_id=previous,
                    switch_task_id=current,
                    pattern=_pulse_pattern(previous, current, following),
                )
            )
            if len(decisions) >= max_decisions:
                return decisions
    return decisions


def is_decision_replayable(
    decision: BranchDecision,
    *,
    taskset: TaskSet,
    reference_progress: torch.Tensor,
) -> bool:
    """判断 stay/switch 两个任务在参考决策时刻是否都可被强制执行。"""

    if reference_progress.ndim != 2:
        raise ValueError('reference_progress must have shape (time, tasks)')
    if reference_progress.shape[1] != len(taskset):
        raise ValueError('reference_progress does not match taskset')
    if not 1 <= decision.decision_time < reference_progress.shape[0]:
        return False

    task_index = {task.id_: index for index, task in enumerate(taskset)}
    for task_id in (decision.stay_task_id, decision.switch_task_id):
        if task_id == -1:
            continue
        index = task_index.get(task_id)
        if index is None:
            return False
        task = taskset[index]
        if not task.release_time <= decision.decision_time <= task.due_time:
            return False
        previous_max = reference_progress[
            :decision.decision_time,
            index,
        ].max()
        if previous_max >= task.duration:
            return False
    return True


class ControlledActionAlgorithm(BaseAlgorithm):
    """只在指定时刻覆盖一颗卫星动作的算法包装器。"""

    def __init__(
        self,
        *args,
        base_algorithm: BaseAlgorithm,
        decision_time: int,
        satellite_index: int,
        forced_task_id: int | None = None,
        forced_candidate_rank: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if (forced_task_id is None) == (forced_candidate_rank is None):
            raise ValueError(
                'provide exactly one of forced_task_id or '
                'forced_candidate_rank'
            )
        if forced_candidate_rank is not None and forced_candidate_rank < 0:
            raise ValueError('forced_candidate_rank must be non-negative')
        self._base_algorithm = base_algorithm
        self._decision_time = decision_time
        self._satellite_index = satellite_index
        self._forced_task_id = forced_task_id
        self._forced_candidate_rank = forced_candidate_rank
        self.override_applied = False
        self.original_task_id: int | None = None
        self.original_assignment: list[int] | None = None
        self.applied_task_id: int | None = None
        self.decision_state_signature: str | None = None
        self._assignment_history: list[list[int]] = []
        self._decision_context: dict[str, Any] | None = None

    def prepare(
        self,
        environment: BaseEnvironment,
        task_manager: TaskManager,
    ) -> None:
        self._task_manager = task_manager
        self._base_algorithm.prepare(environment, task_manager)

    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor) -> bytes:
        return tensor.detach().cpu().contiguous().numpy().tobytes()

    def _state_signature(
        self,
        *,
        constellation: Constellation,
        taskset: TaskSet,
        earth_rotation: torch.Tensor,
        assignment: list[int],
    ) -> str:
        sensor_enabled, dynamic_data = constellation.dynamic_to_tensor()
        digest = hashlib.sha256()
        for tensor in (
            torch.tensor([self._timer.time]),
            sensor_enabled,
            dynamic_data,
            taskset.ids,
            self._task_manager.progress,
            earth_rotation,
            torch.tensor(assignment),
        ):
            digest.update(self._tensor_bytes(tensor))
        return digest.hexdigest()

    def _forced_action(
        self,
        taskset: TaskSet,
        constellation: Constellation,
        forced_task_id: int,
    ) -> Action:
        satellite = constellation.sort()[self._satellite_index]
        if forced_task_id == -1:
            return Action(
                toggle=satellite.sensor.enabled,
                target_location=None,
            )

        matching = [task for task in taskset if task.id_ == forced_task_id]
        if not matching:
            raise ValueError(
                f'forced task {forced_task_id} is not ongoing',
            )
        return Action(
            toggle=not satellite.sensor.enabled,
            target_location=matching[0].coordinate,
        )

    def _resolve_forced_task_id(self) -> int:
        if self._forced_task_id is not None:
            return self._forced_task_id
        logits = getattr(self._base_algorithm, 'last_logits', None)
        task_ids = getattr(self._base_algorithm, 'last_task_ids', None)
        if not isinstance(logits, torch.Tensor
                          ) or not isinstance(task_ids, torch.Tensor):
            raise RuntimeError(
                'ranked candidate forcing requires base algorithm logits '
                'and task ids'
            )
        if logits.ndim != 3 or logits.shape[0] != 1:
            raise ValueError('base algorithm logits have an invalid shape')
        if logits.shape[-1] != task_ids.numel() + 1:
            raise ValueError('base algorithm task ids do not match logits')
        assert self._forced_candidate_rank is not None
        order = logits[0, self._satellite_index].argsort(descending=True)
        if self._forced_candidate_rank >= order.numel():
            raise IndexError('forced candidate rank is out of range')
        relative_index = int(order[self._forced_candidate_rank])
        return -1 if relative_index == 0 else int(task_ids[relative_index - 1])

    def _run_lengths(self, previous: list[int]) -> list[int]:
        lengths = []
        for satellite_index, task_id in enumerate(previous):
            length = 0
            for assignment in reversed(self._assignment_history):
                if assignment[satellite_index] != task_id:
                    break
                length += 1
            lengths.append(length)
        return lengths

    def _switch_counts(self, window: int, num_satellites: int) -> list[int]:
        history = self._assignment_history[-(window + 1):]
        counts = [0] * num_satellites
        for before, after in zip(history, history[1:]):
            for satellite_index in range(num_satellites):
                counts[satellite_index] += int(
                    before[satellite_index] != after[satellite_index]
                )
        return counts

    def _capture_decision_context(
        self,
        *,
        taskset: TaskSet,
        constellation: Constellation,
    ) -> None:
        num_satellites = len(constellation)
        logits = getattr(self._base_algorithm, 'last_logits', None)
        task_ids = getattr(self._base_algorithm, 'last_task_ids', None)
        if not isinstance(logits, torch.Tensor
                          ) or not isinstance(task_ids, torch.Tensor):
            logits = None
            task_ids = None
        previous = (
            self._assignment_history[-1].copy()
            if self._assignment_history else [-1] * num_satellites
        )
        run_lengths = self._run_lengths(previous)
        switches_30 = self._switch_counts(30, num_satellites)
        switches_60 = self._switch_counts(60, num_satellites)
        satellite_sensor_type, satellite_static = (
            constellation.static_to_tensor()
        )
        satellite_enabled, satellite_dynamic = (
            constellation.dynamic_to_tensor()
        )
        satellite_features = torch.cat((
            satellite_static.float(),
            satellite_dynamic.float(),
            satellite_enabled.float().unsqueeze(-1),
            satellite_sensor_type.float().unsqueeze(-1),
            torch.tensor(run_lengths).float().unsqueeze(-1),
            torch.tensor(switches_30).float().unsqueeze(-1),
            torch.tensor(switches_60).float().unsqueeze(-1),
        ), -1)

        task_sensor_type, task_static = taskset.to_tensor()
        task_static = task_static.float().clone()
        task_static[:, :2] -= self._timer.time
        full_taskset = getattr(self._task_manager, 'taskset', taskset)
        full_index = {
            int(task.id_): index
            for index, task in enumerate(full_taskset)
        }
        progress = torch.tensor([
            float(self._task_manager.progress[full_index[int(task.id_)]])
            for task in taskset
        ])
        progress_ratio = progress / taskset.durations.float().clamp_min(1.0)
        task_features = torch.cat((
            task_static,
            progress_ratio.unsqueeze(-1),
            task_sensor_type.float().unsqueeze(-1),
        ), -1)
        self._decision_context = {
            'previous_assignment': previous,
            'run_lengths': run_lengths,
            'switch_counts_30': switches_30,
            'switch_counts_60': switches_60,
            'ongoing_task_ids': None
            if task_ids is None else task_ids.tolist(),
            'actor_logits': None
            if logits is None else logits.squeeze(0).tolist(),
            'satellite_features': satellite_features.tolist(),
            'task_features': task_features.tolist(),
            'satellite_sensor_type': satellite_sensor_type.tolist(),
            'task_sensor_type': task_sensor_type.tolist(),
            'uses_is_visible_as_input': False,
        }

    def step(
        self,
        taskset: TaskSet,
        constellation: Constellation,
        earth_rotation: torch.Tensor,
    ) -> tuple[Actions, list[int]]:
        actions, assignment = self._base_algorithm.step(
            taskset,
            constellation,
            earth_rotation,
        )
        if self._timer.time != self._decision_time:
            self._assignment_history.append(list(assignment))
            return actions, assignment

        if not 0 <= self._satellite_index < len(constellation):
            raise IndexError('satellite_index is out of range')
        actions = Actions(actions)
        assignment = list(assignment)
        self.original_assignment = assignment.copy()
        self.original_task_id = assignment[self._satellite_index]
        self._capture_decision_context(
            taskset=taskset,
            constellation=constellation,
        )
        self.decision_state_signature = self._state_signature(
            constellation=constellation,
            taskset=taskset,
            earth_rotation=earth_rotation,
            assignment=assignment,
        )
        forced_task_id = self._resolve_forced_task_id()
        actions[self._satellite_index] = self._forced_action(
            taskset,
            constellation,
            forced_task_id,
        )
        assignment[self._satellite_index] = forced_task_id
        self.applied_task_id = forced_task_id
        self._assignment_history.append(assignment.copy())
        self.override_applied = True
        return actions, assignment

    @property
    def decision_context(self) -> dict[str, Any]:
        if self._decision_context is None:
            raise RuntimeError('decision context is not ready')
        return self._decision_context


class ControlledCommitmentAlgorithm(ControlledActionAlgorithm):
    """从同一状态强制一颗卫星保持候选动作若干秒。"""

    def __init__(
        self,
        *args,
        commitment_seconds: int,
        **kwargs,
    ) -> None:
        if commitment_seconds not in (1, 5, 15, 30, 60):
            raise ValueError(
                'commitment_seconds must be one of (1, 5, 15, 30, 60)'
            )
        super().__init__(*args, **kwargs)
        if self._decision_time < 0:
            raise ValueError('decision_time must be non-negative')
        self.requested_commitment_seconds = commitment_seconds
        self.actual_commitment_seconds = 0
        self.interruption_reason: str | None = None

    def _capture_start(
        self,
        *,
        taskset: TaskSet,
        constellation: Constellation,
        earth_rotation: torch.Tensor,
        assignment: list[int],
    ) -> None:
        if not 0 <= self._satellite_index < len(constellation):
            raise IndexError('satellite_index is out of range')
        self.original_assignment = assignment.copy()
        self.original_task_id = assignment[self._satellite_index]
        self._capture_decision_context(
            taskset=taskset,
            constellation=constellation,
        )
        self.decision_state_signature = self._state_signature(
            constellation=constellation,
            taskset=taskset,
            earth_rotation=earth_rotation,
            assignment=assignment,
        )
        self.applied_task_id = self._resolve_forced_task_id()

    def step(
        self,
        taskset: TaskSet,
        constellation: Constellation,
        earth_rotation: torch.Tensor,
    ) -> tuple[Actions, list[int]]:
        actions, raw_assignment = self._base_algorithm.step(
            taskset,
            constellation,
            earth_rotation,
        )
        actions = Actions(actions)
        assignment = list(raw_assignment)
        time = int(self._timer.time)
        if time < self._decision_time:
            self._assignment_history.append(assignment.copy())
            return actions, assignment

        if time == self._decision_time and self.applied_task_id is None:
            self._capture_start(
                taskset=taskset,
                constellation=constellation,
                earth_rotation=earth_rotation,
                assignment=assignment,
            )

        if self.applied_task_id is None:
            self._assignment_history.append(assignment.copy())
            return actions, assignment
        if self.interruption_reason is not None:
            self._assignment_history.append(assignment.copy())
            return actions, assignment
        if time - self._decision_time >= self.requested_commitment_seconds:
            self.interruption_reason = 'expired'
            self._assignment_history.append(assignment.copy())
            return actions, assignment
        ongoing_task_ids = {int(value) for value in taskset.ids.tolist()}
        if (
            self.applied_task_id >= 0
            and self.applied_task_id not in ongoing_task_ids
        ):
            self.interruption_reason = 'task_unavailable'
            self._assignment_history.append(assignment.copy())
            return actions, assignment

        actions[self._satellite_index] = self._forced_action(
            taskset,
            constellation,
            self.applied_task_id,
        )
        assignment[self._satellite_index] = self.applied_task_id
        self.actual_commitment_seconds += 1
        self.override_applied = True
        self._assignment_history.append(assignment.copy())
        return actions, assignment


class LocalWindowCallback(BaseCallback):
    """按 ``action[t] -> outcome[t+1]`` 对齐采集局部窗口。"""

    def __init__(
        self,
        *args,
        decision_time: int,
        horizon: int | None = None,
        horizons: tuple[int, ...] | list[int] | None = None,
        target_satellite_index: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if decision_time < 1:
            raise ValueError('decision_time must be at least 1')
        if (horizon is None) == (horizons is None):
            raise ValueError('provide exactly one of horizon or horizons')
        requested = (horizon, ) if horizon is not None else tuple(
            horizons or ()
        )
        if not requested or min(requested) <= 0:
            raise ValueError('horizons must be positive')
        self._decision_time = decision_time
        self._horizons = tuple(sorted(set(int(value) for value in requested)))
        self._horizon = max(self._horizons)
        self._target_satellite_index = target_satellite_index

    def before_run(self) -> None:
        self._previous_assignment: torch.Tensor | None = None
        self._assignments: list[torch.Tensor] = []
        self._progress: list[torch.Tensor] = []
        self._succeeded: list[torch.Tensor] = []
        self._direct_visible: list[torch.Tensor] = []
        self._assignment_before: torch.Tensor | None = None
        self._assignment_after: torch.Tensor | None = None
        self._succeeded_before: torch.Tensor | None = None
        self._succeeded_after: torch.Tensor | None = None
        self._summary: dict[str, Any] | None = None
        self._summaries: dict[int, dict[str, Any]] | None = None
        num_tasks = len(self.controller.task_manager.taskset)
        self._global_max_progress = torch.zeros(num_tasks)
        self._completion_time = torch.full((num_tasks, ), float('inf'))
        self._max_progress_snapshots: list[torch.Tensor] = []
        self._completion_time_snapshots: list[torch.Tensor] = []

    def _aligned_direct_visibility(
        self,
        assignment: torch.Tensor,
    ) -> torch.Tensor:
        is_visible = self.controller.memo['is_visible']
        task_ids = self.controller.task_manager.taskset.ids
        direct = torch.zeros_like(assignment, dtype=torch.bool)
        for satellite_index, task_id in enumerate(assignment.tolist()):
            if task_id < 0:
                continue
            matching = torch.nonzero(task_ids == task_id, as_tuple=False)
            if matching.numel():
                direct[satellite_index] = is_visible[
                    satellite_index,
                    int(matching[0, 0]),
                ]
        return direct

    def after_step(self) -> None:
        time = self.controller.environment.timer.time
        assignment = torch.tensor(self.controller.memo['assignment'])
        current_progress = self.controller.task_manager.progress.float()
        current_succeeded = self.controller.task_manager.succeeded_flags
        self._global_max_progress = torch.maximum(
            self._global_max_progress,
            current_progress,
        )
        self._completion_time[current_succeeded] = torch.minimum(
            self._completion_time[current_succeeded],
            torch.tensor(float(time)),
        )

        if time == self._decision_time:
            if self._previous_assignment is None:
                raise RuntimeError('missing assignment before decision')
            self._assignment_before = self._previous_assignment.clone()
            self._succeeded_before = (
                self.controller.task_manager.succeeded_flags.clone()
            )
            self._progress.append(
                self.controller.task_manager.progress.clone(),
            )
            self._succeeded.append(
                self.controller.task_manager.succeeded_flags.clone(),
            )
            self._max_progress_snapshots.append(
                self._global_max_progress.clone(),
            )
            self._completion_time_snapshots.append(
                self._completion_time.clone(),
            )
            self._assignments.append(assignment.clone())
        elif self._decision_time < time <= (
            self._decision_time + self._horizon
        ):
            if self._previous_assignment is None:
                raise RuntimeError('missing action for aligned outcome')
            self._direct_visible.append(
                self._aligned_direct_visibility(self._previous_assignment),
            )
            self._progress.append(
                self.controller.task_manager.progress.clone(),
            )
            self._succeeded.append(
                self.controller.task_manager.succeeded_flags.clone(),
            )
            self._max_progress_snapshots.append(
                self._global_max_progress.clone(),
            )
            self._completion_time_snapshots.append(
                self._completion_time.clone(),
            )
            if time < self._decision_time + self._horizon:
                self._assignments.append(assignment.clone())
            else:
                self._assignment_after = assignment.clone()
                self._succeeded_after = (
                    self.controller.task_manager.succeeded_flags.clone()
                )

        self._previous_assignment = assignment.clone()

    def after_run(self) -> None:
        required = (
            self._assignment_before,
            self._assignment_after,
            self._succeeded_before,
            self._succeeded_after,
        )
        if any(value is None for value in required):
            raise RuntimeError('local window did not finish')
        if len(self._assignments) != self._horizon:
            raise RuntimeError('local action window has an invalid length')
        if len(self._progress) != self._horizon + 1:
            raise RuntimeError('local progress window has an invalid length')
        if len(self._direct_visible) != self._horizon:
            raise RuntimeError('local visibility window has an invalid length')
        if len(self._succeeded) != self._horizon + 1:
            raise RuntimeError('local success window has an invalid length')
        if len(self._max_progress_snapshots) != self._horizon + 1:
            raise RuntimeError(
                'local prefix metric window has an invalid length'
            )

        satellites = self.controller.environment.get_constellation().sort()
        sensor_power = torch.tensor([
            satellite.sensor.power for satellite in satellites
        ])
        assignments = torch.stack(self._assignments)
        progress = torch.stack(self._progress)
        succeeded = torch.stack(self._succeeded)
        direct_visible = torch.stack(self._direct_visible)
        self._summaries = {}
        for horizon in self._horizons:
            assignment_after = (
                self._assignment_after
                if horizon == self._horizon else assignments[horizon]
            )
            summary = summarize_local_window(
                assignments=assignments[:horizon],
                assignment_before=self._assignment_before,
                assignment_after=assignment_after,
                progress=progress[:horizon + 1],
                succeeded_before=self._succeeded_before,
                succeeded_after=succeeded[horizon],
                direct_visible=direct_visible[:horizon],
                durations=self.controller.task_manager.taskset.durations,
                sensor_power=sensor_power,
                target_satellite_index=self._target_satellite_index,
            )
            summary['prefix_metrics'] = summarize_prefix_paper_metrics(
                max_progress=self._max_progress_snapshots[horizon],
                succeeded=succeeded[horizon],
                completion_time=self._completion_time_snapshots[horizon],
                release_times=(
                    self.controller.task_manager.taskset.release_times
                ),
                durations=self.controller.task_manager.taskset.durations,
                local_pc_wh=float(summary['pc_wh']),
            )
            self._summaries[horizon] = summary
        self._summary = self._summaries[self._horizon]
        self.controller.memo['local_window_summary'] = self._summary
        self.controller.memo['local_window_summaries'] = self._summaries

    @property
    def summary(self) -> dict[str, Any]:
        if self._summary is None:
            raise RuntimeError('local window summary is not ready')
        return self._summary

    @property
    def summaries(self) -> dict[int, dict[str, Any]]:
        if self._summaries is None:
            raise RuntimeError('local window summaries are not ready')
        return self._summaries


def _count_one_second_runs(
    assignments: torch.Tensor,
    assignment_before: torch.Tensor,
    assignment_after: torch.Tensor,
) -> tuple[int, torch.Tensor]:
    extended = torch.cat([
        assignment_before.unsqueeze(0),
        assignments,
        assignment_after.unsqueeze(0),
    ])
    middle = extended[1:-1]
    pulse = ((middle >= 0)
             & (middle != extended[:-2])
             & (middle != extended[2:]))
    return int(pulse.sum()), pulse.sum(0)


def summarize_local_window(
    *,
    assignments: torch.Tensor,
    assignment_before: torch.Tensor,
    assignment_after: torch.Tensor,
    progress: torch.Tensor,
    succeeded_before: torch.Tensor,
    succeeded_after: torch.Tensor,
    direct_visible: torch.Tensor,
    durations: torch.Tensor,
    sensor_power: torch.Tensor,
    target_satellite_index: int,
) -> dict[str, Any]:
    """汇总局部窗口，不提前把不同物理量压成一个奖励。"""

    if assignments.ndim != 2:
        raise ValueError('assignments must have shape (horizon, satellites)')
    horizon, num_satellites = assignments.shape
    if progress.shape[0] != horizon + 1:
        raise ValueError('progress must include start plus every outcome')
    if direct_visible.shape != assignments.shape:
        raise ValueError('direct_visible must align with assignments')
    if assignment_before.shape != (num_satellites, ):
        raise ValueError('assignment_before has an invalid shape')
    if assignment_after.shape != (num_satellites, ):
        raise ValueError('assignment_after has an invalid shape')

    active = assignments >= 0
    new_succeeded = succeeded_after & ~succeeded_before
    max_progress = progress.max(0).values
    progress_gain = (max_progress - progress[0]).clamp_min(0)
    partial_progress_gain = (progress_gain / durations).sum().item()

    working_power = active * sensor_power.unsqueeze(0)
    comparisons = torch.cat([
        assignment_before.unsqueeze(0),
        assignments,
    ])
    switches_by_satellite = (comparisons[1:] != comparisons[:-1]).sum(0)
    one_second_runs, one_second_by_satellite = _count_one_second_runs(
        assignments,
        assignment_before,
        assignment_after,
    )

    redundant = 0
    for row in assignments:
        selected = row[row >= 0]
        redundant += int(selected.numel() - selected.unique().numel())

    return {
        'horizon': horizon,
        'completed_tasks': int(new_succeeded.sum()),
        'completed_duration': int(durations[new_succeeded].sum()),
        'partial_progress_gain': partial_progress_gain,
        'working_satellite_seconds': int(active.sum()),
        'pc_wh': float(working_power.sum().item() / 3600.0),
        'switches': int(switches_by_satellite.sum()),
        'target_satellite_switches': int(
            switches_by_satellite[target_satellite_index]
        ),
        'one_second_runs': one_second_runs,
        'target_satellite_one_second_runs': int(
            one_second_by_satellite[target_satellite_index]
        ),
        'direct_visible_satellite_seconds': int(direct_visible.sum()),
        'target_satellite_direct_visible_seconds': int(
            direct_visible[:, target_satellite_index].sum()
        ),
        'redundant_satellite_seconds': redundant,
    }


def summarize_prefix_paper_metrics(
    *,
    max_progress: torch.Tensor,
    succeeded: torch.Tensor,
    completion_time: torch.Tensor,
    release_times: torch.Tensor,
    durations: torch.Tensor,
    local_pc_wh: float,
) -> dict[str, float | None]:
    """计算决策分支终点的 paper-aligned 前缀快照。

    ``prefix_cost`` 使用完整完成质量和截至该时刻的 TAT，但功耗只加入
    决策窗口内的增量。两个受控分支在决策前完全相同，因此省略共同功耗不会改变
    分支排序。尚无任何完成或进度时 ``quality=0``，该样本不用于排序训练。
    """

    if not (
        max_progress.ndim == succeeded.ndim == completion_time.ndim ==
        release_times.ndim == durations.ndim == 1
    ):
        raise ValueError('prefix metric tensors must be one-dimensional')
    if not (
        max_progress.shape == succeeded.shape == completion_time.shape ==
        release_times.shape == durations.shape
    ):
        raise ValueError('prefix metric tensors must have equal shapes')
    if (durations <= 0).any() or local_pc_wh < 0:
        raise ValueError('durations must be positive and power non-negative')

    progress_ratio = (max_progress.float() / durations.float()).clamp(0, 1)
    progress_ratio = torch.where(succeeded, 1.0, progress_ratio)
    cr = float(succeeded.float().mean().item())
    pcr = float(progress_ratio.mean().item())
    wcr = float(
        durations.float()[succeeded].sum().item()
        / durations.float().sum().item()
    )
    quality = 0.6 * cr + 0.2 * pcr + 0.2 * wcr
    if succeeded.any():
        tat_s = float((completion_time[succeeded]
                       - release_times[succeeded]).mean().item())
    else:
        tat_s = 0.0
    prefix_cost = (
        None if quality <= 0 else 1.0 / quality + tat_s / 700.0
        + local_pc_wh / 100.0
    )
    return {
        'cr': cr,
        'pcr': pcr,
        'wcr': wcr,
        'quality': quality,
        'tat_s': tat_s,
        'local_pc_wh': float(local_pc_wh),
        'prefix_cost': prefix_cost,
    }
