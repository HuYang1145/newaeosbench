"""V2 事件状态机与单条真实 Basilisk 轨迹之间的适配层。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import pathlib
from typing import Any, NamedTuple, Protocol

import torch

from constellation import (
    CONSTELLATIONS_ROOT,
    MAX_TIME_STEP,
    STATISTICS_PATH,
    TASKSETS_ROOT,
    TIMESTAMP,
    TaskManager,
)
from constellation.data import (
    Action,
    Actions,
    Constellation,
    Task,
    TaskSet,
)
from constellation.environments import BasiliskEnvironment
from constellation.new_transformers.dataset import Statistics

from .observation import EventPolicyObservation
from .reward import (
    completion_potential,
    completion_task_weights,
    terminal_completion_quality,
)
from .runtime_state import EventRuntimeState, RuntimeSnapshot
from .state import EventStateTensors
from .transition import JointEventAction


class CompletionSnapshot(NamedTuple):
    progress: torch.Tensor
    required_duration: torch.Tensor
    completed: torch.Tensor

    def validate(self) -> None:
        if (
            self.progress.ndim != 1
            or self.progress.shape != self.required_duration.shape
            or self.progress.shape != self.completed.shape
            or self.progress.numel() == 0
        ):
            raise ValueError('completion snapshot tensors must share one task axis')
        if self.completed.dtype != torch.bool:
            raise ValueError('completion flags must use bool dtype')
        if (
            not self.progress.is_floating_point()
            or not self.required_duration.is_floating_point()
            or not torch.isfinite(self.progress).all()
            or not torch.isfinite(self.required_duration).all()
            or (self.progress < 0).any()
            or (self.required_duration <= 0).any()
        ):
            raise ValueError('completion values must be finite and physically valid')


class EventPhysicsBackend(Protocol):
    num_satellites: int
    time_step: int

    @property
    def done(self) -> bool:
        raise NotImplementedError

    def snapshot(self) -> RuntimeSnapshot:
        raise NotImplementedError

    def completion_snapshot(self) -> CompletionSnapshot:
        raise NotImplementedError

    def apply_assignments(self, global_task_ids: Sequence[int]) -> None:
        raise NotImplementedError

    def step_one_second(self) -> None:
        raise NotImplementedError

    def build_observation(
        self,
        state: EventStateTensors,
        statistics: Statistics,
    ) -> EventPolicyObservation:
        raise NotImplementedError

    def state_dict(self) -> Mapping[str, Any]:
        raise NotImplementedError


class RuntimeStep(NamedTuple):
    observation: EventPolicyObservation | None
    reward: float
    delta_t: int
    done: bool
    final_quality: float | None
    invalid_action_count: int


class BasiliskEventRuntime:
    """只在事件点返回，把中间物理秒封装在一次 ``step`` 内。"""

    def __init__(
        self,
        *,
        backend: EventPhysicsBackend,
        statistics: Statistics,
        safety_review_seconds: int = 5,
    ) -> None:
        self.backend = backend
        self._statistics = statistics
        self._state = EventRuntimeState(
            num_satellites=backend.num_satellites,
            safety_review_seconds=safety_review_seconds,
        )
        self._task_weights: torch.Tensor | None = None
        self._previous_event_potential: torch.Tensor | None = None
        self._total_reward = 0.
        self._terminal = False
        self._reward_reconstruction_error: float | None = None
        self._current_observation: EventPolicyObservation | None = None

    @property
    def reward_reconstruction_error(self) -> float:
        if self._reward_reconstruction_error is None:
            raise RuntimeError('reward reconstruction is only known after terminal')
        return self._reward_reconstruction_error

    @property
    def current_global_task_ids(self) -> tuple[int, ...]:
        return self._state.current_global_task_ids

    @property
    def current_observation(self) -> EventPolicyObservation:
        if self._current_observation is None:
            raise RuntimeError('runtime has no current policy observation')
        return self._current_observation

    def _completion_potential(
        self,
        completion: CompletionSnapshot,
    ) -> torch.Tensor:
        if self._task_weights is None:
            raise RuntimeError('runtime reward has not been initialized')
        return completion_potential(
            completion.progress.unsqueeze(0),
            completion.required_duration.unsqueeze(0),
            self._task_weights.unsqueeze(0),
        ).squeeze(0)

    def _idle_action(self) -> JointEventAction:
        shape = (1, self.backend.num_satellites)
        return JointEventAction(
            terminate=torch.zeros(shape, dtype=torch.bool),
            task_indices=torch.full(shape, -1, dtype=torch.long),
            commitment_indices=torch.full(shape, -1, dtype=torch.long),
        )

    def reset(self) -> EventPolicyObservation:
        if self._task_weights is not None:
            raise RuntimeError('a runtime instance can only execute one scene')
        completion = self.backend.completion_snapshot()
        completion.validate()
        self._task_weights = completion_task_weights(
            completion.required_duration,
        )
        self._previous_event_potential = self._completion_potential(completion)
        snapshot = self.backend.snapshot()
        event = self._state.initial_event(snapshot)

        while not snapshot.ongoing_global_task_ids:
            if self.backend.done:
                raise RuntimeError('scene ended before any task became ongoing')
            self._state.apply_joint_action(self._idle_action(), ())
            self.backend.apply_assignments(self._state.current_global_task_ids)
            self.backend.step_one_second()
            snapshot = self.backend.snapshot()
            event = self._state.advance_one_second(snapshot)
            if event.requires_policy and not snapshot.ongoing_global_task_ids:
                continue

        observation = self.backend.build_observation(
            event.state,
            self._statistics,
        )
        observation.validate()
        self._current_observation = observation
        return observation

    def step(self, action: JointEventAction) -> RuntimeStep:
        if self._terminal:
            raise RuntimeError('runtime scene has already finished')
        if self._previous_event_potential is None:
            raise RuntimeError('runtime must be reset before stepping')
        snapshot = self.backend.snapshot()
        self._state.apply_joint_action(
            action,
            snapshot.ongoing_global_task_ids,
        )
        self.backend.apply_assignments(self._state.current_global_task_ids)
        start_time = self.backend.time_step

        while True:
            previous_time = self.backend.time_step
            self.backend.step_one_second()
            if self.backend.time_step != previous_time + 1:
                raise RuntimeError('backend time did not advance by one second')
            snapshot = self.backend.snapshot()
            event = self._state.advance_one_second(snapshot)
            completion = self.backend.completion_snapshot()
            completion.validate()
            potential = self._completion_potential(completion)

            if self.backend.done:
                terminal_quality = terminal_completion_quality(
                    completion.progress.unsqueeze(0),
                    completion.required_duration.unsqueeze(0),
                    completion.completed.unsqueeze(0),
                ).squeeze(0)
                reward = (
                    potential
                    - self._previous_event_potential
                    + terminal_quality
                    - potential
                )
                reward_value = float(reward.item())
                self._total_reward += reward_value
                self._previous_event_potential = potential
                self._terminal = True
                final_quality = float(terminal_quality.item())
                self._reward_reconstruction_error = abs(
                    self._total_reward - final_quality
                )
                observation = None
                self._current_observation = None
                return RuntimeStep(
                    observation=observation,
                    reward=reward_value,
                    delta_t=self.backend.time_step - start_time,
                    done=True,
                    final_quality=final_quality,
                    invalid_action_count=0,
                )

            if event.requires_policy:
                if not snapshot.ongoing_global_task_ids:
                    self._state.apply_joint_action(self._idle_action(), ())
                    self.backend.apply_assignments(
                        self._state.current_global_task_ids
                    )
                    continue
                reward = potential - self._previous_event_potential
                reward_value = float(reward.item())
                self._total_reward += reward_value
                self._previous_event_potential = potential
                observation = self.backend.build_observation(
                    event.state,
                    self._statistics,
                )
                observation.validate()
                self._current_observation = observation
                return RuntimeStep(
                    observation=observation,
                    reward=reward_value,
                    delta_t=self.backend.time_step - start_time,
                    done=False,
                    final_quality=None,
                    invalid_action_count=0,
                )

    def state_dict(self) -> dict[str, Any]:
        backend_state = self.backend.state_dict()
        return {
            'version': 1,
            'backend': dict(backend_state),
            'event_runtime_state': self._state.state_dict(),
            'task_weights': (
                None
                if self._task_weights is None
                else self._task_weights.detach().cpu().clone()
            ),
            'previous_event_potential': (
                None
                if self._previous_event_potential is None
                else self._previous_event_potential.detach().cpu().clone()
            ),
            'total_reward': self._total_reward,
            'terminal': self._terminal,
            'reward_reconstruction_error': self._reward_reconstruction_error,
            'current_observation': self._current_observation,
        }

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, Any],
        *,
        statistics: Statistics,
    ) -> 'BasiliskEventRuntime':
        if state_dict.get('version') != 1:
            raise ValueError('event runtime checkpoint version does not match')
        backend_state = state_dict.get('backend')
        runtime_state = state_dict.get('event_runtime_state')
        if not isinstance(backend_state, Mapping):
            raise ValueError('event runtime backend checkpoint is invalid')
        if not isinstance(runtime_state, Mapping):
            raise ValueError('event runtime state checkpoint is invalid')
        backend = BasiliskSceneBackend.from_state_dict(backend_state)
        restored_state = EventRuntimeState.from_state_dict(runtime_state)
        runtime = cls(
            backend=backend,
            statistics=statistics,
            safety_review_seconds=int(runtime_state['safety_review_seconds']),
        )
        runtime._state = restored_state
        task_weights = state_dict.get('task_weights')
        previous_potential = state_dict.get('previous_event_potential')
        if not isinstance(task_weights, torch.Tensor):
            raise ValueError('event runtime task weights are invalid')
        if not isinstance(previous_potential, torch.Tensor):
            raise ValueError('event runtime reward potential is invalid')
        runtime._task_weights = task_weights.clone().cpu()
        runtime._previous_event_potential = previous_potential.clone().cpu()
        runtime._total_reward = float(state_dict.get('total_reward'))
        runtime._terminal = bool(state_dict.get('terminal'))
        reconstruction_error = state_dict.get('reward_reconstruction_error')
        runtime._reward_reconstruction_error = (
            None
            if reconstruction_error is None
            else float(reconstruction_error)
        )
        saved_observation = state_dict.get('current_observation')
        if runtime._terminal:
            runtime._current_observation = saved_observation
            return runtime
        if not isinstance(saved_observation, EventPolicyObservation):
            raise ValueError('event runtime current observation is invalid')
        observation = backend.build_observation(
            restored_state.last_event_state,
            statistics,
        )
        for name in observation._fields[:-1]:
            torch.testing.assert_close(
                getattr(observation, name),
                getattr(saved_observation, name),
                rtol=0,
                atol=0,
            )
        for name in observation.event_state._fields:
            torch.testing.assert_close(
                getattr(observation.event_state, name),
                getattr(saved_observation.event_state, name),
                rtol=0,
                atol=0,
            )
        runtime._current_observation = observation
        return runtime


class BasiliskSceneBackend:
    """一个 scene 对应一个 Basilisk 实例，不产生候选分支。"""

    def __init__(
        self,
        *,
        environment: BasiliskEnvironment,
        taskset: TaskSet[Task],
        max_time_step: int,
        split: str | None = None,
        scene_id: int | None = None,
    ) -> None:
        if max_time_step <= environment.timer.time:
            raise ValueError('max time step must be after the initial time')
        self._environment = environment
        self._taskset = taskset
        self._task_manager = TaskManager(
            timer=environment.timer,
            taskset=taskset,
        )
        self._max_time_step = max_time_step
        self._split = split
        self._scene_id = scene_id
        self.num_satellites = environment.num_satellites
        self._task_by_id = {task.id_: task for task in taskset}
        if len(self._task_by_id) != len(taskset):
            raise ValueError('task IDs must be unique within a scene')
        self._assignments = [-1] * self.num_satellites
        self._max_progress = torch.zeros(len(taskset), dtype=torch.float32)
        self._previous_ongoing_ids = set(self._ongoing_ids())
        self._released_ids: tuple[int, ...] = ()
        self._closed_ids: tuple[int, ...] = ()
        self._action_events: list[tuple[int, tuple[int, ...]]] = []
        self._completion_time = torch.full(
            (len(taskset),),
            float('inf'),
            dtype=torch.float32,
        )
        self._working_time_steps = torch.zeros(
            self.num_satellites,
            dtype=torch.float32,
        )
        self._sensor_power = torch.tensor([
            float(satellite.sensor.power)
            for satellite in environment.get_constellation().sort()
        ], dtype=torch.float32)

    @classmethod
    def from_scene_id(
        cls,
        *,
        split: str,
        scene_id: int,
        max_time_step: int = MAX_TIME_STEP,
    ) -> 'BasiliskSceneBackend':
        if scene_id < 0:
            raise ValueError('scene id must be non-negative')
        relative = pathlib.Path(split) / f'{scene_id // 1000:02}' / (
            f'{scene_id:05}.json'
        )
        constellation_path = CONSTELLATIONS_ROOT / relative
        taskset_path = TASKSETS_ROOT / relative
        if not constellation_path.is_file():
            raise FileNotFoundError(
                f'constellation scene not found: {constellation_path}'
            )
        if not taskset_path.is_file():
            raise FileNotFoundError(f'task scene not found: {taskset_path}')
        constellation = Constellation.load(str(constellation_path))
        taskset: TaskSet[Task] = TaskSet.load(str(taskset_path))
        environment = BasiliskEnvironment(
            start_time=0,
            standard_time_init=TIMESTAMP,
            constellation=constellation,
            all_tasks=taskset,
        )
        return cls(
            environment=environment,
            taskset=taskset,
            max_time_step=max_time_step,
            split=split,
            scene_id=scene_id,
        )

    @property
    def time_step(self) -> int:
        return self._environment.timer.time

    @property
    def done(self) -> bool:
        return bool(
            self._task_manager.all_closed
            or self.time_step >= self._max_time_step
        )

    def _ongoing_ids(self) -> list[int]:
        return [
            task.id_
            for task, ongoing in zip(
                self._taskset,
                self._task_manager.ongoing_flags,
            )
            if bool(ongoing)
        ]

    def apply_assignments(self, global_task_ids: Sequence[int]) -> None:
        if len(global_task_ids) != self.num_satellites:
            raise ValueError('assignment must contain one task per satellite')
        constellation = self._environment.get_constellation().sort()
        actions: list[Action] = []
        checked_assignments: list[int] = []
        for satellite, raw_task_id in zip(constellation, global_task_ids):
            task_id = int(raw_task_id)
            task = self._task_by_id.get(task_id)
            if task_id >= 0 and task is None:
                raise ValueError('assignment contains an unknown global task id')
            desired_enabled = task is not None
            actions.append(Action(
                toggle=bool(desired_enabled != satellite.sensor.enabled),
                target_location=(None if task is None else task.coordinate),
            ))
            checked_assignments.append(task_id)
        self._environment.take_actions(Actions(actions))
        self._assignments = checked_assignments
        self._action_events.append((
            self.time_step,
            tuple(checked_assignments),
        ))

    def step_one_second(self) -> None:
        if self.done:
            raise RuntimeError('cannot advance a completed Basilisk scene')
        before = set(self._ongoing_ids())
        self._working_time_steps += torch.tensor(
            [task_id >= 0 for task_id in self._assignments],
            dtype=torch.float32,
        )
        self._environment.timer.step()
        self._environment.step()
        visibility = self._environment.is_visible(self._taskset)
        self._task_manager.record(visibility)
        newly_succeeded = (
            self._task_manager.succeeded_flags
            & torch.isinf(self._completion_time)
        )
        self._completion_time[newly_succeeded] = float(self.time_step)
        self._max_progress = torch.maximum(
            self._max_progress,
            self._task_manager.progress.to(torch.float32),
        )
        after = set(self._ongoing_ids())
        self._released_ids = tuple(sorted(after - before))
        self._closed_ids = tuple(sorted(before - after))
        self._previous_ongoing_ids = after

    def completion_snapshot(self) -> CompletionSnapshot:
        return CompletionSnapshot(
            progress=self._max_progress.clone(),
            required_duration=self._taskset.durations.to(torch.float32),
            completed=self._task_manager.succeeded_flags.clone(),
        )

    def operational_metrics(self) -> dict[str, float]:
        """按论文评估单位返回成功任务 TAT 和传感器功耗。"""

        succeeded = self._task_manager.succeeded_flags
        if bool(succeeded.any()):
            release_times = self._taskset.release_times.to(torch.float32)
            tat_s = float(
                (
                    self._completion_time[succeeded]
                    - release_times[succeeded]
                ).mean().item()
            )
        else:
            tat_s = float('inf')
        pc_wh = float(
            torch.sum(
                self._working_time_steps * self._sensor_power,
            ).item()
            / 3600.0
        )
        return {
            'TAT_s': tat_s,
            'PC_Wh': pc_wh,
        }

    def snapshot(self) -> RuntimeSnapshot:
        ongoing_flags = self._task_manager.ongoing_flags
        ongoing_tasks = [
            task
            for task, flag in zip(self._taskset, ongoing_flags)
            if bool(flag)
        ]
        ongoing_ids = tuple(task.id_ for task in ongoing_tasks)
        constellation = self._environment.get_constellation().sort()
        task_compatibility = torch.tensor([
            [satellite.sensor.type_ == task.sensor_type for task in ongoing_tasks]
            for satellite in constellation
        ], dtype=torch.bool)
        if not ongoing_tasks:
            task_compatibility = torch.empty(
                self.num_satellites,
                0,
                dtype=torch.bool,
            )
        ongoing_id_set = set(ongoing_ids)
        assignment_valid = torch.tensor([
            task_id < 0
            or (
                task_id in ongoing_id_set
                and constellation[satellite_id].sensor.type_
                == self._task_by_id[task_id].sensor_type
            )
            for satellite_id, task_id in enumerate(self._assignments)
        ], dtype=torch.bool)
        return RuntimeSnapshot(
            time_step=self.time_step,
            ongoing_global_task_ids=ongoing_ids,
            task_progress=self._task_manager.progress[
                ongoing_flags
            ].to(torch.float32),
            task_required_duration=torch.tensor(
                [task.duration for task in ongoing_tasks],
                dtype=torch.float32,
            ),
            task_deadline_slack=torch.tensor(
                [max(task.due_time - self.time_step, 0) for task in ongoing_tasks],
                dtype=torch.float32,
            ),
            task_compatibility=task_compatibility,
            assignment_valid=assignment_valid,
            released_global_task_ids=self._released_ids,
            closed_global_task_ids=self._closed_ids,
        )

    def build_observation(
        self,
        state: EventStateTensors,
        statistics: Statistics,
    ) -> EventPolicyObservation:
        ongoing = self._task_manager.ongoing_tasks
        if not ongoing:
            raise ValueError('policy observation requires an ongoing task')
        constellation = self._environment.get_constellation()
        satellite_sensor_type, satellite_static = constellation.static_to_tensor()
        satellite_enabled, satellite_dynamic = constellation.dynamic_to_tensor()
        satellite_data = torch.cat((satellite_static, satellite_dynamic), dim=-1)
        satellite_data = (
            satellite_data.to(torch.float32) - statistics.constellation_mean
        ) / (statistics.constellation_std + 1e-6)

        task_sensor_type, task_static = ongoing.to_tensor()
        task_static = task_static.to(torch.float32)
        task_static[:, 0] -= self.time_step
        task_static[:, 1] -= self.time_step
        task_progress = self._task_manager.progress[
            self._task_manager.ongoing_flags
        ].to(torch.float32).unsqueeze(-1)
        task_data = torch.cat((task_static, task_progress), dim=-1)
        task_data = (
            task_data - statistics.taskset_mean
        ) / (statistics.taskset_std + 1e-6)

        observation = EventPolicyObservation(
            time_steps=torch.tensor([self.time_step], dtype=torch.long),
            constellation_sensor_type=(
                satellite_sensor_type.to(torch.long) - 1
            ).unsqueeze(0),
            constellation_sensor_enabled=satellite_enabled.to(
                torch.long
            ).unsqueeze(0),
            constellation_data=satellite_data.unsqueeze(0),
            constellation_mask=torch.ones(
                1,
                self.num_satellites,
                dtype=torch.bool,
            ),
            tasks_sensor_type=(
                task_sensor_type.to(torch.long) - 1
            ).unsqueeze(0),
            tasks_data=task_data.unsqueeze(0),
            tasks_mask=torch.ones(1, len(ongoing), dtype=torch.bool),
            event_state=state,
        )
        observation.validate()
        return observation

    def state_dict(self) -> dict[str, Any]:
        if self._split is None or self._scene_id is None:
            raise RuntimeError('only scene-id backends can be checkpointed')
        completion = self.completion_snapshot()
        return {
            'version': 1,
            'split': self._split,
            'scene_id': self._scene_id,
            'max_time_step': self._max_time_step,
            'time_step': self.time_step,
            'action_events': tuple(self._action_events),
            'max_progress': completion.progress,
            'completed': completion.completed,
            'completion_time': self._completion_time.clone(),
            'working_time_steps': self._working_time_steps.clone(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, Any],
    ) -> 'BasiliskSceneBackend':
        if state_dict.get('version') != 1:
            raise ValueError('Basilisk backend checkpoint version does not match')
        split = state_dict.get('split')
        scene_id = state_dict.get('scene_id')
        max_time_step = state_dict.get('max_time_step')
        target_time = state_dict.get('time_step')
        action_events = state_dict.get('action_events')
        if not isinstance(split, str) or not split:
            raise ValueError('Basilisk backend split is invalid')
        if not isinstance(scene_id, int) or scene_id < 0:
            raise ValueError('Basilisk backend scene id is invalid')
        if not isinstance(max_time_step, int) or max_time_step <= 0:
            raise ValueError('Basilisk backend max time is invalid')
        if (
            not isinstance(target_time, int)
            or target_time < 0
            or target_time > max_time_step
        ):
            raise ValueError('Basilisk backend target time is invalid')
        if not isinstance(action_events, (list, tuple)):
            raise ValueError('Basilisk backend action trace is invalid')
        checked_events: list[tuple[int, tuple[int, ...]]] = []
        previous_time = -1
        for item in action_events:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError('Basilisk action event has invalid structure')
            time_step, assignments = item
            if (
                not isinstance(time_step, int)
                or time_step < previous_time
                or time_step > target_time
                or not isinstance(assignments, (list, tuple))
            ):
                raise ValueError('Basilisk action event has invalid value')
            checked_events.append((
                time_step,
                tuple(int(value) for value in assignments),
            ))
            previous_time = time_step

        backend = cls.from_scene_id(
            split=split,
            scene_id=scene_id,
            max_time_step=max_time_step,
        )
        event_index = 0
        while backend.time_step < target_time:
            while (
                event_index < len(checked_events)
                and checked_events[event_index][0] == backend.time_step
            ):
                backend.apply_assignments(checked_events[event_index][1])
                event_index += 1
            backend.step_one_second()
        while (
            event_index < len(checked_events)
            and checked_events[event_index][0] == backend.time_step
        ):
            backend.apply_assignments(checked_events[event_index][1])
            event_index += 1
        if event_index != len(checked_events):
            raise ValueError('Basilisk action trace extends beyond target time')
        expected_progress = state_dict.get('max_progress')
        expected_completed = state_dict.get('completed')
        completion = backend.completion_snapshot()
        if not isinstance(expected_progress, torch.Tensor) or not torch.equal(
            completion.progress,
            expected_progress,
        ):
            raise ValueError('replayed Basilisk progress does not match checkpoint')
        if not isinstance(expected_completed, torch.Tensor) or not torch.equal(
            completion.completed,
            expected_completed,
        ):
            raise ValueError('replayed Basilisk completion does not match checkpoint')
        expected_completion_time = state_dict.get('completion_time')
        if (
            expected_completion_time is not None
            and (
                not isinstance(expected_completion_time, torch.Tensor)
                or not torch.equal(
                    backend._completion_time,
                    expected_completion_time,
                )
            )
        ):
            raise ValueError(
                'replayed Basilisk completion time does not match checkpoint',
            )
        expected_working_time = state_dict.get('working_time_steps')
        if (
            expected_working_time is not None
            and (
                not isinstance(expected_working_time, torch.Tensor)
                or not torch.equal(
                    backend._working_time_steps,
                    expected_working_time,
                )
            )
        ):
            raise ValueError(
                'replayed Basilisk power state does not match checkpoint',
            )
        return backend


def load_runtime_statistics(
    path: str | pathlib.Path = STATISTICS_PATH,
) -> Statistics:
    statistics = torch.load(pathlib.Path(path), weights_only=False)
    if not isinstance(statistics, Statistics):
        raise ValueError('statistics checkpoint has an unexpected type')
    return statistics
