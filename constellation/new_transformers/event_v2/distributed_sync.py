"""Event V2 大规模 PPO 的严格同步轮次协议。"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
import math
import queue
import random
from typing import Any

import numpy as np
import torch

from .model import EventJointActorCritic
from .rollout import (
    EventRuntime,
    StoredEventStep,
    SynchronousRuntimeSlot,
    collect_synchronous_rollout,
    replay_rollout_log_probs,
)


@dataclass(frozen=True)
class SyncRoundCommand:
    """命令 actor 使用一个精确策略版本采集一轮事件。"""

    round_id: int
    policy_version: int
    target_events: int | None = None
    stop: bool = False

    def __post_init__(self) -> None:
        if self.round_id < 0 or self.policy_version < 0:
            raise ValueError('sync round command identifiers are invalid')
        if self.target_events is not None and self.target_events <= 0:
            raise ValueError('sync round target events must be positive')


@dataclass(frozen=True)
class SyncActorChunk:
    """一个 actor 在严格 barrier 前提交的完整轮次。"""

    actor_id: int
    round_id: int
    policy_version: int
    steps: tuple[StoredEventStep, ...]
    completed_scene_ids: tuple[int, ...]
    replay_max_abs_error: float
    state: Mapping[str, Any]

    def __post_init__(self) -> None:
        if min(self.actor_id, self.round_id, self.policy_version) < 0:
            raise ValueError('sync actor chunk identifiers are invalid')
        if not self.steps:
            raise ValueError('sync actor chunk must contain rollout events')
        if (
            len(self.completed_scene_ids)
            != len(set(self.completed_scene_ids))
            or any(scene_id < 0 for scene_id in self.completed_scene_ids)
        ):
            raise ValueError('sync actor completed scene IDs are invalid')
        if (
            not math.isfinite(self.replay_max_abs_error)
            or self.replay_max_abs_error < 0
        ):
            raise ValueError('sync actor replay error is invalid')
        if not isinstance(self.state, Mapping):
            raise ValueError('sync actor state must be a mapping')


@dataclass(frozen=True)
class SyncActorDone:
    """actor 已完成其全部 scene 并停在可恢复边界。"""

    actor_id: int
    round_id: int
    policy_version: int
    completed_scene_ids: tuple[int, ...]
    reward_reconstruction_errors: tuple[tuple[int, float], ...]
    state: Mapping[str, Any]

    def __post_init__(self) -> None:
        if min(self.actor_id, self.round_id, self.policy_version) < 0:
            raise ValueError('sync done actor identifiers are invalid')
        if (
            not self.completed_scene_ids
            or len(self.completed_scene_ids)
            != len(set(self.completed_scene_ids))
            or any(scene_id < 0 for scene_id in self.completed_scene_ids)
        ):
            raise ValueError('sync done scene IDs are invalid')
        reward_scene_ids = tuple(
            scene_id for scene_id, _ in self.reward_reconstruction_errors
        )
        if (
            reward_scene_ids != self.completed_scene_ids
            or any(
                not math.isfinite(error) or error < 0
                for _, error in self.reward_reconstruction_errors
            )
        ):
            raise ValueError('sync done reward reconstruction audit is invalid')
        if not isinstance(self.state, Mapping):
            raise ValueError('sync done actor state must be a mapping')


@dataclass(frozen=True)
class SyncWorkerError:
    """子进程异常的完整上下文。"""

    actor_id: int
    error_type: str
    message: str
    traceback: str

    def __post_init__(self) -> None:
        if (
            self.actor_id < 0
            or not self.error_type
            or not self.message
            or not self.traceback
        ):
            raise ValueError('sync worker error context is incomplete')


@dataclass(frozen=True)
class SyncRoundBatch:
    """经过版本和 actor 集合校验、可直接交给 learner 的一轮。"""

    round_id: int
    policy_version: int
    actor_ids: tuple[int, ...]
    steps: tuple[StoredEventStep, ...]
    completed_scene_ids: tuple[int, ...]
    actor_states: tuple[tuple[int, Mapping[str, Any]], ...]
    event_count: int
    processed_physical_seconds: float
    replay_max_abs_error: float
    should_update: bool


@dataclass(frozen=True)
class QueuedRolloutPayload:
    """一个 actor pool 在单轮内采集并重放后的事件。"""

    steps: tuple[StoredEventStep, ...]
    completed_scene_ids: tuple[int, ...]
    replay_max_abs_error: float


def validate_and_merge_sync_round(
    chunks: Sequence[SyncActorChunk],
    *,
    expected_actor_ids: Collection[int],
    round_id: int,
    policy_version: int,
    min_batch_events: int,
) -> SyncRoundBatch:
    """校验一轮严格同步 chunk，并按事件标识确定性聚合。"""

    expected = tuple(sorted(int(actor_id) for actor_id in expected_actor_ids))
    if (
        not expected
        or len(expected) != len(set(expected))
        or any(actor_id < 0 for actor_id in expected)
    ):
        raise ValueError('expected sync actor set is invalid')
    if round_id < 0 or policy_version < 0 or min_batch_events <= 0:
        raise ValueError('sync round validation boundaries are invalid')
    submitted_ids = tuple(chunk.actor_id for chunk in chunks)
    if len(submitted_ids) != len(set(submitted_ids)):
        raise ValueError('sync round contains a duplicate actor submission')
    if set(submitted_ids) != set(expected):
        raise ValueError(
            'sync round actor set does not match the expected actor set',
        )

    transition_ids: set[tuple[int, int, int]] = set()
    all_steps: list[StoredEventStep] = []
    completed_scene_ids: list[int] = []
    replay_errors: list[float] = []
    actor_states: list[tuple[int, Mapping[str, Any]]] = []
    for chunk in sorted(chunks, key=lambda value: value.actor_id):
        if chunk.round_id != round_id:
            raise ValueError('sync actor chunk has the wrong round')
        if chunk.policy_version != policy_version:
            raise ValueError('sync actor chunk has the wrong policy version')
        for step in chunk.steps:
            step.validate()
            if step.policy_version != policy_version:
                raise ValueError(
                    'sync rollout step policy version does not match its round',
                )
            transition_id = (
                int(step.environment_index),
                int(step.episode_id),
                int(step.event_index),
            )
            if transition_id in transition_ids:
                raise ValueError(
                    'sync round contains a duplicate rollout transition',
                )
            transition_ids.add(transition_id)
            all_steps.append(step)
        completed_scene_ids.extend(chunk.completed_scene_ids)
        replay_errors.append(chunk.replay_max_abs_error)
        actor_states.append((chunk.actor_id, chunk.state))

    if (
        len(completed_scene_ids) != len(set(completed_scene_ids))
        or any(scene_id < 0 for scene_id in completed_scene_ids)
    ):
        raise ValueError('sync round contains duplicate completed scenes')
    all_steps.sort(
        key=lambda step: (
            int(step.environment_index),
            int(step.episode_id),
            int(step.event_index),
        ),
    )
    processed_physical_seconds = sum(
        float(step.delta_t.item()) for step in all_steps
    )
    if not math.isfinite(processed_physical_seconds):
        raise ValueError('sync round physical seconds are invalid')
    return SyncRoundBatch(
        round_id=round_id,
        policy_version=policy_version,
        actor_ids=expected,
        steps=tuple(all_steps),
        completed_scene_ids=tuple(sorted(completed_scene_ids)),
        actor_states=tuple(actor_states),
        event_count=len(all_steps),
        processed_physical_seconds=processed_physical_seconds,
        replay_max_abs_error=max(replay_errors),
        should_update=len(all_steps) >= min_batch_events,
    )


class StrictSyncRoundCoordinator:
    """在父进程中阻止 actor 越过尚未完成的同步 barrier。"""

    def __init__(
        self,
        *,
        actor_ids: Collection[int],
        initial_round_id: int = 0,
        initial_policy_version: int = 0,
    ) -> None:
        actor_ids = tuple(sorted(int(actor_id) for actor_id in actor_ids))
        if (
            not actor_ids
            or len(actor_ids) != len(set(actor_ids))
            or any(actor_id < 0 for actor_id in actor_ids)
        ):
            raise ValueError('strict sync actor IDs are invalid')
        if initial_round_id < 0 or initial_policy_version < 0:
            raise ValueError('strict sync initial identifiers are invalid')
        self._actor_ids = actor_ids
        self._round_id = initial_round_id
        self._policy_version = initial_policy_version
        self._chunks: dict[int, SyncActorChunk] = {}
        self._finalized: SyncRoundBatch | None = None

    @property
    def actor_ids(self) -> tuple[int, ...]:
        return self._actor_ids

    @property
    def round_id(self) -> int:
        return self._round_id

    @property
    def policy_version(self) -> int:
        return self._policy_version

    @property
    def submitted_actor_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._chunks))

    def command_for(self, actor_id: int) -> SyncRoundCommand:
        if actor_id not in self._actor_ids:
            raise ValueError('unknown strict sync actor')
        if self._finalized is not None:
            raise RuntimeError(
                'strict sync round is finalized but has not advanced',
            )
        return SyncRoundCommand(
            round_id=self._round_id,
            policy_version=self._policy_version,
        )

    def submit(self, chunk: SyncActorChunk) -> None:
        if self._finalized is not None:
            raise RuntimeError('strict sync round is already finalized')
        if chunk.actor_id not in self._actor_ids:
            raise ValueError('unknown strict sync actor')
        if chunk.actor_id in self._chunks:
            raise ValueError('duplicate actor submission in strict sync round')
        if chunk.round_id != self._round_id:
            raise ValueError('sync actor chunk has the wrong round')
        if chunk.policy_version != self._policy_version:
            raise ValueError('sync actor chunk has the wrong policy version')
        self._chunks[chunk.actor_id] = chunk

    def finalize(self, *, min_batch_events: int) -> SyncRoundBatch:
        if self._finalized is not None:
            return self._finalized
        if set(self._chunks) != set(self._actor_ids):
            missing = sorted(set(self._actor_ids) - set(self._chunks))
            raise RuntimeError(
                f'strict sync round is incomplete; missing actors: {missing}',
            )
        self._finalized = validate_and_merge_sync_round(
            tuple(self._chunks.values()),
            expected_actor_ids=self._actor_ids,
            round_id=self._round_id,
            policy_version=self._policy_version,
            min_batch_events=min_batch_events,
        )
        return self._finalized

    def advance(self, *, next_policy_version: int) -> None:
        if self._finalized is None:
            raise RuntimeError(
                'strict sync round must be finalized before advancing',
            )
        if next_policy_version != self._policy_version + 1:
            raise ValueError(
                'strict sync policy version must increase by exactly one',
            )
        self._round_id += 1
        self._policy_version = next_policy_version
        self._chunks.clear()
        self._finalized = None


def capture_rng_state() -> dict[str, Any]:
    """捕获 actor 断点续训需要的所有随机数状态。"""

    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state().clone(),
        'cuda': (
            tuple(state.clone() for state in torch.cuda.get_rng_state_all())
            if torch.cuda.is_available()
            else ()
        ),
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """恢复由 :func:`capture_rng_state` 保存的随机数状态。"""

    required = {'python', 'numpy', 'torch', 'cuda'}
    if set(state) != required:
        raise ValueError('sync RNG state schema does not match')
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch_state = state['torch']
    if not isinstance(torch_state, torch.Tensor):
        raise ValueError('sync Torch RNG state is invalid')
    torch.set_rng_state(torch_state.cpu())
    cuda_states = state['cuda']
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError(
                'CUDA RNG state cannot be restored without CUDA',
            )
        visible_cuda_states = cuda_states[:torch.cuda.device_count()]
        if visible_cuda_states:
            torch.cuda.set_rng_state_all(
                [value.cpu() for value in visible_cuda_states],
            )


class QueuedEventRuntimePool:
    """为一个 actor 维护有上限的活动环境和确定性 scene 队列。"""

    STATE_VERSION = 1

    def __init__(
        self,
        *,
        assigned_scene_ids: Sequence[int],
        max_active_environments: int,
        runtime_factory: Callable[[int], EventRuntime],
        runtime_state_loader: Callable[
            [int, Mapping[str, Any]],
            EventRuntime,
        ],
        initialize: bool = True,
    ) -> None:
        assigned = tuple(int(scene_id) for scene_id in assigned_scene_ids)
        if (
            not assigned
            or len(assigned) != len(set(assigned))
            or any(scene_id < 0 for scene_id in assigned)
        ):
            raise ValueError('queued runtime assigned scene IDs are invalid')
        if not 1 <= max_active_environments <= len(assigned):
            raise ValueError('queued runtime active environment cap is invalid')
        self._assigned_scene_ids = assigned
        self._max_active_environments = int(max_active_environments)
        self._runtime_factory = runtime_factory
        self._runtime_state_loader = runtime_state_loader
        self._pending_scene_ids = deque(assigned)
        self._active: dict[int, SynchronousRuntimeSlot] = {}
        self._completed_scene_ids: list[int] = []
        self._reward_reconstruction_errors: dict[int, float] = {}
        if initialize:
            self._fill_active_environments()

    @property
    def assigned_scene_ids(self) -> tuple[int, ...]:
        return self._assigned_scene_ids

    @property
    def active_environment_count(self) -> int:
        return len(self._active)

    @property
    def active_scene_ids(self) -> tuple[int, ...]:
        return tuple(self._active)

    @property
    def pending_scene_ids(self) -> tuple[int, ...]:
        return tuple(self._pending_scene_ids)

    @property
    def completed_scene_ids(self) -> tuple[int, ...]:
        return tuple(self._completed_scene_ids)

    @property
    def reward_reconstruction_errors(
        self,
    ) -> tuple[tuple[int, float], ...]:
        return tuple(
            (scene_id, self._reward_reconstruction_errors[scene_id])
            for scene_id in self._completed_scene_ids
        )

    @property
    def is_complete(self) -> bool:
        return (
            not self._pending_scene_ids
            and not self._active
            and tuple(sorted(self._completed_scene_ids))
            == tuple(sorted(self._assigned_scene_ids))
        )

    def _fill_active_environments(self) -> None:
        while (
            self._pending_scene_ids
            and len(self._active) < self._max_active_environments
        ):
            scene_id = self._pending_scene_ids.popleft()
            runtime = self._runtime_factory(scene_id)
            observation = runtime.reset()
            slot = SynchronousRuntimeSlot(
                environment_index=scene_id,
                episode_id=0,
                observation=observation,
                runtime=runtime,
            )
            self._active[scene_id] = slot

    @staticmethod
    def _runtime_state(runtime: EventRuntime) -> Mapping[str, Any]:
        state_dict = getattr(runtime, 'state_dict', None)
        if not callable(state_dict):
            raise TypeError(
                'queued event runtime must support state_dict()',
            )
        state = state_dict()
        if not isinstance(state, Mapping):
            raise TypeError('queued event runtime state must be a mapping')
        return state

    @staticmethod
    def _reward_reconstruction_error(runtime: EventRuntime) -> float:
        try:
            error = float(
                getattr(runtime, 'reward_reconstruction_error'),
            )
        except (AttributeError, RuntimeError):
            total_reward = getattr(runtime, 'total_reward', None)
            final_quality = getattr(runtime, 'final_quality', None)
            if total_reward is None or final_quality is None:
                raise RuntimeError(
                    'finished runtime lacks reward reconstruction audit',
                )
            error = abs(float(total_reward) - float(final_quality))
        if not math.isfinite(error) or error < 0:
            raise RuntimeError(
                'runtime reward reconstruction error is invalid',
            )
        return error

    def _retire_finished_environments(self) -> tuple[int, ...]:
        completed = []
        for scene_id, slot in tuple(self._active.items()):
            if not slot.finished:
                continue
            self._reward_reconstruction_errors[scene_id] = (
                self._reward_reconstruction_error(slot.runtime)
            )
            self._completed_scene_ids.append(scene_id)
            completed.append(scene_id)
            del self._active[scene_id]
        self._fill_active_environments()
        return tuple(completed)

    def collect(
        self,
        *,
        model: EventJointActorCritic,
        policy_version: int,
        max_events: int,
        device: torch.device,
        replay_atol: float,
        amp_enabled: bool = False,
        amp_dtype: torch.dtype = torch.bfloat16,
    ) -> QueuedRolloutPayload:
        """以一个固定 policy version 收集并重放本 actor 的一轮。"""

        if self.is_complete:
            raise ValueError('queued runtime pool has already completed')
        if max_events <= 0 or replay_atol < 0:
            raise ValueError('queued rollout boundaries are invalid')
        steps: list[StoredEventStep] = []
        completed_scene_ids: list[int] = []
        while len(steps) < max_events and self._active:
            collected = collect_synchronous_rollout(
                model,
                tuple(self._active.values()),
                target_events=max_events - len(steps),
                policy_version=policy_version,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            if not collected:
                raise RuntimeError(
                    'queued event collector made no progress',
                )
            steps.extend(collected)
            completed_scene_ids.extend(
                self._retire_finished_environments(),
            )
        replay = replay_rollout_log_probs(
            model,
            steps,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        behavior = torch.stack([
            step.behavior_log_prob for step in steps
        ])
        replay_max_abs_error = float((replay - behavior).abs().max())
        if replay_max_abs_error > replay_atol:
            raise RuntimeError(
                'strict sync actor behavior log-prob replay mismatch: '
                f'{replay_max_abs_error:.8g}',
            )
        return QueuedRolloutPayload(
            steps=tuple(steps),
            completed_scene_ids=tuple(completed_scene_ids),
            replay_max_abs_error=replay_max_abs_error,
        )

    def state_dict(self) -> dict[str, Any]:
        """仅在 actor 到达 barrier 后调用。"""

        active = []
        for scene_id, slot in self._active.items():
            active.append({
                'scene_id': scene_id,
                'environment_index': slot.environment_index,
                'episode_id': slot.episode_id,
                'event_index': slot.event_index,
                'finished': slot.finished,
                'observation': slot.observation,
                'runtime': dict(self._runtime_state(slot.runtime)),
            })
        return {
            'version': self.STATE_VERSION,
            'assigned_scene_ids': self._assigned_scene_ids,
            'max_active_environments': self._max_active_environments,
            'pending_scene_ids': tuple(self._pending_scene_ids),
            'active': tuple(active),
            'completed_scene_ids': tuple(self._completed_scene_ids),
            'reward_reconstruction_errors': (
                self.reward_reconstruction_errors
            ),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """从严格 barrier 保存的状态恢复所有活动环境。"""

        if state.get('version') != self.STATE_VERSION:
            raise ValueError('queued runtime pool state version does not match')
        if tuple(state.get('assigned_scene_ids', ())) != (
            self._assigned_scene_ids
        ):
            raise ValueError('queued runtime scene assignment does not match')
        if int(state.get('max_active_environments', -1)) != (
            self._max_active_environments
        ):
            raise ValueError('queued runtime active environment cap changed')
        pending = tuple(int(value) for value in state['pending_scene_ids'])
        completed = tuple(
            int(value) for value in state['completed_scene_ids']
        )
        active_states = tuple(state['active'])
        if len(active_states) > self._max_active_environments:
            raise ValueError('queued runtime state exceeds its active cap')
        active_scene_ids = tuple(
            int(value['scene_id']) for value in active_states
        )
        combined = pending + active_scene_ids + completed
        if (
            len(combined) != len(set(combined))
            or set(combined) != set(self._assigned_scene_ids)
        ):
            raise ValueError('queued runtime scene partition is invalid')
        reward_errors = {
            int(scene_id): float(error)
            for scene_id, error in state['reward_reconstruction_errors']
        }
        if set(reward_errors) != set(completed):
            raise ValueError(
                'queued runtime reward audit does not match completed scenes',
            )
        self._pending_scene_ids = deque(pending)
        self._completed_scene_ids = list(completed)
        self._reward_reconstruction_errors = reward_errors
        self._active.clear()
        for slot_state in active_states:
            scene_id = int(slot_state['scene_id'])
            runtime_state = slot_state['runtime']
            if not isinstance(runtime_state, Mapping):
                raise ValueError('queued runtime checkpoint is invalid')
            runtime = self._runtime_state_loader(
                scene_id,
                runtime_state,
            )
            slot = SynchronousRuntimeSlot(
                environment_index=int(slot_state['environment_index']),
                episode_id=int(slot_state['episode_id']),
                event_index=int(slot_state['event_index']),
                observation=slot_state['observation'],
                runtime=runtime,
                finished=bool(slot_state['finished']),
            )
            self._active[scene_id] = slot


def run_strict_sync_actor_loop(
    *,
    model: EventJointActorCritic,
    pool: QueuedEventRuntimePool,
    actor_id: int,
    policy_store: Any,
    command_queue: Any,
    result_queue: Any,
    stop_event: Any,
    target_events: int,
    device: torch.device,
    replay_atol: float,
    initial_round_id: int,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """只按 learner 命令运行，并在每个 chunk 后停在 barrier。"""

    if actor_id < 0 or initial_round_id < 0:
        raise ValueError('strict sync actor loop identifiers are invalid')
    if target_events <= 0 or replay_atol < 0:
        raise ValueError('strict sync actor loop boundaries are invalid')
    model.to(device)
    model.eval()
    expected_round_id = initial_round_id
    loaded_policy_version = -1
    while not stop_event.is_set():
        try:
            command = command_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if not isinstance(command, SyncRoundCommand):
            raise TypeError('strict sync actor received an unknown command')
        if command.stop:
            return
        if command.round_id != expected_round_id:
            raise RuntimeError(
                'strict sync actor received an out-of-order round: '
                f'expected {expected_round_id}, got {command.round_id}',
            )
        refresh = policy_store.refresh(
            model,
            last_version=loaded_policy_version,
        )
        loaded_policy_version = refresh.version
        if loaded_policy_version != command.policy_version:
            raise RuntimeError(
                'strict sync actor policy store version does not match '
                f'round command: store={loaded_policy_version}, '
                f'command={command.policy_version}',
            )
        payload = pool.collect(
            model=model,
            policy_version=loaded_policy_version,
            max_events=(
                target_events
                if command.target_events is None
                else command.target_events
            ),
            device=device,
            replay_atol=replay_atol,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        actor_state = {
            'pool': pool.state_dict(),
            'rng': capture_rng_state(),
        }
        result_queue.put(SyncActorChunk(
            actor_id=actor_id,
            round_id=command.round_id,
            policy_version=command.policy_version,
            steps=payload.steps,
            completed_scene_ids=payload.completed_scene_ids,
            replay_max_abs_error=payload.replay_max_abs_error,
            state=actor_state,
        ))
        expected_round_id += 1
        if not pool.is_complete:
            continue
        result_queue.put(SyncActorDone(
            actor_id=actor_id,
            round_id=command.round_id,
            policy_version=command.policy_version,
            completed_scene_ids=pool.completed_scene_ids,
            reward_reconstruction_errors=(
                pool.reward_reconstruction_errors
            ),
            state=actor_state,
        ))
        while not stop_event.is_set():
            stop_event.wait(0.1)
        return
