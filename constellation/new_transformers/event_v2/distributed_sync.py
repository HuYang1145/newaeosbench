"""Event V2 大规模 PPO 的严格同步轮次协议。"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

from .rollout import StoredEventStep


@dataclass(frozen=True)
class SyncRoundCommand:
    """命令 actor 使用一个精确策略版本采集一轮事件。"""

    round_id: int
    policy_version: int
    stop: bool = False

    def __post_init__(self) -> None:
        if self.round_id < 0 or self.policy_version < 0:
            raise ValueError('sync round command identifiers are invalid')


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
    completed_scene_ids: tuple[int, ...]
    state: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.actor_id < 0:
            raise ValueError('sync done actor ID is invalid')
        if (
            not self.completed_scene_ids
            or len(self.completed_scene_ids)
            != len(set(self.completed_scene_ids))
            or any(scene_id < 0 for scene_id in self.completed_scene_ids)
        ):
            raise ValueError('sync done scene IDs are invalid')
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
