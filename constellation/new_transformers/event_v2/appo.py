"""V2-3 异步 PPO 的版本边界与训练组件。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import time
from typing import Any, NamedTuple

import torch

from .model import EventJointActorCritic
from .ppo import (
    PPOConfig,
    PPOUpdateMetrics,
    SynchronousPPOTrainer,
)
from .rollout import (
    StoredEventStep,
    SynchronousRuntimeSlot,
    collect_synchronous_rollout,
    replay_rollout_log_probs,
)


@dataclass(frozen=True)
class APPOConfig:
    """异步采样相对当前 learner 策略允许落后的最大版本数。"""

    max_policy_lag: int = 2

    def __post_init__(self) -> None:
        if self.max_policy_lag < 0:
            raise ValueError('APPO policy lag must be non-negative')


class PolicyLagFilterResult(NamedTuple):
    accepted: tuple[StoredEventStep, ...]
    stale_dropped: int
    minimum_version: int
    maximum_version: int


@dataclass(frozen=True)
class APPOUpdateMetrics:
    """一次异步 learner 更新及其样本版本审计。"""

    ppo: PPOUpdateMetrics
    input_events: int
    accepted_events: int
    stale_dropped_events: int
    minimum_behavior_version: int
    maximum_behavior_version: int


@dataclass(frozen=True)
class APPORolloutChunk:
    actor_id: int
    policy_version: int
    scene_ids: tuple[int, ...]
    steps: tuple[StoredEventStep, ...]
    replay_max_error: float
    physical_seconds: int
    completed_episodes: int

    def __post_init__(self) -> None:
        if self.actor_id < 0 or self.policy_version < 0:
            raise ValueError('APPO actor identifiers must be non-negative')
        if (
            not self.scene_ids
            or len(self.scene_ids) != len(set(self.scene_ids))
            or any(scene_id < 0 for scene_id in self.scene_ids)
        ):
            raise ValueError('APPO actor scene IDs are invalid')
        if (
            not self.steps
            or any(
                step.policy_version != self.policy_version
                for step in self.steps
            )
        ):
            raise ValueError(
                'APPO chunk needs one non-empty behavior policy version',
            )
        if (
            self.replay_max_error < 0
            or self.physical_seconds <= 0
            or not 0 <= self.completed_episodes <= len(self.scene_ids)
        ):
            raise ValueError('APPO actor chunk metrics are invalid')


@dataclass(frozen=True)
class APPOSnapshot:
    actor_id: int
    generation: int
    scene_ids: tuple[int, ...]
    runtime_states: tuple[Mapping[str, Any], ...]
    completed_episodes: int

    def __post_init__(self) -> None:
        if self.actor_id < 0 or self.generation <= 0:
            raise ValueError('APPO snapshot identifiers are invalid')
        if (
            not self.scene_ids
            or len(self.scene_ids) != len(self.runtime_states)
            or not all(
                isinstance(state, Mapping)
                for state in self.runtime_states
            )
            or not 0 <= self.completed_episodes <= len(self.scene_ids)
        ):
            raise ValueError('APPO snapshot runtime states are invalid')


@dataclass(frozen=True)
class APPODone:
    actor_id: int
    scene_ids: tuple[int, ...]
    runtime_states: tuple[Mapping[str, Any], ...]
    completed_episodes: int
    reward_reconstruction_errors: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.actor_id < 0:
            raise ValueError('APPO done actor ID is invalid')
        if (
            not self.scene_ids
            or len(self.scene_ids) != len(self.runtime_states)
            or self.completed_episodes != len(self.scene_ids)
            or len(self.reward_reconstruction_errors) != len(self.scene_ids)
            or any(
                not torch.isfinite(torch.tensor(error)) or error < 0
                for error in self.reward_reconstruction_errors
            )
        ):
            raise ValueError('APPO done message is incomplete')


@dataclass(frozen=True)
class APPOWorkerError:
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
            raise ValueError('APPO worker error context is incomplete')


class SharedPolicyRefresh(NamedTuple):
    version: int
    refreshed: bool


class SharedPolicyStore:
    """通过共享 CPU tensors 原子发布完整策略版本。"""

    def __init__(
        self,
        model: EventJointActorCritic,
        *,
        context: Any,
        initial_version: int,
    ) -> None:
        if initial_version < 0:
            raise ValueError('initial policy version must be non-negative')
        self.model = model.cpu()
        self.model.share_memory()
        self._version = context.Value('q', initial_version)
        self._lock = context.RLock()

    @property
    def version(self) -> int:
        with self._lock:
            return int(self._version.value)

    def publish(
        self,
        source_model: EventJointActorCritic,
        *,
        version: int,
    ) -> None:
        with self._lock:
            if version <= self._version.value:
                raise ValueError(
                    'shared policy versions must increase monotonically',
                )
            source_state = source_model.state_dict()
            shared_state = self.model.state_dict()
            if set(source_state) != set(shared_state):
                raise ValueError('shared policy state keys do not match')
            with torch.no_grad():
                for name, target in shared_state.items():
                    target.copy_(source_state[name].detach().to('cpu'))
            self._version.value = version

    def refresh(
        self,
        target_model: EventJointActorCritic,
        *,
        last_version: int,
    ) -> SharedPolicyRefresh:
        if last_version < -1:
            raise ValueError('last policy version is invalid')
        with self._lock:
            current = int(self._version.value)
            if current == last_version:
                return SharedPolicyRefresh(
                    version=current,
                    refreshed=False,
                )
            if current < last_version:
                raise ValueError('target policy version is ahead of store')
            target_model.load_state_dict(self.model.state_dict())
            return SharedPolicyRefresh(
                version=current,
                refreshed=True,
            )


def filter_policy_lag(
    steps: Sequence[StoredEventStep],
    *,
    current_policy_version: int,
    max_policy_lag: int,
) -> PolicyLagFilterResult:
    """保留版本差在上限内的事件，并严格拒绝来自未来的事件。"""

    if not steps:
        raise ValueError('policy-lag filtering requires at least one event')
    if current_policy_version < 0:
        raise ValueError('current policy version must be non-negative')
    if max_policy_lag < 0:
        raise ValueError('maximum policy lag must be non-negative')
    versions = tuple(int(step.policy_version) for step in steps)
    if any(version > current_policy_version for version in versions):
        raise ValueError('rollout contains a future policy version')
    minimum_allowed = current_policy_version - max_policy_lag
    accepted = tuple(
        step
        for step in steps
        if step.policy_version >= minimum_allowed
    )
    return PolicyLagFilterResult(
        accepted=accepted,
        stale_dropped=len(steps) - len(accepted),
        minimum_version=min(versions),
        maximum_version=max(versions),
    )


def collect_appo_actor_chunk(
    model: EventJointActorCritic,
    slots: Sequence[SynchronousRuntimeSlot],
    *,
    actor_id: int,
    scene_ids: Sequence[int],
    target_events: int,
    policy_version: int,
    device: torch.device,
    replay_atol: float,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> APPORolloutChunk:
    """采集一个版本固定的 actor chunk，并在发送前重放行为概率。"""

    scene_ids = tuple(int(scene_id) for scene_id in scene_ids)
    if len(slots) != len(scene_ids):
        raise ValueError('APPO actor slots must match scene IDs')
    if replay_atol < 0:
        raise ValueError('APPO actor replay tolerance must be non-negative')
    active_slots = [slot for slot in slots if not slot.finished]
    steps = collect_synchronous_rollout(
        model,
        active_slots,
        target_events=target_events,
        policy_version=policy_version,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
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
    replay_max_error = float((replay - behavior).abs().max())
    if replay_max_error > replay_atol:
        raise RuntimeError(
            'APPO actor behavior log-prob replay mismatch: '
            f'{replay_max_error:.8g}',
        )
    return APPORolloutChunk(
        actor_id=actor_id,
        policy_version=policy_version,
        scene_ids=scene_ids,
        steps=tuple(steps),
        replay_max_error=replay_max_error,
        physical_seconds=int(
            sum(step.delta_t.item() for step in steps)
        ),
        completed_episodes=sum(int(slot.finished) for slot in slots),
    )


def _serialize_actor_slots(
    slots: Sequence[SynchronousRuntimeSlot],
) -> tuple[Mapping[str, Any], ...]:
    states = []
    for slot in slots:
        state_dict = getattr(slot.runtime, 'state_dict', None)
        if not callable(state_dict):
            raise TypeError('APPO actor runtime must support state_dict()')
        states.append({
            'environment_index': slot.environment_index,
            'episode_id': slot.episode_id,
            'event_index': slot.event_index,
            'finished': slot.finished,
            'runtime': state_dict(),
        })
    return tuple(states)


def run_appo_actor_loop(
    *,
    model: EventJointActorCritic,
    slots: Sequence[SynchronousRuntimeSlot],
    actor_id: int,
    scene_ids: Sequence[int],
    policy_store: SharedPolicyStore,
    result_queue: Any,
    stop_event: Any,
    target_events: int,
    device: torch.device,
    replay_atol: float,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    checkpoint_request: Any | None = None,
    checkpoint_release: Any | None = None,
) -> None:
    """在 chunk 边界刷新策略，并把完整 actor 结果写入队列。"""

    scene_ids = tuple(int(scene_id) for scene_id in scene_ids)
    if len(slots) != len(scene_ids):
        raise ValueError('APPO actor slots must match scene IDs')
    model.to(device)
    model.eval()
    policy_version = policy_store.refresh(
        model,
        last_version=-1,
    ).version
    last_snapshot_generation = 0
    while not stop_event.is_set() and any(
        not slot.finished for slot in slots
    ):
        refresh = policy_store.refresh(
            model,
            last_version=policy_version,
        )
        policy_version = refresh.version
        chunk = collect_appo_actor_chunk(
            model,
            slots,
            actor_id=actor_id,
            scene_ids=scene_ids,
            target_events=target_events,
            policy_version=policy_version,
            device=device,
            replay_atol=replay_atol,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        result_queue.put(chunk)
        if checkpoint_request is not None:
            generation = int(checkpoint_request.value)
            if generation > last_snapshot_generation:
                result_queue.put(APPOSnapshot(
                    actor_id=actor_id,
                    generation=generation,
                    scene_ids=scene_ids,
                    runtime_states=_serialize_actor_slots(slots),
                    completed_episodes=sum(
                        int(slot.finished) for slot in slots
                    ),
                ))
                last_snapshot_generation = generation
                if checkpoint_release is None:
                    raise ValueError(
                        'APPO checkpoint release signal is missing',
                    )
                while (
                    int(checkpoint_release.value) < generation
                    and not stop_event.is_set()
                ):
                    time.sleep(0.05)

    if stop_event.is_set():
        return
    errors = tuple(
        float(getattr(slot.runtime, 'reward_reconstruction_error'))
        for slot in slots
    )
    result_queue.put(APPODone(
        actor_id=actor_id,
        scene_ids=scene_ids,
        runtime_states=_serialize_actor_slots(slots),
        completed_episodes=len(slots),
        reward_reconstruction_errors=errors,
    ))


class AsynchronousPPOLearner:
    """在严格 policy-lag 边界内复用 event PPO 目标。"""

    def __init__(
        self,
        *,
        model: EventJointActorCritic,
        optimizer: torch.optim.Optimizer,
        ppo_config: PPOConfig,
        appo_config: APPOConfig,
        device: torch.device,
        amp_enabled: bool = False,
        amp_dtype: torch.dtype = torch.bfloat16,
        scaler: torch.amp.GradScaler | None = None,
    ) -> None:
        if model.backbone_is_frozen:
            raise ValueError('V2-3 requires an explicitly unfrozen tail')
        self.appo_config = appo_config
        self._trainer = SynchronousPPOTrainer(
            model=model,
            optimizer=optimizer,
            config=ppo_config,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
            require_fully_frozen_backbone=False,
            verify_behavior_replay=False,
        )

    @property
    def policy_version(self) -> int:
        return self._trainer.policy_version

    @policy_version.setter
    def policy_version(self, value: int) -> None:
        if value < 0:
            raise ValueError('policy version must be non-negative')
        self._trainer.policy_version = value

    def update(
        self,
        steps: Sequence[StoredEventStep],
    ) -> APPOUpdateMetrics:
        filtered = filter_policy_lag(
            steps,
            current_policy_version=self.policy_version,
            max_policy_lag=self.appo_config.max_policy_lag,
        )
        if not filtered.accepted:
            raise ValueError('APPO batch contains only stale events')
        ppo_metrics = self._trainer.update(list(filtered.accepted))
        return APPOUpdateMetrics(
            ppo=ppo_metrics,
            input_events=len(steps),
            accepted_events=len(filtered.accepted),
            stale_dropped_events=filtered.stale_dropped,
            minimum_behavior_version=filtered.minimum_version,
            maximum_behavior_version=filtered.maximum_version,
        )
