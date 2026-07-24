"""V2-3 异步 PPO 的版本边界与训练组件。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch

from .model import EventJointActorCritic
from .ppo import (
    PPOConfig,
    PPOUpdateMetrics,
    SynchronousPPOTrainer,
)
from .rollout import StoredEventStep


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
