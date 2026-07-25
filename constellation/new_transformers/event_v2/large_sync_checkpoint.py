"""大规模严格同步 PPO 的 barrier checkpoint。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
import pathlib
from typing import Any

import torch

from .checkpoint import capture_rng_state, restore_rng_state
from .model import EventJointActorCritic
from .transition import transition_schema_fingerprint


LARGE_SYNC_CHECKPOINT_VERSION = 1
LARGE_SYNC_STAGE = 'V2-2-Large'


@dataclass(frozen=True)
class LargeSyncCounters:
    """checkpoint 所在 barrier 的累计训练进度。"""

    next_round_id: int = 0
    updates: int = 0
    policy_version: int = 0
    processed_physical_seconds: int = 0
    episodes: int = 0
    events: int = 0

    def __post_init__(self) -> None:
        if any(value < 0 for value in (
            self.next_round_id,
            self.updates,
            self.policy_version,
            self.processed_physical_seconds,
            self.episodes,
            self.events,
        )):
            raise ValueError('large sync counters must be non-negative')
        if self.updates != self.policy_version:
            raise ValueError(
                'large sync updates and policy version must match',
            )
        if self.next_round_id < self.updates:
            raise ValueError(
                'large sync next round cannot precede learner updates',
            )


@dataclass(frozen=True)
class LargeSyncRestore:
    counters: LargeSyncCounters
    actor_scene_assignments: dict[int, tuple[int, ...]]
    actor_states: dict[int, Mapping[str, Any]]
    normalizer: Mapping[str, torch.Tensor]
    trainable_parameter_names: tuple[str, ...]
    frozen_parameter_names: tuple[str, ...]
    bootstrap: Mapping[str, Any]


@dataclass(frozen=True)
class LargeSyncPolicyMetadata:
    stage: str
    updates: int
    policy_version: int
    scene_ids: tuple[int, ...]
    config_fingerprint: str
    bootstrap: Mapping[str, Any]


def _normalize_scene_assignments(
    assignments: Mapping[int, Sequence[int]],
) -> dict[int, tuple[int, ...]]:
    normalized = {
        int(actor_id): tuple(int(scene_id) for scene_id in scene_ids)
        for actor_id, scene_ids in assignments.items()
    }
    flattened = tuple(
        scene_id
        for actor_id in sorted(normalized)
        for scene_id in normalized[actor_id]
    )
    if (
        not normalized
        or any(actor_id < 0 for actor_id in normalized)
        or any(not scene_ids for scene_ids in normalized.values())
        or any(scene_id < 0 for scene_id in flattened)
        or len(flattened) != len(set(flattened))
    ):
        raise ValueError(
            'large sync actor scene assignments are invalid',
        )
    return {
        actor_id: normalized[actor_id]
        for actor_id in sorted(normalized)
    }


def _normalize_parameter_names(
    values: Sequence[str],
    *,
    label: str,
) -> tuple[str, ...]:
    names = tuple(str(value) for value in values)
    if (
        not names
        or any(not name for name in names)
        or len(names) != len(set(names))
    ):
        raise ValueError(
            f'large sync {label} parameter names are invalid',
        )
    return names


def _validate_actor_states(
    actor_states: Mapping[int, Mapping[str, Any]],
    *,
    actor_ids: Sequence[int],
) -> dict[int, Mapping[str, Any]]:
    normalized = {
        int(actor_id): state
        for actor_id, state in actor_states.items()
    }
    if set(normalized) != set(actor_ids):
        raise ValueError(
            'large sync actor states do not match scene assignments',
        )
    for state in normalized.values():
        if not isinstance(state, Mapping):
            raise ValueError('large sync actor state must be a mapping')
        if set(state) != {'pool', 'rng'}:
            raise ValueError('large sync actor state schema does not match')
        if not isinstance(state['pool'], Mapping):
            raise ValueError('large sync actor pool state is invalid')
        rng = state['rng']
        if (
            not isinstance(rng, Mapping)
            or set(rng) != {'python', 'numpy', 'torch', 'cuda'}
            or not isinstance(rng['torch'], torch.Tensor)
        ):
            raise ValueError('large sync actor RNG state is invalid')
    return {
        actor_id: normalized[actor_id]
        for actor_id in sorted(normalized)
    }


def _validate_model_parameter_boundary(
    model: EventJointActorCritic,
    *,
    trainable_parameter_names: Sequence[str],
    frozen_parameter_names: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    trainable = _normalize_parameter_names(
        trainable_parameter_names,
        label='trainable',
    )
    frozen = _normalize_parameter_names(
        frozen_parameter_names,
        label='frozen',
    )
    actual_trainable = tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    actual_frozen = tuple(
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    )
    if trainable != actual_trainable:
        raise ValueError(
            'large sync trainable parameter boundary does not match model',
        )
    if frozen != actual_frozen:
        raise ValueError(
            'large sync frozen parameter boundary does not match model',
        )
    if set(trainable) & set(frozen):
        raise ValueError('large sync parameter boundaries overlap')
    return trainable, frozen


def _validate_freeze_state(model: EventJointActorCritic) -> dict[str, Any]:
    transformer_trainable = sum(
        parameter.numel()
        for parameter in model.backbone.transformer.parameters()
        if parameter.requires_grad
    )
    freeze_state = {
        'backbone_is_frozen': model.backbone_is_frozen,
        'transformer_trainable_parameter_count': transformer_trainable,
    }
    if freeze_state != {
        'backbone_is_frozen': True,
        'transformer_trainable_parameter_count': 0,
    }:
        raise ValueError(
            'large sync checkpoint requires a frozen Stage3 transformer',
        )
    return freeze_state


def build_large_sync_checkpoint_payload(
    *,
    model: EventJointActorCritic,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    config_fingerprint_value: str,
    normalizer: Mapping[str, torch.Tensor],
    counters: LargeSyncCounters,
    actor_scene_assignments: Mapping[int, Sequence[int]],
    actor_states: Mapping[int, Mapping[str, Any]],
    trainable_parameter_names: Sequence[str],
    frozen_parameter_names: Sequence[str],
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    """构建只允许在完整 barrier 上保存的 checkpoint payload。"""

    if (
        not isinstance(config_fingerprint_value, str)
        or len(config_fingerprint_value) != 64
    ):
        raise ValueError('large sync config fingerprint is invalid')
    assignments = _normalize_scene_assignments(actor_scene_assignments)
    states = _validate_actor_states(
        actor_states,
        actor_ids=tuple(assignments),
    )
    trainable, frozen = _validate_model_parameter_boundary(
        model,
        trainable_parameter_names=trainable_parameter_names,
        frozen_parameter_names=frozen_parameter_names,
    )
    freeze_state = _validate_freeze_state(model)
    if not isinstance(bootstrap, Mapping):
        raise ValueError('large sync bootstrap metadata is invalid')
    if (
        bootstrap.get('stage') != 'V2-2'
        or int(bootstrap.get('updates', -1)) < 0
        or not str(bootstrap.get('path', ''))
    ):
        raise ValueError('large sync bootstrap source is invalid')
    if not normalizer or not all(
        isinstance(value, torch.Tensor)
        for value in normalizer.values()
    ):
        raise ValueError('large sync normalizer is invalid')
    return {
        'checkpoint_version': LARGE_SYNC_CHECKPOINT_VERSION,
        'stage': LARGE_SYNC_STAGE,
        'barrier_complete': True,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'amp_scaler': scaler.state_dict(),
        'transition_schema_fingerprint': (
            transition_schema_fingerprint()
        ),
        'config_fingerprint': config_fingerprint_value,
        'normalizer': {
            name: value.detach().cpu().clone()
            for name, value in normalizer.items()
        },
        'learner_rng_state': capture_rng_state(),
        'actor_scene_assignments': assignments,
        'actor_states': states,
        'trainable_parameter_names': trainable,
        'frozen_parameter_names': frozen,
        'freeze_state': freeze_state,
        'bootstrap': dict(bootstrap),
        'next_round_id': counters.next_round_id,
        'updates': counters.updates,
        'policy_version': counters.policy_version,
        'processed_physical_seconds': (
            counters.processed_physical_seconds
        ),
        'episodes': counters.episodes,
        'events': counters.events,
    }


def save_large_sync_checkpoint(
    path: str | pathlib.Path,
    *,
    payload: Mapping[str, Any],
    overwrite: bool = False,
) -> None:
    """原子保存；周期 checkpoint 默认禁止覆盖。"""

    path = pathlib.Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f'large sync checkpoint already exists: {path}',
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f'{path.name}.tmp.{os.getpid()}',
    )
    try:
        torch.save(dict(payload), temporary)
        if path.exists() and not overwrite:
            raise FileExistsError(
                f'large sync checkpoint already exists: {path}',
            )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def update_latest_checkpoint(
    *,
    source: str | pathlib.Path,
    latest: str | pathlib.Path,
) -> None:
    """使用原子 hard link 更新 latest，不复制数百 MiB 权重。"""

    source = pathlib.Path(source)
    latest = pathlib.Path(latest)
    if not source.is_file():
        raise FileNotFoundError(
            f'large sync checkpoint source not found: {source}',
        )
    if source.resolve() == latest.resolve():
        raise ValueError('latest checkpoint cannot point to itself')
    latest.parent.mkdir(parents=True, exist_ok=True)
    temporary = latest.with_name(
        f'{latest.name}.tmp.{os.getpid()}',
    )
    try:
        os.link(source, temporary)
        os.replace(temporary, latest)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_and_validate_root(
    path: str | pathlib.Path,
    *,
    expected_schema_fingerprint: str,
    expected_config_fingerprint: str | None,
) -> Mapping[str, Any]:
    checkpoint = torch.load(
        pathlib.Path(path),
        map_location='cpu',
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError('large sync checkpoint root must be a mapping')
    if checkpoint.get('checkpoint_version') != (
        LARGE_SYNC_CHECKPOINT_VERSION
    ):
        raise ValueError('large sync checkpoint version does not match')
    if checkpoint.get('stage') != LARGE_SYNC_STAGE:
        raise ValueError('large sync checkpoint stage does not match')
    if checkpoint.get('barrier_complete') is not True:
        raise ValueError('large sync checkpoint is not at a barrier')
    if checkpoint.get('transition_schema_fingerprint') != (
        expected_schema_fingerprint
    ):
        raise ValueError('large sync checkpoint schema fingerprint mismatch')
    if (
        expected_config_fingerprint is not None
        and checkpoint.get('config_fingerprint')
        != expected_config_fingerprint
    ):
        raise ValueError('large sync checkpoint config fingerprint mismatch')
    return checkpoint


def load_large_sync_checkpoint(
    path: str | pathlib.Path,
    *,
    model: EventJointActorCritic,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    expected_schema_fingerprint: str,
    expected_config_fingerprint: str,
    expected_scene_assignments: Mapping[int, Sequence[int]],
    expected_trainable_parameter_names: Sequence[str],
    expected_frozen_parameter_names: Sequence[str],
) -> LargeSyncRestore:
    """严格校验后恢复 learner 与所有 actor 的 barrier 状态。"""

    checkpoint = _load_and_validate_root(
        path,
        expected_schema_fingerprint=expected_schema_fingerprint,
        expected_config_fingerprint=expected_config_fingerprint,
    )
    expected_assignments = _normalize_scene_assignments(
        expected_scene_assignments,
    )
    saved_assignments = _normalize_scene_assignments(
        checkpoint.get('actor_scene_assignments', {}),
    )
    if saved_assignments != expected_assignments:
        raise ValueError(
            'large sync checkpoint scene assignment does not match',
        )
    saved_trainable = tuple(
        checkpoint.get('trainable_parameter_names', ()),
    )
    saved_frozen = tuple(
        checkpoint.get('frozen_parameter_names', ()),
    )
    expected_trainable, expected_frozen = (
        _validate_model_parameter_boundary(
            model,
            trainable_parameter_names=(
                expected_trainable_parameter_names
            ),
            frozen_parameter_names=expected_frozen_parameter_names,
        )
    )
    if saved_trainable != expected_trainable:
        raise ValueError(
            'large sync checkpoint trainable parameter names do not match',
        )
    if saved_frozen != expected_frozen:
        raise ValueError(
            'large sync checkpoint frozen parameter names do not match',
        )
    if checkpoint.get('freeze_state') != _validate_freeze_state(model):
        raise ValueError(
            'large sync checkpoint frozen boundary does not match',
        )
    actor_states = _validate_actor_states(
        checkpoint.get('actor_states', {}),
        actor_ids=tuple(expected_assignments),
    )
    normalizer = checkpoint.get('normalizer')
    if not isinstance(normalizer, Mapping) or not all(
        isinstance(value, torch.Tensor)
        for value in normalizer.values()
    ):
        raise ValueError('large sync checkpoint normalizer is invalid')
    bootstrap = checkpoint.get('bootstrap')
    if not isinstance(bootstrap, Mapping):
        raise ValueError('large sync checkpoint bootstrap is invalid')
    counters = LargeSyncCounters(
        next_round_id=int(checkpoint.get('next_round_id', -1)),
        updates=int(checkpoint.get('updates', -1)),
        policy_version=int(checkpoint.get('policy_version', -1)),
        processed_physical_seconds=int(
            checkpoint.get('processed_physical_seconds', -1),
        ),
        episodes=int(checkpoint.get('episodes', -1)),
        events=int(checkpoint.get('events', -1)),
    )

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    scaler.load_state_dict(checkpoint['amp_scaler'])
    restore_rng_state(checkpoint['learner_rng_state'])
    return LargeSyncRestore(
        counters=counters,
        actor_scene_assignments=saved_assignments,
        actor_states=actor_states,
        normalizer=dict(normalizer),
        trainable_parameter_names=saved_trainable,
        frozen_parameter_names=saved_frozen,
        bootstrap=dict(bootstrap),
    )


def load_large_sync_policy_checkpoint(
    *,
    path: str | pathlib.Path,
    model: EventJointActorCritic,
) -> LargeSyncPolicyMetadata:
    """只读恢复 policy，供 heldout/Val/Test 使用。"""

    checkpoint = _load_and_validate_root(
        path,
        expected_schema_fingerprint=transition_schema_fingerprint(),
        expected_config_fingerprint=None,
    )
    if checkpoint.get('freeze_state') != _validate_freeze_state(model):
        raise ValueError(
            'large sync policy checkpoint freeze state does not match',
        )
    assignments = _normalize_scene_assignments(
        checkpoint.get('actor_scene_assignments', {}),
    )
    fingerprint = checkpoint.get('config_fingerprint')
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError(
            'large sync policy config fingerprint is invalid',
        )
    bootstrap = checkpoint.get('bootstrap')
    if not isinstance(bootstrap, Mapping):
        raise ValueError('large sync policy bootstrap is invalid')
    model.load_state_dict(checkpoint['model'])
    return LargeSyncPolicyMetadata(
        stage=LARGE_SYNC_STAGE,
        updates=int(checkpoint.get('updates', -1)),
        policy_version=int(checkpoint.get('policy_version', -1)),
        scene_ids=tuple(
            scene_id
            for actor_id in assignments
            for scene_id in assignments[actor_id]
        ),
        config_fingerprint=fingerprint,
        bootstrap=dict(bootstrap),
    )
