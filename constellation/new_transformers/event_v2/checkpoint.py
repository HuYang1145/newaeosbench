"""V2 训练阶段共用的配置指纹、RNG 与原子 checkpoint。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
import pathlib
import random
from typing import Any

import numpy as np
import torch

from .model import EventJointActorCritic
from .rollout import StoredEventStep
from .transition import transition_schema_fingerprint


SYNC_PPO_CHECKPOINT_VERSION = 1
SYNC_PPO_STAGE = 'V2-1'
SYNC_PPO_STAGES = frozenset({'V2-1', 'V2-2'})
APPO_CHECKPOINT_VERSION = 1
APPO_STAGE = 'V2-3'


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f'config value is not fingerprintable: {type(value)!r}')


def config_fingerprint(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _jsonable(config),
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True,
    ).encode('ascii')
    return hashlib.sha256(encoded).hexdigest()


def capture_rng_state() -> dict[str, Any]:
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    required = {'python', 'numpy', 'torch', 'cuda'}
    if set(state) != required:
        raise ValueError('RNG checkpoint fields do not match')
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint_atomic(
    path: str | pathlib.Path,
    checkpoint: Mapping[str, Any],
) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(dict(checkpoint), temporary)
    os.replace(temporary, path)


@dataclass(frozen=True)
class SyncPPOCounters:
    updates: int = 0
    policy_version: int = 0
    processed_physical_seconds: int = 0
    episodes: int = 0
    events: int = 0

    def __post_init__(self) -> None:
        if any(value < 0 for value in (
            self.updates,
            self.policy_version,
            self.processed_physical_seconds,
            self.episodes,
            self.events,
        )):
            raise ValueError('training counters must be non-negative')


@dataclass(frozen=True)
class APPOCounters:
    updates: int = 0
    policy_version: int = 0
    accepted_events: int = 0
    stale_dropped_events: int = 0
    processed_physical_seconds: int = 0
    episodes: int = 0

    def __post_init__(self) -> None:
        if any(value < 0 for value in (
            self.updates,
            self.policy_version,
            self.accepted_events,
            self.stale_dropped_events,
            self.processed_physical_seconds,
            self.episodes,
        )):
            raise ValueError('APPO training counters must be non-negative')


@dataclass(frozen=True)
class SyncPPORestore:
    counters: SyncPPOCounters
    runtime_states: tuple[Mapping[str, Any], ...]
    normalizer: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class SyncPPOBootstrap:
    source_stage: str
    source_updates: int
    source_policy_version: int
    source_scene_ids: tuple[int, ...]


@dataclass(frozen=True)
class SyncPPOPolicyMetadata:
    stage: str
    updates: int
    policy_version: int
    scene_ids: tuple[int, ...]
    config_fingerprint: str


@dataclass(frozen=True)
class APPORestore:
    counters: APPOCounters
    actor_scene_shards: tuple[tuple[int, ...], ...]
    actor_runtime_states: tuple[tuple[Mapping[str, Any], ...], ...]
    pending_steps: tuple[StoredEventStep, ...]
    normalizer: Mapping[str, torch.Tensor]


def _normalize_actor_scene_shards(
    actor_scene_shards: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    shards = tuple(
        tuple(int(scene_id) for scene_id in shard)
        for shard in actor_scene_shards
    )
    flattened = tuple(scene_id for shard in shards for scene_id in shard)
    if (
        not shards
        or any(not shard for shard in shards)
        or any(scene_id < 0 for scene_id in flattened)
        or len(flattened) != len(set(flattened))
    ):
        raise ValueError(
            'APPO actor scene shards must be non-empty, unique and non-negative',
        )
    return shards


def _normalize_actor_runtime_states(
    actor_runtime_states: Sequence[Sequence[Mapping[str, Any]]],
    *,
    scene_shards: tuple[tuple[int, ...], ...],
) -> tuple[tuple[Mapping[str, Any], ...], ...]:
    states = tuple(tuple(shard) for shard in actor_runtime_states)
    if (
        len(states) != len(scene_shards)
        or any(
            len(runtime_shard) != len(scene_shard)
            for runtime_shard, scene_shard in zip(
                states,
                scene_shards,
                strict=True,
            )
        )
        or not all(
            isinstance(state, Mapping)
            for shard in states
            for state in shard
        )
    ):
        raise ValueError('APPO actor runtime states do not match scene shards')
    return states


def build_appo_checkpoint(
    *,
    model: EventJointActorCritic,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    config_fingerprint_value: str,
    normalizer: Mapping[str, torch.Tensor],
    counters: APPOCounters,
    actor_scene_shards: Sequence[Sequence[int]],
    actor_runtime_states: Sequence[Sequence[Mapping[str, Any]]],
    encoder_layers: int,
    decoder_layers: int,
    backbone_lr_scale: float,
    pending_steps: Sequence[StoredEventStep] = (),
) -> dict[str, Any]:
    if model.backbone_is_frozen:
        raise ValueError('APPO checkpoint requires an unfrozen Stage3 tail')
    if encoder_layers <= 0 or decoder_layers <= 0:
        raise ValueError('APPO unfreeze layer counts must be positive')
    if not 0 < backbone_lr_scale <= 1:
        raise ValueError('APPO backbone learning-rate scale is invalid')
    if len(config_fingerprint_value) != 64:
        raise ValueError('config fingerprint must be a SHA-256 hex digest')
    shards = _normalize_actor_scene_shards(actor_scene_shards)
    runtime_states = _normalize_actor_runtime_states(
        actor_runtime_states,
        scene_shards=shards,
    )
    pending_steps = tuple(pending_steps)
    for step in pending_steps:
        if not isinstance(step, StoredEventStep):
            raise ValueError('APPO pending steps have an invalid type')
        step.validate()
    return {
        'checkpoint_version': APPO_CHECKPOINT_VERSION,
        'stage': APPO_STAGE,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'amp_scaler': scaler.state_dict(),
        'policy_version': counters.policy_version,
        'transition_schema_fingerprint': transition_schema_fingerprint(),
        'config_fingerprint': config_fingerprint_value,
        'normalizer': {
            name: value.detach().cpu().clone()
            for name, value in normalizer.items()
        },
        'rng_state': capture_rng_state(),
        'actor_scene_shards': shards,
        'actor_runtime_states': runtime_states,
        'pending_steps': pending_steps,
        'updates': counters.updates,
        'accepted_events': counters.accepted_events,
        'stale_dropped_events': counters.stale_dropped_events,
        'processed_physical_seconds': counters.processed_physical_seconds,
        'episodes': counters.episodes,
        'unfreeze_state': {
            'backbone_is_frozen': False,
            'encoder_layers': encoder_layers,
            'decoder_layers': decoder_layers,
            'backbone_lr_scale': float(backbone_lr_scale),
        },
    }


def load_appo_checkpoint(
    *,
    path: str | pathlib.Path,
    model: EventJointActorCritic,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    expected_config_fingerprint: str,
    expected_actor_scene_shards: Sequence[Sequence[int]],
    expected_encoder_layers: int,
    expected_decoder_layers: int,
    expected_backbone_lr_scale: float,
) -> APPORestore:
    checkpoint = torch.load(
        pathlib.Path(path),
        map_location='cpu',
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError('APPO checkpoint root must be a mapping')
    if checkpoint.get('checkpoint_version') != APPO_CHECKPOINT_VERSION:
        raise ValueError('APPO checkpoint version does not match')
    if checkpoint.get('stage') != APPO_STAGE:
        raise ValueError('APPO checkpoint stage does not match V2-3')
    if checkpoint.get('transition_schema_fingerprint') != (
        transition_schema_fingerprint()
    ):
        raise ValueError('APPO checkpoint schema fingerprint mismatch')
    if checkpoint.get('config_fingerprint') != expected_config_fingerprint:
        raise ValueError('APPO checkpoint config fingerprint mismatch')
    shards = _normalize_actor_scene_shards(expected_actor_scene_shards)
    if tuple(checkpoint.get('actor_scene_shards', ())) != shards:
        raise ValueError('APPO checkpoint actor scene shards do not match')
    expected_unfreeze = {
        'backbone_is_frozen': False,
        'encoder_layers': expected_encoder_layers,
        'decoder_layers': expected_decoder_layers,
        'backbone_lr_scale': float(expected_backbone_lr_scale),
    }
    if checkpoint.get('unfreeze_state') != expected_unfreeze:
        raise ValueError('APPO checkpoint unfreeze state does not match')
    if model.backbone_is_frozen:
        raise ValueError('APPO restore target must unfreeze Stage3 first')
    runtime_states = _normalize_actor_runtime_states(
        checkpoint.get('actor_runtime_states', ()),
        scene_shards=shards,
    )
    pending_steps = checkpoint.get('pending_steps', ())
    if not isinstance(pending_steps, (list, tuple)):
        raise ValueError('APPO checkpoint pending steps are invalid')
    pending_steps = tuple(pending_steps)
    for step in pending_steps:
        if not isinstance(step, StoredEventStep):
            raise ValueError('APPO checkpoint pending step type is invalid')
        step.validate()
    normalizer = checkpoint.get('normalizer')
    if not isinstance(normalizer, Mapping) or not all(
        isinstance(value, torch.Tensor) for value in normalizer.values()
    ):
        raise ValueError('APPO checkpoint normalizer is invalid')
    counters = APPOCounters(
        updates=int(checkpoint.get('updates', -1)),
        policy_version=int(checkpoint.get('policy_version', -1)),
        accepted_events=int(checkpoint.get('accepted_events', -1)),
        stale_dropped_events=int(
            checkpoint.get('stale_dropped_events', -1),
        ),
        processed_physical_seconds=int(
            checkpoint.get('processed_physical_seconds', -1),
        ),
        episodes=int(checkpoint.get('episodes', -1)),
    )
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    scaler.load_state_dict(checkpoint['amp_scaler'])
    restore_rng_state(checkpoint['rng_state'])
    return APPORestore(
        counters=counters,
        actor_scene_shards=shards,
        actor_runtime_states=runtime_states,
        pending_steps=pending_steps,
        normalizer=dict(normalizer),
    )


def build_sync_ppo_checkpoint(
    *,
    stage: str = SYNC_PPO_STAGE,
    model: EventJointActorCritic,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    config_fingerprint_value: str,
    normalizer: Mapping[str, torch.Tensor],
    counters: SyncPPOCounters,
    scene_ids: Sequence[int],
    runtime_states: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if stage not in SYNC_PPO_STAGES:
        raise ValueError(f'unsupported synchronous PPO stage: {stage!r}')
    scene_ids = tuple(int(scene_id) for scene_id in scene_ids)
    if (
        not scene_ids
        or len(set(scene_ids)) != len(scene_ids)
        or any(scene_id < 0 for scene_id in scene_ids)
    ):
        raise ValueError('checkpoint scene IDs must be unique and non-negative')
    if len(runtime_states) != len(scene_ids):
        raise ValueError('checkpoint needs one runtime state per scene')
    if len(config_fingerprint_value) != 64:
        raise ValueError('config fingerprint must be a SHA-256 hex digest')
    return {
        'checkpoint_version': SYNC_PPO_CHECKPOINT_VERSION,
        'stage': stage,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'amp_scaler': scaler.state_dict(),
        'policy_version': counters.policy_version,
        'transition_schema_fingerprint': transition_schema_fingerprint(),
        'config_fingerprint': config_fingerprint_value,
        'normalizer': {
            name: value.detach().cpu().clone()
            for name, value in normalizer.items()
        },
        'rng_state': capture_rng_state(),
        'runtime_states': tuple(runtime_states),
        'scene_ids': scene_ids,
        'processed_physical_seconds': counters.processed_physical_seconds,
        'episodes': counters.episodes,
        'events': counters.events,
        'updates': counters.updates,
        'unfreeze_state': {
            'backbone_is_frozen': model.backbone_is_frozen,
        },
    }


def load_sync_ppo_checkpoint(
    *,
    path: str | pathlib.Path,
    model: EventJointActorCritic,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    expected_stage: str = SYNC_PPO_STAGE,
    expected_config_fingerprint: str,
    expected_scene_ids: Sequence[int],
) -> SyncPPORestore:
    checkpoint = torch.load(
        pathlib.Path(path),
        map_location='cpu',
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError('checkpoint root must be a mapping')
    if checkpoint.get('checkpoint_version') != SYNC_PPO_CHECKPOINT_VERSION:
        raise ValueError('checkpoint version does not match synchronous PPO')
    if expected_stage not in SYNC_PPO_STAGES:
        raise ValueError(
            f'unsupported synchronous PPO stage: {expected_stage!r}',
        )
    if checkpoint.get('stage') != expected_stage:
        raise ValueError(
            f'checkpoint stage does not match {expected_stage}',
        )
    if checkpoint.get('transition_schema_fingerprint') != (
        transition_schema_fingerprint()
    ):
        raise ValueError('checkpoint schema fingerprint mismatch')
    if checkpoint.get('config_fingerprint') != expected_config_fingerprint:
        raise ValueError('checkpoint config fingerprint mismatch')
    expected_scene_ids = tuple(int(value) for value in expected_scene_ids)
    if tuple(checkpoint.get('scene_ids', ())) != expected_scene_ids:
        raise ValueError('checkpoint scene IDs do not match')
    runtime_states = checkpoint.get('runtime_states')
    if (
        not isinstance(runtime_states, (list, tuple))
        or len(runtime_states) != len(expected_scene_ids)
        or not all(isinstance(value, Mapping) for value in runtime_states)
    ):
        raise ValueError('checkpoint runtime states do not match scenes')
    if checkpoint.get('unfreeze_state') != {'backbone_is_frozen': True}:
        raise ValueError('checkpoint does not preserve the V2-1 freeze state')
    normalizer = checkpoint.get('normalizer')
    if not isinstance(normalizer, Mapping) or not all(
        isinstance(value, torch.Tensor) for value in normalizer.values()
    ):
        raise ValueError('checkpoint normalizer is invalid')
    counters = SyncPPOCounters(
        updates=int(checkpoint.get('updates', -1)),
        policy_version=int(checkpoint.get('policy_version', -1)),
        processed_physical_seconds=int(
            checkpoint.get('processed_physical_seconds', -1)
        ),
        episodes=int(checkpoint.get('episodes', -1)),
        events=int(checkpoint.get('events', -1)),
    )

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    scaler.load_state_dict(checkpoint['amp_scaler'])
    restore_rng_state(checkpoint['rng_state'])
    return SyncPPORestore(
        counters=counters,
        runtime_states=tuple(runtime_states),
        normalizer=dict(normalizer),
    )


def load_sync_ppo_bootstrap_checkpoint(
    *,
    path: str | pathlib.Path,
    model: EventJointActorCritic,
    optimizer: torch.optim.Optimizer,
    expected_source_stage: str = SYNC_PPO_STAGE,
) -> SyncPPOBootstrap:
    """只继承策略和 optimizer，不继承旧场景、计数器或 RNG。"""

    checkpoint = torch.load(
        pathlib.Path(path),
        map_location='cpu',
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError('bootstrap checkpoint root must be a mapping')
    if checkpoint.get('checkpoint_version') != SYNC_PPO_CHECKPOINT_VERSION:
        raise ValueError('bootstrap checkpoint version does not match')
    if expected_source_stage not in SYNC_PPO_STAGES:
        raise ValueError(
            f'unsupported bootstrap source stage: {expected_source_stage!r}',
        )
    if checkpoint.get('stage') != expected_source_stage:
        raise ValueError(
            f'bootstrap checkpoint stage does not match {expected_source_stage}',
        )
    if checkpoint.get('transition_schema_fingerprint') != (
        transition_schema_fingerprint()
    ):
        raise ValueError('bootstrap checkpoint schema fingerprint mismatch')
    if checkpoint.get('unfreeze_state') != {'backbone_is_frozen': True}:
        raise ValueError('bootstrap checkpoint does not preserve freeze state')

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if not model.backbone_is_frozen:
        raise ValueError('bootstrap unexpectedly unfreezes Stage3')
    return SyncPPOBootstrap(
        source_stage=str(checkpoint['stage']),
        source_updates=int(checkpoint.get('updates', -1)),
        source_policy_version=int(checkpoint.get('policy_version', -1)),
        source_scene_ids=tuple(
            int(scene_id) for scene_id in checkpoint.get('scene_ids', ())
        ),
    )


def load_sync_ppo_policy_checkpoint(
    *,
    path: str | pathlib.Path,
    model: EventJointActorCritic,
    expected_stages: Sequence[str] = tuple(SYNC_PPO_STAGES),
) -> SyncPPOPolicyMetadata:
    """只读加载同步 PPO policy，不恢复任何训练或 runtime 状态。"""

    checkpoint = torch.load(
        pathlib.Path(path),
        map_location='cpu',
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError('policy checkpoint root must be a mapping')
    if checkpoint.get('checkpoint_version') != SYNC_PPO_CHECKPOINT_VERSION:
        raise ValueError('policy checkpoint version does not match')
    expected_stages = tuple(str(stage) for stage in expected_stages)
    if (
        not expected_stages
        or any(stage not in SYNC_PPO_STAGES for stage in expected_stages)
    ):
        raise ValueError('policy expected stages are invalid')
    stage = checkpoint.get('stage')
    if stage not in expected_stages:
        raise ValueError('policy checkpoint stage does not match')
    if checkpoint.get('transition_schema_fingerprint') != (
        transition_schema_fingerprint()
    ):
        raise ValueError('policy checkpoint schema fingerprint mismatch')
    if checkpoint.get('unfreeze_state') != {'backbone_is_frozen': True}:
        raise ValueError('policy checkpoint does not preserve freeze state')
    fingerprint = checkpoint.get('config_fingerprint')
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError('policy checkpoint config fingerprint is invalid')

    model.load_state_dict(checkpoint['model'])
    if not model.backbone_is_frozen:
        raise ValueError('policy checkpoint unexpectedly unfreezes Stage3')
    return SyncPPOPolicyMetadata(
        stage=str(stage),
        updates=int(checkpoint.get('updates', -1)),
        policy_version=int(checkpoint.get('policy_version', -1)),
        scene_ids=tuple(
            int(scene_id) for scene_id in checkpoint.get('scene_ids', ())
        ),
        config_fingerprint=fingerprint,
    )
