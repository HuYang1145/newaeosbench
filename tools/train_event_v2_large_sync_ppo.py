#!/usr/bin/env python3
"""训练单个 seed 的 Event V2 大规模严格同步 PPO。"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import json
import pathlib
import queue
import random
import runpy
import signal
import sys
import traceback
from typing import Any

import numpy as np
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from constellation.new_transformers.event_v2.appo import (
    SharedPolicyStore,
)
from constellation.new_transformers.event_v2.basilisk_runtime import (
    BasiliskEventRuntime,
    BasiliskSceneBackend,
    load_runtime_statistics,
)
from constellation.new_transformers.event_v2.checkpoint import (
    config_fingerprint,
    load_sync_ppo_bootstrap_checkpoint,
)
from constellation.new_transformers.event_v2.distributed_sync import (
    QueuedEventRuntimePool,
    StrictSyncRoundCoordinator,
    StrictSyncUpdateAccumulator,
    SyncActorChunk,
    SyncActorDone,
    SyncRoundCommand,
    SyncWorkerError,
    capture_rng_state,
    restore_rng_state,
    run_strict_sync_actor_loop,
)
from constellation.new_transformers.event_v2.large_sync_checkpoint import (
    LargeSyncCounters,
    build_large_sync_checkpoint_payload,
    load_large_sync_checkpoint,
    save_large_sync_checkpoint,
    update_latest_checkpoint,
)
from constellation.new_transformers.event_v2.model import (
    EventJointActorCritic,
)
from constellation.new_transformers.event_v2.ppo import (
    PPOConfig,
    SynchronousPPOTrainer,
)
from constellation.new_transformers.event_v2.transition import (
    transition_schema_fingerprint,
)
from tools.train_event_v2_sync_ppo import (
    SyntheticEventRuntime,
    _normalizer_state,
    _sample_action,
    _seed_everything,
    _step_scheduler_without_restart,
    _tiny_model,
)


_STOP_REQUESTED = False


def _request_stop(signum, frame) -> None:
    del signum, frame
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train one Event V2 large strict-sync PPO seed',
    )
    parser.add_argument('--config', type=pathlib.Path, required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--learner-device', default='auto')
    parser.add_argument('--actor-devices', nargs='+')
    parser.add_argument('--actors', type=int)
    parser.add_argument('--active-environments', type=int)
    parser.add_argument('--scene-start', type=int)
    parser.add_argument('--scene-end', type=int)
    parser.add_argument('--max-time-step', type=int)
    parser.add_argument('--max-updates', type=int)
    parser.add_argument('--checkpoint-every-updates', type=int)
    parser.add_argument('--bootstrap-checkpoint', type=pathlib.Path)
    parser.add_argument('--output-dir', type=pathlib.Path)
    parser.add_argument('--resume', type=pathlib.Path)
    parser.add_argument('--synthetic-preflight', action='store_true')
    return parser.parse_args()


def _load_config(path: pathlib.Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f'config not found: {path}')
    config = {
        key: value
        for key, value in runpy.run_path(str(path)).items()
        if not key.startswith('__')
    }
    if config.get('stage') != 'V2-2-Large':
        raise ValueError('large sync config stage must be V2-2-Large')
    if config.get('split') != 'train':
        raise ValueError('large sync PPO may only consume train scenes')
    if config.get('gamma') != 1.0:
        raise ValueError('large sync completion reward requires gamma=1')
    if not config.get('freeze_backbone'):
        raise ValueError('large sync must freeze the Stage3 backbone')
    if config.get('amp_dtype') not in {'bfloat16', 'float16'}:
        raise ValueError('large sync AMP dtype is invalid')
    scene_ids = tuple(int(value) for value in config.get('scene_ids', ()))
    if (
        not scene_ids
        or len(scene_ids) != len(set(scene_ids))
        or any(scene_id < 0 for scene_id in scene_ids)
    ):
        raise ValueError('large sync scene IDs are invalid')
    config['scene_ids'] = scene_ids
    for name in (
        'actor_count',
        'active_environments',
        'events_per_actor_round',
        'min_update_events',
        'max_updates',
        'checkpoint_interval',
        'max_time_step',
        'safety_review_seconds',
        'ppo_epochs',
        'minibatch_events',
    ):
        if int(config.get(name, 0)) <= 0:
            raise ValueError(f'large sync {name} must be positive')
    if int(config['active_environments']) > len(scene_ids):
        raise ValueError(
            'large sync active environments exceed scene count',
        )
    if int(config['actor_count']) > int(config['active_environments']):
        raise ValueError('large sync actor count exceeds active environments')
    optimizer = config.get('optimizer')
    if (
        not isinstance(optimizer, Mapping)
        or float(optimizer.get('lr', 0)) <= 0
    ):
        raise ValueError('large sync optimizer learning rate is invalid')
    if not isinstance(config.get('model'), Mapping):
        raise ValueError('large sync model config is invalid')
    return config


def _device_from_argument(value: str) -> torch.device:
    if value == 'auto':
        return torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu',
        )
    device = torch.device(value)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    return device


def deterministic_scene_assignments(
    scene_ids: Sequence[int],
    *,
    actor_count: int,
) -> dict[int, tuple[int, ...]]:
    """按 actor id 连续、均衡且确定性地切分 scene。"""

    scene_ids = tuple(int(scene_id) for scene_id in scene_ids)
    if (
        not scene_ids
        or not 1 <= actor_count <= len(scene_ids)
        or len(scene_ids) != len(set(scene_ids))
    ):
        raise ValueError('large sync actor count or scene IDs are invalid')
    quotient, remainder = divmod(len(scene_ids), actor_count)
    assignments: dict[int, tuple[int, ...]] = {}
    start = 0
    for actor_id in range(actor_count):
        width = quotient + int(actor_id < remainder)
        assignments[actor_id] = scene_ids[start:start + width]
        start += width
    return assignments


def deterministic_active_environment_caps(
    assignments: Mapping[int, Sequence[int]],
    *,
    total_active_environments: int,
) -> dict[int, int]:
    """把活跃环境预算均衡分给 actor，且不超过各自 scene 数。"""

    actor_ids = tuple(sorted(int(value) for value in assignments))
    total_scenes = sum(len(assignments[actor_id]) for actor_id in actor_ids)
    if (
        not actor_ids
        or total_active_environments < len(actor_ids)
        or total_active_environments > total_scenes
    ):
        raise ValueError('large sync active environment budget is invalid')
    quotient, remainder = divmod(
        total_active_environments,
        len(actor_ids),
    )
    caps = {
        actor_id: quotient + int(index < remainder)
        for index, actor_id in enumerate(actor_ids)
    }
    if any(
        caps[actor_id] > len(assignments[actor_id])
        for actor_id in actor_ids
    ):
        raise ValueError(
            'large sync active environment cap exceeds actor scenes',
        )
    return caps


def resolve_actor_devices(
    device_values: Sequence[str],
    *,
    actor_count: int,
) -> tuple[str, ...]:
    """在给定 GPU 列表上轮转 actor，不改变 actor 数。"""

    values = tuple(str(value) for value in device_values)
    if not values or actor_count <= 0 or len(values) > actor_count:
        raise ValueError('large sync actor device list is invalid')
    for value in values:
        device = torch.device(value)
        if device.type not in {'cpu', 'cuda'}:
            raise ValueError('large sync actor device type is invalid')
    return tuple(
        values[actor_id % len(values)]
        for actor_id in range(actor_count)
    )


def parameter_inventory(
    model: EventJointActorCritic,
) -> dict[str, Any]:
    """记录训练边界，checkpoint 恢复时逐项核对。"""

    trainable_names = tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    frozen_names = tuple(
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    transformer_trainable = sum(
        parameter.numel()
        for parameter in model.backbone.transformer.parameters()
        if parameter.requires_grad
    )
    return {
        'total_parameters': total,
        'trainable_parameters': trainable,
        'frozen_parameters': total - trainable,
        'transformer_trainable_parameters': transformer_trainable,
        'trainable_parameter_names': trainable_names,
        'frozen_parameter_names': frozen_names,
    }


def _optimizer_from_config(
    model: EventJointActorCritic,
    config: Mapping[str, Any],
) -> torch.optim.Optimizer:
    optimizer_config = dict(config['optimizer'])
    learning_rate = float(optimizer_config.pop('lr'))
    return torch.optim.AdamW(
        model.parameter_groups(learning_rate),
        **optimizer_config,
    )


def build_large_sync_model_from_bootstrap(
    *,
    config: Mapping[str, Any],
    checkpoint_path: pathlib.Path,
    device: torch.device,
):
    """只继承 V2-2 model/optimizer，不继承 runtime、计数器或 RNG。"""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f'V2-2 bootstrap checkpoint not found: {checkpoint_path}',
        )
    rng_state = capture_rng_state()
    try:
        model = EventJointActorCritic(**config['model']).to(device)
        optimizer = _optimizer_from_config(model, config)
        metadata = load_sync_ppo_bootstrap_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            expected_source_stage='V2-2',
        )
    finally:
        restore_rng_state(rng_state)
    if not model.backbone_is_frozen or any(
        parameter.requires_grad
        for parameter in model.backbone.transformer.parameters()
    ):
        raise ValueError(
            'large sync bootstrap unexpectedly unfreezes Stage3',
        )
    return model, optimizer, metadata


def _ppo_config(
    config: Mapping[str, Any],
    *,
    synthetic: bool = False,
) -> PPOConfig:
    return PPOConfig(
        clip_ratio=float(config['clip_ratio']),
        value_coefficient=float(config['value_coefficient']),
        entropy_coefficient=float(config['entropy_coefficient']),
        max_grad_norm=float(config['max_grad_norm']),
        max_kl=10.0 if synthetic else float(config['max_kl']),
        ppo_epochs=1 if synthetic else int(config['ppo_epochs']),
        minibatch_events=(
            8 if synthetic else int(config['minibatch_events'])
        ),
        lambda_base=float(config['lambda_base']),
        reference_seconds=float(config['reference_seconds']),
        replay_atol=float(config['logprob_replay_atol']),
    )


def _synthetic_runtime_loader(
    scene_id: int,
    state: Mapping[str, Any],
) -> SyntheticEventRuntime:
    del scene_id
    runtime = SyntheticEventRuntime(num_events=int(state['num_events']))
    runtime.events = int(state['events'])
    runtime.total_reward = float(state['total_reward'])
    return runtime


def run_synthetic_preflight(
    *,
    config: Mapping[str, Any],
    output_dir: pathlib.Path,
    max_updates: int,
) -> dict[str, Any]:
    """用两个 actor 验证严格版本、更新、checkpoint 和冻结边界。"""

    if max_updates != 2:
        raise ValueError(
            'large sync synthetic preflight requires exactly 2 updates',
        )
    _seed_everything(int(config['seed']), torch.device('cpu'))
    learner_model = _tiny_model()
    optimizer = torch.optim.AdamW(
        learner_model.parameter_groups(1e-3),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_updates,
        eta_min=1e-4,
    )
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    trainer = SynchronousPPOTrainer(
        model=learner_model,
        optimizer=optimizer,
        config=_ppo_config(config, synthetic=True),
        device=torch.device('cpu'),
        scaler=scaler,
    )
    inventory = parameter_inventory(learner_model)
    context = torch.multiprocessing.get_context('spawn')
    policy_store = SharedPolicyStore(
        copy.deepcopy(learner_model),
        context=context,
        initial_version=0,
    )
    actor_models = [copy.deepcopy(learner_model) for _ in range(2)]
    actor_versions = [-1, -1]
    assignments = {0: (0,), 1: (1,)}
    pools = {
        actor_id: QueuedEventRuntimePool(
            assigned_scene_ids=scene_ids,
            max_active_environments=1,
            runtime_factory=lambda _scene_id: SyntheticEventRuntime(
                num_events=8,
            ),
            runtime_state_loader=_synthetic_runtime_loader,
        )
        for actor_id, scene_ids in assignments.items()
    }
    actor_states: dict[int, Mapping[str, Any]] = {}
    metrics = []
    replay_max_error = 0.0
    frozen_changes = 0
    counters = LargeSyncCounters()
    for round_id in range(max_updates):
        coordinator = StrictSyncRoundCoordinator(
            actor_ids=(0, 1),
            initial_round_id=round_id,
            initial_policy_version=trainer.policy_version,
        )
        for actor_id in (0, 1):
            refresh = policy_store.refresh(
                actor_models[actor_id],
                last_version=actor_versions[actor_id],
            )
            actor_versions[actor_id] = refresh.version
            if refresh.version != trainer.policy_version:
                raise RuntimeError(
                    'synthetic actor refreshed the wrong policy version',
                )
            payload = pools[actor_id].collect(
                model=actor_models[actor_id],
                policy_version=refresh.version,
                max_events=4,
                device=torch.device('cpu'),
                replay_atol=float(config['logprob_replay_atol']),
            )
            actor_state = {
                'pool': pools[actor_id].state_dict(),
                'rng': capture_rng_state(),
            }
            actor_states[actor_id] = actor_state
            coordinator.submit(SyncActorChunk(
                actor_id=actor_id,
                round_id=round_id,
                policy_version=refresh.version,
                steps=payload.steps,
                completed_scene_ids=payload.completed_scene_ids,
                replay_max_abs_error=payload.replay_max_abs_error,
                state=actor_state,
            ))
        batch = coordinator.finalize(min_batch_events=8)
        if not batch.should_update:
            raise RuntimeError(
                'synthetic strict sync round produced a small batch',
            )
        update = trainer.update(list(batch.steps))
        _step_scheduler_without_restart(scheduler)
        policy_store.publish(
            learner_model,
            version=trainer.policy_version,
        )
        replay_max_error = max(
            replay_max_error,
            batch.replay_max_abs_error,
        )
        frozen_changes += update.frozen_parameter_changes
        metrics.append(update)
        counters = LargeSyncCounters(
            next_round_id=round_id + 1,
            updates=counters.updates + 1,
            policy_version=trainer.policy_version,
            processed_physical_seconds=(
                counters.processed_physical_seconds
                + int(batch.processed_physical_seconds)
            ),
            episodes=sum(
                len(pool.completed_scene_ids)
                for pool in pools.values()
            ),
            events=counters.events + batch.event_count,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = config_fingerprint({
        'mode': 'synthetic_large_sync',
        'seed': int(config['seed']),
        'max_updates': max_updates,
    })
    checkpoint = build_large_sync_checkpoint_payload(
        model=learner_model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config_fingerprint_value=fingerprint,
        normalizer={'mean': torch.zeros(1), 'std': torch.ones(1)},
        counters=counters,
        actor_scene_assignments=assignments,
        actor_states=actor_states,
        trainable_parameter_names=inventory['trainable_parameter_names'],
        frozen_parameter_names=inventory['frozen_parameter_names'],
        bootstrap={
            'path': 'synthetic_v2_2.pth',
            'stage': 'V2-2',
            'updates': 1046,
        },
    )
    checkpoint_path = (
        output_dir
        / f'checkpoint_update_{counters.updates:06d}.pth'
    )
    save_large_sync_checkpoint(checkpoint_path, payload=checkpoint)
    update_latest_checkpoint(
        source=checkpoint_path,
        latest=output_dir / 'checkpoint_latest.pth',
    )
    probe = next(iter(pools.values())).state_dict()
    del probe
    from tools.train_event_v2_sync_ppo import _synthetic_observation

    observation = _synthetic_observation(0)
    expected_action = _sample_action(
        learner_model,
        observation,
        device=torch.device('cpu'),
    )
    load_large_sync_checkpoint(
        checkpoint_path,
        model=learner_model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        expected_schema_fingerprint=transition_schema_fingerprint(),
        expected_config_fingerprint=fingerprint,
        expected_scene_assignments=assignments,
        expected_trainable_parameter_names=(
            inventory['trainable_parameter_names']
        ),
        expected_frozen_parameter_names=(
            inventory['frozen_parameter_names']
        ),
    )
    actual_action = _sample_action(
        learner_model,
        observation,
        device=torch.device('cpu'),
    )
    checkpoint_action_reproduced = bool(
        torch.equal(
            expected_action.actor.action.terminate,
            actual_action.actor.action.terminate,
        )
        and torch.equal(
            expected_action.actor.action.task_indices,
            actual_action.actor.action.task_indices,
        )
        and torch.equal(
            expected_action.actor.action.commitment_indices,
            actual_action.actor.action.commitment_indices,
        )
    )
    reward_error = max(
        error
        for pool in pools.values()
        for _, error in pool.reward_reconstruction_errors
    )
    finite = all(
        np.isfinite(value)
        for item in metrics
        for value in (
            item.total_loss,
            item.policy_loss,
            item.value_loss,
            item.entropy,
            item.approx_kl,
            item.gradient_norm,
        )
    )
    accepted = bool(
        counters.updates == 2
        and counters.policy_version == 2
        and counters.next_round_id == 2
        and counters.events == 16
        and all(pool.is_complete for pool in pools.values())
        and replay_max_error <= float(config['logprob_replay_atol'])
        and reward_error <= 1e-6
        and frozen_changes == 0
        and checkpoint_action_reproduced
        and finite
    )
    summary = {
        'stage': 'V2-2-Large',
        'mode': 'synthetic_strict_sync',
        'accepted': accepted,
        'updates': counters.updates,
        'policy_version': counters.policy_version,
        'next_round_id': counters.next_round_id,
        'events': counters.events,
        'stale_rollout_events': 0,
        'logprob_replay_max_error': replay_max_error,
        'reward_reconstruction_max_error': reward_error,
        'frozen_parameter_changed_count': frozen_changes,
        'checkpoint_first_action_reproduced': (
            checkpoint_action_reproduced
        ),
        'finite': finite,
        'checkpoint': str(checkpoint_path),
        'schema_fingerprint': transition_schema_fingerprint(),
        'config_fingerprint': fingerprint,
    }
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return summary


def _actor_worker_entry(
    *,
    actor_id: int,
    assigned_scene_ids: tuple[int, ...],
    active_environment_cap: int,
    model_config: Mapping[str, Any],
    policy_store: SharedPolicyStore,
    command_queue,
    result_queue,
    stop_event,
    device_value: str,
    seed: int,
    max_time_step: int,
    safety_review_seconds: int,
    target_events: int,
    replay_atol: float,
    amp_enabled: bool,
    amp_dtype_name: str,
    initial_round_id: int,
    initial_actor_state: Mapping[str, Any] | None,
) -> None:
    """spawn actor 入口；异常必须返回 learner，不能静默退出。"""

    try:
        device = _device_from_argument(device_value)
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        torch.set_num_threads(1)
        _seed_everything(seed, device)
        statistics = load_runtime_statistics()

        def runtime_factory(scene_id: int) -> BasiliskEventRuntime:
            return BasiliskEventRuntime(
                backend=BasiliskSceneBackend.from_scene_id(
                    split='train',
                    scene_id=scene_id,
                    max_time_step=max_time_step,
                ),
                statistics=statistics,
                safety_review_seconds=safety_review_seconds,
            )

        def runtime_loader(
            scene_id: int,
            state: Mapping[str, Any],
        ) -> BasiliskEventRuntime:
            del scene_id
            return BasiliskEventRuntime.from_state_dict(
                state,
                statistics=statistics,
            )

        pool = QueuedEventRuntimePool(
            assigned_scene_ids=assigned_scene_ids,
            max_active_environments=active_environment_cap,
            runtime_factory=runtime_factory,
            runtime_state_loader=runtime_loader,
            initialize=initial_actor_state is None,
        )
        if initial_actor_state is not None:
            pool.load_state_dict(initial_actor_state['pool'])
        model = EventJointActorCritic(**model_config)
        if initial_actor_state is not None:
            restore_rng_state(initial_actor_state['rng'])
        amp_dtype = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
        }[amp_dtype_name]
        run_strict_sync_actor_loop(
            model=model,
            pool=pool,
            actor_id=actor_id,
            policy_store=policy_store,
            command_queue=command_queue,
            result_queue=result_queue,
            stop_event=stop_event,
            target_events=target_events,
            device=device,
            replay_atol=replay_atol,
            initial_round_id=initial_round_id,
            amp_enabled=amp_enabled and device.type == 'cuda',
            amp_dtype=amp_dtype,
        )
        close = getattr(result_queue, 'close', None)
        if callable(close):
            close()
        join_thread = getattr(result_queue, 'join_thread', None)
        if callable(join_thread):
            join_thread()
    except BaseException as error:
        result_queue.put(SyncWorkerError(
            actor_id=actor_id,
            error_type=type(error).__name__,
            message=str(error),
            traceback=traceback.format_exc(),
        ))


def _pool_state_is_complete(
    actor_state: Mapping[str, Any],
    *,
    assigned_scene_ids: Sequence[int],
) -> bool:
    pool = actor_state['pool']
    return bool(
        not tuple(pool.get('pending_scene_ids', ()))
        and not tuple(pool.get('active', ()))
        and set(pool.get('completed_scene_ids', ()))
        == set(assigned_scene_ids)
    )


def _training_fingerprint(
    config: Mapping[str, Any],
    *,
    scene_ids: Sequence[int],
    seed: int,
    actor_count: int,
    active_environments: int,
    max_time_step: int,
) -> str:
    names = (
        'stage',
        'split',
        'safety_review_seconds',
        'events_per_actor_round',
        'min_update_events',
        'checkpoint_interval',
        'gamma',
        'lambda_base',
        'reference_seconds',
        'clip_ratio',
        'value_coefficient',
        'entropy_coefficient',
        'max_grad_norm',
        'max_kl',
        'ppo_epochs',
        'minibatch_events',
        'logprob_replay_atol',
        'freeze_backbone',
        'amp',
        'amp_dtype',
        'model',
        'optimizer',
        'max_updates',
    )
    effective = {name: config[name] for name in names}
    effective.update({
        'scene_ids': tuple(scene_ids),
        'seed': seed,
        'actor_count': actor_count,
        'active_environments': active_environments,
        'max_time_step': max_time_step,
    })
    return config_fingerprint(effective)


def run_real_training(
    *,
    config: Mapping[str, Any],
    output_dir: pathlib.Path,
    bootstrap_checkpoint: pathlib.Path,
    scene_ids: tuple[int, ...],
    seed: int,
    actor_count: int,
    active_environments: int,
    max_time_step: int,
    max_updates: int,
    checkpoint_interval: int,
    learner_device: torch.device,
    actor_device_values: Sequence[str],
    resume: pathlib.Path | None,
) -> dict[str, Any]:
    """运行一个 learner 和多个命令驱动 actor 的严格同步 PPO。"""

    if learner_device.type != 'cuda':
        raise ValueError('formal large sync training requires a CUDA learner')
    assignments = deterministic_scene_assignments(
        scene_ids,
        actor_count=actor_count,
    )
    active_caps = deterministic_active_environment_caps(
        assignments,
        total_active_environments=active_environments,
    )
    actor_devices = resolve_actor_devices(
        actor_device_values,
        actor_count=actor_count,
    )
    if any(
        _device_from_argument(value).type != learner_device.type
        for value in actor_devices
    ):
        raise ValueError(
            'large sync learner and actors must use one device type',
        )
    _seed_everything(seed, learner_device)
    learner_model, optimizer, bootstrap_metadata = (
        build_large_sync_model_from_bootstrap(
            config=config,
            checkpoint_path=bootstrap_checkpoint,
            device=learner_device,
        )
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(config['max_updates']), 1),
        eta_min=float(config['optimizer']['lr']) * 0.1,
    )
    scaler = torch.amp.GradScaler(
        learner_device.type,
        enabled=False,
    )
    inventory = parameter_inventory(learner_model)
    fingerprint = _training_fingerprint(
        config,
        scene_ids=scene_ids,
        seed=seed,
        actor_count=actor_count,
        active_environments=active_environments,
        max_time_step=max_time_step,
    )
    counters = LargeSyncCounters()
    actor_states: dict[int, Mapping[str, Any]] = {}
    bootstrap_audit = {
        'path': str(bootstrap_checkpoint),
        'stage': bootstrap_metadata.source_stage,
        'updates': bootstrap_metadata.source_updates,
        'policy_version': bootstrap_metadata.source_policy_version,
        'scene_ids': bootstrap_metadata.source_scene_ids,
    }
    statistics = load_runtime_statistics()
    normalizer = _normalizer_state(statistics)
    if resume is not None:
        restored = load_large_sync_checkpoint(
            resume,
            model=learner_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_schema_fingerprint=transition_schema_fingerprint(),
            expected_config_fingerprint=fingerprint,
            expected_scene_assignments=assignments,
            expected_trainable_parameter_names=(
                inventory['trainable_parameter_names']
            ),
            expected_frozen_parameter_names=(
                inventory['frozen_parameter_names']
            ),
        )
        counters = restored.counters
        actor_states = dict(restored.actor_states)
        if dict(restored.bootstrap) != bootstrap_audit:
            raise ValueError(
                'large sync resume bootstrap metadata changed',
            )

    amp_enabled = bool(
        config['amp'] and learner_device.type == 'cuda'
    )
    amp_dtype = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
    }[str(config['amp_dtype'])]
    trainer = SynchronousPPOTrainer(
        model=learner_model,
        optimizer=optimizer,
        config=_ppo_config(config),
        device=learner_device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        scaler=scaler,
    )
    trainer.policy_version = counters.policy_version

    shared_template = EventJointActorCritic(**config['model'])
    shared_template.load_state_dict(learner_model.state_dict())
    context = torch.multiprocessing.get_context('spawn')
    policy_store = SharedPolicyStore(
        shared_template,
        context=context,
        initial_version=trainer.policy_version,
    )
    del shared_template
    result_queue = context.Queue(maxsize=max(16, actor_count * 4))
    stop_event = context.Event()
    command_queues = {
        actor_id: context.Queue(maxsize=1)
        for actor_id in assignments
    }
    active_actor_ids = {
        actor_id
        for actor_id in assignments
        if (
            actor_id not in actor_states
            or not _pool_state_is_complete(
                actor_states[actor_id],
                assigned_scene_ids=assignments[actor_id],
            )
        )
    }
    processes = {}
    for actor_id in sorted(active_actor_ids):
        process = context.Process(
            target=_actor_worker_entry,
            kwargs={
                'actor_id': actor_id,
                'assigned_scene_ids': assignments[actor_id],
                'active_environment_cap': active_caps[actor_id],
                'model_config': dict(config['model']),
                'policy_store': policy_store,
                'command_queue': command_queues[actor_id],
                'result_queue': result_queue,
                'stop_event': stop_event,
                'device_value': actor_devices[actor_id],
                'seed': seed + 1009 * (actor_id + 1),
                'max_time_step': max_time_step,
                'safety_review_seconds': int(
                    config['safety_review_seconds'],
                ),
                'target_events': int(
                    config['events_per_actor_round'],
                ),
                'replay_atol': float(
                    config['logprob_replay_atol'],
                ),
                'amp_enabled': bool(config['amp']),
                'amp_dtype_name': str(config['amp_dtype']),
                'initial_round_id': counters.next_round_id,
                'initial_actor_state': actor_states.get(actor_id),
            },
            name=f'event-v2-large-sync-actor-{actor_id}',
        )
        process.start()
        processes[actor_id] = process

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'parameter_inventory.json').write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    metrics_path = output_dir / 'metrics.jsonl'
    completed_scene_ids = {
        scene_id
        for actor_id, state in actor_states.items()
        for scene_id in state['pool'].get('completed_scene_ids', ())
    }
    reward_errors = {
        int(scene_id): float(error)
        for state in actor_states.values()
        for scene_id, error in state['pool'].get(
            'reward_reconstruction_errors',
            (),
        )
    }
    replay_max_error = 0.0
    frozen_changes = 0
    finite = True
    latest_checkpoint: pathlib.Path | None = None
    interrupted = False
    failure: BaseException | None = None
    pending_update_events = 0

    def save_at_barrier(*, final: bool) -> pathlib.Path:
        nonlocal latest_checkpoint
        if pending_update_events:
            raise RuntimeError(
                'cannot checkpoint with unconsumed strict sync events',
            )
        if set(actor_states) != set(assignments):
            raise RuntimeError(
                'cannot checkpoint before every actor has a barrier state',
            )
        payload = build_large_sync_checkpoint_payload(
            model=learner_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config_fingerprint_value=fingerprint,
            normalizer=normalizer,
            counters=counters,
            actor_scene_assignments=assignments,
            actor_states=actor_states,
            trainable_parameter_names=(
                inventory['trainable_parameter_names']
            ),
            frozen_parameter_names=(
                inventory['frozen_parameter_names']
            ),
            bootstrap=bootstrap_audit,
        )
        if final:
            path = output_dir / (
                f'checkpoint_final_update_{counters.updates:06d}_'
                f'round_{counters.next_round_id:06d}.pth'
            )
        else:
            path = output_dir / (
                f'checkpoint_update_{counters.updates:06d}.pth'
            )
        if path.exists():
            loaded = torch.load(
                path,
                map_location='cpu',
                weights_only=False,
            )
            if (
                int(loaded.get('updates', -1)) != counters.updates
                or int(loaded.get('next_round_id', -1))
                != counters.next_round_id
            ):
                raise FileExistsError(
                    f'checkpoint collision at {path}',
                )
        else:
            save_large_sync_checkpoint(path, payload=payload)
        update_latest_checkpoint(
            source=path,
            latest=output_dir / 'checkpoint_latest.pth',
        )
        latest_checkpoint = path
        return path

    def collect_sync_round(
        *,
        round_actor_ids: set[int],
        target_events_per_actor: int,
    ):
        coordinator = StrictSyncRoundCoordinator(
            actor_ids=round_actor_ids,
            initial_round_id=counters.next_round_id,
            initial_policy_version=trainer.policy_version,
        )
        for actor_id in sorted(round_actor_ids):
            command = coordinator.command_for(actor_id)
            command_queues[actor_id].put(SyncRoundCommand(
                round_id=command.round_id,
                policy_version=command.policy_version,
                target_events=target_events_per_actor,
            ))

        done_messages: dict[int, SyncActorDone] = {}
        inferred_done: set[int] = set()
        while (
            len(coordinator.submitted_actor_ids) < len(round_actor_ids)
            or not inferred_done.issubset(done_messages)
        ):
            try:
                message = result_queue.get(timeout=60)
            except queue.Empty:
                dead = {
                    actor_id: process.exitcode
                    for actor_id, process in processes.items()
                    if (
                        actor_id in round_actor_ids
                        and not process.is_alive()
                    )
                }
                if dead:
                    raise RuntimeError(
                        f'large sync actors exited without results: {dead}',
                    )
                print(json.dumps({
                    'stage': 'V2-2-Large',
                    'heartbeat': True,
                    'round': counters.next_round_id,
                    'waiting_for_actors': sorted(
                        round_actor_ids
                        - set(coordinator.submitted_actor_ids)
                    ),
                }, sort_keys=True), flush=True)
                continue
            if isinstance(message, SyncWorkerError):
                raise RuntimeError(
                    f'large sync actor {message.actor_id} failed with '
                    f'{message.error_type}: {message.message}\n'
                    f'{message.traceback}',
                )
            if isinstance(message, SyncActorChunk):
                coordinator.submit(message)
                actor_states[message.actor_id] = message.state
                completed_scene_ids.update(message.completed_scene_ids)
                if _pool_state_is_complete(
                    message.state,
                    assigned_scene_ids=assignments[message.actor_id],
                ):
                    inferred_done.add(message.actor_id)
                continue
            if isinstance(message, SyncActorDone):
                if (
                    message.round_id != counters.next_round_id
                    or message.policy_version != trainer.policy_version
                ):
                    raise ValueError(
                        'large sync done message has stale identifiers',
                    )
                done_messages[message.actor_id] = message
                actor_states[message.actor_id] = message.state
                reward_errors.update(
                    dict(message.reward_reconstruction_errors),
                )
                continue
            raise TypeError(
                f'unknown large sync actor message: {type(message)!r}',
            )

        return (
            coordinator.finalize(
                min_batch_events=int(config['min_update_events']),
            ),
            inferred_done,
        )

    try:
        while active_actor_ids and counters.updates < max_updates:
            active_actors_at_update_start = len(active_actor_ids)
            first_round_id = counters.next_round_id
            targets_per_collection_round: list[int] = []
            update_batch = StrictSyncUpdateAccumulator(
                policy_version=trainer.policy_version,
                min_batch_events=int(config['min_update_events']),
            )
            while active_actor_ids and not update_batch.should_update:
                target_per_actor = update_batch.target_events_per_actor(
                    active_actor_count=len(active_actor_ids),
                    default_target_events=int(
                        config['events_per_actor_round'],
                    ),
                )
                targets_per_collection_round.append(target_per_actor)
                round_actor_ids = set(active_actor_ids)
                pending_update_events = -1
                batch, inferred_done = collect_sync_round(
                    round_actor_ids=round_actor_ids,
                    target_events_per_actor=target_per_actor,
                )
                update_batch.add(batch)
                pending_update_events = update_batch.event_count
                replay_max_error = max(
                    replay_max_error,
                    batch.replay_max_abs_error,
                )
                active_actor_ids -= inferred_done
                counters = LargeSyncCounters(
                    next_round_id=counters.next_round_id + 1,
                    updates=counters.updates,
                    policy_version=trainer.policy_version,
                    processed_physical_seconds=(
                        counters.processed_physical_seconds
                        + int(batch.processed_physical_seconds)
                    ),
                    episodes=len(completed_scene_ids),
                    events=counters.events + batch.event_count,
                )

            row: dict[str, Any] = {
                'round': first_round_id,
                'last_collection_round': counters.next_round_id - 1,
                'collection_rounds': update_batch.collection_rounds,
                'top_up_rounds': max(
                    update_batch.collection_rounds - 1,
                    0,
                ),
                'round_event_counts': update_batch.round_event_counts,
                'behavior_policy_version': update_batch.policy_version,
                'events_in_round': update_batch.event_count,
                'events': counters.events,
                'physical_seconds': (
                    counters.processed_physical_seconds
                ),
                'episodes': counters.episodes,
                'active_actors': active_actors_at_update_start,
                'remaining_active_actors': len(active_actor_ids),
                'target_events_per_actor': (
                    targets_per_collection_round[0]
                ),
                'targets_per_collection_round': (
                    targets_per_collection_round
                ),
                'replay_max_abs_error': (
                    update_batch.replay_max_abs_error
                ),
            }
            update_performed = update_batch.should_update
            if update_performed:
                update = trainer.update(list(update_batch.steps))
                _step_scheduler_without_restart(scheduler)
                policy_store.publish(
                    learner_model,
                    version=trainer.policy_version,
                )
                counters = LargeSyncCounters(
                    next_round_id=counters.next_round_id,
                    updates=counters.updates + 1,
                    policy_version=trainer.policy_version,
                    processed_physical_seconds=(
                        counters.processed_physical_seconds
                    ),
                    episodes=counters.episodes,
                    events=counters.events,
                )
                pending_update_events = 0
                frozen_changes += update.frozen_parameter_changes
                row.update({
                    'update': counters.updates,
                    'policy_version': trainer.policy_version,
                    'total_loss': update.total_loss,
                    'policy_loss': update.policy_loss,
                    'value_loss': update.value_loss,
                    'entropy': update.entropy,
                    'approx_kl': update.approx_kl,
                    'clip_fraction': update.clip_fraction,
                    'gradient_norm': update.gradient_norm,
                    'completed_epochs': update.completed_epochs,
                    'early_stopped': update.early_stopped,
                })
                finite = finite and all(
                    np.isfinite(value)
                    for key, value in row.items()
                    if key in {
                        'total_loss',
                        'policy_loss',
                        'value_loss',
                        'entropy',
                        'approx_kl',
                        'clip_fraction',
                        'gradient_norm',
                    }
                )
            else:
                if active_actor_ids:
                    raise RuntimeError(
                        'large sync top-up stopped before reaching '
                        'the minimum batch size',
                    )
                pending_update_events = 0
                row.update({
                    'update': counters.updates,
                    'policy_version': trainer.policy_version,
                    'partial_final_batch_skipped': True,
                })

            with metrics_path.open('a', encoding='utf-8') as file:
                file.write(json.dumps(row, sort_keys=True) + '\n')
            print(json.dumps(row, sort_keys=True), flush=True)
            if (
                update_performed
                and counters.updates % checkpoint_interval == 0
            ):
                save_at_barrier(final=False)
            if _STOP_REQUESTED:
                interrupted = bool(active_actor_ids)
                save_at_barrier(final=True)
                break
            if not active_actor_ids:
                save_at_barrier(final=True)
                break

        if active_actor_ids and counters.updates >= max_updates:
            interrupted = True
            save_at_barrier(final=True)
    except BaseException as error:
        failure = error
        if actor_states and set(actor_states) == set(assignments):
            try:
                save_at_barrier(final=True)
            except BaseException:
                pass
    finally:
        stop_event.set()
        for actor_id, command_queue_value in command_queues.items():
            if actor_id not in processes:
                continue
            try:
                command_queue_value.put_nowait(SyncRoundCommand(
                    round_id=max(counters.next_round_id, 0),
                    policy_version=max(trainer.policy_version, 0),
                    stop=True,
                ))
            except queue.Full:
                pass
        for process in processes.values():
            process.join(timeout=30)
        for process in processes.values():
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)

    if failure is not None:
        raise failure
    all_finished = (
        len(completed_scene_ids) == len(scene_ids)
        and not active_actor_ids
    )
    reward_error = (
        max(reward_errors.values()) if reward_errors else None
    )
    accepted = bool(
        all_finished
        and finite
        and replay_max_error <= float(config['logprob_replay_atol'])
        and frozen_changes == 0
        and reward_error is not None
        and reward_error <= 1e-6
    )
    summary = {
        'stage': 'V2-2-Large',
        'mode': 'real_large_strict_sync',
        'accepted': accepted,
        'resumable': interrupted and not accepted,
        'seed': seed,
        'scene_ids': list(scene_ids),
        'actor_count': actor_count,
        'active_environments': active_environments,
        'actor_devices': list(actor_devices),
        'learner_device': str(learner_device),
        'amp_enabled': amp_enabled,
        'amp_dtype': config['amp_dtype'] if amp_enabled else None,
        'updates': counters.updates,
        'policy_version': counters.policy_version,
        'next_round_id': counters.next_round_id,
        'events': counters.events,
        'episodes': counters.episodes,
        'physical_seconds': counters.processed_physical_seconds,
        'all_scenes_finished': all_finished,
        'stale_rollout_events': 0,
        'logprob_replay_max_error': replay_max_error,
        'reward_reconstruction_max_error': reward_error,
        'frozen_parameter_changed_count': frozen_changes,
        'finite': finite,
        'checkpoint': (
            None if latest_checkpoint is None else str(latest_checkpoint)
        ),
        'config_fingerprint': fingerprint,
        'schema_fingerprint': transition_schema_fingerprint(),
        'parameter_inventory': inventory,
        'cuda_peak_allocated_bytes': int(
            torch.cuda.max_memory_allocated(learner_device)
        ),
        'cuda_peak_reserved_bytes': int(
            torch.cuda.max_memory_reserved(learner_device)
        ),
        'bootstrap': bootstrap_audit,
    }
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return summary


def main() -> None:
    signal.signal(signal.SIGTERM, _request_stop)
    if hasattr(signal, 'SIGUSR1'):
        signal.signal(signal.SIGUSR1, _request_stop)
    args = parse_args()
    config = _load_config(args.config)
    seed = int(config['seed'] if args.seed is None else args.seed)
    if seed < 0:
        raise ValueError('large sync seed must be non-negative')
    output_dir = (
        pathlib.Path(config['output_dir'])
        if args.output_dir is None
        else args.output_dir
    )
    max_updates = int(
        config['max_updates']
        if args.max_updates is None
        else args.max_updates
    )
    if args.synthetic_preflight:
        summary = run_synthetic_preflight(
            config={**config, 'seed': seed},
            output_dir=output_dir,
            max_updates=max_updates,
        )
    else:
        scene_ids = tuple(config['scene_ids'])
        if args.scene_start is not None or args.scene_end is not None:
            if args.scene_start is None or args.scene_end is None:
                raise ValueError(
                    'scene start and end must be provided together',
                )
            if args.scene_end < args.scene_start:
                raise ValueError('large sync scene range is invalid')
            scene_ids = tuple(
                range(args.scene_start, args.scene_end + 1),
            )
        learner_device = _device_from_argument(
            args.learner_device,
        )
        actor_count = int(
            config['actor_count']
            if args.actors is None
            else args.actors
        )
        active_environments = int(
            config['active_environments']
            if args.active_environments is None
            else args.active_environments
        )
        actor_device_values = tuple(
            config['actor_devices']
            if args.actor_devices is None
            else args.actor_devices
        )
        summary = run_real_training(
            config={**config, 'seed': seed},
            output_dir=output_dir,
            bootstrap_checkpoint=(
                pathlib.Path(config['bootstrap_checkpoint'])
                if args.bootstrap_checkpoint is None
                else args.bootstrap_checkpoint
            ),
            scene_ids=scene_ids,
            seed=seed,
            actor_count=actor_count,
            active_environments=active_environments,
            max_time_step=int(
                config['max_time_step']
                if args.max_time_step is None
                else args.max_time_step
            ),
            max_updates=max_updates,
            checkpoint_interval=int(
                config['checkpoint_interval']
                if args.checkpoint_every_updates is None
                else args.checkpoint_every_updates
            ),
            learner_device=learner_device,
            actor_device_values=actor_device_values,
            resume=args.resume,
        )
    print(json.dumps(summary, sort_keys=True), flush=True)
    if not summary['accepted'] and not summary.get('resumable', False):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
