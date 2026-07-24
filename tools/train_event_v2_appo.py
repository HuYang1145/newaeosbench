#!/usr/bin/env python3
"""运行 Event Joint Transformer V2-3 异步 APPO。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import copy
import json
import pathlib
import queue
import random
import runpy
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from constellation.new_transformers.event_v2.appo import (
    APPOConfig,
    APPODone,
    APPORolloutChunk,
    APPOSnapshot,
    APPOWorkerError,
    AsynchronousPPOLearner,
    SharedPolicyStore,
    collect_appo_actor_chunk,
    filter_policy_lag,
    run_appo_actor_loop,
)
from constellation.new_transformers.event_v2.basilisk_runtime import (
    BasiliskEventRuntime,
    BasiliskSceneBackend,
    load_runtime_statistics,
)
from constellation.new_transformers.event_v2.checkpoint import (
    APPOCounters,
    build_appo_checkpoint,
    config_fingerprint,
    load_appo_checkpoint,
    load_sync_ppo_policy_checkpoint,
    save_checkpoint_atomic,
)
from constellation.new_transformers.event_v2.model import (
    EventJointActorCritic,
)
from constellation.new_transformers.event_v2.ppo import PPOConfig
from constellation.new_transformers.event_v2.rollout import (
    StoredEventStep,
    SynchronousRuntimeSlot,
)
from constellation.new_transformers.event_v2.transition import (
    transition_schema_fingerprint,
)
from tools.train_event_v2_sync_ppo import (
    SyntheticEventRuntime,
    _sample_action,
    _synthetic_observation,
    _tiny_model,
)


def _load_config(path: pathlib.Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f'config not found: {path}')
    config = {
        key: value
        for key, value in runpy.run_path(str(path)).items()
        if not key.startswith('__')
    }
    if config.get('stage') != 'V2-3':
        raise ValueError('APPO config stage must be V2-3')
    if config.get('split') != 'train':
        raise ValueError('APPO may only consume train scenes')
    if config.get('gamma') != 1.0:
        raise ValueError('APPO completion reward requires gamma=1')
    scene_ids = tuple(int(value) for value in config.get('scene_ids', ()))
    if (
        not scene_ids
        or len(scene_ids) > 120
        or len(scene_ids) != len(set(scene_ids))
        or any(scene_id < 0 for scene_id in scene_ids)
    ):
        raise ValueError('APPO scene IDs are invalid')
    config['scene_ids'] = scene_ids
    for name in (
        'actor_chunk_events',
        'learner_batch_events',
        'max_updates',
        'checkpoint_interval',
        'encoder_unfreeze_layers',
        'decoder_unfreeze_layers',
    ):
        if int(config.get(name, 0)) <= 0:
            raise ValueError(f'APPO {name} must be positive')
    if int(config.get('max_policy_lag', -1)) < 0:
        raise ValueError('APPO max policy lag must be non-negative')
    if not 0 < float(config.get('backbone_lr_scale', 0)) <= 1:
        raise ValueError('APPO backbone learning-rate scale is invalid')
    optimizer = config.get('optimizer')
    if not isinstance(optimizer, Mapping) or float(
        optimizer.get('lr', 0),
    ) <= 0:
        raise ValueError('APPO optimizer learning rate is invalid')
    if config.get('amp_dtype') not in {'bfloat16', 'float16'}:
        raise ValueError('APPO AMP dtype is invalid')
    return config


def deterministic_scene_shards(
    scene_ids: Sequence[int],
    *,
    actor_count: int,
) -> tuple[tuple[int, ...], ...]:
    scene_ids = tuple(int(scene_id) for scene_id in scene_ids)
    if not scene_ids or not 1 <= actor_count <= len(scene_ids):
        raise ValueError('APPO actor count is invalid')
    quotient, remainder = divmod(len(scene_ids), actor_count)
    shards: list[tuple[int, ...]] = []
    start = 0
    for actor_id in range(actor_count):
        width = quotient + int(actor_id < remainder)
        shards.append(scene_ids[start:start + width])
        start += width
    return tuple(shards)


def build_appo_models_from_bootstrap(
    *,
    config: Mapping[str, Any],
    checkpoint_path: pathlib.Path,
    learner_device: torch.device,
):
    """加载已通过门槛的 V2-2 policy，并只解冻 learner 尾层。"""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f'V2-2 bootstrap checkpoint not found: {checkpoint_path}',
        )
    actor_template = EventJointActorCritic(**config['model'])
    metadata = load_sync_ppo_policy_checkpoint(
        path=checkpoint_path,
        model=actor_template,
        expected_stages=('V2-2',),
    )
    actor_template.eval()
    learner_model = copy.deepcopy(actor_template)
    learner_model.unfreeze_last_layers(
        encoder_layers=int(config['encoder_unfreeze_layers']),
        decoder_layers=int(config['decoder_unfreeze_layers']),
    )
    learner_model.to(learner_device)
    return actor_template, learner_model, metadata


def _device_from_argument(value: str) -> torch.device:
    if value == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(value)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    return device


def _seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def _optimizer_and_schedule(
    model: EventJointActorCritic,
    config: Mapping[str, Any],
    max_updates: int,
) -> tuple[
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.CosineAnnealingLR,
]:
    optimizer_config = dict(config['optimizer'])
    learning_rate = float(optimizer_config.pop('lr'))
    optimizer = torch.optim.AdamW(
        model.parameter_groups(
            new_module_lr=learning_rate,
            backbone_lr_scale=float(config['backbone_lr_scale']),
        ),
        **optimizer_config,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(max_updates, 1),
        eta_min=learning_rate * 0.1,
    )
    return optimizer, scheduler


def _normalizer_state(statistics) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in statistics._asdict().items()
    }


def _build_actor_slots(
    *,
    scene_ids: Sequence[int],
    statistics,
    max_time_step: int,
    safety_review_seconds: int,
    runtime_states: Sequence[Mapping[str, Any]] | None,
) -> list[SynchronousRuntimeSlot]:
    if runtime_states is not None:
        if len(runtime_states) != len(scene_ids):
            raise ValueError('actor resume states do not match scene shard')
        slots = []
        for state in runtime_states:
            runtime = BasiliskEventRuntime.from_state_dict(
                state['runtime'],
                statistics=statistics,
            )
            slots.append(SynchronousRuntimeSlot(
                environment_index=int(state['environment_index']),
                episode_id=int(state['episode_id']),
                event_index=int(state['event_index']),
                observation=runtime.current_observation,
                runtime=runtime,
                finished=bool(state['finished']),
            ))
        return slots

    slots = []
    for scene_id in scene_ids:
        runtime = BasiliskEventRuntime(
            backend=BasiliskSceneBackend.from_scene_id(
                split='train',
                scene_id=int(scene_id),
                max_time_step=max_time_step,
            ),
            statistics=statistics,
            safety_review_seconds=safety_review_seconds,
        )
        slots.append(SynchronousRuntimeSlot(
            environment_index=int(scene_id),
            episode_id=0,
            observation=runtime.reset(),
            runtime=runtime,
        ))
    return slots


def _actor_worker_entry(
    *,
    actor_id: int,
    scene_ids: tuple[int, ...],
    model_config: Mapping[str, Any],
    policy_store: SharedPolicyStore,
    result_queue,
    stop_event,
    checkpoint_request,
    checkpoint_release,
    device_value: str,
    seed: int,
    max_time_step: int,
    safety_review_seconds: int,
    actor_chunk_events: int,
    replay_atol: float,
    amp_enabled: bool,
    amp_dtype_name: str,
    runtime_states: Sequence[Mapping[str, Any]] | None,
) -> None:
    """spawn 子进程入口；所有异常都带 traceback 返回 learner。"""

    try:
        device = _device_from_argument(device_value)
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        _seed_everything(seed, device)
        statistics = load_runtime_statistics()
        model = EventJointActorCritic(**model_config)
        slots = _build_actor_slots(
            scene_ids=scene_ids,
            statistics=statistics,
            max_time_step=max_time_step,
            safety_review_seconds=safety_review_seconds,
            runtime_states=runtime_states,
        )
        amp_dtype = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
        }[amp_dtype_name]
        run_appo_actor_loop(
            model=model,
            slots=slots,
            actor_id=actor_id,
            scene_ids=scene_ids,
            policy_store=policy_store,
            result_queue=result_queue,
            stop_event=stop_event,
            target_events=actor_chunk_events,
            device=device,
            replay_atol=replay_atol,
            amp_enabled=amp_enabled and device.type == 'cuda',
            amp_dtype=amp_dtype,
            checkpoint_request=checkpoint_request,
            checkpoint_release=checkpoint_release,
        )
        result_queue.close()
        result_queue.join_thread()
    except BaseException as error:
        result_queue.put(APPOWorkerError(
            actor_id=actor_id,
            error_type=type(error).__name__,
            message=str(error) or repr(error),
            traceback=traceback.format_exc(),
        ))
        stop_event.set()


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
        max_kl=(10. if synthetic else float(config['max_kl'])),
        ppo_epochs=(1 if synthetic else int(config['ppo_epochs'])),
        minibatch_events=(
            4 if synthetic else int(config['minibatch_events'])
        ),
        lambda_base=float(config['lambda_base']),
        reference_seconds=float(config['reference_seconds']),
        replay_atol=float(config['logprob_replay_atol']),
    )


def run_synthetic_preflight(
    *,
    config: Mapping[str, Any],
    output_dir: pathlib.Path,
    max_updates: int,
) -> dict[str, Any]:
    """用两个错开版本的合成 actor 验证 APPO 数据和 checkpoint。"""

    if max_updates != 3:
        raise ValueError('synthetic APPO preflight requires exactly 3 updates')
    torch.manual_seed(int(config['seed']))
    base_model = _tiny_model()
    learner_model = copy.deepcopy(base_model)
    learner_model.unfreeze_last_layers(
        encoder_layers=1,
        decoder_layers=1,
    )
    optimizer_config = dict(config['optimizer'])
    optimizer_config['lr'] = 1e-3
    learning_rate = float(optimizer_config.pop('lr'))
    optimizer = torch.optim.AdamW(
        learner_model.parameter_groups(
            new_module_lr=learning_rate,
            backbone_lr_scale=float(config['backbone_lr_scale']),
        ),
        **optimizer_config,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_updates,
        eta_min=learning_rate * 0.1,
    )
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    learner = AsynchronousPPOLearner(
        model=learner_model,
        optimizer=optimizer,
        ppo_config=_ppo_config(config, synthetic=True),
        appo_config=APPOConfig(
            max_policy_lag=int(config['max_policy_lag']),
        ),
        device=torch.device('cpu'),
        scaler=scaler,
    )
    context = torch.multiprocessing.get_context('spawn')
    store = SharedPolicyStore(
        copy.deepcopy(base_model),
        context=context,
        initial_version=0,
    )
    actor_models = [copy.deepcopy(base_model), copy.deepcopy(base_model)]
    actor_versions = [
        store.refresh(model, last_version=-1).version
        for model in actor_models
    ]
    runtimes = [SyntheticEventRuntime(num_events=8) for _ in range(2)]
    slots = [
        [
            SynchronousRuntimeSlot(
                environment_index=actor_id,
                episode_id=0,
                observation=runtime.reset(),
                runtime=runtime,
            ),
        ]
        for actor_id, runtime in enumerate(runtimes)
    ]
    accepted_events = 0
    stale_dropped_events = 0
    replay_max_error = 0.
    frozen_changes = 0
    update_metrics = []

    def collect(actor_id: int):
        nonlocal replay_max_error
        chunk = collect_appo_actor_chunk(
            actor_models[actor_id],
            slots[actor_id],
            actor_id=actor_id,
            scene_ids=(actor_id,),
            target_events=4,
            policy_version=actor_versions[actor_id],
            device=torch.device('cpu'),
            replay_atol=float(config['logprob_replay_atol']),
        )
        replay_max_error = max(replay_max_error, chunk.replay_max_error)
        return chunk

    for actor_id in (0, 1):
        chunk = collect(actor_id)
        metrics = learner.update(chunk.steps)
        update_metrics.append(metrics)
        accepted_events += metrics.accepted_events
        stale_dropped_events += metrics.stale_dropped_events
        frozen_changes += metrics.ppo.frozen_parameter_changes
        scheduler.step()
        store.publish(
            learner_model,
            version=learner.policy_version,
        )

    refreshed = store.refresh(
        actor_models[0],
        last_version=actor_versions[0],
    )
    actor_versions[0] = refreshed.version
    chunk = collect(0)
    metrics = learner.update(chunk.steps)
    update_metrics.append(metrics)
    accepted_events += metrics.accepted_events
    stale_dropped_events += metrics.stale_dropped_events
    frozen_changes += metrics.ppo.frozen_parameter_changes
    scheduler.step()
    store.publish(learner_model, version=learner.policy_version)

    stale_chunk = collect(1)
    stale = filter_policy_lag(
        stale_chunk.steps,
        current_policy_version=learner.policy_version,
        max_policy_lag=int(config['max_policy_lag']),
    )
    stale_dropped_events += stale.stale_dropped
    if stale.accepted:
        raise RuntimeError('synthetic stale APPO chunk was unexpectedly accepted')

    counters = APPOCounters(
        updates=len(update_metrics),
        policy_version=learner.policy_version,
        accepted_events=accepted_events,
        stale_dropped_events=stale_dropped_events,
        processed_physical_seconds=sum(
            runtime.events * 5 for runtime in runtimes
        ),
        episodes=sum(int(slot[0].finished) for slot in slots),
    )
    fingerprint = config_fingerprint(config)
    checkpoint = build_appo_checkpoint(
        model=learner_model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config_fingerprint_value=fingerprint,
        normalizer={'mean': torch.zeros(1), 'std': torch.ones(1)},
        counters=counters,
        actor_scene_shards=((0,), (1,)),
        actor_runtime_states=tuple(
            (runtime.state_dict(),) for runtime in runtimes
        ),
        encoder_layers=1,
        decoder_layers=1,
        backbone_lr_scale=float(config['backbone_lr_scale']),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / 'checkpoint_update_000003.pth'
    save_checkpoint_atomic(checkpoint_path, checkpoint)
    probe = _synthetic_observation(0)
    expected = _sample_action(
        learner_model,
        probe,
        device=torch.device('cpu'),
    )
    restored = load_appo_checkpoint(
        path=checkpoint_path,
        model=learner_model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        expected_config_fingerprint=fingerprint,
        expected_actor_scene_shards=((0,), (1,)),
        expected_encoder_layers=1,
        expected_decoder_layers=1,
        expected_backbone_lr_scale=float(config['backbone_lr_scale']),
    )
    actual = _sample_action(
        learner_model,
        probe,
        device=torch.device('cpu'),
    )
    action_reproduced = bool(
        torch.equal(
            actual.actor.action.terminate,
            expected.actor.action.terminate,
        )
        and torch.equal(
            actual.actor.action.task_indices,
            expected.actor.action.task_indices,
        )
        and torch.equal(
            actual.actor.action.commitment_indices,
            expected.actor.action.commitment_indices,
        )
    )
    finite = all(
        np.isfinite(value)
        for metrics in update_metrics
        for value in (
            metrics.ppo.total_loss,
            metrics.ppo.policy_loss,
            metrics.ppo.value_loss,
            metrics.ppo.entropy,
            metrics.ppo.approx_kl,
            metrics.ppo.gradient_norm,
        )
    )
    reward_error = max(
        abs(runtime.total_reward - runtime.final_quality)
        for runtime in runtimes
    )
    accepted = bool(
        restored.counters == counters
        and counters.updates == max_updates
        and counters.policy_version == max_updates
        and accepted_events == 12
        and stale_dropped_events == 4
        and replay_max_error <= float(config['logprob_replay_atol'])
        and frozen_changes == 0
        and reward_error <= 1e-6
        and action_reproduced
        and finite
    )
    summary = {
        'stage': 'V2-3',
        'mode': 'synthetic_appo',
        'accepted': accepted,
        'updates': counters.updates,
        'policy_version': counters.policy_version,
        'accepted_events': accepted_events,
        'stale_dropped_events': stale_dropped_events,
        'actor_replay_max_error': replay_max_error,
        'frozen_parameter_changed_count': frozen_changes,
        'reward_reconstruction_max_error': reward_error,
        'checkpoint_first_action_reproduced': action_reproduced,
        'finite': finite,
        'schema_fingerprint': transition_schema_fingerprint(),
        'config_fingerprint': fingerprint,
        'checkpoint': str(checkpoint_path),
    }
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n',
    )
    return summary


def run_real_training(
    *,
    config: Mapping[str, Any],
    output_dir: pathlib.Path,
    checkpoint_path: pathlib.Path,
    scene_ids: tuple[int, ...],
    max_time_step: int,
    max_updates: int,
    learner_device: torch.device,
    actor_device_values: tuple[str, ...],
    resume: pathlib.Path | None,
) -> dict[str, Any]:
    """运行多进程 actor 与单 learner 的 V2-3 正式训练。"""

    if not actor_device_values:
        raise ValueError('real APPO requires at least one actor device')
    if len(actor_device_values) > len(scene_ids):
        raise ValueError('APPO actor count exceeds the scene count')
    if max_updates <= 0 or max_time_step <= 0:
        raise ValueError('APPO runtime limits must be positive')
    actor_devices = tuple(
        _device_from_argument(value) for value in actor_device_values
    )
    if any(device.type != learner_device.type for device in actor_devices):
        raise ValueError('APPO learner and actors must use one device type')
    _seed_everything(int(config['seed']), learner_device)
    shards = deterministic_scene_shards(
        scene_ids,
        actor_count=len(actor_devices),
    )
    actor_template, learner_model, bootstrap_metadata = (
        build_appo_models_from_bootstrap(
            config=config,
            checkpoint_path=checkpoint_path,
            learner_device=learner_device,
        )
    )
    optimizer, scheduler = _optimizer_and_schedule(
        learner_model,
        config,
        max_updates,
    )
    scaler = torch.amp.GradScaler(
        learner_device.type,
        enabled=False,
    )
    amp_enabled = bool(
        config['amp'] and learner_device.type == 'cuda'
    )
    amp_dtype = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
    }[str(config['amp_dtype'])]
    learner = AsynchronousPPOLearner(
        model=learner_model,
        optimizer=optimizer,
        ppo_config=_ppo_config(config),
        appo_config=APPOConfig(
            max_policy_lag=int(config['max_policy_lag']),
        ),
        device=learner_device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        scaler=scaler,
    )
    effective_config = dict(config)
    effective_config['scene_ids'] = scene_ids
    effective_config['max_time_step'] = max_time_step
    effective_config['max_updates'] = max_updates
    effective_config['actor_devices'] = actor_device_values
    effective_config['learner_device'] = str(learner_device)
    fingerprint = config_fingerprint(effective_config)
    counters = APPOCounters()
    pending_steps: list[StoredEventStep] = []
    initial_runtime_states: tuple[
        tuple[Mapping[str, Any], ...], ...
    ] | None = None
    if resume is not None:
        restored = load_appo_checkpoint(
            path=resume,
            model=learner_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_config_fingerprint=fingerprint,
            expected_actor_scene_shards=shards,
            expected_encoder_layers=int(
                config['encoder_unfreeze_layers'],
            ),
            expected_decoder_layers=int(
                config['decoder_unfreeze_layers'],
            ),
            expected_backbone_lr_scale=float(
                config['backbone_lr_scale'],
            ),
        )
        counters = restored.counters
        pending_steps = list(restored.pending_steps)
        initial_runtime_states = restored.actor_runtime_states
        learner.policy_version = counters.policy_version

    shared_initial = copy.deepcopy(actor_template)
    shared_initial.load_state_dict(learner_model.state_dict())
    context = torch.multiprocessing.get_context('spawn')
    policy_store = SharedPolicyStore(
        shared_initial,
        context=context,
        initial_version=learner.policy_version,
    )
    result_queue = context.Queue(maxsize=max(8, len(shards) * 4))
    stop_event = context.Event()
    checkpoint_request = context.Value('q', 0)
    checkpoint_release = context.Value('q', 0)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / 'metrics.jsonl'
    statistics = load_runtime_statistics()
    normalizer = _normalizer_state(statistics)
    processes = []
    for actor_id, (shard, device) in enumerate(
        zip(shards, actor_devices, strict=True),
    ):
        process = context.Process(
            target=_actor_worker_entry,
            kwargs={
                'actor_id': actor_id,
                'scene_ids': shard,
                'model_config': dict(config['model']),
                'policy_store': policy_store,
                'result_queue': result_queue,
                'stop_event': stop_event,
                'checkpoint_request': checkpoint_request,
                'checkpoint_release': checkpoint_release,
                'device_value': str(device),
                'seed': int(config['seed']) + actor_id + 1,
                'max_time_step': max_time_step,
                'safety_review_seconds': int(
                    config['safety_review_seconds'],
                ),
                'actor_chunk_events': int(
                    config['actor_chunk_events'],
                ),
                'replay_atol': float(
                    config['logprob_replay_atol'],
                ),
                'amp_enabled': bool(config['amp']),
                'amp_dtype_name': str(config['amp_dtype']),
                'runtime_states': (
                    None
                    if initial_runtime_states is None
                    else initial_runtime_states[actor_id]
                ),
            },
            name=f'event-v2-appo-actor-{actor_id}',
        )
        process.start()
        processes.append(process)

    actor_states: list[
        tuple[Mapping[str, Any], ...] | None
    ] = (
        [None] * len(shards)
        if initial_runtime_states is None
        else list(initial_runtime_states)
    )
    actor_completed = [0] * len(shards)
    done_actors: set[int] = set()
    reward_errors: dict[int, tuple[float, ...]] = {}
    actor_physical_seconds = counters.processed_physical_seconds
    actor_replay_max_error = 0.
    stale_dropped = counters.stale_dropped_events
    frozen_changes = 0
    metric_rows: list[dict[str, Any]] = []
    last_checkpoint_update = counters.updates
    latest_checkpoint: pathlib.Path | None = None

    def current_counters() -> APPOCounters:
        return APPOCounters(
            updates=len(metric_rows) + counters.updates,
            policy_version=learner.policy_version,
            accepted_events=(
                counters.accepted_events
                + sum(
                    int(row['accepted_events'])
                    for row in metric_rows
                )
            ),
            stale_dropped_events=stale_dropped,
            processed_physical_seconds=actor_physical_seconds,
            episodes=sum(actor_completed),
        )

    def handle_message(message: Any) -> APPOSnapshot | None:
        nonlocal actor_physical_seconds
        nonlocal actor_replay_max_error
        if isinstance(message, APPORolloutChunk):
            pending_steps.extend(message.steps)
            actor_physical_seconds += message.physical_seconds
            actor_completed[message.actor_id] = max(
                actor_completed[message.actor_id],
                message.completed_episodes,
            )
            actor_replay_max_error = max(
                actor_replay_max_error,
                message.replay_max_error,
            )
            return None
        if isinstance(message, APPOSnapshot):
            actor_states[message.actor_id] = message.runtime_states
            actor_completed[message.actor_id] = max(
                actor_completed[message.actor_id],
                message.completed_episodes,
            )
            return message
        if isinstance(message, APPODone):
            actor_states[message.actor_id] = message.runtime_states
            actor_completed[message.actor_id] = (
                message.completed_episodes
            )
            reward_errors[message.actor_id] = (
                message.reward_reconstruction_errors
            )
            done_actors.add(message.actor_id)
            return None
        if isinstance(message, APPOWorkerError):
            raise RuntimeError(
                f'APPO actor {message.actor_id} failed with '
                f'{message.error_type}: {message.message}\n'
                f'{message.traceback}',
            )
        raise TypeError(f'unknown APPO worker message: {type(message)!r}')

    def update_ready(*, flush: bool) -> None:
        nonlocal pending_steps
        nonlocal stale_dropped
        nonlocal frozen_changes
        while pending_steps and (
            flush
            or len(pending_steps) >= int(
                config['learner_batch_events'],
            )
        ):
            filtered = filter_policy_lag(
                pending_steps,
                current_policy_version=learner.policy_version,
                max_policy_lag=int(config['max_policy_lag']),
            )
            stale_dropped += filtered.stale_dropped
            pending_steps = list(filtered.accepted)
            if not pending_steps:
                return
            if (
                not flush
                and len(pending_steps)
                < int(config['learner_batch_events'])
            ):
                return
            width = (
                len(pending_steps)
                if flush
                else int(config['learner_batch_events'])
            )
            batch = pending_steps[:width]
            pending_steps = pending_steps[width:]
            metrics = learner.update(batch)
            scheduler.step()
            policy_store.publish(
                learner_model,
                version=learner.policy_version,
            )
            row = {
                'update': current_counters().updates + 1,
                'policy_version': learner.policy_version,
                'input_events': metrics.input_events,
                'accepted_events': metrics.accepted_events,
                'stale_dropped_events': stale_dropped,
                'minimum_behavior_version': (
                    metrics.minimum_behavior_version
                ),
                'maximum_behavior_version': (
                    metrics.maximum_behavior_version
                ),
                'total_loss': metrics.ppo.total_loss,
                'policy_loss': metrics.ppo.policy_loss,
                'value_loss': metrics.ppo.value_loss,
                'entropy': metrics.ppo.entropy,
                'approx_kl': metrics.ppo.approx_kl,
                'clip_fraction': metrics.ppo.clip_fraction,
                'gradient_norm': metrics.ppo.gradient_norm,
                'completed_epochs': metrics.ppo.completed_epochs,
                'early_stopped': metrics.ppo.early_stopped,
            }
            frozen_changes += metrics.ppo.frozen_parameter_changes
            metric_rows.append(row)
            with metrics_path.open('a', encoding='utf-8') as file:
                file.write(json.dumps(row, sort_keys=True) + '\n')
            print(json.dumps(row, sort_keys=True), flush=True)
            if current_counters().updates >= max_updates:
                stop_event.set()
                return

    def save_checkpoint(path: pathlib.Path) -> None:
        nonlocal latest_checkpoint
        if any(state is None for state in actor_states):
            raise RuntimeError(
                'cannot checkpoint APPO before every actor has a snapshot',
            )
        checkpoint = build_appo_checkpoint(
            model=learner_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config_fingerprint_value=fingerprint,
            normalizer=normalizer,
            counters=current_counters(),
            actor_scene_shards=shards,
            actor_runtime_states=tuple(
                state for state in actor_states if state is not None
            ),
            pending_steps=tuple(pending_steps),
            encoder_layers=int(config['encoder_unfreeze_layers']),
            decoder_layers=int(config['decoder_unfreeze_layers']),
            backbone_lr_scale=float(config['backbone_lr_scale']),
        )
        save_checkpoint_atomic(path, checkpoint)
        latest_checkpoint = path

    def coordinated_checkpoint() -> None:
        nonlocal last_checkpoint_update
        generation = int(checkpoint_request.value) + 1
        checkpoint_request.value = generation
        waiting = set(range(len(shards))) - done_actors
        while waiting:
            try:
                message = result_queue.get(timeout=30)
            except queue.Empty:
                dead = [
                    process.name
                    for process in processes
                    if not process.is_alive()
                    and process.exitcode not in (None, 0)
                ]
                if dead:
                    raise RuntimeError(
                        f'APPO actors exited during checkpoint: {dead}',
                    )
                continue
            snapshot = handle_message(message)
            if (
                snapshot is not None
                and snapshot.generation == generation
            ):
                waiting.discard(snapshot.actor_id)
            waiting -= done_actors
        save_checkpoint(output_dir / 'checkpoint_latest.pth')
        checkpoint_release.value = generation
        last_checkpoint_update = current_counters().updates

    error: BaseException | None = None
    try:
        while (
            len(done_actors) < len(shards)
            and current_counters().updates < max_updates
        ):
            try:
                message = result_queue.get(timeout=30)
            except queue.Empty:
                dead = [
                    process.name
                    for process in processes
                    if not process.is_alive()
                    and process.exitcode not in (None, 0)
                ]
                if dead:
                    raise RuntimeError(
                        f'APPO actors exited unexpectedly: {dead}',
                    )
                continue
            handle_message(message)
            update_ready(flush=False)
            if stop_event.is_set():
                break
            if (
                current_counters().updates > last_checkpoint_update
                and current_counters().updates
                % int(config['checkpoint_interval']) == 0
            ):
                coordinated_checkpoint()

        if len(done_actors) == len(shards):
            update_ready(flush=True)
        else:
            stop_event.set()
        if len(done_actors) == len(shards):
            final_path = output_dir / (
                f'checkpoint_update_{current_counters().updates:06d}.pth'
            )
            save_checkpoint(final_path)
    except BaseException as caught:
        error = caught
        stop_event.set()
    finally:
        stop_event.set()
        checkpoint_release.value = max(
            checkpoint_release.value,
            checkpoint_request.value,
        )
        for process in processes:
            process.join(timeout=20)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
        result_queue.close()
        result_queue.join_thread()
    if error is not None:
        raise error
    if latest_checkpoint is None:
        raise RuntimeError('APPO training produced no checkpoint')

    probe = _synthetic_observation(0)
    expected_action = _sample_action(
        learner_model,
        probe,
        device=learner_device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )
    restored = load_appo_checkpoint(
        path=latest_checkpoint,
        model=learner_model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        expected_config_fingerprint=fingerprint,
        expected_actor_scene_shards=shards,
        expected_encoder_layers=int(config['encoder_unfreeze_layers']),
        expected_decoder_layers=int(config['decoder_unfreeze_layers']),
        expected_backbone_lr_scale=float(config['backbone_lr_scale']),
    )
    actual_action = _sample_action(
        learner_model,
        probe,
        device=learner_device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )
    action_reproduced = bool(
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
    flattened_reward_errors = tuple(
        error_value
        for actor_id in sorted(reward_errors)
        for error_value in reward_errors[actor_id]
    )
    finite = all(
        np.isfinite(value)
        for row in metric_rows
        for name, value in row.items()
        if name not in {'early_stopped'}
        and isinstance(value, (int, float))
    )
    all_finished = len(done_actors) == len(shards)
    final_counters = restored.counters
    accepted = bool(
        all_finished
        and final_counters.updates > 0
        and final_counters.episodes == len(scene_ids)
        and len(flattened_reward_errors) == len(scene_ids)
        and max(flattened_reward_errors, default=float('inf')) <= 1e-6
        and actor_replay_max_error <= float(
            config['logprob_replay_atol'],
        )
        and frozen_changes == 0
        and action_reproduced
        and finite
    )
    summary = {
        'stage': 'V2-3',
        'mode': 'real_appo',
        'accepted': accepted,
        'scene_ids': list(scene_ids),
        'actor_scene_shards': [list(shard) for shard in shards],
        'actor_devices': list(actor_device_values),
        'learner_device': str(learner_device),
        'amp_enabled': amp_enabled,
        'amp_dtype': config['amp_dtype'] if amp_enabled else None,
        'updates': final_counters.updates,
        'policy_version': final_counters.policy_version,
        'accepted_events': final_counters.accepted_events,
        'stale_dropped_events': final_counters.stale_dropped_events,
        'physical_seconds': final_counters.processed_physical_seconds,
        'episodes': final_counters.episodes,
        'reward_reconstruction_max_error': max(
            flattened_reward_errors,
            default=None,
        ),
        'actor_replay_max_error': actor_replay_max_error,
        'frozen_parameter_changed_count': frozen_changes,
        'checkpoint_first_action_reproduced': action_reproduced,
        'finite': finite,
        'all_scenes_finished': all_finished,
        'schema_fingerprint': transition_schema_fingerprint(),
        'config_fingerprint': fingerprint,
        'checkpoint': str(latest_checkpoint),
        'bootstrap': {
            'stage': bootstrap_metadata.stage,
            'updates': bootstrap_metadata.updates,
            'policy_version': bootstrap_metadata.policy_version,
            'scene_ids': list(bootstrap_metadata.scene_ids),
            'checkpoint': str(checkpoint_path),
        },
    }
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n',
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=pathlib.Path, required=True)
    parser.add_argument('--bootstrap-checkpoint', type=pathlib.Path)
    parser.add_argument('--output', type=pathlib.Path)
    parser.add_argument('--resume', type=pathlib.Path)
    parser.add_argument('--scene-ids', type=int, nargs='+')
    parser.add_argument('--max-time-step', type=int)
    parser.add_argument('--max-updates', type=int)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--actor-devices', nargs='+')
    parser.add_argument('--synthetic-preflight', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    output_dir = args.output or pathlib.Path(config['output_dir'])
    if args.synthetic_preflight:
        summary = run_synthetic_preflight(
            config=config,
            output_dir=output_dir,
            max_updates=3,
        )
    else:
        learner_device = _device_from_argument(args.device)
        if not args.actor_devices:
            raise ValueError(
                'real APPO requires explicit --actor-devices',
            )
        scene_ids = tuple(args.scene_ids or config['scene_ids'])
        checkpoint_path = (
            args.bootstrap_checkpoint
            or pathlib.Path(config['bootstrap_checkpoint'])
        )
        summary = run_real_training(
            config=config,
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            scene_ids=scene_ids,
            max_time_step=(
                args.max_time_step
                if args.max_time_step is not None
                else int(config['max_time_step'])
            ),
            max_updates=(
                args.max_updates
                if args.max_updates is not None
                else int(config['max_updates'])
            ),
            learner_device=learner_device,
            actor_device_values=tuple(args.actor_devices),
            resume=args.resume,
        )
    print(json.dumps(summary, sort_keys=True), flush=True)
    if not summary['accepted']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
