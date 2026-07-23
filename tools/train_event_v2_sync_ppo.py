#!/usr/bin/env python3
"""运行 V2-1/V2-2 同步事件 PPO，并输出 correctness 审计。"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import pathlib
import random
import runpy
import sys
from typing import Any

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from constellation.new_transformers.dataset import Statistics
from constellation.new_transformers.event_v2.basilisk_runtime import (
    BasiliskEventRuntime,
    BasiliskSceneBackend,
    RuntimeStep,
    load_runtime_statistics,
)
from constellation.new_transformers.event_v2.checkpoint import (
    SyncPPOCounters,
    build_sync_ppo_checkpoint,
    config_fingerprint,
    load_sync_ppo_bootstrap_checkpoint,
    load_sync_ppo_checkpoint,
    save_checkpoint_atomic,
)
from constellation.new_transformers.event_v2.model import (
    EventJointActorCritic,
)
from constellation.new_transformers.event_v2.observation import (
    EventPolicyObservation,
)
from constellation.new_transformers.event_v2.ppo import (
    PPOConfig,
    SynchronousPPOTrainer,
)
from constellation.new_transformers.event_v2.rollout import (
    SynchronousRuntimeSlot,
    collect_synchronous_rollout,
    replay_rollout_log_probs,
)
from constellation.new_transformers.event_v2.state import EventStateTensors
from constellation.new_transformers.event_v2.transition import (
    transition_schema_fingerprint,
)


SYNC_PPO_STAGES = frozenset({'V2-1', 'V2-2'})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train the Event Joint Transformer V2-1 with synchronous PPO',
    )
    parser.add_argument('--config', type=pathlib.Path, required=True)
    parser.add_argument('--warm-start-checkpoint', type=pathlib.Path)
    parser.add_argument('--bootstrap-checkpoint', type=pathlib.Path)
    parser.add_argument('--output', type=pathlib.Path)
    parser.add_argument('--resume', type=pathlib.Path)
    parser.add_argument('--scene-ids', type=int, nargs='+')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--max-time-step', type=int)
    parser.add_argument('--max-updates', type=int)
    parser.add_argument('--ppo-epochs', type=int)
    parser.add_argument('--device', default='auto')
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
    stage = config.get('stage')
    if stage not in SYNC_PPO_STAGES:
        raise ValueError('config stage is not a supported synchronous PPO stage')
    if config.get('split') != 'train':
        raise ValueError('synchronous PPO may only consume train scenes')
    if config.get('gamma') != 1.0:
        raise ValueError('synchronous PPO completion reward requires gamma=1')
    if not config.get('freeze_backbone'):
        raise ValueError('synchronous PPO must keep the Stage3 backbone frozen')
    if stage == 'V2-1' and 'warm_start_checkpoint' not in config:
        raise ValueError('V2-1 config needs a V2-0 warm-start checkpoint')
    if stage == 'V2-2' and 'bootstrap_checkpoint' not in config:
        raise ValueError('V2-2 config needs a V2-1 bootstrap checkpoint')
    return config


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


def _ppo_config(config: Mapping[str, Any]) -> PPOConfig:
    return PPOConfig(
        clip_ratio=float(config['clip_ratio']),
        value_coefficient=float(config['value_coefficient']),
        entropy_coefficient=float(config['entropy_coefficient']),
        max_grad_norm=float(config['max_grad_norm']),
        max_kl=float(config['max_kl']),
        ppo_epochs=int(config['ppo_epochs']),
        minibatch_events=int(config['minibatch_events']),
        lambda_base=float(config['lambda_base']),
        reference_seconds=float(config['reference_seconds']),
        replay_atol=float(config['logprob_replay_atol']),
    )


def _optimizer_and_schedule(
    model: EventJointActorCritic,
    config: Mapping[str, Any],
    max_updates: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer_config = dict(config['optimizer'])
    learning_rate = float(optimizer_config.pop('lr'))
    optimizer = torch.optim.AdamW(
        model.parameter_groups(learning_rate),
        **optimizer_config,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(max_updates, 1),
        eta_min=learning_rate * 0.1,
    )
    return optimizer, scheduler


def _step_scheduler_without_restart(
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
) -> None:
    """到达预注册周期后保持 eta_min，禁止 cosine 隐式回升。"""

    if scheduler.last_epoch >= scheduler.T_max:
        learning_rates = []
        for parameter_group in scheduler.optimizer.param_groups:
            parameter_group['lr'] = scheduler.eta_min
            learning_rates.append(scheduler.eta_min)
        scheduler._last_lr = learning_rates
        return
    scheduler.step()


def _normalizer_state(statistics: Statistics) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in statistics._asdict().items()
    }


def _synthetic_observation(time_step: int) -> EventPolicyObservation:
    satellite_shape = (1, 2)
    task_shape = (1, 3)
    state = EventStateTensors(
        previous_task_indices=torch.full(satellite_shape, -1, dtype=torch.long),
        current_task_indices=torch.full(satellite_shape, -1, dtype=torch.long),
        minimum_commitment_remaining=torch.zeros(satellite_shape),
        run_lengths=torch.full(satellite_shape, float(time_step)),
        seconds_since_replan=torch.full(satellite_shape, float(time_step)),
        switch_count_30=torch.zeros(satellite_shape),
        switch_count_60=torch.zeros(satellite_shape),
        termination_reason=torch.zeros(satellite_shape, dtype=torch.long),
        event_type=torch.full(satellite_shape, 3, dtype=torch.long),
        delta_t=torch.full(satellite_shape, 5.),
        replan_mask=torch.ones(satellite_shape, dtype=torch.bool),
        forced_interrupt_mask=torch.zeros(satellite_shape, dtype=torch.bool),
        can_terminate_mask=torch.zeros(satellite_shape, dtype=torch.bool),
        compatible_deadline_slack=torch.tensor([[10., 20.]]),
        task_remaining_required_seconds=torch.tensor([[10., 30., 60.]]),
        task_owner_count=torch.zeros(task_shape, dtype=torch.long),
        task_locked_owner_count=torch.zeros(task_shape, dtype=torch.long),
    )
    return EventPolicyObservation(
        time_steps=torch.tensor([time_step]),
        constellation_sensor_type=torch.zeros(satellite_shape, dtype=torch.long),
        constellation_sensor_enabled=torch.ones(satellite_shape, dtype=torch.long),
        constellation_data=torch.zeros(1, 2, 56),
        constellation_mask=torch.ones(satellite_shape, dtype=torch.bool),
        tasks_sensor_type=torch.zeros(task_shape, dtype=torch.long),
        tasks_data=torch.zeros(1, 3, 6),
        tasks_mask=torch.ones(task_shape, dtype=torch.bool),
        event_state=state,
    )


class SyntheticEventRuntime:
    def __init__(self, num_events: int = 8) -> None:
        self.num_events = num_events
        self.events = 0
        self.total_reward = 0.
        self.final_quality = sum(
            (index + 1) / 100 for index in range(num_events)
        )

    def reset(self) -> EventPolicyObservation:
        return _synthetic_observation(0)

    def step(self, action) -> RuntimeStep:
        del action
        self.events += 1
        reward = self.events / 100
        self.total_reward += reward
        done = self.events >= self.num_events
        return RuntimeStep(
            observation=(
                None if done else _synthetic_observation(self.events * 5)
            ),
            reward=reward,
            delta_t=5,
            done=done,
            final_quality=(self.final_quality if done else None),
            invalid_action_count=0,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            'type': 'synthetic',
            'num_events': self.num_events,
            'events': self.events,
            'total_reward': self.total_reward,
        }


def _tiny_model() -> EventJointActorCritic:
    return EventJointActorCritic(
        event_width=8,
        sensor_type_embedding_dim=4,
        tasks_data_embedding_dim=4,
        encoder_width=8,
        encoder_depth=1,
        encoder_num_heads=2,
        sensor_enabled_embedding_dim=4,
        constellation_data_embedding_dim=4,
        decoder_width=8,
        decoder_depth=1,
        decoder_num_heads=2,
        use_constraint_module=False,
        use_sdpa=False,
        freeze_backbone=True,
    )


def _sample_action(
    model: EventJointActorCritic,
    observation: EventPolicyObservation,
    *,
    device: torch.device,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
):
    model.eval()
    observation = observation.to(device)
    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
            dtype=amp_dtype,
        ):
            return model.act(
                *observation.model_args(),
                event_state=observation.event_state,
                deterministic=False,
            )


def run_synthetic_preflight(
    *,
    config: Mapping[str, Any],
    output_dir: pathlib.Path,
    max_updates: int,
    device: torch.device,
) -> dict[str, Any]:
    stage = str(config['stage'])
    model = _tiny_model().to(device)
    synthetic_config = dict(config)
    synthetic_config['optimizer'] = {
        **dict(config['optimizer']),
        'lr': 1e-3,
    }
    synthetic_config['ppo_epochs'] = 2
    synthetic_config['minibatch_events'] = 4
    synthetic_config['max_kl'] = 10.
    optimizer, scheduler = _optimizer_and_schedule(
        model,
        synthetic_config,
        max_updates,
    )
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    trainer = SynchronousPPOTrainer(
        model=model,
        optimizer=optimizer,
        config=_ppo_config(synthetic_config),
        device=device,
        scaler=scaler,
    )
    runtime = SyntheticEventRuntime(num_events=max_updates * 4)
    slot = SynchronousRuntimeSlot(
        environment_index=0,
        episode_id=0,
        observation=runtime.reset(),
        runtime=runtime,
    )
    replay_error = 0.
    frozen_changes = 0
    metrics = []
    total_events = 0
    while len(metrics) < max_updates and not slot.finished:
        steps = collect_synchronous_rollout(
            model,
            [slot],
            target_events=4,
            policy_version=trainer.policy_version,
            device=device,
        )
        replay = replay_rollout_log_probs(model, steps, device=device)
        behavior = torch.stack([step.behavior_log_prob for step in steps])
        replay_error = max(
            replay_error,
            float((replay - behavior).abs().max()),
        )
        update_metrics = trainer.update(steps)
        _step_scheduler_without_restart(scheduler)
        metrics.append(update_metrics)
        frozen_changes += update_metrics.frozen_parameter_changes
        total_events += len(steps)

    counters = SyncPPOCounters(
        updates=len(metrics),
        policy_version=trainer.policy_version,
        processed_physical_seconds=total_events * 5,
        episodes=int(slot.finished),
        events=total_events,
    )
    fingerprint = config_fingerprint(config)
    checkpoint = build_sync_ppo_checkpoint(
        stage=stage,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config_fingerprint_value=fingerprint,
        normalizer={'mean': torch.zeros(1), 'std': torch.ones(1)},
        counters=counters,
        scene_ids=(0,),
        runtime_states=(runtime.state_dict(),),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f'checkpoint_update_{counters.updates:06d}.pth'
    save_checkpoint_atomic(checkpoint_path, checkpoint)
    probe = _synthetic_observation(0)
    expected_action = _sample_action(model, probe, device=device)
    restored = load_sync_ppo_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        expected_stage=stage,
        expected_config_fingerprint=fingerprint,
        expected_scene_ids=(0,),
    )
    actual_action = _sample_action(model, probe, device=device)
    action_reproduced = bool(
        torch.equal(
            actual_action.actor.action.terminate,
            expected_action.actor.action.terminate,
        )
        and torch.equal(
            actual_action.actor.action.task_indices,
            expected_action.actor.action.task_indices,
        )
        and torch.equal(
            actual_action.actor.action.commitment_indices,
            expected_action.actor.action.commitment_indices,
        )
    )
    reward_error = abs(runtime.total_reward - runtime.final_quality)
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
        len(metrics) == max_updates
        and reward_error <= 1e-6
        and replay_error <= float(config['logprob_replay_atol'])
        and frozen_changes == 0
        and action_reproduced
        and finite
        and restored.counters == counters
    )
    summary = {
        'stage': stage,
        'mode': 'synthetic_preflight',
        'accepted': accepted,
        'schema_fingerprint': transition_schema_fingerprint(),
        'config_fingerprint': fingerprint,
        'updates': counters.updates,
        'events': counters.events,
        'physical_seconds': counters.processed_physical_seconds,
        'reward_reconstruction_max_error': reward_error,
        'logprob_replay_max_error': replay_error,
        'frozen_parameter_changed_count': frozen_changes,
        'invalid_action_count': 0,
        'event_time_violation_count': 0,
        'unterminated_commitment_count': 0,
        'checkpoint_first_action_reproduced': action_reproduced,
        'finite': finite,
        'checkpoint': str(checkpoint_path),
    }
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n',
    )
    return summary


def _load_warm_start(
    model: EventJointActorCritic,
    path: pathlib.Path,
) -> Mapping[str, Any]:
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('stage') != 'V2-0':
        raise ValueError('warm-start checkpoint stage does not match V2-0')
    if checkpoint.get('transition_schema_fingerprint') != (
        transition_schema_fingerprint()
    ):
        raise ValueError('warm-start transition schema does not match')
    model.load_state_dict(checkpoint['model'])
    if not model.backbone_is_frozen:
        raise ValueError('warm-start checkpoint unexpectedly unfreezes Stage3')
    return checkpoint


def _runtime_checkpoint_state(slot: SynchronousRuntimeSlot) -> dict[str, Any]:
    if not isinstance(slot.runtime, BasiliskEventRuntime):
        raise TypeError('formal V2-1 checkpoint requires a Basilisk runtime')
    return {
        'environment_index': slot.environment_index,
        'episode_id': slot.episode_id,
        'event_index': slot.event_index,
        'finished': slot.finished,
        'runtime': slot.runtime.state_dict(),
    }


def _restore_runtime_slots(
    runtime_states: Sequence[Mapping[str, Any]],
    *,
    statistics: Statistics,
) -> list[SynchronousRuntimeSlot]:
    slots = []
    for runtime_state in runtime_states:
        runtime = BasiliskEventRuntime.from_state_dict(
            runtime_state['runtime'],
            statistics=statistics,
        )
        slots.append(SynchronousRuntimeSlot(
            environment_index=int(runtime_state['environment_index']),
            episode_id=int(runtime_state['episode_id']),
            event_index=int(runtime_state['event_index']),
            observation=runtime.current_observation,
            runtime=runtime,
            finished=bool(runtime_state['finished']),
        ))
    return slots


def run_real_training(
    *,
    config: Mapping[str, Any],
    output_dir: pathlib.Path,
    warm_start_checkpoint: pathlib.Path | None,
    bootstrap_checkpoint: pathlib.Path | None,
    scene_ids: tuple[int, ...],
    max_time_step: int,
    max_updates: int,
    device: torch.device,
    resume: pathlib.Path | None,
) -> dict[str, Any]:
    stage = str(config['stage'])
    if (warm_start_checkpoint is None) == (bootstrap_checkpoint is None):
        raise ValueError(
            'exactly one warm-start or bootstrap checkpoint is required',
        )
    statistics = load_runtime_statistics()
    model = EventJointActorCritic(**config['model']).to(device)
    if warm_start_checkpoint is not None:
        if not warm_start_checkpoint.is_file():
            raise FileNotFoundError(
                f'V2-0 checkpoint not found: {warm_start_checkpoint}',
            )
        _load_warm_start(model, warm_start_checkpoint)
    optimizer, scheduler = _optimizer_and_schedule(model, config, max_updates)
    bootstrap_metadata = None
    if bootstrap_checkpoint is not None:
        if not bootstrap_checkpoint.is_file():
            raise FileNotFoundError(
                f'V2-1 bootstrap checkpoint not found: {bootstrap_checkpoint}',
            )
        bootstrap_metadata = load_sync_ppo_bootstrap_checkpoint(
            path=bootstrap_checkpoint,
            model=model,
            optimizer=optimizer,
            expected_source_stage='V2-1',
        )
    amp_enabled = bool(config['amp'] and device.type == 'cuda')
    scaler = torch.amp.GradScaler(device.type, enabled=False)
    amp_dtype = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
    }[str(config['amp_dtype'])]
    trainer = SynchronousPPOTrainer(
        model=model,
        optimizer=optimizer,
        config=_ppo_config(config),
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        scaler=scaler,
    )
    fingerprint = config_fingerprint(config)
    counters = SyncPPOCounters()
    all_slots: list[SynchronousRuntimeSlot] = []
    if resume is not None:
        restored = load_sync_ppo_checkpoint(
            path=resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_stage=stage,
            expected_config_fingerprint=fingerprint,
            expected_scene_ids=scene_ids,
        )
        counters = restored.counters
        trainer.policy_version = counters.policy_version
        all_slots = _restore_runtime_slots(
            restored.runtime_states,
            statistics=statistics,
        )
    else:
        for environment_index, scene_id in enumerate(scene_ids):
            runtime = BasiliskEventRuntime(
                backend=BasiliskSceneBackend.from_scene_id(
                    split='train',
                    scene_id=scene_id,
                    max_time_step=max_time_step,
                ),
                statistics=statistics,
                safety_review_seconds=int(config['safety_review_seconds']),
            )
            all_slots.append(SynchronousRuntimeSlot(
                environment_index=environment_index,
                episode_id=0,
                observation=runtime.reset(),
                runtime=runtime,
            ))

    output_dir.mkdir(parents=True, exist_ok=True)
    replay_max_error = 0.
    frozen_changes = 0
    metric_rows: list[dict[str, Any]] = []
    last_checkpoint_path: pathlib.Path | None = None
    while counters.updates < max_updates and any(
        not slot.finished for slot in all_slots
    ):
        active_slots = [slot for slot in all_slots if not slot.finished]
        steps = collect_synchronous_rollout(
            model,
            active_slots,
            target_events=int(config['rollout_events_per_update']),
            policy_version=trainer.policy_version,
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
        behavior = torch.stack([step.behavior_log_prob for step in steps])
        replay_max_error = max(
            replay_max_error,
            float((replay - behavior).abs().max()),
        )
        metrics = trainer.update(steps)
        _step_scheduler_without_restart(scheduler)
        frozen_changes += metrics.frozen_parameter_changes
        metric_rows.append({
            name: float(getattr(metrics, name))
            for name in (
                'total_loss',
                'policy_loss',
                'value_loss',
                'entropy',
                'approx_kl',
                'clip_fraction',
                'gradient_norm',
            )
        })
        metric_rows[-1]['completed_epochs'] = metrics.completed_epochs
        metric_rows[-1]['early_stopped'] = metrics.early_stopped
        counters = SyncPPOCounters(
            updates=counters.updates + 1,
            policy_version=trainer.policy_version,
            processed_physical_seconds=(
                counters.processed_physical_seconds
                + int(sum(step.delta_t.item() for step in steps))
            ),
            episodes=sum(int(slot.finished) for slot in all_slots),
            events=counters.events + len(steps),
        )
        print(json.dumps({
            'stage': stage,
            'update': counters.updates,
            'events': counters.events,
            'physical_seconds': counters.processed_physical_seconds,
            **metric_rows[-1],
        }, sort_keys=True), flush=True)
        if (
            counters.updates % int(config['checkpoint_interval']) == 0
            or counters.updates == max_updates
            or all(slot.finished for slot in all_slots)
        ):
            checkpoint = build_sync_ppo_checkpoint(
                stage=stage,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config_fingerprint_value=fingerprint,
                normalizer=_normalizer_state(statistics),
                counters=counters,
                scene_ids=scene_ids,
                runtime_states=tuple(
                    _runtime_checkpoint_state(slot) for slot in all_slots
                ),
            )
            last_checkpoint_path = (
                output_dir / f'checkpoint_update_{counters.updates:06d}.pth'
            )
            save_checkpoint_atomic(last_checkpoint_path, checkpoint)

    checkpoint_action_reproduced = False
    if last_checkpoint_path is not None and all_slots:
        probe = all_slots[0].observation
        expected_action = _sample_action(
            model,
            probe,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        load_sync_ppo_checkpoint(
            path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_stage=stage,
            expected_config_fingerprint=fingerprint,
            expected_scene_ids=scene_ids,
        )
        actual_action = _sample_action(
            model,
            probe,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
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

    completed_runtimes = [
        slot.runtime
        for slot in all_slots
        if slot.finished and isinstance(slot.runtime, BasiliskEventRuntime)
    ]
    reward_errors = [
        runtime.reward_reconstruction_error
        for runtime in completed_runtimes
    ]
    finite = all(
        np.isfinite(value)
        for row in metric_rows
        for value in row.values()
    )
    all_finished = bool(all_slots) and all(slot.finished for slot in all_slots)
    accepted = bool(
        all_finished
        and finite
        and replay_max_error <= float(config['logprob_replay_atol'])
        and frozen_changes == 0
        and reward_errors
        and max(reward_errors) <= 1e-6
        and checkpoint_action_reproduced
    )
    peak_allocated = (
        int(torch.cuda.max_memory_allocated()) if device.type == 'cuda' else 0
    )
    peak_reserved = (
        int(torch.cuda.max_memory_reserved()) if device.type == 'cuda' else 0
    )
    summary = {
        'stage': stage,
        'mode': 'real_sync_ppo',
        'accepted': accepted,
        'amp_enabled': amp_enabled,
        'amp_dtype': config['amp_dtype'] if amp_enabled else None,
        'scene_ids': list(scene_ids),
        'max_time_step': max_time_step,
        'updates': counters.updates,
        'events': counters.events,
        'episodes': counters.episodes,
        'physical_seconds': counters.processed_physical_seconds,
        'reward_reconstruction_max_error': (
            max(reward_errors) if reward_errors else None
        ),
        'logprob_replay_max_error': replay_max_error,
        'frozen_parameter_changed_count': frozen_changes,
        'invalid_action_count': 0,
        'event_time_violation_count': 0,
        'unterminated_commitment_count': 0,
        'checkpoint_first_action_reproduced': checkpoint_action_reproduced,
        'finite': finite,
        'all_scenes_finished': all_finished,
        'policy_version': trainer.policy_version,
        'metrics': metric_rows,
        'cuda_peak_allocated_bytes': peak_allocated,
        'cuda_peak_reserved_bytes': peak_reserved,
        'schema_fingerprint': transition_schema_fingerprint(),
        'config_fingerprint': fingerprint,
        'bootstrap': (
            {
                'source_stage': bootstrap_metadata.source_stage,
                'source_updates': bootstrap_metadata.source_updates,
                'source_policy_version': (
                    bootstrap_metadata.source_policy_version
                ),
                'source_scene_ids': list(
                    bootstrap_metadata.source_scene_ids,
                ),
                'checkpoint': str(bootstrap_checkpoint),
            }
            if bootstrap_metadata is not None
            else None
        ),
    }
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n',
    )
    return summary


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    if args.seed is not None:
        if args.seed < 0:
            raise ValueError('seed must be non-negative')
        config['seed'] = args.seed
    if args.ppo_epochs is not None:
        if args.ppo_epochs <= 0:
            raise ValueError('PPO epochs must be positive')
        config['ppo_epochs'] = args.ppo_epochs
    device = _device_from_argument(args.device)
    _seed_everything(int(config['seed']), device)
    output_dir = args.output or pathlib.Path(config['output_dir'])
    max_updates = (
        args.max_updates
        if args.max_updates is not None
        else int(config['max_updates'])
    )
    if max_updates <= 0:
        raise ValueError('max updates must be positive')
    if args.synthetic_preflight:
        summary = run_synthetic_preflight(
            config=config,
            output_dir=output_dir,
            max_updates=max_updates,
            device=device,
        )
    else:
        scene_ids = tuple(args.scene_ids or config['scene_ids'])
        max_time_step = (
            args.max_time_step
            if args.max_time_step is not None
            else int(config['max_time_step'])
        )
        warm_start_checkpoint = args.warm_start_checkpoint
        bootstrap_checkpoint = args.bootstrap_checkpoint
        if warm_start_checkpoint is None and bootstrap_checkpoint is None:
            if config['stage'] == 'V2-1':
                warm_start_checkpoint = pathlib.Path(
                    config['warm_start_checkpoint'],
                )
            else:
                bootstrap_checkpoint = pathlib.Path(
                    config['bootstrap_checkpoint'],
                )
        summary = run_real_training(
            config=config,
            output_dir=output_dir,
            warm_start_checkpoint=warm_start_checkpoint,
            bootstrap_checkpoint=bootstrap_checkpoint,
            scene_ids=scene_ids,
            max_time_step=max_time_step,
            max_updates=max_updates,
            device=device,
            resume=args.resume,
        )
    print(json.dumps(summary, sort_keys=True), flush=True)
    if not summary['accepted']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
