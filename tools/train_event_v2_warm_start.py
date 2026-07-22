#!/usr/bin/env python3
"""训练 V2-0 离线 warm start，并保存可审计恢复点。"""

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
import pathlib
import random
import runpy
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from constellation.new_transformers.event_v2.dataset import (
    EventV2OfflineDataset,
)
from constellation.new_transformers.event_v2.model import (
    EventJointActorCritic,
)
from constellation.new_transformers.event_v2.offline import (
    event_v2_offline_loss,
)
from constellation.new_transformers.event_v2.transition import (
    transition_schema_fingerprint,
)


CHECKPOINT_VERSION = 1
STAGE = 'V2-0'


@dataclass(frozen=True)
class TrainingCounters:
    steps: int = 0
    processed_physical_seconds: int = 0
    episodes: int = 0
    events: int = 0


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
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def build_training_checkpoint(
    *,
    model: EventJointActorCritic,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    config_fingerprint_value: str,
    normalizer: Mapping[str, torch.Tensor],
    counters: TrainingCounters,
) -> dict[str, Any]:
    return {
        'checkpoint_version': CHECKPOINT_VERSION,
        'stage': STAGE,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'amp_scaler': scaler.state_dict(),
        'policy_version': 0,
        'transition_schema_fingerprint': transition_schema_fingerprint(),
        'config_fingerprint': config_fingerprint_value,
        'normalizer': dict(normalizer),
        'rng_state': capture_rng_state(),
        'processed_physical_seconds': counters.processed_physical_seconds,
        'episodes': counters.episodes,
        'events': counters.events,
        'steps': counters.steps,
        'unfreeze_state': {
            'backbone_is_frozen': model.backbone_is_frozen,
        },
    }


def save_checkpoint_atomic(
    path: str | pathlib.Path,
    checkpoint: Mapping[str, Any],
) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(dict(checkpoint), temporary)
    os.replace(temporary, path)


def load_training_checkpoint(
    *,
    path: str | pathlib.Path,
    model: EventJointActorCritic,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    expected_config_fingerprint: str,
) -> TrainingCounters:
    checkpoint = torch.load(
        pathlib.Path(path),
        map_location='cpu',
        weights_only=False,
    )
    if checkpoint.get('checkpoint_version') != CHECKPOINT_VERSION:
        raise ValueError('checkpoint version does not match trainer')
    if checkpoint.get('stage') != STAGE:
        raise ValueError('checkpoint stage does not match V2-0')
    if checkpoint.get('transition_schema_fingerprint') != (
        transition_schema_fingerprint()
    ):
        raise ValueError('transition schema fingerprint mismatch')
    if checkpoint.get('config_fingerprint') != expected_config_fingerprint:
        raise ValueError('config fingerprint mismatch')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    scaler.load_state_dict(checkpoint['amp_scaler'])
    restore_rng_state(checkpoint['rng_state'])
    return TrainingCounters(
        steps=int(checkpoint['steps']),
        processed_physical_seconds=int(
            checkpoint['processed_physical_seconds']
        ),
        episodes=int(checkpoint['episodes']),
        events=int(checkpoint['events']),
    )


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=device.type == 'cuda')
    if isinstance(value, tuple) and hasattr(value, '_fields'):
        return type(value)(*(_move_to_device(item, device) for item in value))
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {
            key: _move_to_device(item, device)
            for key, item in value.items()
        }
    return value


def _load_config(path: pathlib.Path) -> dict[str, Any]:
    namespace = runpy.run_path(str(path))
    keys = (
        'stage',
        'max_hours',
        'seed',
        'annotation_file',
        'stage3_checkpoint',
        'output_dir',
        'max_steps',
        'event_batch_size',
        'num_workers',
        'log_interval',
        'checkpoint_interval',
        'amp',
        'amp_dtype',
        'model',
        'optimizer',
        'loss_weights',
    )
    config = {key: namespace[key] for key in keys}
    if config['stage'] != STAGE:
        raise ValueError('warm-start config must use stage V2-0')
    return config


def _normalizer_state(dataset: EventV2OfflineDataset) -> dict[str, torch.Tensor]:
    statistics = dataset._statistics
    return {
        name: value.detach().cpu()
        for name, value in statistics._asdict().items()
    }


def _device_from_argument(name: str) -> torch.device:
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    return device


def resolve_amp_dtype(name: str) -> torch.dtype:
    dtypes = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
    }
    try:
        return dtypes[name]
    except KeyError as error:
        raise ValueError(f'unsupported AMP dtype: {name}') from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train the Event Joint Transformer V2-0 warm start',
    )
    parser.add_argument('--config', type=pathlib.Path, required=True)
    parser.add_argument('--stage3-checkpoint', type=pathlib.Path)
    parser.add_argument('--output', type=pathlib.Path)
    parser.add_argument('--resume', type=pathlib.Path)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--device', default='auto')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    checkpoint_path = (
        args.stage3_checkpoint
        if args.stage3_checkpoint is not None
        else pathlib.Path(config['stage3_checkpoint'])
    )
    output_dir = (
        args.output
        if args.output is not None
        else pathlib.Path(config['output_dir'])
    )
    max_steps = (
        args.max_steps if args.max_steps is not None else config['max_steps']
    )
    if max_steps <= 0:
        raise ValueError('max_steps must be positive')
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Stage3 checkpoint not found: {checkpoint_path}')
    device = _device_from_argument(args.device)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config['seed'])

    preflight = max_steps == 1
    dataset = EventV2OfflineDataset(
        split='train',
        annotation_file=config['annotation_file'],
        batch_size=(1 if preflight else config['event_batch_size']),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=True,
        num_workers=(0 if preflight else config['num_workers']),
        pin_memory=device.type == 'cuda',
    )
    model = EventJointActorCritic(**config['model'])
    model.load_stage3_checkpoint(checkpoint_path)
    model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    optimizer_config = dict(config['optimizer'])
    learning_rate = optimizer_config.pop('lr')
    optimizer = torch.optim.AdamW(
        model.parameter_groups(learning_rate),
        **optimizer_config,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_steps,
        eta_min=learning_rate * 0.05,
    )
    amp_enabled = bool(config['amp'] and device.type == 'cuda')
    amp_dtype = resolve_amp_dtype(config['amp_dtype'])
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_enabled and amp_dtype is torch.float16,
    )
    fingerprint = config_fingerprint(config)
    counters = TrainingCounters()
    if args.resume is not None:
        counters = load_training_checkpoint(
            path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_config_fingerprint=fingerprint,
        )
    print(json.dumps({
        'stage': STAGE,
        'device': str(device),
        'schema_fingerprint': transition_schema_fingerprint(),
        'config_fingerprint': fingerprint,
        'parameters': parameter_count,
        'trainable_parameters': trainable_count,
        'amp_dtype': str(amp_dtype) if amp_enabled else None,
        'start_step': counters.steps,
        'max_steps': max_steps,
    }, sort_keys=True))

    model.train()
    iterator = iter(dataloader)
    while counters.steps < max_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)
        batch = _move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
            dtype=amp_dtype,
        ):
            losses = event_v2_offline_loss(
                model,
                batch,
                task_weight=config['loss_weights']['task'],
                termination_weight=config['loss_weights']['termination'],
                commitment_weight=config['loss_weights']['commitment'],
                value_weight=config['loss_weights']['value'],
            )
        if not all(torch.isfinite(value) for value in losses):
            raise FloatingPointError('V2-0 produced a non-finite loss')
        scaler.scale(losses.total).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ],
            max_norm=1.0,
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError('V2-0 produced a non-finite gradient')
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        counters = TrainingCounters(
            steps=counters.steps + 1,
            processed_physical_seconds=(
                counters.processed_physical_seconds
                + int(batch.event_state.delta_t.max(dim=1).values.sum().item())
            ),
            episodes=counters.episodes + 1,
            events=counters.events + len(batch.stage3_batch.time_steps),
        )
        if (
            counters.steps == 1
            or counters.steps % config['log_interval'] == 0
            or counters.steps == max_steps
        ):
            print(json.dumps({
                'step': counters.steps,
                'loss': float(losses.total.detach()),
                'task_distillation': float(losses.task_distillation.detach()),
                'termination': float(losses.termination.detach()),
                'commitment': float(losses.commitment.detach()),
                'value': float(losses.value.detach()),
                'gradient_norm': float(gradient_norm.detach()),
                'events': counters.events,
                'physical_seconds': counters.processed_physical_seconds,
            }, sort_keys=True))
        if (
            counters.steps % config['checkpoint_interval'] == 0
            or counters.steps == max_steps
        ):
            checkpoint = build_training_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config_fingerprint_value=fingerprint,
                normalizer=_normalizer_state(dataset),
                counters=counters,
            )
            save_checkpoint_atomic(
                output_dir / f'checkpoint_step_{counters.steps:06d}.pth',
                checkpoint,
            )


if __name__ == '__main__':
    main()
