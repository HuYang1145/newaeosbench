#!/usr/bin/env python3
"""在固定未见轨迹上公平比较随机 V2 与 V2-0 warm start。"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import NamedTuple
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from constellation.new_transformers.event_v2.dataset import (
    EventV2OfflineDataset,
    OfflineEventBatch,
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


LOSS_COMPONENTS = (
    'task_distillation',
    'termination',
    'commitment',
    'value',
)
CHECKPOINT_VERSION = 1
STAGE = 'V2-0'


class PairedModels(NamedTuple):
    random_model: EventJointActorCritic
    trained_model: EventJointActorCritic
    audit: dict[str, Any]


DEFAULT_CONFIG = (
    ROOT / 'constellation/new_transformers/config_event_v2_warm_start.py'
)
DEFAULT_TRAINED_CHECKPOINT = (
    ROOT
    / 'work_dirs/event_joint_transformer_v2/v2_0_warm_start'
    / 'checkpoint_step_010000.pth'
)
DEFAULT_OUTPUT = (
    ROOT
    / 'work_dirs/event_joint_transformer_v2/v2_0_unseen_offline'
    / 'summary.json'
)


def _set_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def audit_training_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    expected_config_fingerprint: str,
    expected_steps: int,
) -> dict[str, Any]:
    """在加载权重前验证 V2-0 checkpoint 的不可变元数据。"""

    if checkpoint.get('checkpoint_version') != CHECKPOINT_VERSION:
        raise ValueError('checkpoint version mismatch')
    if checkpoint.get('stage') != STAGE:
        raise ValueError('checkpoint stage mismatch')
    if checkpoint.get('steps') != expected_steps:
        raise ValueError('checkpoint step mismatch')
    schema_fingerprint = transition_schema_fingerprint()
    if checkpoint.get('transition_schema_fingerprint') != schema_fingerprint:
        raise ValueError('checkpoint schema fingerprint mismatch')
    if checkpoint.get('config_fingerprint') != expected_config_fingerprint:
        raise ValueError('checkpoint config fingerprint mismatch')
    model_state = checkpoint.get('model')
    if not isinstance(model_state, Mapping):
        raise ValueError('checkpoint model state is missing')
    return {
        'checkpoint_version': CHECKPOINT_VERSION,
        'stage': STAGE,
        'steps': expected_steps,
        'transition_schema_fingerprint': schema_fingerprint,
        'config_fingerprint': expected_config_fingerprint,
    }


def build_paired_models(
    *,
    model_kwargs: Mapping[str, Any],
    stage3_checkpoint: str | Path,
    trained_checkpoint: str | Path,
    expected_config_fingerprint: str,
    seed: int,
    device: torch.device,
) -> PairedModels:
    """构造共享 Stage3 backbone 的随机基线和严格加载的训练模型。"""

    stage3_checkpoint = Path(stage3_checkpoint)
    trained_checkpoint = Path(trained_checkpoint)
    if not stage3_checkpoint.is_file():
        raise FileNotFoundError(f'Stage3 checkpoint not found: {stage3_checkpoint}')
    if not trained_checkpoint.is_file():
        raise FileNotFoundError(
            f'V2 trained checkpoint not found: {trained_checkpoint}'
        )
    checkpoint = torch.load(
        trained_checkpoint,
        map_location='cpu',
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError('V2 checkpoint must be a mapping')
    audit = audit_training_checkpoint(
        checkpoint,
        expected_config_fingerprint=expected_config_fingerprint,
        expected_steps=10_000,
    )

    _set_seed(seed, device)
    random_model = EventJointActorCritic(**dict(model_kwargs))
    random_model.load_stage3_checkpoint(stage3_checkpoint)
    _set_seed(seed, device)
    trained_model = EventJointActorCritic(**dict(model_kwargs))
    trained_model.load_stage3_checkpoint(stage3_checkpoint)
    trained_model.load_state_dict(checkpoint['model'], strict=True)

    random_backbone = random_model.backbone.transformer.state_dict()
    trained_backbone = trained_model.backbone.transformer.state_dict()
    backbone_exact_match = (
        random_backbone.keys() == trained_backbone.keys()
        and all(
            torch.equal(value, trained_backbone[name])
            for name, value in random_backbone.items()
        )
    )
    if not backbone_exact_match:
        raise ValueError('random and trained Stage3 backbones do not match')

    random_model.eval().to(device)
    trained_model.eval().to(device)
    audit.update({
        'seed': seed,
        'strict_model_load': True,
        'backbone_exact_match': True,
    })
    return PairedModels(random_model, trained_model, audit)


def _map_tensors(value: Any, transform) -> Any:
    if isinstance(value, torch.Tensor):
        return transform(value)
    if isinstance(value, tuple) and hasattr(value, '_fields'):
        return type(value)(*(_map_tensors(item, transform) for item in value))
    if isinstance(value, tuple):
        return tuple(_map_tensors(item, transform) for item in value)
    if isinstance(value, list):
        return [_map_tensors(item, transform) for item in value]
    if isinstance(value, dict):
        return {
            key: _map_tensors(item, transform)
            for key, item in value.items()
        }
    return value


def _move_to_device(value: Any, device: torch.device) -> Any:
    if device.type == 'cuda':
        value = _map_tensors(
            value,
            lambda tensor: (
                tensor.pin_memory()
                if tensor.device.type == 'cpu' and not tensor.is_pinned()
                else tensor
            ),
        )
    return _map_tensors(
        value,
        lambda tensor: tensor.to(
            device,
            non_blocking=device.type == 'cuda',
        ),
    )


def batch_supports(batch: OfflineEventBatch | Any) -> dict[str, int]:
    """计算每个 loss 真实参与平均的标签数。"""

    constellation_mask = batch.stage3_batch.constellation_mask.bool()
    targets = batch.targets
    return {
        'task_distillation': int(
            (targets.task_observed.bool() & constellation_mask).sum().item()
        ),
        'termination': int(
            (
                targets.termination_observed.bool()
                & constellation_mask
            ).sum().item()
        ),
        'commitment': int(
            (
                targets.commitment_observed.bool()
                & constellation_mask
            ).sum().item()
        ),
        'value': int(targets.value_returns.numel()),
    }


def _loss_dict(losses: Any) -> dict[str, float]:
    return {
        name: float(getattr(losses, name).detach().float().cpu().item())
        for name in (*LOSS_COMPONENTS, 'total')
    }


def evaluate_dataset(
    *,
    dataset: Any,
    random_model: Any,
    trained_model: Any,
    loss_weights: Mapping[str, float],
    device: torch.device,
    seed: int,
    limit: int | None = None,
    loss_fn=event_v2_offline_loss,
) -> list[dict[str, Any]]:
    """每场只构造一个 batch，再依次评价两个模型。"""

    scene_count = len(dataset) if limit is None else min(len(dataset), limit)
    if scene_count <= 0:
        raise ValueError('evaluation requires at least one scene')
    records: list[dict[str, Any]] = []
    loss_arguments = {
        'task_weight': float(loss_weights['task_distillation']),
        'termination_weight': float(loss_weights['termination']),
        'commitment_weight': float(loss_weights['commitment']),
        'value_weight': float(loss_weights['value']),
    }
    amp_enabled = device.type == 'cuda'
    with torch.inference_mode():
        for index in range(scene_count):
            _set_seed(seed + index, device)
            cpu_batch = dataset[index]
            support = batch_supports(cpu_batch)
            batch = _move_to_device(cpu_batch, device)
            with torch.autocast(
                device_type=device.type,
                enabled=amp_enabled,
                dtype=torch.bfloat16,
            ):
                random_losses = loss_fn(
                    random_model,
                    batch,
                    **loss_arguments,
                )
                trained_losses = loss_fn(
                    trained_model,
                    batch,
                    **loss_arguments,
                )
            records.append({
                'scene_index': index,
                'scene_id': int(cpu_batch.stage3_batch.annotation_id),
                'event_times': [
                    int(value)
                    for value in cpu_batch.stage3_batch.time_steps
                ],
                'actual_events': len(cpu_batch.stage3_batch.time_steps),
                'supports': support,
                'random': _loss_dict(random_losses),
                'trained': _loss_dict(trained_losses),
            })
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return records


def cuda_memory_snapshot(
    device: torch.device,
    *,
    requested_event_batch_size: int,
    maximum_actual_events: int,
) -> dict[str, Any]:
    """返回稳定的 GPU 峰值审计字段；CPU smoke 保留同一 schema。"""

    if device.type != 'cuda':
        return {
            'device': str(device),
            'device_name': None,
            'requested_event_batch_size': requested_event_batch_size,
            'maximum_actual_events': maximum_actual_events,
            'max_memory_allocated_bytes': None,
            'max_memory_reserved_bytes': None,
            'total_memory_bytes': None,
            'max_reserved_fraction': None,
        }
    properties = torch.cuda.get_device_properties(device)
    total_memory = int(properties.total_memory)
    maximum_reserved = int(torch.cuda.max_memory_reserved(device))
    return {
        'device': str(device),
        'device_name': properties.name,
        'requested_event_batch_size': requested_event_batch_size,
        'maximum_actual_events': maximum_actual_events,
        'max_memory_allocated_bytes': int(
            torch.cuda.max_memory_allocated(device)
        ),
        'max_memory_reserved_bytes': maximum_reserved,
        'total_memory_bytes': total_memory,
        'max_reserved_fraction': maximum_reserved / total_memory,
    }


def write_json_atomic(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool,
) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f'output already exists: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def aggregate_weighted_losses(
    scenes: Sequence[Mapping[str, Any]],
    loss_weights: Mapping[str, float],
) -> dict[str, Any]:
    """按各监督头自己的事实标签数聚合跨场景 loss。"""

    if not scenes:
        raise ValueError('at least one scene is required')
    if set(loss_weights) != set(LOSS_COMPONENTS):
        raise ValueError('loss weights must cover every loss component')
    weights = {name: float(loss_weights[name]) for name in LOSS_COMPONENTS}
    if any(not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError('loss weights must be finite and non-negative')

    supports = {name: 0 for name in LOSS_COMPONENTS}
    weighted = {
        model_name: {name: 0.0 for name in LOSS_COMPONENTS}
        for model_name in ('random', 'trained')
    }
    for scene in scenes:
        for name in LOSS_COMPONENTS:
            support = int(scene['supports'][name])
            if support < 0:
                raise ValueError('loss support must be non-negative')
            supports[name] += support
            for model_name in ('random', 'trained'):
                value = float(scene[model_name][name])
                weighted[model_name][name] += value * support

    aggregated: dict[str, dict[str, float]] = {}
    for model_name in ('random', 'trained'):
        components = {
            name: (
                weighted[model_name][name] / supports[name]
                if supports[name] > 0 else math.nan
            )
            for name in LOSS_COMPONENTS
        }
        components['total'] = sum(
            components[name] * weights[name]
            for name in LOSS_COMPONENTS
        )
        aggregated[model_name] = components

    metric_names = (*LOSS_COMPONENTS, 'total')
    delta = {
        name: aggregated['trained'][name] - aggregated['random'][name]
        for name in metric_names
    }
    relative_reduction = {
        name: (
            aggregated['random'][name] - aggregated['trained'][name]
        ) / max(aggregated['random'][name], 1e-12)
        for name in metric_names
    }
    return {
        'supports': supports,
        'loss_weights': weights,
        'random': aggregated['random'],
        'trained': aggregated['trained'],
        'delta': delta,
        'relative_reduction': relative_reduction,
    }


def decide_acceptance(
    *,
    metrics: Mapping[str, Any],
    scene_count: int,
    expected_scene_count: int,
    audit_passed: bool,
) -> dict[str, Any]:
    """执行固定的严格验收门槛，并返回全部拒绝原因。"""

    reasons: list[str] = []
    if scene_count != expected_scene_count:
        reasons.append(
            f'scene count mismatch: {scene_count} != {expected_scene_count}'
        )
    if not audit_passed:
        reasons.append('checkpoint or paired-input audit failed')

    supports = metrics.get('supports', {})
    for name in LOSS_COMPONENTS:
        if int(supports.get(name, 0)) <= 0:
            reasons.append(f'{name} support is not positive')

    for section in ('random', 'trained', 'delta', 'relative_reduction'):
        values = metrics.get(section, {})
        for name in (*LOSS_COMPONENTS, 'total'):
            value = float(values.get(name, math.nan))
            if not math.isfinite(value):
                reasons.append(f'{section}.{name} is not finite')

    for name in (*LOSS_COMPONENTS, 'total'):
        random_value = float(metrics.get('random', {}).get(name, math.nan))
        trained_value = float(metrics.get('trained', {}).get(name, math.nan))
        if math.isfinite(random_value) and math.isfinite(trained_value):
            if not trained_value < random_value:
                reasons.append(f'{name} did not strictly decrease')

    return {'accepted': not reasons, 'reasons': reasons}


def build_summary(
    *,
    records: Sequence[Mapping[str, Any]],
    loss_weights: Mapping[str, float],
    checkpoint_audit: Mapping[str, Any],
    resources: Mapping[str, Any],
    formal: bool,
    event_batch_size: int,
    annotation_file: str,
) -> dict[str, Any]:
    """组合完整可复核结果；probe 永远不能冒充正式验收。"""

    metrics = aggregate_weighted_losses(records, loss_weights)
    checkpoint_ok = bool(
        checkpoint_audit.get('strict_model_load')
        and checkpoint_audit.get('backbone_exact_match')
    )
    decision = decide_acceptance(
        metrics=metrics,
        scene_count=len(records),
        expected_scene_count=64,
        audit_passed=checkpoint_ok and formal,
    )
    return {
        'schema_version': 1,
        'stage': STAGE,
        'scope': {
            'split': 'val_unseen',
            'annotation_file': annotation_file,
            'expected_scenes': 64,
            'processed_scenes': len(records),
            'event_batch_size': event_batch_size,
            'formal': formal,
        },
        'audit': {
            **dict(checkpoint_audit),
            'paired_batch_reuse': True,
            'called_basilisk': False,
            'read_test': False,
        },
        'resources': dict(resources),
        'metrics': metrics,
        'decision': decision,
        'scenes': list(records),
    }


def _resolve_from_root(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _device_from_name(name: str) -> torch.device:
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    return device


def validate_evaluation_scope(
    *,
    annotation_file: str,
    formal: bool,
    limit: int | None,
    dataset_scene_count: int | None,
) -> None:
    """在读取数据前锁定 split，并在构造后核对正式场景数。"""

    annotation_name = Path(annotation_file).name
    if 'test' in Path(annotation_name).stem.lower():
        raise ValueError('Test annotations are forbidden in V2-0 acceptance')
    if formal and annotation_name != 'val_unseen.json':
        raise ValueError('formal evaluation requires val_unseen.json')
    if formal and limit is not None:
        raise ValueError('formal evaluation cannot use --limit')
    if formal and dataset_scene_count is not None and dataset_scene_count != 64:
        raise ValueError(
            'formal val_unseen annotation must contain 64 scenes, '
            f'got {dataset_scene_count}'
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate V2-0 warm start on fixed val_unseen trajectories',
    )
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--annotation-file', default='val_unseen.json')
    parser.add_argument('--stage3-checkpoint', type=Path)
    parser.add_argument(
        '--trained-checkpoint',
        type=Path,
        default=DEFAULT_TRAINED_CHECKPOINT,
    )
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--event-batch-size', type=int, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--formal', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main() -> None:
    from tools.train_event_v2_warm_start import (
        _load_config,
        config_fingerprint,
    )

    args = parse_args()
    if args.event_batch_size <= 0:
        raise ValueError('event batch size must be positive')
    if args.limit is not None and args.limit <= 0:
        raise ValueError('limit must be positive')
    validate_evaluation_scope(
        annotation_file=args.annotation_file,
        formal=args.formal,
        limit=args.limit,
        dataset_scene_count=None,
    )
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f'output already exists: {args.output}')

    config_path = _resolve_from_root(args.config)
    config = _load_config(config_path)
    fingerprint = config_fingerprint(config)
    stage3_checkpoint = _resolve_from_root(
        args.stage3_checkpoint or config['stage3_checkpoint']
    )
    trained_checkpoint = _resolve_from_root(args.trained_checkpoint)
    device = _device_from_name(args.device)
    if device.type == 'cuda':
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.set_float32_matmul_precision('high')

    dataset = EventV2OfflineDataset(
        split='val_unseen',
        annotation_file=args.annotation_file,
        batch_size=args.event_batch_size,
    )
    validate_evaluation_scope(
        annotation_file=args.annotation_file,
        formal=args.formal,
        limit=args.limit,
        dataset_scene_count=len(dataset),
    )
    paired = build_paired_models(
        model_kwargs=config['model'],
        stage3_checkpoint=stage3_checkpoint,
        trained_checkpoint=trained_checkpoint,
        expected_config_fingerprint=fingerprint,
        seed=args.seed,
        device=device,
    )
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    loss_weights = {
        'task_distillation': float(config['loss_weights']['task']),
        'termination': float(config['loss_weights']['termination']),
        'commitment': float(config['loss_weights']['commitment']),
        'value': float(config['loss_weights']['value']),
    }
    records = evaluate_dataset(
        dataset=dataset,
        random_model=paired.random_model,
        trained_model=paired.trained_model,
        loss_weights=loss_weights,
        device=device,
        seed=args.seed,
        limit=args.limit,
    )
    resources = cuda_memory_snapshot(
        device,
        requested_event_batch_size=args.event_batch_size,
        maximum_actual_events=max(
            int(record['actual_events']) for record in records
        ),
    )
    summary = build_summary(
        records=records,
        loss_weights=loss_weights,
        checkpoint_audit={
            **paired.audit,
            'config_path': str(config_path),
            'stage3_checkpoint': str(stage3_checkpoint),
            'trained_checkpoint': str(trained_checkpoint),
        },
        resources=resources,
        formal=args.formal,
        event_batch_size=args.event_batch_size,
        annotation_file=args.annotation_file,
    )
    write_json_atomic(args.output, summary, overwrite=args.overwrite)
    print(json.dumps({
        'output': str(args.output),
        'processed_scenes': len(records),
        'resources': resources,
        'decision': summary['decision'],
    }, sort_keys=True))


if __name__ == '__main__':
    main()
