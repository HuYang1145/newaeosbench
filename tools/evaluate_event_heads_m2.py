#!/usr/bin/env python3
"""离线评价 M2 continue、duration 与短窗口事实结果头。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Sequence

import torch


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _mean_defined(values: Sequence[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return None if not defined else sum(defined) / len(defined)


def binary_classification_metrics(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    observed: torch.Tensor | None = None,
) -> dict:
    """按零 logit 阈值汇总二分类，同时保留两类召回。"""
    if logits.shape != targets.shape:
        raise ValueError('binary logits and targets must share a shape')
    if observed is None:
        observed = torch.ones_like(targets, dtype=torch.bool)
    if observed.shape != targets.shape:
        raise ValueError('binary observed mask must match targets')
    observed = observed.bool()
    if not observed.any():
        raise ValueError('binary observed scope is empty')

    targets = targets[observed].bool()
    predictions = logits[observed] >= 0
    true_positive = int((predictions & targets).sum())
    true_negative = int((~predictions & ~targets).sum())
    false_positive = int((predictions & ~targets).sum())
    false_negative = int((~predictions & targets).sum())
    support = targets.numel()
    positive_recall = _safe_ratio(
        true_positive,
        true_positive + false_negative,
    )
    negative_recall = _safe_ratio(
        true_negative,
        true_negative + false_positive,
    )
    return {
        'support': support,
        'positive_count': int(targets.sum()),
        'negative_count': int((~targets).sum()),
        'accuracy': _safe_ratio(
            true_positive + true_negative,
            support,
        ),
        'balanced_accuracy': (
            None
            if positive_recall is None or negative_recall is None
            else (positive_recall + negative_recall) / 2
        ),
        'positive_precision': _safe_ratio(
            true_positive,
            true_positive + false_positive,
        ),
        'positive_recall': positive_recall,
        'negative_precision': _safe_ratio(
            true_negative,
            true_negative + false_negative,
        ),
        'negative_recall': negative_recall,
        'predicted_positive_rate': _safe_ratio(
            true_positive + false_positive,
            support,
        ),
        'confusion': {
            'true_negative': true_negative,
            'false_positive': false_positive,
            'false_negative': false_negative,
            'true_positive': true_positive,
        },
    }


def multiclass_classification_metrics(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_names: Sequence[str],
    observed: torch.Tensor | None = None,
) -> dict:
    """汇总多分类混淆矩阵与宏平均召回，显式暴露多数类塌缩。"""
    if logits.ndim != targets.ndim + 1:
        raise ValueError('multiclass logits must add one class dimension')
    if logits.shape[:-1] != targets.shape:
        raise ValueError('multiclass logits and targets must align')
    names = tuple(str(name) for name in class_names)
    num_classes = logits.shape[-1]
    if len(names) != num_classes:
        raise ValueError('class names must match the logits class dimension')
    if observed is None:
        observed = torch.ones_like(targets, dtype=torch.bool)
    if observed.shape != targets.shape:
        raise ValueError('multiclass observed mask must match targets')
    observed = observed.bool()
    if not observed.any():
        raise ValueError('multiclass observed scope is empty')

    targets = targets[observed].long()
    if (targets < 0).any() or (targets >= num_classes).any():
        raise ValueError('multiclass target is out of range')
    predictions = logits[observed].argmax(-1)
    confusion = torch.zeros(
        num_classes,
        num_classes,
        dtype=torch.long,
    )
    indices = targets * num_classes + predictions
    confusion.view(-1).scatter_add_(
        0,
        indices.cpu(),
        torch.ones_like(indices, device='cpu'),
    )
    target_counts = confusion.sum(1)
    predicted_counts = confusion.sum(0)
    recalls = [
        _safe_ratio(
            int(confusion[index, index]),
            int(target_counts[index]),
        )
        for index in range(num_classes)
    ]
    precisions = [
        _safe_ratio(
            int(confusion[index, index]),
            int(predicted_counts[index]),
        )
        for index in range(num_classes)
    ]
    f1_scores = [
        (
            None
            if precision is None or recall is None
            else (
                0.0
                if precision + recall == 0
                else 2 * precision * recall / (precision + recall)
            )
        )
        for precision, recall in zip(precisions, recalls)
    ]
    support = targets.numel()
    return {
        'support': support,
        'accuracy': _safe_ratio(int((predictions == targets).sum()), support),
        'balanced_accuracy': _mean_defined(recalls),
        'macro_f1': _mean_defined(f1_scores),
        'target_counts': {
            name: int(target_counts[index])
            for index, name in enumerate(names)
        },
        'predicted_counts': {
            name: int(predicted_counts[index])
            for index, name in enumerate(names)
        },
        'precision': {
            name: precisions[index]
            for index, name in enumerate(names)
        },
        'recall': {
            name: recalls[index]
            for index, name in enumerate(names)
        },
        'confusion': confusion.tolist(),
    }


def _gather_executed_edges(
    values: torch.Tensor,
    actions_task_id: torch.Tensor,
) -> torch.Tensor:
    if values.shape[:2] != actions_task_id.shape:
        raise ValueError('prediction and actions must share batch/satellites')
    indices = actions_task_id.clamp_min(0)
    if values.ndim == 3:
        return values.gather(2, indices.unsqueeze(-1)).squeeze(-1)
    if values.ndim == 4:
        expanded = indices.unsqueeze(-1).unsqueeze(-1).expand(
            -1,
            -1,
            1,
            values.shape[-1],
        )
        return values.gather(2, expanded).squeeze(2)
    raise ValueError('edge prediction must have rank 3 or 4')


def _observed_binary(
    logits: torch.Tensor,
    targets: torch.Tensor,
    observed: torch.Tensor,
) -> dict[str, torch.Tensor]:
    observed = observed.bool()
    return {
        'logits': logits[observed].detach().cpu(),
        'targets': targets[observed].bool().detach().cpu(),
    }


def collect_temporal_predictions(
    *,
    output,
    temporal,
    actions_task_id: torch.Tensor,
) -> dict:
    """抽取当前 batch 中实际执行且已观测的事件与结果预测。"""
    valid = temporal.outcome_valid.bool() & (actions_task_id >= 0)
    horizons = temporal.horizons
    if horizons.ndim == 2:
        if not torch.equal(horizons, horizons[:1].expand_as(horizons)):
            raise ValueError('all samples must share temporal horizons')
        horizons = horizons[0]
    if horizons.ndim != 1:
        raise ValueError('temporal horizons must be one-dimensional')
    horizon_names = tuple(str(int(value)) for value in horizons)

    continue_logits = _gather_executed_edges(
        output.continue_logits,
        actions_task_id,
    )
    duration_logits = _gather_executed_edges(
        output.duration_logits,
        actions_task_id,
    )
    duration_observed = valid & temporal.event_duration_observed.bool()
    duration = {
        'logits': duration_logits[duration_observed].detach().cpu(),
        'targets': (
            temporal.event_duration_index[duration_observed]
            .long()
            .detach()
            .cpu()
        ),
    }

    outcomes = {}
    specs = {
        'visible': (
            output.visible_next_logits,
            output.visible_logits,
            temporal.visible_next,
            temporal.visible,
            temporal.visible_observed,
        ),
        'progress': (
            output.progress_next_logits,
            output.progress_logits,
            temporal.progress_next,
            temporal.progress,
            temporal.progress_observed,
        ),
        'completion': (
            output.completed_next_logits,
            output.completed_logits,
            temporal.completed_next,
            temporal.completed,
            temporal.completion_observed,
        ),
    }
    for name, (
        next_logits,
        horizon_logits,
        next_targets,
        horizon_targets,
        horizon_observed,
    ) in specs.items():
        executed_next = _gather_executed_edges(
            next_logits,
            actions_task_id,
        )
        executed_horizons = _gather_executed_edges(
            horizon_logits,
            actions_task_id,
        )
        windows = {
            'next': _observed_binary(
                executed_next,
                next_targets,
                valid,
            ),
        }
        for index, horizon_name in enumerate(horizon_names):
            windows[horizon_name] = _observed_binary(
                executed_horizons[..., index],
                horizon_targets[..., index],
                valid & horizon_observed[..., index].bool(),
            )
        outcomes[name] = windows

    return {
        'continue': _observed_binary(
            continue_logits,
            temporal.event_continue,
            valid,
        ),
        'duration': duration,
        'outcomes': outcomes,
        'horizons': horizon_names,
    }


def _state_dict(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(payload, dict) and 'state_dict' in payload:
        payload = payload['state_dict']
    if not isinstance(payload, dict):
        raise TypeError('checkpoint must contain a state dict')
    return payload


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple) and hasattr(value, '_fields'):
        return type(value)(*(
            _move_to_device(item, device)
            for item in value
        ))
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


def _concatenate(
    chunks: Sequence[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    return {
        name: torch.cat([chunk[name] for chunk in chunks])
        for name in ('logits', 'targets')
    }


def _optional_binary_metrics(
    values: dict[str, torch.Tensor],
) -> dict:
    if values['logits'].numel() == 0:
        return {'support': 0}
    return binary_classification_metrics(**values)


def aggregate_temporal_predictions(
    chunks: Sequence[dict],
    *,
    duration_names: Sequence[str] = ('1', '5', '15', '30', '60'),
) -> dict:
    """跨场景合并原始预测后计算一次指标，避免平均比例失真。"""
    if not chunks:
        raise ValueError('prediction chunks are empty')
    horizons = chunks[0]['horizons']
    if any(chunk['horizons'] != horizons for chunk in chunks):
        raise ValueError('prediction chunks use different horizons')

    continue_values = _concatenate([
        chunk['continue'] for chunk in chunks
    ])
    duration_values = _concatenate([
        chunk['duration'] for chunk in chunks
    ])
    outcomes = {}
    for outcome_name in ('visible', 'progress', 'completion'):
        outcomes[outcome_name] = {}
        for window in ('next', *horizons):
            values = _concatenate([
                chunk['outcomes'][outcome_name][window]
                for chunk in chunks
            ])
            outcomes[outcome_name][window] = _optional_binary_metrics(values)

    return {
        'continue': binary_classification_metrics(**continue_values),
        'duration': multiclass_classification_metrics(
            **duration_values,
            class_names=duration_names,
        ),
        'outcomes': outcomes,
    }


def evaluate_checkpoint(
    *,
    checkpoint: Path,
    annotation_file: Path,
    split: str,
    limit: int | None,
    batch_size: int,
    seed: int,
    device_name: str,
) -> dict:
    """在旧专家轨迹上离线评价 M2，不启动 Basilisk。"""
    from todd.configs import PyConfig

    from constellation.new_transformers.dataset import JointBatch, JointDataset
    from constellation.new_transformers.model import JointModel

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not annotation_file.is_file():
        raise FileNotFoundError(annotation_file)
    if limit is not None and limit <= 0:
        raise ValueError('limit must be positive')
    if batch_size <= 0:
        raise ValueError('batch-size must be positive')

    device = torch.device(device_name)
    config = PyConfig.load(
        'constellation/new_transformers/config_event_heads_m2.py'
    )
    model_kwargs = dict(config.trainer.model)
    model_kwargs.pop('type')
    model = JointModel(**model_kwargs)
    incompatible = model.load_state_dict(
        _state_dict(checkpoint),
        strict=False,
    )
    model.to(device).eval()

    dataset = JointDataset(
        split=split,
        annotation_file=annotation_file.name,
        batch_size=batch_size,
        constraint_batch_size=2,
        include_temporal_history=True,
        temporal_horizons=(5, 15, 30, 60),
    )
    scene_count = len(dataset) if limit is None else min(limit, len(dataset))
    chunks = []
    scenes = []
    with torch.no_grad():
        for index in range(scene_count):
            random.seed(seed + index)
            torch.manual_seed(seed + index)
            batch = _move_to_device(dataset[index], device)
            if not isinstance(batch, JointBatch):
                batch = JointBatch(*batch)
            _, temporal_output = model._predict_with_temporal_output(
                batch.time_steps,
                batch.constellation_sensor_type,
                batch.constellation_sensor_enabled,
                batch.constellation_data,
                batch.constellation_mask,
                batch.tasks_sensor_type,
                batch.tasks_data,
                batch.tasks_mask,
                temporal_history=model._history_from_batch(batch),
            )
            if temporal_output is None or batch.temporal is None:
                raise RuntimeError('M2 temporal output/targets are missing')
            chunk = collect_temporal_predictions(
                output=temporal_output,
                temporal=batch.temporal,
                actions_task_id=batch.actions_task_id,
            )
            chunks.append(chunk)
            scenes.append({
                'index': index,
                'scene_id': int(batch.annotation_id),
                'time_steps': list(batch.time_steps),
                'continue_support': int(
                    chunk['continue']['targets'].numel()
                ),
                'duration_support': int(
                    chunk['duration']['targets'].numel()
                ),
            })
            print(
                f'[evaluate] {index + 1}/{scene_count} '
                f'scene={batch.annotation_id}',
                flush=True,
            )

    return {
        'purpose': 'M2 offline event-head evaluation; no Basilisk',
        'checkpoint': str(checkpoint),
        'annotation_file': str(annotation_file),
        'split': split,
        'scene_count': scene_count,
        'batch_size_per_scene': batch_size,
        'seed': seed,
        'device': str(device),
        'missing_keys': list(incompatible.missing_keys),
        'unexpected_keys': list(incompatible.unexpected_keys),
        'scenes': scenes,
        'metrics': aggregate_temporal_predictions(chunks),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--annotation-file', type=Path, required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_checkpoint(
        checkpoint=args.checkpoint,
        annotation_file=args.annotation_file,
        split=args.split,
        limit=args.limit,
        batch_size=args.batch_size,
        seed=args.seed,
        device_name=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'[done] output={args.output}', flush=True)


if __name__ == '__main__':
    main()
