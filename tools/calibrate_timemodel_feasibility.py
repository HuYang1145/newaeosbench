"""离线校准 TimeModel feasibility score。"""

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import torch

from constellation.new_transformers.dataset import JointDataset, TimeSpans
from constellation.new_transformers.constants import TIME_SCALE
from constellation.new_transformers.feasibility import (
    binary_calibration_metrics,
    hard_negative_indices,
)
from constellation.new_transformers.time_model import TimeModel

TIME_MODEL_PREFIX = '_transformer._time_model.'


def build_span_rows(
    positives: TimeSpans,
    negatives: TimeSpans,
) -> torch.Tensor:
    """展开训练语义中的正负时间片段，并过滤空动作伪任务。"""
    rows = [
        positives._to_data(index)  # pylint: disable=protected-access
        for index in range(positives.total_length)
    ]
    rows.extend(
        negatives._to_data(  # pylint: disable=protected-access
            index,
            with_duration=False,
        )
        for index in range(negatives.total_length)
    )
    rows = [row for row in rows if row[3] >= 0]
    if not rows:
        return torch.empty((0, 4), dtype=torch.int)
    return torch.tensor(rows, dtype=torch.int)


def extract_time_model_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """从 JointModel checkpoint 中提取 TimeModel 参数。"""
    return {
        key.removeprefix(TIME_MODEL_PREFIX): value
        for key, value in state_dict.items()
        if key.startswith(TIME_MODEL_PREFIX)
    }


def duration_regression_metrics(
    predicted_normalized: torch.Tensor,
    target_seconds: torch.Tensor,
) -> dict[str, int | float]:
    """计算正样本持续时间的秒级 MAE 和训练尺度 MSE。"""
    positive = target_seconds >= 0
    if not positive.any():
        return {
            'duration_sample_count': 0,
            'duration_mae_s': 0.0,
            'duration_mse_normalized': 0.0,
        }
    errors = (
        predicted_normalized[positive]
        - target_seconds[positive].float() / TIME_SCALE
    )
    return {
        'duration_sample_count': int(positive.sum().item()),
        'duration_mae_s': float((errors.abs() * TIME_SCALE).mean().item()),
        'duration_mse_normalized': float(errors.square().mean().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Calibrate TimeModel feasibility scores offline.',
    )
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument(
        '--split',
        choices=['val_seen', 'val_unseen'],
        required=True,
    )
    parser.add_argument('--annotation-file')
    parser.add_argument('--max-scenes', type=int)
    parser.add_argument(
        '--thresholds',
        type=float,
        nargs='+',
        default=[0.001, 0.01, 0.05, 0.1, 0.3, 0.5],
    )
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--num-bins', type=int, default=10)
    parser.add_argument('--hard-negative-threshold', type=float, default=0.9)
    parser.add_argument('--max-hard-negatives', type=int, default=1000)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def load_time_model(checkpoint: Path, device: torch.device) -> TimeModel:
    if not checkpoint.is_file():
        raise FileNotFoundError(f'checkpoint does not exist: {checkpoint}')
    payload = torch.load(
        checkpoint,
        map_location='cpu',
        weights_only=False,
    )
    if not isinstance(payload, Mapping):
        raise TypeError('checkpoint must contain a state dict mapping')
    state_dict = extract_time_model_state_dict(payload)
    if not state_dict:
        raise ValueError(f'checkpoint has no TimeModel parameters: {checkpoint}')

    model = TimeModel()
    incompatible = model.load_state_dict(state_dict, strict=False)
    critical_missing = [
        key for key in incompatible.missing_keys
        if key == '_time_embedding' or key.startswith('_mlp.')
    ]
    if critical_missing:
        raise ValueError(
            f'checkpoint is missing TimeModel parameters: {critical_missing}',
        )
    model.to(device)
    model.eval()
    return model


def load_scene_context(
    dataset: JointDataset,
    index: int,
) -> tuple[int, int, dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    """加载单场景归一化特征和与训练一致的约束行。"""
    id_, epoch, trajectory = dataset._load_trajectory(  # pylint: disable=protected-access
        index,
    )
    time_count = trajectory['constellation']['data'].shape[0]
    time_indices = list(range(time_count))
    constellation_data = dataset._load_constellation(  # pylint: disable=protected-access
        trajectory['constellation'],
        id_,
        time_indices,
    )[2]
    tasks_data = dataset._load_tasks(  # pylint: disable=protected-access
        trajectory['taskset'],
        id_,
    )[1]
    if dataset.normalize:
        statistics = dataset._statistics  # pylint: disable=protected-access
        constellation_data = (
            (constellation_data - statistics.constellation_mean)
            / (statistics.constellation_std + 1e-6)
        )
        tasks_data = (
            (tasks_data - statistics.taskset_mean)
            / (statistics.taskset_std + 1e-6)
        )

    positives, negatives = dataset._parse_time_spans(  # pylint: disable=protected-access
        trajectory['actions']['task_id'],
        trajectory['is_visible'],
    )
    rows = build_span_rows(positives, negatives)
    return id_, epoch, trajectory, constellation_data, tasks_data, rows


def predict_scene(
    model: TimeModel,
    constellation_data: torch.Tensor,
    tasks_data: torch.Tensor,
    rows: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    probabilities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    predicted_durations: list[torch.Tensor] = []
    target_durations: list[torch.Tensor] = []
    for chunk in rows.split(batch_size):
        time_steps, durations, satellite_ids, task_ids = chunk.unbind(-1)
        with torch.inference_mode():
            predicted_duration, feasibility_logits = model._predict(  # pylint: disable=protected-access
                time_steps.to(device),
                constellation_data[
                    time_steps,
                    satellite_ids,
                ].to(device),
                tasks_data[time_steps, task_ids].to(device),
            )
        probabilities.append(feasibility_logits.sigmoid().cpu())
        targets.append((durations >= 0).cpu())
        predicted_durations.append(predicted_duration.cpu())
        target_durations.append(durations.cpu())

    if not probabilities:
        return (
            torch.empty(0),
            torch.empty(0, dtype=torch.bool),
            torch.empty(0),
            torch.empty(0, dtype=torch.long),
        )
    return (
        torch.cat(probabilities),
        torch.cat(targets),
        torch.cat(predicted_durations),
        torch.cat(target_durations),
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.max_scenes is not None and args.max_scenes <= 0:
        raise ValueError('max-scenes must be positive')
    if args.batch_size <= 0:
        raise ValueError('batch-size must be positive')
    if args.num_bins <= 0:
        raise ValueError('num-bins must be positive')
    if args.max_hard_negatives < 0:
        raise ValueError('max-hard-negatives must be non-negative')
    for threshold in [*args.thresholds, args.hard_negative_threshold]:
        if not 0 <= threshold <= 1:
            raise ValueError('all thresholds must be in [0, 1]')


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'CUDA is unavailable for requested device {device}')

    model = load_time_model(args.checkpoint, device)
    dataset = JointDataset(
        split=args.split,
        annotation_file=args.annotation_file,
        batch_size=1,
        constraint_batch_size=1,
    )
    scene_count = len(dataset)
    if args.max_scenes is not None:
        scene_count = min(scene_count, args.max_scenes)

    all_probabilities: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_predicted_durations: list[torch.Tensor] = []
    all_target_durations: list[torch.Tensor] = []
    scene_summaries: list[dict[str, int]] = []
    hard_negatives: list[dict[str, int | float | str]] = []
    hard_negative_count = 0

    for index in range(scene_count):
        (
            id_,
            epoch,
            _trajectory,
            constellation_data,
            tasks_data,
            rows,
        ) = load_scene_context(dataset, index)
        (
            probabilities,
            targets,
            predicted_durations,
            target_durations,
        ) = predict_scene(
            model,
            constellation_data,
            tasks_data,
            rows,
            batch_size=args.batch_size,
            device=device,
        )
        if probabilities.numel() == 0:
            continue
        all_probabilities.append(probabilities)
        all_targets.append(targets)
        all_predicted_durations.append(predicted_durations)
        all_target_durations.append(target_durations)
        scene_summaries.append(dict(
            id=id_,
            epoch=epoch,
            sample_count=probabilities.numel(),
        ))

        indices = hard_negative_indices(
            probabilities,
            targets,
            threshold=args.hard_negative_threshold,
        )
        hard_negative_count += indices.numel()
        remaining = args.max_hard_negatives - len(hard_negatives)
        for local_index in indices[:max(remaining, 0)].tolist():
            time_step, _duration, satellite_id, task_id = rows[local_index].tolist()
            hard_negatives.append(dict(
                split=args.split,
                scene_id=id_,
                trajectory_epoch=epoch,
                time_step=time_step,
                satellite_id=satellite_id,
                task_id=task_id,
                feasibility_probability=float(probabilities[local_index]),
            ))

    if not all_probabilities:
        raise ValueError('calibration scope produced no real satellite-task pairs')
    probabilities = torch.cat(all_probabilities)
    targets = torch.cat(all_targets)
    duration_metrics = duration_regression_metrics(
        torch.cat(all_predicted_durations),
        torch.cat(all_target_durations),
    )
    threshold_metrics = [
        binary_calibration_metrics(
            probabilities,
            targets,
            threshold=threshold,
            num_bins=args.num_bins,
        )
        for threshold in args.thresholds
    ]

    report = dict(
        checkpoint=str(args.checkpoint),
        split=args.split,
        annotation_file=args.annotation_file or f'{args.split}.json',
        scene_count=len(scene_summaries),
        sample_count=probabilities.numel(),
        label_semantics='selected_action_continuous_visibility',
        **duration_metrics,
        thresholds=threshold_metrics,
        hard_negative_threshold=args.hard_negative_threshold,
        hard_negative_count=hard_negative_count,
        hard_negatives=hard_negatives,
        scenes=scene_summaries,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(args.output)


if __name__ == '__main__':
    main()
