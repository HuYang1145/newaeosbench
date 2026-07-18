#!/usr/bin/env python3
"""审计既有轨迹中的多时间尺度卫星—任务边标签。"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import torch

from constellation.new_transformers.multi_horizon_edge_labels import (
    aggregate_edge_label_summaries,
    summarize_trajectory_edge_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory_root', type=Path, nargs='?')
    parser.add_argument('--annotation-file', type=Path)
    parser.add_argument('--split')
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument(
        '--taskset-root', type=Path, default=Path('data/tasksets')
    )
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--limit', type=int)
    parser.add_argument(
        '--horizons', type=int, nargs='+', default=[5, 15, 30]
    )
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


@dataclasses.dataclass(frozen=True)
class AuditTarget:
    scene_id: int
    epoch: int | None
    split: str
    trajectory_path: Path
    taskset_path: Path


def annotation_targets(
    annotation_file: Path,
    *,
    split: str,
    data_root: Path,
    taskset_root: Path,
) -> list[AuditTarget]:
    """严格按 annotation 的 scene id 与 epoch 路由轨迹。"""
    payload = json.loads(annotation_file.read_text())
    if not isinstance(payload, dict):
        raise TypeError('annotation must be an object')
    ids = payload.get('ids')
    epochs = payload.get('epochs')
    if not isinstance(ids, list) or not isinstance(epochs, list):
        raise TypeError('annotation ids and epochs must be lists')
    if len(ids) != len(epochs):
        raise ValueError('annotation ids and epochs must have equal length')
    normalized_ids = [int(value) for value in ids]
    normalized_epochs = [int(value) for value in epochs]
    duplicates = sorted(
        scene_id
        for scene_id, count in Counter(normalized_ids).items()
        if count > 1
    )
    if duplicates:
        raise ValueError(f'annotation has duplicate scene ids: {duplicates}')

    targets = []
    missing = []
    for scene_id, epoch in zip(normalized_ids, normalized_epochs):
        if scene_id < 0 or epoch <= 0:
            raise ValueError(
                'scene ids must be non-negative and epochs positive'
            )
        relative = (
            Path(split) / f'{scene_id // 1000:02}' / f'{scene_id:05}'
        )
        trajectory_path = (
            data_root / f'trajectories.{epoch}'
            / relative.with_suffix('.pth')
        )
        taskset_path = taskset_root / relative.with_suffix('.json')
        for path in (trajectory_path, taskset_path):
            if not path.is_file():
                missing.append(path)
        targets.append(AuditTarget(
            scene_id=scene_id,
            epoch=epoch,
            split=split,
            trajectory_path=trajectory_path,
            taskset_path=taskset_path,
        ))
    if missing:
        preview = ', '.join(str(path) for path in missing[:5])
        suffix = '' if len(missing) <= 5 else f' (+{len(missing) - 5} more)'
        raise FileNotFoundError(f'missing routed files: {preview}{suffix}')
    return targets


def taskset_path_for_trajectory(
    trajectory_path: Path,
    *,
    trajectory_root: Path,
    taskset_root: Path,
) -> Path:
    relative = trajectory_path.relative_to(trajectory_root)
    return (taskset_root / relative).with_suffix('.json')


def _task_durations(path: Path, *, num_tasks: int) -> torch.Tensor:
    tasks = json.loads(path.read_text())
    if not isinstance(tasks, list):
        raise TypeError(f'taskset must be a list: {path}')
    by_id = {int(task['id']): int(task['duration']) for task in tasks}
    expected_ids = set(range(num_tasks))
    if set(by_id) != expected_ids:
        raise ValueError(f'task ids do not align with trajectory: {path}')
    return torch.tensor([by_id[index] for index in range(num_tasks)])


def _audit_paths(
    trajectory_path: Path,
    *,
    taskset_path: Path,
    horizons: tuple[int, ...],
) -> dict[str, Any]:
    payload = torch.load(
        trajectory_path,
        map_location='cpu',
        weights_only=False,
    )
    actions = torch.as_tensor(payload['actions']['task_id']).long()
    is_visible = torch.as_tensor(payload['is_visible']).bool()
    progress = torch.as_tensor(payload['taskset']['progress'])
    durations = _task_durations(taskset_path, num_tasks=progress.shape[1])
    summary = summarize_trajectory_edge_labels(
        actions=actions,
        is_visible=is_visible,
        progress=progress,
        task_durations=durations,
        horizons=horizons,
    )
    return {
        'trajectory': str(trajectory_path),
        'taskset': str(taskset_path),
        'summary': summary,
    }


def audit_trajectory(
    trajectory_path: Path,
    *,
    trajectory_root: Path,
    taskset_root: Path,
    horizons: tuple[int, ...],
) -> dict[str, Any]:
    taskset_path = taskset_path_for_trajectory(
        trajectory_path,
        trajectory_root=trajectory_root,
        taskset_root=taskset_root,
    )
    return _audit_paths(
        trajectory_path,
        taskset_path=taskset_path,
        horizons=horizons,
    )


def _audit_target(target: AuditTarget, horizons: tuple[int, ...]) -> dict[str, Any]:
    result = _audit_paths(
        target.trajectory_path,
        taskset_path=target.taskset_path,
        horizons=horizons,
    )
    result.update({
        'scene_id': target.scene_id,
        'epoch': target.epoch,
        'split': target.split,
    })
    return result


def _worker_init() -> None:
    torch.set_num_threads(1)


def _audit_target_job(
    args: tuple[AuditTarget, tuple[int, ...]],
) -> dict[str, Any]:
    return _audit_target(*args)


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError('limit must be positive')
    if args.workers <= 0:
        raise ValueError('workers must be positive')
    horizons = tuple(int(value) for value in args.horizons)
    if any(value <= 0 for value in horizons) or len(set(horizons)) != len(
        horizons
    ):
        raise ValueError('horizons must be positive and unique')

    annotation_mode = args.annotation_file is not None
    if annotation_mode:
        if args.trajectory_root is not None:
            raise ValueError(
                'trajectory_root and --annotation-file are mutually exclusive'
            )
        if args.split is None:
            raise ValueError('--split is required with --annotation-file')
        targets = annotation_targets(
            args.annotation_file,
            split=args.split,
            data_root=args.data_root,
            taskset_root=args.taskset_root,
        )
    else:
        if args.trajectory_root is None:
            raise ValueError(
                'provide trajectory_root or --annotation-file'
            )
        paths = sorted(args.trajectory_root.rglob('*.pth'))
        if not paths:
            raise FileNotFoundError(
                f'no trajectory files found under {args.trajectory_root}'
            )
        targets = [
            AuditTarget(
                scene_id=int(path.stem),
                epoch=None,
                split=path.relative_to(args.trajectory_root).parts[0],
                trajectory_path=path,
                taskset_path=taskset_path_for_trajectory(
                    path,
                    trajectory_root=args.trajectory_root,
                    taskset_root=args.taskset_root,
                ),
            )
            for path in paths
        ]
    if args.limit is not None:
        targets = targets[:args.limit]

    jobs = [(target, horizons) for target in targets]
    if args.workers == 1:
        results = map(_audit_target_job, jobs)
    else:
        executor = ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_worker_init,
        )
        results = executor.map(_audit_target_job, jobs, chunksize=1)
    scenes = []
    try:
        for index, (target, result) in enumerate(
            zip(targets, results),
            start=1,
        ):
            print(
                f'[audit] {index}/{len(targets)} '
                f'{target.trajectory_path}',
                flush=True,
            )
            scenes.append(result)
    finally:
        if args.workers != 1:
            executor.shutdown()
    combined = aggregate_edge_label_summaries([
        scene['summary'] for scene in scenes
    ])
    output = {
        'purpose': 'multi-horizon executed-edge label audit; Actor unchanged',
        'config': {
            'mode': 'annotation' if annotation_mode else 'trajectory_root',
            'trajectory_root': (
                None
                if args.trajectory_root is None
                else str(args.trajectory_root)
            ),
            'annotation_file': (
                None
                if args.annotation_file is None
                else str(args.annotation_file)
            ),
            'split': args.split,
            'data_root': str(args.data_root),
            'taskset_root': str(args.taskset_root),
            'horizons': list(horizons),
            'limit': args.limit,
            'workers': args.workers,
            'causal_alignment': 'action[t] -> outcome[t+1]',
            'early_switch_without_event': 'censored',
        },
        'routing': {
            'scene_count': len(targets),
            'unique_scene_count': len({target.scene_id for target in targets}),
            'epochs': dict(sorted(Counter(
                target.epoch for target in targets
            ).items(), key=lambda item: (item[0] is None, item[0]))),
        },
        'scenes': scenes,
        'combined': combined,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(f'[done] output={args.output}', flush=True)


if __name__ == '__main__':
    main()
