#!/usr/bin/env python3
"""审计既有轨迹中的多时间尺度卫星—任务边标签。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from constellation.new_transformers.multi_horizon_edge_labels import (
    aggregate_edge_label_summaries,
    summarize_trajectory_edge_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory_root', type=Path)
    parser.add_argument(
        '--taskset-root', type=Path, default=Path('data/tasksets')
    )
    parser.add_argument('--limit', type=int)
    parser.add_argument(
        '--horizons', type=int, nargs='+', default=[5, 15, 30]
    )
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


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


def audit_trajectory(
    trajectory_path: Path,
    *,
    trajectory_root: Path,
    taskset_root: Path,
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
    taskset_path = taskset_path_for_trajectory(
        trajectory_path,
        trajectory_root=trajectory_root,
        taskset_root=taskset_root,
    )
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


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError('limit must be positive')
    horizons = tuple(int(value) for value in args.horizons)
    if any(value <= 0 for value in horizons) or len(set(horizons)) != len(
        horizons
    ):
        raise ValueError('horizons must be positive and unique')

    paths = sorted(args.trajectory_root.rglob('*.pth'))
    if args.limit is not None:
        paths = paths[:args.limit]
    if not paths:
        raise FileNotFoundError(
            f'no trajectory files found under {args.trajectory_root}'
        )

    scenes = []
    for index, path in enumerate(paths, start=1):
        print(f'[audit] {index}/{len(paths)} {path}', flush=True)
        scenes.append(
            audit_trajectory(
                path,
                trajectory_root=args.trajectory_root,
                taskset_root=args.taskset_root,
                horizons=horizons,
            )
        )
    combined = aggregate_edge_label_summaries([
        scene['summary'] for scene in scenes
    ])
    output = {
        'purpose': 'multi-horizon executed-edge label audit; Actor unchanged',
        'config': {
            'trajectory_root': str(args.trajectory_root),
            'taskset_root': str(args.taskset_root),
            'horizons': list(horizons),
            'limit': args.limit,
            'causal_alignment': 'action[t] -> outcome[t+1]',
            'early_switch_without_event': 'censored',
        },
        'scenes': scenes,
        'combined': combined,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(f'[done] output={args.output}', flush=True)


if __name__ == '__main__':
    main()
