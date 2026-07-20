#!/usr/bin/env python3
"""审计 M2 事件持续标签与短窗口事实结果覆盖。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import torch

from constellation.new_transformers.multi_horizon_edge_labels import (
    aggregate_edge_label_summaries,
    build_event_supervision,
    summarize_trajectory_edge_labels,
)
from tools.audit_multi_horizon_edge_labels import (
    AuditTarget,
    annotation_targets,
)


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def summarize_event_targets(
    actions: torch.Tensor,
    *,
    commitments: Sequence[int] = (1, 5, 15, 30, 60),
) -> dict[str, Any]:
    """汇总单条轨迹的 continue、duration 和 censor 标签。"""
    normalized = tuple(int(value) for value in commitments)
    targets = build_event_supervision(actions, normalized)
    valid = targets.valid
    observed = valid & targets.duration_observed
    edge_count = int(valid.sum())
    continue_count = int((targets.continue_next & valid).sum())
    duration_counts = {
        str(commitment): int(
            (
                observed
                & (targets.duration_index == index)
            ).sum()
        )
        for index, commitment in enumerate(normalized)
    }
    duration_observed_count = int(observed.sum())
    duration_censored_count = int((valid & ~observed).sum())
    return {
        'edge_count': edge_count,
        'continue_count': continue_count,
        'stop_count': edge_count - continue_count,
        'duration_observed_count': duration_observed_count,
        'duration_censored_count': duration_censored_count,
        'duration_counts': duration_counts,
        'rates': {
            'continue_rate': _ratio(continue_count, edge_count),
            'duration_censored_rate': _ratio(
                duration_censored_count,
                edge_count,
            ),
        },
    }


def aggregate_event_summaries(
    summaries: Sequence[dict[str, Any]],
    *,
    commitments: Sequence[int] = (1, 5, 15, 30, 60),
) -> dict[str, Any]:
    """按原始计数聚合场景并重新计算比例和建议类别权重。"""
    if not summaries:
        raise ValueError('at least one event summary is required')
    normalized = tuple(int(value) for value in commitments)
    edge_count = sum(int(summary['edge_count']) for summary in summaries)
    continue_count = sum(
        int(summary['continue_count']) for summary in summaries
    )
    observed_count = sum(
        int(summary['duration_observed_count']) for summary in summaries
    )
    censored_count = sum(
        int(summary['duration_censored_count']) for summary in summaries
    )
    duration_counts = {
        str(commitment): sum(
            int(summary['duration_counts'][str(commitment)])
            for summary in summaries
        )
        for commitment in normalized
    }
    class_counts = [duration_counts[str(value)] for value in normalized]
    duration_class_weights = None
    if all(count > 0 for count in class_counts):
        inverse = [observed_count / count for count in class_counts]
        mean = sum(inverse) / len(inverse)
        duration_class_weights = [value / mean for value in inverse]
    stop_count = edge_count - continue_count
    return {
        'scene_count': len(summaries),
        'edge_count': edge_count,
        'continue_count': continue_count,
        'stop_count': stop_count,
        'duration_observed_count': observed_count,
        'duration_censored_count': censored_count,
        'duration_counts': duration_counts,
        'suggested_weights': {
            'continue_positive_weight': _ratio(
                stop_count,
                continue_count,
            ),
            'duration_class_weights': duration_class_weights,
        },
        'rates': {
            'continue_rate': _ratio(continue_count, edge_count),
            'duration_censored_rate': _ratio(censored_count, edge_count),
        },
    }


def _task_durations(path: Path, *, num_tasks: int) -> torch.Tensor:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise TypeError(f'taskset must be a list: {path}')
    by_id = {int(task['id']): int(task['duration']) for task in payload}
    if set(by_id) != set(range(num_tasks)):
        raise ValueError(f'task ids do not align with trajectory: {path}')
    return torch.tensor([by_id[index] for index in range(num_tasks)])


def _audit_target_job(
    args: tuple[AuditTarget, tuple[int, ...], tuple[int, ...]],
) -> dict[str, Any]:
    target, commitments, horizons = args
    payload = torch.load(
        target.trajectory_path,
        map_location='cpu',
        weights_only=False,
    )
    actions = torch.as_tensor(payload['actions']['task_id']).long()
    progress = torch.as_tensor(payload['taskset']['progress'])
    event_summary = summarize_event_targets(
        actions,
        commitments=commitments,
    )
    outcome_summary = summarize_trajectory_edge_labels(
        actions=actions,
        is_visible=torch.as_tensor(payload['is_visible']).bool(),
        progress=progress,
        task_durations=_task_durations(
            target.taskset_path,
            num_tasks=progress.shape[1],
        ),
        horizons=horizons,
    )
    return {
        'scene_id': target.scene_id,
        'epoch': target.epoch,
        'trajectory': str(target.trajectory_path),
        'taskset': str(target.taskset_path),
        'event': event_summary,
        'outcome': outcome_summary,
    }


def _worker_init() -> None:
    torch.set_num_threads(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotation-file', type=Path, required=True)
    parser.add_argument('--split', default='train')
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument(
        '--taskset-root',
        type=Path,
        default=Path('data/tasksets'),
    )
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--limit', type=int)
    parser.add_argument(
        '--commitments',
        type=int,
        nargs='+',
        default=[1, 5, 15, 30, 60],
    )
    parser.add_argument(
        '--horizons',
        type=int,
        nargs='+',
        default=[5, 15, 30, 60],
    )
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError('limit must be positive')
    if args.workers <= 0:
        raise ValueError('workers must be positive')
    commitments = tuple(int(value) for value in args.commitments)
    horizons = tuple(int(value) for value in args.horizons)
    targets = annotation_targets(
        args.annotation_file,
        split=args.split,
        data_root=args.data_root,
        taskset_root=args.taskset_root,
    )
    if args.limit is not None:
        targets = targets[:args.limit]

    jobs = [(target, commitments, horizons) for target in targets]
    executor = None
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
        if executor is not None:
            executor.shutdown()

    event_combined = aggregate_event_summaries(
        [scene['event'] for scene in scenes],
        commitments=commitments,
    )
    outcome_combined = aggregate_edge_label_summaries([
        scene['outcome'] for scene in scenes
    ])
    output = {
        'purpose': 'M2 event supervision audit; Actor unchanged',
        'config': {
            'annotation_file': str(args.annotation_file),
            'split': args.split,
            'data_root': str(args.data_root),
            'taskset_root': str(args.taskset_root),
            'workers': args.workers,
            'limit': args.limit,
            'commitments': list(commitments),
            'horizons': list(horizons),
            'causal_alignment': 'state[t-1], history[:t] -> action/outcome[t:]',
            'idle_duration': 'masked',
            'trajectory_end': 'censored below max commitment',
        },
        'routing': {
            'scene_count': len(targets),
            'unique_scene_count': len({target.scene_id for target in targets}),
            'epochs': dict(Counter(target.epoch for target in targets)),
        },
        'scenes': scenes,
        'event_combined': event_combined,
        'outcome_combined': outcome_combined,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(f'[done] output={args.output}', flush=True)


if __name__ == '__main__':
    main()
