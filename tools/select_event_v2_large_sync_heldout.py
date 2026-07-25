#!/usr/bin/env python3
"""只用固定 train-heldout 对大规模同步 PPO checkpoint 排序。"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
import os
import pathlib
import re
from typing import Any

from constellation.new_transformers.event_v2.large_sync_checkpoint import (
    update_latest_checkpoint,
)


METRIC_NAMES = ('CR', 'PCR', 'WCR', 'Q')
_CANDIDATE_PATTERN = re.compile(
    r'^seed_(?P<seed>\d+).*update_(?P<update>\d+)(?:_.*)?$',
)


def _validate_summary(
    summary: Mapping[str, Any],
    *,
    expected_stage: str,
    expected_scene_ids: tuple[int, ...],
    candidate: bool,
) -> dict[str, Any]:
    label = summary.get('label')
    if not isinstance(label, str) or not label:
        raise ValueError('large heldout summary label is invalid')
    if summary.get('stage') != expected_stage:
        raise ValueError(f'{label} stage does not match {expected_stage}')
    if summary.get('split') != 'train':
        raise ValueError(f'{label} must use held-out train scenes')
    if tuple(summary.get('scene_ids', ())) != expected_scene_ids:
        raise ValueError(f'{label} scene IDs do not match held-out protocol')
    if summary.get('max_time_step') != 3600:
        raise ValueError(f'{label} max time must be 3600 seconds')
    if summary.get('deterministic') is not True:
        raise ValueError(f'{label} evaluation must be deterministic')
    if summary.get('finite') is not True:
        raise ValueError(f'{label} evaluation is not finite')
    reward_error = float(
        summary.get('reward_reconstruction_max_error', math.inf),
    )
    if not math.isfinite(reward_error) or reward_error > 1e-6:
        raise ValueError(f'{label} reward reconstruction failed')
    aggregate = summary.get('aggregate')
    if not isinstance(aggregate, Mapping):
        raise ValueError(f'{label} aggregate metrics are missing')
    metrics = {
        name: float(aggregate.get(name, math.nan))
        for name in METRIC_NAMES
    }
    if not all(
        math.isfinite(value) and 0 <= value <= 1
        for value in metrics.values()
    ):
        raise ValueError(f'{label} aggregate metrics are invalid')
    expected_q = (
        0.6 * metrics['CR']
        + 0.2 * metrics['PCR']
        + 0.2 * metrics['WCR']
    )
    if abs(metrics['Q'] - expected_q) > 1e-9:
        raise ValueError(f'{label} Q does not match registered formula')
    checkpoint = summary.get('checkpoint')
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError(f'{label} checkpoint path is missing')
    update = int(summary.get('checkpoint_updates', -1))
    if update < 0:
        raise ValueError(f'{label} checkpoint update is invalid')
    row = {
        'label': label,
        'checkpoint': checkpoint,
        'stage': expected_stage,
        'update': update,
        'aggregate': metrics,
    }
    if not candidate:
        return row
    match = _CANDIDATE_PATTERN.fullmatch(label)
    if match is None:
        raise ValueError(
            f'{label} must encode seed and update in its label',
        )
    label_update = int(match.group('update'))
    if label_update != update:
        raise ValueError(
            f'{label} checkpoint update does not match its label',
        )
    row['seed'] = int(match.group('seed'))
    return row


def select_large_sync_heldout_summaries(
    *,
    baseline: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    expected_scene_ids: Sequence[int],
) -> dict[str, Any]:
    """按 Q、最小单项提升、较早 update、较小 seed 依次排序。"""

    scene_ids = tuple(int(value) for value in expected_scene_ids)
    if (
        scene_ids != tuple(range(196, 204))
        or len(scene_ids) != len(set(scene_ids))
    ):
        raise ValueError(
            'large heldout selection must use train scenes 196-203',
        )
    if len(candidates) < 2:
        raise ValueError(
            'large heldout selection requires at least two candidates',
        )
    baseline_row = _validate_summary(
        baseline,
        expected_stage='V2-2',
        expected_scene_ids=scene_ids,
        candidate=False,
    )
    candidate_rows = [
        _validate_summary(
            summary,
            expected_stage='V2-2-Large',
            expected_scene_ids=scene_ids,
            candidate=True,
        )
        for summary in candidates
    ]
    labels = tuple(row['label'] for row in candidate_rows)
    checkpoints = tuple(row['checkpoint'] for row in candidate_rows)
    if len(labels) != len(set(labels)):
        raise ValueError('large heldout candidate labels must be unique')
    if len(checkpoints) != len(set(checkpoints)):
        raise ValueError(
            'large heldout candidate checkpoints must be unique',
        )
    for row in candidate_rows:
        row['delta_vs_baseline'] = {
            name: (
                row['aggregate'][name]
                - baseline_row['aggregate'][name]
            )
            for name in METRIC_NAMES
        }
        row['minimum_metric_delta'] = min(
            row['delta_vs_baseline'][name]
            for name in ('CR', 'PCR', 'WCR')
        )
    ranking = sorted(
        candidate_rows,
        key=lambda row: (
            -row['aggregate']['Q'],
            -row['minimum_metric_delta'],
            row['update'],
            row['seed'],
            row['label'],
        ),
    )
    selected = ranking[0]
    return {
        'protocol': {
            'split': 'train',
            'scene_ids': list(scene_ids),
            'max_time_step': 3600,
            'deterministic': True,
            'selection_metric': 'Q=0.6CR+0.2PCR+0.2WCR',
            'tie_break': [
                'minimum_CR_PCR_WCR_delta',
                'earlier_update',
                'smaller_seed',
            ],
        },
        'baseline': baseline_row,
        'ranking': ranking,
        'selected': selected,
        'delta_vs_baseline': dict(selected['delta_vs_baseline']),
    }


def write_selection_artifacts(
    selection: Mapping[str, Any],
    *,
    output: pathlib.Path,
    best_link: pathlib.Path,
) -> None:
    """原子写 selection，并用 hard link 锁定最佳 checkpoint。"""

    selected = selection.get('selected')
    if not isinstance(selected, Mapping):
        raise ValueError('large heldout selection has no selected row')
    checkpoint = selected.get('checkpoint')
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError('large heldout selected checkpoint is invalid')
    source = pathlib.Path(checkpoint)
    if not source.is_file():
        raise FileNotFoundError(
            f'large heldout selected checkpoint not found: {source}',
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    try:
        temporary.write_text(
            json.dumps(selection, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    update_latest_checkpoint(source=source, latest=best_link)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Select a large strict-sync PPO heldout checkpoint',
    )
    parser.add_argument('--baseline', type=pathlib.Path, required=True)
    parser.add_argument(
        '--candidates',
        type=pathlib.Path,
        nargs='+',
        required=True,
    )
    parser.add_argument(
        '--expected-scene-ids',
        type=int,
        nargs=8,
        required=True,
    )
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--best-link', type=pathlib.Path, required=True)
    return parser.parse_args()


def _load(path: pathlib.Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, Mapping):
        raise ValueError(f'summary root must be an object: {path}')
    return value


def main() -> None:
    args = parse_args()
    selection = select_large_sync_heldout_summaries(
        baseline=_load(args.baseline),
        candidates=[_load(path) for path in args.candidates],
        expected_scene_ids=args.expected_scene_ids,
    )
    write_selection_artifacts(
        selection,
        output=args.output,
        best_link=args.best_link,
    )
    print(json.dumps(selection, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
