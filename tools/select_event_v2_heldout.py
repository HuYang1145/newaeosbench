#!/usr/bin/env python3
"""按固定 held-out train Q 选择 Event V2-2 checkpoint。"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
import pathlib
from typing import Any


METRIC_NAMES = ('CR', 'PCR', 'WCR', 'Q')


def _validated_summary(
    summary: Mapping[str, Any],
    *,
    expected_stage: str,
    expected_scene_ids: tuple[int, ...],
) -> dict[str, Any]:
    label = summary.get('label')
    if not isinstance(label, str) or not label:
        raise ValueError('held-out summary label is invalid')
    if summary.get('stage') != expected_stage:
        raise ValueError(f'{label} stage does not match {expected_stage}')
    if summary.get('split') != 'train':
        raise ValueError(f'{label} must use held-out train scenes')
    if tuple(summary.get('scene_ids', ())) != expected_scene_ids:
        raise ValueError(f'{label} scene IDs do not match held-out protocol')
    if summary.get('max_time_step') != 3600:
        raise ValueError(f'{label} max time does not match 3600 seconds')
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
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in metrics.values()):
        raise ValueError(f'{label} aggregate metrics are not finite rates')
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
    return {
        'label': label,
        'checkpoint': checkpoint,
        'stage': expected_stage,
        'aggregate': metrics,
    }


def select_heldout_summaries(
    *,
    baseline: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    expected_scene_ids: Sequence[int],
) -> dict[str, Any]:
    expected_scene_ids = tuple(int(value) for value in expected_scene_ids)
    if (
        not expected_scene_ids
        or len(set(expected_scene_ids)) != len(expected_scene_ids)
    ):
        raise ValueError('expected held-out scene IDs must be unique')
    if len(candidates) != 4:
        raise ValueError('held-out selection requires four V2-2 candidates')
    baseline_row = _validated_summary(
        baseline,
        expected_stage='V2-1',
        expected_scene_ids=expected_scene_ids,
    )
    candidate_rows = [
        _validated_summary(
            summary,
            expected_stage='V2-2',
            expected_scene_ids=expected_scene_ids,
        )
        for summary in candidates
    ]
    labels = [row['label'] for row in candidate_rows]
    if len(set(labels)) != len(labels):
        raise ValueError('V2-2 candidate labels must be unique')
    ranking = sorted(
        candidate_rows,
        key=lambda row: (-row['aggregate']['Q'], row['label']),
    )
    selected = ranking[0]
    delta = {
        name: (
            selected['aggregate'][name]
            - baseline_row['aggregate'][name]
        )
        for name in METRIC_NAMES
    }
    return {
        'protocol': {
            'split': 'train',
            'scene_ids': list(expected_scene_ids),
            'max_time_step': 3600,
            'deterministic': True,
            'selection_metric': 'Q=0.6CR+0.2PCR+0.2WCR',
        },
        'baseline': baseline_row,
        'ranking': ranking,
        'selected': selected,
        'delta_vs_baseline': delta,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Select the best Event V2-2 held-out checkpoint',
    )
    parser.add_argument('--baseline', type=pathlib.Path, required=True)
    parser.add_argument(
        '--candidates',
        type=pathlib.Path,
        nargs=4,
        required=True,
    )
    parser.add_argument(
        '--expected-scene-ids',
        type=int,
        nargs='+',
        required=True,
    )
    parser.add_argument('--output', type=pathlib.Path, required=True)
    return parser.parse_args()


def _load(path: pathlib.Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise ValueError(f'summary root must be an object: {path}')
    return value


def main() -> None:
    args = parse_args()
    selection = select_heldout_summaries(
        baseline=_load(args.baseline),
        candidates=[_load(path) for path in args.candidates],
        expected_scene_ids=args.expected_scene_ids,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + '\n',
    )
    print(json.dumps(selection, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
