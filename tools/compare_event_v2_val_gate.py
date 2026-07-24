#!/usr/bin/env python3
"""验证 Event V2-2 是否通过预注册 Val 8+8 完成率门槛。"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
import pathlib
from typing import Any


METRIC_NAMES = ('CR', 'PCR', 'WCR', 'Q')


def _validate(
    summary: Mapping[str, Any],
    *,
    stage: str,
    split: str,
    scene_ids: tuple[int, ...],
) -> dict[str, Any]:
    if summary.get('stage') != stage:
        raise ValueError(f'{split} checkpoint stage does not match {stage}')
    if summary.get('split') != split:
        raise ValueError(f'{split} summary split does not match')
    if tuple(summary.get('scene_ids', ())) != scene_ids:
        raise ValueError(f'{split} scene IDs do not match Val 8 protocol')
    if summary.get('max_time_step') != 3600:
        raise ValueError(f'{split} max time does not match')
    if summary.get('deterministic') is not True:
        raise ValueError(f'{split} evaluation must be deterministic')
    if summary.get('finite') is not True:
        raise ValueError(f'{split} evaluation is not finite')
    if float(summary.get('reward_reconstruction_max_error', math.inf)) > 1e-6:
        raise ValueError(f'{split} reward reconstruction failed')
    aggregate = summary.get('aggregate')
    if not isinstance(aggregate, Mapping):
        raise ValueError(f'{split} aggregate metrics are missing')
    metrics = {
        name: float(aggregate.get(name, math.nan))
        for name in METRIC_NAMES
    }
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in metrics.values()):
        raise ValueError(f'{split} metrics are invalid')
    expected_q = (
        0.6 * metrics['CR']
        + 0.2 * metrics['PCR']
        + 0.2 * metrics['WCR']
    )
    if abs(metrics['Q'] - expected_q) > 1e-9:
        raise ValueError(f'{split} Q formula does not match')
    return {
        'label': summary.get('label'),
        'checkpoint': summary.get('checkpoint'),
        'aggregate': metrics,
    }


def compare_val_gate(
    *,
    baseline_seen: Mapping[str, Any],
    candidate_seen: Mapping[str, Any],
    baseline_unseen: Mapping[str, Any],
    candidate_unseen: Mapping[str, Any],
    expected_scene_ids: Sequence[int],
    minimum_q_improvement: float = 0.005,
) -> dict[str, Any]:
    scene_ids = tuple(int(value) for value in expected_scene_ids)
    if len(scene_ids) != 8 or len(set(scene_ids)) != 8:
        raise ValueError('Val gate requires eight unique scene IDs')
    if minimum_q_improvement <= 0:
        raise ValueError('minimum Q improvement must be positive')
    summaries = {
        'val_seen': (
            _validate(
                baseline_seen,
                stage='V2-1',
                split='val_seen',
                scene_ids=scene_ids,
            ),
            _validate(
                candidate_seen,
                stage='V2-2',
                split='val_seen',
                scene_ids=scene_ids,
            ),
        ),
        'val_unseen': (
            _validate(
                baseline_unseen,
                stage='V2-1',
                split='val_unseen',
                scene_ids=scene_ids,
            ),
            _validate(
                candidate_unseen,
                stage='V2-2',
                split='val_unseen',
                scene_ids=scene_ids,
            ),
        ),
    }
    split_results = {}
    for split, (baseline, candidate) in summaries.items():
        delta = {
            name: (
                candidate['aggregate'][name]
                - baseline['aggregate'][name]
            )
            for name in METRIC_NAMES
        }
        q_threshold_met = delta['Q'] + 1e-12 >= minimum_q_improvement
        metrics_non_decreasing = all(
            delta[name] >= -1e-12
            for name in ('CR', 'PCR', 'WCR')
        )
        split_results[split] = {
            'baseline': baseline,
            'candidate': candidate,
            'delta': delta,
            'q_threshold_met': q_threshold_met,
            'metrics_non_decreasing': metrics_non_decreasing,
            'passed': q_threshold_met and metrics_non_decreasing,
        }
    return {
        'protocol': {
            'scene_ids': list(scene_ids),
            'max_time_step': 3600,
            'deterministic': True,
            'minimum_q_improvement': minimum_q_improvement,
            'metrics_must_not_decrease': ['CR', 'PCR', 'WCR'],
        },
        'splits': split_results,
        'passed': all(row['passed'] for row in split_results.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compare Event V2 baseline and candidate on Val 8+8',
    )
    parser.add_argument('--baseline-seen', type=pathlib.Path, required=True)
    parser.add_argument('--candidate-seen', type=pathlib.Path, required=True)
    parser.add_argument('--baseline-unseen', type=pathlib.Path, required=True)
    parser.add_argument('--candidate-unseen', type=pathlib.Path, required=True)
    parser.add_argument('--expected-scene-ids', type=int, nargs=8, required=True)
    parser.add_argument('--minimum-q-improvement', type=float, default=0.005)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    return parser.parse_args()


def _load(path: pathlib.Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise ValueError(f'summary root must be an object: {path}')
    return value


def main() -> None:
    args = parse_args()
    result = compare_val_gate(
        baseline_seen=_load(args.baseline_seen),
        candidate_seen=_load(args.candidate_seen),
        baseline_unseen=_load(args.baseline_unseen),
        candidate_unseen=_load(args.candidate_unseen),
        expected_scene_ids=args.expected_scene_ids,
        minimum_q_improvement=args.minimum_q_improvement,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + '\n',
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    if not result['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
