#!/usr/bin/env python3
"""验证锁定的 Event V2 checkpoint 是否通过完整 Val 64+64。"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import pathlib
from typing import Any

from tools.compare_event_v2_val_gate import METRIC_NAMES, _validate


def compare_full_val(
    *,
    baseline_seen: Mapping[str, Any],
    candidate_seen: Mapping[str, Any],
    baseline_unseen: Mapping[str, Any],
    candidate_unseen: Mapping[str, Any],
    expected_scene_ids: Sequence[int],
    minimum_q_improvement: float = 0.005,
    baseline_stage: str = 'V2-2',
    candidate_stage: str = 'V2-2-Large',
) -> dict[str, Any]:
    scene_ids = tuple(int(value) for value in expected_scene_ids)
    if scene_ids != tuple(range(64)):
        raise ValueError('full Val must use exactly scenes 0-63')
    if minimum_q_improvement <= 0:
        raise ValueError('minimum Q improvement must be positive')
    raw = {
        'val_seen': (baseline_seen, candidate_seen),
        'val_unseen': (baseline_unseen, candidate_unseen),
    }
    split_results = {}
    for split, (baseline_summary, candidate_summary) in raw.items():
        baseline = _validate(
            baseline_summary,
            stage=baseline_stage,
            split=split,
            scene_ids=scene_ids,
        )
        candidate = _validate(
            candidate_summary,
            stage=candidate_stage,
            split=split,
            scene_ids=scene_ids,
        )
        delta = {
            name: (
                candidate['aggregate'][name]
                - baseline['aggregate'][name]
            )
            for name in METRIC_NAMES
        }
        q_threshold_met = (
            delta['Q'] + 1e-12 >= minimum_q_improvement
        )
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
            'recorded_only': {
                'baseline': {
                    name: baseline_summary['aggregate'].get(name)
                    for name in ('TAT_s', 'PC_Wh', 'CS_paper')
                },
                'candidate': {
                    name: candidate_summary['aggregate'].get(name)
                    for name in ('TAT_s', 'PC_Wh', 'CS_paper')
                },
            },
        }
    return {
        'protocol': {
            'scene_ids': list(scene_ids),
            'max_time_step': 3600,
            'deterministic': True,
            'minimum_q_improvement': minimum_q_improvement,
            'metrics_must_not_decrease': ['CR', 'PCR', 'WCR'],
            'baseline_stage': baseline_stage,
            'candidate_stage': candidate_stage,
            'recorded_not_selected': [
                'TAT_s',
                'PC_Wh',
                'CS_paper',
            ],
        },
        'splits': split_results,
        'passed': all(
            row['passed'] for row in split_results.values()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compare V2-2 and V2-2-Large on full Val 64+64',
    )
    parser.add_argument('--baseline-seen', type=pathlib.Path, required=True)
    parser.add_argument('--candidate-seen', type=pathlib.Path, required=True)
    parser.add_argument('--baseline-unseen', type=pathlib.Path, required=True)
    parser.add_argument('--candidate-unseen', type=pathlib.Path, required=True)
    parser.add_argument(
        '--expected-scene-ids',
        type=int,
        nargs=64,
        required=True,
    )
    parser.add_argument('--minimum-q-improvement', type=float, default=0.005)
    parser.add_argument('--baseline-stage', default='V2-2')
    parser.add_argument('--candidate-stage', default='V2-2-Large')
    parser.add_argument('--output', type=pathlib.Path, required=True)
    return parser.parse_args()


def _load(path: pathlib.Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, Mapping):
        raise ValueError(f'full Val summary must be an object: {path}')
    return value


def main() -> None:
    args = parse_args()
    result = compare_full_val(
        baseline_seen=_load(args.baseline_seen),
        candidate_seen=_load(args.candidate_seen),
        baseline_unseen=_load(args.baseline_unseen),
        candidate_unseen=_load(args.candidate_unseen),
        expected_scene_ids=args.expected_scene_ids,
        minimum_q_improvement=args.minimum_q_improvement,
        baseline_stage=args.baseline_stage,
        candidate_stage=args.candidate_stage,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    if not result['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
