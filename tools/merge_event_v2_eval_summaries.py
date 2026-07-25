#!/usr/bin/env python3
"""合并同一 Event V2 checkpoint 的互斥 scene 评估分片。"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
import pathlib
from typing import Any

from tools.evaluate_event_v2_policy import aggregate_scene_metrics


_IDENTITY_FIELDS = (
    'label',
    'checkpoint',
    'stage',
    'checkpoint_updates',
    'checkpoint_policy_version',
    'checkpoint_train_scene_ids',
    'config_fingerprint',
    'split',
    'max_time_step',
    'deterministic',
    'amp_enabled',
    'amp_dtype',
)


def merge_eval_summaries(
    summaries: Sequence[Mapping[str, Any]],
    *,
    expected_scene_ids: Sequence[int],
) -> dict[str, Any]:
    """按 scene id 排序后重新计算 macro 指标，禁止平均 summary 均值。"""

    if not summaries:
        raise ValueError('eval merge requires at least one summary')
    expected = tuple(int(value) for value in expected_scene_ids)
    if not expected or len(expected) != len(set(expected)):
        raise ValueError('eval merge expected scene IDs are invalid')
    reference = summaries[0]
    rows: list[Mapping[str, Any]] = []
    reward_errors: list[float] = []
    for summary in summaries:
        if not isinstance(summary, Mapping):
            raise ValueError('eval merge summary must be an object')
        for field in _IDENTITY_FIELDS:
            if summary.get(field) != reference.get(field):
                readable = field.replace('_', ' ')
                raise ValueError(
                    f'eval merge {readable} does not match',
                )
        if summary.get('max_time_step') != 3600:
            raise ValueError('eval merge max time must be 3600 seconds')
        if summary.get('deterministic') is not True:
            raise ValueError('eval merge evaluation must be deterministic')
        if summary.get('finite') is not True:
            raise ValueError('eval merge summary is not finite')
        error = float(
            summary.get(
                'reward_reconstruction_max_error',
                math.inf,
            ),
        )
        if not math.isfinite(error) or error > 1e-6:
            raise ValueError('eval merge reward reconstruction failed')
        reward_errors.append(error)
        summary_scene_ids = tuple(summary.get('scene_ids', ()))
        scene_rows = summary.get('scenes')
        if (
            not isinstance(scene_rows, Sequence)
            or tuple(row.get('scene_id') for row in scene_rows)
            != summary_scene_ids
        ):
            raise ValueError('eval merge scene rows do not match scene IDs')
        rows.extend(scene_rows)
    row_by_scene = {
        int(row['scene_id']): row
        for row in rows
    }
    if (
        len(row_by_scene) != len(rows)
        or set(row_by_scene) != set(expected)
    ):
        raise ValueError(
            'eval merge scene coverage is missing or duplicated',
        )
    ordered_rows = [row_by_scene[scene_id] for scene_id in expected]
    aggregate = aggregate_scene_metrics(ordered_rows)
    result = {
        field: reference.get(field)
        for field in _IDENTITY_FIELDS
    }
    result.update({
        'scene_ids': list(expected),
        'scenes': ordered_rows,
        'aggregate': aggregate,
        'finite': all(
            math.isfinite(float(value))
            for row in ordered_rows
            for name, value in row.items()
            if name in {
                'CR',
                'PCR',
                'WCR',
                'Q',
                'TAT_s',
                'PC_Wh',
                'CS_paper',
            }
        ),
        'reward_reconstruction_max_error': max(reward_errors),
        'merged_shards': len(summaries),
    })
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Merge disjoint Event V2 evaluation summaries',
    )
    parser.add_argument(
        '--inputs',
        type=pathlib.Path,
        nargs='+',
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
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, Mapping):
        raise ValueError(f'eval summary root must be an object: {path}')
    return value


def main() -> None:
    args = parse_args()
    merged = merge_eval_summaries(
        [_load(path) for path in args.inputs],
        expected_scene_ids=args.expected_scene_ids,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(merged, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
