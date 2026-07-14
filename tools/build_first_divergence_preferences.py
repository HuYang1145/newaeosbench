"""把同场景候选偏好转换为第一动作分歧点数据集。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
from typing import Any

import torch

from constellation.new_transformers.preference_divergence import (
    build_first_divergence_record,
)


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            'count': 0,
            'mean': None,
            'min': None,
            'p25': None,
            'median': None,
            'p75': None,
            'max': None,
        }
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        'count': len(values),
        'mean': statistics.fmean(values),
        'min': min(values),
        'p25': float(torch.quantile(tensor, 0.25).item()),
        'median': statistics.median(values),
        'p75': float(torch.quantile(tensor, 0.75).item()),
        'max': max(values),
    }


def build_divergence_dataset(
    summary_path: Path,
    *,
    min_cost_margin: float = 0.05,
) -> dict[str, object]:
    """逐场景读取轨迹，避免把全部 ``is_visible`` 张量留在内存。"""

    if min_cost_margin < 0:
        raise ValueError('minimum cost margin must be non-negative')
    source = json.loads(summary_path.read_text(encoding='utf-8'))
    source_pairs = source['preference_pairs']
    pairs_by_scene: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for pair in source_pairs:
        pairs_by_scene[int(pair['scene_id'])].append(pair)

    records = []
    num_identical = 0
    for scene_id in sorted(pairs_by_scene):
        scene_pairs = pairs_by_scene[scene_id]
        paths = sorted({
            str(pair[key])
            for pair in scene_pairs
            for key in ('better_trajectory_path', 'worse_trajectory_path')
        })
        trajectories = {
            path: torch.load(path, map_location='cpu', weights_only=False)
            for path in paths
        }
        for pair in scene_pairs:
            better_path = str(pair['better_trajectory_path'])
            worse_path = str(pair['worse_trajectory_path'])
            record = build_first_divergence_record(
                scene_id=scene_id,
                better_candidate=str(pair['better_candidate']),
                worse_candidate=str(pair['worse_candidate']),
                better_cost=float(pair['better_cost']),
                worse_cost=float(pair['worse_cost']),
                better_trajectory_path=better_path,
                worse_trajectory_path=worse_path,
                better_trajectory=trajectories[better_path],
                worse_trajectory=trajectories[worse_path],
            )
            if record is None:
                num_identical += 1
                continue
            record['usable_for_graph_q'] = bool(
                record['shared_state_match']
                and float(record['cost_margin']) >= min_cost_margin
            )
            records.append(record)

    usable = [record for record in records if record['usable_for_graph_q']]
    divergence_indices = [
        float(record['divergence_index']) for record in usable
    ]
    divergence_fractions = [
        float(record['divergence_fraction']) for record in usable
    ]
    margins = [float(record['cost_margin']) for record in usable]
    changed_satellites = [
        float(record['changed_satellites']) for record in usable
    ]
    duplicate_delta = [
        int(record['better_action_summary']['duplicate_assignments'])
        - int(record['worse_action_summary']['duplicate_assignments'])
        for record in usable
    ]
    unique_task_delta = [
        int(record['better_action_summary']['unique_tasks'])
        - int(record['worse_action_summary']['unique_tasks'])
        for record in usable
    ]
    return {
        'purpose': 'P1 同场景第一分歧点偏好数据集',
        'source_summary': str(summary_path),
        'score_definition': source.get('score_definition'),
        'min_cost_margin': min_cost_margin,
        'input_contract': {
            'uses_current_state_only': True,
            'keeps_exact_joint_actions': True,
            'uses_is_visible_as_input': False,
            'basilisk_online_inference': False,
        },
        'summary': {
            'num_scenes': len(pairs_by_scene),
            'num_source_pairs': len(source_pairs),
            'num_divergence_records': len(records),
            'num_identical_action_pairs': num_identical,
            'num_shared_state_records': sum(
                bool(record['shared_state_match']) for record in records
            ),
            'num_unreconstructable_initial_state_records': sum(
                not bool(record['current_state_reconstructable'])
                for record in records
            ),
            'num_reconstructable_state_mismatch_records': sum(
                bool(record['current_state_reconstructable'])
                and not bool(record['shared_state_match'])
                for record in records
            ),
            'num_below_margin_records': sum(
                float(record['cost_margin']) < min_cost_margin
                for record in records
            ),
            'num_usable_records': len(usable),
            'divergence_index': _distribution(divergence_indices),
            'divergence_fraction': _distribution(divergence_fractions),
            'cost_margin': _distribution(margins),
            'changed_satellites': _distribution(changed_satellites),
            'better_has_fewer_duplicates': sum(
                delta < 0 for delta in duplicate_delta
            ),
            'same_duplicate_count': sum(delta == 0 for delta in duplicate_delta),
            'better_has_more_duplicates': sum(
                delta > 0 for delta in duplicate_delta
            ),
            'better_covers_more_unique_tasks': sum(
                delta > 0 for delta in unique_task_delta
            ),
            'same_unique_task_count': sum(
                delta == 0 for delta in unique_task_delta
            ),
            'better_covers_fewer_unique_tasks': sum(
                delta < 0 for delta in unique_task_delta
            ),
        },
        'records': records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build first-divergence preferences from same-scene rollouts',
    )
    parser.add_argument('summary', type=Path)
    parser.add_argument('--min-cost-margin', type=float, default=0.05)
    parser.add_argument(
        '--output', type=Path,
        default=Path(
            'work_dirs/same_scene_preference_critic_64/'
            'first_divergence_preferences.json'
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_divergence_dataset(
        args.summary,
        min_cost_margin=args.min_cost_margin,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))
    print(f'[done] output={args.output}')


if __name__ == '__main__':
    main()
