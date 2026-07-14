"""汇总同一场景的多条模型候选轨迹，并构造偏好对。"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import torch

from constellation.new_transformers.offline_critic import (
    compute_cs_paper_from_metrics,
)


def _action_signature(path: Path) -> str:
    trajectory = torch.load(path, map_location='cpu', weights_only=False)
    task_ids = torch.as_tensor(trajectory['actions']['task_id']).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(task_ids.shape)).encode())
    digest.update(str(task_ids.dtype).encode())
    digest.update(task_ids.numpy().tobytes())
    return digest.hexdigest()


def _safe_cost(metrics: dict[str, float]) -> tuple[float | None, str | None]:
    try:
        cost = compute_cs_paper_from_metrics(metrics)
    except (KeyError, TypeError, ValueError) as error:
        return None, str(error)
    if not math.isfinite(cost):
        return None, 'CS_paper is not finite'
    return cost, None


def summarize_candidates(
    root: Path,
    *,
    split: str,
    greedy_candidate: str,
    min_cost_margin: float = 1e-6,
) -> dict[str, Any]:
    """按 scene 聚合候选，只为有效且动作不同的轨迹构造偏好对。"""

    if min_cost_margin < 0:
        raise ValueError('min_cost_margin must be non-negative')
    grouped: dict[int, list[dict[str, Any]]] = {}
    for candidate_root in sorted(root.glob('candidate_*')):
        candidate = candidate_root.name
        for metrics_path in sorted((candidate_root / split).rglob('*.json')):
            scene_id = int(metrics_path.stem)
            trajectory_path = metrics_path.with_suffix('.pth')
            metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
            cost, error = _safe_cost(metrics)
            action_signature = None
            if not trajectory_path.is_file():
                error = 'trajectory file is missing'
                cost = None
            else:
                action_signature = _action_signature(trajectory_path)
            grouped.setdefault(scene_id, []).append({
                'candidate': candidate,
                'valid': cost is not None,
                'cost': cost,
                'error': error,
                'action_signature': action_signature,
                'metrics': metrics,
                'metrics_path': str(metrics_path),
                'trajectory_path': str(trajectory_path),
            })

    scenes = []
    preference_pairs = []
    improvements = []
    num_scenes_with_action_diversity = 0
    num_scenes_with_preference = 0
    for scene_id, candidates in sorted(grouped.items()):
        candidates.sort(key=lambda item: item['candidate'])
        valid = [item for item in candidates if item['valid']]

        # 同一动作序列只保留 cost 最低的代表，不制造伪偏好对。
        representatives: dict[str, dict[str, Any]] = {}
        for item in valid:
            signature = item['action_signature']
            previous = representatives.get(signature)
            if previous is None or (item['cost'], item['candidate']) < (
                previous['cost'], previous['candidate'],
            ):
                representatives[signature] = item
        unique_actions = sorted(
            representatives.values(),
            key=lambda item: item['candidate'],
        )
        if len(unique_actions) >= 2:
            num_scenes_with_action_diversity += 1

        scene_pairs = []
        for left, right in itertools.combinations(unique_actions, 2):
            margin = abs(left['cost'] - right['cost'])
            if margin <= min_cost_margin:
                continue
            better, worse = (
                (left, right) if left['cost'] < right['cost'] else (right, left)
            )
            pair = {
                'scene_id': scene_id,
                'better_candidate': better['candidate'],
                'worse_candidate': worse['candidate'],
                'better_cost': better['cost'],
                'worse_cost': worse['cost'],
                'cost_margin': margin,
                'better_trajectory_path': better['trajectory_path'],
                'worse_trajectory_path': worse['trajectory_path'],
            }
            scene_pairs.append(pair)
            preference_pairs.append(pair)
        if scene_pairs:
            num_scenes_with_preference += 1

        greedy = next(
            (item for item in valid if item['candidate'] == greedy_candidate),
            None,
        )
        best = min(valid, key=lambda item: (item['cost'], item['candidate'])) \
            if valid else None
        improvement = None
        if greedy is not None and best is not None:
            improvement = greedy['cost'] - best['cost']
            improvements.append(improvement)
        scenes.append({
            'scene_id': scene_id,
            'num_candidates': len(candidates),
            'num_valid_candidates': len(valid),
            'num_distinct_actions': len(unique_actions),
            'num_preference_pairs': len(scene_pairs),
            'greedy_cost': None if greedy is None else greedy['cost'],
            'best_candidate': None if best is None else best['candidate'],
            'best_cost': None if best is None else best['cost'],
            'best_improvement_vs_greedy': improvement,
            'candidates': candidates,
        })

    return {
        'score_definition': (
            'CS_paper = (0.6*CR + 0.2*PCR + 0.2*WCR)^(-1) '
            '+ TAT_s/700 + PC_Wh/100'
        ),
        'root': str(root),
        'split': split,
        'greedy_candidate': greedy_candidate,
        'num_scenes': len(scenes),
        'num_scenes_with_action_diversity': num_scenes_with_action_diversity,
        'num_scenes_with_preference': num_scenes_with_preference,
        'num_candidate_pairs': len(preference_pairs),
        'mean_best_improvement_vs_greedy': (
            None if not improvements else sum(improvements) / len(improvements)
        ),
        'scenes': scenes,
        'preference_pairs': preference_pairs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Summarize same-scene model candidate trajectories',
    )
    parser.add_argument('root', type=Path)
    parser.add_argument('--split', default='train')
    parser.add_argument('--greedy-candidate', default='candidate_000_greedy')
    parser.add_argument('--min-cost-margin', type=float, default=1e-6)
    parser.add_argument('--output', type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_candidates(
        args.root,
        split=args.split,
        greedy_candidate=args.greedy_candidate,
        min_cost_margin=args.min_cost_margin,
    )
    output = args.output or args.root / 'summary.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    print(json.dumps({
        key: summary[key] for key in (
            'num_scenes',
            'num_scenes_with_action_diversity',
            'num_scenes_with_preference',
            'num_candidate_pairs',
            'mean_best_improvement_vs_greedy',
        )
    }, indent=2, ensure_ascii=False))
    print(f'[done] summary={output}')


if __name__ == '__main__':
    main()
