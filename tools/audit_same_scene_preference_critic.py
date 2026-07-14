"""审计同场景裁判模型（Critic）的跨场景排序误差。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

import torch

from constellation.new_transformers.offline_critic import (
    ActionConditionedCritic,
    StateValueBaseline,
)
from constellation.new_transformers.preference_audit import (
    audit_pairwise_predictions,
)
from constellation.new_transformers.preference_critic import (
    CandidateTensors,
    PreferenceCriticBundle,
)
from tools.train_same_scene_preference_critic import (
    load_candidate_features,
    split_candidates_by_scene,
)


def load_saved_bundle(
    checkpoint_path: Path,
    *,
    state_dim: int,
    action_dim: int,
    hidden_dim: int,
) -> PreferenceCriticBundle:
    """恢复裁判模型及训练集归一化统计。"""

    payload = torch.load(
        checkpoint_path,
        map_location='cpu',
        weights_only=True,
    )
    baseline = StateValueBaseline(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
    )
    critic = ActionConditionedCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
    )
    baseline.load_state_dict(payload['baseline'])
    critic.load_state_dict(payload['critic'])
    return PreferenceCriticBundle(
        baseline=baseline,
        critic=critic,
        state_mean=payload['state_mean'],
        state_std=payload['state_std'],
        action_mean=payload['action_mean'],
        action_std=payload['action_std'],
    )


def _select(candidates: CandidateTensors, mask: torch.Tensor) -> CandidateTensors:
    return CandidateTensors(*(tensor[mask] for tensor in candidates))


def _candidate_exploration(payload: dict[str, Any]) -> dict[str, object]:
    scenes = payload['scenes']
    improvements = [
        float(scene['best_improvement_vs_greedy']) for scene in scenes
    ]
    outlier = max(
        scenes,
        key=lambda scene: float(scene['best_improvement_vs_greedy']),
    )
    without_max = improvements.copy()
    without_max.remove(max(without_max))
    return {
        'num_scenes': len(scenes),
        'num_candidate_pairs': int(payload['num_candidate_pairs']),
        'num_action_diverse_scenes': int(
            payload['num_scenes_with_action_diversity'],
        ),
        'sampled_beats_greedy_scenes': sum(value > 1e-12 for value in improvements),
        'mean_best_improvement_vs_greedy': statistics.fmean(improvements),
        'median_best_improvement_vs_greedy': statistics.median(improvements),
        'mean_without_max_outlier': statistics.fmean(without_max),
        'max_improvement_scene_id': int(outlier['scene_id']),
        'max_improvement': float(outlier['best_improvement_vs_greedy']),
    }


def audit_saved_folds(
    summary_path: Path,
    critic_root: Path,
    *,
    tasksets_root: Path,
    constellations_root: Path,
    samples_per_candidate: int,
    margin_thresholds: tuple[float, ...],
    device: torch.device,
) -> dict[str, object]:
    """把四折未见场景预测拼接成一次完整的 64 场审计。"""

    source = json.loads(summary_path.read_text(encoding='utf-8'))
    candidates, records = load_candidate_features(
        summary_path,
        tasksets_root=tasksets_root,
        constellations_root=constellations_root,
        samples_per_candidate=samples_per_candidate,
    )
    all_scene_ids = []
    all_names = []
    all_costs = []
    all_baseline = []
    all_critic = []
    fold_results = []
    fold_dirs = sorted(
        path for path in critic_root.glob('fold*')
        if (path / 'summary.json').is_file()
    )
    if not fold_dirs:
        raise FileNotFoundError(f'no completed folds found in {critic_root}')

    for fold_dir in fold_dirs:
        fold_summary = json.loads(
            (fold_dir / 'summary.json').read_text(encoding='utf-8'),
        )
        config = fold_summary['config']
        _, val, _, val_scene_ids = split_candidates_by_scene(
            candidates,
            val_fraction=float(config['val_fraction']),
            seed=int(config['seed']),
            num_folds=int(config['num_folds']),
            fold_index=int(config['fold_index']),
        )
        bundle = load_saved_bundle(
            fold_dir / 'critic.pth',
            state_dim=int(fold_summary['data']['state_dim']),
            action_dim=int(fold_summary['data']['action_dim']),
            hidden_dim=int(config['hidden_dim']),
        )
        baseline_prediction, critic_prediction = bundle.predict(
            val,
            device=device,
        )
        val_ids = set(val_scene_ids)
        val_records = [
            record for record in records
            if int(record['scene_id']) in val_ids
        ]
        names = [str(record['candidate']) for record in val_records]
        audit = audit_pairwise_predictions(
            scene_ids=val.scene_ids,
            candidate_names=names,
            costs=val.cost,
            baseline_predictions=baseline_prediction,
            critic_predictions=critic_prediction,
            margin_thresholds=margin_thresholds,
            greedy_candidate=str(source['greedy_candidate']),
        )
        fold_results.append({
            'fold_index': int(config['fold_index']),
            'validation_scene_ids': val_scene_ids,
            'audit': audit,
        })
        all_scene_ids.append(val.scene_ids)
        all_names.extend(names)
        all_costs.append(val.cost)
        all_baseline.append(baseline_prediction)
        all_critic.append(critic_prediction)

    combined_scene_ids = torch.cat(all_scene_ids)
    combined_costs = torch.cat(all_costs)
    combined_baseline = torch.cat(all_baseline)
    combined_critic = torch.cat(all_critic)
    overall = audit_pairwise_predictions(
        scene_ids=combined_scene_ids,
        candidate_names=all_names,
        costs=combined_costs,
        baseline_predictions=combined_baseline,
        critic_predictions=combined_critic,
        margin_thresholds=margin_thresholds,
        greedy_candidate=str(source['greedy_candidate']),
    )
    exploration = _candidate_exploration(source)
    outlier_scene_id = int(exploration['max_improvement_scene_id'])
    without_outlier_mask = combined_scene_ids != outlier_scene_id
    without_outlier_names = [
        name for name, keep in zip(all_names, without_outlier_mask.tolist())
        if keep
    ]
    without_outlier = audit_pairwise_predictions(
        scene_ids=combined_scene_ids[without_outlier_mask],
        candidate_names=without_outlier_names,
        costs=combined_costs[without_outlier_mask],
        baseline_predictions=combined_baseline[without_outlier_mask],
        critic_predictions=combined_critic[without_outlier_mask],
        margin_thresholds=margin_thresholds,
        greedy_candidate=str(source['greedy_candidate']),
    )
    return {
        'purpose': 'P0 裁判模型跨场景误差与候选选择审计',
        'source_summary': str(summary_path),
        'critic_root': str(critic_root),
        'score_definition': source['score_definition'],
        'candidate_exploration': exploration,
        'folds': fold_results,
        'overall': overall,
        'without_max_improvement_outlier': {
            'excluded_scene_id': outlier_scene_id,
            'audit': without_outlier,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Audit same-scene preference Critic checkpoints',
    )
    parser.add_argument('summary', type=Path)
    parser.add_argument('critic_root', type=Path)
    parser.add_argument(
        '--tasksets-root', type=Path, default=Path('data/tasksets'),
    )
    parser.add_argument(
        '--constellations-root', type=Path, default=Path('data/constellations'),
    )
    parser.add_argument('--samples-per-candidate', type=int, default=8)
    parser.add_argument(
        '--margin-thresholds', type=float, nargs='+',
        default=[0.0, 0.01, 0.05, 0.1, 0.5, 1.0],
    )
    parser.add_argument('--device', default='cpu')
    parser.add_argument(
        '--output', type=Path,
        default=Path('work_dirs/same_scene_preference_critic_64/audit.json'),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_saved_folds(
        args.summary,
        args.critic_root,
        tasksets_root=args.tasksets_root,
        constellations_root=args.constellations_root,
        samples_per_candidate=args.samples_per_candidate,
        margin_thresholds=tuple(args.margin_thresholds),
        device=torch.device(args.device),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    overall_brief = {
        key: value for key, value in result['overall'].items()
        if key != 'scenes'
    }
    without_outlier = result['without_max_improvement_outlier']
    without_outlier_brief = {
        key: value for key, value in without_outlier['audit'].items()
        if key != 'scenes'
    }
    print(json.dumps({
        'candidate_exploration': result['candidate_exploration'],
        'overall': overall_brief,
        'without_max_improvement_outlier': {
            'excluded_scene_id': without_outlier['excluded_scene_id'],
            'audit': without_outlier_brief,
        },
    }, ensure_ascii=False, indent=2, allow_nan=False))
    print(f'[done] output={args.output}')


if __name__ == '__main__':
    main()
