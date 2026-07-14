"""用同场景多候选轨迹训练 pairwise Critic，不更新 Actor。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
from typing import Any

os.environ.setdefault('MPLCONFIGDIR', '/tmp/aeos-matplotlib')

import torch

from constellation.new_transformers.offline_critic import (
    build_transition_tensors,
    sample_time_indices,
)
from constellation.new_transformers.preference_critic import (
    CandidateTensors,
    PreferenceCriticBundle,
    fit_preference_critics,
)
from tools.train_offline_critic import load_scene_context


def _trajectory_feature(tensor: torch.Tensor) -> torch.Tensor:
    """用时间均值和标准差表示一条候选轨迹。"""

    return torch.cat((
        tensor.float().mean(0),
        tensor.float().std(0, unbiased=False),
    ))


def load_candidate_features(
    summary_path: Path,
    *,
    tasksets_root: Path,
    constellations_root: Path,
    samples_per_candidate: int,
) -> tuple[CandidateTensors, list[dict[str, Any]]]:
    """读取有效候选，聚合前半轨迹的 state/action 特征。"""

    payload = json.loads(summary_path.read_text(encoding='utf-8'))
    split = str(payload['split'])
    scene_ids = []
    states = []
    actions = []
    costs = []
    records = []
    context_cache: dict[int, dict[str, torch.Tensor]] = {}
    for scene in payload['scenes']:
        scene_id = int(scene['scene_id'])
        context = context_cache.get(scene_id)
        if context is None:
            relative_path = Path(split) / f'{scene_id // 1000:02}' / f'{scene_id:05}.json'
            context = load_scene_context(
                tasksets_root / relative_path,
                constellations_root / relative_path,
            )
            context_cache[scene_id] = context
        for candidate in scene['candidates']:
            if not candidate['valid']:
                continue
            trajectory_path = Path(candidate['trajectory_path'])
            trajectory = torch.load(
                trajectory_path,
                map_location='cpu',
                weights_only=False,
            )
            num_time_steps = int(trajectory['taskset']['progress'].shape[0])
            # 只使用前半轨迹，避免依赖接近终点的结果状态。
            early_transitions = max(1, (num_time_steps - 1) // 2)
            indices = sample_time_indices(
                num_time_steps=early_transitions + 1,
                num_samples=samples_per_candidate,
            )
            transitions = build_transition_tensors(
                trajectory,
                task_durations=context['task_durations'],
                episode_cost=float(candidate['cost']),
                time_indices=indices,
                task_static_data=context['task_static_data'],
                constellation_static_data=context['constellation_static_data'],
                task_sensor_type=context['task_sensor_type'],
                constellation_sensor_type=context['constellation_sensor_type'],
            )
            scene_ids.append(scene_id)
            states.append(_trajectory_feature(transitions.state))
            actions.append(_trajectory_feature(transitions.action))
            costs.append(float(candidate['cost']))
            records.append({
                'scene_id': scene_id,
                'candidate': candidate['candidate'],
                'cost': float(candidate['cost']),
                'trajectory_path': str(trajectory_path),
                'time_indices': indices,
            })
    if not records:
        raise ValueError('summary contains no valid candidates')
    return CandidateTensors(
        scene_ids=torch.tensor(scene_ids, dtype=torch.long),
        state=torch.stack(states),
        action=torch.stack(actions),
        cost=torch.tensor(costs, dtype=torch.float32),
    ), records


def split_candidates_by_scene(
    candidates: CandidateTensors,
    *,
    val_fraction: float,
    seed: int,
    num_folds: int | None = None,
    fold_index: int | None = None,
) -> tuple[CandidateTensors, CandidateTensors, list[int], list[int]]:
    """按 scene 切分，确保同场景候选不跨 train/val。"""

    if not 0 < val_fraction < 1:
        raise ValueError('val_fraction must be between zero and one')
    scene_ids = candidates.scene_ids.unique(sorted=True).tolist()
    if len(scene_ids) < 2:
        raise ValueError('at least two scenes are required')
    if (num_folds is None) != (fold_index is None):
        raise ValueError('num_folds and fold_index must be provided together')
    if num_folds is None:
        random.Random(seed).shuffle(scene_ids)
        num_val = min(
            len(scene_ids) - 1,
            max(1, round(len(scene_ids) * val_fraction)),
        )
        val_scene_ids = sorted(scene_ids[:num_val])
    else:
        if not 2 <= num_folds <= len(scene_ids):
            raise ValueError('num_folds must be between 2 and num scenes')
        assert fold_index is not None
        if not 0 <= fold_index < num_folds:
            raise ValueError('fold_index is outside the fold range')
        val_scene_ids = scene_ids[fold_index::num_folds]
    train_scene_ids = sorted(set(scene_ids) - set(val_scene_ids))
    val_mask = torch.zeros_like(candidates.scene_ids, dtype=torch.bool)
    for scene_id in val_scene_ids:
        val_mask |= candidates.scene_ids == scene_id
    train_mask = ~val_mask

    def select(mask: torch.Tensor) -> CandidateTensors:
        return CandidateTensors(*(tensor[mask] for tensor in candidates))

    return (
        select(train_mask),
        select(val_mask),
        train_scene_ids,
        val_scene_ids,
    )


def _bundle_state_dict(bundle: PreferenceCriticBundle) -> dict[str, Any]:
    return {
        'baseline': bundle.baseline.state_dict(),
        'critic': bundle.critic.state_dict(),
        'state_mean': bundle.state_mean,
        'state_std': bundle.state_std,
        'action_mean': bundle.action_mean,
        'action_std': bundle.action_std,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train a same-scene pairwise preference critic',
    )
    parser.add_argument('summary', type=Path)
    parser.add_argument('--tasksets-root', type=Path, default=Path('data/tasksets'))
    parser.add_argument(
        '--constellations-root',
        type=Path,
        default=Path('data/constellations'),
    )
    parser.add_argument('--samples-per-candidate', type=int, default=8)
    parser.add_argument('--val-fraction', type=float, default=0.25)
    parser.add_argument('--num-folds', type=int, default=None)
    parser.add_argument('--fold-index', type=int, default=None)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num-threads', type=int, default=8)
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('work_dirs/same_scene_preference_critic'),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.num_threads)
    device = torch.device(args.device)
    candidates, records = load_candidate_features(
        args.summary,
        tasksets_root=args.tasksets_root,
        constellations_root=args.constellations_root,
        samples_per_candidate=args.samples_per_candidate,
    )
    train, val, train_scene_ids, val_scene_ids = split_candidates_by_scene(
        candidates,
        val_fraction=args.val_fraction,
        seed=args.seed,
        num_folds=args.num_folds,
        fold_index=args.fold_index,
    )
    bundle, training_summary = fit_preference_critics(
        train,
        val,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=device,
    )
    output = {
        'purpose': (
            'rank multiple model-generated trajectories from the same scene; '
            'Actor remains frozen and no PPO is used'
        ),
        'config': {
            'summary': str(args.summary),
            'samples_per_candidate': args.samples_per_candidate,
            'val_fraction': args.val_fraction,
            'num_folds': args.num_folds,
            'fold_index': args.fold_index,
            'hidden_dim': args.hidden_dim,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'seed': args.seed,
            'device': str(device),
            'num_threads': args.num_threads,
        },
        'data': {
            'num_candidates': len(candidates.scene_ids),
            'train_scene_ids': train_scene_ids,
            'val_scene_ids': val_scene_ids,
            'state_dim': candidates.state.shape[1],
            'action_dim': candidates.action.shape[1],
            'records': records,
        },
        'training': training_summary,
        'decision': (
            'candidate_signal_passed'
            if training_summary['accepted']
            else 'stop_before_actor_update'
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(_bundle_state_dict(bundle), args.output_dir / 'critic.pth')
    (args.output_dir / 'summary.json').write_text(
        json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    print(json.dumps({
        'training': training_summary,
        'decision': output['decision'],
    }, indent=2, ensure_ascii=False))
    print(f'[done] output_dir={args.output_dir}')


if __name__ == '__main__':
    main()
