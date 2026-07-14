"""用现有轨迹训练诊断型离线 Critic，不更新 Actor 或运行 Basilisk。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
from typing import Any

os.environ.setdefault('MPLCONFIGDIR', '/tmp/aeos-matplotlib')

import torch

from constellation.data import Constellation, TaskSet
from constellation.new_transformers.offline_critic import (
    DiagnosticCriticBundle,
    OfflineDatasetTensors,
    TrajectoryRecord,
    audit_candidate_coverage,
    build_transition_tensors,
    combine_transition_tensors,
    fit_diagnostic_critics,
    load_routed_records,
    sample_time_indices,
    split_records_by_scene,
)


def select_records(
    records: list[TrajectoryRecord],
    *,
    limit: int | None,
    seed: int,
) -> list[TrajectoryRecord]:
    """按 scene 抽样；若一个 scene 有多个候选，则一起保留。"""

    if limit is None:
        return list(records)
    if limit <= 0:
        raise ValueError('limit must be positive')
    grouped: dict[int, list[TrajectoryRecord]] = defaultdict(list)
    for record in records:
        grouped[record.scene_id].append(record)
    scene_ids = sorted(grouped)
    random.Random(seed).shuffle(scene_ids)
    selected_ids = set(scene_ids[:limit])
    return [record for record in records if record.scene_id in selected_ids]


def load_scene_context(
    taskset_path: Path,
    constellation_path: Path,
) -> dict[str, torch.Tensor]:
    """读取静态场景张量；不调用仿真或轨道传播。"""

    taskset = TaskSet.load(str(taskset_path))
    task_sensor_type, task_static_data = taskset.to_tensor()
    constellation = Constellation.load(str(constellation_path))
    constellation_sensor_type, constellation_static_data = (
        constellation.static_to_tensor()
    )
    return {
        'task_durations': taskset.durations.float(),
        'task_static_data': task_static_data.float(),
        'constellation_static_data': constellation_static_data.float(),
        'task_sensor_type': task_sensor_type,
        'constellation_sensor_type': constellation_sensor_type,
    }


def load_transition_dataset(
    records: list[TrajectoryRecord],
    *,
    tasksets_root: Path,
    constellations_root: Path | None = None,
    split: str,
    samples_per_trajectory: int,
) -> OfflineDatasetTensors:
    """把路由后的轨迹加载为紧凑 ``(s,a,r,s')`` 张量。"""

    items = []
    for trajectory_id, record in enumerate(records):
        if record.trajectory_path is None:
            raise ValueError('trajectory path is required for training')
        trajectory: dict[str, Any] = torch.load(
            record.trajectory_path,
            map_location='cpu',
            weights_only=False,
        )
        taskset_path = (
            tasksets_root / split / f'{record.scene_id // 1000:02}'
            / f'{record.scene_id:05}.json'
        )
        if constellations_root is None:
            constellations_root = tasksets_root.parent / 'constellations'
        constellation_path = (
            constellations_root / split / f'{record.scene_id // 1000:02}'
            / f'{record.scene_id:05}.json'
        )
        context = load_scene_context(taskset_path, constellation_path)
        num_time_steps = int(trajectory['taskset']['progress'].shape[0])
        indices = sample_time_indices(
            num_time_steps=num_time_steps,
            num_samples=samples_per_trajectory,
        )
        items.append((trajectory_id, build_transition_tensors(
            trajectory,
            task_durations=context.pop('task_durations'),
            episode_cost=record.episode_cost,
            time_indices=indices,
            **context,
        )))
        if (trajectory_id + 1) % 100 == 0:
            print(
                f'[load] {trajectory_id + 1}/{len(records)} trajectories',
                flush=True,
            )
    return combine_transition_tensors(items)


def _bundle_state_dict(bundle: DiagnosticCriticBundle) -> dict[str, Any]:
    return {
        'baseline': bundle.baseline.state_dict(),
        'critic': bundle.critic.state_dict(),
        'state_mean': bundle.state_mean,
        'state_std': bundle.state_std,
        'action_mean': bundle.action_mean,
        'action_std': bundle.action_std,
        'cost_mean': bundle.cost_mean,
        'cost_std': bundle.cost_std,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train an offline diagnostic critic from saved trajectories',
    )
    parser.add_argument(
        '--annotation',
        type=Path,
        default=Path('data/annotations/train.json'),
    )
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument(
        '--tasksets-root',
        type=Path,
        default=Path('data/tasksets'),
    )
    parser.add_argument(
        '--constellations-root',
        type=Path,
        default=Path('data/constellations'),
    )
    parser.add_argument('--split', default='train')
    parser.add_argument('--max-scenes', type=int, default=1024)
    parser.add_argument('--samples-per-trajectory', type=int, default=8)
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num-threads', type=int, default=8)
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('work_dirs/offline_critic_pilot'),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.num_threads)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')

    all_records = load_routed_records(
        annotation_path=args.annotation,
        data_root=args.data_root,
        split=args.split,
    )
    records = select_records(
        all_records,
        limit=args.max_scenes,
        seed=args.seed,
    )
    train_records, val_records = split_records_by_scene(
        records,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(
        f'[data] routed={len(all_records)} selected={len(records)} '
        f'train={len(train_records)} val={len(val_records)}',
        flush=True,
    )
    train = load_transition_dataset(
        train_records,
        tasksets_root=args.tasksets_root,
        constellations_root=args.constellations_root,
        split=args.split,
        samples_per_trajectory=args.samples_per_trajectory,
    )
    val = load_transition_dataset(
        val_records,
        tasksets_root=args.tasksets_root,
        constellations_root=args.constellations_root,
        split=args.split,
        samples_per_trajectory=args.samples_per_trajectory,
    )
    bundle, training_summary = fit_diagnostic_critics(
        train,
        val,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=device,
    )

    coverage = audit_candidate_coverage(
        data_root=args.data_root,
        split=args.split,
    )
    output = {
        'purpose': (
            'diagnose whether action features improve trajectory CS_paper '
            'ranking beyond a state-only baseline; Actor remains frozen'
        ),
        'score_definition': (
            'CS_paper = (0.6*CR + 0.2*PCR + 0.2*WCR)^(-1) '
            '+ TAT_s/700 + PC_Wh/100'
        ),
        'config': {
            'annotation': str(args.annotation),
            'split': args.split,
            'max_scenes': args.max_scenes,
            'samples_per_trajectory': args.samples_per_trajectory,
            'val_fraction': args.val_fraction,
            'hidden_dim': args.hidden_dim,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'seed': args.seed,
            'device': str(device),
            'num_threads': args.num_threads,
        },
        'data': {
            'num_routed_records': len(all_records),
            'num_selected_records': len(records),
            'num_train_records': len(train_records),
            'num_val_records': len(val_records),
            'candidate_coverage': coverage,
            'warning': (
                'Repeated-scene candidates are required to distinguish action '
                'quality from scene difficulty reliably.'
            ),
        },
        'training': training_summary,
        'decision': (
            'proceed_to_advantage_adapter'
            if training_summary['accepted']
            else 'stop_before_actor_update'
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / 'critic.pth'
    summary_path = args.output_dir / 'summary.json'
    torch.save(_bundle_state_dict(bundle), checkpoint_path)
    summary_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(output, indent=2, ensure_ascii=False), flush=True)
    print(f'[done] checkpoint={checkpoint_path}', flush=True)
    print(f'[done] summary={summary_path}', flush=True)


if __name__ == '__main__':
    main()
