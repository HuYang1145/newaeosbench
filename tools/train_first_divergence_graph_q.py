"""用第一分歧点偏好训练身份敏感的 Graph-Q 裁判模型。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from constellation.data import Constellation, TaskSet
from constellation.new_transformers.graph_q_critic import (
    GraphQCriticBundle,
    GraphQSample,
    GraphQSceneContext,
    audit_pairwise_tournament,
    fit_graph_q_critics,
)


def _map_action_to_indices(
    action: list[int],
    task_ids: torch.Tensor,
) -> torch.Tensor:
    mapping = {int(task_id): index for index, task_id in enumerate(task_ids)}
    indices = []
    for task_id in action:
        task_id = int(task_id)
        if task_id < 0:
            indices.append(-1)
        elif task_id not in mapping:
            raise ValueError(f'unknown task id {task_id} in joint action')
        else:
            indices.append(mapping[task_id])
    return torch.tensor(indices, dtype=torch.long)


def _action_summary(
    action: torch.Tensor,
    *,
    previous_action: torch.Tensor,
    task_progress_ratio: torch.Tensor,
    compatibility: torch.Tensor,
) -> torch.Tensor:
    num_satellites = max(action.numel(), 1)
    active = action >= 0
    if not active.any():
        return torch.zeros(7)
    satellite_ids = active.nonzero().flatten()
    task_indices = action[active]
    unique = int(task_indices.unique().numel())
    continued = active & (action == previous_action)
    counts = torch.bincount(
        task_indices,
        minlength=task_progress_ratio.numel(),
    )
    return torch.tensor([
        float(active.sum()) / num_satellites,
        unique / num_satellites,
        (float(active.sum()) - unique) / num_satellites,
        float(task_progress_ratio[task_indices].mean()),
        float(compatibility[satellite_ids, task_indices].mean()),
        float(counts.max()) / num_satellites,
        float(continued.sum()) / float(active.sum()),
    ])


def build_graph_q_sample(
    *,
    record: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    context: GraphQSceneContext,
) -> GraphQSample:
    """只从决策前状态与精确动作构造一个 Graph-Q 偏好样本。"""

    divergence = int(record['divergence_index'])
    sensor_index = record['sensor_enabled_source_index']
    if sensor_index is None:
        raise ValueError('pre-action sensor state is required for Graph-Q')
    sensor_index = int(sensor_index)
    better_action = _map_action_to_indices(
        list(record['better_action']),
        context.task_ids,
    )
    worse_action = _map_action_to_indices(
        list(record['worse_action']),
        context.task_ids,
    )

    constellation = trajectory['constellation']
    taskset = trajectory['taskset']
    if not isinstance(constellation,
                      Mapping) or not isinstance(taskset, Mapping):
        raise TypeError('trajectory state sections must be mappings')
    sensor_enabled = torch.as_tensor(
        constellation['sensor_enabled'],
    )[sensor_index].float()
    dynamic_data = torch.as_tensor(
        constellation['data'],
    )[divergence].float()
    progress = torch.as_tensor(taskset['progress'])[divergence].float()
    actions = trajectory['actions']
    if not isinstance(actions, Mapping):
        raise TypeError('trajectory actions must be a mapping')
    if divergence <= 0:
        raise ValueError('Graph-Q requires a reconstructable previous action')
    previous_action = _map_action_to_indices(
        torch.as_tensor(actions['task_id'])[divergence - 1].tolist(),
        context.task_ids,
    )
    durations = context.task_durations.float().clamp_min(1.0)
    progress_ratio = (progress / durations).clamp(0, 1)

    task_static = context.task_static_data.float().clone()
    task_static[:, :2] -= divergence
    satellite_features = torch.cat((
        context.constellation_static_data.float(),
        dynamic_data,
        sensor_enabled.unsqueeze(-1),
        context.constellation_sensor_type.float().unsqueeze(-1),
    ), -1)
    task_features = torch.cat((
        task_static,
        progress_ratio.unsqueeze(-1),
        (progress_ratio >= 1.0).float().unsqueeze(-1),
        context.task_sensor_type.float().unsqueeze(-1),
    ), -1)
    compatibility = (
        context.constellation_sensor_type[:, None] == context.task_sensor_type[
            None, :]
    ).float()
    return GraphQSample(
        scene_id=int(record['scene_id']),
        satellite_features=satellite_features,
        task_features=task_features,
        compatibility=compatibility,
        previous_action=previous_action,
        better_action=better_action,
        worse_action=worse_action,
        better_summary=_action_summary(
            better_action,
            previous_action=previous_action,
            task_progress_ratio=progress_ratio,
            compatibility=compatibility,
        ),
        worse_summary=_action_summary(
            worse_action,
            previous_action=previous_action,
            task_progress_ratio=progress_ratio,
            compatibility=compatibility,
        ),
        margin=float(record['cost_margin']),
        better_candidate=str(record['better_candidate']),
        worse_candidate=str(record['worse_candidate']),
        better_cost=float(record['better_cost']),
        worse_cost=float(record['worse_cost']),
    )


def load_graph_q_scene_context(
    *,
    taskset_path: Path,
    constellation_path: Path,
) -> GraphQSceneContext:
    """加载静态节点特征，不运行 Basilisk 或轨道传播。"""

    taskset = TaskSet.load(str(taskset_path))
    task_sensor_type, task_static_data = taskset.to_tensor()
    constellation = Constellation.load(str(constellation_path))
    constellation_sensor_type, constellation_static_data = (
        constellation.static_to_tensor()
    )
    return GraphQSceneContext(
        task_ids=taskset.ids.long(),
        task_durations=taskset.durations.float(),
        task_static_data=task_static_data.float(),
        task_sensor_type=task_sensor_type.long(),
        constellation_static_data=constellation_static_data.float(),
        constellation_sensor_type=constellation_sensor_type.long(),
    )


def load_graph_q_samples(
    divergence_path: Path,
    *,
    tasksets_root: Path,
    constellations_root: Path,
    split: str,
) -> tuple[list[GraphQSample], dict[str, object]]:
    """从 P1 索引加载可用样本，并保持每次只缓存一个场景。"""

    payload = json.loads(divergence_path.read_text(encoding='utf-8'))
    contract = payload.get('input_contract', {})
    if contract.get('uses_is_visible_as_input') is not False:
        raise ValueError('Graph-Q input contract must exclude is_visible')
    if contract.get('basilisk_online_inference') is not False:
        raise ValueError('Graph-Q input contract must exclude online Basilisk')
    records = [
        record for record in payload['records']
        if record.get('usable_for_graph_q') is True
    ]
    samples = []
    current_scene_id: int | None = None
    trajectory_cache: dict[str, Mapping[str, Any]] = {}
    context: GraphQSceneContext | None = None
    for record in records:
        scene_id = int(record['scene_id'])
        if scene_id != current_scene_id:
            relative = Path(
                split
            ) / f'{scene_id // 1000:02}' / f'{scene_id:05}.json'
            context = load_graph_q_scene_context(
                taskset_path=tasksets_root / relative,
                constellation_path=constellations_root / relative,
            )
            trajectory_cache.clear()
            current_scene_id = scene_id
        assert context is not None
        trajectory_path = str(record['better_trajectory_path'])
        trajectory = trajectory_cache.get(trajectory_path)
        if trajectory is None:
            trajectory = torch.load(
                trajectory_path,
                map_location='cpu',
                weights_only=False,
            )
            trajectory_cache[trajectory_path] = trajectory
        samples.append(
            build_graph_q_sample(
                record=record,
                trajectory=trajectory,
                context=context,
            )
        )
    if not samples:
        raise ValueError(
            'divergence dataset contains no usable Graph-Q samples'
        )
    return samples, {
        'source': str(divergence_path),
        'num_source_records': len(payload['records']),
        'num_usable_samples': len(samples),
        'num_scenes': len({sample.scene_id
                           for sample in samples}),
        'satellite_dim': samples[0].satellite_features.shape[-1],
        'task_dim': samples[0].task_features.shape[-1],
        'uses_is_visible_as_input': False,
        'basilisk_online_inference': False,
    }


def split_samples_by_scene(
    samples: Sequence[GraphQSample],
    *,
    num_folds: int,
    fold_index: int,
) -> tuple[list[GraphQSample], list[GraphQSample], list[int], list[int]]:
    scene_ids = sorted({item.scene_id for item in samples})
    if not 2 <= num_folds <= len(scene_ids):
        raise ValueError('number of folds must be between 2 and scene count')
    if not 0 <= fold_index < num_folds:
        raise ValueError('fold index is outside the fold range')
    val_ids = scene_ids[fold_index::num_folds]
    val_set = set(val_ids)
    train_ids = sorted(set(scene_ids) - val_set)
    return (
        [item for item in samples if item.scene_id not in val_set],
        [item for item in samples if item.scene_id in val_set],
        train_ids,
        val_ids,
    )


def _bundle_state_dict(bundle: GraphQCriticBundle) -> dict[str, Any]:
    return {
        'baseline': bundle.baseline.state_dict(),
        'graph_q': bundle.graph_q.state_dict(),
        'satellite_mean': bundle.satellite_mean,
        'satellite_std': bundle.satellite_std,
        'task_mean': bundle.task_mean,
        'task_std': bundle.task_std,
        'summary_mean': bundle.summary_mean,
        'summary_std': bundle.summary_std,
    }


def _without_scene(
    samples: Sequence[GraphQSample],
    predictions: dict[str, torch.Tensor],
    scene_id: int,
) -> tuple[list[GraphQSample], dict[str, torch.Tensor]]:
    indices = [
        index for index, sample in enumerate(samples)
        if sample.scene_id != scene_id
    ]
    return (
        [samples[index] for index in indices],
        {
            key: value[torch.tensor(indices)]
            for key, value in predictions.items()
        },
    )


def _max_greedy_improvement_scene(samples: Sequence[GraphQSample]) -> int:
    scene_costs: dict[int, dict[str, float]] = {}
    for sample in samples:
        costs = scene_costs.setdefault(sample.scene_id, {})
        costs[sample.better_candidate] = sample.better_cost
        costs[sample.worse_candidate] = sample.worse_cost
    return max(
        scene_costs,
        key=lambda scene_id: (
            scene_costs[scene_id].get(
                'candidate_000_greedy',
                min(scene_costs[scene_id].values()),
            ) - min(scene_costs[scene_id].values())
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=
        'Train scene-fold Graph-Q from first-divergence preferences',
    )
    parser.add_argument('divergence_path', type=Path)
    parser.add_argument(
        '--tasksets-root', type=Path, default=Path('data/tasksets')
    )
    parser.add_argument(
        '--constellations-root',
        type=Path,
        default=Path('data/constellations'),
    )
    parser.add_argument('--split', default='train')
    parser.add_argument('--num-folds', type=int, default=4)
    parser.add_argument('--hidden-dim', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--margin-clip', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num-threads', type=int, default=16)
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('work_dirs/first_divergence_graph_q'),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.num_threads)
    device = torch.device(args.device)
    samples, data_metadata = load_graph_q_samples(
        args.divergence_path,
        tasksets_root=args.tasksets_root,
        constellations_root=args.constellations_root,
        split=args.split,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds = []
    held_out_samples: list[GraphQSample] = []
    held_out_predictions: dict[str, list[torch.Tensor]] = {
        'baseline_better': [],
        'baseline_worse': [],
        'graph_q_better': [],
        'graph_q_worse': [],
    }
    for fold_index in range(args.num_folds):
        train, val, train_ids, val_ids = split_samples_by_scene(
            samples,
            num_folds=args.num_folds,
            fold_index=fold_index,
        )
        bundle, training = fit_graph_q_critics(
            train,
            val,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            margin_clip=args.margin_clip,
            seed=args.seed + fold_index,
            device=device,
        )
        predictions = bundle.predict(
            val,
            batch_size=args.batch_size,
            device=device,
        )
        fold_dir = args.output_dir / f'fold{fold_index}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(_bundle_state_dict(bundle), fold_dir / 'critic.pth')
        fold_summary = {
            'fold_index': fold_index,
            'train_scene_ids': train_ids,
            'val_scene_ids': val_ids,
            'training': training,
        }
        (fold_dir / 'summary.json').write_text(
            json.dumps(fold_summary, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )
        folds.append(fold_summary)
        held_out_samples.extend(val)
        for key, value in predictions.items():
            held_out_predictions[key].append(value)
        print(
            f'[fold {fold_index}] graph_q='
            f"{training['graph_q']['pairwise_accuracy']:.4f} "
            f"gain={training['pairwise_accuracy_gain']:+.4f}",
            flush=True,
        )

    combined_predictions = {
        key: torch.cat(values)
        for key, values in held_out_predictions.items()
    }
    baseline_audit = audit_pairwise_tournament(
        held_out_samples,
        better_scores=combined_predictions['baseline_better'],
        worse_scores=combined_predictions['baseline_worse'],
        greedy_candidate='candidate_000_greedy',
    )
    graph_audit = audit_pairwise_tournament(
        held_out_samples,
        better_scores=combined_predictions['graph_q_better'],
        worse_scores=combined_predictions['graph_q_worse'],
        greedy_candidate='candidate_000_greedy',
    )
    outlier_scene = _max_greedy_improvement_scene(held_out_samples)
    filtered_samples, filtered_predictions = _without_scene(
        held_out_samples,
        combined_predictions,
        outlier_scene,
    )
    filtered_graph = audit_pairwise_tournament(
        filtered_samples,
        better_scores=filtered_predictions['graph_q_better'],
        worse_scores=filtered_predictions['graph_q_worse'],
        greedy_candidate='candidate_000_greedy',
    )
    gain = (
        float(graph_audit['pairwise_accuracy'])
        - float(baseline_audit['pairwise_accuracy'])
    )
    accepted_folds = sum(bool(fold['training']['accepted']) for fold in folds)
    accepted = bool(
        float(graph_audit['pairwise_accuracy']) >= 0.6 and gain >= 0.05
        and float(graph_audit['mean_regret']
                  ) <= float(baseline_audit['mean_regret'])
        and float(filtered_graph['selected_vs_greedy_mean_cost_delta']) <= 0
        and accepted_folds >= 3
    )
    output = {
        'purpose': 'P2 first-divergence identity-aware Graph-Q; Actor frozen',
        'config': {
            'num_folds': args.num_folds,
            'hidden_dim': args.hidden_dim,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'margin_clip': args.margin_clip,
            'seed': args.seed,
            'device': str(device),
            'num_threads': args.num_threads,
        },
        'data': data_metadata,
        'folds': folds,
        'combined': {
            'baseline': baseline_audit,
            'graph_q': graph_audit,
            'pairwise_accuracy_gain': gain,
            'accepted_folds': accepted_folds,
            'without_max_greedy_improvement_outlier': {
                'excluded_scene_id': outlier_scene,
                'graph_q': filtered_graph,
            },
            'accepted': accepted,
        },
        'decision': 'ready_for_actor_pilot'
        if accepted else 'stop_before_actor',
    }
    (args.output_dir / 'summary.json').write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(output['combined'], indent=2, ensure_ascii=False))
    print(f'[done] output_dir={args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
