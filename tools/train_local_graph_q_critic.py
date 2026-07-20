"""训练300秒受控局部标签上的 Graph-Q 裁判模型。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from constellation.new_transformers.local_graph_q_critic import (
    LOCAL_OUTCOME_KEYS,
    LocalGraphQCriticBundle,
    LocalGraphQSample,
    fit_local_graph_q_critics,
    samples_from_branch_summary,
    split_samples_by_scene,
)


def load_local_samples(
    paths: Sequence[Path],
    *,
    min_cost_margin: float,
) -> tuple[list[LocalGraphQSample], dict[str, Any]]:
    if min_cost_margin < 0:
        raise ValueError('min_cost_margin must be non-negative')
    samples: list[LocalGraphQSample] = []
    source_samples = 0
    filtered = 0
    for path in paths:
        payload = json.loads(path.read_text(encoding='utf-8'))
        loaded = samples_from_branch_summary(payload)
        source_samples += len(loaded)
        for sample in loaded:
            if sample.margin < min_cost_margin:
                filtered += 1
            else:
                samples.append(sample)
    if not samples:
        raise ValueError('no local Graph-Q samples remain after filtering')
    return samples, {
        'num_source_summaries': len(paths),
        'num_source_samples': source_samples,
        'num_filtered_small_margin': filtered,
        'num_samples': len(samples),
        'num_scenes': len({sample.scene_id
                           for sample in samples}),
        'outcome_keys': list(LOCAL_OUTCOME_KEYS),
        'uses_is_visible_as_input': False,
        'basilisk_online_inference': False,
    }


def audit_horizon_consistency(
    payloads: Sequence[Mapping[str, Any]],
    *,
    primary_horizon: int,
    check_horizon: int,
) -> dict[str, float | int | None]:
    comparable = 0
    agreeing = 0
    reversed_pairs = 0
    ties = 0
    for payload in payloads:
        for record in payload['records']:
            branches = record['branches']
            for pair in record['preference_pairs']:
                if int(pair['primary_horizon']) != primary_horizon:
                    continue
                better = branches[pair['better_branch']]
                worse = branches[pair['worse_branch']]
                better_cost = better['horizons'][
                    str(check_horizon)]['prefix_metrics']['prefix_cost']
                worse_cost = worse['horizons'][
                    str(check_horizon)]['prefix_metrics']['prefix_cost']
                if better_cost is None or worse_cost is None:
                    continue
                comparable += 1
                if float(better_cost) < float(worse_cost):
                    agreeing += 1
                elif float(better_cost) > float(worse_cost):
                    reversed_pairs += 1
                else:
                    ties += 1
    return {
        'primary_horizon': primary_horizon,
        'check_horizon': check_horizon,
        'comparable_pairs': comparable,
        'agreeing_pairs': agreeing,
        'reversed_pairs': reversed_pairs,
        'ties': ties,
        'agreement': None if comparable == 0 else agreeing / comparable,
    }


def summarize_cross_validation(
    folds: Sequence[Mapping[str, Any]],
) -> dict[str, float | int | bool]:
    if not folds:
        raise ValueError('at least one fold summary is required')
    accepted_folds = sum(bool(fold['training']['accepted']) for fold in folds)
    baseline_accuracy = sum(
        float(fold['training']['baseline']['pairwise_accuracy'])
        for fold in folds
    ) / len(folds)
    graph_accuracy = sum(
        float(fold['training']['graph_q']['pairwise_accuracy'])
        for fold in folds
    ) / len(folds)
    baseline_regret = sum(
        float(fold['training']['baseline']['mean_regret']) for fold in folds
    ) / len(folds)
    graph_regret = sum(
        float(fold['training']['graph_q']['mean_regret']) for fold in folds
    ) / len(folds)
    gain = graph_accuracy - baseline_accuracy
    required_folds = min(3, len(folds))
    accepted = bool(
        graph_accuracy >= 0.6 and gain >= 0.05
        and graph_regret <= baseline_regret
        and accepted_folds >= required_folds
    )
    return {
        'num_folds': len(folds),
        'accepted_folds': accepted_folds,
        'mean_baseline_pairwise_accuracy': baseline_accuracy,
        'mean_graph_q_pairwise_accuracy': graph_accuracy,
        'mean_pairwise_accuracy_gain': gain,
        'mean_baseline_regret': baseline_regret,
        'mean_graph_q_regret': graph_regret,
        'accepted': accepted,
    }


def _bundle_payload(bundle: LocalGraphQCriticBundle) -> dict[str, Any]:
    return {
        'baseline': bundle.baseline.state_dict(),
        'graph_q': bundle.graph_q.state_dict(),
        'satellite_mean': bundle.satellite_mean,
        'satellite_std': bundle.satellite_std,
        'task_mean': bundle.task_mean,
        'task_std': bundle.task_std,
        'outcome_mean': bundle.outcome_mean,
        'outcome_std': bundle.outcome_std,
        'outcome_keys': LOCAL_OUTCOME_KEYS,
        'uses_is_visible_as_input': False,
        'basilisk_online_inference': False,
    }


def _expand_paths(paths: Sequence[Path]) -> list[Path]:
    expanded = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.rglob('summary.json')))
        elif path.is_file():
            expanded.append(path)
        else:
            raise FileNotFoundError(path)
    unique = list(dict.fromkeys(path.resolve() for path in expanded))
    if not unique:
        raise ValueError('no local rollout summary files found')
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train scene-fold local Graph-Q Critic'
    )
    parser.add_argument('summaries', type=Path, nargs='+')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--num-folds', type=int, default=4)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--outcome-loss-weight', type=float, default=0.2)
    parser.add_argument('--margin-clip', type=float, default=1.0)
    parser.add_argument('--min-cost-margin', type=float, default=0.01)
    parser.add_argument('--primary-horizon', type=int, default=300)
    parser.add_argument('--check-horizon', type=int, default=600)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num-threads', type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.num_threads)
    paths = _expand_paths(args.summaries)
    payloads = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    samples, data_metadata = load_local_samples(
        paths, min_cost_margin=args.min_cost_margin
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    folds = []
    for fold_index in range(args.num_folds):
        train, val, train_ids, val_ids = split_samples_by_scene(
            samples,
            num_folds=args.num_folds,
            fold_index=fold_index,
        )
        bundle, training = fit_local_graph_q_critics(
            train,
            val,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            outcome_loss_weight=args.outcome_loss_weight,
            margin_clip=args.margin_clip,
            seed=args.seed + fold_index,
            device=device,
        )
        fold_dir = args.output_dir / f'fold{fold_index}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(_bundle_payload(bundle), fold_dir / 'critic.pth')
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
        print(
            f'[fold {fold_index}] graph_q='
            f"{training['graph_q']['pairwise_accuracy']:.4f} "
            f"gain={training['pairwise_accuracy_gain']:+.4f}",
            flush=True,
        )
    combined = summarize_cross_validation(folds)
    output = {
        'purpose': '300-second controlled local Graph-Q; Actor frozen',
        'config': vars(args) | {
            'device': str(device)
        },
        'data': data_metadata,
        'horizon_consistency': audit_horizon_consistency(
            payloads,
            primary_horizon=args.primary_horizon,
            check_horizon=args.check_horizon,
        ),
        'folds': folds,
        'combined': combined,
        'decision': (
            'ready_for_critic_rerank_pilot'
            if combined['accepted'] else 'stop_before_actor_or_reranking'
        ),
    }
    output['config']['summaries'] = [str(path) for path in paths]
    output['config']['output_dir'] = str(args.output_dir)
    (args.output_dir / 'summary.json').write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(combined, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
