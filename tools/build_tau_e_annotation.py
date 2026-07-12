"""从 rollout 轨迹构建 tau_e 过滤后的训练标注。

标注(annotation)的作用：
  标注是一个路由表，格式为 {"ids": [场景编号...], "epochs": [轮次编号...]}，
  告诉 Dataset 每个场景应加载第几轮的轨迹文件。例如 {"ids": [0,1,2], "epochs": [3,1,2]}
  表示场景 0 用 trajectories.3 的轨迹，场景 1 用 trajectories.1 的轨迹。
  标注在每轮 Stage-2 专家迭代中更新：训练模型 → rollout 生成候选轨迹 → 本脚本按
  CS ≤ tau_e 过滤 → 更新标注中对应场景的 epoch → 用扩大的数据池重新训练。

本脚本的工作：
  读取基础标注，逐场景检查 data/trajectories.N 中候选轨迹的综合评分(CS)，
  CS ≤ tau_e 的轨迹被接受（epoch 更新为新轮次），否则保留旧 epoch。
  CS = 1/(0.6*CR+0.2*PCR+0.2*WCR) + TAT_100s/7 + PC_Wh/100，越小代表轨迹质量越好。
  输出新的标注 JSON 及可选摘要。
"""

import argparse
import json
import pathlib
from typing import Any

from todd.patches.py_ import json_dump, json_load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Merge candidate trajectories into a stage-2 annotation',
    )
    parser.add_argument('base_annotation', type=pathlib.Path)
    parser.add_argument('candidate_root', type=pathlib.Path)
    parser.add_argument('output_annotation', type=pathlib.Path)
    parser.add_argument('--split', default='train')
    parser.add_argument('--candidate-epoch', type=int, required=True)
    parser.add_argument('--tau-e', type=float, default=4.5)
    parser.add_argument(
        '--comparison',
        choices=['le', 'lt'],
        default='le',
        help='How candidate score is compared against tau_e',
    )
    parser.add_argument(
        '--score-key',
        default='CS',
        help='Metric key to use if present in the metrics json',
    )
    parser.add_argument(
        '--formula',
        choices=[
            'paper_full',
            'paper_no_tat',
            'simple_fixed_pc',
            'legacy_raw_pc',
        ],
        default='paper_full',
        help='Fallback formula used when score-key is absent',
    )
    parser.add_argument(
        '--tat-scale',
        type=float,
        default=100.0,
        help='Scale factor used to convert raw TAT_s into paper table units',
    )
    parser.add_argument(
        '--pc-scale',
        type=float,
        default=100.0,
        help='Scale factor used in the final PC_Wh/scale score term',
    )
    parser.add_argument(
        '--tat-weight-scale',
        type=float,
        default=7.0,
        help='Divisor used for the TAT_h term in paper_full',
    )
    parser.add_argument(
        '--summary-path',
        type=pathlib.Path,
        default=None,
    )
    parser.add_argument(
        '--merge-mode',
        choices=['replace_existing', 'union_candidates'],
        default='replace_existing',
        help='Whether accepted candidates replace epochs for base ids only, or are unioned into the base annotation',
    )
    return parser.parse_args()


def load_annotation(
    path: pathlib.Path,
) -> tuple[list[int], list[int], str]:
    payload = json_load(str(path))
    if isinstance(payload, dict):
        ids = list(payload['ids'])
        epochs = list(payload['epochs'])
        return ids, epochs, 'dict'
    if isinstance(payload, list):
        ids = list(payload)
        epochs = [0] * len(ids)
        return ids, epochs, 'list'
    raise TypeError(f'Unsupported annotation format in {path}')


def iter_candidate_ids(candidate_root: pathlib.Path, split: str) -> list[int]:
    split_root = candidate_root / split
    ids = sorted(int(path.stem) for path in split_root.rglob('*.json'))
    return ids


def score_from_metrics(
    metrics: dict[str, Any],
    *,
    score_key: str,
    formula: str,
    tat_scale: float,
    pc_scale: float,
    tat_weight_scale: float,
) -> tuple[float, str]:
    if score_key in metrics:
        return float(metrics[score_key]), score_key

    if 'PC_Wh' in metrics:
        pc_wh = float(metrics['PC_Wh'])
        pc_source = 'PC_Wh'
    else:
        pc_wh = float(metrics.get('PC', 0.0)) / 3600.0
        pc_source = 'PC/3600'

    if formula == 'legacy_raw_pc':
        cr = max(float(metrics['CR']), 1e-6)
        tat_h = float(metrics.get('TAT', 0.0)) / tat_scale
        pc_raw = float(metrics.get('PC', 0.0)) / pc_scale
        return (
            1.0 / cr + tat_h + pc_raw,
            f'computed(1/CR + TAT/{tat_scale:g} + PC/{pc_scale:g})',
        )

    if formula == 'simple_fixed_pc':
        cr = max(float(metrics['CR']), 1e-6)
        tat_h = float(metrics.get('TAT', 0.0)) / tat_scale
        return (
            1.0 / cr + tat_h + pc_wh / pc_scale,
            f'computed(1/CR + TAT/{tat_scale:g} + {pc_source}/{pc_scale:g})',
        )

    cr = float(metrics['CR'])
    pcr = float(metrics['PCR'])
    wcr = float(metrics['WCR'])
    quality = max(0.6 * cr + 0.2 * pcr + 0.2 * wcr, 1e-6)

    if formula == 'paper_no_tat':
        return (
            1.0 / quality + pc_wh / pc_scale,
            f'computed((0.6*CR+0.2*PCR+0.2*WCR)^(-1) + {pc_source}/{pc_scale:g})',
        )

    if formula == 'paper_full':
        tat_h = float(metrics.get('TAT', 0.0)) / tat_scale
        return (
            1.0 / quality + tat_h / tat_weight_scale + pc_wh / pc_scale,
            f'computed((0.6*CR+0.2*PCR+0.2*WCR)^(-1) + '
            f'(TAT/{tat_scale:g})/{tat_weight_scale:g} + {pc_source}/{pc_scale:g})',
        )

    raise ValueError(f'Unsupported formula: {formula}')


def main() -> None:
    args = parse_args()
    ids, epochs, base_format = load_annotation(args.base_annotation)

    accepted = 0
    rejected = 0
    missing = 0
    candidate_total = 0
    score_formula = None

    if args.merge_mode == 'replace_existing':
        new_epochs = []
        for id_, base_epoch in zip(ids, epochs):
            metrics_path = (
                args.candidate_root / args.split / f'{id_ // 1000:02}' / f'{id_:05}.json'
            )
            trajectory_path = (
                args.candidate_root / args.split / f'{id_ // 1000:02}' / f'{id_:05}.pth'
            )

            if not metrics_path.exists() or not trajectory_path.exists():
                new_epochs.append(base_epoch)
                missing += 1
                continue

            candidate_total += 1
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            score, formula = score_from_metrics(
                metrics,
                score_key=args.score_key,
                formula=args.formula,
                tat_scale=args.tat_scale,
                pc_scale=args.pc_scale,
                tat_weight_scale=args.tat_weight_scale,
            )
            score_formula = formula

            accept = score <= args.tau_e if args.comparison == 'le' else score < args.tau_e
            if accept:
                new_epochs.append(args.candidate_epoch)
                accepted += 1
            else:
                new_epochs.append(base_epoch)
                rejected += 1
        new_ids = ids
    else:
        new_ids = list(ids)
        new_epochs = list(epochs)
        id_to_index = {id_: i for i, id_ in enumerate(new_ids)}

        for id_ in iter_candidate_ids(args.candidate_root, args.split):
            metrics_path = (
                args.candidate_root / args.split / f'{id_ // 1000:02}' / f'{id_:05}.json'
            )
            trajectory_path = (
                args.candidate_root / args.split / f'{id_ // 1000:02}' / f'{id_:05}.pth'
            )
            if not metrics_path.exists() or not trajectory_path.exists():
                missing += 1
                continue

            candidate_total += 1
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            score, formula = score_from_metrics(
                metrics,
                score_key=args.score_key,
                formula=args.formula,
                tat_scale=args.tat_scale,
                pc_scale=args.pc_scale,
                tat_weight_scale=args.tat_weight_scale,
            )
            score_formula = formula
            accept = score <= args.tau_e if args.comparison == 'le' else score < args.tau_e
            if accept:
                if id_ in id_to_index:
                    new_epochs[id_to_index[id_]] = args.candidate_epoch
                else:
                    id_to_index[id_] = len(new_ids)
                    new_ids.append(id_)
                    new_epochs.append(args.candidate_epoch)
                accepted += 1
            else:
                rejected += 1

    args.output_annotation.parent.mkdir(parents=True, exist_ok=True)
    json_dump(
        dict(ids=new_ids, epochs=new_epochs),
        str(args.output_annotation),
    )

    epoch_distribution: dict[str, int] = {}
    for epoch in new_epochs:
        key = str(epoch)
        epoch_distribution[key] = epoch_distribution.get(key, 0) + 1

    summary = dict(
        base_annotation=str(args.base_annotation),
        candidate_root=str(args.candidate_root),
        candidate_epoch=args.candidate_epoch,
        merge_mode=args.merge_mode,
        split=args.split,
        tau_e=args.tau_e,
        comparison=args.comparison,
        score_formula=score_formula,
        base_format=base_format,
        num_total=len(new_ids),
        num_base=len(ids),
        num_candidates=candidate_total,
        num_accepted=accepted,
        num_rejected=rejected,
        num_missing=missing,
        epoch_distribution=epoch_distribution,
        output_annotation=str(args.output_annotation),
    )

    summary_path = args.summary_path
    if summary_path is None:
        summary_path = args.output_annotation.with_suffix('.summary.json')
    json_dump(summary, str(summary_path))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
