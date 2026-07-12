"""按论文对齐口径汇总 AEOS 调度评估结果。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


METRIC_KEYS = ('CR', 'PCR', 'WCR')


def compute_scores(
    *,
    cr: float,
    pcr: float,
    wcr: float,
    tat_s: float,
    pc_wh: float,
) -> dict[str, float]:
    """计算论文对齐的完成质量、TAT 缩放与综合得分。"""
    quality = 0.6 * cr + 0.2 * pcr + 0.2 * wcr
    if quality <= 0:
        raise ValueError('completion quality must be positive')
    if tat_s < 0:
        raise ValueError('TAT_s must be non-negative')
    if pc_wh < 0:
        raise ValueError('PC_Wh must be non-negative')

    tat_100s = tat_s / 100.0
    cs_no_tat = 1.0 / quality + pc_wh / 100.0
    cs_paper = cs_no_tat + tat_100s / 7.0
    return {
        'quality': quality,
        'TAT_100s': tat_100s,
        'CS_no_TAT': cs_no_tat,
        'CS_paper': cs_paper,
    }


def _load_metrics(path: Path) -> dict[str, float] | None:
    if path.name.endswith('_trace.json') or path.name == '_trace_summary.json':
        return None
    try:
        data: Any = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    metrics = data.get('metrics', data) if isinstance(data, dict) else None
    if not isinstance(metrics, dict):
        return None
    if not all(key in metrics for key in METRIC_KEYS):
        return None

    output: dict[str, float] = {}
    for key in ('CR', 'PCR', 'WCR', 'WPCR', 'TAT', 'PC', 'PC_Wh'):
        if key not in metrics or metrics[key] is None:
            continue
        try:
            output[key] = float(metrics[key])
        except (TypeError, ValueError):
            continue
    return output


def summarize_split(root: Path) -> dict[str, Any]:
    records = []
    for path in sorted(root.rglob('*.json')):
        metrics = _load_metrics(path)
        if metrics is not None:
            records.append(metrics)
    if not records:
        raise RuntimeError(f'No metric records found under {root}')

    means: dict[str, float] = {}
    for key in ('CR', 'PCR', 'WCR', 'WPCR', 'TAT', 'PC', 'PC_Wh'):
        values = [
            record[key]
            for record in records
            if key in record and math.isfinite(record[key])
        ]
        if values:
            means[key] = sum(values) / len(values)

    if 'TAT' not in means:
        raise RuntimeError(f'No TAT values found under {root}')
    if 'PC_Wh' not in means:
        if 'PC' not in means:
            raise RuntimeError(f'No PC or PC_Wh values found under {root}')
        means['PC_Wh'] = means['PC'] / 3600.0

    scores = compute_scores(
        cr=means['CR'],
        pcr=means['PCR'],
        wcr=means['WCR'],
        tat_s=means['TAT'],
        pc_wh=means['PC_Wh'],
    )
    return {
        'root': str(root),
        'scene_count': len(records),
        'means': means,
        'table': {
            'CS_paper': scores['CS_paper'],
            'CS_no_TAT': scores['CS_no_TAT'],
            'quality': scores['quality'],
            'CR_percent': means['CR'] * 100.0,
            'PCR_percent': means['PCR'] * 100.0,
            'WCR_percent': means['WCR'] * 100.0,
            'TAT_s': means['TAT'],
            'TAT_100s': scores['TAT_100s'],
            'PC_Wh': means['PC_Wh'],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('split_roots', nargs='+', type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = {root.name: summarize_split(root) for root in args.split_roots}
    output = {
        'score_definition': (
            'CS_paper = (0.6*CR + 0.2*PCR + 0.2*WCR)^(-1) '
            '+ TAT_s/700 + PC_Wh/100'
        ),
        'tat_scaling': 'TAT_100s = TAT_s/100',
        'note': (
            'The paper table label TAT/h is inconsistent with the 3600-second '
            'horizon; TAT_s/100 reproduces its reported comprehensive scores.'
        ),
        'splits': summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )


if __name__ == '__main__':
    main()
