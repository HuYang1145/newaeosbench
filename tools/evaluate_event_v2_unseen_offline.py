#!/usr/bin/env python3
"""在固定未见轨迹上公平比较随机 V2 与 V2-0 warm start。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any


LOSS_COMPONENTS = (
    'task_distillation',
    'termination',
    'commitment',
    'value',
)


def aggregate_weighted_losses(
    scenes: Sequence[Mapping[str, Any]],
    loss_weights: Mapping[str, float],
) -> dict[str, Any]:
    """按各监督头自己的事实标签数聚合跨场景 loss。"""

    if not scenes:
        raise ValueError('at least one scene is required')
    if set(loss_weights) != set(LOSS_COMPONENTS):
        raise ValueError('loss weights must cover every loss component')
    weights = {name: float(loss_weights[name]) for name in LOSS_COMPONENTS}
    if any(not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError('loss weights must be finite and non-negative')

    supports = {name: 0 for name in LOSS_COMPONENTS}
    weighted = {
        model_name: {name: 0.0 for name in LOSS_COMPONENTS}
        for model_name in ('random', 'trained')
    }
    for scene in scenes:
        for name in LOSS_COMPONENTS:
            support = int(scene['supports'][name])
            if support < 0:
                raise ValueError('loss support must be non-negative')
            supports[name] += support
            for model_name in ('random', 'trained'):
                value = float(scene[model_name][name])
                weighted[model_name][name] += value * support

    aggregated: dict[str, dict[str, float]] = {}
    for model_name in ('random', 'trained'):
        components = {
            name: (
                weighted[model_name][name] / supports[name]
                if supports[name] > 0 else math.nan
            )
            for name in LOSS_COMPONENTS
        }
        components['total'] = sum(
            components[name] * weights[name]
            for name in LOSS_COMPONENTS
        )
        aggregated[model_name] = components

    metric_names = (*LOSS_COMPONENTS, 'total')
    delta = {
        name: aggregated['trained'][name] - aggregated['random'][name]
        for name in metric_names
    }
    relative_reduction = {
        name: (
            aggregated['random'][name] - aggregated['trained'][name]
        ) / max(aggregated['random'][name], 1e-12)
        for name in metric_names
    }
    return {
        'supports': supports,
        'loss_weights': weights,
        'random': aggregated['random'],
        'trained': aggregated['trained'],
        'delta': delta,
        'relative_reduction': relative_reduction,
    }


def decide_acceptance(
    *,
    metrics: Mapping[str, Any],
    scene_count: int,
    expected_scene_count: int,
    audit_passed: bool,
) -> dict[str, Any]:
    """执行固定的严格验收门槛，并返回全部拒绝原因。"""

    reasons: list[str] = []
    if scene_count != expected_scene_count:
        reasons.append(
            f'scene count mismatch: {scene_count} != {expected_scene_count}'
        )
    if not audit_passed:
        reasons.append('checkpoint or paired-input audit failed')

    supports = metrics.get('supports', {})
    for name in LOSS_COMPONENTS:
        if int(supports.get(name, 0)) <= 0:
            reasons.append(f'{name} support is not positive')

    for section in ('random', 'trained', 'delta', 'relative_reduction'):
        values = metrics.get(section, {})
        for name in (*LOSS_COMPONENTS, 'total'):
            value = float(values.get(name, math.nan))
            if not math.isfinite(value):
                reasons.append(f'{section}.{name} is not finite')

    for name in (*LOSS_COMPONENTS, 'total'):
        random_value = float(metrics.get('random', {}).get(name, math.nan))
        trained_value = float(metrics.get('trained', {}).get(name, math.nan))
        if math.isfinite(random_value) and math.isfinite(trained_value):
            if not trained_value < random_value:
                reasons.append(f'{name} did not strictly decrease')

    return {'accepted': not reasons, 'reasons': reasons}
