#!/usr/bin/env python3
"""在固定未见轨迹上公平比较随机 V2 与 V2-0 warm start。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
import random
from typing import NamedTuple
from typing import Any

import numpy as np
import torch

from constellation.new_transformers.event_v2.model import (
    EventJointActorCritic,
)
from constellation.new_transformers.event_v2.transition import (
    transition_schema_fingerprint,
)


LOSS_COMPONENTS = (
    'task_distillation',
    'termination',
    'commitment',
    'value',
)
CHECKPOINT_VERSION = 1
STAGE = 'V2-0'


class PairedModels(NamedTuple):
    random_model: EventJointActorCritic
    trained_model: EventJointActorCritic
    audit: dict[str, Any]


def _set_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def audit_training_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    expected_config_fingerprint: str,
    expected_steps: int,
) -> dict[str, Any]:
    """在加载权重前验证 V2-0 checkpoint 的不可变元数据。"""

    if checkpoint.get('checkpoint_version') != CHECKPOINT_VERSION:
        raise ValueError('checkpoint version mismatch')
    if checkpoint.get('stage') != STAGE:
        raise ValueError('checkpoint stage mismatch')
    if checkpoint.get('steps') != expected_steps:
        raise ValueError('checkpoint step mismatch')
    schema_fingerprint = transition_schema_fingerprint()
    if checkpoint.get('transition_schema_fingerprint') != schema_fingerprint:
        raise ValueError('checkpoint schema fingerprint mismatch')
    if checkpoint.get('config_fingerprint') != expected_config_fingerprint:
        raise ValueError('checkpoint config fingerprint mismatch')
    model_state = checkpoint.get('model')
    if not isinstance(model_state, Mapping):
        raise ValueError('checkpoint model state is missing')
    return {
        'checkpoint_version': CHECKPOINT_VERSION,
        'stage': STAGE,
        'steps': expected_steps,
        'transition_schema_fingerprint': schema_fingerprint,
        'config_fingerprint': expected_config_fingerprint,
    }


def build_paired_models(
    *,
    model_kwargs: Mapping[str, Any],
    stage3_checkpoint: str | Path,
    trained_checkpoint: str | Path,
    expected_config_fingerprint: str,
    seed: int,
    device: torch.device,
) -> PairedModels:
    """构造共享 Stage3 backbone 的随机基线和严格加载的训练模型。"""

    stage3_checkpoint = Path(stage3_checkpoint)
    trained_checkpoint = Path(trained_checkpoint)
    if not stage3_checkpoint.is_file():
        raise FileNotFoundError(f'Stage3 checkpoint not found: {stage3_checkpoint}')
    if not trained_checkpoint.is_file():
        raise FileNotFoundError(
            f'V2 trained checkpoint not found: {trained_checkpoint}'
        )
    checkpoint = torch.load(
        trained_checkpoint,
        map_location='cpu',
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError('V2 checkpoint must be a mapping')
    audit = audit_training_checkpoint(
        checkpoint,
        expected_config_fingerprint=expected_config_fingerprint,
        expected_steps=10_000,
    )

    _set_seed(seed, device)
    random_model = EventJointActorCritic(**dict(model_kwargs))
    random_model.load_stage3_checkpoint(stage3_checkpoint)
    _set_seed(seed, device)
    trained_model = EventJointActorCritic(**dict(model_kwargs))
    trained_model.load_stage3_checkpoint(stage3_checkpoint)
    trained_model.load_state_dict(checkpoint['model'], strict=True)

    random_backbone = random_model.backbone.transformer.state_dict()
    trained_backbone = trained_model.backbone.transformer.state_dict()
    backbone_exact_match = (
        random_backbone.keys() == trained_backbone.keys()
        and all(
            torch.equal(value, trained_backbone[name])
            for name, value in random_backbone.items()
        )
    )
    if not backbone_exact_match:
        raise ValueError('random and trained Stage3 backbones do not match')

    random_model.eval().to(device)
    trained_model.eval().to(device)
    audit.update({
        'seed': seed,
        'strict_model_load': True,
        'backbone_exact_match': True,
    })
    return PairedModels(random_model, trained_model, audit)


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
