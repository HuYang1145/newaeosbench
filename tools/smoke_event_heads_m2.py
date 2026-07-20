#!/usr/bin/env python3
"""在真实轨迹上验证 M2 冻结主干的一次 forward/backward/step。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from todd.configs import PyConfig

from constellation.new_transformers.dataset import JointDataset
from constellation.new_transformers.model import JointModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--annotation-file', type=Path, required=True)
    parser.add_argument('--split', default='train')
    parser.add_argument('--scene-index', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def _state_dict(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(payload, dict) and 'state_dict' in payload:
        payload = payload['state_dict']
    if not isinstance(payload, dict):
        raise TypeError('checkpoint must contain a state dict')
    return payload


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.annotation_file.is_file():
        raise FileNotFoundError(args.annotation_file)
    if args.scene_index < 0:
        raise ValueError('scene-index must be non-negative')
    if args.batch_size <= 0 or args.lr <= 0:
        raise ValueError('batch-size and lr must be positive')

    config = PyConfig.load(
        'constellation/new_transformers/config_event_heads_m2.py'
    )
    model_kwargs = dict(config.trainer.model)
    model_kwargs.pop('type')
    model = JointModel(**model_kwargs)
    incompatible = model.load_state_dict(
        _state_dict(args.checkpoint),
        strict=False,
    )
    dataset = JointDataset(
        split=args.split,
        annotation_file=args.annotation_file.name,
        batch_size=args.batch_size,
        constraint_batch_size=args.batch_size,
        include_temporal_history=True,
        temporal_horizons=(5, 15, 30, 60),
    )
    if args.scene_index >= len(dataset):
        raise IndexError('scene-index exceeds annotation length')
    batch = dataset[args.scene_index]

    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    frozen = {
        name: parameter
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    if not trainable or not all(
        '_temporal_adapter.' in name for name in trainable
    ):
        raise RuntimeError('only Temporal Adapter parameters may train')
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in frozen.items()
    }
    event_before = (
        model._transformer._temporal_adapter.event_head.weight
        .detach()
        .clone()
    )
    optimizer = torch.optim.AdamW(trainable.values(), lr=args.lr)

    model.train()
    memo = model(type('Runner', (), {'iter_': 0})(), batch, {})
    loss = memo['loss']
    if not torch.isfinite(loss):
        raise RuntimeError(f'non-finite M2 loss: {loss}')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    frozen_unchanged = all(
        parameter.grad is None
        and torch.equal(parameter.detach(), frozen_before[name])
        for name, parameter in frozen.items()
    )
    event_head_changed = not torch.equal(
        model._transformer._temporal_adapter.event_head.weight.detach(),
        event_before,
    )
    if not frozen_unchanged:
        raise RuntimeError('a frozen Stage3 parameter changed')
    if not event_head_changed:
        raise RuntimeError('event head did not update')

    loss_names = (
        'loss',
        'temporal_continue_loss',
        'temporal_duration_loss',
        'temporal_visible_loss',
        'temporal_progress_loss',
        'temporal_completion_loss',
        'temporal_event_time_loss',
    )
    output = {
        'checkpoint': str(args.checkpoint),
        'annotation_file': str(args.annotation_file),
        'split': args.split,
        'scene_index': args.scene_index,
        'scene_id': int(batch.annotation_id),
        'sampled_time_steps': list(batch.time_steps),
        'trainable_parameter_count': sum(
            parameter.numel() for parameter in trainable.values()
        ),
        'frozen_parameter_count': sum(
            parameter.numel() for parameter in frozen.values()
        ),
        'frozen_unchanged': frozen_unchanged,
        'event_head_changed': event_head_changed,
        'missing_keys': list(incompatible.missing_keys),
        'unexpected_keys': list(incompatible.unexpected_keys),
        'losses': {
            name: float(memo[name].detach())
            for name in loss_names
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2), flush=True)


if __name__ == '__main__':
    main()
