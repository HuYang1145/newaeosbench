#!/usr/bin/env python3
"""确定性评估单个 Event Transformer V2 同步 PPO checkpoint。"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import pathlib
import runpy
import sys
from typing import Any

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from constellation.new_transformers.event_v2.basilisk_runtime import (
    BasiliskEventRuntime,
    BasiliskSceneBackend,
    CompletionSnapshot,
    load_runtime_statistics,
)
from constellation.new_transformers.event_v2.checkpoint import (
    load_sync_ppo_policy_checkpoint,
)
from constellation.new_transformers.event_v2.model import (
    EventJointActorCritic,
)
from constellation.new_transformers.event_v2.transition import (
    JointEventAction,
)


METRIC_NAMES = ('CR', 'PCR', 'WCR', 'Q')


def completion_metrics(
    snapshot: CompletionSnapshot,
) -> dict[str, float]:
    snapshot.validate()
    progress_ratio = (
        snapshot.progress / snapshot.required_duration
    ).clamp(0, 1)
    completed = snapshot.completed.to(snapshot.required_duration.dtype)
    cr = float(completed.mean())
    pcr = float(progress_ratio.mean())
    wcr = float(
        (completed * snapshot.required_duration).sum()
        / snapshot.required_duration.sum()
    )
    return {
        'CR': cr,
        'PCR': pcr,
        'WCR': wcr,
        'Q': 0.6 * cr + 0.2 * pcr + 0.2 * wcr,
    }


def aggregate_scene_metrics(
    rows: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    if not rows:
        raise ValueError('at least one scene metric row is required')
    means = {
        name: sum(float(row[name]) for row in rows) / len(rows)
        for name in ('CR', 'PCR', 'WCR')
    }
    means['Q'] = (
        0.6 * means['CR']
        + 0.2 * means['PCR']
        + 0.2 * means['WCR']
    )
    return means


def _detach_action(action: JointEventAction) -> JointEventAction:
    return JointEventAction(
        terminate=action.terminate.detach().to('cpu').clone(),
        task_indices=action.task_indices.detach().to('cpu').clone(),
        commitment_indices=(
            action.commitment_indices.detach().to('cpu').clone()
        ),
    )


def evaluate_runtime(
    *,
    model: EventJointActorCritic,
    runtime: BasiliskEventRuntime,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, float | int]:
    observation = runtime.reset()
    events = 0
    physical_seconds = 0
    final_quality = None
    model.eval()
    with torch.inference_mode():
        while True:
            observation_device = observation.to(device)
            with torch.autocast(
                device_type=device.type,
                enabled=amp_enabled,
                dtype=amp_dtype,
            ):
                output = model.act(
                    *observation_device.model_args(),
                    event_state=observation_device.event_state,
                    deterministic=True,
                )
            result = runtime.step(_detach_action(output.actor.action))
            if result.invalid_action_count != 0:
                raise RuntimeError('deterministic policy produced invalid action')
            if result.delta_t <= 0:
                raise RuntimeError('deterministic runtime did not advance time')
            events += 1
            physical_seconds += int(result.delta_t)
            if result.done:
                final_quality = result.final_quality
                break
            if result.observation is None:
                raise RuntimeError('non-terminal evaluation has no observation')
            observation = result.observation

    metrics = completion_metrics(runtime.backend.completion_snapshot())
    if final_quality is None or abs(metrics['Q'] - final_quality) > 1e-6:
        raise RuntimeError('terminal quality does not match completion metrics')
    return {
        **metrics,
        'events': events,
        'physical_seconds': physical_seconds,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Deterministically evaluate one Event V2 checkpoint',
    )
    parser.add_argument('--config', type=pathlib.Path, required=True)
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument(
        '--split',
        choices=('train', 'val_seen', 'val_unseen', 'test'),
        required=True,
    )
    parser.add_argument('--scene-ids', type=int, nargs='+', required=True)
    parser.add_argument('--max-time-step', type=int, default=3600)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output', type=pathlib.Path, required=True)
    return parser.parse_args()


def _load_config(path: pathlib.Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f'config not found: {path}')
    return {
        key: value
        for key, value in runpy.run_path(str(path)).items()
        if not key.startswith('__')
    }


def _resolve_device(value: str) -> torch.device:
    if value == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(value)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    return device


def main() -> None:
    args = parse_args()
    if len(set(args.scene_ids)) != len(args.scene_ids):
        raise ValueError('scene IDs must be unique')
    if args.max_time_step <= 0:
        raise ValueError('max time step must be positive')
    config = _load_config(args.config)
    device = _resolve_device(args.device)
    model = EventJointActorCritic(**config['model']).to(device)
    metadata = load_sync_ppo_policy_checkpoint(
        path=args.checkpoint,
        model=model,
        expected_stages=('V2-1', 'V2-2'),
    )
    amp_enabled = bool(config.get('amp', True) and device.type == 'cuda')
    amp_dtype = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
    }[str(config.get('amp_dtype', 'bfloat16'))]
    statistics = load_runtime_statistics()
    scene_rows = []
    for scene_id in args.scene_ids:
        runtime = BasiliskEventRuntime(
            backend=BasiliskSceneBackend.from_scene_id(
                split=args.split,
                scene_id=scene_id,
                max_time_step=args.max_time_step,
            ),
            statistics=statistics,
            safety_review_seconds=int(config['safety_review_seconds']),
        )
        row = {
            'scene_id': scene_id,
            **evaluate_runtime(
                model=model,
                runtime=runtime,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            ),
            'reward_reconstruction_error': (
                runtime.reward_reconstruction_error
            ),
        }
        scene_rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    aggregate = aggregate_scene_metrics(scene_rows)
    summary = {
        'label': args.label,
        'checkpoint': str(args.checkpoint.resolve()),
        'stage': metadata.stage,
        'checkpoint_updates': metadata.updates,
        'checkpoint_policy_version': metadata.policy_version,
        'checkpoint_train_scene_ids': list(metadata.scene_ids),
        'config_fingerprint': metadata.config_fingerprint,
        'split': args.split,
        'scene_ids': list(args.scene_ids),
        'max_time_step': args.max_time_step,
        'deterministic': True,
        'amp_enabled': amp_enabled,
        'amp_dtype': (
            str(config.get('amp_dtype', 'bfloat16'))
            if amp_enabled
            else None
        ),
        'scenes': scene_rows,
        'aggregate': aggregate,
        'finite': all(
            torch.isfinite(torch.tensor(row[name]))
            for row in scene_rows
            for name in METRIC_NAMES
        ),
        'reward_reconstruction_max_error': max(
            float(row['reward_reconstruction_error'])
            for row in scene_rows
        ),
    }
    output_path = (
        args.output
        if args.output.suffix == '.json'
        else args.output / 'summary.json'
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n',
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
