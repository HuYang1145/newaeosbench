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
    load_appo_policy_checkpoint,
    load_sync_ppo_policy_checkpoint,
)
from constellation.new_transformers.event_v2.large_sync_checkpoint import (
    load_large_sync_policy_checkpoint,
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
    if all('TAT_s' in row and 'PC_Wh' in row for row in rows):
        means['TAT_s'] = sum(
            float(row['TAT_s']) for row in rows
        ) / len(rows)
        means['PC_Wh'] = sum(
            float(row['PC_Wh']) for row in rows
        ) / len(rows)
        means['CS_paper'] = (
            float('inf')
            if means['Q'] <= 0
            else (
                1 / means['Q']
                + means['TAT_s'] / 700
                + means['PC_Wh'] / 100
            )
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
    operational_metrics = getattr(
        runtime.backend,
        'operational_metrics',
        None,
    )
    if not callable(operational_metrics):
        raise RuntimeError(
            'evaluation backend does not expose operational metrics',
        )
    operational = dict(operational_metrics())
    tat_s = float(operational['TAT_s'])
    pc_wh = float(operational['PC_Wh'])
    cs_paper = (
        float('inf')
        if metrics['Q'] <= 0
        else 1 / metrics['Q'] + tat_s / 700 + pc_wh / 100
    )
    return {
        **metrics,
        'TAT_s': tat_s,
        'PC_Wh': pc_wh,
        'CS_paper': cs_paper,
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


def load_policy_for_evaluation(
    *,
    path: pathlib.Path,
    model: EventJointActorCritic,
):
    """按 checkpoint 阶段只读加载 policy，并关闭所有评估梯度。"""

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError('policy checkpoint root must be a mapping')
    stage = checkpoint.get('stage')
    if stage == 'V2-3':
        model.unfreeze_last_layers(
            encoder_layers=1,
            decoder_layers=1,
        )
        metadata = load_appo_policy_checkpoint(
            path=path,
            model=model,
            expected_encoder_layers=1,
            expected_decoder_layers=1,
            expected_backbone_lr_scale=0.1,
        )
    elif stage == 'V2-2-Large':
        metadata = load_large_sync_policy_checkpoint(
            path=path,
            model=model,
        )
    else:
        metadata = load_sync_ppo_policy_checkpoint(
            path=path,
            model=model,
            expected_stages=('V2-1', 'V2-2'),
        )
    model.requires_grad_(False)
    model.eval()
    return metadata


def main() -> None:
    args = parse_args()
    if len(set(args.scene_ids)) != len(args.scene_ids):
        raise ValueError('scene IDs must be unique')
    if args.max_time_step <= 0:
        raise ValueError('max time step must be positive')
    config = _load_config(args.config)
    device = _resolve_device(args.device)
    model = EventJointActorCritic(**config['model']).to(device)
    metadata = load_policy_for_evaluation(
        path=args.checkpoint,
        model=model,
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
            for name in (
                *METRIC_NAMES,
                'TAT_s',
                'PC_Wh',
                'CS_paper',
            )
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
