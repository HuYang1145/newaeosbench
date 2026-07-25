from __future__ import annotations

import pathlib

import pytest
import torch

from constellation.new_transformers.event_v2.checkpoint import (
    SyncPPOCounters,
    build_sync_ppo_checkpoint,
    config_fingerprint,
    save_checkpoint_atomic,
)
from constellation.new_transformers.event_v2.model import (
    EventJointActorCritic,
)
from tools.train_event_v2_large_sync_ppo import (
    _load_config,
    build_large_sync_model_from_bootstrap,
    deterministic_active_environment_caps,
    deterministic_scene_assignments,
    parameter_inventory,
    resolve_actor_devices,
    run_synthetic_preflight,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / 'constellation/new_transformers/'
    'config_event_v2_large_sync_ppo.py'
)


def _tiny_model_config() -> dict:
    return {
        'event_width': 8,
        'sensor_type_embedding_dim': 4,
        'tasks_data_embedding_dim': 4,
        'encoder_width': 8,
        'encoder_depth': 1,
        'encoder_num_heads': 2,
        'sensor_enabled_embedding_dim': 4,
        'constellation_data_embedding_dim': 4,
        'decoder_width': 8,
        'decoder_depth': 1,
        'decoder_num_heads': 2,
        'use_constraint_module': False,
        'use_sdpa': False,
        'freeze_backbone': True,
    }


def _v2_2_checkpoint(path: pathlib.Path) -> None:
    model = EventJointActorCritic(**_tiny_model_config())
    optimizer = torch.optim.AdamW(model.parameter_groups(1e-3))
    loss = sum(
        parameter.square().mean()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    loss.backward()
    optimizer.step()
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda _: 1.0,
    )
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    checkpoint = build_sync_ppo_checkpoint(
        stage='V2-2',
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config_fingerprint_value=config_fingerprint(
            {'stage': 'V2-2'},
        ),
        normalizer={'mean': torch.zeros(1), 'std': torch.ones(1)},
        counters=SyncPPOCounters(
            updates=1046,
            policy_version=1046,
            processed_physical_seconds=999,
            episodes=48,
            events=66906,
        ),
        scene_ids=(4, 5),
        runtime_states=({'cursor': 1}, {'cursor': 2}),
    )
    save_checkpoint_atomic(path, checkpoint)


def test_large_sync_config_preregisters_two_gpu_single_seed_protocol() -> None:
    config = _load_config(CONFIG)

    assert config['stage'] == 'V2-2-Large'
    assert config['split'] == 'train'
    assert config['scene_ids'] == tuple(range(205, 325))
    assert config['actor_count'] == 12
    assert config['active_environments'] == 60
    assert config['events_per_actor_round'] == 8
    assert config['min_update_events'] == 64
    assert config['checkpoint_interval'] == 100
    assert config['gamma'] == 1.0
    assert config['clip_ratio'] == pytest.approx(0.2)
    assert config['max_kl'] == pytest.approx(0.03)
    assert config['ppo_epochs'] == 4
    assert config['minibatch_events'] == 16
    assert config['model']['freeze_backbone'] is True
    assert config['amp_dtype'] == 'bfloat16'
    assert config['bootstrap_checkpoint'].endswith(
        'replica_0/checkpoint_update_001046.pth',
    )


def test_scene_assignment_and_active_caps_are_deterministic_exhaustive() -> None:
    scenes = tuple(range(205, 325))

    assignments = deterministic_scene_assignments(
        scenes,
        actor_count=12,
    )
    caps = deterministic_active_environment_caps(
        assignments,
        total_active_environments=60,
    )

    assert tuple(
        scene_id
        for actor_id in assignments
        for scene_id in assignments[actor_id]
    ) == scenes
    assert set(assignments) == set(range(12))
    assert {len(shard) for shard in assignments.values()} == {10}
    assert caps == {actor_id: 5 for actor_id in range(12)}
    assert sum(caps.values()) == 60


@pytest.mark.parametrize(
    ('actor_count', 'active_environments', 'message'),
    [
        (0, 60, 'actor count'),
        (121, 60, 'actor count'),
        (12, 11, 'active environment'),
        (12, 121, 'active environment'),
    ],
)
def test_scene_assignment_rejects_invalid_resource_boundaries(
    actor_count: int,
    active_environments: int,
    message: str,
) -> None:
    scenes = tuple(range(205, 325))
    with pytest.raises(ValueError, match=message):
        assignments = deterministic_scene_assignments(
            scenes,
            actor_count=actor_count,
        )
        deterministic_active_environment_caps(
            assignments,
            total_active_environments=active_environments,
        )


def test_actor_devices_repeat_over_two_gpus_without_changing_actor_count(
) -> None:
    assert resolve_actor_devices(
        ('cuda:0', 'cuda:1'),
        actor_count=12,
    ) == tuple(
        f'cuda:{actor_id % 2}' for actor_id in range(12)
    )
    with pytest.raises(ValueError, match='actor device'):
        resolve_actor_devices((), actor_count=12)


def test_bootstrap_inherits_v2_2_model_optimizer_but_resets_progress_and_rng(
    tmp_path: pathlib.Path,
) -> None:
    source = tmp_path / 'v2_2.pth'
    _v2_2_checkpoint(source)
    torch.manual_seed(999)
    rng_before = torch.get_rng_state().clone()

    model, optimizer, metadata = build_large_sync_model_from_bootstrap(
        config={
            'model': _tiny_model_config(),
            'optimizer': {
                'lr': 1e-3,
                'betas': (0.9, 0.98),
                'weight_decay': 1e-4,
                'eps': 1e-8,
            },
        },
        checkpoint_path=source,
        device=torch.device('cpu'),
    )

    assert metadata.source_stage == 'V2-2'
    assert metadata.source_updates == 1046
    assert metadata.source_policy_version == 1046
    assert model.backbone_is_frozen is True
    assert optimizer.state_dict()['state']
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_parameter_inventory_records_full_frozen_and_trainable_boundaries(
) -> None:
    model = EventJointActorCritic(
        event_width=256,
        freeze_backbone=True,
        use_constraint_module=True,
        use_sdpa=True,
    )

    inventory = parameter_inventory(model)

    assert inventory['total_parameters'] == 93_056_272
    assert inventory['trainable_parameters'] == 1_674_507
    assert inventory['frozen_parameters'] == (
        inventory['total_parameters']
        - inventory['trainable_parameters']
    )
    assert inventory['transformer_trainable_parameters'] == 0
    assert inventory['trainable_parameter_names']
    assert inventory['frozen_parameter_names']
    assert set(inventory['trainable_parameter_names']).isdisjoint(
        inventory['frozen_parameter_names'],
    )


def test_synthetic_large_sync_preflight_runs_two_exact_barrier_updates(
    tmp_path: pathlib.Path,
) -> None:
    config = _load_config(CONFIG)

    summary = run_synthetic_preflight(
        config=config,
        output_dir=tmp_path,
        max_updates=2,
    )

    assert summary['accepted'] is True
    assert summary['updates'] == 2
    assert summary['policy_version'] == 2
    assert summary['next_round_id'] == 2
    assert summary['events'] == 16
    assert summary['stale_rollout_events'] == 0
    assert summary['logprob_replay_max_error'] <= 1e-6
    assert summary['reward_reconstruction_max_error'] <= 1e-6
    assert summary['frozen_parameter_changed_count'] == 0
    assert pathlib.Path(summary['checkpoint']).is_file()
    assert (tmp_path / 'checkpoint_latest.pth').is_file()
