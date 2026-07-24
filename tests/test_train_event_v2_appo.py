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
from tools.train_event_v2_appo import (
    _load_config,
    build_appo_models_from_bootstrap,
    deterministic_scene_shards,
    run_synthetic_preflight,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / 'constellation/new_transformers/config_event_v2_appo.py'
)


def test_appo_config_preregisters_training_and_staleness_boundaries() -> None:
    config = _load_config(CONFIG)

    assert config['stage'] == 'V2-3'
    assert config['split'] == 'train'
    assert config['scene_ids'] == tuple(range(205, 325))
    assert config['gamma'] == 1.0
    assert config['encoder_unfreeze_layers'] == 1
    assert config['decoder_unfreeze_layers'] == 1
    assert config['backbone_lr_scale'] == pytest.approx(0.1)
    assert config['optimizer']['lr'] == pytest.approx(1e-6)
    assert config['actor_chunk_events'] == 32
    assert config['learner_batch_events'] == 128
    assert config['max_policy_lag'] == 2
    assert config['ppo_epochs'] == 2
    assert config['minibatch_events'] == 32


def test_scene_shards_are_deterministic_balanced_and_exhaustive() -> None:
    scene_ids = tuple(range(205, 325))

    shards = deterministic_scene_shards(scene_ids, actor_count=3)

    assert tuple(value for shard in shards for value in shard) == scene_ids
    assert [len(shard) for shard in shards] == [40, 40, 40]
    assert len(set(value for shard in shards for value in shard)) == 120
    assert deterministic_scene_shards(scene_ids, actor_count=3) == shards


@pytest.mark.parametrize('actor_count', [0, 121])
def test_scene_shards_reject_invalid_actor_counts(actor_count: int) -> None:
    with pytest.raises(ValueError, match='actor count'):
        deterministic_scene_shards(
            tuple(range(205, 325)),
            actor_count=actor_count,
        )


def test_synthetic_appo_preflight_updates_and_drops_stale_chunk(
    tmp_path,
) -> None:
    config = _load_config(CONFIG)

    summary = run_synthetic_preflight(
        config=config,
        output_dir=tmp_path,
        max_updates=3,
    )

    assert summary['accepted'] is True
    assert summary['updates'] == 3
    assert summary['policy_version'] == 3
    assert summary['accepted_events'] == 12
    assert summary['stale_dropped_events'] == 4
    assert summary['actor_replay_max_error'] <= 1e-6
    assert summary['frozen_parameter_changed_count'] == 0
    assert summary['checkpoint_first_action_reproduced'] is True
    assert pathlib.Path(summary['checkpoint']).is_file()


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


def test_appo_bootstrap_loads_v2_2_then_unfreezes_only_learner_tail(
    tmp_path,
) -> None:
    source = EventJointActorCritic(**_tiny_model_config())
    optimizer = torch.optim.AdamW(source.parameter_groups(1e-3))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    checkpoint = build_sync_ppo_checkpoint(
        stage='V2-2',
        model=source,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config_fingerprint_value=config_fingerprint({'stage': 'V2-2'}),
        normalizer={'mean': torch.zeros(1), 'std': torch.ones(1)},
        counters=SyncPPOCounters(updates=9, policy_version=9),
        scene_ids=(4,),
        runtime_states=({'cursor': 1},),
    )
    path = tmp_path / 'v2_2.pth'
    save_checkpoint_atomic(path, checkpoint)
    config = {
        'model': _tiny_model_config(),
        'encoder_unfreeze_layers': 1,
        'decoder_unfreeze_layers': 1,
    }

    actor_template, learner_model, metadata = (
        build_appo_models_from_bootstrap(
            config=config,
            checkpoint_path=path,
            learner_device=torch.device('cpu'),
        )
    )

    assert metadata.stage == 'V2-2'
    assert metadata.policy_version == 9
    assert actor_template.backbone_is_frozen is True
    assert learner_model.backbone_is_frozen is False
    assert not any(
        parameter.requires_grad
        for parameter in actor_template.backbone.transformer.parameters()
    )
    assert any(
        parameter.requires_grad
        for parameter in learner_model.backbone.transformer.parameters()
    )
    for name, value in actor_template.state_dict().items():
        torch.testing.assert_close(value, learner_model.state_dict()[name])
