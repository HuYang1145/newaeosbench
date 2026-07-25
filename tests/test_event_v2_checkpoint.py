import pathlib
import random

import numpy as np
import pytest
import torch

from constellation.new_transformers.event_v2.checkpoint import (
    APPOCounters,
    build_appo_checkpoint,
    SyncPPOCounters,
    build_sync_ppo_checkpoint,
    config_fingerprint,
    load_appo_checkpoint,
    load_appo_policy_checkpoint,
    load_sync_ppo_bootstrap_checkpoint,
    load_sync_ppo_checkpoint,
    load_sync_ppo_policy_checkpoint,
    save_checkpoint_atomic,
)
from constellation.new_transformers.event_v2.model import EventJointActorCritic
from constellation.new_transformers.event_v2.observation import (
    EventPolicyObservation,
)
from constellation.new_transformers.event_v2.state import EventStateTensors
from constellation.new_transformers.event_v2.transition import (
    transition_schema_fingerprint,
)


def _model() -> EventJointActorCritic:
    return EventJointActorCritic(
        event_width=8,
        sensor_type_embedding_dim=4,
        tasks_data_embedding_dim=4,
        encoder_width=8,
        encoder_depth=1,
        encoder_num_heads=2,
        sensor_enabled_embedding_dim=4,
        constellation_data_embedding_dim=4,
        decoder_width=8,
        decoder_depth=1,
        decoder_num_heads=2,
        use_constraint_module=False,
        use_sdpa=False,
        freeze_backbone=True,
    ).eval()


def _observation() -> EventPolicyObservation:
    satellite_shape = (1, 2)
    task_shape = (1, 3)
    state = EventStateTensors(
        previous_task_indices=torch.full(satellite_shape, -1, dtype=torch.long),
        current_task_indices=torch.full(satellite_shape, -1, dtype=torch.long),
        minimum_commitment_remaining=torch.zeros(satellite_shape),
        run_lengths=torch.zeros(satellite_shape),
        seconds_since_replan=torch.zeros(satellite_shape),
        switch_count_30=torch.zeros(satellite_shape),
        switch_count_60=torch.zeros(satellite_shape),
        termination_reason=torch.zeros(satellite_shape, dtype=torch.long),
        event_type=torch.zeros(satellite_shape, dtype=torch.long),
        delta_t=torch.zeros(satellite_shape),
        replan_mask=torch.ones(satellite_shape, dtype=torch.bool),
        forced_interrupt_mask=torch.zeros(satellite_shape, dtype=torch.bool),
        can_terminate_mask=torch.zeros(satellite_shape, dtype=torch.bool),
        compatible_deadline_slack=torch.ones(satellite_shape),
        task_remaining_required_seconds=torch.tensor([[10., 20., 30.]]),
        task_owner_count=torch.zeros(task_shape, dtype=torch.long),
        task_locked_owner_count=torch.zeros(task_shape, dtype=torch.long),
    )
    return EventPolicyObservation(
        time_steps=torch.tensor([0]),
        constellation_sensor_type=torch.zeros(satellite_shape, dtype=torch.long),
        constellation_sensor_enabled=torch.ones(satellite_shape, dtype=torch.long),
        constellation_data=torch.zeros(1, 2, 56),
        constellation_mask=torch.ones(satellite_shape, dtype=torch.bool),
        tasks_sensor_type=torch.zeros(task_shape, dtype=torch.long),
        tasks_data=torch.zeros(1, 3, 6),
        tasks_mask=torch.ones(task_shape, dtype=torch.bool),
        event_state=state,
    )


def _training_objects():
    model = _model()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    return model, optimizer, scheduler, scaler


def _sample(model: EventJointActorCritic):
    observation = _observation()
    with torch.inference_mode():
        return model.act(
            *observation.model_args(),
            event_state=observation.event_state,
            deterministic=False,
        )


def _build_checkpoint():
    model, optimizer, scheduler, scaler = _training_objects()
    config = {'stage': 'V2-1', 'seed': 3407, 'path': pathlib.Path('train')}
    checkpoint = build_sync_ppo_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config_fingerprint_value=config_fingerprint(config),
        normalizer={'mean': torch.zeros(2), 'std': torch.ones(2)},
        counters=SyncPPOCounters(
            updates=3,
            policy_version=3,
            processed_physical_seconds=120,
            episodes=2,
            events=24,
        ),
        scene_ids=(0, 1),
        runtime_states=({'cursor': 12}, {'cursor': 8}),
    )
    return model, optimizer, scheduler, scaler, config, checkpoint


def test_checkpoint_restores_rng_metadata_and_first_actions(tmp_path) -> None:
    random.seed(3407)
    np.random.seed(3407)
    torch.manual_seed(3407)
    model, optimizer, scheduler, scaler, config, checkpoint = _build_checkpoint()
    path = tmp_path / 'v2_1.pth'
    save_checkpoint_atomic(path, checkpoint)
    expected = _sample(model)

    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(1)
    random.random()
    np.random.rand()
    torch.rand(3)
    restored = load_sync_ppo_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        expected_config_fingerprint=config_fingerprint(config),
        expected_scene_ids=(0, 1),
    )
    actual = _sample(model)

    assert restored.counters.policy_version == 3
    assert restored.counters.events == 24
    assert restored.runtime_states == ({'cursor': 12}, {'cursor': 8})
    torch.testing.assert_close(actual.actor.log_prob, expected.actor.log_prob)
    torch.testing.assert_close(actual.value, expected.value)
    assert torch.equal(actual.actor.action.terminate, expected.actor.action.terminate)
    assert torch.equal(actual.actor.action.task_indices, expected.actor.action.task_indices)
    assert torch.equal(
        actual.actor.action.commitment_indices,
        expected.actor.action.commitment_indices,
    )
    assert not path.with_suffix('.pth.tmp').exists()


@pytest.mark.parametrize(
    ('field', 'value', 'message'),
    [
        ('stage', 'V2-0', 'stage'),
        ('transition_schema_fingerprint', 'bad', 'schema'),
        ('config_fingerprint', 'bad', 'config'),
        ('scene_ids', (9,), 'scene'),
    ],
)
def test_checkpoint_rejects_incompatible_metadata(
    tmp_path,
    field: str,
    value,
    message: str,
) -> None:
    model, optimizer, scheduler, scaler, config, checkpoint = _build_checkpoint()
    checkpoint[field] = value
    path = tmp_path / 'bad.pth'
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match=message):
        load_sync_ppo_checkpoint(
            path=path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_config_fingerprint=config_fingerprint(config),
            expected_scene_ids=(0, 1),
        )


def test_checkpoint_uses_current_transition_schema() -> None:
    *_, checkpoint = _build_checkpoint()

    assert checkpoint['transition_schema_fingerprint'] == (
        transition_schema_fingerprint()
    )


def test_checkpoint_round_trip_uses_requested_sync_ppo_stage(tmp_path) -> None:
    model, optimizer, scheduler, scaler, config, _ = _build_checkpoint()
    checkpoint = build_sync_ppo_checkpoint(
        stage='V2-2',
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config_fingerprint_value=config_fingerprint(config),
        normalizer={'mean': torch.zeros(2), 'std': torch.ones(2)},
        counters=SyncPPOCounters(updates=7, policy_version=7),
        scene_ids=(4, 5),
        runtime_states=({'cursor': 4}, {'cursor': 5}),
    )
    path = tmp_path / 'v2_2.pth'
    save_checkpoint_atomic(path, checkpoint)

    restored = load_sync_ppo_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        expected_stage='V2-2',
        expected_config_fingerprint=config_fingerprint(config),
        expected_scene_ids=(4, 5),
    )

    assert checkpoint['stage'] == 'V2-2'
    assert restored.counters.updates == 7
    with pytest.raises(ValueError, match='stage'):
        load_sync_ppo_checkpoint(
            path=path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_stage='V2-1',
            expected_config_fingerprint=config_fingerprint(config),
            expected_scene_ids=(4, 5),
        )


def test_bootstrap_loads_v2_1_policy_optimizer_without_runtime_or_rng(
    tmp_path,
) -> None:
    source_model, source_optimizer, _, _, _, checkpoint = _build_checkpoint()
    source_parameters = {
        name: value.detach().clone()
        for name, value in source_model.state_dict().items()
    }
    path = tmp_path / 'v2_1_source.pth'
    save_checkpoint_atomic(path, checkpoint)

    target_model, target_optimizer, _, _ = _training_objects()
    with torch.no_grad():
        for parameter in target_model.parameters():
            if parameter.requires_grad:
                parameter.add_(1)
    target_optimizer.param_groups[0]['lr'] = 9e-4
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    rng_before = torch.get_rng_state().clone()

    bootstrap = load_sync_ppo_bootstrap_checkpoint(
        path=path,
        model=target_model,
        optimizer=target_optimizer,
        expected_source_stage='V2-1',
    )

    assert bootstrap.source_stage == 'V2-1'
    assert bootstrap.source_updates == 3
    assert bootstrap.source_policy_version == 3
    assert bootstrap.source_scene_ids == (0, 1)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert target_optimizer.param_groups[0]['lr'] == pytest.approx(
        source_optimizer.param_groups[0]['lr'],
    )
    for name, value in target_model.state_dict().items():
        torch.testing.assert_close(value, source_parameters[name])


def test_policy_loader_only_restores_model_and_returns_metadata(tmp_path) -> None:
    source_model, _, _, _, _, checkpoint = _build_checkpoint()
    checkpoint['stage'] = 'V2-2'
    checkpoint['updates'] = 914
    checkpoint['policy_version'] = 914
    path = tmp_path / 'v2_2_policy.pth'
    save_checkpoint_atomic(path, checkpoint)

    target_model = _model()
    with torch.no_grad():
        for parameter in target_model.parameters():
            if parameter.requires_grad:
                parameter.add_(1)
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    rng_before = torch.get_rng_state().clone()

    metadata = load_sync_ppo_policy_checkpoint(
        path=path,
        model=target_model,
        expected_stages=('V2-1', 'V2-2'),
    )

    assert metadata.stage == 'V2-2'
    assert metadata.updates == 914
    assert metadata.policy_version == 914
    assert metadata.scene_ids == (0, 1)
    assert metadata.config_fingerprint == checkpoint['config_fingerprint']
    assert torch.equal(torch.get_rng_state(), rng_before)
    for name, value in target_model.state_dict().items():
        torch.testing.assert_close(value, source_model.state_dict()[name])


@pytest.mark.parametrize(
    ('field', 'value', 'message'),
    [
        ('checkpoint_version', 99, 'version'),
        ('stage', 'V2-0', 'stage'),
        ('transition_schema_fingerprint', 'bad', 'schema'),
        ('unfreeze_state', {'backbone_is_frozen': False}, 'freeze'),
    ],
)
def test_policy_loader_rejects_incompatible_checkpoint(
    tmp_path,
    field: str,
    value,
    message: str,
) -> None:
    model, _, _, _, _, checkpoint = _build_checkpoint()
    checkpoint[field] = value
    path = tmp_path / 'bad_policy.pth'
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match=message):
        load_sync_ppo_policy_checkpoint(
            path=path,
            model=model,
            expected_stages=('V2-1', 'V2-2'),
        )


def _appo_training_objects():
    model = _model()
    model.unfreeze_last_layers(encoder_layers=1, decoder_layers=1)
    optimizer = torch.optim.AdamW(
        model.parameter_groups(
            new_module_lr=1e-3,
            backbone_lr_scale=0.1,
        ),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    return model, optimizer, scheduler, scaler


def _build_appo_test_checkpoint():
    model, optimizer, scheduler, scaler = _appo_training_objects()
    config = {'stage': 'V2-3', 'seed': 5407}
    scene_shards = ((205, 207), (206,))
    runtime_states = (
        ({'scene_id': 205, 'cursor': 12}, {'scene_id': 207, 'cursor': 8}),
        ({'scene_id': 206, 'cursor': 5},),
    )
    checkpoint = build_appo_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config_fingerprint_value=config_fingerprint(config),
        normalizer={'mean': torch.zeros(2), 'std': torch.ones(2)},
        counters=APPOCounters(
            updates=4,
            policy_version=4,
            accepted_events=512,
            stale_dropped_events=32,
            processed_physical_seconds=2400,
            episodes=1,
        ),
        actor_scene_shards=scene_shards,
        actor_runtime_states=runtime_states,
        encoder_layers=1,
        decoder_layers=1,
        backbone_lr_scale=0.1,
    )
    return (
        model,
        optimizer,
        scheduler,
        scaler,
        config,
        checkpoint,
        scene_shards,
        runtime_states,
    )


def test_appo_checkpoint_restores_training_and_actor_runtime_state(
    tmp_path,
) -> None:
    (
        model,
        optimizer,
        scheduler,
        scaler,
        config,
        checkpoint,
        scene_shards,
        runtime_states,
    ) = _build_appo_test_checkpoint()
    expected = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    path = tmp_path / 'v2_3.pth'
    save_checkpoint_atomic(path, checkpoint)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(1)

    restored = load_appo_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        expected_config_fingerprint=config_fingerprint(config),
        expected_actor_scene_shards=scene_shards,
        expected_encoder_layers=1,
        expected_decoder_layers=1,
        expected_backbone_lr_scale=0.1,
    )

    assert restored.counters.policy_version == 4
    assert restored.counters.accepted_events == 512
    assert restored.counters.stale_dropped_events == 32
    assert restored.actor_scene_shards == scene_shards
    assert restored.actor_runtime_states == runtime_states
    assert restored.pending_steps == ()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name])


def test_appo_policy_loader_only_restores_model_without_rng_or_runtime(
    tmp_path,
) -> None:
    (
        source_model,
        _,
        _,
        _,
        _,
        checkpoint,
        scene_shards,
        _,
    ) = _build_appo_test_checkpoint()
    checkpoint['updates'] = 832
    checkpoint['policy_version'] = 832
    path = tmp_path / 'v2_3_policy.pth'
    save_checkpoint_atomic(path, checkpoint)

    target_model, _, _, _ = _appo_training_objects()
    with torch.no_grad():
        for parameter in target_model.parameters():
            if parameter.requires_grad:
                parameter.add_(1)
    torch.manual_seed(999)
    rng_before = torch.get_rng_state().clone()

    metadata = load_appo_policy_checkpoint(
        path=path,
        model=target_model,
        expected_encoder_layers=1,
        expected_decoder_layers=1,
        expected_backbone_lr_scale=0.1,
    )

    assert metadata.stage == 'V2-3'
    assert metadata.updates == 832
    assert metadata.policy_version == 832
    assert metadata.scene_ids == tuple(
        scene_id for shard in scene_shards for scene_id in shard
    )
    assert metadata.encoder_layers == 1
    assert metadata.decoder_layers == 1
    assert metadata.backbone_lr_scale == pytest.approx(0.1)
    assert torch.equal(torch.get_rng_state(), rng_before)
    for name, value in target_model.state_dict().items():
        torch.testing.assert_close(value, source_model.state_dict()[name])


@pytest.mark.parametrize(
    ('field', 'value', 'message'),
    [
        ('checkpoint_version', 99, 'version'),
        ('stage', 'V2-2', 'stage'),
        ('transition_schema_fingerprint', 'bad', 'schema'),
        ('config_fingerprint', 'bad', 'config'),
        ('actor_scene_shards', ((999,),), 'scene'),
        (
            'unfreeze_state',
            {
                'backbone_is_frozen': False,
                'encoder_layers': 2,
                'decoder_layers': 1,
                'backbone_lr_scale': 0.1,
            },
            'unfreeze',
        ),
    ],
)
def test_appo_checkpoint_rejects_incompatible_metadata(
    tmp_path,
    field: str,
    value,
    message: str,
) -> None:
    (
        model,
        optimizer,
        scheduler,
        scaler,
        config,
        checkpoint,
        scene_shards,
        _,
    ) = _build_appo_test_checkpoint()
    checkpoint[field] = value
    path = tmp_path / 'bad_v2_3.pth'
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match=message):
        load_appo_checkpoint(
            path=path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_config_fingerprint=config_fingerprint(config),
            expected_actor_scene_shards=scene_shards,
            expected_encoder_layers=1,
            expected_decoder_layers=1,
            expected_backbone_lr_scale=0.1,
        )
