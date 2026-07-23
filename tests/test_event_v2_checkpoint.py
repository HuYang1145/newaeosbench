import pathlib
import random

import numpy as np
import pytest
import torch

from constellation.new_transformers.event_v2.checkpoint import (
    SyncPPOCounters,
    build_sync_ppo_checkpoint,
    config_fingerprint,
    load_sync_ppo_bootstrap_checkpoint,
    load_sync_ppo_checkpoint,
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
