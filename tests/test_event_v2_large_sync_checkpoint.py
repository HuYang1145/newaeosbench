from __future__ import annotations

import pathlib
import random

import numpy as np
import pytest
import torch

from constellation.new_transformers.event_v2.checkpoint import (
    config_fingerprint,
    restore_rng_state as restore_learner_rng_state,
)
from constellation.new_transformers.event_v2.distributed_sync import (
    capture_rng_state,
    restore_rng_state as restore_actor_rng_state,
)
from constellation.new_transformers.event_v2.large_sync_checkpoint import (
    LargeSyncCounters,
    build_large_sync_checkpoint_payload,
    load_large_sync_checkpoint,
    save_large_sync_checkpoint,
    update_latest_checkpoint,
)
from constellation.new_transformers.event_v2.model import (
    EventJointActorCritic,
)
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
    )


def _training_objects():
    model = _model()
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=10,
    )
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    return model, optimizer, scheduler, scaler


def _parameter_names(model: EventJointActorCritic):
    trainable = tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    frozen = tuple(
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    )
    return trainable, frozen


def _actor_states():
    return {
        0: {
            'pool': {
                'assigned_scene_ids': (205, 206),
                'pending_scene_ids': (206,),
                'active': ({'scene_id': 205, 'cursor': 7},),
                'completed_scene_ids': (),
            },
            'rng': capture_rng_state(),
        },
        1: {
            'pool': {
                'assigned_scene_ids': (207, 208),
                'pending_scene_ids': (),
                'active': ({'scene_id': 208, 'cursor': 2},),
                'completed_scene_ids': (207,),
            },
            'rng': capture_rng_state(),
        },
    }


@pytest.mark.parametrize(
    'restore_rng_state',
    [restore_learner_rng_state, restore_actor_rng_state],
)
def test_rng_restore_uses_only_states_for_visible_cuda_devices(
    monkeypatch: pytest.MonkeyPatch,
    restore_rng_state,
) -> None:
    restored_cuda_states = []
    first_cuda_state = torch.tensor([1], dtype=torch.uint8)
    second_cuda_state = torch.tensor([2], dtype=torch.uint8)
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': (first_cuda_state, second_cuda_state),
    }
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        'set_rng_state_all',
        lambda values: restored_cuda_states.extend(values),
    )

    restore_rng_state(state)

    assert len(restored_cuda_states) == 1
    torch.testing.assert_close(restored_cuda_states[0], first_cuda_state)


def _payload():
    model, optimizer, scheduler, scaler = _training_objects()
    trainable, frozen = _parameter_names(model)
    assignments = {0: (205, 206), 1: (207, 208)}
    payload = build_large_sync_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config_fingerprint_value=config_fingerprint(
            {'stage': 'V2-2-Large', 'seed': 5408},
        ),
        normalizer={'mean': torch.zeros(2), 'std': torch.ones(2)},
        counters=LargeSyncCounters(
            next_round_id=3,
            updates=3,
            policy_version=3,
            processed_physical_seconds=480,
            episodes=1,
            events=96,
        ),
        actor_scene_assignments=assignments,
        actor_states=_actor_states(),
        trainable_parameter_names=trainable,
        frozen_parameter_names=frozen,
        bootstrap={
            'path': 'checkpoint_update_001046.pth',
            'stage': 'V2-2',
            'updates': 1046,
        },
    )
    return (
        model,
        optimizer,
        scheduler,
        scaler,
        trainable,
        frozen,
        assignments,
        payload,
    )


def test_large_sync_checkpoint_round_trip_restores_every_barrier_state(
    tmp_path: pathlib.Path,
) -> None:
    random.seed(5408)
    np.random.seed(5408)
    torch.manual_seed(5408)
    (
        model,
        optimizer,
        scheduler,
        scaler,
        trainable,
        frozen,
        assignments,
        payload,
    ) = _payload()
    path = tmp_path / 'checkpoint_update_000003.pth'
    save_large_sync_checkpoint(path, payload=payload)
    expected_random = torch.rand(4)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(1)
    torch.rand(4)

    restored = load_large_sync_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        expected_schema_fingerprint=transition_schema_fingerprint(),
        expected_config_fingerprint=payload['config_fingerprint'],
        expected_scene_assignments=assignments,
        expected_trainable_parameter_names=trainable,
        expected_frozen_parameter_names=frozen,
    )
    actual_random = torch.rand(4)

    assert restored.counters == LargeSyncCounters(
        next_round_id=3,
        updates=3,
        policy_version=3,
        processed_physical_seconds=480,
        episodes=1,
        events=96,
    )
    assert restored.actor_scene_assignments == assignments
    assert set(restored.actor_states) == {0, 1}
    assert restored.actor_states[0]['pool']['pending_scene_ids'] == (206,)
    assert restored.normalizer['std'].tolist() == [1.0, 1.0]
    assert restored.bootstrap['updates'] == 1046
    torch.testing.assert_close(actual_random, expected_random)


def test_large_sync_checkpoint_preserves_all_model_parameters() -> None:
    model, *_, trainable, frozen, _, payload = _payload()

    assert payload['trainable_parameter_names'] == trainable
    assert payload['frozen_parameter_names'] == frozen
    assert set(trainable).isdisjoint(frozen)
    assert set(trainable) | set(frozen) == {
        name for name, _ in model.named_parameters()
    }
    assert payload['freeze_state'] == {
        'backbone_is_frozen': True,
        'transformer_trainable_parameter_count': 0,
    }
    assert payload['barrier_complete'] is True


@pytest.mark.parametrize(
    ('field', 'value', 'message'),
    [
        ('transition_schema_fingerprint', 'bad', 'schema'),
        ('config_fingerprint', 'bad', 'config'),
        ('barrier_complete', False, 'barrier'),
        (
            'trainable_parameter_names',
            ('wrong.parameter',),
            'trainable',
        ),
    ],
)
def test_large_sync_checkpoint_rejects_incompatible_metadata(
    tmp_path: pathlib.Path,
    field: str,
    value,
    message: str,
) -> None:
    (
        model,
        optimizer,
        scheduler,
        scaler,
        trainable,
        frozen,
        assignments,
        payload,
    ) = _payload()
    payload[field] = value
    path = tmp_path / 'bad.pth'
    torch.save(payload, path)

    with pytest.raises(ValueError, match=message):
        load_large_sync_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_schema_fingerprint=transition_schema_fingerprint(),
            expected_config_fingerprint=config_fingerprint(
                {'stage': 'V2-2-Large', 'seed': 5408},
            ),
            expected_scene_assignments=assignments,
            expected_trainable_parameter_names=trainable,
            expected_frozen_parameter_names=frozen,
        )


def test_large_sync_checkpoint_rejects_scene_or_freeze_boundary_drift(
    tmp_path: pathlib.Path,
) -> None:
    (
        model,
        optimizer,
        scheduler,
        scaler,
        trainable,
        frozen,
        assignments,
        payload,
    ) = _payload()
    path = tmp_path / 'checkpoint.pth'
    save_large_sync_checkpoint(path, payload=payload)

    with pytest.raises(ValueError, match='scene assignment'):
        load_large_sync_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_schema_fingerprint=transition_schema_fingerprint(),
            expected_config_fingerprint=payload['config_fingerprint'],
            expected_scene_assignments={0: (205,), 1: (207, 208)},
            expected_trainable_parameter_names=trainable,
            expected_frozen_parameter_names=frozen,
        )
    with pytest.raises(ValueError, match='frozen'):
        load_large_sync_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_schema_fingerprint=transition_schema_fingerprint(),
            expected_config_fingerprint=payload['config_fingerprint'],
            expected_scene_assignments=assignments,
            expected_trainable_parameter_names=trainable,
            expected_frozen_parameter_names=frozen[:-1],
        )


def test_permanent_checkpoint_is_never_overwritten_and_latest_is_atomic(
    tmp_path: pathlib.Path,
) -> None:
    *_, payload = _payload()
    permanent = tmp_path / 'checkpoint_update_000003.pth'
    latest = tmp_path / 'checkpoint_latest.pth'
    save_large_sync_checkpoint(permanent, payload=payload)

    with pytest.raises(FileExistsError, match='already exists'):
        save_large_sync_checkpoint(permanent, payload=payload)
    update_latest_checkpoint(source=permanent, latest=latest)

    assert latest.is_file()
    assert permanent.stat().st_ino == latest.stat().st_ino
    assert not latest.with_suffix('.pth.tmp').exists()
    loaded = torch.load(latest, map_location='cpu', weights_only=False)
    assert loaded['updates'] == 3
