import random

import numpy as np
import pytest
import torch
from todd.configs import PyConfig

from constellation.new_transformers.event_v2.model import (
    EventJointActorCritic,
)
from constellation.new_transformers.event_v2.transition import (
    transition_schema_fingerprint,
)
from tools.train_event_v2_warm_start import (
    TrainingCounters,
    build_training_checkpoint,
    config_fingerprint,
    load_training_checkpoint,
    save_checkpoint_atomic,
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


def test_warm_start_config_freezes_stage3_and_uses_separate_output() -> None:
    config = PyConfig.load(
        'constellation/new_transformers/config_event_v2_warm_start.py',
    )

    assert config.stage == 'V2-0'
    assert config.max_hours == 4
    assert config.annotation_file == 'train_paper_stage3_tau_e_existing.json'
    assert config.stage3_checkpoint.endswith('iter_200000/model.pth')
    assert config.output_dir == (
        'work_dirs/event_joint_transformer_v2/v2_0_warm_start'
    )
    assert config.model.freeze_backbone is True
    assert config.model.event_width == 256
    assert config.max_steps == 10_000
    assert config.checkpoint_interval == 1_000


def test_config_fingerprint_is_stable_and_sensitive() -> None:
    config = {'stage': 'V2-0', 'model': {'width': 8}, 'seed': 3407}

    first = config_fingerprint(config)
    second = config_fingerprint(config)
    changed = config_fingerprint({**config, 'seed': 1})

    assert first == second
    assert first != changed
    assert len(first) == 64


def test_checkpoint_round_trip_restores_model_optimizer_and_metadata(
    tmp_path,
) -> None:
    torch.manual_seed(3)
    random.seed(3)
    np.random.seed(3)
    model = _model()
    optimizer = torch.optim.AdamW(model.parameter_groups(1e-3))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    expected_parameter = next(model.parameters()).detach().clone()
    fingerprint = config_fingerprint({'stage': 'V2-0'})
    checkpoint = build_training_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config_fingerprint_value=fingerprint,
        normalizer={'task_mean': torch.tensor([1.])},
        counters=TrainingCounters(
            steps=7,
            processed_physical_seconds=123,
            episodes=4,
            events=19,
        ),
    )
    path = tmp_path / 'checkpoint.pth'
    save_checkpoint_atomic(path, checkpoint)
    with torch.no_grad():
        next(model.parameters()).add_(10.)

    counters = load_training_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        expected_config_fingerprint=fingerprint,
    )

    torch.testing.assert_close(next(model.parameters()), expected_parameter)
    assert counters == TrainingCounters(7, 123, 4, 19)
    loaded = torch.load(path, weights_only=False)
    assert loaded['stage'] == 'V2-0'
    assert loaded['policy_version'] == 0
    assert loaded['transition_schema_fingerprint'] == (
        transition_schema_fingerprint()
    )
    assert loaded['unfreeze_state'] == {'backbone_is_frozen': True}
    assert set(loaded['rng_state']) == {'python', 'numpy', 'torch', 'cuda'}
    assert not path.with_suffix('.pth.tmp').exists()


def test_checkpoint_rejects_schema_or_config_mismatch(tmp_path) -> None:
    model = _model()
    optimizer = torch.optim.AdamW(model.parameter_groups(1e-3))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    checkpoint = build_training_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config_fingerprint_value='a' * 64,
        normalizer={},
        counters=TrainingCounters(),
    )
    checkpoint['transition_schema_fingerprint'] = 'bad'
    path = tmp_path / 'bad.pth'
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match='schema fingerprint'):
        load_training_checkpoint(
            path=path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_config_fingerprint='a' * 64,
        )

    checkpoint['transition_schema_fingerprint'] = transition_schema_fingerprint()
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='config fingerprint'):
        load_training_checkpoint(
            path=path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_config_fingerprint='b' * 64,
        )
