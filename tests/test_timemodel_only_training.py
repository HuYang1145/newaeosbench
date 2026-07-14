import torch
from todd.configs import PyConfig

from constellation.new_transformers.dataset import JointBatch
from constellation.new_transformers.model import JointModel
from constellation.new_transformers.time_model import TimeModel


def _tiny_model_kwargs() -> dict[str, object]:
    return dict(
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
        feasibility_loss_weight=0.,
        assignment_loss_weight=0.,
        train_duration_head_only=True,
    )


def _tiny_joint_batch() -> JointBatch:
    return JointBatch(
        id_=0,
        annotation_id=0,
        time_steps=[0],
        constellation_sensor_type=torch.zeros(1, 1, dtype=torch.long),
        constellation_sensor_enabled=torch.ones(1, 1, dtype=torch.long),
        constellation_data=torch.zeros(1, 1, 56),
        constellation_mask=torch.ones(1, 1, dtype=torch.bool),
        tasks_sensor_type=torch.zeros(1, 1, dtype=torch.long),
        tasks_data=torch.zeros(1, 1, 6),
        tasks_mask=torch.ones(1, 1, dtype=torch.bool),
        actions_task_id=torch.zeros(1, 1, dtype=torch.long),
        constraint_time_steps=torch.tensor([0, 0]),
        constraint_constellation_data=torch.zeros(2, 56),
        constraint_tasks_data=torch.zeros(2, 6),
        constraint_durations=torch.tensor([1., -1.]),
    )


def test_duration_head_starts_as_zero_residual() -> None:
    model = TimeModel(
        input_dim=2,
        time_embedding_dim=2,
        hidden_dim=4,
    )
    time_steps = torch.tensor([0, 1])
    constellation_data = torch.randn(2, 1)
    tasks_data = torch.randn(2, 1)

    assert torch.count_nonzero(model._duration_head.weight) == 0
    assert torch.count_nonzero(model._duration_head.bias) == 0

    with torch.no_grad():
        data = torch.cat([
            constellation_data,
            tasks_data,
            model._time_embedding[time_steps],
        ], -1)
        legacy_duration, legacy_feasibility = model._mlp(data).unbind(-1)
        duration, feasibility = model._predict(
            time_steps,
            constellation_data,
            tasks_data,
        )

    torch.testing.assert_close(duration, legacy_duration)
    torch.testing.assert_close(feasibility, legacy_feasibility)


def test_duration_head_only_mode_freezes_every_other_parameter() -> None:
    model = JointModel(**_tiny_model_kwargs())

    trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    assert trainable == [
        '_transformer._time_model._duration_head.weight',
        '_transformer._time_model._duration_head.bias',
    ]


def test_old_checkpoint_cancels_legacy_duration_in_head_only_mode() -> None:
    source = JointModel(**_tiny_model_kwargs())
    old_state = {
        name: value
        for name, value in source.state_dict().items()
        if '_duration_head.' not in name
    }
    model = JointModel(**_tiny_model_kwargs())

    incompatible = model.load_state_dict(old_state, strict=False)
    batch = _tiny_joint_batch()
    with torch.no_grad():
        duration, _ = model._transformer._time_model._predict(
            batch.constraint_time_steps,
            batch.constraint_constellation_data,
            batch.constraint_tasks_data,
        )

    assert incompatible.missing_keys == [
        '_transformer._time_model._duration_head.weight',
        '_transformer._time_model._duration_head.bias',
    ]
    torch.testing.assert_close(duration, torch.zeros_like(duration), atol=1e-6, rtol=0)


def test_checkpoint_with_duration_head_preserves_trained_head() -> None:
    source = JointModel(**_tiny_model_kwargs())
    with torch.no_grad():
        source._transformer._time_model._duration_head.weight.fill_(0.25)
        source._transformer._time_model._duration_head.bias.fill_(0.5)
    model = JointModel(**_tiny_model_kwargs())

    incompatible = model.load_state_dict(source.state_dict(), strict=False)

    assert incompatible.missing_keys == []
    torch.testing.assert_close(
        model._transformer._time_model._duration_head.weight,
        source._transformer._time_model._duration_head.weight,
    )
    torch.testing.assert_close(
        model._transformer._time_model._duration_head.bias,
        source._transformer._time_model._duration_head.bias,
    )


def test_duration_head_only_forward_skips_action_prediction(monkeypatch) -> None:
    model = JointModel(**_tiny_model_kwargs())

    def fail_predict(*args, **kwargs):
        raise AssertionError('the action branch must stay frozen and unused')

    monkeypatch.setattr(model, 'predict', fail_predict)
    memo = model(type('Runner', (), {'iter_': 0})(), _tiny_joint_batch(), {})

    assert 'logits' not in memo
    assert memo['assignment_loss'].item() == 0.
    torch.testing.assert_close(memo['loss'], memo['lt_loss'])
    assert memo['duration_mae_s'].item() >= 0.

    memo['loss'].backward()
    gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
    }
    assert gradients[
        '_transformer._time_model._duration_head.bias'
    ] is not None
    assert all(
        gradient is None
        for name, gradient in gradients.items()
        if '_duration_head.' not in name
    )


def test_duration_head_step_preserves_feasibility_output() -> None:
    model = JointModel(**_tiny_model_kwargs())
    time_model = model._transformer._time_model
    batch = _tiny_joint_batch()

    with torch.no_grad():
        duration_before, feasibility_before = time_model._predict(
            batch.constraint_time_steps,
            batch.constraint_constellation_data,
            batch.constraint_tasks_data,
        )
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in time_model.named_parameters()
        if not name.startswith('_duration_head.')
    }

    optimizer = torch.optim.SGD(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=0.1,
    )
    memo = model(type('Runner', (), {'iter_': 0})(), batch, {})
    memo['loss'].backward()
    optimizer.step()

    with torch.no_grad():
        duration_after, feasibility_after = time_model._predict(
            batch.constraint_time_steps,
            batch.constraint_constellation_data,
            batch.constraint_tasks_data,
        )
    frozen_after = {
        name: parameter.detach()
        for name, parameter in time_model.named_parameters()
        if not name.startswith('_duration_head.')
    }

    assert not torch.equal(duration_before, duration_after)
    torch.testing.assert_close(feasibility_after, feasibility_before)
    for name in frozen_before:
        torch.testing.assert_close(frozen_after[name], frozen_before[name])


def test_duration_head_pilot_config_is_small_and_action_free() -> None:
    config = PyConfig.load(
        'constellation/new_transformers/config_timemodel_scale_pilot.py',
    )

    model = config.trainer.model
    assert config.trainer.iters == 2_000
    assert model.train_duration_head_only is True
    assert model.feasibility_loss_weight == 0.
    assert model.assignment_loss_weight == 0.
    assert model.collision_loss_weight == 0.
    assert model.coverage_loss_weight == 0.
    assert config.trainer.dataset.annotation_file == (
        'train_paper_stage3_tau_e_existing.json'
    )
    checkpoint = next(
        callback
        for callback in config.trainer.callbacks
        if callback.type == 'CheckpointCallback'
    )
    assert checkpoint.interval == 500
