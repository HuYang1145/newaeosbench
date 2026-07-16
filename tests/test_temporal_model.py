import torch

from constellation.new_transformers.dataset import JointBatch, TemporalBatch
from constellation.new_transformers.model import JointModel, Model
from constellation.new_transformers.temporal_adapter import TemporalHistoryTensors


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
    )


def _predict_inputs() -> tuple[object, ...]:
    return (
        [1],
        torch.zeros(1, 2, dtype=torch.long),
        torch.ones(1, 2, dtype=torch.long),
        torch.randn(1, 2, 56),
        torch.ones(1, 2, dtype=torch.bool),
        torch.zeros(1, 2, dtype=torch.long),
        torch.randn(1, 2, 6),
        torch.ones(1, 2, dtype=torch.bool),
    )


def _history() -> TemporalHistoryTensors:
    return TemporalHistoryTensors(
        previous_task_indices=torch.tensor([[0, -1]]),
        previous_task_available=torch.tensor([[True, False]]),
        previous_was_idle=torch.tensor([[False, True]]),
        run_lengths=torch.tensor([[2., 1.]]),
        switch_count_30=torch.tensor([[0., 1.]]),
        switch_count_60=torch.tensor([[1., 2.]]),
    )


def _temporal_targets() -> TemporalBatch:
    return TemporalBatch(
        previous_task_indices=torch.tensor([[0, -1]]),
        previous_task_available=torch.tensor([[True, False]]),
        previous_was_idle=torch.tensor([[False, True]]),
        run_lengths=torch.tensor([[2., 1.]]),
        switch_count_30=torch.tensor([[0., 1.]]),
        switch_count_60=torch.tensor([[1., 2.]]),
        outcome_valid=torch.tensor([[True, True]]),
        visible_next=torch.tensor([[True, False]]),
        progress_next=torch.tensor([[False, True]]),
        completed_next=torch.tensor([[False, False]]),
        horizons=torch.tensor([1]),
        visible=torch.tensor([[[True], [False]]]),
        visible_observed=torch.ones(1, 2, 1, dtype=torch.bool),
        progress=torch.tensor([[[False], [True]]]),
        progress_observed=torch.ones(1, 2, 1, dtype=torch.bool),
        completed=torch.zeros(1, 2, 1, dtype=torch.bool),
        completion_observed=torch.ones(1, 2, 1, dtype=torch.bool),
        time_to_first_visible=torch.tensor([[[1], [0]]]),
        time_to_first_progress=torch.tensor([[[0], [1]]]),
        time_to_completion=torch.zeros(1, 2, 1, dtype=torch.long),
    )


def _joint_batch() -> JointBatch:
    inputs = _predict_inputs()
    return JointBatch(
        id_=0,
        annotation_id=0,
        time_steps=inputs[0],
        constellation_sensor_type=inputs[1],
        constellation_sensor_enabled=inputs[2],
        constellation_data=inputs[3],
        constellation_mask=inputs[4],
        tasks_sensor_type=inputs[5],
        tasks_data=inputs[6],
        tasks_mask=inputs[7],
        actions_task_id=torch.tensor([[0, 1]]),
        constraint_time_steps=torch.tensor([1, 1]),
        constraint_constellation_data=torch.zeros(2, 56),
        constraint_tasks_data=torch.zeros(2, 6),
        constraint_durations=torch.tensor([1., -1.]),
        temporal=_temporal_targets(),
    )


def test_temporal_model_starts_with_exact_baseline_logits() -> None:
    baseline = Model(**_tiny_model_kwargs()).eval()
    temporal = Model(
        **_tiny_model_kwargs(),
        use_temporal_adapter=True,
        temporal_adapter_hidden_width=16,
        temporal_horizons=(1,),
    ).eval()
    incompatible = temporal.load_state_dict(baseline.state_dict(), strict=False)
    inputs = _predict_inputs()

    with torch.no_grad():
        baseline_logits = baseline.predict(*inputs)
        temporal_logits = temporal.predict(*inputs, temporal_history=_history())

    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all('_temporal_adapter.' in key for key in incompatible.missing_keys)
    torch.testing.assert_close(temporal_logits, baseline_logits, rtol=0, atol=0)


def test_temporal_model_requires_history_only_when_enabled() -> None:
    baseline = Model(**_tiny_model_kwargs()).eval()
    baseline.predict(*_predict_inputs())
    temporal = Model(
        **_tiny_model_kwargs(),
        use_temporal_adapter=True,
        temporal_horizons=(1,),
    ).eval()

    try:
        temporal.predict(*_predict_inputs())
    except ValueError as error:
        assert 'temporal history' in str(error)
    else:
        raise AssertionError('enabled temporal adapter must require history')


def test_temporal_residual_scale_changes_null_and_task_logits() -> None:
    model = Model(
        **_tiny_model_kwargs(),
        use_temporal_adapter=True,
        temporal_adapter_hidden_width=16,
        temporal_horizons=(1,),
        temporal_residual_scale=2.,
    ).eval()
    with torch.no_grad():
        adapter = model._transformer._temporal_adapter
        assert adapter is not None
        adapter.null_residual.bias.fill_(0.25)
        adapter.task_residual.bias.fill_(0.5)
    inputs = _predict_inputs()

    with torch.no_grad():
        changed = model.predict(*inputs, temporal_history=_history())
        adapter.null_residual.bias.zero_()
        adapter.task_residual.bias.zero_()
        baseline = model.predict(*inputs, temporal_history=_history())

    torch.testing.assert_close(
        changed[..., 0] - baseline[..., 0],
        torch.full((1, 2), 2 * torch.tanh(torch.tensor(0.25))),
    )
    torch.testing.assert_close(
        changed[..., 1:] - baseline[..., 1:],
        torch.full((1, 2, 2), 2 * torch.tanh(torch.tensor(0.5))),
    )


def test_freeze_temporal_backbone_only_leaves_adapter_trainable() -> None:
    model = JointModel(
        **_tiny_model_kwargs(),
        use_temporal_adapter=True,
        temporal_horizons=(1,),
        freeze_temporal_backbone=True,
    )

    trainable = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    assert trainable
    assert all('_temporal_adapter.' in name for name in trainable)


def test_temporal_joint_model_backward_and_step_only_update_adapter() -> None:
    model = JointModel(
        **_tiny_model_kwargs(),
        use_temporal_adapter=True,
        temporal_adapter_hidden_width=16,
        temporal_horizons=(1,),
        freeze_temporal_backbone=True,
        feasibility_loss_weight=0.,
        time_loss_weight=0.,
        assignment_loss_weight=1.,
        temporal_visible_loss_weight=1.,
        temporal_progress_loss_weight=1.,
        temporal_completion_loss_weight=1.,
        temporal_event_time_loss_weight=1.,
    )
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if '_temporal_adapter.' not in name
    }
    adapter_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if '_temporal_adapter.' in name
    }
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.1,
    )

    memo = model(type('Runner', (), {'iter_': 0})(), _joint_batch(), {})
    assert torch.isfinite(memo['loss'])
    memo['loss'].backward()
    optimizer.step()

    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if '_temporal_adapter.' not in name
    )
    for name, before in frozen_before.items():
        torch.testing.assert_close(model.state_dict()[name], before)
    assert any(
        not torch.equal(model.state_dict()[name], before)
        for name, before in adapter_before.items()
    )
    for key in (
        'temporal_visible_loss',
        'temporal_progress_loss',
        'temporal_completion_loss',
        'temporal_event_time_loss',
    ):
        assert key in memo
