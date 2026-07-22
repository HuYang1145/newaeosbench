import inspect

import torch

from constellation.new_transformers.event_v2.model import (
    EventJointActorCritic,
)
from constellation.new_transformers.event_v2.state import EventStateTensors
from constellation.new_transformers.model import Model


def _model_kwargs(*, depth: int = 1) -> dict[str, object]:
    return dict(
        event_width=8,
        sensor_type_embedding_dim=4,
        tasks_data_embedding_dim=4,
        encoder_width=8,
        encoder_depth=depth,
        encoder_num_heads=2,
        sensor_enabled_embedding_dim=4,
        constellation_data_embedding_dim=4,
        decoder_width=8,
        decoder_depth=depth,
        decoder_num_heads=2,
        use_constraint_module=False,
        use_sdpa=False,
    )


def _inputs() -> tuple[object, ...]:
    return (
        [1],
        torch.zeros(1, 2, dtype=torch.long),
        torch.ones(1, 2, dtype=torch.long),
        torch.randn(1, 2, 56),
        torch.ones(1, 2, dtype=torch.bool),
        torch.zeros(1, 3, dtype=torch.long),
        torch.randn(1, 3, 6),
        torch.ones(1, 3, dtype=torch.bool),
    )


def _state() -> EventStateTensors:
    return EventStateTensors(
        previous_task_indices=torch.tensor([[-1, 0]]),
        current_task_indices=torch.tensor([[-1, -1]]),
        minimum_commitment_remaining=torch.zeros(1, 2),
        run_lengths=torch.tensor([[1., 4.]]),
        seconds_since_replan=torch.tensor([[5., 10.]]),
        switch_count_30=torch.tensor([[0., 1.]]),
        switch_count_60=torch.tensor([[0., 2.]]),
        termination_reason=torch.tensor([[0, 1]]),
        event_type=torch.tensor([[0, 1]]),
        delta_t=torch.tensor([[5., 5.]]),
        replan_mask=torch.ones(1, 2, dtype=torch.bool),
        forced_interrupt_mask=torch.zeros(1, 2, dtype=torch.bool),
        can_terminate_mask=torch.zeros(1, 2, dtype=torch.bool),
        compatible_deadline_slack=torch.tensor([[20., 5.]]),
        task_remaining_required_seconds=torch.tensor([[1., 10., 30.]]),
        task_owner_count=torch.zeros(1, 3, dtype=torch.long),
        task_locked_owner_count=torch.zeros(1, 3, dtype=torch.long),
    )


def test_event_joint_actor_critic_act_and_replay() -> None:
    model = EventJointActorCritic(**_model_kwargs()).eval()
    inputs = _inputs()

    output = model.act(*inputs, event_state=_state(), deterministic=True)
    replay, replay_value = model.evaluate_actions(
        *inputs,
        event_state=_state(),
        action=output.actor.action,
        trace=output.actor.trace,
    )

    assert output.value.shape == (1,)
    assert torch.isfinite(output.value).all()
    torch.testing.assert_close(replay.log_prob, output.actor.log_prob)
    torch.testing.assert_close(replay.entropy, output.actor.entropy)
    torch.testing.assert_close(replay_value, output.value)


def test_optimizer_step_keeps_frozen_stage3_parameters_unchanged() -> None:
    model = EventJointActorCritic(**_model_kwargs(), freeze_backbone=True)
    before = {
        name: value.detach().clone()
        for name, value in model.backbone.transformer.state_dict().items()
    }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )

    output = model.act(*_inputs(), event_state=_state(), deterministic=True)
    loss = -output.actor.log_prob.mean() + output.value.square().mean()
    loss.backward()
    optimizer.step()

    for name, expected in before.items():
        torch.testing.assert_close(
            model.backbone.transformer.state_dict()[name],
            expected,
            rtol=0,
            atol=0,
        )
    assert any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if not name.startswith('backbone.transformer.')
    )


def test_model_forward_interfaces_have_no_privileged_inputs() -> None:
    for method in (
        EventJointActorCritic.act,
        EventJointActorCritic.evaluate_actions,
    ):
        parameters = inspect.signature(method).parameters
        assert 'is_visible' not in parameters
        assert 'future_state' not in parameters
        assert 'basilisk' not in parameters


def test_state_dict_round_trip_preserves_deterministic_action_and_value(
    tmp_path,
) -> None:
    model = EventJointActorCritic(**_model_kwargs()).eval()
    inputs = _inputs()
    expected = model.act(*inputs, event_state=_state(), deterministic=True)
    path = tmp_path / 'v2.pth'
    torch.save(model.state_dict(), path)
    restored = EventJointActorCritic(**_model_kwargs()).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))

    actual = restored.act(*inputs, event_state=_state(), deterministic=True)

    torch.testing.assert_close(actual.value, expected.value, rtol=0, atol=0)
    torch.testing.assert_close(
        actual.actor.log_prob,
        expected.actor.log_prob,
        rtol=0,
        atol=0,
    )
    assert actual.actor.action.task_indices.tolist() == (
        expected.actor.action.task_indices.tolist()
    )
    assert actual.actor.action.commitment_indices.tolist() == (
        expected.actor.action.commitment_indices.tolist()
    )


def test_load_stage3_checkpoint_accepts_model_wrapper(tmp_path) -> None:
    stage3_kwargs = dict(_model_kwargs())
    stage3_kwargs.pop('event_width')
    source = Model(**stage3_kwargs)
    path = tmp_path / 'stage3.pth'
    torch.save({'model': source.state_dict()}, path)
    model = EventJointActorCritic(**_model_kwargs())

    model.load_stage3_checkpoint(path)

    source_value = source._transformer._time_embedding
    actual_value = model.backbone.transformer._time_embedding
    torch.testing.assert_close(actual_value, source_value)


def test_unfreeze_last_layers_only_opens_requested_tail() -> None:
    model = EventJointActorCritic(
        **_model_kwargs(depth=2),
        freeze_backbone=True,
    )

    model.unfreeze_last_layers(encoder_layers=1, decoder_layers=1)

    encoder_blocks = list(
        model.backbone.transformer._encoder._blocks.children()
    )
    decoder_blocks = list(
        model.backbone.transformer._decoder._blocks.children()
    )
    assert all(not parameter.requires_grad for parameter in encoder_blocks[0].parameters())
    assert all(parameter.requires_grad for parameter in encoder_blocks[1].parameters())
    assert all(not parameter.requires_grad for parameter in decoder_blocks[0].parameters())
    assert all(parameter.requires_grad for parameter in decoder_blocks[1].parameters())


def test_parameter_groups_apply_scaled_lr_only_to_unfrozen_stage3() -> None:
    model = EventJointActorCritic(
        **_model_kwargs(depth=2),
        freeze_backbone=True,
    )
    frozen_groups = model.parameter_groups(
        new_module_lr=1e-3,
        backbone_lr_scale=0.1,
    )
    model.unfreeze_last_layers(encoder_layers=1, decoder_layers=1)
    unfrozen_groups = model.parameter_groups(
        new_module_lr=1e-3,
        backbone_lr_scale=0.1,
    )

    assert [group['lr'] for group in frozen_groups] == [1e-3]
    assert {group['lr'] for group in unfrozen_groups} == {1e-3, 1e-4}
    parameter_ids = [
        id(parameter)
        for group in unfrozen_groups
        for parameter in group['params']
    ]
    assert len(parameter_ids) == len(set(parameter_ids))
