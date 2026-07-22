import torch

from constellation.new_transformers.event_v2.backbone import (
    Stage3BackboneOutput,
    Stage3FeatureBackbone,
)
from constellation.new_transformers.event_v2.critic import (
    CentralizedValueCritic,
    EventStateEncoder,
    EventStateEncoding,
)
from constellation.new_transformers.event_v2.state import EventStateTensors


def _state() -> EventStateTensors:
    return EventStateTensors(
        previous_task_indices=torch.tensor([[0, 1]]),
        current_task_indices=torch.tensor([[1, -1]]),
        minimum_commitment_remaining=torch.tensor([[0., 5.]]),
        run_lengths=torch.tensor([[8., 3.]]),
        seconds_since_replan=torch.tensor([[2., 4.]]),
        switch_count_30=torch.tensor([[0., 2.]]),
        switch_count_60=torch.tensor([[1., 3.]]),
        termination_reason=torch.tensor([[0, 1]]),
        event_type=torch.tensor([[0, 1]]),
        delta_t=torch.tensor([[5., 1.]]),
        replan_mask=torch.tensor([[True, True]]),
        forced_interrupt_mask=torch.tensor([[False, True]]),
        can_terminate_mask=torch.tensor([[True, False]]),
        compatible_deadline_slack=torch.tensor([[20., 5.]]),
        task_remaining_required_seconds=torch.tensor([[1., 4., 30.]]),
        task_owner_count=torch.tensor([[0, 1, 2]]),
        task_locked_owner_count=torch.tensor([[0, 1, 1]]),
    )


def _backbone_output() -> Stage3BackboneOutput:
    return Stage3BackboneOutput(
        task_tokens=torch.randn(1, 3, 6),
        satellite_tokens=torch.randn(1, 2, 5),
        edge_features=torch.randn(1, 2, 3, 4),
        teacher_null_logits=torch.zeros(1, 2),
        teacher_task_logits=torch.zeros(1, 2, 3),
        feasibility_logits=None,
    )


def _encoder() -> EventStateEncoder:
    return EventStateEncoder(
        satellite_width=5,
        task_width=6,
        edge_width=4,
        event_width=8,
        num_termination_reasons=4,
        num_event_types=4,
    ).eval()


def _remap_task_indices(
    indices: torch.Tensor,
    permutation: torch.Tensor,
) -> torch.Tensor:
    inverse = torch.argsort(permutation)
    return torch.where(indices >= 0, inverse[indices.clamp_min(0)], -1)


def test_event_state_encoder_is_task_permutation_equivariant() -> None:
    torch.manual_seed(4)
    encoder = _encoder()
    backbone = _backbone_output()
    state = _state()
    satellite_mask = torch.ones(1, 2, dtype=torch.bool)
    task_mask = torch.ones(1, 3, dtype=torch.bool)
    permutation = torch.tensor([2, 0, 1])

    expected = encoder(backbone, state, satellite_mask, task_mask)
    permuted_backbone = Stage3BackboneOutput(
        task_tokens=backbone.task_tokens[:, permutation],
        satellite_tokens=backbone.satellite_tokens,
        edge_features=backbone.edge_features[:, :, permutation],
        teacher_null_logits=backbone.teacher_null_logits,
        teacher_task_logits=backbone.teacher_task_logits[:, :, permutation],
        feasibility_logits=None,
    )
    permuted_state = state._replace(
        previous_task_indices=_remap_task_indices(
            state.previous_task_indices,
            permutation,
        ),
        current_task_indices=_remap_task_indices(
            state.current_task_indices,
            permutation,
        ),
        task_remaining_required_seconds=(
            state.task_remaining_required_seconds[:, permutation]
        ),
        task_owner_count=state.task_owner_count[:, permutation],
        task_locked_owner_count=(
            state.task_locked_owner_count[:, permutation]
        ),
    )

    actual = encoder(
        permuted_backbone,
        permuted_state,
        satellite_mask,
        task_mask[:, permutation],
    )

    torch.testing.assert_close(actual.satellite_tokens, expected.satellite_tokens)
    torch.testing.assert_close(actual.task_tokens, expected.task_tokens[:, permutation])
    torch.testing.assert_close(actual.edge_tokens, expected.edge_tokens[:, :, permutation])


def test_centralized_critic_ignores_masked_tokens() -> None:
    torch.manual_seed(5)
    critic = CentralizedValueCritic(event_width=4).eval()
    encoding = EventStateEncoding(
        satellite_tokens=torch.randn(1, 2, 4),
        task_tokens=torch.randn(1, 3, 4),
        edge_tokens=torch.randn(1, 2, 3, 4),
    )
    satellite_mask = torch.tensor([[True, False]])
    task_mask = torch.tensor([[True, True, False]])
    changed = EventStateEncoding(
        satellite_tokens=encoding.satellite_tokens.clone(),
        task_tokens=encoding.task_tokens.clone(),
        edge_tokens=encoding.edge_tokens.clone(),
    )
    changed.satellite_tokens[0, 1] = 1e6
    changed.task_tokens[0, 2] = -1e6
    changed.edge_tokens[0, 1] = 1e6
    changed.edge_tokens[0, :, 2] = -1e6

    expected = critic(encoding, satellite_mask, task_mask)
    actual = critic(changed, satellite_mask, task_mask)

    torch.testing.assert_close(actual, expected)


def test_centralized_critic_outputs_one_value_per_scene() -> None:
    critic = CentralizedValueCritic(event_width=4)
    encoding = EventStateEncoding(
        satellite_tokens=torch.randn(3, 2, 4),
        task_tokens=torch.randn(3, 5, 4),
        edge_tokens=torch.randn(3, 2, 5, 4),
    )

    value = critic(
        encoding,
        torch.ones(3, 2, dtype=torch.bool),
        torch.ones(3, 5, dtype=torch.bool),
    )

    assert value.shape == (3,)
    assert torch.isfinite(value).all()


def test_centralized_critic_rejects_scene_without_valid_tokens() -> None:
    critic = CentralizedValueCritic(event_width=4)
    encoding = EventStateEncoding(
        satellite_tokens=torch.randn(1, 2, 4),
        task_tokens=torch.randn(1, 3, 4),
        edge_tokens=torch.randn(1, 2, 3, 4),
    )

    try:
        critic(
            encoding,
            torch.zeros(1, 2, dtype=torch.bool),
            torch.ones(1, 3, dtype=torch.bool),
        )
    except ValueError as error:
        assert 'valid satellite' in str(error)
    else:
        raise AssertionError('a scene without valid satellites must fail')


def test_frozen_stage3_receives_no_gradient_from_critic() -> None:
    backbone = Stage3FeatureBackbone(
        edge_width=4,
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
    backbone.freeze()
    encoder = EventStateEncoder(
        satellite_width=8,
        task_width=8,
        edge_width=4,
        event_width=8,
        num_termination_reasons=4,
        num_event_types=4,
    )
    critic = CentralizedValueCritic(event_width=8)
    inputs = (
        [1],
        torch.zeros(1, 2, dtype=torch.long),
        torch.ones(1, 2, dtype=torch.long),
        torch.randn(1, 2, 56),
        torch.ones(1, 2, dtype=torch.bool),
        torch.zeros(1, 3, dtype=torch.long),
        torch.randn(1, 3, 6),
        torch.ones(1, 3, dtype=torch.bool),
    )

    encoded = encoder(
        backbone_output=backbone(*inputs),
        state=_state(),
        satellite_mask=inputs[4],
        task_mask=inputs[7],
    )
    critic(encoded, inputs[4], inputs[7]).sum().backward()

    assert all(
        parameter.grad is None
        for parameter in backbone.transformer.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in backbone.satellite_edge_projection.parameters()
    )
    assert any(parameter.grad is not None for parameter in encoder.parameters())
    assert any(parameter.grad is not None for parameter in critic.parameters())
