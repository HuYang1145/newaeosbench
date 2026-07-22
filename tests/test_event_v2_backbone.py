import inspect

import pytest
import torch

from constellation.new_transformers.event_v2.backbone import (
    Stage3FeatureBackbone,
)
from constellation.new_transformers.model import Model


def _model_kwargs(*, depth: int = 1) -> dict[str, object]:
    return dict(
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
        torch.tensor([[True, True, False]]),
    )


def test_backbone_reproduces_stage3_tokens_and_logits() -> None:
    source = Model(**_model_kwargs()).eval()
    backbone = Stage3FeatureBackbone(
        **_model_kwargs(),
        edge_width=6,
    ).eval()
    backbone.load_stage3_state_dict(source.state_dict())
    captured: dict[str, torch.Tensor] = {}

    def capture_tasks(module, inputs, output) -> None:
        del module, inputs
        captured['tasks'] = output

    def capture_satellites(module, inputs, output) -> None:
        del module, inputs
        captured['satellites'] = output[2]

    encoder_handle = source._transformer._encoder.register_forward_hook(
        capture_tasks,
    )
    decoder_handle = source._transformer._decoder.register_forward_hook(
        capture_satellites,
    )
    inputs = _inputs()
    with torch.no_grad():
        expected_logits = source.predict(*inputs)
        actual = backbone(*inputs)
    encoder_handle.remove()
    decoder_handle.remove()

    torch.testing.assert_close(actual.task_tokens, captured['tasks'])
    torch.testing.assert_close(
        actual.satellite_tokens,
        captured['satellites'],
    )
    torch.testing.assert_close(
        actual.teacher_null_logits,
        expected_logits[..., 0],
    )
    torch.testing.assert_close(
        actual.teacher_task_logits,
        expected_logits[..., 1:],
    )
    assert actual.edge_features.shape == (1, 2, 3, 6)
    assert actual.feasibility_logits is None


def test_freeze_only_locks_checkpoint_backbone_parameters() -> None:
    backbone = Stage3FeatureBackbone(**_model_kwargs(), edge_width=6)

    backbone.freeze()

    assert all(
        not parameter.requires_grad
        for parameter in backbone.transformer.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in backbone.satellite_edge_projection.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in backbone.task_edge_projection.parameters()
    )


def test_unfreeze_last_layers_does_not_open_input_embeddings() -> None:
    backbone = Stage3FeatureBackbone(
        **_model_kwargs(depth=2),
        edge_width=6,
    )
    backbone.freeze()

    backbone.unfreeze_last_layers(encoder_layers=1, decoder_layers=1)

    encoder_blocks = list(backbone.transformer._encoder._blocks.children())
    decoder_blocks = list(backbone.transformer._decoder._blocks.children())
    assert all(not parameter.requires_grad for parameter in encoder_blocks[0].parameters())
    assert all(parameter.requires_grad for parameter in encoder_blocks[1].parameters())
    assert all(not parameter.requires_grad for parameter in decoder_blocks[0].parameters())
    assert all(parameter.requires_grad for parameter in decoder_blocks[1].parameters())
    assert all(
        not parameter.requires_grad
        for parameter in backbone.transformer._encoder._in_projector.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in backbone.transformer._decoder._in_projector.parameters()
    )


def test_backbone_forward_has_no_privileged_simulator_input() -> None:
    parameters = inspect.signature(Stage3FeatureBackbone.forward).parameters

    assert 'is_visible' not in parameters
    assert 'future_state' not in parameters
    assert 'basilisk' not in parameters


def test_stage3_loader_rejects_missing_checkpoint_parameter() -> None:
    source = Model(**_model_kwargs())
    state_dict = source.state_dict()
    state_dict.pop(next(iter(state_dict)))
    backbone = Stage3FeatureBackbone(**_model_kwargs(), edge_width=6)

    with pytest.raises(ValueError, match='missing Stage3 backbone keys'):
        backbone.load_stage3_state_dict(state_dict)


def test_stage3_loader_rejects_unexpected_checkpoint_parameter() -> None:
    source = Model(**_model_kwargs())
    state_dict = dict(source.state_dict())
    state_dict['_transformer.not_a_real_parameter'] = torch.tensor(1.)
    backbone = Stage3FeatureBackbone(**_model_kwargs(), edge_width=6)

    with pytest.raises(ValueError, match='unexpected Stage3 backbone keys'):
        backbone.load_stage3_state_dict(state_dict)


@pytest.mark.parametrize(
    ('encoder_layers', 'decoder_layers'),
    [(-1, 0), (0, -1), (3, 0), (0, 3)],
)
def test_unfreeze_rejects_invalid_layer_count(
    encoder_layers: int,
    decoder_layers: int,
) -> None:
    backbone = Stage3FeatureBackbone(
        **_model_kwargs(depth=2),
        edge_width=6,
    )

    with pytest.raises(ValueError, match='layer count'):
        backbone.unfreeze_last_layers(encoder_layers, decoder_layers)
