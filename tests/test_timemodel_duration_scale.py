import torch

from constellation.new_transformers.constants import TIME_SCALE
from constellation.new_transformers.time_model import TimeModel


def test_duration_scale_is_fifty_time_steps() -> None:
    assert TIME_SCALE == 50


def test_timemodel_predict_restores_duration_to_time_steps() -> None:
    model = TimeModel(
        input_dim=2,
        time_embedding_dim=2,
        hidden_dim=2,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model._mlp[-1].bias[0] = 1.

    internal_duration, _ = model._predict(
        torch.tensor([0]),
        torch.zeros(1, 1),
        torch.zeros(1, 1),
    )
    public_duration, _ = model.predict(
        torch.tensor([0]),
        torch.zeros(1, 1, 1),
        torch.ones(1, 1, dtype=torch.bool),
        torch.zeros(1, 1, 1),
        torch.ones(1, 1, dtype=torch.bool),
    )

    torch.testing.assert_close(internal_duration, torch.tensor([1.]))
    torch.testing.assert_close(
        public_duration,
        torch.tensor([[[float(TIME_SCALE)]]]),
    )
