import torch

from constellation.new_transformers.temporal_adapter import (
    TemporalAdapter,
    TemporalAdapterOutput,
    TemporalHistoryTensors,
    TemporalOutcomePositiveWeights,
    masked_binary_cross_entropy,
    temporal_outcome_loss,
)
from constellation.new_transformers.dataset import TemporalBatch


def _history() -> TemporalHistoryTensors:
    return TemporalHistoryTensors(
        previous_task_indices=torch.tensor([[1, -1]]),
        previous_task_available=torch.tensor([[True, False]]),
        previous_was_idle=torch.tensor([[False, True]]),
        run_lengths=torch.tensor([[3., 2.]]),
        switch_count_30=torch.tensor([[1., 0.]]),
        switch_count_60=torch.tensor([[2., 0.]]),
    )


def _forward(adapter: TemporalAdapter):
    return adapter(
        satellite_features=torch.randn(1, 2, 8),
        task_features=torch.randn(1, 3, 8),
        null_logits=torch.randn(1, 2),
        task_logits=torch.randn(1, 2, 3),
        satellite_mask=torch.tensor([[True, False]]),
        task_mask=torch.tensor([[True, False, True]]),
        history=_history(),
    )


def test_temporal_adapter_starts_as_exact_noop() -> None:
    adapter = TemporalAdapter(
        satellite_width=8,
        task_width=8,
        hidden_width=16,
        horizons=(5, 15),
    )

    result = _forward(adapter)

    torch.testing.assert_close(
        result.null_delta,
        torch.zeros_like(result.null_delta),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        result.task_delta,
        torch.zeros_like(result.task_delta),
        rtol=0,
        atol=0,
    )


def test_temporal_adapter_residual_only_changes_valid_edges() -> None:
    adapter = TemporalAdapter(
        satellite_width=8,
        task_width=8,
        hidden_width=16,
        horizons=(5,),
    )
    with torch.no_grad():
        adapter.task_residual.bias.fill_(0.5)
        adapter.null_residual.bias.fill_(0.25)

    result = _forward(adapter)

    torch.testing.assert_close(
        result.task_delta,
        torch.tensor([[[0.5, 0.0, 0.5], [0.0, 0.0, 0.0]]]),
    )
    torch.testing.assert_close(
        result.null_delta,
        torch.tensor([[0.25, 0.0]]),
    )


def test_temporal_adapter_outputs_all_outcome_shapes() -> None:
    adapter = TemporalAdapter(
        satellite_width=8,
        task_width=8,
        hidden_width=16,
        horizons=(5, 15, 30),
    )

    result = _forward(adapter)

    assert result.visible_next_logits.shape == (1, 2, 3)
    assert result.progress_next_logits.shape == (1, 2, 3)
    assert result.completed_next_logits.shape == (1, 2, 3)
    assert result.visible_logits.shape == (1, 2, 3, 3)
    assert result.progress_logits.shape == (1, 2, 3, 3)
    assert result.completed_logits.shape == (1, 2, 3, 3)
    assert result.time_to_first_visible.shape == (1, 2, 3, 3)
    assert result.time_to_first_progress.shape == (1, 2, 3, 3)
    assert result.time_to_completion.shape == (1, 2, 3, 3)


def test_temporal_adapter_rejects_available_out_of_range_previous_task() -> None:
    adapter = TemporalAdapter(
        satellite_width=8,
        task_width=8,
        hidden_width=16,
        horizons=(5,),
    )
    history = _history()
    history = history._replace(
        previous_task_indices=torch.tensor([[3, -1]]),
    )

    try:
        adapter(
            satellite_features=torch.randn(1, 2, 8),
            task_features=torch.randn(1, 3, 8),
            null_logits=torch.randn(1, 2),
            task_logits=torch.randn(1, 2, 3),
            satellite_mask=torch.ones(1, 2, dtype=torch.bool),
            task_mask=torch.ones(1, 3, dtype=torch.bool),
            history=history,
        )
    except ValueError as error:
        assert 'previous_task_indices' in str(error)
    else:
        raise AssertionError('out-of-range previous task must fail')


def test_history_shape_validation_can_skip_value_reductions(monkeypatch) -> None:
    history = _history()

    def fail_any(self, *args, **kwargs):
        raise AssertionError('hot-path validation must not reduce tensor values')

    monkeypatch.setattr(torch.Tensor, 'any', fail_any)

    history.validate(
        batch_size=1,
        num_satellites=2,
        num_tasks=3,
        check_values=False,
    )


def test_masked_bce_ignores_censored_entries() -> None:
    logits = torch.tensor([10., -10.], requires_grad=True)

    loss = masked_binary_cross_entropy(
        logits,
        torch.tensor([1., 1.]),
        torch.tensor([True, False]),
    )
    expected = masked_binary_cross_entropy(
        logits[:1],
        torch.tensor([1.]),
        torch.tensor([True]),
    )

    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[1].item() == 0.


def test_masked_bce_all_censored_returns_differentiable_zero() -> None:
    logits = torch.randn(3, requires_grad=True)

    loss = masked_binary_cross_entropy(
        logits,
        torch.ones(3),
        torch.zeros(3, dtype=torch.bool),
    )

    assert loss.item() == 0.
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def test_masked_bce_applies_training_count_positive_weight() -> None:
    logits = torch.zeros(2)
    targets = torch.tensor([1., 0.])
    observed = torch.ones(2, dtype=torch.bool)

    unweighted = masked_binary_cross_entropy(logits, targets, observed)
    weighted = masked_binary_cross_entropy(
        logits,
        targets,
        observed,
        positive_weight=torch.tensor(3.),
    )

    torch.testing.assert_close(weighted, 2 * unweighted)


def test_zero_residual_layer_receives_gradient() -> None:
    adapter = TemporalAdapter(
        satellite_width=8,
        task_width=8,
        hidden_width=16,
        horizons=(5,),
    )

    result = _forward(adapter)
    result.task_delta.sum().backward()

    assert adapter.task_residual.weight.grad is not None
    assert adapter.task_residual.weight.grad.abs().sum() > 0


def _outcome_output(visible_horizon_logits: torch.Tensor) -> TemporalAdapterOutput:
    shape = (1, 2, 1)
    horizon_shape = (1, 2, 1, 1)
    zero_next = torch.zeros(shape, requires_grad=True)
    zero_horizon = torch.zeros(horizon_shape, requires_grad=True)
    return TemporalAdapterOutput(
        null_delta=torch.zeros(1, 2),
        task_delta=torch.zeros(shape),
        visible_next_logits=zero_next,
        progress_next_logits=zero_next.clone(),
        completed_next_logits=zero_next.clone(),
        visible_logits=visible_horizon_logits,
        progress_logits=zero_horizon,
        completed_logits=zero_horizon.clone(),
        time_to_first_visible=zero_horizon.clone(),
        time_to_first_progress=zero_horizon.clone(),
        time_to_completion=zero_horizon.clone(),
    )


def _outcome_targets() -> TemporalBatch:
    bool_2 = torch.tensor([[True, True]])
    bool_h = torch.tensor([[[True], [True]]])
    return TemporalBatch(
        previous_task_indices=torch.tensor([[-1, -1]]),
        previous_task_available=torch.tensor([[False, False]]),
        previous_was_idle=torch.tensor([[True, True]]),
        run_lengths=torch.zeros(1, 2),
        switch_count_30=torch.zeros(1, 2),
        switch_count_60=torch.zeros(1, 2),
        outcome_valid=bool_2,
        visible_next=bool_2,
        progress_next=torch.zeros_like(bool_2),
        completed_next=torch.zeros_like(bool_2),
        horizons=torch.tensor([5]),
        visible=bool_h,
        visible_observed=torch.tensor([[[True], [False]]]),
        progress=torch.zeros_like(bool_h),
        progress_observed=torch.ones_like(bool_h),
        completed=torch.zeros_like(bool_h),
        completion_observed=torch.ones_like(bool_h),
        time_to_first_visible=torch.tensor([[[2], [0]]]),
        time_to_first_progress=torch.zeros(1, 2, 1, dtype=torch.long),
        time_to_completion=torch.zeros(1, 2, 1, dtype=torch.long),
    )


def test_temporal_outcome_loss_ignores_censored_horizon_logit() -> None:
    first = _outcome_output(
        torch.tensor([[[[2.]], [[-100.]]]], requires_grad=True),
    )
    second = _outcome_output(
        torch.tensor([[[[2.]], [[100.]]]], requires_grad=True),
    )
    targets = _outcome_targets()
    actions = torch.zeros(1, 2, dtype=torch.long)

    first_losses = temporal_outcome_loss(first, targets, actions)
    second_losses = temporal_outcome_loss(second, targets, actions)

    torch.testing.assert_close(first_losses.visible, second_losses.visible)
    assert torch.isfinite(first_losses.event_time)


def test_temporal_outcome_loss_uses_next_and_horizon_positive_weights() -> None:
    output = _outcome_output(
        torch.zeros(1, 2, 1, 1, requires_grad=True),
    )
    targets = _outcome_targets()
    actions = torch.zeros(1, 2, dtype=torch.long)

    unweighted = temporal_outcome_loss(output, targets, actions)
    weighted = temporal_outcome_loss(
        output,
        targets,
        actions,
        positive_weights=TemporalOutcomePositiveWeights(
            visible=torch.tensor([2., 3.]),
            progress=torch.ones(2),
            completion=torch.ones(2),
        ),
    )

    assert weighted.visible > unweighted.visible
    torch.testing.assert_close(weighted.progress, unweighted.progress)
    torch.testing.assert_close(weighted.completion, unweighted.completion)


def test_temporal_outcome_loss_returns_differentiable_zero_for_idle_batch() -> None:
    output = _outcome_output(
        torch.zeros(1, 2, 1, 1, requires_grad=True),
    )
    targets = _outcome_targets()._replace(
        outcome_valid=torch.zeros(1, 2, dtype=torch.bool),
    )

    losses = temporal_outcome_loss(
        output,
        targets,
        torch.full((1, 2), -1, dtype=torch.long),
    )
    total = losses.visible + losses.progress + losses.completion + losses.event_time

    assert total.item() == 0.
    total.backward()
    assert output.visible_logits.grad is not None
