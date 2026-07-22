import torch

from constellation.new_transformers.dataset import Batch
from constellation.new_transformers.event_v2.dataset import (
    OfflineEventBatch,
    OfflineEventTargets,
)
from constellation.new_transformers.event_v2.model import EventJointActorCritic
from constellation.new_transformers.event_v2.offline import (
    event_v2_offline_loss,
)
from constellation.new_transformers.event_v2.state import EventStateTensors


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


def _batch(*, observed: bool = True) -> OfflineEventBatch:
    stage3 = Batch(
        id_=0,
        annotation_id=0,
        time_steps=[1],
        constellation_sensor_type=torch.zeros(1, 2, dtype=torch.long),
        constellation_sensor_enabled=torch.ones(1, 2, dtype=torch.long),
        constellation_data=torch.randn(1, 2, 56),
        constellation_mask=torch.ones(1, 2, dtype=torch.bool),
        tasks_sensor_type=torch.zeros(1, 2, dtype=torch.long),
        tasks_data=torch.randn(1, 2, 6),
        tasks_mask=torch.ones(1, 2, dtype=torch.bool),
        actions_task_id=torch.tensor([[0, 1]]),
        temporal=None,
    )
    state = EventStateTensors(
        previous_task_indices=torch.tensor([[-1, -1]]),
        current_task_indices=torch.tensor([[-1, -1]]),
        minimum_commitment_remaining=torch.zeros(1, 2),
        run_lengths=torch.ones(1, 2),
        seconds_since_replan=torch.ones(1, 2),
        switch_count_30=torch.zeros(1, 2),
        switch_count_60=torch.zeros(1, 2),
        termination_reason=torch.zeros(1, 2, dtype=torch.long),
        event_type=torch.zeros(1, 2, dtype=torch.long),
        delta_t=torch.ones(1, 2),
        replan_mask=torch.ones(1, 2, dtype=torch.bool),
        forced_interrupt_mask=torch.zeros(1, 2, dtype=torch.bool),
        can_terminate_mask=torch.ones(1, 2, dtype=torch.bool),
        compatible_deadline_slack=torch.full((1, 2), 10.),
        task_remaining_required_seconds=torch.tensor([[10., 20.]]),
        task_owner_count=torch.zeros(1, 2, dtype=torch.long),
        task_locked_owner_count=torch.zeros(1, 2, dtype=torch.long),
    )
    observed_mask = torch.full((1, 2), observed, dtype=torch.bool)
    targets = OfflineEventTargets(
        termination=torch.tensor([[False, True]]),
        termination_observed=observed_mask,
        task_indices=torch.tensor([[0, 1]]),
        task_observed=observed_mask,
        commitment_indices=torch.tensor([[1, 1]]),
        commitment_observed=observed_mask,
        value_returns=torch.tensor([0.5]),
    )
    return OfflineEventBatch(stage3, state, targets)


def test_offline_losses_are_finite_and_train_new_modules() -> None:
    model = _model()

    losses = event_v2_offline_loss(model, _batch())
    losses.total.backward()

    for value in losses:
        assert torch.isfinite(value)
    assert model.actor.task_value_head.weight.grad is not None
    assert model.actor.termination_head.weight.grad is not None
    assert model.actor.commitment_head.weight.grad is not None
    assert model.critic.value_head[-1].weight.grad is not None


def test_offline_loss_does_not_supervise_owner_marginal_from_expert_duplicates() -> None:
    model = _model()

    event_v2_offline_loss(model, _batch()).total.backward()

    assert model.actor.owner_marginal_head.weight.grad is None
    assert model.actor.owner_marginal_head.bias.grad is None


def test_unobserved_action_targets_contribute_zero_loss() -> None:
    model = _model()

    losses = event_v2_offline_loss(model, _batch(observed=False))

    torch.testing.assert_close(losses.task_distillation, torch.tensor(0.))
    torch.testing.assert_close(losses.termination, torch.tensor(0.))
    torch.testing.assert_close(losses.commitment, torch.tensor(0.))
    assert losses.value.item() > 0


def test_offline_loss_weights_control_total() -> None:
    model = _model()
    losses = event_v2_offline_loss(
        model,
        _batch(),
        task_weight=0.,
        termination_weight=0.,
        commitment_weight=0.,
        value_weight=2.,
    )

    torch.testing.assert_close(losses.total, 2 * losses.value)
