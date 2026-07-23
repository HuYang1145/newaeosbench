import copy

import pytest
import torch

import constellation.new_transformers.event_v2.ppo as ppo_module
from constellation.new_transformers.event_v2.model import EventJointActorCritic
from constellation.new_transformers.event_v2.observation import (
    EventPolicyObservation,
)
from constellation.new_transformers.event_v2.ppo import (
    PPOConfig,
    PPOUpdateRejected,
    SynchronousPPOTrainer,
    clipped_ppo_objective,
    compute_rollout_targets,
    event_action_component_counts,
)
from constellation.new_transformers.event_v2.rollout import StoredEventStep
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


def _observation(time_step: int) -> EventPolicyObservation:
    satellite_shape = (1, 2)
    task_shape = (1, 3)
    state = EventStateTensors(
        previous_task_indices=torch.full(satellite_shape, -1, dtype=torch.long),
        current_task_indices=torch.full(satellite_shape, -1, dtype=torch.long),
        minimum_commitment_remaining=torch.zeros(satellite_shape),
        run_lengths=torch.full(satellite_shape, float(time_step)),
        seconds_since_replan=torch.full(satellite_shape, float(time_step)),
        switch_count_30=torch.zeros(satellite_shape),
        switch_count_60=torch.zeros(satellite_shape),
        termination_reason=torch.zeros(satellite_shape, dtype=torch.long),
        event_type=torch.full(satellite_shape, 3, dtype=torch.long),
        delta_t=torch.full(satellite_shape, 5.),
        replan_mask=torch.ones(satellite_shape, dtype=torch.bool),
        forced_interrupt_mask=torch.zeros(satellite_shape, dtype=torch.bool),
        can_terminate_mask=torch.zeros(satellite_shape, dtype=torch.bool),
        compatible_deadline_slack=torch.tensor([[10., 20.]]),
        task_remaining_required_seconds=torch.tensor([[10., 30., 60.]]),
        task_owner_count=torch.zeros(task_shape, dtype=torch.long),
        task_locked_owner_count=torch.zeros(task_shape, dtype=torch.long),
    )
    return EventPolicyObservation(
        time_steps=torch.tensor([time_step]),
        constellation_sensor_type=torch.zeros(satellite_shape, dtype=torch.long),
        constellation_sensor_enabled=torch.ones(satellite_shape, dtype=torch.long),
        constellation_data=torch.zeros(1, 2, 56),
        constellation_mask=torch.ones(satellite_shape, dtype=torch.bool),
        tasks_sensor_type=torch.zeros(task_shape, dtype=torch.long),
        tasks_data=torch.zeros(1, 3, 6),
        tasks_mask=torch.ones(task_shape, dtype=torch.bool),
        event_state=state,
    )


def _rollout(model: EventJointActorCritic, count: int = 4) -> list[StoredEventStep]:
    model.eval()
    steps: list[StoredEventStep] = []
    with torch.inference_mode():
        for index in range(count):
            observation = _observation(index * 5)
            output = model.act(
                *observation.model_args(),
                event_state=observation.event_state,
                deterministic=False,
            )
            done = index == count - 1
            step = StoredEventStep(
                environment_index=0,
                episode_id=0,
                event_index=index,
                observation=observation,
                action=output.actor.action,
                trace=output.actor.trace,
                behavior_log_prob=output.actor.log_prob[0].detach().clone(),
                value=output.value[0].detach().clone(),
                reward=torch.tensor((index + 1) / 10),
                delta_t=torch.tensor(5.),
                next_observation=None if done else _observation((index + 1) * 5),
                next_value=(
                    torch.tensor(0.)
                    if done
                    else output.value[0].detach().clone()
                ),
                done=torch.tensor(done),
                policy_version=0,
            )
            step.validate()
            steps.append(step)
    return steps


def _state_dict_clone(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }


def test_clipped_objective_uses_joint_event_probability() -> None:
    output = clipped_ppo_objective(
        new_log_prob=torch.log(torch.tensor([1.3, 0.7])),
        behavior_log_prob=torch.zeros(2),
        advantages=torch.tensor([1., -1.]),
        clip_ratio=0.2,
    )

    assert output.ratio.tolist() == pytest.approx([1.3, 0.7])
    assert output.policy_loss.item() == pytest.approx(-0.2)
    assert output.clip_fraction.item() == pytest.approx(1.)


def test_rollout_targets_preserve_episode_order_and_terminal_bootstrap() -> None:
    model = _model()
    steps = _rollout(model, count=3)

    targets = compute_rollout_targets(
        steps,
        lambda_base=0.95,
        reference_seconds=5.,
        normalize_advantages=False,
    )

    assert targets.advantages.shape == (3,)
    assert targets.returns.shape == (3,)
    assert torch.isfinite(targets.advantages).all()
    assert targets.returns[-1] == pytest.approx(steps[-1].reward.item())


def test_entropy_is_normalized_by_actual_joint_action_components() -> None:
    model = _model()
    step = _rollout(model, count=1)[0]

    counts = event_action_component_counts([step])

    expected = (
        step.trace.termination_mask.sum()
        + (step.trace.action_order >= 0).sum()
        + (step.action.commitment_indices >= 0).sum()
    )
    assert counts.tolist() == [float(expected)]


def test_update_changes_new_modules_but_not_frozen_stage3() -> None:
    torch.manual_seed(12)
    model = _model()
    steps = _rollout(model)
    frozen_before = {
        name: value.detach().clone()
        for name, value in model.backbone.transformer.state_dict().items()
    }
    trainable_before = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if value.requires_grad
    }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    trainer = SynchronousPPOTrainer(
        model=model,
        optimizer=optimizer,
        config=PPOConfig(ppo_epochs=2, minibatch_events=4, max_kl=10.),
        device=torch.device('cpu'),
    )

    metrics = trainer.update(steps)

    for name, expected in frozen_before.items():
        torch.testing.assert_close(
            model.backbone.transformer.state_dict()[name],
            expected,
            rtol=0,
            atol=0,
        )
    assert any(
        not torch.equal(parameter.detach(), trainable_before[name])
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    assert metrics.frozen_parameter_changes == 0
    assert metrics.policy_version == 1
    assert torch.isfinite(torch.tensor(metrics.total_loss))


def test_later_epoch_kl_breach_rolls_back_that_epoch_and_early_stops(
    monkeypatch,
) -> None:
    torch.manual_seed(120)
    model = _model()
    steps = _rollout(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    original_objective = ppo_module.clipped_ppo_objective
    calls = 0

    def force_second_epoch_kl(*args, **kwargs):
        nonlocal calls
        calls += 1
        objective = original_objective(*args, **kwargs)
        if calls == 2:
            return objective._replace(
                approx_kl=objective.approx_kl.new_tensor(11.),
            )
        return objective

    monkeypatch.setattr(
        ppo_module,
        'clipped_ppo_objective',
        force_second_epoch_kl,
    )
    trainer = SynchronousPPOTrainer(
        model=model,
        optimizer=optimizer,
        config=PPOConfig(
            ppo_epochs=4,
            minibatch_events=4,
            max_kl=10.,
        ),
        device=torch.device('cpu'),
    )

    metrics = trainer.update(steps)

    assert metrics.early_stopped is True
    assert metrics.completed_epochs == 1
    assert metrics.policy_version == 1
    assert metrics.approx_kl <= 10.


def test_first_epoch_local_kl_breach_keeps_globally_safe_partial_epoch(
    monkeypatch,
) -> None:
    torch.manual_seed(121)
    model = _model()
    steps = _rollout(model, count=8)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    original_objective = ppo_module.clipped_ppo_objective
    calls = 0

    def force_second_minibatch_kl(*args, **kwargs):
        nonlocal calls
        calls += 1
        objective = original_objective(*args, **kwargs)
        if calls == 2:
            return objective._replace(
                approx_kl=objective.approx_kl.new_tensor(11.),
            )
        return objective

    monkeypatch.setattr(
        ppo_module,
        'clipped_ppo_objective',
        force_second_minibatch_kl,
    )
    trainer = SynchronousPPOTrainer(
        model=model,
        optimizer=optimizer,
        config=PPOConfig(
            ppo_epochs=4,
            minibatch_events=4,
            max_kl=10.,
        ),
        device=torch.device('cpu'),
    )

    metrics = trainer.update(steps)

    assert metrics.early_stopped is True
    assert metrics.completed_epochs == 1
    assert metrics.policy_version == 1
    assert metrics.approx_kl <= 10.


class CorruptingAdamW(torch.optim.AdamW):
    def step(self, closure=None):
        loss = super().step(closure)
        with torch.no_grad():
            self.param_groups[0]['params'][0].flatten()[0] = float('nan')
        return loss


def test_nonfinite_update_restores_preupdate_model() -> None:
    torch.manual_seed(13)
    model = _model()
    steps = _rollout(model)
    before = _state_dict_clone(model)
    optimizer = CorruptingAdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    trainer = SynchronousPPOTrainer(
        model=model,
        optimizer=optimizer,
        config=PPOConfig(ppo_epochs=1, minibatch_events=4, max_kl=10.),
        device=torch.device('cpu'),
    )

    with pytest.raises(PPOUpdateRejected, match='non-finite'):
        trainer.update(steps)

    for name, expected in before.items():
        torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)


def test_behavior_replay_mismatch_is_rejected_before_update() -> None:
    model = _model()
    steps = _rollout(model)
    steps[0] = steps[0]._replace(
        behavior_log_prob=steps[0].behavior_log_prob + 1,
    )
    before = _state_dict_clone(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    trainer = SynchronousPPOTrainer(
        model=model,
        optimizer=optimizer,
        config=PPOConfig(),
        device=torch.device('cpu'),
    )

    with pytest.raises(PPOUpdateRejected, match='behavior log-prob'):
        trainer.update(steps)

    for name, expected in before.items():
        torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)
