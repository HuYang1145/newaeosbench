import pytest
import torch

from constellation.new_transformers.event_v2.appo import (
    APPOConfig,
    AsynchronousPPOLearner,
    filter_policy_lag,
)
from constellation.new_transformers.event_v2.model import (
    EventJointActorCritic,
)
from constellation.new_transformers.event_v2.observation import (
    EventPolicyObservation,
)
from constellation.new_transformers.event_v2.rollout import StoredEventStep
from constellation.new_transformers.event_v2.state import EventStateTensors
from constellation.new_transformers.event_v2.ppo import PPOConfig


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


def _observation(time_step: int = 0) -> EventPolicyObservation:
    satellite_shape = (1, 2)
    task_shape = (1, 3)
    state = EventStateTensors(
        previous_task_indices=torch.full(
            satellite_shape,
            -1,
            dtype=torch.long,
        ),
        current_task_indices=torch.full(
            satellite_shape,
            -1,
            dtype=torch.long,
        ),
        minimum_commitment_remaining=torch.zeros(satellite_shape),
        run_lengths=torch.full(satellite_shape, float(time_step)),
        seconds_since_replan=torch.full(
            satellite_shape,
            float(time_step),
        ),
        switch_count_30=torch.zeros(satellite_shape),
        switch_count_60=torch.zeros(satellite_shape),
        termination_reason=torch.zeros(
            satellite_shape,
            dtype=torch.long,
        ),
        event_type=torch.full(satellite_shape, 3, dtype=torch.long),
        delta_t=torch.full(satellite_shape, 5.),
        replan_mask=torch.ones(satellite_shape, dtype=torch.bool),
        forced_interrupt_mask=torch.zeros(
            satellite_shape,
            dtype=torch.bool,
        ),
        can_terminate_mask=torch.zeros(
            satellite_shape,
            dtype=torch.bool,
        ),
        compatible_deadline_slack=torch.tensor([[10., 20.]]),
        task_remaining_required_seconds=torch.tensor([[10., 30., 60.]]),
        task_owner_count=torch.zeros(task_shape, dtype=torch.long),
        task_locked_owner_count=torch.zeros(task_shape, dtype=torch.long),
    )
    return EventPolicyObservation(
        time_steps=torch.tensor([time_step]),
        constellation_sensor_type=torch.zeros(
            satellite_shape,
            dtype=torch.long,
        ),
        constellation_sensor_enabled=torch.ones(
            satellite_shape,
            dtype=torch.long,
        ),
        constellation_data=torch.zeros(1, 2, 56),
        constellation_mask=torch.ones(satellite_shape, dtype=torch.bool),
        tasks_sensor_type=torch.zeros(task_shape, dtype=torch.long),
        tasks_data=torch.zeros(1, 3, 6),
        tasks_mask=torch.ones(task_shape, dtype=torch.bool),
        event_state=state,
    )


def _step(policy_version: int = 0) -> StoredEventStep:
    model = _model().eval()
    observation = _observation()
    with torch.inference_mode():
        output = model.act(
            *observation.model_args(),
            event_state=observation.event_state,
            deterministic=False,
        )
    step = StoredEventStep(
        environment_index=0,
        episode_id=0,
        event_index=0,
        observation=observation,
        action=output.actor.action,
        trace=output.actor.trace,
        behavior_log_prob=output.actor.log_prob[0].detach().clone(),
        value=output.value[0].detach().clone(),
        reward=torch.tensor(0.1),
        delta_t=torch.tensor(5.),
        next_observation=None,
        next_value=torch.tensor(0.),
        done=torch.tensor(True),
        policy_version=policy_version,
    )
    step.validate()
    return step


def test_policy_lag_filter_keeps_current_and_bounded_old_events() -> None:
    template = _step()
    steps = [
        template._replace(event_index=index, policy_version=version)
        for index, version in enumerate((7, 6, 5, 4))
    ]

    result = filter_policy_lag(
        steps,
        current_policy_version=7,
        max_policy_lag=2,
    )

    assert [step.policy_version for step in result.accepted] == [7, 6, 5]
    assert result.stale_dropped == 1
    assert result.minimum_version == 4
    assert result.maximum_version == 7


def test_policy_lag_filter_rejects_future_events() -> None:
    with pytest.raises(ValueError, match='future policy version'):
        filter_policy_lag(
            [_step(policy_version=8)],
            current_policy_version=7,
            max_policy_lag=2,
        )


def test_policy_lag_configuration_rejects_invalid_boundaries() -> None:
    with pytest.raises(ValueError, match='policy lag'):
        APPOConfig(max_policy_lag=-1)
    with pytest.raises(ValueError, match='at least one'):
        filter_policy_lag(
            [],
            current_policy_version=0,
            max_policy_lag=0,
        )
    with pytest.raises(ValueError, match='current policy version'):
        filter_policy_lag(
            [_step()],
            current_policy_version=-1,
            max_policy_lag=0,
        )


def test_appo_learner_accepts_bounded_old_behavior_and_updates_only_trainable(
) -> None:
    torch.manual_seed(31)
    behavior_model = _model().eval()
    learner_model = _model()
    learner_model.load_state_dict(behavior_model.state_dict())
    learner_model.unfreeze_last_layers(
        encoder_layers=1,
        decoder_layers=1,
    )
    parameter_groups = learner_model.parameter_groups(
        new_module_lr=1e-3,
        backbone_lr_scale=0.1,
    )
    assert [group['lr'] for group in parameter_groups] == pytest.approx(
        [1e-3, 1e-4],
    )
    optimizer = torch.optim.AdamW(parameter_groups)
    template = _step_from_model(behavior_model, policy_version=0)
    steps = [
        template._replace(environment_index=index, episode_id=index)
        for index in range(4)
    ]
    with torch.no_grad():
        next(learner_model.actor.parameters()).add_(0.01)
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in learner_model.backbone.transformer.named_parameters()
        if not parameter.requires_grad
    }
    trainable_before = {
        name: parameter.detach().clone()
        for name, parameter in learner_model.named_parameters()
        if parameter.requires_grad
    }
    learner = AsynchronousPPOLearner(
        model=learner_model,
        optimizer=optimizer,
        ppo_config=PPOConfig(
            ppo_epochs=1,
            minibatch_events=4,
            max_kl=10.,
        ),
        appo_config=APPOConfig(max_policy_lag=2),
        device=torch.device('cpu'),
    )
    learner.policy_version = 1

    metrics = learner.update(steps)

    assert metrics.input_events == 4
    assert metrics.accepted_events == 4
    assert metrics.stale_dropped_events == 0
    assert metrics.minimum_behavior_version == 0
    assert metrics.maximum_behavior_version == 0
    assert metrics.ppo.policy_version == 2
    assert metrics.ppo.frozen_parameter_changes == 0
    for name, expected in frozen_before.items():
        torch.testing.assert_close(
            learner_model.backbone.transformer.get_parameter(name),
            expected,
            rtol=0,
            atol=0,
        )
    assert any(
        not torch.equal(parameter.detach(), trainable_before[name])
        for name, parameter in learner_model.named_parameters()
        if parameter.requires_grad
    )


def test_appo_learner_counts_stale_events_before_updating() -> None:
    model = _model()
    model.unfreeze_last_layers(encoder_layers=1, decoder_layers=1)
    optimizer = torch.optim.AdamW(model.parameter_groups(1e-3))
    learner = AsynchronousPPOLearner(
        model=model,
        optimizer=optimizer,
        ppo_config=PPOConfig(
            ppo_epochs=1,
            minibatch_events=2,
            max_kl=10.,
        ),
        appo_config=APPOConfig(max_policy_lag=1),
        device=torch.device('cpu'),
    )
    learner.policy_version = 2
    fresh = _step_from_model(model, policy_version=1)
    stale = fresh._replace(
        environment_index=1,
        episode_id=1,
        policy_version=0,
    )

    metrics = learner.update([fresh, stale])

    assert metrics.input_events == 2
    assert metrics.accepted_events == 1
    assert metrics.stale_dropped_events == 1


def _step_from_model(
    model: EventJointActorCritic,
    *,
    policy_version: int,
) -> StoredEventStep:
    observation = _observation()
    model.eval()
    with torch.inference_mode():
        output = model.act(
            *observation.model_args(),
            event_state=observation.event_state,
            deterministic=False,
        )
    step = StoredEventStep(
        environment_index=0,
        episode_id=0,
        event_index=0,
        observation=observation,
        action=output.actor.action,
        trace=output.actor.trace,
        behavior_log_prob=output.actor.log_prob[0].detach().clone(),
        value=output.value[0].detach().clone(),
        reward=torch.tensor(0.1),
        delta_t=torch.tensor(5.),
        next_observation=None,
        next_value=torch.tensor(0.),
        done=torch.tensor(True),
        policy_version=policy_version,
    )
    step.validate()
    return step
