import pytest
import torch

from constellation.new_transformers.event_v2.observation import (
    EventPolicyObservation,
    stack_event_observations,
)
from constellation.new_transformers.event_v2.state import EventStateTensors


def _state(batch_size: int, num_satellites: int, num_tasks: int) -> EventStateTensors:
    satellite_shape = (batch_size, num_satellites)
    task_shape = (batch_size, num_tasks)
    return EventStateTensors(
        previous_task_indices=torch.full(satellite_shape, -1, dtype=torch.long),
        current_task_indices=torch.full(satellite_shape, -1, dtype=torch.long),
        minimum_commitment_remaining=torch.zeros(satellite_shape),
        run_lengths=torch.zeros(satellite_shape),
        seconds_since_replan=torch.zeros(satellite_shape),
        switch_count_30=torch.zeros(satellite_shape),
        switch_count_60=torch.zeros(satellite_shape),
        termination_reason=torch.zeros(satellite_shape, dtype=torch.long),
        event_type=torch.zeros(satellite_shape, dtype=torch.long),
        delta_t=torch.zeros(satellite_shape),
        replan_mask=torch.ones(satellite_shape, dtype=torch.bool),
        forced_interrupt_mask=torch.zeros(satellite_shape, dtype=torch.bool),
        can_terminate_mask=torch.zeros(satellite_shape, dtype=torch.bool),
        compatible_deadline_slack=torch.full(satellite_shape, 60.),
        task_remaining_required_seconds=torch.ones(task_shape),
        task_owner_count=torch.zeros(task_shape, dtype=torch.long),
        task_locked_owner_count=torch.zeros(task_shape, dtype=torch.long),
    )


def _observation(
    batch_size: int = 1,
    num_satellites: int = 2,
    num_tasks: int = 3,
) -> EventPolicyObservation:
    return EventPolicyObservation(
        time_steps=torch.arange(batch_size, dtype=torch.long),
        constellation_sensor_type=torch.zeros(
            batch_size,
            num_satellites,
            dtype=torch.long,
        ),
        constellation_sensor_enabled=torch.ones(
            batch_size,
            num_satellites,
            dtype=torch.long,
        ),
        constellation_data=torch.randn(batch_size, num_satellites, 56),
        constellation_mask=torch.ones(
            batch_size,
            num_satellites,
            dtype=torch.bool,
        ),
        tasks_sensor_type=torch.zeros(
            batch_size,
            num_tasks,
            dtype=torch.long,
        ),
        tasks_data=torch.randn(batch_size, num_tasks, 6),
        tasks_mask=torch.ones(
            batch_size,
            num_tasks,
            dtype=torch.bool,
        ),
        event_state=_state(batch_size, num_satellites, num_tasks),
    )


def test_event_policy_observation_validates_and_moves_named_tensors() -> None:
    observation = _observation(batch_size=2, num_satellites=3, num_tasks=4)

    observation.validate()
    moved = observation.to(torch.device('cpu'))

    assert observation.batch_size == 2
    assert observation.num_satellites == 3
    assert observation.num_tasks == 4
    assert moved.event_state.replan_mask.shape == (2, 3)
    assert all(value.device.type == 'cpu' for value in moved.model_args())


def test_event_policy_observation_rejects_task_mask_shape_mismatch() -> None:
    observation = _observation()

    with pytest.raises(ValueError, match='task mask'):
        observation._replace(
            tasks_mask=torch.ones(1, 4, dtype=torch.bool),
        ).validate()


def test_event_policy_observation_rejects_non_boolean_masks() -> None:
    observation = _observation()

    with pytest.raises(ValueError, match='constellation mask'):
        observation._replace(
            constellation_mask=torch.ones(1, 2),
        ).validate()


def test_stack_event_observations_concatenates_all_fields() -> None:
    first = _observation()
    second = _observation()._replace(time_steps=torch.tensor([7]))

    stacked = stack_event_observations([first, second])

    stacked.validate()
    assert stacked.batch_size == 2
    assert stacked.time_steps.tolist() == [0, 7]
    assert stacked.event_state.task_owner_count.shape == (2, 3)


def test_stack_event_observations_rejects_different_scene_shapes() -> None:
    with pytest.raises(ValueError, match='same satellite and task shapes'):
        stack_event_observations([
            _observation(num_tasks=3),
            _observation(num_tasks=4),
        ])


def test_stack_event_observations_requires_single_scene_items() -> None:
    with pytest.raises(ValueError, match='one scene'):
        stack_event_observations([_observation(batch_size=2)])
