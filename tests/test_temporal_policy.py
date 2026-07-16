import numpy as np
import torch

from constellation.data import SensorType
from constellation.new_transformers.temporal_history import (
    CausalAssignmentHistory,
)
from constellation.rl.environment import (
    MAX_NUM_SATELLITES,
    MAX_NUM_TASKS,
    Padding,
    history_to_observation,
    map_relative_actions_to_global,
    null_observation,
)
from constellation.rl.policy import ActorCritic, FeatureExtractor


def test_null_observation_contains_reset_temporal_history() -> None:
    assert null_observation['previous_task_index'].shape == (
        MAX_NUM_SATELLITES,
    )
    assert np.all(null_observation['previous_task_index'] == -1)
    assert np.all(null_observation['previous_task_available'] == 0)
    assert np.all(null_observation['previous_was_idle'] == 1)
    assert np.all(null_observation['run_length'] == 0)
    assert np.all(null_observation['switch_count_30'] == 0)
    assert np.all(null_observation['switch_count_60'] == 0)


def test_history_to_observation_maps_current_candidates() -> None:
    state = CausalAssignmentHistory(num_satellites=2)
    state.record([7, -1])
    state.record([7, 9])

    values = history_to_observation(state, [3, 7, 9])

    assert values['previous_task_index'].tolist() == [1, 2]
    assert values['previous_task_available'].tolist() == [1, 1]
    assert values['previous_was_idle'].tolist() == [0, 0]
    assert values['run_length'].tolist() == [2., 1.]
    assert values['switch_count_30'].tolist() == [0., 1.]


def test_relative_actions_map_to_stable_global_ids() -> None:
    result = map_relative_actions_to_global(
        [1, -1, 3],
        [4, 8, 12],
    )

    assert result == [8, -1, -1]


def _policy_observation() -> dict[str, torch.Tensor]:
    num_satellites = torch.zeros(1, MAX_NUM_SATELLITES)
    num_satellites[0, 2] = 1
    num_tasks = torch.zeros(1, MAX_NUM_TASKS)
    num_tasks[0, 2] = 1
    time_step = torch.zeros(1, 3600)
    time_step[0, 1] = 1
    sensor_types = len(SensorType)
    constellation_sensor_type = torch.zeros(
        1, MAX_NUM_SATELLITES, sensor_types
    )
    constellation_sensor_type[..., 0] = 1
    tasks_sensor_type = torch.zeros(1, MAX_NUM_TASKS, sensor_types)
    tasks_sensor_type[..., 0] = 1
    return {
        'num_satellites': num_satellites,
        'num_tasks': num_tasks,
        'time_step': time_step,
        'constellation_sensor_type': constellation_sensor_type.flatten(1),
        'constellation_sensor_enabled': torch.ones(1, MAX_NUM_SATELLITES),
        'constellation_data': torch.zeros(1, MAX_NUM_SATELLITES, 56),
        'tasks_sensor_type': tasks_sensor_type.flatten(1),
        'tasks_data': torch.zeros(1, MAX_NUM_TASKS, 6),
        'previous_task_index': torch.tensor(
            [[1, -1] + [-1] * (MAX_NUM_SATELLITES - 2)]
        ),
        'previous_task_available': torch.tensor(
            [[1, 0] + [0] * (MAX_NUM_SATELLITES - 2)]
        ),
        'previous_was_idle': torch.tensor(
            [[0, 1] + [1] * (MAX_NUM_SATELLITES - 2)]
        ),
        'run_length': torch.tensor(
            [[3., 2.] + [0.] * (MAX_NUM_SATELLITES - 2)]
        ),
        'switch_count_30': torch.tensor(
            [[1., 0.] + [0.] * (MAX_NUM_SATELLITES - 2)]
        ),
        'switch_count_60': torch.tensor(
            [[2., 0.] + [0.] * (MAX_NUM_SATELLITES - 2)]
        ),
    }


def test_feature_extractor_preserves_temporal_history_fields() -> None:
    extractor = object.__new__(FeatureExtractor)

    batch = extractor.forward(_policy_observation())

    assert batch.previous_task_indices.tolist() == [[1, -1]]
    assert batch.previous_task_available.tolist() == [[True, False]]
    assert batch.previous_was_idle.tolist() == [[False, True]]
    assert batch.run_lengths.tolist() == [[3., 2.]]
    assert batch.switch_count_30.tolist() == [[1., 0.]]
    assert batch.switch_count_60.tolist() == [[2., 0.]]


def test_actor_critic_passes_history_to_temporal_model() -> None:
    extractor = object.__new__(FeatureExtractor)
    batch = extractor.forward(_policy_observation())
    model = ActorCritic(actor_model_kwargs=dict(
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
        use_temporal_adapter=True,
        temporal_adapter_hidden_width=8,
        temporal_horizons=(1,),
    )).eval()
    with torch.no_grad():
        model.actor._transformer._decoder._null_task.zero_()

    with torch.no_grad():
        logits = model.forward_actor(batch)

    assert logits.shape == (1, MAX_NUM_SATELLITES, MAX_NUM_TASKS)
    assert torch.isfinite(logits[:, :2, :3]).all()
