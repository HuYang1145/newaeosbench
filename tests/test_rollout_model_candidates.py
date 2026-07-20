import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from constellation.data import Coordinate, Task, TaskSet
from constellation.data.constellations import SensorType
from constellation.new_transformers.event_policy import EventActorRuntime
from tools.rollout_model_trajectories import (
    EventGreedyModelAlgorithm,
    GreedyModelAlgorithm,
    actions_from_assignment,
    ranked_task_candidates,
    select_action_indices,
    summarize_event_history,
    validate_event_options,
    write_rollout_metadata,
)


class _PredictOnlyModel:

    def __init__(self):
        self.calls = 0

    def predict(self, *args):
        del args
        self.calls += 1
        return torch.tensor([[[0.0, 3.0, 2.0]]])


class _SingleSatelliteConstellation:

    def __len__(self):
        return 1

    def sort(self):
        sensor = type('Sensor', (), {'enabled': False})()
        return [type('Satellite', (), {'sensor': sensor})()]


def _candidate_tasks() -> TaskSet:
    return TaskSet([
        Task(
            id_=31,
            release_time=0,
            due_time=100,
            duration=10,
            coordinate=Coordinate(0.0, 0.0),
            sensor_type=SensorType.VISIBLE,
        ),
        Task(
            id_=44,
            release_time=0,
            due_time=100,
            duration=10,
            coordinate=Coordinate(10.0, 10.0),
            sensor_type=SensorType.VISIBLE,
        ),
    ])


def _replacement_candidate_tasks() -> TaskSet:
    return TaskSet([
        Task(
            id_=44,
            release_time=0,
            due_time=100,
            duration=10,
            coordinate=Coordinate(10.0, 10.0),
            sensor_type=SensorType.VISIBLE,
        ),
        Task(
            id_=57,
            release_time=0,
            due_time=100,
            duration=10,
            coordinate=Coordinate(20.0, 20.0),
            sensor_type=SensorType.VISIBLE,
        ),
    ])


def test_greedy_candidate_matches_existing_argmax() -> None:
    logits = torch.tensor([
        [[1.0, 4.0, 2.0], [3.0, 2.0, 5.0]],
    ])

    selected = select_action_indices(logits, strategy='greedy')

    assert torch.equal(selected, logits.argmax(-1))


def test_top_k_candidate_is_seeded_and_stays_inside_top_k() -> None:
    logits = torch.tensor([
        [[5.0, 4.0, 3.0, -100.0], [1.0, 3.0, 2.0, -100.0]],
    ])

    first = select_action_indices(
        logits,
        strategy='top_k_sample',
        top_k=2,
        temperature=0.7,
        generator=torch.Generator().manual_seed(17),
    )
    second = select_action_indices(
        logits,
        strategy='top_k_sample',
        top_k=2,
        temperature=0.7,
        generator=torch.Generator().manual_seed(17),
    )

    assert torch.equal(first, second)
    assert first[0, 0].item() in {0, 1}
    assert first[0, 1].item() in {1, 2}


def test_ranked_task_candidates_map_null_and_ongoing_task_ids() -> None:
    candidates = ranked_task_candidates(
        torch.tensor([5.0, 9.0, 7.0, 8.0]),
        task_ids=torch.tensor([31, 44, 57]),
        top_k=3,
    )

    assert candidates == [31, 57, 44]


def test_ranked_task_candidates_can_include_idle() -> None:
    candidates = ranked_task_candidates(
        torch.tensor([10.0, 9.0, 8.0]),
        task_ids=torch.tensor([31, 44]),
        top_k=2,
    )

    assert candidates == [-1, 31]


def test_greedy_algorithm_exposes_current_logits_and_task_ids() -> None:
    algorithm = object.__new__(GreedyModelAlgorithm)
    algorithm._model = _PredictOnlyModel()
    algorithm._build_inputs = lambda *args: ()
    algorithm._strategy = 'greedy'
    algorithm._top_k = 3
    algorithm._temperature = 0.7
    algorithm._generator = torch.Generator().manual_seed(1)

    _, assignment = algorithm.step(
        _candidate_tasks(),
        _SingleSatelliteConstellation(),
        torch.eye(3),
    )

    assert assignment == [31]
    assert torch.equal(
        algorithm.last_logits, torch.tensor([[[0.0, 3.0, 2.0]]])
    )
    assert torch.equal(algorithm.last_task_ids, torch.tensor([31, 44]))


def _event_algorithm(
    *,
    commitment_seconds: int = 5,
) -> EventGreedyModelAlgorithm:
    algorithm = object.__new__(EventGreedyModelAlgorithm)
    algorithm._model = _PredictOnlyModel()
    algorithm._build_inputs = lambda *args: ()
    algorithm._strategy = 'greedy'
    algorithm._top_k = 3
    algorithm._temperature = 0.7
    algorithm._generator = torch.Generator().manual_seed(1)
    algorithm._timer = SimpleNamespace(time=0)
    algorithm._runtime = EventActorRuntime(num_satellites=1)
    algorithm._event_commitment_seconds = commitment_seconds
    algorithm.model_call_count = 0
    algorithm.event_history = []
    return algorithm


def test_event_actor_skips_model_until_commitment_expires() -> None:
    algorithm = _event_algorithm(commitment_seconds=5)

    _, first = algorithm.step(
        _candidate_tasks(),
        _SingleSatelliteConstellation(),
        torch.eye(3),
    )
    algorithm._timer.time = 1
    _, second = algorithm.step(
        _candidate_tasks(),
        _SingleSatelliteConstellation(),
        torch.eye(3),
    )

    assert first == second == [31]
    assert algorithm.model_call_count == 1
    assert algorithm._model.calls == 1

    algorithm._timer.time = 5
    algorithm.step(
        _candidate_tasks(),
        _SingleSatelliteConstellation(),
        torch.eye(3),
    )
    assert algorithm.model_call_count == 2
    assert algorithm._model.calls == 2


def test_event_actor_replans_when_committed_task_disappears() -> None:
    algorithm = _event_algorithm(commitment_seconds=30)
    algorithm.step(
        _candidate_tasks(),
        _SingleSatelliteConstellation(),
        torch.eye(3),
    )

    algorithm._timer.time = 1
    _, assignment = algorithm.step(
        _replacement_candidate_tasks(),
        _SingleSatelliteConstellation(),
        torch.eye(3),
    )

    assert assignment == [44]
    assert algorithm.model_call_count == 2
    assert algorithm.event_history[-1]['trigger'] == 'task_unavailable'


def test_actions_from_assignment_preserves_global_task_identity() -> None:
    actions = actions_from_assignment(
        assignment=[44],
        taskset=_candidate_tasks(),
        constellation=_SingleSatelliteConstellation(),
    )

    assert len(actions) == 1
    assert actions[0].target_location == Coordinate(10.0, 10.0)
    assert actions[0].toggle is True


def test_event_history_summary_separates_idle_and_task_commitments() -> None:
    summary = summarize_event_history([
        {
            'time': 0,
            'satellite_index': 0,
            'task_id': -1,
            'commitment_seconds': 1,
            'trigger': 'initial',
        },
        {
            'time': 1,
            'satellite_index': 0,
            'task_id': 31,
            'commitment_seconds': 5,
            'trigger': 'expired',
        },
        {
            'time': 6,
            'satellite_index': 0,
            'task_id': 44,
            'commitment_seconds': 1,
            'trigger': 'task_unavailable',
        },
    ])

    assert summary['commitment_count'] == 3
    assert summary['task_commitment_count'] == 2
    assert summary['task_one_second_commitment_rate'] == 0.5
    assert summary['task_mean_commitment_seconds'] == 3.0
    assert summary['duration_counts'] == {'1': 2, '5': 1}
    assert summary['trigger_counts'] == {
        'initial': 1,
        'expired': 1,
        'task_unavailable': 1,
    }


@pytest.mark.parametrize('duration', [1, 5, 15, 30, 60])
def test_event_options_accept_supported_commitments(duration: int) -> None:
    validate_event_options(
        event_actor=True,
        event_commitment_seconds=duration,
    )


def test_event_options_require_explicit_commitment() -> None:
    with pytest.raises(ValueError, match='event-commitment-seconds'):
        validate_event_options(
            event_actor=True,
            event_commitment_seconds=None,
        )


def test_event_options_reject_commitment_when_disabled() -> None:
    with pytest.raises(ValueError, match='requires --event-actor'):
        validate_event_options(
            event_actor=False,
            event_commitment_seconds=5,
        )


@pytest.mark.parametrize(
    ('top_k', 'temperature'),
    [(0, 0.7), (3, 0.0), (3, -1.0)],
)
def test_top_k_candidate_rejects_invalid_sampling_parameters(
    top_k: int,
    temperature: float,
) -> None:
    with pytest.raises(ValueError):
        select_action_indices(
            torch.zeros(1, 2, 4),
            strategy='top_k_sample',
            top_k=top_k,
            temperature=temperature,
        )


def test_only_rank_zero_writes_shared_rollout_metadata(tmp_path: Path) -> None:
    metadata = {'strategy': 'top_k_sample', 'seed': 17}

    write_rollout_metadata(tmp_path, metadata=metadata, rank=1)
    assert not (tmp_path / 'rollout_metadata.json').exists()

    write_rollout_metadata(tmp_path, metadata=metadata, rank=0)
    assert json.loads((tmp_path
                       / 'rollout_metadata.json').read_text(encoding='utf-8'
                                                            ), ) == metadata
