import torch

from constellation.data import Coordinate, Task, TaskSet
from constellation.data.constellations import SensorType
from tools.generate_local_action_branches import (
    build_candidate_pair_records,
    candidate_branch_specs,
    build_pair_summary,
    select_replayable_decisions,
    unique_candidate_branch_specs,
)


def _taskset() -> TaskSet:
    return TaskSet([
        Task(
            id_=3,
            release_time=0,
            due_time=100,
            duration=15,
            coordinate=Coordinate(0.0, 0.0),
            sensor_type=SensorType.VISIBLE,
        ),
        Task(
            id_=4,
            release_time=0,
            due_time=100,
            duration=15,
            coordinate=Coordinate(10.0, 10.0),
            sensor_type=SensorType.VISIBLE,
        ),
    ])


def test_select_replayable_decisions_reserves_the_longest_horizon() -> None:
    actions = torch.tensor([
        [-1],
        [3],
        [-1],
        [-1],
        [4],
        [-1],
    ])
    reference_progress = torch.zeros((6, 2), dtype=torch.uint8)

    decisions = select_replayable_decisions(
        actions=actions,
        reference_progress=reference_progress,
        taskset=_taskset(),
        horizons=[1, 3],
        max_decisions=2,
    )

    assert len(decisions) == 1
    assert decisions[0].decision_time == 1
    assert decisions[0].switch_task_id == 3


def test_select_replayable_decisions_spreads_across_scene_time() -> None:
    actions = torch.full((1000, 1), -1, dtype=torch.long)
    actions[1, 0] = 3
    actions[301, 0] = 3
    actions[601, 0] = 3
    reference_progress = torch.zeros((1000, 2), dtype=torch.uint8)
    taskset = TaskSet([
        Task(
            id_=3,
            release_time=0,
            due_time=1000,
            duration=15,
            coordinate=Coordinate(0.0, 0.0),
            sensor_type=SensorType.VISIBLE,
        ),
        Task(
            id_=4,
            release_time=0,
            due_time=1000,
            duration=15,
            coordinate=Coordinate(10.0, 10.0),
            sensor_type=SensorType.VISIBLE,
        ),
    ])

    decisions = select_replayable_decisions(
        actions=actions,
        reference_progress=reference_progress,
        taskset=taskset,
        horizons=[100],
        max_decisions=2,
    )

    assert [decision.decision_time for decision in decisions] == [1, 601]


def test_build_pair_summary_keeps_raw_deltas() -> None:
    stay = {
        'completed_tasks': 1,
        'partial_progress_gain': 0.5,
        'pc_wh': 2.0,
        'forced_task_id': -1,
        'decision_time': 8,
        'decision_state_signature': 'same-state',
        'branch': 'stay',
    }
    switch = {
        'completed_tasks': 2,
        'partial_progress_gain': 0.25,
        'pc_wh': 3.5,
        'forced_task_id': 86,
        'decision_time': 8,
        'decision_state_signature': 'same-state',
        'branch': 'switch',
    }

    summary = build_pair_summary(stay=stay, switch=switch)

    assert summary['stay'] == stay
    assert summary['switch'] == switch
    assert summary['switch_minus_stay'] == {
        'completed_tasks': 1.0,
        'partial_progress_gain': -0.25,
        'pc_wh': 1.5,
    }


def test_build_pair_summary_rejects_different_initial_states() -> None:
    stay = {'decision_state_signature': 'state-a'}
    switch = {'decision_state_signature': 'state-b'}

    try:
        build_pair_summary(stay=stay, switch=switch)
    except ValueError as error:
        assert 'different decision states' in str(error)
    else:
        raise AssertionError('different decision states must be rejected')


def test_candidate_branch_specs_include_stay_and_actor_top_k() -> None:
    specs = candidate_branch_specs(stay_task_id=31, top_k=3)

    assert specs == [
        {
            'name': 'stay',
            'forced_task_id': 31
        },
        {
            'name': 'actor_rank_0',
            'forced_candidate_rank': 0
        },
        {
            'name': 'actor_rank_1',
            'forced_candidate_rank': 1
        },
        {
            'name': 'actor_rank_2',
            'forced_candidate_rank': 2
        },
    ]


def test_candidate_branch_specs_cap_top_k_to_available_actions() -> None:
    specs = candidate_branch_specs(
        stay_task_id=-1,
        top_k=3,
        num_available_candidates=2,
    )

    assert [spec['name'] for spec in specs] == [
        'stay',
        'actor_rank_0',
        'actor_rank_1',
    ]


def test_unique_candidate_branch_specs_skip_duplicate_stay_action() -> None:
    specs = unique_candidate_branch_specs(
        stay_task_id=-1,
        actor_logits=[0.5, 1.0],
        ongoing_task_ids=[86],
        top_k=3,
    )

    assert specs == [
        {
            'name': 'stay',
            'forced_task_id': -1,
            'resolved_task_id': -1
        },
        {
            'name': 'actor_rank_0',
            'forced_candidate_rank': 0,
            'resolved_task_id': 86,
        },
    ]


def test_build_candidate_pair_records_uses_primary_prefix_cost() -> None:
    common = {
        'decision_state_signature': 'same',
        'decision_context': {
            'previous_assignment': [31]
        },
    }
    branches = {
        'stay': {
            **common,
            'applied_task_id': 31,
            'horizons': {
                '300': {
                    'prefix_metrics': {
                        'prefix_cost': 2.0
                    }
                }
            },
        },
        'actor_rank_0': {
            **common,
            'applied_task_id': 44,
            'horizons': {
                '300': {
                    'prefix_metrics': {
                        'prefix_cost': 1.5
                    }
                }
            },
        },
        'actor_rank_1': {
            **common,
            'applied_task_id': 57,
            'horizons': {
                '300': {
                    'prefix_metrics': {
                        'prefix_cost': None
                    }
                }
            },
        },
    }

    records = build_candidate_pair_records(branches, primary_horizon=300)

    assert records == [{
        'better_branch': 'actor_rank_0',
        'worse_branch': 'stay',
        'better_task_id': 44,
        'worse_task_id': 31,
        'better_cost': 1.5,
        'worse_cost': 2.0,
        'cost_margin': 0.5,
        'primary_horizon': 300,
    }]


def test_build_candidate_pair_records_deduplicates_identical_actions() -> None:
    common = {
        'decision_state_signature': 'same',
        'decision_context': {
            'previous_assignment': [-1]
        },
        'original_assignment': [31],
        'satellite_index': 0,
    }

    def branch(task_id: int, cost: float) -> dict:
        return {
            **common,
            'applied_task_id': task_id,
            'horizons': {
                '300': {
                    'prefix_metrics': {
                        'prefix_cost': cost
                    }
                }
            },
        }

    records = build_candidate_pair_records(
        {
            'stay': branch(-1, 1.0),
            'actor_rank_0': branch(31, 2.0),
            'actor_rank_1': branch(-1, 1.0),
        },
        primary_horizon=300,
    )

    assert len(records) == 1
    assert records[0]['better_branch'] == 'stay'
    assert records[0]['worse_branch'] == 'actor_rank_0'
