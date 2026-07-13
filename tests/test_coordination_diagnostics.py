import torch
from types import SimpleNamespace

from constellation.new_transformers.model import GLOBALS
from constellation.rl import coordination_diagnostics
from constellation.rl import policy as policy_module


def test_map_topk_logits_to_global_task_ids() -> None:
    logits = torch.tensor([
        [
            [0.0, 5.0, 4.0, 1.0, float('-inf')],
            [0.0, 2.0, 3.0, 6.0, float('-inf')],
        ],
    ])

    mapped = coordination_diagnostics.map_topk_task_ids(
        logits,
        ongoing_task_ids=[[10, 20, 30]],
        num_satellites=[2],
        top_k=2,
    )

    assert mapped == [[[10, 20], [30, 20]]]


def test_build_step_diagnostics_maps_relative_actions_and_progress() -> None:
    result = coordination_diagnostics.build_step_diagnostics(
        time_step=42,
        action=[1, 1, 0],
        ongoing_task_ids=[10, 20],
        all_task_ids=[10, 20, 30],
        progress_before=torch.tensor([0, 0, 0]),
        progress_after=torch.tensor([1, 0, 0]),
        is_visible=torch.tensor([
            [True, False, False],
            [False, False, False],
            [False, False, False],
        ]),
    )

    assert result == {
        'assignment': [10, 10, -1],
        'selected_visible': [True, False, False],
        'progress_made_task_ids': [10],
        'ongoing_task_ids': [10, 20],
        'time_step': 42,
    }


def test_policy_can_capture_actor_logits_for_diagnostics() -> None:
    policy = SimpleNamespace(action_dist=SimpleNamespace())
    logits = torch.tensor([[[1.0, 2.0]]])
    GLOBALS['capture_actor_logits'] = True
    try:
        policy_module.Policy._get_action_dist_from_latent(policy, logits)

        assert torch.equal(GLOBALS['actor_logits'], logits)
        assert GLOBALS['actor_logits'].device.type == 'cpu'
    finally:
        GLOBALS.pop('capture_actor_logits', None)
        GLOBALS.pop('actor_logits', None)


def test_scene_recorder_classifies_duplicates_relay_and_topk_coverage() -> None:
    recorder = coordination_diagnostics.SceneRecorder(scene_id=7, top_k=2)

    recorder.record_step(
        time_step=10,
        assignment=[10, 10],
        topk_task_ids=[[10, 20], [10, 20]],
        selected_visible=[True, False],
        progress_made_task_ids=[10],
    )
    recorder.record_step(
        time_step=11,
        assignment=[10, 10],
        topk_task_ids=[[10, 20], [10, 20]],
        selected_visible=[False, True],
        progress_made_task_ids=[10],
    )
    recorder.record_step(
        time_step=20,
        assignment=[20, 20],
        topk_task_ids=[[10, 20], [10, 20]],
        selected_visible=[False, False],
        progress_made_task_ids=[],
    )

    result = recorder.finalize(
        succeeded_task_ids=[10],
        failed_task_ids=[20],
        open_task_ids=[30],
    )

    assert result['duplicate_group_events'] == 3
    assert result['redundant_satellite_selections'] == 3
    assert result['duplicate_progress_events'] == 2
    assert result['duplicate_stalled_events'] == 1
    assert result['duplicate_tasks'] == 2
    assert result['duplicate_tasks_succeeded'] == 1
    assert result['duplicate_tasks_failed'] == 1
    assert result['duplicate_tasks_open'] == 0
    assert result['relay_supported_tasks'] == 1
    assert result['unfinished_tasks'] == 2
    assert result['unfinished_never_topk'] == 1
    assert result['unfinished_never_topk_ids'] == [30]
    assert result['unfinished_ever_topk_never_selected'] == 0
    assert result['time_step_stats'] == {
        '10': {
            'active_satellite_selections': 2,
            'duplicate_group_events': 1,
            'redundant_satellite_selections': 1,
        },
        '11': {
            'active_satellite_selections': 2,
            'duplicate_group_events': 1,
            'redundant_satellite_selections': 1,
        },
        '20': {
            'active_satellite_selections': 2,
            'duplicate_group_events': 1,
            'redundant_satellite_selections': 1,
        },
    }


def test_summarize_scene_results_reports_diagnostic_rates() -> None:
    summary = coordination_diagnostics.summarize_scene_results([
        {
            'duplicate_group_events': 4,
            'duplicate_progress_events': 1,
            'duplicate_stalled_events': 3,
            'duplicate_tasks': 2,
            'duplicate_tasks_succeeded': 1,
            'duplicate_tasks_failed': 1,
            'duplicate_tasks_open': 0,
            'relay_supported_tasks': 1,
            'redundant_satellite_selections': 5,
            'active_satellite_selections': 20,
            'unfinished_tasks': 4,
            'unfinished_never_topk': 2,
            'unfinished_ever_topk_never_selected': 1,
            'time_step_stats': {
                '10': {
                    'active_satellite_selections': 4,
                    'duplicate_group_events': 1,
                    'redundant_satellite_selections': 2,
                },
            },
        },
    ])

    assert summary['duplicate_selection_rate'] == 0.25
    assert summary['duplicate_progress_rate'] == 0.25
    assert summary['duplicate_task_success_rate'] == 0.5
    assert summary['relay_support_rate'] == 0.5
    assert summary['unfinished_never_topk_rate'] == 0.5
    assert summary['unfinished_ever_topk_never_selected_rate'] == 0.25
    assert summary['time_step_stats'] == {
        '10': {
            'active_satellite_selections': 4,
            'duplicate_group_events': 1,
            'redundant_satellite_selections': 2,
            'duplicate_selection_rate': 0.5,
        },
    }
