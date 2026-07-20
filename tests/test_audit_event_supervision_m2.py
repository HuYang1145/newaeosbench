import runpy
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]
TOOL = ROOT / 'tools' / 'audit_event_supervision_m2.py'


def _module() -> dict:
    assert TOOL.is_file(), 'M2 event supervision audit tool is missing'
    return runpy.run_path(str(TOOL))


def test_event_audit_summarizes_continue_duration_and_censor() -> None:
    summarize = _module()['summarize_event_targets']
    actions = torch.tensor([
        [0, 1, -1],
        [0, 1, -1],
        [2, 1, -1],
        [2, 3, -1],
        [2, 3, -1],
    ])

    summary = summarize(actions, commitments=(1, 5))

    assert summary['edge_count'] == 8
    assert summary['continue_count'] == 6
    assert summary['stop_count'] == 2
    assert summary['duration_observed_count'] == 5
    assert summary['duration_censored_count'] == 3
    assert summary['duration_counts'] == {'1': 5, '5': 0}


def test_event_audit_aggregation_recomputes_rates() -> None:
    module = _module()
    summarize = module['summarize_event_targets']
    aggregate = module['aggregate_event_summaries']
    first = summarize(torch.tensor([[0], [1]]), commitments=(1, 5))
    second = summarize(
        torch.tensor([[0], [0], [0], [1]]),
        commitments=(1, 5),
    )

    combined = aggregate([first, second], commitments=(1, 5))

    assert combined['scene_count'] == 2
    assert combined['edge_count'] == 4
    assert combined['continue_count'] == 2
    assert combined['rates']['continue_rate'] == 0.5
