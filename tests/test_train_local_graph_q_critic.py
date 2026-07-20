import json
from pathlib import Path

import pytest

from tools.train_local_graph_q_critic import (
    audit_horizon_consistency,
    load_local_samples,
    summarize_cross_validation,
)

from test_local_graph_q_critic import _summary_payload


def test_load_local_samples_combines_summaries_and_filters_margin(
    tmp_path: Path,
) -> None:
    first = _summary_payload()
    second = _summary_payload()
    second['scene_id'] = 8
    second['records'][0]['preference_pairs'][0]['cost_margin'] = 0.01
    paths = []
    for index, payload in enumerate((first, second)):
        path = tmp_path / f'{index}.json'
        path.write_text(json.dumps(payload), encoding='utf-8')
        paths.append(path)

    samples, metadata = load_local_samples(paths, min_cost_margin=0.05)

    assert len(samples) == 1
    assert samples[0].scene_id == 7
    assert metadata['num_source_summaries'] == 2
    assert metadata['num_filtered_small_margin'] == 1


def test_audit_horizon_consistency_counts_preference_agreement() -> None:
    payload = _summary_payload()
    branches = payload['records'][0]['branches']
    branches['stay']['horizons']['600'] = {
        'prefix_metrics': {
            'prefix_cost': 3.0
        },
    }
    branches['actor_rank_0']['horizons']['600'] = {
        'prefix_metrics': {
            'prefix_cost': 2.0
        },
    }

    audit = audit_horizon_consistency([payload],
                                      primary_horizon=300,
                                      check_horizon=600)

    assert audit['comparable_pairs'] == 1
    assert audit['agreeing_pairs'] == 1
    assert audit['agreement'] == 1.0


def test_summarize_cross_validation_requires_three_accepted_folds() -> None:
    folds = [{
        'training': {
            'accepted': accepted,
            'baseline': {
                'pairwise_accuracy': 0.5,
                'mean_regret': 0.5
            },
            'graph_q': {
                'pairwise_accuracy': 0.7,
                'mean_regret': 0.2
            },
        }
    } for accepted in (True, True, True, False)]

    summary = summarize_cross_validation(folds)

    assert summary['accepted_folds'] == 3
    assert summary['mean_graph_q_pairwise_accuracy'] == pytest.approx(0.7)
    assert summary['mean_pairwise_accuracy_gain'] == pytest.approx(0.2)
    assert summary['accepted'] is True
