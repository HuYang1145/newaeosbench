import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).parents[1]
TOOL = ROOT / 'tools' / 'evaluate_event_heads_m2.py'


def _module() -> dict:
    assert TOOL.is_file(), 'M2 event-head evaluator is missing'
    return runpy.run_path(str(TOOL))


def test_binary_metrics_report_both_classes_and_balanced_accuracy() -> None:
    metrics = _module()['binary_classification_metrics'](
        logits=torch.tensor([2.0, -2.0, 1.0, -1.0]),
        targets=torch.tensor([True, True, False, False]),
    )

    assert metrics['support'] == 4
    assert metrics['confusion'] == {
        'true_negative': 1,
        'false_positive': 1,
        'false_negative': 1,
        'true_positive': 1,
    }
    assert metrics['accuracy'] == pytest.approx(0.5)
    assert metrics['balanced_accuracy'] == pytest.approx(0.5)
    assert metrics['positive_recall'] == pytest.approx(0.5)
    assert metrics['negative_recall'] == pytest.approx(0.5)
    assert metrics['predicted_positive_rate'] == pytest.approx(0.5)


def test_binary_metrics_respect_observed_mask() -> None:
    metrics = _module()['binary_classification_metrics'](
        logits=torch.tensor([2.0, -2.0, 2.0]),
        targets=torch.tensor([True, False, False]),
        observed=torch.tensor([True, True, False]),
    )

    assert metrics['support'] == 2
    assert metrics['accuracy'] == pytest.approx(1.0)


def test_binary_balanced_accuracy_is_unknown_when_one_class_is_absent() -> None:
    metrics = _module()['binary_classification_metrics'](
        logits=torch.tensor([2.0, -2.0]),
        targets=torch.tensor([True, True]),
    )

    assert metrics['negative_count'] == 0
    assert metrics['negative_recall'] is None
    assert metrics['balanced_accuracy'] is None


def test_multiclass_metrics_expose_majority_collapse() -> None:
    metrics = _module()['multiclass_classification_metrics'](
        logits=torch.tensor([
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]),
        targets=torch.tensor([0, 0, 1, 2]),
        class_names=('1', '5', '15'),
    )

    assert metrics['support'] == 4
    assert metrics['accuracy'] == pytest.approx(0.5)
    assert metrics['balanced_accuracy'] == pytest.approx(1 / 3)
    assert metrics['target_counts'] == {'1': 2, '5': 1, '15': 1}
    assert metrics['predicted_counts'] == {'1': 4, '5': 0, '15': 0}
    assert metrics['recall'] == {'1': 1.0, '5': 0.0, '15': 0.0}
    assert metrics['confusion'] == [
        [2, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
    ]


def test_metrics_reject_empty_observed_scope() -> None:
    module = _module()

    with pytest.raises(ValueError, match='empty'):
        module['binary_classification_metrics'](
            logits=torch.tensor([0.0]),
            targets=torch.tensor([True]),
            observed=torch.tensor([False]),
        )


def test_collect_predictions_gathers_only_executed_observed_edges() -> None:
    collect = _module()['collect_temporal_predictions']
    output = SimpleNamespace(
        continue_logits=torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
        duration_logits=torch.tensor([[
            [[10.0, 11.0], [20.0, 21.0]],
            [[30.0, 31.0], [40.0, 41.0]],
        ]]),
        visible_next_logits=torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]),
        progress_next_logits=torch.tensor([[[9.0, 10.0], [11.0, 12.0]]]),
        completed_next_logits=torch.tensor([[[13.0, 14.0], [15.0, 16.0]]]),
        visible_logits=torch.tensor([[
            [[50.0], [60.0]],
            [[70.0], [80.0]],
        ]]),
        progress_logits=torch.tensor([[
            [[90.0], [100.0]],
            [[110.0], [120.0]],
        ]]),
        completed_logits=torch.tensor([[
            [[130.0], [140.0]],
            [[150.0], [160.0]],
        ]]),
    )
    temporal = SimpleNamespace(
        outcome_valid=torch.tensor([[True, True]]),
        event_continue=torch.tensor([[True, False]]),
        event_duration_index=torch.tensor([[1, 0]]),
        event_duration_observed=torch.tensor([[True, True]]),
        visible_next=torch.tensor([[True, False]]),
        progress_next=torch.tensor([[False, True]]),
        completed_next=torch.tensor([[True, False]]),
        visible=torch.tensor([[[False], [True]]]),
        visible_observed=torch.tensor([[[True], [True]]]),
        progress=torch.tensor([[[True], [False]]]),
        progress_observed=torch.tensor([[[False], [True]]]),
        completed=torch.tensor([[[False], [True]]]),
        completion_observed=torch.tensor([[[True], [True]]]),
        horizons=torch.tensor([5]),
    )

    predictions = collect(
        output=output,
        temporal=temporal,
        actions_task_id=torch.tensor([[1, -1]]),
    )

    assert predictions['continue']['logits'].tolist() == [2.0]
    assert predictions['continue']['targets'].tolist() == [True]
    assert predictions['duration']['logits'].tolist() == [[20.0, 21.0]]
    assert predictions['duration']['targets'].tolist() == [1]
    assert predictions['outcomes']['visible']['next']['logits'].tolist() == [
        6.0,
    ]
    assert predictions['outcomes']['visible']['5']['logits'].tolist() == [
        60.0,
    ]
    assert predictions['outcomes']['progress']['5']['logits'].numel() == 0
