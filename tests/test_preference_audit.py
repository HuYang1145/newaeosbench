import pytest
import torch

from constellation.new_transformers.preference_audit import (
    audit_pairwise_predictions,
)


def test_audit_separates_margin_accuracy_and_top1_regret() -> None:
    summary = audit_pairwise_predictions(
        scene_ids=torch.tensor([0, 0, 0, 1, 1, 1]),
        candidate_names=['greedy', 'a', 'b', 'greedy', 'a', 'b'],
        costs=torch.tensor([3.0, 1.0, 2.0, 1.0, 1.01, 4.0]),
        baseline_predictions=torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]),
        critic_predictions=torch.tensor([3.0, 1.0, 2.0, 2.0, 1.0, 3.0]),
        margin_thresholds=(0.0, 0.05),
        greedy_candidate='greedy',
    )

    assert summary['overall']['num_pairs'] == 6
    assert summary['overall']['baseline_accuracy'] == pytest.approx(4 / 6)
    assert summary['overall']['critic_accuracy'] == pytest.approx(5 / 6)
    assert summary['overall']['accuracy_gain'] == pytest.approx(1 / 6)
    assert summary['margin_at_least']['0.05']['num_pairs'] == 5
    assert summary['margin_at_least']['0.05']['critic_accuracy'] == 1.0
    assert summary['top1']['num_scenes'] == 2
    assert summary['top1']['critic_exact_best_scenes'] == 1
    assert summary['top1']['critic_mean_regret'] == pytest.approx(0.005)
    assert summary['top1']['baseline_mean_regret'] == pytest.approx(1.0)
    assert summary['top1']['critic_vs_greedy_mean_cost_delta'] == pytest.approx(
        -0.995,
    )
    assert summary['top1']['critic_better_than_greedy_scenes'] == 1
    assert summary['top1']['critic_equal_to_greedy_scenes'] == 0
    assert summary['top1']['critic_worse_than_greedy_scenes'] == 1
    assert summary['top1']['critic_vs_greedy_median_cost_delta'] == pytest.approx(
        -0.995,
    )


def test_audit_rejects_candidates_without_one_greedy_per_scene() -> None:
    with pytest.raises(ValueError, match='exactly one greedy candidate'):
        audit_pairwise_predictions(
            scene_ids=torch.tensor([0, 0]),
            candidate_names=['a', 'b'],
            costs=torch.tensor([1.0, 2.0]),
            baseline_predictions=torch.tensor([1.0, 2.0]),
            critic_predictions=torch.tensor([1.0, 2.0]),
            greedy_candidate='greedy',
        )


def test_audit_uses_null_for_empty_margin_group() -> None:
    summary = audit_pairwise_predictions(
        scene_ids=torch.tensor([0, 0]),
        candidate_names=['greedy', 'sample'],
        costs=torch.tensor([1.0, 2.0]),
        baseline_predictions=torch.tensor([1.0, 2.0]),
        critic_predictions=torch.tensor([1.0, 2.0]),
        margin_thresholds=(10.0,),
        greedy_candidate='greedy',
    )

    assert summary['margin_at_least']['10']['num_pairs'] == 0
    assert summary['margin_at_least']['10']['baseline_accuracy'] is None
    assert summary['margin_at_least']['10']['critic_accuracy'] is None
