"""同场景裁判模型（Critic）的排序与候选选择审计。"""

from __future__ import annotations

from collections.abc import Sequence
import statistics

import torch


def _accuracy(correct: list[bool]) -> float | None:
    if not correct:
        return None
    return sum(correct) / len(correct)


def audit_pairwise_predictions(
    *,
    scene_ids: torch.Tensor,
    candidate_names: Sequence[str],
    costs: torch.Tensor,
    baseline_predictions: torch.Tensor,
    critic_predictions: torch.Tensor,
    margin_thresholds: Sequence[float] = (0.0, 0.01, 0.05, 0.1, 0.5),
    greedy_candidate: str,
) -> dict[str, object]:
    """审计未见场景上的成对排序准确率与 top-1 regret。"""

    scene_ids = scene_ids.long().flatten()
    costs = costs.float().flatten()
    baseline_predictions = baseline_predictions.float().flatten()
    critic_predictions = critic_predictions.float().flatten()
    size = scene_ids.numel()
    if not (
        len(candidate_names) == size
        == costs.numel()
        == baseline_predictions.numel()
        == critic_predictions.numel()
    ):
        raise ValueError('candidate audit inputs must have equal length')
    if any(threshold < 0 for threshold in margin_thresholds):
        raise ValueError('margin thresholds must be non-negative')

    pairs = []
    scene_summaries = []
    for scene_id_tensor in scene_ids.unique(sorted=True):
        scene_id = int(scene_id_tensor.item())
        indices = (scene_ids == scene_id).nonzero().flatten().tolist()
        greedy_indices = [
            index for index in indices
            if candidate_names[index] == greedy_candidate
        ]
        if len(greedy_indices) != 1:
            raise ValueError(
                'each scene must contain exactly one greedy candidate',
            )
        for offset, left in enumerate(indices):
            for right in indices[offset + 1:]:
                if costs[left] == costs[right]:
                    continue
                better, worse = (
                    (left, right) if costs[left] < costs[right]
                    else (right, left)
                )
                pairs.append({
                    'scene_id': scene_id,
                    'margin': float((costs[worse] - costs[better]).item()),
                    'baseline_correct': bool(
                        baseline_predictions[better]
                        < baseline_predictions[worse]
                    ),
                    'critic_correct': bool(
                        critic_predictions[better]
                        < critic_predictions[worse]
                    ),
                })

        oracle_cost = float(costs[indices].min().item())
        baseline_index = min(
            indices,
            key=lambda index: float(baseline_predictions[index].item()),
        )
        critic_index = min(
            indices,
            key=lambda index: float(critic_predictions[index].item()),
        )
        greedy_index = greedy_indices[0]
        scene_summaries.append({
            'scene_id': scene_id,
            'oracle_cost': oracle_cost,
            'greedy_cost': float(costs[greedy_index].item()),
            'baseline_candidate': candidate_names[baseline_index],
            'baseline_cost': float(costs[baseline_index].item()),
            'baseline_regret': float(costs[baseline_index].item()) - oracle_cost,
            'critic_candidate': candidate_names[critic_index],
            'critic_cost': float(costs[critic_index].item()),
            'critic_regret': float(costs[critic_index].item()) - oracle_cost,
        })

    baseline_correct = [bool(pair['baseline_correct']) for pair in pairs]
    critic_correct = [bool(pair['critic_correct']) for pair in pairs]
    if not pairs:
        raise ValueError('at least one distinct-cost preference pair is required')
    baseline_accuracy = _accuracy(baseline_correct)
    critic_accuracy = _accuracy(critic_correct)
    assert baseline_accuracy is not None and critic_accuracy is not None
    margin_at_least = {}
    for threshold in sorted(set(float(value) for value in margin_thresholds)):
        selected = [pair for pair in pairs if pair['margin'] >= threshold]
        margin_at_least[f'{threshold:g}'] = {
            'num_pairs': len(selected),
            'baseline_accuracy': _accuracy([
                bool(pair['baseline_correct']) for pair in selected
            ]),
            'critic_accuracy': _accuracy([
                bool(pair['critic_correct']) for pair in selected
            ]),
        }

    num_scenes = len(scene_summaries)
    critic_regrets = [float(scene['critic_regret']) for scene in scene_summaries]
    baseline_regrets = [
        float(scene['baseline_regret']) for scene in scene_summaries
    ]
    critic_vs_greedy = [
        float(scene['critic_cost']) - float(scene['greedy_cost'])
        for scene in scene_summaries
    ]
    return {
        'overall': {
            'num_pairs': len(pairs),
            'baseline_accuracy': baseline_accuracy,
            'critic_accuracy': critic_accuracy,
            'accuracy_gain': critic_accuracy - baseline_accuracy,
        },
        'margin_at_least': margin_at_least,
        'top1': {
            'num_scenes': num_scenes,
            'baseline_exact_best_scenes': sum(
                regret <= 1e-8 for regret in baseline_regrets
            ),
            'critic_exact_best_scenes': sum(
                regret <= 1e-8 for regret in critic_regrets
            ),
            'baseline_mean_regret': sum(baseline_regrets) / num_scenes,
            'critic_mean_regret': sum(critic_regrets) / num_scenes,
            'critic_vs_greedy_mean_cost_delta': (
                sum(critic_vs_greedy) / num_scenes
            ),
            'critic_vs_greedy_median_cost_delta': statistics.median(
                critic_vs_greedy,
            ),
            'critic_better_than_greedy_scenes': sum(
                delta < -1e-8 for delta in critic_vs_greedy
            ),
            'critic_equal_to_greedy_scenes': sum(
                abs(delta) <= 1e-8 for delta in critic_vs_greedy
            ),
            'critic_worse_than_greedy_scenes': sum(
                delta > 1e-8 for delta in critic_vs_greedy
            ),
        },
        'scenes': scene_summaries,
    }
