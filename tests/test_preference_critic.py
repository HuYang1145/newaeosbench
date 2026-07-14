import torch

from constellation.new_transformers.preference_critic import (
    CandidateTensors,
    build_preference_pairs,
    fit_preference_critics,
)


def test_preference_pairs_compare_only_distinct_costs_in_same_scene() -> None:
    pairs = build_preference_pairs(
        scene_ids=torch.tensor([1, 1, 1, 2, 2]),
        costs=torch.tensor([3.0, 2.0, 2.0, 5.0, 6.0]),
    )

    assert pairs.better.tolist() == [1, 2, 3]
    assert pairs.worse.tolist() == [0, 0, 4]


def _synthetic_candidates(scene_start: int, scene_end: int) -> CandidateTensors:
    scene_ids = []
    states = []
    actions = []
    costs = []
    for scene_id in range(scene_start, scene_end):
        difficulty = scene_id / 10.0
        for action_quality in (0.0, 1.0, 2.0, 3.0):
            scene_ids.append(scene_id)
            states.append([difficulty, 1.0])
            actions.append([action_quality])
            costs.append(difficulty + action_quality)
    return CandidateTensors(
        scene_ids=torch.tensor(scene_ids),
        state=torch.tensor(states),
        action=torch.tensor(actions),
        cost=torch.tensor(costs),
    )


def test_pairwise_critic_learns_action_signal_beyond_state_baseline() -> None:
    train = _synthetic_candidates(0, 30)
    val = _synthetic_candidates(30, 40)

    _, summary = fit_preference_critics(
        train,
        val,
        hidden_dim=16,
        epochs=80,
        batch_size=64,
        learning_rate=3e-3,
        seed=13,
        device=torch.device('cpu'),
    )

    assert summary['num_train_pairs'] == 180
    assert summary['num_val_pairs'] == 60
    assert summary['critic_pairwise_accuracy'] > 0.95
    assert summary['baseline_pairwise_accuracy'] < 0.6
    assert summary['pairwise_accuracy_gain'] > 0.35
    assert summary['accepted'] is True
