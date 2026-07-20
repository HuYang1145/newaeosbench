import torch

from constellation.new_transformers.local_graph_q_critic import (
    LOCAL_OUTCOME_KEYS,
    LocalGraphQCritic,
    LocalGraphQSample,
    collate_local_graph_q_samples,
    fit_local_graph_q_critics,
    rerank_candidate_actions,
    samples_from_branch_summary,
    split_samples_by_scene,
)


def _summary_payload() -> dict:
    context = {
        'previous_assignment': [31, -1],
        'ongoing_task_ids': [31, 44],
        'actor_logits': [[0.0, 3.0, 2.0], [4.0, 1.0, 0.0]],
        'satellite_features': [[1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
        'task_features': [[10.0, 20.0, 0.5], [30.0, 40.0, 0.0]],
        'satellite_sensor_type': [1, 2],
        'task_sensor_type': [1, 2],
        'uses_is_visible_as_input': False,
    }

    def branch(task_id: int, cost: float, completed: int) -> dict:
        metrics = {
            'completed_tasks': completed,
            'partial_progress_gain': float(completed),
            'pc_wh': 0.2 + completed,
            'switches': 2 + completed,
            'one_second_runs': 1,
            'redundant_satellite_seconds': completed,
            'prefix_metrics': {
                'prefix_cost': cost
            },
        }
        return {
            'decision_state_signature': 'same-state',
            'decision_context': context,
            'original_assignment': [31, -1],
            'satellite_index': 0,
            'applied_task_id': task_id,
            'horizons': {
                '300': metrics
            },
        }

    return {
        'scene_id': 7,
        'primary_horizon': 300,
        'records': [{
            'decision': {
                'decision_time': 10,
                'satellite_index': 0
            },
            'branches': {
                'stay': branch(31, 2.0, 0),
                'actor_rank_0': branch(44, 1.0, 1),
            },
            'preference_pairs': [{
                'better_branch': 'actor_rank_0',
                'worse_branch': 'stay',
                'better_task_id': 44,
                'worse_task_id': 31,
                'better_cost': 1.0,
                'worse_cost': 2.0,
                'cost_margin': 1.0,
                'primary_horizon': 300,
            }],
        }],
    }


def test_samples_from_branch_summary_preserve_identity_and_outcomes() -> None:
    samples = samples_from_branch_summary(_summary_payload())

    assert len(samples) == 1
    sample = samples[0]
    assert sample.scene_id == 7
    assert sample.decision_time == 10
    assert sample.previous_action.tolist() == [0, -1]
    assert sample.better_action.tolist() == [1, -1]
    assert sample.worse_action.tolist() == [0, -1]
    assert sample.compatibility.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert sample.actor_logits.shape == (2, 3)
    assert sample.better_outcomes.shape == (len(LOCAL_OUTCOME_KEYS), )
    assert sample.better_outcomes[0].item() == 1
    assert sample.worse_outcomes[0].item() == 0


def test_samples_treat_unavailable_previous_task_as_not_continuable() -> None:
    payload = _summary_payload()
    context = payload['records'][0]['branches']['stay']['decision_context']
    context['previous_assignment'] = [999, -1]

    sample = samples_from_branch_summary(payload)[0]

    assert sample.previous_action.tolist() == [-1, -1]


def test_local_graph_q_critic_scores_pairs_and_predicts_outcomes() -> None:
    sample = samples_from_branch_summary(_summary_payload())[0]
    batch = collate_local_graph_q_samples([sample])
    model = LocalGraphQCritic(
        satellite_dim=sample.satellite_features.shape[-1],
        task_dim=sample.task_features.shape[-1],
        outcome_dim=len(LOCAL_OUTCOME_KEYS),
        hidden_dim=16,
    )

    better_score, worse_score, better_outcomes, worse_outcomes = model(batch)

    assert better_score.shape == (1, )
    assert worse_score.shape == (1, )
    assert better_outcomes.shape == (1, len(LOCAL_OUTCOME_KEYS))
    assert worse_outcomes.shape == (1, len(LOCAL_OUTCOME_KEYS))
    loss = (
        better_score.mean() + worse_score.mean() + better_outcomes.mean()
        + worse_outcomes.mean()
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def _synthetic_sample(scene_id: int) -> LocalGraphQSample:
    flip = scene_id % 2
    task_features = torch.tensor([
        [float(flip), 0.0],
        [float(1 - flip), 0.0],
    ])
    better_index = flip
    worse_index = 1 - flip
    better_outcomes = torch.zeros(len(LOCAL_OUTCOME_KEYS))
    worse_outcomes = torch.ones(len(LOCAL_OUTCOME_KEYS))
    return LocalGraphQSample(
        scene_id=scene_id,
        decision_time=10,
        satellite_features=torch.tensor([[1.0, 0.0]]),
        task_features=task_features,
        compatibility=torch.ones((1, 2)),
        actor_logits=torch.zeros((1, 3)),
        previous_action=torch.tensor([-1]),
        better_action=torch.tensor([better_index]),
        worse_action=torch.tensor([worse_index]),
        better_outcomes=better_outcomes,
        worse_outcomes=worse_outcomes,
        margin=1.0,
        better_branch=f'candidate_{better_index}',
        worse_branch=f'candidate_{worse_index}',
        better_cost=0.0,
        worse_cost=1.0,
    )


def test_split_samples_by_scene_keeps_scenes_isolated() -> None:
    samples = [_synthetic_sample(scene_id) for scene_id in range(8)]

    train, val, train_ids, val_ids = split_samples_by_scene(
        samples,
        num_folds=4,
        fold_index=1,
    )

    assert set(train_ids).isdisjoint(val_ids)
    assert {item.scene_id for item in train} == set(train_ids)
    assert {item.scene_id for item in val} == set(val_ids)


def test_fit_local_graph_q_critics_learns_identity_signal() -> None:
    samples = [_synthetic_sample(scene_id) for scene_id in range(48)]
    train = samples[:40]
    val = samples[40:]

    bundle, summary = fit_local_graph_q_critics(
        train,
        val,
        hidden_dim=16,
        epochs=80,
        batch_size=16,
        learning_rate=3e-3,
        outcome_loss_weight=0.2,
        margin_clip=1.0,
        seed=7,
        device=torch.device('cpu'),
    )

    assert summary['graph_q']['pairwise_accuracy'] > 0.9
    assert summary['pairwise_accuracy_gain'] > 0.4
    assert summary['graph_q']['mean_regret'] < summary['baseline'][
        'mean_regret']
    assert set(summary['outcome_mae']) == set(LOCAL_OUTCOME_KEYS)
    assert max(summary['outcome_mae'].values()) < 0.2

    batch = collate_local_graph_q_samples([val[0]])
    candidates = torch.stack(
        (batch.better_action, batch.worse_action),
        dim=1,
    )
    selected, scores = rerank_candidate_actions(
        bundle,
        batch,
        candidate_actions=candidates,
        device=torch.device('cpu'),
    )
    assert scores.shape == (1, 2)
    assert torch.equal(selected, scores.argmin(-1))
