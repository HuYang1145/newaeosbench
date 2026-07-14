import torch

from constellation.new_transformers.graph_q_critic import (
    GraphQCritic,
    GraphQSample,
    audit_pairwise_tournament,
    collate_graph_q_samples,
    fit_graph_q_critics,
    pairwise_accuracy,
)


def _sample(
    *,
    scene_id: int,
    satellite_features: torch.Tensor,
    task_features: torch.Tensor,
    better_action: list[int],
    worse_action: list[int],
    better_candidate: str = 'sample',
    worse_candidate: str = 'greedy',
    better_cost: float = 3.0,
    worse_cost: float = 4.0,
) -> GraphQSample:
    compatibility = torch.ones(
        satellite_features.shape[0],
        task_features.shape[0],
    )
    return GraphQSample(
        scene_id=scene_id,
        satellite_features=satellite_features,
        task_features=task_features,
        compatibility=compatibility,
        previous_action=torch.full(
            (satellite_features.shape[0], ),
            -1,
            dtype=torch.long,
        ),
        better_action=torch.tensor(better_action),
        worse_action=torch.tensor(worse_action),
        better_summary=torch.tensor([
            0.5,
            0.5,
            0.0,
            0.5,
            1.0,
            0.5,
            0.5,
        ]),
        worse_summary=torch.tensor([
            0.5,
            0.5,
            0.0,
            0.5,
            1.0,
            0.5,
            0.5,
        ]),
        margin=worse_cost - better_cost,
        better_candidate=better_candidate,
        worse_candidate=worse_candidate,
        better_cost=better_cost,
        worse_cost=worse_cost,
    )


def test_collate_graph_q_samples_pads_variable_graphs() -> None:
    samples = [
        _sample(
            scene_id=1,
            satellite_features=torch.zeros(2, 3),
            task_features=torch.zeros(3, 4),
            better_action=[0, -1],
            worse_action=[1, -1],
        ),
        _sample(
            scene_id=2,
            satellite_features=torch.zeros(1, 3),
            task_features=torch.zeros(2, 4),
            better_action=[1],
            worse_action=[0],
        ),
    ]

    batch = collate_graph_q_samples(samples)

    assert batch.satellite_features.shape == (2, 2, 3)
    assert batch.task_features.shape == (2, 3, 4)
    assert batch.compatibility.shape == (2, 2, 3)
    assert batch.satellite_mask.tolist() == [[True, True], [True, False]]
    assert batch.task_mask.tolist() == [[True, True, True],
                                        [True, True, False]]
    assert batch.previous_action.tolist() == [[-1, -1], [-1, -1]]
    assert batch.better_action.tolist() == [[0, -1], [1, -1]]


def test_graph_q_preserves_satellite_task_identity() -> None:
    torch.manual_seed(7)
    model = GraphQCritic(
        satellite_dim=2,
        task_dim=2,
        hidden_dim=8,
    ).eval()
    sample = _sample(
        scene_id=3,
        satellite_features=torch.tensor([[2.0, 0.0], [0.0, 1.0]]),
        task_features=torch.tensor([[1.0, 0.0], [0.0, 3.0]]),
        better_action=[0, 1],
        worse_action=[1, 0],
    )
    batch = collate_graph_q_samples([sample])

    with torch.inference_mode():
        better = model.score_batch(batch, batch.better_action)
        worse = model.score_batch(batch, batch.worse_action)

    assert not torch.allclose(better, worse)


def test_pairwise_accuracy_counts_ties_as_half() -> None:
    accuracy = pairwise_accuracy(
        better_scores=torch.tensor([0.0, 2.0, 1.0]),
        worse_scores=torch.tensor([1.0, 1.0, 1.0]),
    )

    assert accuracy == 0.5


def test_pairwise_tournament_reports_scene_top1_regret() -> None:
    samples = [
        _sample(
            scene_id=5,
            satellite_features=torch.zeros(1, 2),
            task_features=torch.zeros(2, 2),
            better_action=[0],
            worse_action=[1],
            better_candidate='sample',
            worse_candidate='candidate_000_greedy',
            better_cost=3.0,
            worse_cost=4.0,
        ),
    ]

    audit = audit_pairwise_tournament(
        samples,
        better_scores=torch.tensor([0.0]),
        worse_scores=torch.tensor([2.0]),
        greedy_candidate='candidate_000_greedy',
    )

    assert audit['pairwise_accuracy'] == 1.0
    assert audit['top1_exact_best_scenes'] == 1
    assert audit['mean_regret'] == 0.0
    assert audit['selected_vs_greedy_mean_cost_delta'] == -1.0


def test_pairwise_tournament_handles_large_score_gaps() -> None:
    samples = [
        _sample(
            scene_id=6,
            satellite_features=torch.zeros(1, 2),
            task_features=torch.zeros(2, 2),
            better_action=[0],
            worse_action=[1],
            better_candidate='sample',
            worse_candidate='candidate_000_greedy',
        ),
    ]

    audit = audit_pairwise_tournament(
        samples,
        better_scores=torch.tensor([-1000.0]),
        worse_scores=torch.tensor([1000.0]),
        greedy_candidate='candidate_000_greedy',
    )

    assert audit['pairwise_accuracy'] == 1.0


def test_fit_graph_q_learns_identity_beyond_summary_baseline() -> None:
    samples = [
        _sample(
            scene_id=scene_id,
            satellite_features=torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
            task_features=torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
            better_action=[0, 1],
            worse_action=[1, 0],
        ) for scene_id in range(16)
    ]

    _, summary = fit_graph_q_critics(
        samples[:12],
        samples[12:],
        hidden_dim=8,
        epochs=40,
        batch_size=4,
        learning_rate=1e-2,
        margin_clip=1.0,
        seed=7,
        device=torch.device('cpu'),
    )

    assert summary['baseline']['pairwise_accuracy'] == 0.5
    assert summary['graph_q']['pairwise_accuracy'] == 1.0
