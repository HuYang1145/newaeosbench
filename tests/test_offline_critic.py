import json

import pytest
import torch

from constellation.new_transformers.offline_critic import (
    ActionConditionedCritic,
    DenseRewardTargets,
    OfflineDatasetTensors,
    TrajectoryRecord,
    aggregate_by_trajectory,
    audit_candidate_coverage,
    build_transition_tensors,
    build_dense_reward_targets,
    combine_transition_tensors,
    compute_cs_paper_from_metrics,
    evaluate_ranking,
    fit_diagnostic_critics,
    load_routed_records,
    sample_time_indices,
    split_records_by_scene,
)


def _trajectory() -> dict:
    progress = torch.tensor([
        [0, 0],
        [1, 0],
        [2, 0],
        [2, 1],
    ], dtype=torch.uint8)
    actions = torch.tensor([
        [0, -1],
        [0, 0],
        [1, -1],
        [-1, -1],
    ])
    constellation_data = torch.arange(4 * 2 * 8, dtype=torch.float32).reshape(
        4, 2, 8,
    )
    return {
        'constellation': {
            'sensor_enabled': torch.ones(4, 2, dtype=torch.long),
            'data': constellation_data,
        },
        'taskset': {'progress': progress},
        'actions': {'task_id': actions},
        # 这是离线监督标签，不能泄漏进 Critic 输入特征。
        'is_visible': torch.ones(4, 2, 2, dtype=torch.bool),
    }


def test_compute_cs_paper_uses_pc_wh_or_converts_pc() -> None:
    metrics = {
        'CR': 0.5,
        'PCR': 0.4,
        'WCR': 0.3,
        'TAT': 700.0,
        'PC': 36000.0,
    }
    expected = 1.0 / (0.6 * 0.5 + 0.2 * 0.4 + 0.2 * 0.3) + 1 + 0.1

    assert compute_cs_paper_from_metrics(metrics) == pytest.approx(expected)
    metrics['PC_Wh'] = 20.0
    assert compute_cs_paper_from_metrics(metrics) == pytest.approx(
        expected + 0.1,
    )


def test_build_transition_tensors_preserves_s_a_r_s_prime() -> None:
    tensors = build_transition_tensors(
        _trajectory(),
        task_durations=torch.tensor([2.0, 1.0]),
        episode_cost=4.25,
        time_indices=[0, 2],
    )

    assert tensors.state.shape[0] == 2
    assert tensors.action.shape == (2, 6)
    assert tensors.next_state.shape == tensors.state.shape
    assert tensors.episode_cost.tolist() == pytest.approx([4.25, 4.25])
    assert tensors.reward.tolist() == pytest.approx([0.0, -4.25])
    assert tensors.return_to_go.tolist() == pytest.approx([-4.25, -4.25])
    assert tensors.done.tolist() == [False, True]
    assert torch.isfinite(tensors.state).all()
    assert torch.isfinite(tensors.action).all()
    assert torch.isfinite(tensors.next_state).all()


def test_transition_features_do_not_use_visibility_labels() -> None:
    trajectory = _trajectory()
    expected = build_transition_tensors(
        trajectory,
        task_durations=torch.tensor([2.0, 1.0]),
        episode_cost=3.0,
        time_indices=[0, 1],
    )
    trajectory['is_visible'].zero_()
    actual = build_transition_tensors(
        trajectory,
        task_durations=torch.tensor([2.0, 1.0]),
        episode_cost=3.0,
        time_indices=[0, 1],
    )

    assert torch.equal(actual.state, expected.state)
    assert torch.equal(actual.action, expected.action)
    assert torch.equal(actual.next_state, expected.next_state)


def test_pair_context_distinguishes_satellite_task_assignment() -> None:
    aligned = _trajectory()
    swapped = _trajectory()
    aligned['actions']['task_id'][0] = torch.tensor([0, 1])
    swapped['actions']['task_id'][0] = torch.tensor([1, 0])
    context = {
        'task_static_data': torch.tensor([
            [0.0, 3.0, 2.0, 10.0, 20.0],
            [0.0, 3.0, 1.0, -10.0, -20.0],
        ]),
        'constellation_static_data': torch.arange(8.0).repeat(2, 1),
        'task_sensor_type': torch.tensor([1, 2]),
        'constellation_sensor_type': torch.tensor([1, 2]),
    }

    aligned_tensors = build_transition_tensors(
        aligned,
        task_durations=torch.tensor([2.0, 1.0]),
        episode_cost=3.0,
        time_indices=[0],
        **context,
    )
    swapped_tensors = build_transition_tensors(
        swapped,
        task_durations=torch.tensor([2.0, 1.0]),
        episode_cost=3.0,
        time_indices=[0],
        **context,
    )

    assert aligned_tensors.action.shape[1] > 6
    assert torch.equal(
        aligned_tensors.action[:, :6],
        swapped_tensors.action[:, :6],
    )
    assert not torch.equal(aligned_tensors.action, swapped_tensors.action)
    assert torch.equal(aligned_tensors.state, swapped_tensors.state)


def test_split_records_keeps_each_scene_in_one_partition() -> None:
    records = [
        TrajectoryRecord(scene_id=i, epoch=1, trajectory_path=None,
                         metrics_path=None, episode_cost=float(i))
        for i in range(20)
    ]
    records.append(
        TrajectoryRecord(scene_id=3, epoch=2, trajectory_path=None,
                         metrics_path=None, episode_cost=2.0),
    )

    train, val = split_records_by_scene(records, val_fraction=0.25, seed=7)

    train_ids = {record.scene_id for record in train}
    val_ids = {record.scene_id for record in val}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == set(range(20))
    assert 3 not in train_ids or sum(r.scene_id == 3 for r in train) == 2
    assert 3 not in val_ids or sum(r.scene_id == 3 for r in val) == 2


def test_action_conditioned_critic_depends_on_action_features() -> None:
    torch.manual_seed(0)
    model = ActionConditionedCritic(state_dim=4, action_dim=2, hidden_dim=8)
    state = torch.zeros(3, 4)
    action_a = torch.zeros(3, 2)
    action_b = torch.ones(3, 2)

    prediction_a = model(state, action_a)
    prediction_b = model(state, action_b)

    assert prediction_a.shape == (3,)
    assert not torch.equal(prediction_a, prediction_b)


def test_evaluate_ranking_reports_order_and_baseline_gain() -> None:
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    baseline = torch.tensor([1.0, 3.0, 2.0, 4.0])
    critic = target.clone()

    result = evaluate_ranking(
        target_cost=target,
        critic_cost=critic,
        baseline_cost=baseline,
    )

    assert result['critic_spearman'] == pytest.approx(1.0)
    assert result['critic_pairwise_accuracy'] == pytest.approx(1.0)
    assert result['spearman_gain'] > 0
    assert result['accepted'] is True


def test_evaluate_ranking_rejects_scene_only_tie() -> None:
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    same_prediction = torch.tensor([1.0, 3.0, 2.0, 4.0])

    result = evaluate_ranking(
        target_cost=target,
        critic_cost=same_prediction,
        baseline_cost=same_prediction,
    )

    assert result['spearman_gain'] == pytest.approx(0.0)
    assert result['accepted'] is False


def test_load_routed_records_uses_annotation_epoch(tmp_path) -> None:
    annotation_path = tmp_path / 'train.json'
    annotation_path.write_text(
        json.dumps({'ids': [1], 'epochs': [2]}),
        encoding='utf-8',
    )
    trajectory_root = tmp_path / 'trajectories.2' / 'train' / '00'
    trajectory_root.mkdir(parents=True)
    (trajectory_root / '00001.pth').touch()
    metrics = {
        'CR': 0.5,
        'PCR': 0.4,
        'WCR': 0.3,
        'TAT': 700.0,
        'PC': 36000.0,
    }
    (trajectory_root / '00001.json').write_text(
        json.dumps(metrics),
        encoding='utf-8',
    )

    records = load_routed_records(
        annotation_path=annotation_path,
        data_root=tmp_path,
        split='train',
    )

    assert len(records) == 1
    assert records[0].scene_id == 1
    assert records[0].epoch == 2
    assert records[0].trajectory_path == trajectory_root / '00001.pth'
    assert records[0].episode_cost == pytest.approx(
        compute_cs_paper_from_metrics(metrics),
    )


def test_audit_candidate_coverage_detects_repeated_scenes(tmp_path) -> None:
    for epoch, ids in {1: [1, 2], 2: [2, 3]}.items():
        root = tmp_path / f'trajectories.{epoch}' / 'train' / '00'
        root.mkdir(parents=True)
        for scene_id in ids:
            (root / f'{scene_id:05}.json').touch()

    audit = audit_candidate_coverage(data_root=tmp_path, split='train')

    assert audit['epoch_counts'] == {'1': 2, '2': 2}
    assert audit['unique_scene_count'] == 3
    assert audit['repeated_scene_count'] == 1
    assert audit['max_candidates_per_scene'] == 2


def test_sample_time_indices_is_deterministic_and_includes_terminal() -> None:
    assert sample_time_indices(num_time_steps=10, num_samples=4) == [0, 3, 5, 8]
    assert sample_time_indices(num_time_steps=3, num_samples=8) == [0, 1]


def test_aggregate_by_trajectory_averages_predictions() -> None:
    trajectory_ids = torch.tensor([10, 10, 20, 20])
    target = torch.tensor([1.0, 1.0, 3.0, 3.0])
    prediction = torch.tensor([0.0, 2.0, 2.0, 6.0])

    ids, target_mean, prediction_mean = aggregate_by_trajectory(
        trajectory_ids=trajectory_ids,
        target_cost=target,
        predicted_cost=prediction,
    )

    assert ids.tolist() == [10, 20]
    assert target_mean.tolist() == pytest.approx([1.0, 3.0])
    assert prediction_mean.tolist() == pytest.approx([1.0, 4.0])


def test_combine_transition_tensors_keeps_trajectory_ids() -> None:
    first = build_transition_tensors(
        _trajectory(),
        task_durations=torch.tensor([2.0, 1.0]),
        episode_cost=2.0,
        time_indices=[0, 2],
    )
    second = build_transition_tensors(
        _trajectory(),
        task_durations=torch.tensor([2.0, 1.0]),
        episode_cost=4.0,
        time_indices=[1],
    )

    combined = combine_transition_tensors([(7, first), (9, second)])

    assert isinstance(combined, OfflineDatasetTensors)
    assert combined.trajectory_ids.tolist() == [7, 7, 9]
    assert combined.state.shape[0] == 3
    assert combined.episode_cost.tolist() == pytest.approx([2.0, 2.0, 4.0])


def test_fit_diagnostic_critics_detects_action_signal() -> None:
    generator = torch.Generator().manual_seed(3)
    action = torch.rand(100, 1, generator=generator)
    state = torch.zeros(100, 2)
    cost = 2.0 + 3.0 * action[:, 0]
    tensors = OfflineDatasetTensors(
        trajectory_ids=torch.arange(100),
        state=state,
        action=action,
        reward=-cost,
        next_state=state,
        done=torch.ones(100, dtype=torch.bool),
        episode_cost=cost,
        return_to_go=-cost,
    )
    train = OfflineDatasetTensors(*(tensor[:80] for tensor in tensors))
    val = OfflineDatasetTensors(*(tensor[80:] for tensor in tensors))

    _, summary = fit_diagnostic_critics(
        train,
        val,
        hidden_dim=16,
        epochs=120,
        batch_size=32,
        learning_rate=3e-3,
        seed=5,
        device=torch.device('cpu'),
    )

    assert summary['all']['critic_spearman'] > 0.9
    assert summary['all']['spearman_gain'] > 0.5
    assert summary['accepted'] is True


def test_dense_reward_components_sum_to_negative_cs_paper() -> None:
    trajectory = {
        'constellation': {
            'sensor_enabled': torch.ones(4, 2, dtype=torch.long),
            'data': torch.zeros(4, 2, 8),
        },
        'taskset': {
            'progress': torch.tensor([
                [0, 0],
                [1, 0],
                [2, 0],
                [0, 1],
            ], dtype=torch.uint8),
        },
        'actions': {
            'task_id': torch.tensor([
                [0, -1],
                [0, -1],
                [-1, 1],
                [-1, -1],
            ]),
        },
        'is_visible': torch.zeros(4, 2, 2, dtype=torch.bool),
    }
    tat_s = 2.5
    pc_term = (36.0 + 36.0 + 72.0) / 360000.0
    episode_cost = 1.0 + tat_s / 700.0 + pc_term

    targets = build_dense_reward_targets(
        trajectory,
        task_durations=torch.tensor([2.0, 1.0]),
        task_release_times=torch.tensor([0.0, 0.0]),
        satellite_sensor_power=torch.tensor([36.0, 72.0]),
        episode_cost=episode_cost,
    )

    assert isinstance(targets, DenseRewardTargets)
    assert targets.reward.shape == (3,)
    assert targets.quality_delta.sum().item() == pytest.approx(1.0)
    assert targets.tat_cost.sum().item() == pytest.approx(tat_s / 700.0)
    assert targets.power_cost.sum().item() == pytest.approx(pc_term)
    assert targets.reward.sum().item() == pytest.approx(-episode_cost)
    assert targets.return_to_go[0].item() == pytest.approx(-episode_cost)
    assert targets.return_to_go[-1] == targets.reward[-1]
    assert targets.terminal_correction.item() == pytest.approx(-2.0)


def test_dense_reward_does_not_use_visibility_as_input() -> None:
    trajectory = _trajectory()
    kwargs = dict(
        task_durations=torch.tensor([2.0, 1.0]),
        task_release_times=torch.tensor([0.0, 0.0]),
        satellite_sensor_power=torch.tensor([10.0, 20.0]),
        episode_cost=3.0,
    )
    expected = build_dense_reward_targets(trajectory, **kwargs)
    trajectory['is_visible'].zero_()
    actual = build_dense_reward_targets(trajectory, **kwargs)

    assert torch.equal(actual.reward, expected.reward)
    assert torch.equal(actual.return_to_go, expected.return_to_go)


def test_transition_tensors_select_dense_reward_and_return_to_go() -> None:
    trajectory = _trajectory()
    targets = build_dense_reward_targets(
        trajectory,
        task_durations=torch.tensor([2.0, 1.0]),
        task_release_times=torch.tensor([0.0, 0.0]),
        satellite_sensor_power=torch.tensor([10.0, 20.0]),
        episode_cost=3.0,
    )

    transitions = build_transition_tensors(
        trajectory,
        task_durations=torch.tensor([2.0, 1.0]),
        episode_cost=3.0,
        time_indices=[0, 2],
        dense_reward_targets=targets,
    )

    assert transitions.reward.tolist() == pytest.approx(
        targets.reward[[0, 2]].tolist(),
    )
    assert transitions.return_to_go.tolist() == pytest.approx(
        targets.return_to_go[[0, 2]].tolist(),
    )


def test_fit_diagnostic_critics_can_target_dense_cost_to_go() -> None:
    generator = torch.Generator().manual_seed(11)
    action = torch.rand(100, 1, generator=generator)
    state = torch.zeros(100, 2)
    dense_cost_to_go = 2.0 + 3.0 * action[:, 0]
    tensors = OfflineDatasetTensors(
        trajectory_ids=torch.arange(100),
        state=state,
        action=action,
        reward=-dense_cost_to_go,
        next_state=state,
        done=torch.ones(100, dtype=torch.bool),
        episode_cost=torch.full((100,), 9.0),
        return_to_go=-dense_cost_to_go,
    )
    train = OfflineDatasetTensors(*(tensor[:80] for tensor in tensors))
    val = OfflineDatasetTensors(*(tensor[80:] for tensor in tensors))

    _, summary = fit_diagnostic_critics(
        train,
        val,
        hidden_dim=16,
        epochs=120,
        batch_size=32,
        learning_rate=3e-3,
        seed=5,
        device=torch.device('cpu'),
        target_mode='dense_cost_to_go',
    )

    assert summary['target_mode'] == 'dense_cost_to_go'
    assert summary['all']['critic_spearman'] > 0.9
    assert summary['all']['spearman_gain'] > 0.5
