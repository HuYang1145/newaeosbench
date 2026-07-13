import importlib
import importlib.util

import torch

from constellation.new_transformers.dataset import JointBatch
from constellation.new_transformers.model import JointModel, Model


def _assignment_module():
    module_name = 'constellation.new_transformers.assignment'
    assert importlib.util.find_spec(module_name) is not None, (
        'bipartite assignment module is not implemented'
    )
    return importlib.import_module(module_name)


def _head():
    module = _assignment_module()
    return module.BipartiteAssignmentHead(
        satellite_width=4,
        task_width=6,
        hidden_width=8,
    )


def test_assignment_head_starts_as_exact_residual_noop() -> None:
    head = _head()
    null_logits = torch.tensor([[[-0.2], [0.3]]])
    task_logits = torch.tensor([[[1.0, 2.0, -3.0], [0.5, -1.0, 4.0]]])
    satellite_features = torch.randn(1, 2, 4)
    task_features = torch.randn(1, 3, 6)
    satellite_mask = torch.tensor([[True, True]])
    task_mask = torch.tensor([[True, True, False]])

    output = head(
        null_logits,
        task_logits,
        satellite_features,
        task_features,
        satellite_mask,
        task_mask,
    )

    torch.testing.assert_close(output, task_logits, rtol=0, atol=0)


def test_assignment_head_only_updates_valid_satellite_task_edges() -> None:
    head = _head()
    with torch.no_grad():
        head.residual_score.bias.fill_(0.5)

    null_logits = torch.zeros(1, 2, 1)
    task_logits = torch.zeros(1, 2, 3)
    output = head(
        null_logits,
        task_logits,
        torch.randn(1, 2, 4),
        torch.randn(1, 3, 6),
        torch.tensor([[True, False]]),
        torch.tensor([[True, False, True]]),
    )

    expected = torch.tensor([[[0.5, 0.0, 0.5], [0.0, 0.0, 0.0]]])
    torch.testing.assert_close(output, expected)


def test_assignment_head_receives_gradient_from_assignment_loss() -> None:
    head = _head()
    output = head(
        torch.zeros(1, 2, 1),
        torch.zeros(1, 2, 3),
        torch.randn(1, 2, 4),
        torch.randn(1, 3, 6),
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, 3, dtype=torch.bool),
    )

    output.sum().backward()

    assert head.residual_score.weight.grad is not None
    assert head.residual_score.weight.grad.abs().sum() > 0


def test_collision_loss_is_higher_for_competing_satellites() -> None:
    loss_fn = _assignment_module().AssignmentAuxiliaryLoss()
    competing = torch.tensor([
        [[-5.0, 5.0, -5.0], [-5.0, 5.0, -5.0]],
    ])
    distributed = torch.tensor([
        [[-5.0, 5.0, -5.0], [-5.0, -5.0, 5.0]],
    ])
    satellite_mask = torch.ones(1, 2, dtype=torch.bool)
    task_mask = torch.ones(1, 2, dtype=torch.bool)
    targets = torch.tensor([[0, 1]])

    competing_loss = loss_fn(
        competing,
        targets,
        satellite_mask,
        task_mask,
    ).collision
    distributed_loss = loss_fn(
        distributed,
        targets,
        satellite_mask,
        task_mask,
    ).collision

    assert competing_loss > distributed_loss


def test_coverage_loss_rewards_covering_unique_expert_tasks() -> None:
    loss_fn = _assignment_module().AssignmentAuxiliaryLoss()
    covered = torch.tensor([
        [[-5.0, 5.0, -5.0], [-5.0, -5.0, 5.0]],
    ])
    missed = torch.tensor([
        [[5.0, -5.0, -5.0], [5.0, -5.0, -5.0]],
    ])
    satellite_mask = torch.ones(1, 2, dtype=torch.bool)
    task_mask = torch.ones(1, 2, dtype=torch.bool)
    targets = torch.tensor([[0, 1]])

    covered_loss = loss_fn(
        covered,
        targets,
        satellite_mask,
        task_mask,
    ).coverage
    missed_loss = loss_fn(
        missed,
        targets,
        satellite_mask,
        task_mask,
    ).coverage

    assert covered_loss < missed_loss


def test_auxiliary_losses_ignore_padding() -> None:
    loss_fn = _assignment_module().AssignmentAuxiliaryLoss()
    logits = torch.tensor([
        [
            [-5.0, 5.0, -5.0],
            [-5.0, -5.0, 5.0],
            [-5.0, 100.0, 100.0],
        ],
    ])
    targets = torch.tensor([[0, 1, 0]])
    satellite_mask = torch.tensor([[True, True, False]])
    task_mask = torch.tensor([[True, True]])

    padded = loss_fn(logits, targets, satellite_mask, task_mask)
    unpadded = loss_fn(
        logits[:, :2],
        targets[:, :2],
        satellite_mask[:, :2],
        task_mask,
    )

    torch.testing.assert_close(padded.collision, unpadded.collision)
    torch.testing.assert_close(padded.coverage, unpadded.coverage)


def _tiny_model_kwargs() -> dict[str, object]:
    return dict(
        sensor_type_embedding_dim=4,
        tasks_data_embedding_dim=4,
        encoder_width=8,
        encoder_depth=1,
        encoder_num_heads=2,
        sensor_enabled_embedding_dim=4,
        constellation_data_embedding_dim=4,
        decoder_width=8,
        decoder_depth=1,
        decoder_num_heads=2,
        use_constraint_module=False,
        use_sdpa=False,
    )


def _tiny_predict_inputs() -> tuple[object, ...]:
    return (
        [0],
        torch.zeros(1, 2, dtype=torch.long),
        torch.ones(1, 2, dtype=torch.long),
        torch.randn(1, 2, 56),
        torch.ones(1, 2, dtype=torch.bool),
        torch.zeros(1, 3, dtype=torch.long),
        torch.randn(1, 3, 6),
        torch.ones(1, 3, dtype=torch.bool),
    )


def test_model_assignment_switch_preserves_baseline_logits_at_init() -> None:
    baseline = Model(**_tiny_model_kwargs()).eval()
    assignment = Model(
        **_tiny_model_kwargs(),
        use_assignment_head=True,
        assignment_head_hidden_width=4,
    ).eval()
    incompatible = assignment.load_state_dict(
        baseline.state_dict(),
        strict=False,
    )
    inputs = _tiny_predict_inputs()

    with torch.no_grad():
        baseline_logits = baseline.predict(*inputs)
        assignment_logits = assignment.predict(*inputs)

    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all('_assignment_head.' in key for key in incompatible.missing_keys)
    torch.testing.assert_close(
        assignment_logits,
        baseline_logits,
        rtol=0,
        atol=0,
    )


def test_freeze_assignment_backbone_only_leaves_head_trainable() -> None:
    model = JointModel(
        **_tiny_model_kwargs(),
        use_assignment_head=True,
        freeze_assignment_backbone=True,
    )
    trainable = [name for name, parameter in model.named_parameters()
                 if parameter.requires_grad]

    assert trainable
    assert all('_assignment_head.' in name for name in trainable)


def test_joint_model_reports_weighted_assignment_auxiliary_losses() -> None:
    model = JointModel(
        **_tiny_model_kwargs(),
        use_assignment_head=True,
        feasibility_loss_weight=0.,
        time_loss_weight=0.,
        assignment_loss_weight=1.,
        collision_loss_weight=0.2,
        coverage_loss_weight=0.3,
    ).eval()
    inputs = _tiny_predict_inputs()
    batch = JointBatch(
        id_=0,
        annotation_id=0,
        time_steps=[0],
        constellation_sensor_type=inputs[1],
        constellation_sensor_enabled=inputs[2],
        constellation_data=inputs[3],
        constellation_mask=inputs[4],
        tasks_sensor_type=inputs[5],
        tasks_data=inputs[6],
        tasks_mask=inputs[7],
        actions_task_id=torch.tensor([[0, 1]]),
        constraint_time_steps=torch.tensor([0, 0]),
        constraint_constellation_data=torch.randn(2, 56),
        constraint_tasks_data=torch.randn(2, 6),
        constraint_durations=torch.tensor([-50., -50.]),
    )

    memo = model(type('Runner', (), {'iter_': 0})(), batch, {})

    collision = memo['assignment_collision_loss']
    coverage = memo['assignment_coverage_loss']
    assignment = memo['assignment_loss']
    expected = assignment + 0.2 * collision + 0.3 * coverage
    torch.testing.assert_close(memo['loss'], expected)
