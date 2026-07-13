import sys

from gymnasium import spaces
import pytest
import torch
from torch import nn

from constellation.new_transformers import feasibility
from constellation.new_transformers.model import Transformer
from constellation.rl import eval_all
from constellation.rl import policy as policy_module


def test_feasibility_gate_exists() -> None:
    gate = getattr(feasibility, 'apply_feasibility_threshold', None)
    assert callable(gate)


def test_threshold_none_preserves_logits() -> None:
    logits = torch.tensor([[1.0, 2.0]])
    feasibility_logits = torch.tensor([[-1.0, 1.0]])

    output = feasibility.apply_feasibility_threshold(
        logits,
        feasibility_logits,
        None,
    )

    assert output is logits


def test_gate_masks_probabilities_at_or_below_threshold() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    feasibility_logits = torch.tensor([
        [-2.1972246, 0.0, 2.1972246],
    ])

    output = feasibility.apply_feasibility_threshold(
        logits,
        feasibility_logits,
        0.5,
    )

    assert torch.isneginf(output[0, 0])
    assert torch.isneginf(output[0, 1])
    assert output[0, 2].item() == 3.0


@pytest.mark.parametrize('threshold', [-0.01, 1.01])
def test_gate_rejects_threshold_outside_probability_range(
    threshold: float,
) -> None:
    with pytest.raises(ValueError, match=r'\[0, 1\]'):
        feasibility.apply_feasibility_threshold(
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            threshold,
        )


def test_gate_requires_feasibility_logits_when_enabled() -> None:
    with pytest.raises(ValueError, match='feasibility_logits'):
        feasibility.apply_feasibility_threshold(
            torch.zeros(1, 1),
            None,
            0.5,
        )


def test_all_masked_tasks_fall_back_to_null_action() -> None:
    task_logits = feasibility.apply_feasibility_threshold(
        torch.tensor([[4.0, 3.0]]),
        torch.tensor([[-10.0, -10.0]]),
        0.5,
    )
    all_logits = torch.cat([torch.tensor([[0.0]]), task_logits], -1)

    assert all_logits.argmax(-1).item() == 0


def test_bounded_soft_penalty_only_reduces_low_probability_logits() -> None:
    probabilities = torch.tensor([[0.003, 0.015, 0.03, 0.06]])
    feasibility_logits = torch.logit(probabilities)

    output = feasibility.apply_feasibility_penalty(
        torch.zeros_like(feasibility_logits),
        feasibility_logits,
        threshold=0.03,
        strength=2.0,
    )

    assert torch.isfinite(output).all()
    assert output == pytest.approx(
        torch.tensor([[-1.8, -1.0, 0.0, 0.0]]),
        abs=1e-6,
    )


def test_soft_penalty_disabled_preserves_logits() -> None:
    logits = torch.tensor([[1.0, 2.0]])

    output = feasibility.apply_feasibility_penalty(
        logits,
        torch.zeros_like(logits),
        threshold=None,
        strength=None,
    )

    assert output is logits


@pytest.mark.parametrize(
    ('threshold', 'strength'),
    [(0.03, None), (None, 0.5), (0.0, 0.5), (1.01, 0.5), (0.03, -0.1)],
)
def test_soft_penalty_rejects_invalid_configuration(
    threshold: float | None,
    strength: float | None,
) -> None:
    with pytest.raises(ValueError):
        feasibility.apply_feasibility_penalty(
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            threshold=threshold,
            strength=strength,
        )


def _build_transformer(**kwargs) -> Transformer:
    return Transformer(
        sensor_type_embedding_dim=2,
        tasks_data_embedding_dim=2,
        encoder_width=4,
        encoder_depth=1,
        encoder_num_heads=1,
        sensor_enabled_embedding_dim=2,
        constellation_data_embedding_dim=2,
        decoder_width=4,
        decoder_depth=1,
        decoder_num_heads=1,
        **kwargs,
    )


def test_transformer_stores_configured_feasibility_threshold() -> None:
    transformer = _build_transformer(feasibility_threshold=0.25)

    assert transformer._feasibility_threshold == 0.25


def test_transformer_stores_configured_feasibility_penalty() -> None:
    transformer = _build_transformer(
        feasibility_penalty_threshold=0.03,
        feasibility_penalty_strength=0.5,
    )

    assert transformer._feasibility_penalty_threshold == 0.03
    assert transformer._feasibility_penalty_strength == 0.5


def test_transformer_rejects_simultaneous_hard_and_soft_constraints() -> None:
    with pytest.raises(ValueError, match='simultaneously'):
        _build_transformer(
            feasibility_threshold=0.03,
            feasibility_penalty_threshold=0.03,
            feasibility_penalty_strength=0.5,
        )


def test_transformer_rejects_threshold_without_constraint_module() -> None:
    with pytest.raises(ValueError, match='constraint module'):
        _build_transformer(
            use_constraint_module=False,
            feasibility_threshold=0.25,
        )


@pytest.mark.parametrize('threshold', [-0.01, 1.01])
def test_transformer_rejects_threshold_outside_probability_range(
    threshold: float,
) -> None:
    with pytest.raises(ValueError, match=r'\[0, 1\]'):
        _build_transformer(feasibility_threshold=threshold)


def test_eval_policy_kwargs_include_feasibility_constraints() -> None:
    build_policy_kwargs = getattr(eval_all, 'build_policy_kwargs', None)
    assert callable(build_policy_kwargs)

    kwargs = build_policy_kwargs(
        ['model.pth'],
        0.25,
        feasibility_penalty_threshold=None,
        feasibility_penalty_strength=None,
    )

    assert kwargs == {
        'load_model_from': ['model.pth'],
        'actor_model_kwargs': {
            'use_constraint_module': True,
            'use_sdpa': True,
            'feasibility_threshold': 0.25,
            'feasibility_penalty_threshold': None,
            'feasibility_penalty_strength': None,
        },
    }


def test_eval_policy_kwargs_can_enable_assignment_head() -> None:
    kwargs = eval_all.build_policy_kwargs(
        ['assignment.pth'],
        None,
        feasibility_penalty_threshold=None,
        feasibility_penalty_strength=None,
        use_assignment_head=True,
        assignment_head_hidden_width=16,
    )

    assert kwargs['actor_model_kwargs']['use_assignment_head'] is True
    assert kwargs['actor_model_kwargs']['assignment_head_hidden_width'] == 16


def test_policy_uses_actor_model_kwargs_during_actor_construction(
    monkeypatch,
) -> None:
    class TinyActorCritic(nn.Module):

        def __init__(
            self,
            *args,
            actor_model_kwargs=None,
            **kwargs,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.latent_dim_pi = 4
            self.latent_dim_vf = 4
            self.actor_model_kwargs = actor_model_kwargs

    monkeypatch.setattr(policy_module, 'ActorCritic', TinyActorCritic)
    policy = policy_module.Policy(
        observation_space=spaces.Dict({
            'dummy': spaces.Box(-1.0, 1.0, shape=(1,)),
        }),
        action_space=spaces.MultiDiscrete([2]),
        lr_schedule=lambda _: 1e-3,
        load_model_from=[],
        actor_model_kwargs={'feasibility_threshold': 0.25},
    )

    assert policy.mlp_extractor.actor_model_kwargs == {
        'feasibility_threshold': 0.25,
    }


def test_eval_metadata_records_feasibility_threshold() -> None:
    metadata = eval_all.build_eval_metadata(
        split='val_seen',
        world_size=96,
        max_scenes=8,
        load_model_from=['model.pth'],
        feasibility_threshold=0.25,
        feasibility_penalty_threshold=None,
        feasibility_penalty_strength=None,
        coordination_diagnostics_top_k=5,
        use_assignment_head=False,
        assignment_head_hidden_width=32,
        owner_assignment=True,
        owner_continuation_bonus=0.25,
    )

    assert metadata == {
        'split': 'val_seen',
        'world_size': 96,
        'max_scenes': 8,
        'load_model_from': ['model.pth'],
        'feasibility_threshold': 0.25,
        'feasibility_penalty_threshold': None,
        'feasibility_penalty_strength': None,
        'coordination_diagnostics_top_k': 5,
        'use_assignment_head': False,
        'assignment_head_hidden_width': 32,
        'owner_assignment': True,
        'owner_continuation_bonus': 0.25,
    }


def test_eval_cli_parses_feasibility_threshold(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'eval_all',
            'run_name',
            'constellation/rl/config_eval.py',
            '--feasibility-threshold',
            '0.25',
            '--feasibility-penalty-threshold',
            '0.03',
            '--feasibility-penalty-strength',
            '0.5',
            '--coordination-diagnostics-top-k',
            '5',
            '--max-scenes',
            '8',
            '--use-assignment-head',
            '--assignment-head-hidden-width',
            '16',
            '--owner-assignment',
            '--owner-continuation-bonus',
            '0.25',
        ],
    )

    args = eval_all.parse_args()

    assert args.feasibility_threshold == 0.25
    assert args.feasibility_penalty_threshold == 0.03
    assert args.feasibility_penalty_strength == 0.5
    assert args.coordination_diagnostics_top_k == 5
    assert args.max_scenes == 8
    assert args.use_assignment_head is True
    assert args.assignment_head_hidden_width == 16
    assert args.owner_assignment is True
    assert args.owner_continuation_bonus == 0.25


def test_limit_annotations_keeps_requested_prefix() -> None:
    assert eval_all.limit_annotations([10, 20, 30], 2) == [10, 20]


def test_limit_annotations_none_preserves_all_annotations() -> None:
    annotations = [10, 20, 30]

    assert eval_all.limit_annotations(annotations, None) is annotations


@pytest.mark.parametrize('max_scenes', [0, -1])
def test_limit_annotations_rejects_non_positive_limit(
    max_scenes: int,
) -> None:
    with pytest.raises(ValueError, match='positive'):
        eval_all.limit_annotations([10, 20, 30], max_scenes)


def test_binary_calibration_metrics_match_hand_calculation() -> None:
    calculate = getattr(feasibility, 'binary_calibration_metrics', None)
    assert callable(calculate)

    metrics = calculate(
        torch.tensor([0.9, 0.8, 0.4, 0.1]),
        torch.tensor([True, False, True, False]),
        threshold=0.5,
        num_bins=2,
    )

    assert metrics['support'] == 4
    assert metrics['positive_support'] == 2
    assert metrics['negative_support'] == 2
    assert metrics['tp'] == 1
    assert metrics['fp'] == 1
    assert metrics['fn'] == 1
    assert metrics['tn'] == 1
    assert metrics['precision'] == pytest.approx(0.5)
    assert metrics['recall'] == pytest.approx(0.5)
    assert metrics['fpr'] == pytest.approx(0.5)
    assert metrics['fnr'] == pytest.approx(0.5)
    assert metrics['f1'] == pytest.approx(0.5)
    assert metrics['brier_score'] == pytest.approx(0.255)
    assert metrics['ece'] == pytest.approx(0.3)
    assert [item['count'] for item in metrics['calibration_bins']] == [2, 2]


def test_binary_calibration_metrics_reject_empty_input() -> None:
    calculate = getattr(feasibility, 'binary_calibration_metrics', None)
    assert callable(calculate)

    with pytest.raises(ValueError, match='empty'):
        calculate(
            torch.tensor([]),
            torch.tensor([], dtype=torch.bool),
            threshold=0.5,
        )


def test_hard_negative_indices_select_false_high_confidence_pairs() -> None:
    select = getattr(feasibility, 'hard_negative_indices', None)
    assert callable(select)

    indices = select(
        torch.tensor([0.9, 0.8, 0.4, 0.1]),
        torch.tensor([True, False, False, False]),
        threshold=0.5,
    )

    assert indices.tolist() == [1]
