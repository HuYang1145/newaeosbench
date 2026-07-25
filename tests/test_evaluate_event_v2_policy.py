from types import SimpleNamespace

import pytest
import torch

from constellation.new_transformers.event_v2.basilisk_runtime import (
    CompletionSnapshot,
    RuntimeStep,
)
from constellation.new_transformers.event_v2.transition import (
    JointEventAction,
)
from tools.evaluate_event_v2_policy import (
    aggregate_scene_metrics,
    completion_metrics,
    evaluate_runtime,
    load_policy_for_evaluation,
)


def _completion() -> CompletionSnapshot:
    return CompletionSnapshot(
        progress=torch.tensor([10., 5.]),
        required_duration=torch.tensor([10., 10.]),
        completed=torch.tensor([True, False]),
    )


def test_completion_metrics_match_registered_quality_formula() -> None:
    metrics = completion_metrics(_completion())

    assert metrics == pytest.approx({
        'CR': 0.5,
        'PCR': 0.75,
        'WCR': 0.5,
        'Q': 0.55,
    })


def test_aggregate_scene_metrics_use_macro_mean() -> None:
    aggregate = aggregate_scene_metrics([
        {'CR': 0.5, 'PCR': 0.75, 'WCR': 0.5, 'Q': 0.55},
        {'CR': 1.0, 'PCR': 1.0, 'WCR': 1.0, 'Q': 1.0},
    ])

    assert aggregate == pytest.approx({
        'CR': 0.75,
        'PCR': 0.875,
        'WCR': 0.75,
        'Q': 0.775,
    })


def test_aggregate_scene_metrics_reports_tat_power_and_paper_score() -> None:
    aggregate = aggregate_scene_metrics([
        {
            'CR': 0.5,
            'PCR': 0.75,
            'WCR': 0.5,
            'Q': 0.55,
            'TAT_s': 100.0,
            'PC_Wh': 2.0,
        },
        {
            'CR': 1.0,
            'PCR': 1.0,
            'WCR': 1.0,
            'Q': 1.0,
            'TAT_s': 200.0,
            'PC_Wh': 4.0,
        },
    ])

    assert aggregate['TAT_s'] == pytest.approx(150.0)
    assert aggregate['PC_Wh'] == pytest.approx(3.0)
    assert aggregate['CS_paper'] == pytest.approx(
        1 / 0.775 + 150 / 700 + 3 / 100,
    )


def test_evaluate_runtime_uses_deterministic_actor_and_one_trajectory() -> None:
    action = JointEventAction(
        terminate=torch.zeros(1, 1, dtype=torch.bool),
        task_indices=torch.full((1, 1), -1, dtype=torch.long),
        commitment_indices=torch.full((1, 1), -1, dtype=torch.long),
    )

    class FakeObservation:
        event_state = object()

        def to(self, device):
            assert device == torch.device('cpu')
            return self

        def model_args(self):
            return ()

    class FakeModel:
        def __init__(self):
            self.deterministic_calls = []

        def eval(self):
            return self

        def act(self, *args, event_state, deterministic):
            del args, event_state
            self.deterministic_calls.append(deterministic)
            return SimpleNamespace(
                actor=SimpleNamespace(action=action),
            )

    class FakeBackend:
        def completion_snapshot(self):
            return _completion()

        def operational_metrics(self):
            return {
                'TAT_s': 100.0,
                'PC_Wh': 2.0,
            }

    class FakeRuntime:
        backend = FakeBackend()

        def __init__(self):
            self.reset_calls = 0
            self.step_calls = 0

        def reset(self):
            self.reset_calls += 1
            return FakeObservation()

        def step(self, sampled_action):
            self.step_calls += 1
            assert sampled_action.task_indices.device.type == 'cpu'
            return RuntimeStep(
                observation=None,
                reward=0.55,
                delta_t=5,
                done=True,
                final_quality=0.55,
                invalid_action_count=0,
            )

    model = FakeModel()
    runtime = FakeRuntime()
    result = evaluate_runtime(
        model=model,
        runtime=runtime,
        device=torch.device('cpu'),
        amp_enabled=False,
        amp_dtype=torch.bfloat16,
    )

    assert model.deterministic_calls == [True]
    assert runtime.reset_calls == 1
    assert runtime.step_calls == 1
    assert result['events'] == 1
    assert result['physical_seconds'] == 5
    assert result['CR'] == pytest.approx(0.5)
    assert result['Q'] == pytest.approx(0.55)
    assert result['TAT_s'] == pytest.approx(100.0)
    assert result['PC_Wh'] == pytest.approx(2.0)
    assert result['CS_paper'] == pytest.approx(
        1 / 0.55 + 100 / 700 + 2 / 100,
    )


def test_v2_3_evaluation_unfreezes_tail_then_disables_all_gradients(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / 'v2_3.pth'
    torch.save({'stage': 'V2-3'}, checkpoint)
    calls = []

    class FakeModel:
        def unfreeze_last_layers(self, *, encoder_layers, decoder_layers):
            calls.append(('unfreeze', encoder_layers, decoder_layers))

        def requires_grad_(self, enabled):
            calls.append(('requires_grad', enabled))
            return self

        def eval(self):
            calls.append(('eval',))
            return self

    metadata = SimpleNamespace(stage='V2-3')

    def fake_loader(
        *,
        path,
        model,
        expected_encoder_layers,
        expected_decoder_layers,
        expected_backbone_lr_scale,
    ):
        assert path == checkpoint
        assert isinstance(model, FakeModel)
        assert expected_encoder_layers == 1
        assert expected_decoder_layers == 1
        assert expected_backbone_lr_scale == pytest.approx(0.1)
        calls.append(('load',))
        return metadata

    monkeypatch.setattr(
        'tools.evaluate_event_v2_policy.load_appo_policy_checkpoint',
        fake_loader,
    )

    actual = load_policy_for_evaluation(
        path=checkpoint,
        model=FakeModel(),
    )

    assert actual is metadata
    assert calls == [
        ('unfreeze', 1, 1),
        ('load',),
        ('requires_grad', False),
        ('eval',),
    ]


def test_large_sync_evaluation_uses_large_checkpoint_loader(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / 'large_sync.pth'
    torch.save({'stage': 'V2-2-Large'}, checkpoint)
    calls = []

    class FakeModel:
        def requires_grad_(self, enabled):
            calls.append(('requires_grad', enabled))
            return self

        def eval(self):
            calls.append(('eval',))
            return self

    metadata = SimpleNamespace(stage='V2-2-Large')

    def fake_loader(*, path, model):
        assert path == checkpoint
        assert isinstance(model, FakeModel)
        calls.append(('load_large',))
        return metadata

    monkeypatch.setattr(
        'tools.evaluate_event_v2_policy.'
        'load_large_sync_policy_checkpoint',
        fake_loader,
    )

    actual = load_policy_for_evaluation(
        path=checkpoint,
        model=FakeModel(),
    )

    assert actual is metadata
    assert calls == [
        ('load_large',),
        ('requires_grad', False),
        ('eval',),
    ]
