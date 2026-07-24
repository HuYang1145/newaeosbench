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
