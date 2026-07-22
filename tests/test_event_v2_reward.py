import pytest
import torch

from constellation.new_transformers.event_v2.reward import (
    build_completion_event_rewards,
    completion_potential,
    terminal_completion_quality,
    time_aware_gae,
)


def test_completion_reward_telescopes_to_exact_q() -> None:
    weights = torch.tensor([0.3, 0.7])
    progress = [
        torch.tensor([0., 0.]),
        torch.tensor([5., 0.]),
        torch.tensor([10., 5.]),
    ]

    rewards = build_completion_event_rewards(
        progress=progress,
        required_duration=torch.tensor([10., 10.]),
        task_weights=weights,
        completed=torch.tensor([True, False]),
    )

    torch.testing.assert_close(sum(rewards), torch.tensor(0.3))


def test_terminal_correction_reclaims_unfinished_partial_progress() -> None:
    rewards = build_completion_event_rewards(
        progress=[torch.tensor([0.]), torch.tensor([9.])],
        required_duration=torch.tensor([10.]),
        task_weights=torch.tensor([1.]),
        completed=torch.tensor([False]),
    )

    torch.testing.assert_close(rewards[0], torch.tensor(0.))


def test_completion_reward_supports_batched_scenes() -> None:
    rewards = build_completion_event_rewards(
        progress=[
            torch.tensor([[0., 0.], [0., 0.]]),
            torch.tensor([[10., 5.], [0., 10.]]),
        ],
        required_duration=torch.tensor([[10., 10.], [10., 10.]]),
        task_weights=torch.tensor([[0.3, 0.7], [0.2, 0.8]]),
        completed=torch.tensor([[True, False], [False, True]]),
    )

    torch.testing.assert_close(rewards[0], torch.tensor([0.3, 0.8]))


def test_potential_clamps_progress_to_required_duration() -> None:
    potential = completion_potential(
        progress=torch.tensor([[-1., 20.]]),
        required_duration=torch.tensor([[10., 10.]]),
        task_weights=torch.tensor([[0.4, 0.6]]),
    )

    torch.testing.assert_close(potential, torch.tensor([0.6]))


def test_terminal_quality_only_counts_completed_tasks() -> None:
    quality = terminal_completion_quality(
        completed=torch.tensor([[True, False], [True, True]]),
        task_weights=torch.tensor([[0.4, 0.6], [0.2, 0.8]]),
    )

    torch.testing.assert_close(quality, torch.tensor([0.4, 1.0]))


def test_time_aware_gae_uses_physical_delta_t() -> None:
    result = time_aware_gae(
        rewards=torch.tensor([1., 2.]),
        values=torch.tensor([0.5, 0.25]),
        next_values=torch.tensor([0.25, 0.]),
        delta_t=torch.tensor([5., 10.]),
        done=torch.tensor([False, True]),
        lambda_base=0.95,
        reference_seconds=5.,
    )
    expected_last = torch.tensor(1.75)
    expected_first = torch.tensor(0.75) + 0.95 * expected_last

    torch.testing.assert_close(
        result.advantages,
        torch.stack((expected_first, expected_last)),
    )
    torch.testing.assert_close(
        result.returns,
        result.advantages + torch.tensor([0.5, 0.25]),
    )


def test_time_aware_gae_stops_bootstrap_and_trace_at_terminal() -> None:
    result = time_aware_gae(
        rewards=torch.tensor([1., 2.]),
        values=torch.tensor([0., 0.]),
        next_values=torch.tensor([100., 100.]),
        delta_t=torch.tensor([5., 5.]),
        done=torch.tensor([True, True]),
    )

    torch.testing.assert_close(result.advantages, torch.tensor([1., 2.]))


@pytest.mark.parametrize(
    ('required_duration', 'task_weights', 'error'),
    [
        (torch.tensor([0.]), torch.tensor([1.]), 'duration'),
        (torch.tensor([1.]), torch.tensor([-1.]), 'weight'),
        (torch.tensor([1.]), torch.tensor([float('nan')]), 'finite'),
    ],
)
def test_completion_potential_rejects_invalid_inputs(
    required_duration: torch.Tensor,
    task_weights: torch.Tensor,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        completion_potential(
            progress=torch.tensor([0.]),
            required_duration=required_duration,
            task_weights=task_weights,
        )


def test_completion_rewards_require_at_least_two_states() -> None:
    with pytest.raises(ValueError, match='two states'):
        build_completion_event_rewards(
            progress=[torch.tensor([0.])],
            required_duration=torch.tensor([1.]),
            task_weights=torch.tensor([1.]),
            completed=torch.tensor([False]),
        )


def test_time_aware_gae_rejects_nonpositive_delta_t() -> None:
    with pytest.raises(ValueError, match='delta_t'):
        time_aware_gae(
            rewards=torch.tensor([1.]),
            values=torch.tensor([0.]),
            next_values=torch.tensor([0.]),
            delta_t=torch.tensor([0.]),
            done=torch.tensor([False]),
        )
