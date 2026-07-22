import inspect

import torch

from constellation.new_transformers.event_v2.dataset import (
    EventV2OfflineDataset,
    OfflineEventBatch,
    build_capped_owner_counts,
    build_commitment_targets,
    compress_expert_actions_to_events,
)


def test_event_compression_keeps_action_validity_and_completion_events() -> None:
    actions = torch.tensor([[-1], [-1], [0], [0], [0], [1], [1]])
    task_valid = torch.tensor([
        [False, False],
        [False, False],
        [True, True],
        [True, True],
        [True, True],
        [True, True],
        [True, True],
    ])
    progress = torch.tensor([
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [5., 0.],
        [10., 0.],
        [10., 0.],
        [10., 1.],
    ])

    events = compress_expert_actions_to_events(
        actions=actions,
        task_valid=task_valid,
        progress=progress,
        durations=torch.tensor([10., 10.]),
    )

    assert events == [2, 4, 5]


def test_continuous_same_action_does_not_create_per_second_events() -> None:
    actions = torch.zeros(8, 2, dtype=torch.long)
    task_valid = torch.ones(8, 1, dtype=torch.bool)
    progress = torch.zeros(8, 1)

    events = compress_expert_actions_to_events(
        actions=actions,
        task_valid=task_valid,
        progress=progress,
        durations=torch.tensor([10.]),
    )

    assert events == [1]


def test_commitment_targets_use_longest_observed_legal_minimum() -> None:
    actions = torch.tensor([
        [0, 1],
        [0, 1],
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 0],
    ])
    remaining = torch.full((7, 2), 30.)
    remaining[1, 0] = 1.

    indices, observed = build_commitment_targets(
        actions=actions,
        event_indices=[1],
        task_remaining_required_seconds=remaining,
    )

    assert indices.tolist() == [[0, 1]]
    assert observed.tolist() == [[True, True]]


def test_commitment_target_ignores_too_short_nonterminal_segment() -> None:
    actions = torch.tensor([[0], [0], [1]])
    remaining = torch.full((3, 2), 30.)

    indices, observed = build_commitment_targets(
        actions=actions,
        event_indices=[1],
        task_remaining_required_seconds=remaining,
    )

    assert indices.tolist() == [[0]]
    assert observed.tolist() == [[False]]


def test_offline_batch_contract_contains_no_visibility_tensor() -> None:
    assert 'is_visible' not in OfflineEventBatch._fields
    signature = inspect.signature(EventV2OfflineDataset.__getitem__)
    assert 'is_visible' not in signature.parameters


def test_legacy_expert_owner_counts_saturate_at_v2_safety_cap() -> None:
    counts = build_capped_owner_counts(
        torch.tensor([[0, 0, 0, 0, 1]]),
        num_tasks=2,
    )

    assert counts.tolist() == [[3, 1]]
