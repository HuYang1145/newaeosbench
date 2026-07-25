from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from constellation.new_transformers.event_v2.distributed_sync import (
    StrictSyncRoundCoordinator,
    SyncActorChunk,
    SyncRoundCommand,
    SyncWorkerError,
    validate_and_merge_sync_round,
)


@dataclass(frozen=True)
class _StubStep:
    environment_index: int
    episode_id: int
    event_index: int
    policy_version: int
    delta_t: torch.Tensor = torch.tensor(5.0)

    def validate(self) -> None:
        if min(
            self.environment_index,
            self.episode_id,
            self.event_index,
            self.policy_version,
        ) < 0:
            raise ValueError('invalid stub rollout identifier')


def _chunk(
    actor_id: int,
    *,
    round_id: int = 3,
    policy_version: int = 7,
    events: int = 32,
    environment_offset: int | None = None,
) -> SyncActorChunk:
    offset = actor_id * 100 if environment_offset is None else environment_offset
    return SyncActorChunk(
        actor_id=actor_id,
        round_id=round_id,
        policy_version=policy_version,
        steps=tuple(
            _StubStep(
                environment_index=offset + event_index,
                episode_id=0,
                event_index=event_index,
                policy_version=policy_version,
            )
            for event_index in range(events)
        ),
        completed_scene_ids=(offset,) if events else (),
        replay_max_abs_error=5e-7,
        state={'actor_id': actor_id, 'round_id': round_id},
    )


def test_sync_round_merges_one_exact_version_in_deterministic_order() -> None:
    batch = validate_and_merge_sync_round(
        [_chunk(1), _chunk(0)],
        expected_actor_ids=(0, 1),
        round_id=3,
        policy_version=7,
        min_batch_events=64,
    )

    assert batch.round_id == 3
    assert batch.policy_version == 7
    assert batch.actor_ids == (0, 1)
    assert batch.event_count == 64
    assert batch.should_update is True
    assert batch.completed_scene_ids == (0, 100)
    assert batch.replay_max_abs_error == pytest.approx(5e-7)
    assert [
        (step.environment_index, step.event_index)
        for step in batch.steps
    ] == sorted(
        (step.environment_index, step.event_index)
        for step in batch.steps
    )


@pytest.mark.parametrize(
    ('chunks', 'message'),
    [
        ([_chunk(0)], 'actor set'),
        ([_chunk(0), _chunk(0)], 'duplicate actor'),
        ([_chunk(0), _chunk(1, round_id=4)], 'round'),
        ([_chunk(0), _chunk(1, policy_version=6)], 'policy version'),
        ([_chunk(0), _chunk(1, policy_version=8)], 'policy version'),
    ],
)
def test_sync_round_rejects_missing_duplicate_or_mixed_chunks(
    chunks: list[SyncActorChunk],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_and_merge_sync_round(
            chunks,
            expected_actor_ids=(0, 1),
            round_id=3,
            policy_version=7,
            min_batch_events=64,
        )


def test_sync_round_rejects_mixed_step_policy_versions() -> None:
    chunk = _chunk(1)
    mixed = chunk.steps[0]
    bad = chunk.__class__(
        **{
            **chunk.__dict__,
            'steps': (
                mixed.__class__(
                    environment_index=mixed.environment_index,
                    episode_id=mixed.episode_id,
                    event_index=mixed.event_index,
                    policy_version=6,
                ),
                *chunk.steps[1:],
            ),
        },
    )

    with pytest.raises(ValueError, match='step policy version'):
        validate_and_merge_sync_round(
            [_chunk(0), bad],
            expected_actor_ids=(0, 1),
            round_id=3,
            policy_version=7,
            min_batch_events=64,
        )


def test_sync_round_never_updates_from_a_small_final_batch() -> None:
    batch = validate_and_merge_sync_round(
        [_chunk(0, events=12), _chunk(1, events=13)],
        expected_actor_ids=(0, 1),
        round_id=3,
        policy_version=7,
        min_batch_events=64,
    )

    assert batch.event_count == 25
    assert batch.should_update is False


def test_round_coordinator_blocks_advance_until_every_actor_arrives() -> None:
    coordinator = StrictSyncRoundCoordinator(
        actor_ids=(0, 1),
        initial_round_id=3,
        initial_policy_version=7,
    )

    assert coordinator.command_for(0) == SyncRoundCommand(
        round_id=3,
        policy_version=7,
    )
    coordinator.submit(_chunk(0))
    with pytest.raises(RuntimeError, match='incomplete'):
        coordinator.finalize(min_batch_events=64)
    with pytest.raises(RuntimeError, match='finalized'):
        coordinator.advance(next_policy_version=8)

    coordinator.submit(_chunk(1))
    batch = coordinator.finalize(min_batch_events=64)
    assert batch.should_update is True
    coordinator.advance(next_policy_version=8)

    assert coordinator.command_for(1) == SyncRoundCommand(
        round_id=4,
        policy_version=8,
    )
    with pytest.raises(ValueError, match='round'):
        coordinator.submit(_chunk(0))


def test_round_coordinator_rejects_duplicate_actor_submission() -> None:
    coordinator = StrictSyncRoundCoordinator(
        actor_ids=(0, 1),
        initial_round_id=3,
        initial_policy_version=7,
    )
    coordinator.submit(_chunk(0))

    with pytest.raises(ValueError, match='duplicate actor'):
        coordinator.submit(_chunk(0))


def test_sync_worker_error_requires_complete_context() -> None:
    error = SyncWorkerError(
        actor_id=2,
        error_type='RuntimeError',
        message='synthetic failure',
        traceback='Traceback: synthetic failure',
    )

    assert error.actor_id == 2
    with pytest.raises(ValueError, match='incomplete'):
        SyncWorkerError(
            actor_id=2,
            error_type='RuntimeError',
            message='',
            traceback='Traceback',
        )
