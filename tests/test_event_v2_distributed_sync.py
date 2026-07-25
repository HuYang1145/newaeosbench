from __future__ import annotations

from dataclasses import dataclass
import queue
import threading

import pytest
import torch

from constellation.new_transformers.event_v2.appo import SharedPolicyStore
from constellation.new_transformers.event_v2.distributed_sync import (
    QueuedEventRuntimePool,
    StrictSyncRoundCoordinator,
    SyncActorChunk,
    SyncActorDone,
    SyncRoundCommand,
    SyncWorkerError,
    run_strict_sync_actor_loop,
    validate_and_merge_sync_round,
)
from tools.train_event_v2_sync_ppo import (
    SyntheticEventRuntime,
    _tiny_model,
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


def _runtime_factory(
    created_scene_ids: list[int],
    *,
    num_events: int,
):
    def factory(scene_id: int) -> SyntheticEventRuntime:
        created_scene_ids.append(scene_id)
        return SyntheticEventRuntime(num_events=num_events)

    return factory


def _synthetic_runtime_loader(
    scene_id: int,
    state: dict,
) -> SyntheticEventRuntime:
    del scene_id
    runtime = SyntheticEventRuntime(num_events=int(state['num_events']))
    runtime.events = int(state['events'])
    runtime.total_reward = float(state['total_reward'])
    return runtime


def test_runtime_pool_caps_active_scenes_and_refills_its_own_queue() -> None:
    created: list[int] = []
    pool = QueuedEventRuntimePool(
        assigned_scene_ids=tuple(range(205, 215)),
        max_active_environments=5,
        runtime_factory=_runtime_factory(created, num_events=1),
        runtime_state_loader=_synthetic_runtime_loader,
    )

    payload = pool.collect(
        model=_tiny_model(),
        policy_version=0,
        max_events=8,
        device=torch.device('cpu'),
        replay_atol=1e-6,
    )

    assert len(payload.steps) == 8
    assert payload.completed_scene_ids == tuple(range(205, 213))
    assert pool.active_environment_count == 2
    assert pool.pending_scene_ids == ()
    assert pool.completed_scene_ids == tuple(range(205, 213))
    assert created == list(range(205, 215))
    assert {
        step.environment_index for step in payload.steps
    } == set(range(205, 213))

    final = pool.collect(
        model=_tiny_model(),
        policy_version=0,
        max_events=8,
        device=torch.device('cpu'),
        replay_atol=1e-6,
    )

    assert len(final.steps) == 2
    assert final.completed_scene_ids == (213, 214)
    assert pool.is_complete is True
    assert pool.active_environment_count == 0
    assert pool.completed_scene_ids == tuple(range(205, 215))
    assert len(set(pool.completed_scene_ids)) == 10


def test_runtime_pool_state_restores_active_pending_and_completed_scenes() -> None:
    created: list[int] = []
    pool = QueuedEventRuntimePool(
        assigned_scene_ids=(205, 206, 207, 208),
        max_active_environments=2,
        runtime_factory=_runtime_factory(created, num_events=2),
        runtime_state_loader=_synthetic_runtime_loader,
    )
    first = pool.collect(
        model=_tiny_model(),
        policy_version=3,
        max_events=3,
        device=torch.device('cpu'),
        replay_atol=1e-6,
    )
    state = pool.state_dict()
    restored = QueuedEventRuntimePool(
        assigned_scene_ids=(205, 206, 207, 208),
        max_active_environments=2,
        runtime_factory=_runtime_factory([], num_events=2),
        runtime_state_loader=_synthetic_runtime_loader,
        initialize=False,
    )

    restored.load_state_dict(state)

    assert len(first.steps) == 3
    assert restored.state_dict() == state
    assert restored.active_environment_count <= 2
    assert (
        set(restored.pending_scene_ids)
        | set(restored.active_scene_ids)
        | set(restored.completed_scene_ids)
    ) == {205, 206, 207, 208}


def test_runtime_pool_rejects_behavior_replay_mismatch(monkeypatch) -> None:
    pool = QueuedEventRuntimePool(
        assigned_scene_ids=(205,),
        max_active_environments=1,
        runtime_factory=lambda _: SyntheticEventRuntime(num_events=2),
        runtime_state_loader=_synthetic_runtime_loader,
    )

    def mismatched_replay(*args, **kwargs):
        del args, kwargs
        return torch.tensor([10.0])

    monkeypatch.setattr(
        'constellation.new_transformers.event_v2.distributed_sync.'
        'replay_rollout_log_probs',
        mismatched_replay,
    )

    with pytest.raises(RuntimeError, match='replay mismatch'):
        pool.collect(
            model=_tiny_model(),
            policy_version=0,
            max_events=1,
            device=torch.device('cpu'),
            replay_atol=1e-6,
        )


def test_actor_loop_waits_for_exact_round_commands_and_done_ack() -> None:
    context = torch.multiprocessing.get_context('spawn')
    store = SharedPolicyStore(
        _tiny_model(),
        context=context,
        initial_version=0,
    )
    pool = QueuedEventRuntimePool(
        assigned_scene_ids=(205,),
        max_active_environments=1,
        runtime_factory=lambda _: SyntheticEventRuntime(num_events=2),
        runtime_state_loader=_synthetic_runtime_loader,
    )
    commands: queue.Queue = queue.Queue()
    results: queue.Queue = queue.Queue()
    stop = threading.Event()
    worker = threading.Thread(
        target=run_strict_sync_actor_loop,
        kwargs={
            'model': _tiny_model(),
            'pool': pool,
            'actor_id': 0,
            'policy_store': store,
            'command_queue': commands,
            'result_queue': results,
            'stop_event': stop,
            'target_events': 1,
            'device': torch.device('cpu'),
            'replay_atol': 1e-6,
            'initial_round_id': 0,
        },
    )
    worker.start()

    with pytest.raises(queue.Empty):
        results.get(timeout=0.1)
    commands.put(SyncRoundCommand(round_id=0, policy_version=0))
    first = results.get(timeout=3)
    assert isinstance(first, SyncActorChunk)
    assert first.round_id == 0
    assert first.policy_version == 0
    assert first.state['rng']['torch'] is not None
    with pytest.raises(queue.Empty):
        results.get(timeout=0.1)

    source = _tiny_model()
    with torch.no_grad():
        next(source.parameters()).add_(0.25)
    store.publish(source, version=1)
    commands.put(SyncRoundCommand(round_id=1, policy_version=1))
    second = results.get(timeout=3)
    done = results.get(timeout=3)

    assert isinstance(second, SyncActorChunk)
    assert second.round_id == 1
    assert second.policy_version == 1
    assert isinstance(done, SyncActorDone)
    assert done.completed_scene_ids == (205,)
    assert worker.is_alive()
    stop.set()
    worker.join(timeout=3)
    assert not worker.is_alive()


def test_actor_loop_rejects_skipped_or_unpublished_versions() -> None:
    context = torch.multiprocessing.get_context('spawn')
    store = SharedPolicyStore(
        _tiny_model(),
        context=context,
        initial_version=0,
    )
    pool = QueuedEventRuntimePool(
        assigned_scene_ids=(205,),
        max_active_environments=1,
        runtime_factory=lambda _: SyntheticEventRuntime(num_events=2),
        runtime_state_loader=_synthetic_runtime_loader,
    )
    commands: queue.Queue = queue.Queue()
    results: queue.Queue = queue.Queue()
    stop = threading.Event()
    commands.put(SyncRoundCommand(round_id=1, policy_version=1))

    with pytest.raises(RuntimeError, match='round'):
        run_strict_sync_actor_loop(
            model=_tiny_model(),
            pool=pool,
            actor_id=0,
            policy_store=store,
            command_queue=commands,
            result_queue=results,
            stop_event=stop,
            target_events=1,
            device=torch.device('cpu'),
            replay_atol=1e-6,
            initial_round_id=0,
        )
