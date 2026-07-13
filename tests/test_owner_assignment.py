import numpy as np

from constellation.rl import eval_all
from constellation.rl.owner_assignment import resolve_owner_assignments


def test_resolver_finds_global_assignment_instead_of_independent_argmax() -> None:
    logits = np.array([
        [0.0, 9.0, 8.0],
        [0.0, 8.5, 0.1],
    ])

    actions, owner_task_ids = resolve_owner_assignments(
        logits,
        task_ids=[10, 20],
        num_satellites=2,
    )

    assert actions.tolist() == [2, 1]
    assert owner_task_ids.tolist() == [20, 10]


def test_resolver_uses_null_action_when_task_is_worse_than_null() -> None:
    logits = np.array([
        [2.0, 1.0, 0.5],
        [0.0, 3.0, 2.0],
    ])

    actions, owner_task_ids = resolve_owner_assignments(
        logits,
        task_ids=[10, 20],
        num_satellites=2,
    )

    assert actions.tolist() == [0, 1]
    assert owner_task_ids.tolist() == [-1, 10]


def test_continuation_bonus_tracks_global_task_id_after_reordering() -> None:
    logits = np.array([
        [0.0, 0.7, 0.8],
        [0.0, 0.1, 1.2],
    ])

    actions, owner_task_ids = resolve_owner_assignments(
        logits,
        task_ids=[20, 10],
        num_satellites=2,
        previous_owner_task_ids=[10, -1],
        continuation_bonus=1.1,
    )

    assert actions.tolist() == [2, 1]
    assert owner_task_ids.tolist() == [10, 20]


def test_resolver_keeps_padded_satellites_on_null_action() -> None:
    logits = np.array([
        [0.0, 3.0],
        [0.0, 2.0],
        [0.0, 100.0],
    ])

    actions, owner_task_ids = resolve_owner_assignments(
        logits,
        task_ids=[10],
        num_satellites=2,
    )

    assert actions.tolist() == [1, 0, 0]
    assert owner_task_ids.tolist() == [10, -1, -1]


def test_resolver_rejects_negative_continuation_bonus() -> None:
    with np.testing.assert_raises_regex(
        ValueError,
        'continuation_bonus must be non-negative',
    ):
        resolve_owner_assignments(
            np.zeros((1, 2)),
            task_ids=[10],
            num_satellites=1,
            continuation_bonus=-0.1,
        )


def test_finished_worker_ignores_owner_payload_without_controller() -> None:
    environment = eval_all.EvalEnvironment.__new__(
        eval_all.EvalEnvironment,
    )
    environment._counter = 0
    environment._world_size = 1
    environment._rank = 0
    environment._annotations = []
    environment._enable_owner_assignment = True

    _, reward, terminated, truncated, info = environment.step((
        np.zeros(1, dtype=np.int64),
        {'owner_logits': np.zeros((1, 1))},
    ))

    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info == {'all_done': True}
