import torch

from constellation.new_transformers.event_v2.actor import (
    AutoregressiveJointActor,
)
from constellation.new_transformers.event_v2.critic import EventStateEncoding
from constellation.new_transformers.event_v2.state import EventStateTensors


def _state(
    *,
    replan_mask: torch.Tensor | None = None,
    forced_interrupt_mask: torch.Tensor | None = None,
    can_terminate_mask: torch.Tensor | None = None,
    commitment_remaining: torch.Tensor | None = None,
    current_task_indices: torch.Tensor | None = None,
    owner_count: torch.Tensor | None = None,
    task_remaining: torch.Tensor | None = None,
) -> EventStateTensors:
    return EventStateTensors(
        previous_task_indices=torch.tensor([[-1, -1]]),
        current_task_indices=(
            torch.tensor([[-1, -1]])
            if current_task_indices is None else current_task_indices
        ),
        minimum_commitment_remaining=(
            torch.zeros(1, 2)
            if commitment_remaining is None else commitment_remaining
        ),
        run_lengths=torch.tensor([[5., 10.]]),
        seconds_since_replan=torch.tensor([[2., 12.]]),
        switch_count_30=torch.tensor([[0., 1.]]),
        switch_count_60=torch.tensor([[1., 2.]]),
        termination_reason=torch.tensor([[0, 0]]),
        event_type=torch.tensor([[0, 0]]),
        delta_t=torch.tensor([[5., 5.]]),
        replan_mask=(
            torch.ones(1, 2, dtype=torch.bool)
            if replan_mask is None else replan_mask
        ),
        forced_interrupt_mask=(
            torch.zeros(1, 2, dtype=torch.bool)
            if forced_interrupt_mask is None else forced_interrupt_mask
        ),
        can_terminate_mask=(
            torch.zeros(1, 2, dtype=torch.bool)
            if can_terminate_mask is None else can_terminate_mask
        ),
        compatible_deadline_slack=torch.tensor([[10., 5.]]),
        task_remaining_required_seconds=(
            torch.tensor([[30., 30.]])
            if task_remaining is None else task_remaining
        ),
        task_owner_count=(
            torch.zeros(1, 2, dtype=torch.long)
            if owner_count is None else owner_count
        ),
        task_locked_owner_count=torch.zeros(1, 2, dtype=torch.long),
    )


def _encoding(
    *,
    edge_tokens: torch.Tensor | None = None,
) -> EventStateEncoding:
    return EventStateEncoding(
        satellite_tokens=torch.zeros(1, 2, 4),
        task_tokens=torch.zeros(1, 2, 4),
        edge_tokens=(
            torch.zeros(1, 2, 2, 4)
            if edge_tokens is None else edge_tokens
        ),
    )


def _actor() -> AutoregressiveJointActor:
    actor = AutoregressiveJointActor(event_width=4)
    with torch.no_grad():
        actor.termination_head.weight.zero_()
        actor.termination_head.bias.fill_(-10.)
        actor.idle_head.weight.zero_()
        actor.idle_head.bias.fill_(-10.)
        actor.task_value_head.weight.zero_()
        actor.task_value_head.bias.fill_(5.)
        actor.owner_marginal_head.weight.zero_()
        actor.owner_marginal_head.bias.fill_(-1.)
        actor.commitment_head.weight.zero_()
        actor.commitment_head.bias.zero_()
    return actor.eval()


def test_termination_excludes_locked_and_forced_interruptions() -> None:
    actor = _actor()
    with torch.no_grad():
        actor.termination_head.bias.fill_(10.)
    state = _state(
        replan_mask=torch.zeros(1, 2, dtype=torch.bool),
        forced_interrupt_mask=torch.tensor([[False, True]]),
        can_terminate_mask=torch.tensor([[True, True]]),
        commitment_remaining=torch.tensor([[5., 0.]]),
    )

    output = actor.sample_actions(
        _encoding(),
        state,
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, 2, dtype=torch.bool),
        deterministic=True,
    )

    assert output.action.terminate.tolist() == [[False, False]]
    assert output.trace.termination_mask.tolist() == [[False, False]]
    assert output.trace.action_order.tolist() == [[1, -1]]


def test_active_termination_enters_replan_order_and_log_prob() -> None:
    actor = _actor()
    with torch.no_grad():
        actor.termination_head.bias.fill_(10.)
    state = _state(
        replan_mask=torch.zeros(1, 2, dtype=torch.bool),
        can_terminate_mask=torch.tensor([[True, False]]),
    )

    output = actor.sample_actions(
        _encoding(),
        state,
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, 2, dtype=torch.bool),
        deterministic=True,
    )

    assert output.action.terminate.tolist() == [[True, False]]
    assert output.trace.termination_mask.tolist() == [[True, False]]
    assert output.trace.action_order.tolist() == [[0, -1]]
    expected = torch.distributions.Bernoulli(
        logits=torch.tensor(10.),
    ).log_prob(torch.tensor(1.))
    assert output.log_prob.item() < 0
    assert output.log_prob.item() <= expected.item()


def test_autoregressive_owner_state_changes_second_assignment() -> None:
    actor = _actor()

    output = actor.sample_actions(
        _encoding(),
        _state(),
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, 2, dtype=torch.bool),
        deterministic=True,
    )

    assert output.action.task_indices.tolist() == [[1, 0]]
    assert output.trace.action_order.tolist() == [[1, 0]]
    assert output.trace.owner_state[0, 0].tolist() == [0, 0]
    assert output.trace.owner_state[0, 1].tolist() == [1, 0]


def test_owner_count_three_is_always_physically_masked() -> None:
    actor = _actor()
    output = actor.sample_actions(
        _encoding(),
        _state(owner_count=torch.tensor([[3, 0]])),
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, 2, dtype=torch.bool),
        deterministic=True,
    )

    assert output.trace.task_masks[0, 0, 1].item() is False
    assert 0 not in output.action.task_indices[0].tolist()


def test_negative_marginal_blocks_duplicate_in_deterministic_mode() -> None:
    actor = _actor()
    edge_tokens = torch.zeros(1, 2, 2, 4)
    edge_tokens[:, :, 0, 0] = 10.
    with torch.no_grad():
        actor.task_value_head.weight[0, 0] = 1.
        actor.task_value_head.bias.zero_()
    output = actor.sample_actions(
        _encoding(edge_tokens=edge_tokens),
        _state(owner_count=torch.tensor([[1, 0]])),
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, 2, dtype=torch.bool),
        deterministic=True,
    )

    assert output.trace.task_masks[0, 0, 1].item() is False
    assert output.action.task_indices[0, 1].item() == 1


def test_commitment_mask_uses_selected_task_remaining_duration() -> None:
    actor = _actor()
    output = actor.sample_actions(
        _encoding(),
        _state(task_remaining=torch.tensor([[1., 30.]])),
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, 2, dtype=torch.bool),
        deterministic=True,
    )
    order = output.trace.action_order[0].tolist()
    first_task = output.action.task_indices[0, order[0]].item()
    first_mask = output.trace.commitment_masks[0, 0]

    if first_task == 0:
        assert first_mask.tolist() == [True] * 5
    else:
        assert first_mask.tolist() == [False, True, True, True, True]


def test_replayed_joint_log_prob_matches_behavior_exactly() -> None:
    torch.manual_seed(17)
    actor = _actor()
    actor.train()
    encoding = _encoding()
    state = _state(
        can_terminate_mask=torch.tensor([[True, False]]),
    )
    satellite_mask = torch.ones(1, 2, dtype=torch.bool)
    task_mask = torch.ones(1, 2, dtype=torch.bool)

    sampled = actor.sample_actions(
        encoding,
        state,
        satellite_mask,
        task_mask,
        deterministic=False,
    )
    replayed = actor.evaluate_actions(
        encoding,
        state,
        satellite_mask,
        task_mask,
        sampled.action,
        sampled.trace,
    )

    torch.testing.assert_close(replayed.log_prob, sampled.log_prob, rtol=0, atol=0)
    torch.testing.assert_close(replayed.entropy, sampled.entropy, rtol=0, atol=0)


def test_actor_outputs_are_finite_for_legal_batch() -> None:
    actor = _actor()
    output = actor.sample_actions(
        _encoding(),
        _state(),
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, 2, dtype=torch.bool),
        deterministic=False,
    )

    assert torch.isfinite(output.log_prob).all()
    assert torch.isfinite(output.entropy).all()
    assert output.action.commitment_indices.shape == (1, 2)
