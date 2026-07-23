"""带合法 termination、软容量和可重放 trace 的自回归联合 Actor。"""

import math
from typing import NamedTuple

import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical

from .critic import EventStateEncoding
from .state import (
    COMMITMENT_SECONDS,
    MAX_TASK_OWNERS,
    EventStateTensors,
    build_commitment_mask,
    build_replan_order,
)
from .transition import ActionTrace, JointEventAction


class ActorOutput(NamedTuple):
    action: JointEventAction
    log_prob: torch.Tensor
    entropy: torch.Tensor
    trace: ActionTrace


class ActionEvaluation(NamedTuple):
    log_prob: torch.Tensor
    entropy: torch.Tensor


class AutoregressiveJointActor(nn.Module):
    """按紧迫度顺序联合解码 termination、任务和最低承诺。"""

    def __init__(self, *, event_width: int) -> None:
        super().__init__()
        if event_width <= 0:
            raise ValueError('event_width must be positive')
        self.event_width = event_width
        self.termination_head = nn.Linear(event_width, 1)
        self.idle_head = nn.Linear(event_width, 1)
        self.task_value_head = nn.Linear(event_width, 1)
        self.owner_marginal_head = nn.Linear(event_width, 2)
        self.commitment_head = nn.Linear(
            3 * event_width,
            len(COMMITMENT_SECONDS),
        )
        self.commitment_embedding = nn.Embedding(
            len(COMMITMENT_SECONDS),
            event_width,
        )
        self.idle_context = nn.Parameter(torch.zeros(event_width))
        self.prefix_update = nn.Linear(2 * event_width, event_width)
        self.prefix_norm = nn.LayerNorm(event_width)

    def _validate_inputs(
        self,
        encoding: EventStateEncoding,
        state: EventStateTensors,
        satellite_mask: torch.Tensor,
        task_mask: torch.Tensor,
    ) -> tuple[int, int, int]:
        state.validate()
        batch_size, num_satellites, width = (
            encoding.satellite_tokens.shape
        )
        task_batch, num_tasks, task_width = encoding.task_tokens.shape
        if width != self.event_width or task_width != self.event_width:
            raise ValueError('event token width does not match actor')
        if task_batch != batch_size:
            raise ValueError('task and satellite tokens need same batch size')
        if encoding.edge_tokens.shape != (
            batch_size,
            num_satellites,
            num_tasks,
            self.event_width,
        ):
            raise ValueError('edge token shape does not match actor inputs')
        if state.replan_mask.shape != (batch_size, num_satellites):
            raise ValueError('state does not match actor satellite axis')
        if state.task_owner_count.shape != (batch_size, num_tasks):
            raise ValueError('state does not match actor task axis')
        if satellite_mask.shape != (batch_size, num_satellites):
            raise ValueError('satellite mask has invalid actor shape')
        if task_mask.shape != (batch_size, num_tasks):
            raise ValueError('task mask has invalid actor shape')
        if satellite_mask.dtype != torch.bool or task_mask.dtype != torch.bool:
            raise ValueError('actor masks must use bool dtype')
        if not all(torch.isfinite(value).all() for value in encoding):
            raise ValueError('actor tokens must contain finite values')
        return batch_size, num_satellites, num_tasks

    def _termination_distribution(
        self,
        encoding: EventStateEncoding,
    ) -> Bernoulli:
        return Bernoulli(
            logits=self.termination_head(
                encoding.satellite_tokens,
            ).squeeze(-1),
        )

    def _task_logits(
        self,
        encoding: EventStateEncoding,
        batch_index: int,
        satellite_id: int,
        prefix_context: torch.Tensor,
        owner_count: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = (
            encoding.satellite_tokens[batch_index, satellite_id]
            + prefix_context
        )
        task_tokens = encoding.task_tokens[batch_index]
        edge_tokens = encoding.edge_tokens[batch_index, satellite_id]
        base_logits = (
            torch.einsum('d,td->t', query, task_tokens)
            / math.sqrt(self.event_width)
            + self.task_value_head(edge_tokens).squeeze(-1)
        )
        marginal_logits = self.owner_marginal_head(edge_tokens)
        marginal_adjustment = torch.zeros_like(base_logits)
        marginal_adjustment = torch.where(
            owner_count == 1,
            marginal_logits[:, 0],
            marginal_adjustment,
        )
        marginal_adjustment = torch.where(
            owner_count == 2,
            marginal_logits[:, 1],
            marginal_adjustment,
        )
        task_logits = base_logits + marginal_adjustment
        idle_logit = self.idle_head(query).squeeze(-1)
        return (
            torch.cat((idle_logit.unsqueeze(0), task_logits)),
            marginal_adjustment,
            query,
        )

    def _commitment_logits(
        self,
        encoding: EventStateEncoding,
        batch_index: int,
        satellite_id: int,
        task_id: int,
        query: torch.Tensor,
    ) -> torch.Tensor:
        return self.commitment_head(torch.cat((
            query,
            encoding.task_tokens[batch_index, task_id],
            encoding.edge_tokens[batch_index, satellite_id, task_id],
        )))

    def _updated_prefix(
        self,
        prefix_context: torch.Tensor,
        selected_task: torch.Tensor,
        commitment_index: int | None,
    ) -> torch.Tensor:
        if commitment_index is None:
            commitment = torch.zeros_like(selected_task)
        else:
            commitment = self.commitment_embedding.weight[commitment_index]
        update = self.prefix_update(torch.cat((selected_task, commitment)))
        return self.prefix_norm(prefix_context + update)

    @staticmethod
    def _masked_categorical(
        logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> Categorical:
        if mask.dtype != torch.bool or mask.shape != logits.shape:
            raise ValueError('categorical mask must match logits')
        if not mask.any():
            raise ValueError('categorical action requires one legal category')
        return Categorical(logits=logits.masked_fill(~mask, float('-inf')))

    def sample_actions(
        self,
        encoding: EventStateEncoding,
        state: EventStateTensors,
        satellite_mask: torch.Tensor,
        task_mask: torch.Tensor,
        *,
        deterministic: bool,
        task_compatibility: torch.Tensor | None = None,
    ) -> ActorOutput:
        batch_size, num_satellites, num_tasks = self._validate_inputs(
            encoding,
            state,
            satellite_mask,
            task_mask,
        )
        termination_distribution = self._termination_distribution(encoding)
        if task_compatibility is None:
            task_compatibility = torch.ones(
                batch_size,
                num_satellites,
                num_tasks,
                dtype=torch.bool,
                device=task_mask.device,
            )
        elif (
            task_compatibility.shape
            != (batch_size, num_satellites, num_tasks)
            or task_compatibility.dtype != torch.bool
        ):
            raise ValueError('task compatibility has invalid actor shape')
        termination_mask = (
            satellite_mask
            & state.can_terminate_mask
            & ~state.forced_interrupt_mask
            & (state.minimum_commitment_remaining <= 0)
        )
        if deterministic:
            terminate = termination_distribution.logits >= 0
        else:
            terminate = termination_distribution.sample().bool()
        terminate &= termination_mask
        termination_log_prob = torch.where(
            termination_mask,
            termination_distribution.log_prob(terminate.float()),
            0.,
        ).sum(dim=-1)
        termination_entropy = torch.where(
            termination_mask,
            termination_distribution.entropy(),
            0.,
        ).sum(dim=-1)

        active_replan = (
            state.replan_mask | state.forced_interrupt_mask | terminate
        ) & satellite_mask
        planning_state = state._replace(replan_mask=active_replan)
        orders = build_replan_order(planning_state)

        task_indices = torch.full(
            (batch_size, num_satellites),
            -1,
            dtype=torch.long,
            device=satellite_mask.device,
        )
        commitment_indices = torch.full_like(task_indices, -1)
        action_order = torch.full_like(task_indices, -1)
        task_masks = torch.zeros(
            batch_size,
            num_satellites,
            num_tasks + 1,
            dtype=torch.bool,
            device=satellite_mask.device,
        )
        commitment_masks = torch.zeros(
            batch_size,
            num_satellites,
            len(COMMITMENT_SECONDS),
            dtype=torch.bool,
            device=satellite_mask.device,
        )
        owner_state = torch.zeros(
            batch_size,
            num_satellites,
            num_tasks,
            dtype=torch.long,
            device=satellite_mask.device,
        )
        scene_log_probs: list[torch.Tensor] = []
        scene_entropies: list[torch.Tensor] = []
        for batch_index, order in enumerate(orders):
            scene_log_prob = termination_log_prob[batch_index]
            scene_entropy = termination_entropy[batch_index]
            prefix_context = encoding.satellite_tokens.new_zeros(
                self.event_width,
            )
            current_owner_count = state.task_owner_count[
                batch_index
            ].clone().long()
            for satellite_id in order.tolist():
                current_task = int(state.current_task_indices[
                    batch_index,
                    satellite_id,
                ].item())
                if current_task >= 0:
                    if current_owner_count[current_task] <= 0:
                        raise ValueError(
                            'current task owner count is inconsistent'
                        )
                    current_owner_count[current_task] -= 1

            for position, satellite_id in enumerate(order.tolist()):
                action_order[batch_index, position] = satellite_id
                owner_state[batch_index, position] = current_owner_count
                logits, marginal, query = self._task_logits(
                    encoding,
                    batch_index,
                    satellite_id,
                    prefix_context,
                    current_owner_count,
                )
                legal_tasks = (
                    task_mask[batch_index]
                    & task_compatibility[batch_index, satellite_id]
                    & (current_owner_count < MAX_TASK_OWNERS)
                )
                if deterministic:
                    legal_tasks &= ~(
                        (current_owner_count > 0) & (marginal <= 0)
                    )
                legal = torch.cat((
                    torch.ones(1, dtype=torch.bool, device=legal_tasks.device),
                    legal_tasks,
                ))
                task_masks[batch_index, position] = legal
                task_distribution = self._masked_categorical(logits, legal)
                if deterministic:
                    category = task_distribution.logits.argmax()
                else:
                    category = task_distribution.sample()
                scene_log_prob = (
                    scene_log_prob + task_distribution.log_prob(category)
                )
                scene_entropy = scene_entropy + task_distribution.entropy()
                task_id = int(category.item()) - 1
                task_indices[batch_index, satellite_id] = task_id
                if task_id < 0:
                    prefix_context = self._updated_prefix(
                        prefix_context,
                        self.idle_context,
                        None,
                    )
                    continue

                commitment_mask = build_commitment_mask(
                    state.task_remaining_required_seconds[
                        batch_index, task_id
                    ].reshape(1),
                    torch.ones(1, dtype=torch.bool, device=legal.device),
                ).squeeze(0)
                commitment_masks[batch_index, position] = commitment_mask
                commitment_logits = self._commitment_logits(
                    encoding,
                    batch_index,
                    satellite_id,
                    task_id,
                    query,
                )
                commitment_distribution = self._masked_categorical(
                    commitment_logits,
                    commitment_mask,
                )
                if deterministic:
                    commitment_index = (
                        commitment_distribution.logits.argmax()
                    )
                else:
                    commitment_index = commitment_distribution.sample()
                scene_log_prob = (
                    scene_log_prob
                    + commitment_distribution.log_prob(commitment_index)
                )
                scene_entropy = (
                    scene_entropy + commitment_distribution.entropy()
                )
                commitment_id = int(commitment_index.item())
                commitment_indices[
                    batch_index,
                    satellite_id,
                ] = commitment_id
                current_owner_count[task_id] += 1
                prefix_context = self._updated_prefix(
                    prefix_context,
                    encoding.task_tokens[batch_index, task_id],
                    commitment_id,
                )
            scene_log_probs.append(scene_log_prob)
            scene_entropies.append(scene_entropy)

        return ActorOutput(
            action=JointEventAction(
                terminate=terminate,
                task_indices=task_indices,
                commitment_indices=commitment_indices,
            ),
            log_prob=torch.stack(scene_log_probs),
            entropy=torch.stack(scene_entropies),
            trace=ActionTrace(
                action_order=action_order,
                termination_mask=termination_mask,
                task_masks=task_masks,
                commitment_masks=commitment_masks,
                owner_state=owner_state,
            ),
        )

    def evaluate_actions(
        self,
        encoding: EventStateEncoding,
        state: EventStateTensors,
        satellite_mask: torch.Tensor,
        task_mask: torch.Tensor,
        action: JointEventAction,
        trace: ActionTrace,
    ) -> ActionEvaluation:
        batch_size, num_satellites, num_tasks = self._validate_inputs(
            encoding,
            state,
            satellite_mask,
            task_mask,
        )
        if action.terminate.shape != (batch_size, num_satellites):
            raise ValueError('termination action shape does not match actor')
        if action.terminate.dtype != torch.bool:
            raise ValueError('termination action must use bool dtype')
        if trace.termination_mask.shape != action.terminate.shape:
            raise ValueError('termination trace shape does not match action')
        if (action.terminate & ~trace.termination_mask).any():
            raise ValueError('termination action is outside behavior mask')
        expected_trace_shapes = (
            (trace.action_order, (batch_size, num_satellites)),
            (trace.task_masks, (batch_size, num_satellites, num_tasks + 1)),
            (
                trace.commitment_masks,
                (batch_size, num_satellites, len(COMMITMENT_SECONDS)),
            ),
            (trace.owner_state, (batch_size, num_satellites, num_tasks)),
        )
        if any(value.shape != shape for value, shape in expected_trace_shapes):
            raise ValueError('behavior trace shape does not match actor inputs')

        termination_distribution = self._termination_distribution(encoding)
        termination_log_prob = torch.where(
            trace.termination_mask,
            termination_distribution.log_prob(action.terminate.float()),
            0.,
        ).sum(dim=-1)
        termination_entropy = torch.where(
            trace.termination_mask,
            termination_distribution.entropy(),
            0.,
        ).sum(dim=-1)
        scene_log_probs: list[torch.Tensor] = []
        scene_entropies: list[torch.Tensor] = []
        for batch_index in range(batch_size):
            scene_log_prob = termination_log_prob[batch_index]
            scene_entropy = termination_entropy[batch_index]
            prefix_context = encoding.satellite_tokens.new_zeros(
                self.event_width,
            )
            active_order = trace.action_order[batch_index]
            active_order = active_order[active_order >= 0]
            if active_order.unique().numel() != active_order.numel():
                raise ValueError('behavior action order contains duplicates')
            for position, satellite_tensor in enumerate(active_order):
                satellite_id = int(satellite_tensor.item())
                owner_count = trace.owner_state[batch_index, position]
                logits, _, query = self._task_logits(
                    encoding,
                    batch_index,
                    satellite_id,
                    prefix_context,
                    owner_count,
                )
                legal = trace.task_masks[batch_index, position]
                task_distribution = self._masked_categorical(logits, legal)
                task_id = int(action.task_indices[
                    batch_index,
                    satellite_id,
                ].item())
                category = torch.tensor(
                    task_id + 1,
                    dtype=torch.long,
                    device=logits.device,
                )
                if category < 0 or category >= legal.numel() or not legal[category]:
                    raise ValueError('task action is outside behavior mask')
                scene_log_prob = (
                    scene_log_prob + task_distribution.log_prob(category)
                )
                scene_entropy = scene_entropy + task_distribution.entropy()
                if task_id < 0:
                    prefix_context = self._updated_prefix(
                        prefix_context,
                        self.idle_context,
                        None,
                    )
                    continue

                commitment_mask = trace.commitment_masks[
                    batch_index,
                    position,
                ]
                commitment_logits = self._commitment_logits(
                    encoding,
                    batch_index,
                    satellite_id,
                    task_id,
                    query,
                )
                commitment_distribution = self._masked_categorical(
                    commitment_logits,
                    commitment_mask,
                )
                commitment_index = action.commitment_indices[
                    batch_index,
                    satellite_id,
                ]
                if (
                    commitment_index < 0
                    or commitment_index >= commitment_mask.numel()
                    or not commitment_mask[commitment_index]
                ):
                    raise ValueError(
                        'commitment action is outside behavior mask'
                    )
                scene_log_prob = (
                    scene_log_prob
                    + commitment_distribution.log_prob(commitment_index)
                )
                scene_entropy = (
                    scene_entropy + commitment_distribution.entropy()
                )
                prefix_context = self._updated_prefix(
                    prefix_context,
                    encoding.task_tokens[batch_index, task_id],
                    int(commitment_index.item()),
                )
            scene_log_probs.append(scene_log_prob)
            scene_entropies.append(scene_entropy)
        return ActionEvaluation(
            log_prob=torch.stack(scene_log_probs),
            entropy=torch.stack(scene_entropies),
        )
