"""V2-0 旧轨迹 warm start 损失。"""

import math
from typing import NamedTuple

import torch
from torch.nn import functional as F

from .dataset import OfflineEventBatch
from .model import EventJointActorCritic


class OfflineLosses(NamedTuple):
    total: torch.Tensor
    task_distillation: torch.Tensor
    termination: torch.Tensor
    commitment: torch.Tensor
    value: torch.Tensor


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if value.shape != mask.shape:
        raise ValueError('masked loss value and mask must share shape')
    if mask.dtype != torch.bool:
        raise ValueError('loss mask must use bool dtype')
    if mask.any():
        return value[mask].mean()
    return value.sum() * 0.


def event_v2_offline_loss(
    model: EventJointActorCritic,
    batch: OfflineEventBatch,
    *,
    task_weight: float = 1.0,
    termination_weight: float = 1.0,
    commitment_weight: float = 1.0,
    value_weight: float = 1.0,
) -> OfflineLosses:
    """蒸馏基础任务价值并训练事实 termination/commitment/value。"""

    weights = (
        task_weight,
        termination_weight,
        commitment_weight,
        value_weight,
    )
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError('offline loss weights must be finite and non-negative')
    stage3 = batch.stage3_batch
    targets = batch.targets
    backbone_output = model.backbone(
        stage3.time_steps,
        stage3.constellation_sensor_type,
        stage3.constellation_sensor_enabled,
        stage3.constellation_data,
        stage3.constellation_mask,
        stage3.tasks_sensor_type,
        stage3.tasks_data,
        stage3.tasks_mask,
    )
    encoding = model.state_encoder(
        backbone_output,
        batch.event_state,
        stage3.constellation_mask,
        stage3.tasks_mask,
    )
    batch_size, num_satellites, num_tasks = (
        encoding.edge_tokens.shape[:3]
    )
    satellite_shape = (batch_size, num_satellites)
    for name in (
        'termination',
        'termination_observed',
        'task_indices',
        'task_observed',
        'commitment_indices',
        'commitment_observed',
    ):
        if getattr(targets, name).shape != satellite_shape:
            raise ValueError(f'offline target {name} has invalid shape')
    if targets.value_returns.shape != (batch_size,):
        raise ValueError('offline value return has invalid shape')

    query = encoding.satellite_tokens
    student_task_logits = (
        torch.einsum('bsd,btd->bst', query, encoding.task_tokens)
        / math.sqrt(model.actor.event_width)
        + model.actor.task_value_head(encoding.edge_tokens).squeeze(-1)
    )
    student_logits = torch.cat((
        model.actor.idle_head(query),
        student_task_logits,
    ), dim=-1)
    teacher_logits = torch.cat((
        backbone_output.teacher_null_logits.unsqueeze(-1),
        backbone_output.teacher_task_logits,
    ), dim=-1).detach()
    categorical_mask = torch.cat((
        torch.ones(
            batch_size,
            1,
            dtype=torch.bool,
            device=stage3.tasks_mask.device,
        ),
        stage3.tasks_mask,
    ), dim=-1).unsqueeze(1).expand(-1, num_satellites, -1)
    student_logits = student_logits.masked_fill(~categorical_mask, -1e4)
    teacher_logits = teacher_logits.masked_fill(~categorical_mask, -1e4)
    task_kl = F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        reduction='none',
    ).sum(dim=-1)
    task_distillation_loss = _masked_mean(
        task_kl,
        targets.task_observed & stage3.constellation_mask,
    )

    termination_logits = model.actor.termination_head(query).squeeze(-1)
    termination_elementwise = F.binary_cross_entropy_with_logits(
        termination_logits,
        targets.termination.float(),
        reduction='none',
    )
    termination_loss = _masked_mean(
        termination_elementwise,
        targets.termination_observed & stage3.constellation_mask,
    )

    commitment_terms: list[torch.Tensor] = []
    observed_commitment = (
        targets.commitment_observed & stage3.constellation_mask
    )
    for batch_index, satellite_id in observed_commitment.nonzero().tolist():
        task_id = int(targets.task_indices[
            batch_index,
            satellite_id,
        ].item())
        if not 0 <= task_id < num_tasks:
            raise ValueError('observed commitment has no valid selected task')
        logits = model.actor._commitment_logits(
            encoding,
            batch_index,
            satellite_id,
            task_id,
            query[batch_index, satellite_id],
        ).unsqueeze(0)
        target = targets.commitment_indices[
            batch_index,
            satellite_id,
        ].reshape(1)
        commitment_terms.append(F.cross_entropy(logits, target))
    if commitment_terms:
        commitment_loss = torch.stack(commitment_terms).mean()
    else:
        commitment_loss = encoding.edge_tokens.sum() * 0.

    predicted_value = model.critic(
        encoding,
        stage3.constellation_mask,
        stage3.tasks_mask,
    )
    value_loss = F.smooth_l1_loss(
        predicted_value,
        targets.value_returns.float(),
    )
    total = (
        task_weight * task_distillation_loss
        + termination_weight * termination_loss
        + commitment_weight * commitment_loss
        + value_weight * value_loss
    )
    return OfflineLosses(
        total=total,
        task_distillation=task_distillation_loss,
        termination=termination_loss,
        commitment=commitment_loss,
        value=value_loss,
    )
