"""历史感知的轻量 Temporal Adapter 与带 censor mask 的损失。"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


__all__ = [
    'TemporalHistoryTensors',
    'TemporalAdapterOutput',
    'TemporalOutcomeLosses',
    'TemporalOutcomePositiveWeights',
    'TemporalAdapter',
    'masked_binary_cross_entropy',
    'temporal_outcome_loss',
]


class TemporalHistoryTensors(NamedTuple):
    previous_task_indices: torch.Tensor
    previous_task_available: torch.Tensor
    previous_was_idle: torch.Tensor
    run_lengths: torch.Tensor
    switch_count_30: torch.Tensor
    switch_count_60: torch.Tensor

    def validate(
        self,
        *,
        batch_size: int,
        num_satellites: int,
        num_tasks: int,
        check_values: bool = True,
    ) -> None:
        shape = (batch_size, num_satellites)
        for name, value in self._asdict().items():
            if value.shape != shape:
                raise ValueError(f'{name} must have shape {shape}')
        if not check_values:
            return
        available = self.previous_task_available.bool()
        indices = self.previous_task_indices
        invalid_available = available & ((indices < 0) | (indices >= num_tasks))
        if invalid_available.any():
            raise ValueError(
                'available previous_task_indices must reference current tasks'
            )
        if ((~available) & (indices != -1)).any():
            raise ValueError(
                'unavailable previous_task_indices must equal -1'
            )
        if (self.run_lengths < 0).any():
            raise ValueError('run_lengths must be non-negative')
        if (self.switch_count_30 < 0).any():
            raise ValueError('switch_count_30 must be non-negative')
        if (self.switch_count_60 < 0).any():
            raise ValueError('switch_count_60 must be non-negative')


class TemporalAdapterOutput(NamedTuple):
    null_delta: torch.Tensor
    task_delta: torch.Tensor
    visible_next_logits: torch.Tensor
    progress_next_logits: torch.Tensor
    completed_next_logits: torch.Tensor
    visible_logits: torch.Tensor
    progress_logits: torch.Tensor
    completed_logits: torch.Tensor
    time_to_first_visible: torch.Tensor
    time_to_first_progress: torch.Tensor
    time_to_completion: torch.Tensor


class TemporalOutcomeLosses(NamedTuple):
    visible: torch.Tensor
    progress: torch.Tensor
    completion: torch.Tensor
    event_time: torch.Tensor


class TemporalOutcomePositiveWeights(NamedTuple):
    """顺序为 next、随后各 horizon 的训练集负正样本比。"""

    visible: torch.Tensor
    progress: torch.Tensor
    completion: torch.Tensor


def masked_binary_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    observed: torch.Tensor,
    positive_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """只对事实已观测位置计算 BCE，全 censored 时返回可反传零。"""
    if logits.shape != targets.shape or logits.shape != observed.shape:
        raise ValueError('logits, targets and observed must share a shape')
    observed = observed.bool()
    if not observed.any():
        return logits.sum() * 0.
    losses = F.binary_cross_entropy_with_logits(
        logits,
        targets.to(dtype=logits.dtype),
        reduction='none',
        pos_weight=(
            None
            if positive_weight is None
            else positive_weight.to(
                device=logits.device,
                dtype=logits.dtype,
            )
        ),
    )
    return losses[observed].mean()


def _gather_executed_edges(
    values: torch.Tensor,
    actions_task_id: torch.Tensor,
) -> torch.Tensor:
    if values.shape[:2] != actions_task_id.shape:
        raise ValueError('outcome logits and actions must share batch/satellites')
    task_indices = actions_task_id.clamp_min(0)
    if values.ndim == 3:
        return values.gather(2, task_indices.unsqueeze(-1)).squeeze(-1)
    if values.ndim == 4:
        index = task_indices.unsqueeze(-1).unsqueeze(-1).expand(
            -1,
            -1,
            1,
            values.shape[-1],
        )
        return values.gather(2, index).squeeze(2)
    raise ValueError('outcome logits must have rank 3 or 4')


def _classification_outcome_loss(
    *,
    next_logits: torch.Tensor,
    horizon_logits: torch.Tensor,
    next_targets: torch.Tensor,
    horizon_targets: torch.Tensor,
    horizon_observed: torch.Tensor,
    valid: torch.Tensor,
    actions_task_id: torch.Tensor,
    positive_weights: torch.Tensor | None,
) -> torch.Tensor:
    executed_next = _gather_executed_edges(next_logits, actions_task_id)
    executed_horizons = _gather_executed_edges(
        horizon_logits,
        actions_task_id,
    )
    next_loss = masked_binary_cross_entropy(
        executed_next,
        next_targets,
        valid,
        (
            None
            if positive_weights is None
            else positive_weights[0]
        ),
    )
    horizon_loss = masked_binary_cross_entropy(
        executed_horizons,
        horizon_targets,
        valid.unsqueeze(-1) & horizon_observed,
        (
            None
            if positive_weights is None
            else positive_weights[1:]
        ),
    )
    return next_loss + horizon_loss


def temporal_outcome_loss(
    output: TemporalAdapterOutput,
    targets,
    actions_task_id: torch.Tensor,
    positive_weights: TemporalOutcomePositiveWeights | None = None,
) -> TemporalOutcomeLosses:
    """计算实际执行边的 pointwise 结果损失，censored 不作负样本。"""
    if actions_task_id.ndim != 2:
        raise ValueError('actions_task_id must have shape (batch, satellites)')
    num_tasks = output.visible_next_logits.shape[2]
    selected = actions_task_id[actions_task_id >= 0]
    if selected.numel() and int(selected.max()) >= num_tasks:
        raise ValueError('actions_task_id exceeds temporal task logits')
    valid = targets.outcome_valid.bool() & (actions_task_id >= 0)
    if positive_weights is not None:
        expected_shape = (1 + output.visible_logits.shape[-1],)
        normalized = []
        for name, values in positive_weights._asdict().items():
            values = values.to(
                device=output.visible_logits.device,
                dtype=output.visible_logits.dtype,
            )
            if values.shape != expected_shape:
                raise ValueError(
                    f'{name} positive weights must have shape '
                    f'{expected_shape}'
                )
            if not torch.isfinite(values).all() or (values <= 0).any():
                raise ValueError('positive weights must be finite and positive')
            normalized.append(values)
        positive_weights = TemporalOutcomePositiveWeights(*normalized)

    visible = _classification_outcome_loss(
        next_logits=output.visible_next_logits,
        horizon_logits=output.visible_logits,
        next_targets=targets.visible_next,
        horizon_targets=targets.visible,
        horizon_observed=targets.visible_observed,
        valid=valid,
        actions_task_id=actions_task_id,
        positive_weights=(
            None if positive_weights is None else positive_weights.visible
        ),
    )
    progress = _classification_outcome_loss(
        next_logits=output.progress_next_logits,
        horizon_logits=output.progress_logits,
        next_targets=targets.progress_next,
        horizon_targets=targets.progress,
        horizon_observed=targets.progress_observed,
        valid=valid,
        actions_task_id=actions_task_id,
        positive_weights=(
            None if positive_weights is None else positive_weights.progress
        ),
    )
    completion = _classification_outcome_loss(
        next_logits=output.completed_next_logits,
        horizon_logits=output.completed_logits,
        next_targets=targets.completed_next,
        horizon_targets=targets.completed,
        horizon_observed=targets.completion_observed,
        valid=valid,
        actions_task_id=actions_task_id,
        positive_weights=(
            None if positive_weights is None else positive_weights.completion
        ),
    )

    horizons = targets.horizons
    if horizons.ndim == 2:
        if not torch.equal(horizons, horizons[:1].expand_as(horizons)):
            raise ValueError('all samples must use the same temporal horizons')
        horizons = horizons[0]
    if horizons.ndim != 1:
        raise ValueError('horizons must have shape (num_horizons,)')
    if horizons.numel() != output.visible_logits.shape[-1]:
        raise ValueError('target horizons do not match temporal adapter')
    horizon_scale = horizons.to(
        device=output.visible_logits.device,
        dtype=output.visible_logits.dtype,
    ).view(1, 1, -1)

    event_predictions = []
    event_targets = []
    event_observed = []
    for prediction, target in (
        (output.time_to_first_visible, targets.time_to_first_visible),
        (output.time_to_first_progress, targets.time_to_first_progress),
        (output.time_to_completion, targets.time_to_completion),
    ):
        executed = _gather_executed_edges(prediction, actions_task_id)
        event_predictions.append(torch.sigmoid(executed))
        event_targets.append(
            target.to(dtype=executed.dtype) / horizon_scale
        )
        event_observed.append(valid.unsqueeze(-1) & (target > 0))
    flat_predictions = torch.cat(
        [value.reshape(-1) for value in event_predictions]
    )
    flat_targets = torch.cat([value.reshape(-1) for value in event_targets])
    flat_observed = torch.cat([
        value.reshape(-1) for value in event_observed
    ])
    if flat_observed.any():
        event_time = F.smooth_l1_loss(
            flat_predictions[flat_observed],
            flat_targets[flat_observed],
        )
    else:
        event_time = flat_predictions.sum() * 0.

    return TemporalOutcomeLosses(
        visible=visible,
        progress=progress,
        completion=completion,
        event_time=event_time,
    )


class TemporalAdapter(nn.Module):
    """基于历史的任务边/空动作软残差与 pointwise outcome heads。"""

    def __init__(
        self,
        *args,
        satellite_width: int,
        task_width: int,
        hidden_width: int = 64,
        horizons: tuple[int, ...] = (5, 15, 30, 300),
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.horizons = tuple(int(value) for value in horizons)
        if (
            not self.horizons
            or any(value <= 0 for value in self.horizons)
            or len(set(self.horizons)) != len(self.horizons)
        ):
            raise ValueError('horizons must be unique and positive')

        edge_width = satellite_width + task_width + 7
        null_width = satellite_width + 6
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_width, hidden_width),
            nn.GELU(),
            nn.LayerNorm(hidden_width),
        )
        self.null_mlp = nn.Sequential(
            nn.Linear(null_width, hidden_width),
            nn.GELU(),
            nn.LayerNorm(hidden_width),
        )
        self.task_residual = nn.Linear(hidden_width, 1)
        self.null_residual = nn.Linear(hidden_width, 1)
        self.outcome_head = nn.Linear(
            hidden_width,
            3 + 6 * len(self.horizons),
        )
        nn.init.zeros_(self.task_residual.weight)
        nn.init.zeros_(self.task_residual.bias)
        nn.init.zeros_(self.null_residual.weight)
        nn.init.zeros_(self.null_residual.bias)

    @staticmethod
    def _history_features(
        history: TemporalHistoryTensors,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        run_lengths = torch.log1p(
            history.run_lengths.to(dtype=dtype).clamp_max(300)
        ) / math.log1p(300)
        return torch.stack((
            run_lengths,
            history.switch_count_30.to(dtype=dtype) / 30.,
            history.switch_count_60.to(dtype=dtype) / 60.,
            history.previous_was_idle.to(dtype=dtype),
            history.previous_task_available.to(dtype=dtype),
        ), -1)

    @staticmethod
    def _previous_task_match(
        history: TemporalHistoryTensors,
        num_tasks: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        matches = F.one_hot(
            history.previous_task_indices.clamp_min(0).long(),
            num_classes=num_tasks,
        ).to(dtype=dtype)
        return matches * history.previous_task_available.unsqueeze(-1)

    def forward(
        self,
        *,
        satellite_features: torch.Tensor,
        task_features: torch.Tensor,
        null_logits: torch.Tensor,
        task_logits: torch.Tensor,
        satellite_mask: torch.Tensor,
        task_mask: torch.Tensor,
        history: TemporalHistoryTensors,
    ) -> TemporalAdapterOutput:
        if satellite_features.ndim != 3 or task_features.ndim != 3:
            raise ValueError('satellite and task features must be rank 3')
        batch_size, num_satellites, satellite_width = (
            satellite_features.shape
        )
        task_batch, num_tasks, task_width = task_features.shape
        if task_batch != batch_size:
            raise ValueError('satellite and task features must share a batch')
        if null_logits.shape != (batch_size, num_satellites):
            raise ValueError('null_logits shape does not match satellites')
        if task_logits.shape != (batch_size, num_satellites, num_tasks):
            raise ValueError('task_logits shape does not match features')
        if satellite_mask.shape != (batch_size, num_satellites):
            raise ValueError('satellite_mask shape does not match features')
        if task_mask.shape != (batch_size, num_tasks):
            raise ValueError('task_mask shape does not match features')
        history.validate(
            batch_size=batch_size,
            num_satellites=num_satellites,
            num_tasks=num_tasks,
            check_values=not task_logits.is_cuda,
        )

        dtype = satellite_features.dtype
        history_features = self._history_features(history, dtype=dtype)
        task_matches = self._previous_task_match(
            history,
            num_tasks,
            dtype,
        )
        satellite_edges = satellite_features.unsqueeze(2).expand(
            -1, -1, num_tasks, satellite_width
        )
        task_edges = task_features.unsqueeze(1).expand(
            -1, num_satellites, -1, task_width
        )
        common_edges = history_features.unsqueeze(2).expand(
            -1, -1, num_tasks, -1
        )
        edge_features = torch.cat((
            satellite_edges,
            task_edges,
            task_logits.unsqueeze(-1),
            task_matches.unsqueeze(-1),
            common_edges,
        ), -1)
        edge_hidden = self.edge_mlp(edge_features)

        valid_edges = satellite_mask.unsqueeze(-1) & task_mask.unsqueeze(1)
        task_delta = self.task_residual(edge_hidden).squeeze(-1)
        task_delta = task_delta.masked_fill(~valid_edges, 0.)

        null_features = torch.cat((
            satellite_features,
            null_logits.unsqueeze(-1),
            history_features,
        ), -1)
        null_hidden = self.null_mlp(null_features)
        null_delta = self.null_residual(null_hidden).squeeze(-1)
        null_delta = null_delta.masked_fill(~satellite_mask, 0.)

        outcome = self.outcome_head(edge_hidden)
        outcome = outcome.masked_fill(~valid_edges.unsqueeze(-1), 0.)
        num_horizons = len(self.horizons)
        next_logits = outcome[..., :3]
        horizon_logits = outcome[..., 3:3 + 3 * num_horizons]
        horizon_logits = horizon_logits.reshape(
            batch_size,
            num_satellites,
            num_tasks,
            3,
            num_horizons,
        )
        event_times = outcome[..., 3 + 3 * num_horizons:]
        event_times = event_times.reshape(
            batch_size,
            num_satellites,
            num_tasks,
            3,
            num_horizons,
        )
        return TemporalAdapterOutput(
            null_delta=null_delta,
            task_delta=task_delta,
            visible_next_logits=next_logits[..., 0],
            progress_next_logits=next_logits[..., 1],
            completed_next_logits=next_logits[..., 2],
            visible_logits=horizon_logits[..., 0, :],
            progress_logits=horizon_logits[..., 1, :],
            completed_logits=horizon_logits[..., 2, :],
            time_to_first_visible=event_times[..., 0, :],
            time_to_first_progress=event_times[..., 1, :],
            time_to_completion=event_times[..., 2, :],
        )
