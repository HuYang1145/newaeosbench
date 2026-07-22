"""显式事件状态编码与 centralized state-value Critic。"""

import math
from typing import NamedTuple

import torch
from torch import nn

from .backbone import Stage3BackboneOutput
from .state import MAX_TASK_OWNERS, EventStateTensors


class EventStateEncoding(NamedTuple):
    satellite_tokens: torch.Tensor
    task_tokens: torch.Tensor
    edge_tokens: torch.Tensor


class EventStateEncoder(nn.Module):
    """把可部署事件统计注入 Stage3 token。

    previous/current task 只转换成当前候选边上的 boolean relation，不建立跨场景
    task-id embedding。
    """

    def __init__(
        self,
        *,
        satellite_width: int,
        task_width: int,
        edge_width: int,
        event_width: int,
        num_termination_reasons: int = 8,
        num_event_types: int = 8,
    ) -> None:
        super().__init__()
        widths = (satellite_width, task_width, edge_width, event_width)
        if any(width <= 0 for width in widths):
            raise ValueError('all event encoder widths must be positive')
        if num_termination_reasons <= 0 or num_event_types <= 0:
            raise ValueError('event category counts must be positive')
        category_width = max(2, event_width // 8)
        self.num_termination_reasons = num_termination_reasons
        self.num_event_types = num_event_types
        self.termination_reason_embedding = nn.Embedding(
            num_termination_reasons,
            category_width,
        )
        self.event_type_embedding = nn.Embedding(
            num_event_types,
            category_width,
        )
        self.satellite_base_projection = nn.Linear(
            satellite_width,
            event_width,
        )
        self.task_base_projection = nn.Linear(task_width, event_width)
        self.edge_base_projection = nn.Linear(edge_width, event_width)
        self.satellite_state_projection = nn.Sequential(
            nn.Linear(9 + 2 * category_width, event_width),
            nn.GELU(),
            nn.Linear(event_width, event_width),
        )
        self.task_state_projection = nn.Sequential(
            nn.Linear(3, event_width),
            nn.GELU(),
            nn.Linear(event_width, event_width),
        )
        self.edge_relation_projection = nn.Linear(2, event_width)
        self.satellite_norm = nn.LayerNorm(event_width)
        self.task_norm = nn.LayerNorm(event_width)
        self.edge_norm = nn.LayerNorm(event_width)

    @staticmethod
    def _log_time(value: torch.Tensor) -> torch.Tensor:
        return torch.log1p(value.float()) / math.log1p(3600.0)

    def forward(
        self,
        backbone_output: Stage3BackboneOutput,
        state: EventStateTensors,
        satellite_mask: torch.Tensor,
        task_mask: torch.Tensor,
    ) -> EventStateEncoding:
        state.validate()
        batch_size, num_satellites, _ = (
            backbone_output.satellite_tokens.shape
        )
        task_batch, num_tasks, _ = backbone_output.task_tokens.shape
        if task_batch != batch_size:
            raise ValueError('task and satellite tokens must share batch size')
        if state.replan_mask.shape != (batch_size, num_satellites):
            raise ValueError('event state does not match satellite tokens')
        if state.task_owner_count.shape != (batch_size, num_tasks):
            raise ValueError('event state does not match task tokens')
        if satellite_mask.shape != (batch_size, num_satellites):
            raise ValueError('satellite mask has invalid shape')
        if task_mask.shape != (batch_size, num_tasks):
            raise ValueError('task mask has invalid shape')
        if satellite_mask.dtype != torch.bool or task_mask.dtype != torch.bool:
            raise ValueError('event token masks must use bool dtype')
        if backbone_output.edge_features.shape[:3] != (
            batch_size,
            num_satellites,
            num_tasks,
        ):
            raise ValueError('edge features do not match satellite-task axes')
        if (
            (state.termination_reason >= self.num_termination_reasons).any()
            or (state.event_type >= self.num_event_types).any()
        ):
            raise ValueError('event category index is outside embedding range')

        reason_embedding = self.termination_reason_embedding(
            state.termination_reason.long(),
        )
        event_embedding = self.event_type_embedding(
            state.event_type.long(),
        )
        satellite_numeric = torch.stack((
            state.minimum_commitment_remaining.float() / 60.0,
            self._log_time(state.run_lengths),
            self._log_time(state.seconds_since_replan),
            state.switch_count_30.float() / 30.0,
            state.switch_count_60.float() / 60.0,
            state.delta_t.float() / 60.0,
            state.can_terminate_mask.float(),
            state.replan_mask.float(),
            state.forced_interrupt_mask.float(),
        ), dim=-1)
        satellite_state = torch.cat((
            satellite_numeric,
            reason_embedding,
            event_embedding,
        ), dim=-1)
        satellite_tokens = self.satellite_norm(
            self.satellite_base_projection(backbone_output.satellite_tokens)
            + self.satellite_state_projection(satellite_state)
        )

        task_state = torch.stack((
            self._log_time(state.task_remaining_required_seconds),
            state.task_owner_count.float() / MAX_TASK_OWNERS,
            state.task_locked_owner_count.float() / MAX_TASK_OWNERS,
        ), dim=-1)
        task_tokens = self.task_norm(
            self.task_base_projection(backbone_output.task_tokens)
            + self.task_state_projection(task_state)
        )

        task_indices = torch.arange(
            num_tasks,
            device=state.previous_task_indices.device,
        ).view(1, 1, num_tasks)
        previous_relation = (
            state.previous_task_indices.unsqueeze(-1) == task_indices
        )
        current_relation = (
            state.current_task_indices.unsqueeze(-1) == task_indices
        )
        relation = torch.stack((
            previous_relation,
            current_relation,
        ), dim=-1).to(backbone_output.edge_features.dtype)
        edge_tokens = self.edge_norm(
            self.edge_base_projection(backbone_output.edge_features)
            + self.edge_relation_projection(relation)
        )
        return EventStateEncoding(
            satellite_tokens=satellite_tokens,
            task_tokens=task_tokens,
            edge_tokens=edge_tokens,
        )


class CentralizedValueCritic(nn.Module):
    """聚合完整轻量星座状态，输出每个 event state 的标量价值。"""

    def __init__(self, *, event_width: int) -> None:
        super().__init__()
        if event_width <= 0:
            raise ValueError('event_width must be positive')
        self.event_width = event_width
        self.value_head = nn.Sequential(
            nn.LayerNorm(6 * event_width),
            nn.Linear(6 * event_width, 2 * event_width),
            nn.GELU(),
            nn.Linear(2 * event_width, 1),
        )

    @staticmethod
    def _masked_mean_max(
        tokens: torch.Tensor,
        mask: torch.Tensor,
        *,
        entity_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.shape[:-1] != mask.shape:
            raise ValueError(f'{entity_name} mask does not match tokens')
        if mask.dtype != torch.bool:
            raise ValueError(f'{entity_name} mask must use bool dtype')
        batch_size = tokens.shape[0]
        flat_tokens = tokens.reshape(batch_size, -1, tokens.shape[-1])
        flat_mask = mask.reshape(batch_size, -1)
        if not flat_mask.any(dim=1).all():
            raise ValueError(f'each scene needs a valid {entity_name}')
        expanded_mask = flat_mask.unsqueeze(-1)
        mean = (flat_tokens * expanded_mask).sum(dim=1) / (
            expanded_mask.sum(dim=1).clamp_min(1)
        )
        maximum = flat_tokens.masked_fill(
            ~expanded_mask,
            float('-inf'),
        ).max(dim=1).values
        return mean, maximum

    def forward(
        self,
        encoding: EventStateEncoding,
        satellite_mask: torch.Tensor,
        task_mask: torch.Tensor,
    ) -> torch.Tensor:
        if encoding.satellite_tokens.shape[-1] != self.event_width:
            raise ValueError('satellite token width does not match critic')
        if encoding.task_tokens.shape[-1] != self.event_width:
            raise ValueError('task token width does not match critic')
        if encoding.edge_tokens.shape[-1] != self.event_width:
            raise ValueError('edge token width does not match critic')
        satellite_mean, satellite_max = self._masked_mean_max(
            encoding.satellite_tokens,
            satellite_mask,
            entity_name='satellite',
        )
        task_mean, task_max = self._masked_mean_max(
            encoding.task_tokens,
            task_mask,
            entity_name='task',
        )
        edge_mask = satellite_mask.unsqueeze(2) & task_mask.unsqueeze(1)
        edge_mean, edge_max = self._masked_mean_max(
            encoding.edge_tokens,
            edge_mask,
            entity_name='satellite-task edge',
        )
        pooled = torch.cat((
            satellite_mean,
            satellite_max,
            task_mean,
            task_max,
            edge_mean,
            edge_max,
        ), dim=-1)
        return self.value_head(pooled).squeeze(-1)
