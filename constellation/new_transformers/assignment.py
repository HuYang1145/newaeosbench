"""轻量二部图联合分配头及其训练辅助损失。"""

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


class AssignmentLosses(NamedTuple):
    collision: torch.Tensor
    coverage: torch.Tensor


class BipartiteAssignmentHead(nn.Module):
    """通过卫星和任务间的软消息传递，对原始任务 logits 做残差修正。"""

    def __init__(
        self,
        *,
        satellite_width: int,
        task_width: int,
        hidden_width: int = 32,
    ) -> None:
        super().__init__()
        self.satellite_projection = nn.Linear(satellite_width, hidden_width)
        self.task_projection = nn.Linear(task_width, hidden_width)
        self.satellite_message_projection = nn.Linear(
            hidden_width,
            hidden_width,
        )
        self.task_message_projection = nn.Linear(hidden_width, hidden_width)
        self.demand_projection = nn.Linear(1, hidden_width)
        self.edge_norm = nn.LayerNorm(hidden_width)
        self.residual_score = nn.Linear(hidden_width, 1)

        # 新头启用但尚未训练时，输出必须与原 checkpoint 完全一致。
        nn.init.zeros_(self.residual_score.weight)
        nn.init.zeros_(self.residual_score.bias)

    def forward(
        self,
        null_logits: torch.Tensor,
        task_logits: torch.Tensor,
        satellite_features: torch.Tensor,
        task_features: torch.Tensor,
        satellite_mask: torch.Tensor,
        task_mask: torch.Tensor,
    ) -> torch.Tensor:
        masked_task_logits = task_logits.masked_fill(
            ~task_mask[:, None, :],
            float('-inf'),
        )
        probabilities = torch.cat((null_logits, masked_task_logits), -1)
        probabilities = probabilities.softmax(-1)[..., 1:]
        valid_edges = satellite_mask[:, :, None] & task_mask[:, None, :]
        probabilities = probabilities * valid_edges

        satellite_nodes = self.satellite_projection(satellite_features)
        task_nodes = self.task_projection(task_features)

        task_demand = probabilities.sum(1)
        task_messages = torch.einsum(
            'bst,bsh->bth',
            probabilities,
            satellite_nodes,
        ) / task_demand.unsqueeze(-1).clamp_min(1e-6)
        satellite_demand = probabilities.sum(-1)
        satellite_messages = torch.einsum(
            'bst,bth->bsh',
            probabilities,
            task_nodes,
        ) / satellite_demand.unsqueeze(-1).clamp_min(1e-6)

        edge_features = (
            satellite_nodes[:, :, None, :]
            + task_nodes[:, None, :, :]
            + self.satellite_message_projection(
                satellite_messages,
            )[:, :, None, :]
            + self.task_message_projection(task_messages)[:, None, :, :]
            + self.demand_projection(
                task_demand.log1p().unsqueeze(-1),
            )[:, None, :, :]
        )
        residual = self.residual_score(
            F.gelu(self.edge_norm(edge_features)),
        ).squeeze(-1)
        residual = torch.where(valid_edges, residual, 0.)
        return task_logits + residual


class AssignmentAuxiliaryLoss(nn.Module):
    """减少软分配冲突，并保持专家动作覆盖的可微辅助损失。"""

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        satellite_mask: torch.Tensor,
        task_mask: torch.Tensor,
    ) -> AssignmentLosses:
        task_logits = logits[..., 1:].masked_fill(
            ~task_mask[:, None, :],
            float('-inf'),
        )
        probabilities = torch.cat((logits[..., :1], task_logits), -1)
        probabilities = probabilities.softmax(-1)[..., 1:]
        valid_edges = satellite_mask[:, :, None] & task_mask[:, None, :]
        probabilities = probabilities * valid_edges

        expected_count = probabilities.sum(1)
        excess = (expected_count - 1.).relu()
        bounded_excess = excess / (1. + excess)
        collision_per_task = bounded_excess.square() * task_mask
        collision = collision_per_task.sum() / task_mask.sum().clamp_min(1)

        num_tasks = task_mask.shape[-1]
        valid_targets = (
            satellite_mask
            & (targets >= 0)
            & (targets < num_tasks)
        )
        target_one_hot = F.one_hot(
            targets.clamp(0, max(num_tasks - 1, 0)),
            num_classes=num_tasks,
        ).bool()
        target_task_mask = (
            target_one_hot & valid_targets.unsqueeze(-1)
        ).any(1)
        target_task_mask &= task_mask

        coverage_probability = 1. - (1. - probabilities).prod(1)
        coverage_error = (1. - coverage_probability) * target_task_mask
        coverage = (
            coverage_error.sum() / target_task_mask.sum().clamp_min(1)
        )
        return AssignmentLosses(collision=collision, coverage=coverage)
