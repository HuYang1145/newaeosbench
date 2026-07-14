"""第一分歧点上的轻量卫星—任务二部图裁判模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GraphQSceneContext:
    """不含可见性标签的静态场景张量。"""

    task_ids: torch.Tensor
    task_durations: torch.Tensor
    task_static_data: torch.Tensor
    task_sensor_type: torch.Tensor
    constellation_static_data: torch.Tensor
    constellation_sensor_type: torch.Tensor


@dataclass(frozen=True)
class GraphQSample:
    """共享决策前状态上的一对精确联合动作。"""

    scene_id: int
    satellite_features: torch.Tensor
    task_features: torch.Tensor
    compatibility: torch.Tensor
    previous_action: torch.Tensor
    better_action: torch.Tensor
    worse_action: torch.Tensor
    better_summary: torch.Tensor
    worse_summary: torch.Tensor
    margin: float
    better_candidate: str
    worse_candidate: str
    better_cost: float
    worse_cost: float


class GraphQBatch(NamedTuple):
    scene_ids: torch.Tensor
    satellite_features: torch.Tensor
    task_features: torch.Tensor
    compatibility: torch.Tensor
    satellite_mask: torch.Tensor
    task_mask: torch.Tensor
    previous_action: torch.Tensor
    better_action: torch.Tensor
    worse_action: torch.Tensor
    better_summary: torch.Tensor
    worse_summary: torch.Tensor
    margins: torch.Tensor


def collate_graph_q_samples(samples: Sequence[GraphQSample]) -> GraphQBatch:
    """把不同卫星数、任务数的图补齐为一个 batch。"""

    if not samples:
        raise ValueError('at least one Graph-Q sample is required')
    satellite_dim = samples[0].satellite_features.shape[-1]
    task_dim = samples[0].task_features.shape[-1]
    summary_dim = samples[0].better_summary.numel()
    max_satellites = max(item.satellite_features.shape[0] for item in samples)
    max_tasks = max(item.task_features.shape[0] for item in samples)
    batch_size = len(samples)
    satellite_features = torch.zeros(
        batch_size,
        max_satellites,
        satellite_dim,
    )
    task_features = torch.zeros(batch_size, max_tasks, task_dim)
    compatibility = torch.zeros(batch_size, max_satellites, max_tasks)
    satellite_mask = torch.zeros(batch_size, max_satellites, dtype=torch.bool)
    task_mask = torch.zeros(batch_size, max_tasks, dtype=torch.bool)
    better_action = torch.full(
        (batch_size, max_satellites),
        -1,
        dtype=torch.long,
    )
    previous_action = better_action.clone()
    worse_action = better_action.clone()
    better_summary = torch.zeros(batch_size, summary_dim)
    worse_summary = torch.zeros(batch_size, summary_dim)
    for index, item in enumerate(samples):
        num_satellites = item.satellite_features.shape[0]
        num_tasks = item.task_features.shape[0]
        if item.satellite_features.shape[-1] != satellite_dim:
            raise ValueError('satellite feature dimensions must be equal')
        if item.task_features.shape[-1] != task_dim:
            raise ValueError('task feature dimensions must be equal')
        if item.compatibility.shape != (num_satellites, num_tasks):
            raise ValueError('compatibility must match the bipartite graph')
        if item.better_action.shape != (num_satellites, ):
            raise ValueError('better action must have one entry per satellite')
        if item.worse_action.shape != (num_satellites, ):
            raise ValueError('worse action must have one entry per satellite')
        if item.previous_action.shape != (num_satellites, ):
            raise ValueError(
                'previous action must have one entry per satellite'
            )
        satellite_features[
            index, :num_satellites] = (item.satellite_features.float())
        task_features[index, :num_tasks] = item.task_features.float()
        compatibility[
            index, :num_satellites, :num_tasks] = (item.compatibility.float())
        satellite_mask[index, :num_satellites] = True
        task_mask[index, :num_tasks] = True
        previous_action[index, :num_satellites] = item.previous_action.long()
        better_action[index, :num_satellites] = item.better_action.long()
        worse_action[index, :num_satellites] = item.worse_action.long()
        better_summary[index] = item.better_summary.float()
        worse_summary[index] = item.worse_summary.float()
    return GraphQBatch(
        scene_ids=torch.tensor([item.scene_id for item in samples]),
        satellite_features=satellite_features,
        task_features=task_features,
        compatibility=compatibility,
        satellite_mask=satellite_mask,
        task_mask=task_mask,
        previous_action=previous_action,
        better_action=better_action,
        worse_action=worse_action,
        better_summary=better_summary,
        worse_summary=worse_summary,
        margins=torch.tensor([item.margin for item in samples]).float(),
    )


class GraphQCritic(nn.Module):
    """按任务聚合已分配卫星，保留具体卫星—任务对应关系。"""

    def __init__(
        self,
        *,
        satellite_dim: int,
        task_dim: int,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.satellite_encoder = nn.Sequential(
            nn.Linear(satellite_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.task_encoder = nn.Sequential(
            nn.Linear(task_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.satellite_message = nn.Linear(hidden_dim, hidden_dim)
        self.demand_projection = nn.Linear(1, hidden_dim)
        self.compatibility_projection = nn.Linear(1, hidden_dim)
        self.continuity_projection = nn.Linear(1, hidden_dim)
        self.task_score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.null_score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.action_score = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def score_batch(
        self,
        batch: GraphQBatch,
        action: torch.Tensor,
    ) -> torch.Tensor:
        satellite_nodes = self.satellite_encoder(batch.satellite_features)
        task_nodes = self.task_encoder(batch.task_features)
        num_tasks = batch.task_features.shape[1]
        safe_action = action.clamp(0, max(num_tasks - 1, 0))
        selected = (
            batch.satellite_mask
            & (action >= 0)
            & (action < num_tasks)
        )
        selected_task_valid = batch.task_mask.gather(1, safe_action)
        selected &= selected_task_valid
        assignment = F.one_hot(
            safe_action,
            num_classes=num_tasks,
        ).float() * selected.unsqueeze(-1)

        demand = assignment.sum(1)
        satellite_messages = torch.einsum(
            'bst,bsh->bth',
            assignment,
            satellite_nodes,
        ) / demand.unsqueeze(-1).clamp_min(1.0)
        selected_compatibility = (assignment * batch.compatibility
                                  ).sum(1) / demand.clamp_min(1.0)
        continued = selected & (action == batch.previous_action)
        continued_fraction = (assignment * continued.unsqueeze(-1)
                              ).sum(1) / demand.clamp_min(1.0)
        task_action_nodes = (
            task_nodes + self.satellite_message(satellite_messages)
            + self.demand_projection(demand.log1p().unsqueeze(-1))
            + self.compatibility_projection(
                selected_compatibility.unsqueeze(-1),
            ) + self.continuity_projection(continued_fraction.unsqueeze(-1))
        )
        active_tasks = (demand > 0) & batch.task_mask
        denominator = batch.satellite_mask.sum(1).clamp_min(1).float()
        task_cost = (
            self.task_score(task_action_nodes).squeeze(-1) * active_tasks
        ).sum(1) / denominator

        null_satellites = batch.satellite_mask & ~selected
        null_cost = (
            self.null_score(satellite_nodes).squeeze(-1) * null_satellites
        ).sum(1) / denominator

        active_count = selected.sum(1).float()
        unique_count = active_tasks.sum(1).float()
        duplicate_count = active_count - unique_count
        max_demand = demand.max(1).values
        selected_compatibility_sum = (assignment * batch.compatibility).sum(
            (1, 2)
        )
        action_summary = torch.stack((
            active_count / denominator,
            unique_count / denominator,
            duplicate_count / denominator,
            max_demand / denominator,
            selected_compatibility_sum / active_count.clamp_min(1.0),
            continued.sum(1).float() / active_count.clamp_min(1.0),
        ), -1)
        global_cost = self.action_score(action_summary).squeeze(-1)
        return task_cost + null_cost + global_cost

    def forward(
        self,
        batch: GraphQBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.score_batch(batch, batch.better_action),
            self.score_batch(batch, batch.worse_action),
        )


class SummaryActionCritic(nn.Module):
    """不保留卫星—任务身份关系的动作汇总基线。"""

    def __init__(self, *, summary_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        return self.layers(summary).squeeze(-1)


@dataclass
class GraphQCriticBundle:
    """模型与仅由训练场景计算的特征归一化统计。"""

    baseline: SummaryActionCritic
    graph_q: GraphQCritic
    satellite_mean: torch.Tensor
    satellite_std: torch.Tensor
    task_mean: torch.Tensor
    task_std: torch.Tensor
    summary_mean: torch.Tensor
    summary_std: torch.Tensor

    def _prepare(
        self,
        batch: GraphQBatch,
        *,
        device: torch.device,
    ) -> GraphQBatch:
        satellite_mask = batch.satellite_mask.to(device)
        task_mask = batch.task_mask.to(device)
        satellite_features = (
            batch.satellite_features.to(device)
            - self.satellite_mean.to(device)
        ) / self.satellite_std.to(device)
        task_features = (
            batch.task_features.to(device) - self.task_mean.to(device)
        ) / self.task_std.to(device)
        satellite_features = torch.where(
            satellite_mask.unsqueeze(-1),
            satellite_features,
            0.0,
        )
        task_features = torch.where(
            task_mask.unsqueeze(-1),
            task_features,
            0.0,
        )
        return GraphQBatch(
            scene_ids=batch.scene_ids.to(device),
            satellite_features=satellite_features,
            task_features=task_features,
            compatibility=batch.compatibility.to(device),
            satellite_mask=satellite_mask,
            task_mask=task_mask,
            previous_action=batch.previous_action.to(device),
            better_action=batch.better_action.to(device),
            worse_action=batch.worse_action.to(device),
            better_summary=(
                batch.better_summary.to(device) - self.summary_mean.to(device)
            ) / self.summary_std.to(device),
            worse_summary=(
                batch.worse_summary.to(device) - self.summary_mean.to(device)
            ) / self.summary_std.to(device),
            margins=batch.margins.to(device),
        )

    def predict(
        self,
        samples: Sequence[GraphQSample],
        *,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        self.baseline.to(device).eval()
        self.graph_q.to(device).eval()
        output: dict[str, list[torch.Tensor]] = {
            'baseline_better': [],
            'baseline_worse': [],
            'graph_q_better': [],
            'graph_q_worse': [],
        }
        with torch.inference_mode():
            for start in range(0, len(samples), batch_size):
                batch = self._prepare(
                    collate_graph_q_samples(samples[start:start + batch_size]),
                    device=device,
                )
                baseline_better = self.baseline(batch.better_summary)
                baseline_worse = self.baseline(batch.worse_summary)
                graph_better, graph_worse = self.graph_q(batch)
                for key, value in (
                    ('baseline_better', baseline_better),
                    ('baseline_worse', baseline_worse),
                    ('graph_q_better', graph_better),
                    ('graph_q_worse', graph_worse),
                ):
                    output[key].append(value.cpu())
        return {key: torch.cat(values) for key, values in output.items()}


def _feature_statistics(
    samples: Sequence[GraphQSample],
) -> tuple[torch.Tensor, ...]:
    satellites = torch.cat([
        item.satellite_features.float() for item in samples
    ])
    tasks = torch.cat([item.task_features.float() for item in samples])
    summaries = torch.stack([
        summary.float()
        for item in samples
        for summary in (item.better_summary, item.worse_summary)
    ])

    def statistics(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            tensor.mean(0),
            tensor.std(0, unbiased=False).clamp_min(1e-6),
        )

    satellite_mean, satellite_std = statistics(satellites)
    task_mean, task_std = statistics(tasks)
    summary_mean, summary_std = statistics(summaries)
    return (
        satellite_mean,
        satellite_std,
        task_mean,
        task_std,
        summary_mean,
        summary_std,
    )


def fit_graph_q_critics(
    train_samples: Sequence[GraphQSample],
    val_samples: Sequence[GraphQSample],
    *,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    margin_clip: float,
    seed: int,
    device: torch.device,
) -> tuple[GraphQCriticBundle, dict[str, object]]:
    """用稳健 margin 加权的成对 logistic loss 训练两个裁判。"""

    if not train_samples or not val_samples:
        raise ValueError('train and validation samples must be non-empty')
    if epochs <= 0 or batch_size <= 0 or margin_clip <= 0:
        raise ValueError('epochs, batch size and margin clip must be positive')
    torch.manual_seed(seed)
    statistics = _feature_statistics(train_samples)
    baseline = SummaryActionCritic(
        summary_dim=train_samples[0].better_summary.numel(),
        hidden_dim=hidden_dim,
    ).to(device)
    graph_q = GraphQCritic(
        satellite_dim=train_samples[0].satellite_features.shape[-1],
        task_dim=train_samples[0].task_features.shape[-1],
        hidden_dim=hidden_dim,
    ).to(device)
    bundle = GraphQCriticBundle(
        baseline=baseline,
        graph_q=graph_q,
        satellite_mean=statistics[0],
        satellite_std=statistics[1],
        task_mean=statistics[2],
        task_std=statistics[3],
        summary_mean=statistics[4],
        summary_std=statistics[5],
    )
    optimizer = torch.optim.AdamW(
        list(baseline.parameters()) + list(graph_q.parameters()),
        lr=learning_rate,
    )
    generator = torch.Generator().manual_seed(seed)
    final_loss = float('nan')
    for _ in range(epochs):
        permutation = torch.randperm(len(train_samples), generator=generator)
        for start in range(0, len(train_samples), batch_size):
            indices = permutation[start:start + batch_size].tolist()
            batch = bundle._prepare(
                collate_graph_q_samples([
                    train_samples[index] for index in indices
                ]),
                device=device,
            )
            baseline_better = baseline(batch.better_summary)
            baseline_worse = baseline(batch.worse_summary)
            graph_better, graph_worse = graph_q(batch)
            weights = batch.margins.clamp(max=margin_clip)
            weights = weights / weights.mean().clamp_min(1e-6)
            baseline_loss = (
                F.softplus(baseline_better - baseline_worse) * weights
            ).mean()
            graph_loss = (F.softplus(graph_better - graph_worse)
                          * weights).mean()
            loss = baseline_loss + graph_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().item())

    baseline.cpu()
    graph_q.cpu()
    predictions = bundle.predict(
        val_samples,
        batch_size=batch_size,
        device=device,
    )
    baseline_audit = audit_pairwise_tournament(
        val_samples,
        better_scores=predictions['baseline_better'],
        worse_scores=predictions['baseline_worse'],
        greedy_candidate='candidate_000_greedy',
    )
    graph_audit = audit_pairwise_tournament(
        val_samples,
        better_scores=predictions['graph_q_better'],
        worse_scores=predictions['graph_q_worse'],
        greedy_candidate='candidate_000_greedy',
    )
    gain = (
        float(graph_audit['pairwise_accuracy'])
        - float(baseline_audit['pairwise_accuracy'])
    )
    accepted = bool(
        float(graph_audit['pairwise_accuracy']) >= 0.6 and gain >= 0.05
        and float(graph_audit['mean_regret']
                  ) <= float(baseline_audit['mean_regret'])
    )
    return bundle, {
        'num_train_samples': len(train_samples),
        'num_val_samples': len(val_samples),
        'train_final_loss': final_loss,
        'baseline': baseline_audit,
        'graph_q': graph_audit,
        'pairwise_accuracy_gain': gain,
        'accepted': accepted,
    }


def pairwise_accuracy(
    *,
    better_scores: torch.Tensor,
    worse_scores: torch.Tensor,
) -> float:
    """返回排序准确率；完全相同的分数按随机猜测 0.5 计。"""

    if better_scores.shape != worse_scores.shape or not better_scores.numel():
        raise ValueError(
            'pairwise score tensors must be non-empty and aligned'
        )
    correct = (better_scores < worse_scores).float()
    ties = (better_scores == worse_scores).float() * 0.5
    return float((correct + ties).mean().item())


def audit_pairwise_tournament(
    samples: Sequence[GraphQSample],
    *,
    better_scores: torch.Tensor,
    worse_scores: torch.Tensor,
    greedy_candidate: str,
) -> dict[str, float | int]:
    """用成对胜率组成场景内锦标赛，并计算 top-1 regret。"""

    if len(samples) != better_scores.numel(
    ) or (better_scores.shape != worse_scores.shape):
        raise ValueError('samples and pairwise predictions must be aligned')
    scene_points: dict[int, dict[str, float]] = {}
    scene_costs: dict[int, dict[str, float]] = {}
    for sample, better_score, worse_score in zip(
        samples,
        better_scores.tolist(),
        worse_scores.tolist(),
    ):
        points = scene_points.setdefault(sample.scene_id, {})
        costs = scene_costs.setdefault(sample.scene_id, {})
        probability = float(
            torch.sigmoid(torch.tensor(
                worse_score - better_score,
            )).item()
        )
        points[sample.better_candidate
               ] = (points.get(sample.better_candidate, 0.0) + probability)
        points[
            sample.worse_candidate
        ] = (points.get(sample.worse_candidate, 0.0) + 1.0 - probability)
        costs[sample.better_candidate] = sample.better_cost
        costs[sample.worse_candidate] = sample.worse_cost

    regrets = []
    selected_vs_greedy = []
    exact = 0
    for scene_id, points in scene_points.items():
        costs = scene_costs[scene_id]
        selected = min(
            points,
            key=lambda name: (-points[name], name != greedy_candidate, name),
        )
        oracle_cost = min(costs.values())
        selected_cost = costs[selected]
        regrets.append(selected_cost - oracle_cost)
        if abs(selected_cost - oracle_cost) <= 1e-8:
            exact += 1
        greedy_cost = costs.get(greedy_candidate)
        if greedy_cost is not None:
            selected_vs_greedy.append(selected_cost - greedy_cost)
    return {
        'num_pairs': len(samples),
        'num_scenes': len(scene_points),
        'pairwise_accuracy': pairwise_accuracy(
            better_scores=better_scores,
            worse_scores=worse_scores,
        ),
        'top1_exact_best_scenes': exact,
        'mean_regret': sum(regrets) / len(regrets),
        'selected_vs_greedy_mean_cost_delta': (
            None if not selected_vs_greedy else sum(selected_vs_greedy)
            / len(selected_vs_greedy)
        ),
    }
