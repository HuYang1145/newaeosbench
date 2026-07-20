"""基于受控短窗口标签的局部 Graph-Q 裁判模型。

模型只消费 Actor 在线已有的状态、候选 logits 与动作历史摘要。Basilisk 结果仅在
离线数据中形成排序标签和多目标回归目标，不进入裁判在线输入。
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

LOCAL_OUTCOME_KEYS = (
    'completed_tasks',
    'partial_progress_gain',
    'pc_wh',
    'switches',
    'one_second_runs',
    'redundant_satellite_seconds',
)


@dataclasses.dataclass(frozen=True)
class LocalGraphQSample:
    scene_id: int
    decision_time: int
    satellite_features: torch.Tensor
    task_features: torch.Tensor
    compatibility: torch.Tensor
    actor_logits: torch.Tensor
    previous_action: torch.Tensor
    better_action: torch.Tensor
    worse_action: torch.Tensor
    better_outcomes: torch.Tensor
    worse_outcomes: torch.Tensor
    margin: float
    better_branch: str
    worse_branch: str
    better_cost: float
    worse_cost: float


class LocalGraphQBatch(NamedTuple):
    scene_ids: torch.Tensor
    decision_times: torch.Tensor
    satellite_features: torch.Tensor
    task_features: torch.Tensor
    compatibility: torch.Tensor
    actor_logits: torch.Tensor
    satellite_mask: torch.Tensor
    task_mask: torch.Tensor
    previous_action: torch.Tensor
    better_action: torch.Tensor
    worse_action: torch.Tensor
    better_outcomes: torch.Tensor
    worse_outcomes: torch.Tensor
    margins: torch.Tensor


def _map_action(
    task_ids: Sequence[int],
    ongoing_ids: Sequence[int],
    *,
    unavailable_to_idle: bool = False,
) -> torch.Tensor:
    mapping = {
        int(task_id): index
        for index, task_id in enumerate(ongoing_ids)
    }
    output = []
    for task_id in task_ids:
        task_id = int(task_id)
        if task_id < 0:
            output.append(-1)
        elif task_id in mapping:
            output.append(mapping[task_id])
        elif unavailable_to_idle:
            output.append(-1)
        else:
            raise ValueError(f'action task {task_id} is not ongoing')
    return torch.tensor(output, dtype=torch.long)


def _outcomes(branch: Mapping[str, Any], horizon: int) -> torch.Tensor:
    metrics = branch['horizons'][str(horizon)]
    return torch.tensor([float(metrics[key]) for key in LOCAL_OUTCOME_KEYS])


def samples_from_branch_summary(
    payload: Mapping[str, Any],
) -> list[LocalGraphQSample]:
    """把受控 rollout JSON 汇总转换为局部 Graph-Q 样本。"""

    horizon = int(payload.get('primary_horizon', 300))
    scene_id = int(payload['scene_id'])
    samples = []
    for record in payload['records']:
        branches = record['branches']
        for pair in record['preference_pairs']:
            if int(pair['primary_horizon']) != horizon:
                continue
            better = branches[pair['better_branch']]
            worse = branches[pair['worse_branch']]
            if better['decision_state_signature'] != worse[
                'decision_state_signature']:
                raise ValueError('local Critic pair does not share one state')
            context = better['decision_context']
            if context != worse['decision_context']:
                raise ValueError('local Critic pair contexts differ')
            if context.get('uses_is_visible_as_input') is not False:
                raise ValueError('local Critic input must exclude is_visible')
            ongoing_ids = [int(value) for value in context['ongoing_task_ids']]
            satellite_features = torch.tensor(
                context['satellite_features'], dtype=torch.float32
            )
            task_features = torch.tensor(
                context['task_features'], dtype=torch.float32
            )
            actor_logits = torch.tensor(
                context['actor_logits'], dtype=torch.float32
            )
            satellite_types = torch.tensor(context['satellite_sensor_type'])
            task_types = torch.tensor(context['task_sensor_type'])
            compatibility = (satellite_types[:, None] == task_types[None, :]
                             ).float()

            def branch_action(branch: Mapping[str, Any]) -> torch.Tensor:
                assignment = list(branch['original_assignment'])
                assignment[int(branch['satellite_index'])
                           ] = int(branch['applied_task_id'])
                return _map_action(assignment, ongoing_ids)

            samples.append(
                LocalGraphQSample(
                    scene_id=scene_id,
                    decision_time=int(record['decision']['decision_time']),
                    satellite_features=satellite_features,
                    task_features=task_features,
                    compatibility=compatibility,
                    actor_logits=actor_logits,
                    previous_action=_map_action(
                        context['previous_assignment'],
                        ongoing_ids,
                        unavailable_to_idle=True,
                    ),
                    better_action=branch_action(better),
                    worse_action=branch_action(worse),
                    better_outcomes=_outcomes(better, horizon),
                    worse_outcomes=_outcomes(worse, horizon),
                    margin=float(pair['cost_margin']),
                    better_branch=str(pair['better_branch']),
                    worse_branch=str(pair['worse_branch']),
                    better_cost=float(pair['better_cost']),
                    worse_cost=float(pair['worse_cost']),
                )
            )
    return samples


def collate_local_graph_q_samples(
    samples: Sequence[LocalGraphQSample],
) -> LocalGraphQBatch:
    if not samples:
        raise ValueError('at least one local Graph-Q sample is required')
    batch_size = len(samples)
    max_satellites = max(item.satellite_features.shape[0] for item in samples)
    max_tasks = max(item.task_features.shape[0] for item in samples)
    satellite_dim = samples[0].satellite_features.shape[-1]
    task_dim = samples[0].task_features.shape[-1]
    outcome_dim = samples[0].better_outcomes.numel()
    satellite_features = torch.zeros(batch_size, max_satellites, satellite_dim)
    task_features = torch.zeros(batch_size, max_tasks, task_dim)
    compatibility = torch.zeros(batch_size, max_satellites, max_tasks)
    actor_logits = torch.full((batch_size, max_satellites, max_tasks + 1),
                              -1e9)
    satellite_mask = torch.zeros(batch_size, max_satellites, dtype=torch.bool)
    task_mask = torch.zeros(batch_size, max_tasks, dtype=torch.bool)
    previous_action = torch.full((batch_size, max_satellites),
                                 -1,
                                 dtype=torch.long)
    better_action = previous_action.clone()
    worse_action = previous_action.clone()
    better_outcomes = torch.zeros(batch_size, outcome_dim)
    worse_outcomes = torch.zeros(batch_size, outcome_dim)
    for index, item in enumerate(samples):
        num_satellites = item.satellite_features.shape[0]
        num_tasks = item.task_features.shape[0]
        if item.satellite_features.shape[-1] != satellite_dim:
            raise ValueError('satellite feature dimensions must match')
        if item.task_features.shape[-1] != task_dim:
            raise ValueError('task feature dimensions must match')
        if item.compatibility.shape != (num_satellites, num_tasks):
            raise ValueError('compatibility shape is invalid')
        if item.actor_logits.shape != (num_satellites, num_tasks + 1):
            raise ValueError('actor logits shape is invalid')
        satellite_features[index, :num_satellites] = item.satellite_features
        task_features[index, :num_tasks] = item.task_features
        compatibility[index, :num_satellites, :num_tasks] = item.compatibility
        actor_logits[index, :num_satellites, :num_tasks
                     + 1] = item.actor_logits
        satellite_mask[index, :num_satellites] = True
        task_mask[index, :num_tasks] = True
        previous_action[index, :num_satellites] = item.previous_action
        better_action[index, :num_satellites] = item.better_action
        worse_action[index, :num_satellites] = item.worse_action
        better_outcomes[index] = item.better_outcomes
        worse_outcomes[index] = item.worse_outcomes
    return LocalGraphQBatch(
        scene_ids=torch.tensor([item.scene_id for item in samples]),
        decision_times=torch.tensor([item.decision_time for item in samples]),
        satellite_features=satellite_features,
        task_features=task_features,
        compatibility=compatibility,
        actor_logits=actor_logits,
        satellite_mask=satellite_mask,
        task_mask=task_mask,
        previous_action=previous_action,
        better_action=better_action,
        worse_action=worse_action,
        better_outcomes=better_outcomes,
        worse_outcomes=worse_outcomes,
        margins=torch.tensor([item.margin for item in samples]).float(),
    )


class LocalGraphQCritic(nn.Module):
    """对一个完整联合候选动作预测局部代价及可解释结果分量。"""

    def __init__(
        self,
        *,
        satellite_dim: int,
        task_dim: int,
        outcome_dim: int,
        hidden_dim: int = 64,
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
        self.null_task = nn.Parameter(torch.zeros(hidden_dim))
        self.action_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 4, hidden_dim),
            nn.GELU(),
        )
        self.global_projection = nn.Sequential(
            nn.Linear(hidden_dim + 6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.score_head = nn.Linear(hidden_dim, 1)
        self.outcome_head = nn.Linear(hidden_dim, outcome_dim)

    def encode_action(
        self,
        batch: LocalGraphQBatch,
        action: torch.Tensor,
    ) -> torch.Tensor:
        satellite_nodes = self.satellite_encoder(batch.satellite_features)
        task_nodes = self.task_encoder(batch.task_features)
        num_tasks = task_nodes.shape[1]
        safe_action = action.clamp(0, max(num_tasks - 1, 0))
        selected = (
            batch.satellite_mask
            & (action >= 0)
            & (action < num_tasks)
        )
        gathered_tasks = task_nodes.gather(
            1,
            safe_action.unsqueeze(-1).expand(-1, -1, task_nodes.shape[-1]),
        )
        selected_tasks = torch.where(
            selected.unsqueeze(-1),
            gathered_tasks,
            self.null_task.view(1, 1, -1),
        )
        actor_index = (action + 1).clamp(0, num_tasks)
        selected_logit = batch.actor_logits.gather(
            2, actor_index.unsqueeze(-1)
        ).squeeze(-1)
        best_logit = batch.actor_logits.max(-1).values
        actor_margin = selected_logit - best_logit
        selected_compatibility = batch.compatibility.gather(
            2, safe_action.unsqueeze(-1)
        ).squeeze(-1)
        selected_compatibility = torch.where(
            selected, selected_compatibility, 1.0
        )
        continuity = (action == batch.previous_action).float()
        assignment = F.one_hot(safe_action, num_classes=num_tasks).float()
        assignment = assignment * selected.unsqueeze(-1)
        demand = assignment.sum(1)
        selected_demand = demand.gather(1, safe_action)
        selected_demand = torch.where(selected, selected_demand, 0.0)
        scalar_context = torch.stack((
            actor_margin,
            continuity,
            selected_compatibility,
            selected_demand.log1p(),
        ), -1)
        satellite_action = self.action_projection(
            torch.cat((satellite_nodes, selected_tasks, scalar_context), -1)
        )
        denominator = batch.satellite_mask.sum(1).clamp_min(1).unsqueeze(-1)
        pooled = (satellite_action
                  * batch.satellite_mask.unsqueeze(-1)).sum(1) / denominator

        active = selected.sum(1).float()
        unique = (demand > 0).sum(1).float()
        duplicate = active - unique
        max_demand = demand.max(1).values
        active_denominator = active.clamp_min(1.0)
        global_features = torch.stack((
            active / denominator.squeeze(-1),
            unique / denominator.squeeze(-1),
            duplicate / denominator.squeeze(-1),
            max_demand / denominator.squeeze(-1),
            (continuity * selected).sum(1) / active_denominator,
            (selected_compatibility * selected).sum(1) / active_denominator,
        ), -1)
        return self.global_projection(torch.cat((pooled, global_features), -1))

    def score_action(
        self,
        batch: LocalGraphQBatch,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode_action(batch, action)
        return self.score_head(hidden).squeeze(-1), self.outcome_head(hidden)

    def forward(
        self,
        batch: LocalGraphQBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        better_score, better_outcomes = self.score_action(
            batch, batch.better_action
        )
        worse_score, worse_outcomes = self.score_action(
            batch, batch.worse_action
        )
        return better_score, worse_score, better_outcomes, worse_outcomes


def split_samples_by_scene(
    samples: Sequence[LocalGraphQSample],
    *,
    num_folds: int,
    fold_index: int,
) -> tuple[
    list[LocalGraphQSample],
    list[LocalGraphQSample],
    list[int],
    list[int],
]:
    scene_ids = sorted({sample.scene_id for sample in samples})
    if not 2 <= num_folds <= len(scene_ids):
        raise ValueError('number of folds must be between 2 and scene count')
    if not 0 <= fold_index < num_folds:
        raise ValueError('fold index is outside the fold range')
    val_ids = scene_ids[fold_index::num_folds]
    val_set = set(val_ids)
    train_ids = sorted(set(scene_ids) - val_set)
    return (
        [sample for sample in samples if sample.scene_id not in val_set],
        [sample for sample in samples if sample.scene_id in val_set],
        train_ids,
        val_ids,
    )


def _action_summary(
    batch: LocalGraphQBatch,
    action: torch.Tensor,
) -> torch.Tensor:
    num_tasks = batch.task_features.shape[1]
    safe_action = action.clamp(0, max(num_tasks - 1, 0))
    selected = (batch.satellite_mask & (action >= 0) & (action < num_tasks))
    assignment = F.one_hot(safe_action, num_classes=num_tasks).float()
    assignment = assignment * selected.unsqueeze(-1)
    demand = assignment.sum(1)
    active = selected.sum(1).float()
    unique = (demand > 0).sum(1).float()
    duplicate = active - unique
    denominator = batch.satellite_mask.sum(1).clamp_min(1).float()
    continuity = selected & (action == batch.previous_action)
    compatibility = batch.compatibility.gather(2, safe_action.unsqueeze(-1)
                                               ).squeeze(-1)
    actor_index = (action + 1).clamp(0, num_tasks)
    selected_logit = batch.actor_logits.gather(2, actor_index.unsqueeze(-1)
                                               ).squeeze(-1)
    actor_margin = selected_logit - batch.actor_logits.max(-1).values
    active_denominator = active.clamp_min(1.0)
    return torch.stack((
        active / denominator,
        unique / denominator,
        duplicate / denominator,
        demand.max(1).values / denominator,
        continuity.sum(1).float() / active_denominator,
        (compatibility * selected).sum(1) / active_denominator,
        (actor_margin * selected).sum(1) / active_denominator,
    ), -1)


class LocalActionSummaryBaseline(nn.Module):

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        batch: LocalGraphQBatch,
        action: torch.Tensor,
    ) -> torch.Tensor:
        return self.layers(_action_summary(batch, action)).squeeze(-1)


@dataclasses.dataclass
class LocalGraphQCriticBundle:
    baseline: LocalActionSummaryBaseline
    graph_q: LocalGraphQCritic
    satellite_mean: torch.Tensor
    satellite_std: torch.Tensor
    task_mean: torch.Tensor
    task_std: torch.Tensor
    outcome_mean: torch.Tensor
    outcome_std: torch.Tensor

    def prepare(
        self,
        batch: LocalGraphQBatch,
        *,
        device: torch.device,
    ) -> LocalGraphQBatch:
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
            satellite_mask.unsqueeze(-1), satellite_features, 0.0
        )
        task_features = torch.where(
            task_mask.unsqueeze(-1), task_features, 0.0
        )
        return LocalGraphQBatch(
            scene_ids=batch.scene_ids.to(device),
            decision_times=batch.decision_times.to(device),
            satellite_features=satellite_features,
            task_features=task_features,
            compatibility=batch.compatibility.to(device),
            actor_logits=batch.actor_logits.to(device),
            satellite_mask=satellite_mask,
            task_mask=task_mask,
            previous_action=batch.previous_action.to(device),
            better_action=batch.better_action.to(device),
            worse_action=batch.worse_action.to(device),
            better_outcomes=(
                batch.better_outcomes.to(device)
                - self.outcome_mean.to(device)
            ) / self.outcome_std.to(device),
            worse_outcomes=(
                batch.worse_outcomes.to(device) - self.outcome_mean.to(device)
            ) / self.outcome_std.to(device),
            margins=batch.margins.to(device),
        )

    def predict(
        self,
        samples: Sequence[LocalGraphQSample],
        *,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        self.baseline.to(device).eval()
        self.graph_q.to(device).eval()
        output: dict[str, list[torch.Tensor]] = {
            key: []
            for key in (
                'baseline_better',
                'baseline_worse',
                'graph_q_better',
                'graph_q_worse',
                'outcome_better',
                'outcome_worse',
            )
        }
        with torch.inference_mode():
            for start in range(0, len(samples), batch_size):
                batch = self.prepare(
                    collate_local_graph_q_samples(
                        samples[start:start + batch_size]
                    ),
                    device=device,
                )
                baseline_better = self.baseline(batch, batch.better_action)
                baseline_worse = self.baseline(batch, batch.worse_action)
                (
                    graph_better,
                    graph_worse,
                    outcome_better,
                    outcome_worse,
                ) = self.graph_q(batch)
                for key, value in (
                    ('baseline_better', baseline_better),
                    ('baseline_worse', baseline_worse),
                    ('graph_q_better', graph_better),
                    ('graph_q_worse', graph_worse),
                    ('outcome_better', outcome_better),
                    ('outcome_worse', outcome_worse),
                ):
                    output[key].append(value.cpu())
        return {key: torch.cat(values) for key, values in output.items()}


def _statistics(
    samples: Sequence[LocalGraphQSample],
) -> tuple[torch.Tensor, ...]:
    satellites = torch.cat([sample.satellite_features for sample in samples])
    tasks = torch.cat([sample.task_features for sample in samples])
    outcomes = torch.stack([
        outcome for sample in samples
        for outcome in (sample.better_outcomes, sample.worse_outcomes)
    ])

    def mean_std(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return tensor.mean(0), tensor.std(0, unbiased=False).clamp_min(1e-6)

    return (*mean_std(satellites), *mean_std(tasks), *mean_std(outcomes))


def _pairwise_accuracy(
    better_scores: torch.Tensor,
    worse_scores: torch.Tensor,
) -> float:
    correct = (better_scores < worse_scores).float()
    ties = (better_scores == worse_scores).float() * 0.5
    return float((correct + ties).mean().item())


def audit_local_tournament(
    samples: Sequence[LocalGraphQSample],
    *,
    better_scores: torch.Tensor,
    worse_scores: torch.Tensor,
) -> dict[str, float | int]:
    if len(samples) != better_scores.numel(
    ) or (better_scores.shape != worse_scores.shape):
        raise ValueError('samples and predictions must be aligned')
    points_by_state: dict[tuple[int, int], dict[str, float]] = {}
    costs_by_state: dict[tuple[int, int], dict[str, float]] = {}
    for sample, better_score, worse_score in zip(
        samples, better_scores.tolist(), worse_scores.tolist()
    ):
        key = (sample.scene_id, sample.decision_time)
        points = points_by_state.setdefault(key, {})
        costs = costs_by_state.setdefault(key, {})
        better_probability = float(
            torch.sigmoid(torch.tensor(worse_score - better_score)).item()
        )
        points[
            sample.better_branch
        ] = (points.get(sample.better_branch, 0.0) + better_probability)
        points[
            sample.worse_branch
        ] = (points.get(sample.worse_branch, 0.0) + 1.0 - better_probability)
        costs[sample.better_branch] = sample.better_cost
        costs[sample.worse_branch] = sample.worse_cost
    regrets = []
    exact = 0
    for key, points in points_by_state.items():
        costs = costs_by_state[key]
        selected = min(points, key=lambda name: (-points[name], name))
        oracle_cost = min(costs.values())
        regret = costs[selected] - oracle_cost
        regrets.append(regret)
        exact += int(abs(regret) <= 1e-8)
    return {
        'num_pairs': len(samples),
        'num_states': len(points_by_state),
        'pairwise_accuracy': _pairwise_accuracy(better_scores, worse_scores),
        'top1_exact_states': exact,
        'mean_regret': sum(regrets) / len(regrets),
    }


def fit_local_graph_q_critics(
    train_samples: Sequence[LocalGraphQSample],
    val_samples: Sequence[LocalGraphQSample],
    *,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    outcome_loss_weight: float,
    margin_clip: float,
    seed: int,
    device: torch.device,
) -> tuple[LocalGraphQCriticBundle, dict[str, Any]]:
    if not train_samples or not val_samples:
        raise ValueError('train and validation samples must be non-empty')
    if min(epochs, batch_size) <= 0 or learning_rate <= 0:
        raise ValueError('training configuration must be positive')
    if outcome_loss_weight < 0 or margin_clip <= 0:
        raise ValueError('loss weights and margin clip are invalid')
    torch.manual_seed(seed)
    stats = _statistics(train_samples)
    baseline = LocalActionSummaryBaseline(hidden_dim).to(device)
    graph_q = LocalGraphQCritic(
        satellite_dim=train_samples[0].satellite_features.shape[-1],
        task_dim=train_samples[0].task_features.shape[-1],
        outcome_dim=train_samples[0].better_outcomes.numel(),
        hidden_dim=hidden_dim,
    ).to(device)
    bundle = LocalGraphQCriticBundle(
        baseline=baseline,
        graph_q=graph_q,
        satellite_mean=stats[0],
        satellite_std=stats[1],
        task_mean=stats[2],
        task_std=stats[3],
        outcome_mean=stats[4],
        outcome_std=stats[5],
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
            batch = bundle.prepare(
                collate_local_graph_q_samples([
                    train_samples[index] for index in indices
                ]),
                device=device,
            )
            baseline_better = baseline(batch, batch.better_action)
            baseline_worse = baseline(batch, batch.worse_action)
            (
                graph_better,
                graph_worse,
                predicted_better_outcomes,
                predicted_worse_outcomes,
            ) = graph_q(batch)
            weights = batch.margins.clamp(max=margin_clip)
            weights = weights / weights.mean().clamp_min(1e-6)
            baseline_rank_loss = (
                F.softplus(baseline_better - baseline_worse) * weights
            ).mean()
            graph_rank_loss = (
                F.softplus(graph_better - graph_worse) * weights
            ).mean()
            outcome_loss = 0.5 * (
                F.smooth_l1_loss(
                    predicted_better_outcomes, batch.better_outcomes
                ) + F.
                smooth_l1_loss(predicted_worse_outcomes, batch.worse_outcomes)
            )
            loss = (
                baseline_rank_loss + graph_rank_loss
                + outcome_loss_weight * outcome_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().item())
    baseline.cpu()
    graph_q.cpu()
    predictions = bundle.predict(
        val_samples, batch_size=batch_size, device=device
    )
    baseline_audit = audit_local_tournament(
        val_samples,
        better_scores=predictions['baseline_better'],
        worse_scores=predictions['baseline_worse'],
    )
    graph_audit = audit_local_tournament(
        val_samples,
        better_scores=predictions['graph_q_better'],
        worse_scores=predictions['graph_q_worse'],
    )
    predicted_outcomes = torch.cat((
        predictions['outcome_better'],
        predictions['outcome_worse'],
    ))
    predicted_outcomes = (
        predicted_outcomes * bundle.outcome_std + bundle.outcome_mean
    )
    target_outcomes = torch.cat((
        torch.stack([sample.better_outcomes for sample in val_samples]),
        torch.stack([sample.worse_outcomes for sample in val_samples]),
    ))
    outcome_mae_values = (predicted_outcomes - target_outcomes).abs().mean(0)
    outcome_mae = {
        key: float(value)
        for key, value in zip(LOCAL_OUTCOME_KEYS, outcome_mae_values.tolist())
    }
    gain = float(
        graph_audit['pairwise_accuracy'] - baseline_audit['pairwise_accuracy']
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
        'outcome_mae': outcome_mae,
        'pairwise_accuracy_gain': gain,
        'accepted': accepted,
    }


def rerank_candidate_actions(
    bundle: LocalGraphQCriticBundle,
    batch: LocalGraphQBatch,
    *,
    candidate_actions: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """用已训练裁判从 Actor top-k 联合动作中选择预测代价最低者。"""

    if candidate_actions.ndim != 3:
        raise ValueError(
            'candidate_actions must have shape (batch, candidates, satellites)'
        )
    if candidate_actions.shape[0] != batch.scene_ids.numel(
    ) or (candidate_actions.shape[2] != batch.satellite_features.shape[1]):
        raise ValueError('candidate actions do not match the state batch')
    prepared = bundle.prepare(batch, device=device)
    bundle.graph_q.to(device).eval()
    scores = []
    with torch.inference_mode():
        for candidate_index in range(candidate_actions.shape[1]):
            score, _ = bundle.graph_q.score_action(
                prepared,
                candidate_actions[:, candidate_index].to(device),
            )
            scores.append(score.cpu())
    bundle.graph_q.cpu()
    stacked = torch.stack(scores, -1)
    return stacked.argmin(-1), stacked
