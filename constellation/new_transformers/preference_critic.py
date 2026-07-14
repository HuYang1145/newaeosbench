"""同场景模型候选轨迹的 pairwise Critic。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn

from .offline_critic import ActionConditionedCritic, StateValueBaseline


class CandidateTensors(NamedTuple):
    """每条候选轨迹的聚合特征与最终 cost。"""

    scene_ids: torch.Tensor
    state: torch.Tensor
    action: torch.Tensor
    cost: torch.Tensor


class PreferencePairs(NamedTuple):
    """下标对：``better`` 的真实 cost 低于 ``worse``。"""

    better: torch.Tensor
    worse: torch.Tensor


def build_preference_pairs(
    *,
    scene_ids: torch.Tensor,
    costs: torch.Tensor,
    min_cost_margin: float = 1e-6,
) -> PreferencePairs:
    """只在同一 scene 内构造 cost 差异足够大的偏好对。"""

    scene_ids = scene_ids.flatten()
    costs = costs.float().flatten()
    if scene_ids.numel() != costs.numel():
        raise ValueError('scene ids and costs must have equal length')
    if min_cost_margin < 0:
        raise ValueError('min_cost_margin must be non-negative')
    better = []
    worse = []
    for scene_id in scene_ids.unique(sorted=True):
        indices = (scene_ids == scene_id).nonzero().flatten().tolist()
        for offset, left in enumerate(indices):
            for right in indices[offset + 1:]:
                delta = float(costs[left] - costs[right])
                if abs(delta) <= min_cost_margin:
                    continue
                if delta < 0:
                    better.append(left)
                    worse.append(right)
                else:
                    better.append(right)
                    worse.append(left)
    return PreferencePairs(
        better=torch.tensor(better, dtype=torch.long),
        worse=torch.tensor(worse, dtype=torch.long),
    )


@dataclass
class PreferenceCriticBundle:
    """成对排序模型及训练集归一化统计。"""

    baseline: StateValueBaseline
    critic: ActionConditionedCritic
    state_mean: torch.Tensor
    state_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor

    def predict(
        self,
        candidates: CandidateTensors,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.baseline.to(device).eval()
        self.critic.to(device).eval()
        state = (
            (candidates.state.to(device) - self.state_mean.to(device))
            / self.state_std.to(device)
        )
        action = (
            (candidates.action.to(device) - self.action_mean.to(device))
            / self.action_std.to(device)
        )
        with torch.inference_mode():
            baseline = self.baseline(state)
            critic = self.critic(state, action)
        return baseline.cpu(), critic.cpu()


def _feature_statistics(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = tensor.float().mean(0)
    std = tensor.float().std(0, unbiased=False).clamp_min(1e-6)
    return mean, std


def _pairwise_accuracy(
    predictions: torch.Tensor,
    pairs: PreferencePairs,
) -> float:
    if not pairs.better.numel():
        raise ValueError('at least one preference pair is required')
    correct = predictions[pairs.better] < predictions[pairs.worse]
    return float(correct.float().mean().item())


def fit_preference_critics(
    train: CandidateTensors,
    val: CandidateTensors,
    *,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    min_cost_margin: float = 1e-6,
) -> tuple[PreferenceCriticBundle, dict[str, float | int | bool]]:
    """用同场景 pairwise logistic loss 训练 baseline 和 action Critic。"""

    if epochs <= 0 or batch_size <= 0:
        raise ValueError('epochs and batch_size must be positive')
    train_pairs = build_preference_pairs(
        scene_ids=train.scene_ids,
        costs=train.cost,
        min_cost_margin=min_cost_margin,
    )
    val_pairs = build_preference_pairs(
        scene_ids=val.scene_ids,
        costs=val.cost,
        min_cost_margin=min_cost_margin,
    )
    if not train_pairs.better.numel() or not val_pairs.better.numel():
        raise ValueError('train and validation sets both need preference pairs')

    torch.manual_seed(seed)
    state_mean, state_std = _feature_statistics(train.state)
    action_mean, action_std = _feature_statistics(train.action)
    baseline = StateValueBaseline(
        state_dim=train.state.shape[1],
        hidden_dim=hidden_dim,
    ).to(device)
    critic = ActionConditionedCritic(
        state_dim=train.state.shape[1],
        action_dim=train.action.shape[1],
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(baseline.parameters()) + list(critic.parameters()),
        lr=learning_rate,
    )
    state = ((train.state - state_mean) / state_std).to(device)
    action = ((train.action - action_mean) / action_std).to(device)
    better = train_pairs.better.to(device)
    worse = train_pairs.worse.to(device)
    generator = torch.Generator().manual_seed(seed)
    final_loss = float('nan')
    for _ in range(epochs):
        permutation = torch.randperm(len(better), generator=generator)
        for start in range(0, len(better), batch_size):
            indices = permutation[start:start + batch_size].to(device)
            baseline_prediction = baseline(state)
            critic_prediction = critic(state, action)
            baseline_loss = nn.functional.softplus(
                baseline_prediction[better[indices]]
                - baseline_prediction[worse[indices]],
            ).mean()
            critic_loss = nn.functional.softplus(
                critic_prediction[better[indices]]
                - critic_prediction[worse[indices]],
            ).mean()
            loss = baseline_loss + critic_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().item())

    bundle = PreferenceCriticBundle(
        baseline=baseline.cpu(),
        critic=critic.cpu(),
        state_mean=state_mean.cpu(),
        state_std=state_std.cpu(),
        action_mean=action_mean.cpu(),
        action_std=action_std.cpu(),
    )
    baseline_prediction, critic_prediction = bundle.predict(val, device=device)
    baseline_accuracy = _pairwise_accuracy(baseline_prediction, val_pairs)
    critic_accuracy = _pairwise_accuracy(critic_prediction, val_pairs)
    gain = critic_accuracy - baseline_accuracy
    summary: dict[str, float | int | bool] = {
        'num_train_candidates': len(train.scene_ids),
        'num_val_candidates': len(val.scene_ids),
        'num_train_pairs': len(train_pairs.better),
        'num_val_pairs': len(val_pairs.better),
        'train_final_loss': final_loss,
        'baseline_pairwise_accuracy': baseline_accuracy,
        'critic_pairwise_accuracy': critic_accuracy,
        'pairwise_accuracy_gain': gain,
        'accepted': bool(critic_accuracy >= 0.6 and gain >= 0.05),
    }
    return bundle, summary
