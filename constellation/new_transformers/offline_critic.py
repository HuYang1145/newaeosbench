"""离线 Critic 的数据结构、紧凑特征与排序验收工具。

第一阶段只判断现有轨迹是否足以学习动作质量，不更新 Actor。可见性
``is_visible`` 只能作为离线监督来源，不能进入 Critic 输入特征。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Mapping, NamedTuple

import torch
from torch import nn


@dataclass(frozen=True)
class TrajectoryRecord:
    """一条轨迹及其论文口径 episode cost。"""

    scene_id: int
    epoch: int
    trajectory_path: Path | None
    metrics_path: Path | None
    episode_cost: float


class TransitionTensors(NamedTuple):
    """由离线轨迹构造的 ``(s, a, r, s')`` 张量。"""

    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    done: torch.Tensor
    episode_cost: torch.Tensor
    return_to_go: torch.Tensor


class OfflineDatasetTensors(NamedTuple):
    """多条轨迹拼接后的离线转移数据。"""

    trajectory_ids: torch.Tensor
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    done: torch.Tensor
    episode_cost: torch.Tensor
    return_to_go: torch.Tensor


class DenseRewardTargets(NamedTuple):
    """可解释的单步奖励分量及 ``gamma=1`` Monte Carlo 回报。"""

    reward: torch.Tensor
    return_to_go: torch.Tensor
    quality_delta: torch.Tensor
    tat_cost: torch.Tensor
    power_cost: torch.Tensor
    terminal_correction: torch.Tensor


def compute_cs_paper_from_metrics(
    metrics: Mapping[str, float],
) -> float:
    """从单场景评估指标计算统一的 ``CS_paper``。"""

    quality = (
        0.6 * float(metrics['CR'])
        + 0.2 * float(metrics['PCR'])
        + 0.2 * float(metrics['WCR'])
    )
    tat_s = float(metrics['TAT'])
    pc_wh = float(
        metrics['PC_Wh'] if 'PC_Wh' in metrics
        else float(metrics['PC']) / 3600.0
    )
    if quality <= 0:
        raise ValueError('completion quality must be positive')
    if tat_s < 0:
        raise ValueError('TAT_s must be non-negative')
    if pc_wh < 0:
        raise ValueError('PC_Wh must be non-negative')
    return 1.0 / quality + tat_s / 700.0 + pc_wh / 100.0


def load_routed_records(
    *,
    annotation_path: Path,
    data_root: Path,
    split: str,
) -> list[TrajectoryRecord]:
    """按 annotation 指定的 epoch 加载轨迹索引和真实 episode cost。"""

    payload = json.loads(annotation_path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise TypeError('offline critic annotation must contain ids and epochs')
    ids = payload.get('ids')
    epochs = payload.get('epochs')
    if not isinstance(ids, list) or not isinstance(epochs, list):
        raise TypeError('annotation ids and epochs must be lists')
    if len(ids) != len(epochs):
        raise ValueError('annotation ids and epochs must have equal length')

    records = []
    for scene_id, epoch in zip(ids, epochs):
        root = (
            data_root / f'trajectories.{int(epoch)}' / split
            / f'{int(scene_id) // 1000:02}'
        )
        trajectory_path = root / f'{int(scene_id):05}.pth'
        metrics_path = root / f'{int(scene_id):05}.json'
        if not trajectory_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(
                f'missing trajectory or metrics for scene {scene_id} '
                f'at epoch {epoch}',
            )
        metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        records.append(TrajectoryRecord(
            scene_id=int(scene_id),
            epoch=int(epoch),
            trajectory_path=trajectory_path,
            metrics_path=metrics_path,
            episode_cost=compute_cs_paper_from_metrics(metrics),
        ))
    return records


def audit_candidate_coverage(
    *,
    data_root: Path,
    split: str,
) -> dict[str, object]:
    """统计不同 ``trajectories.N`` 是否提供同场景候选轨迹。"""

    epoch_counts: dict[str, int] = {}
    scene_counts: Counter[int] = Counter()
    for root in sorted(data_root.glob('trajectories.*')):
        if not root.is_dir():
            continue
        epoch = root.name.removeprefix('trajectories.')
        paths = list((root / split).rglob('*.json'))
        if not paths:
            continue
        epoch_counts[epoch] = len(paths)
        scene_counts.update(int(path.stem) for path in paths)
    return {
        'epoch_counts': epoch_counts,
        'unique_scene_count': len(scene_counts),
        'repeated_scene_count': sum(count > 1 for count in scene_counts.values()),
        'max_candidates_per_scene': max(scene_counts.values(), default=0),
    }


def sample_time_indices(
    *,
    num_time_steps: int,
    num_samples: int,
) -> list[int]:
    """均匀抽取转移起点，并始终包含 episode 的最后一个转移。"""

    if num_time_steps < 2:
        raise ValueError('at least two time steps are required')
    if num_samples <= 0:
        raise ValueError('num_samples must be positive')
    num_transitions = num_time_steps - 1
    if num_samples >= num_transitions:
        return list(range(num_transitions))
    return (
        torch.linspace(0, num_transitions - 1, steps=num_samples)
        .round().long().unique(sorted=True).tolist()
    )


def aggregate_by_trajectory(
    *,
    trajectory_ids: torch.Tensor,
    target_cost: torch.Tensor,
    predicted_cost: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把多个转移预测聚合为轨迹级 cost，供排序验收。"""

    if not (
        trajectory_ids.numel() == target_cost.numel()
        == predicted_cost.numel()
    ):
        raise ValueError('aggregation tensors must have equal length')
    unique_ids = trajectory_ids.flatten().unique(sorted=True)
    targets = []
    predictions = []
    for trajectory_id in unique_ids:
        mask = trajectory_ids.flatten() == trajectory_id
        targets.append(target_cost.flatten()[mask].float().mean())
        predictions.append(predicted_cost.flatten()[mask].float().mean())
    return unique_ids, torch.stack(targets), torch.stack(predictions)


def combine_transition_tensors(
    items: list[tuple[int, TransitionTensors]],
) -> OfflineDatasetTensors:
    """拼接转移并保留轨迹编号，确保排序按 episode 聚合。"""

    if not items:
        raise ValueError('at least one trajectory is required')
    trajectory_ids = []
    columns: list[list[torch.Tensor]] = [[] for _ in range(7)]
    for trajectory_id, tensors in items:
        trajectory_ids.append(torch.full(
            (tensors.state.shape[0],),
            int(trajectory_id),
            dtype=torch.long,
        ))
        for column, tensor in zip(columns, tensors):
            column.append(tensor)
    return OfflineDatasetTensors(
        trajectory_ids=torch.cat(trajectory_ids),
        state=torch.cat(columns[0]),
        action=torch.cat(columns[1]),
        reward=torch.cat(columns[2]),
        next_state=torch.cat(columns[3]),
        done=torch.cat(columns[4]),
        episode_cost=torch.cat(columns[5]),
        return_to_go=torch.cat(columns[6]),
    )


def _state_features(
    *,
    time_index: int,
    num_time_steps: int,
    sensor_enabled: torch.Tensor,
    constellation_data: torch.Tensor,
    progress_ratio: torch.Tensor,
    max_progress_ratio: torch.Tensor,
    task_durations: torch.Tensor,
    task_static_data: torch.Tensor | None,
    constellation_static_data: torch.Tensor | None,
    task_sensor_type: torch.Tensor | None,
    constellation_sensor_type: torch.Tensor | None,
) -> torch.Tensor:
    satellite_data = constellation_data.float()
    satellite_mean = satellite_data.mean(0)
    satellite_std = satellite_data.std(0, unbiased=False)
    completed = max_progress_ratio >= 1.0
    weighted_completed = (
        task_durations[completed].sum() / task_durations.sum().clamp_min(1e-6)
    )
    denominator = max(num_time_steps - 1, 1)
    scalars = satellite_data.new_tensor([
        time_index / denominator,
        sensor_enabled.float().mean().item(),
        progress_ratio.mean().item(),
        max_progress_ratio.mean().item(),
        completed.float().mean().item(),
        weighted_completed.item(),
    ])
    features = [scalars, satellite_mean, satellite_std]
    if task_static_data is not None:
        assert constellation_static_data is not None
        assert task_sensor_type is not None
        assert constellation_sensor_type is not None
        current_task_data = task_static_data.float().clone()
        current_task_data[:, :2] -= time_index
        features.extend((
            current_task_data.mean(0),
            current_task_data.std(0, unbiased=False),
            constellation_static_data.float().mean(0),
            constellation_static_data.float().std(0, unbiased=False),
            task_sensor_type.float().new_tensor([
                task_sensor_type.float().mean().item(),
                task_sensor_type.float().std(unbiased=False).item(),
                constellation_sensor_type.float().mean().item(),
                constellation_sensor_type.float().std(unbiased=False).item(),
            ]),
        ))
    return torch.cat(features)


def _action_features(
    *,
    actions: torch.Tensor,
    previous_actions: torch.Tensor | None,
    progress_ratio: torch.Tensor,
    time_index: int,
    sensor_enabled: torch.Tensor,
    constellation_data: torch.Tensor,
    task_static_data: torch.Tensor | None,
    constellation_static_data: torch.Tensor | None,
    task_sensor_type: torch.Tensor | None,
    constellation_sensor_type: torch.Tensor | None,
) -> torch.Tensor:
    actions = actions.long()
    num_satellites = max(actions.numel(), 1)
    active = actions >= 0
    num_active = int(active.sum().item())
    if num_active:
        selected = actions[active]
        num_unique = int(selected.unique().numel())
        selected_progress = progress_ratio[selected].mean().item()
    else:
        num_unique = 0
        selected_progress = 0.0
    switch_fraction = (
        0.0 if previous_actions is None
        else (actions != previous_actions).float().mean().item()
    )
    basic = progress_ratio.new_tensor([
        num_active / num_satellites,
        1.0 - num_active / num_satellites,
        num_unique / num_satellites,
        (num_active - num_unique) / num_satellites,
        switch_fraction,
        selected_progress,
    ])
    if task_static_data is None:
        return basic

    assert constellation_static_data is not None
    assert task_sensor_type is not None
    assert constellation_sensor_type is not None
    context_dim = (
        constellation_static_data.shape[1]
        + constellation_data.shape[1]
        + task_static_data.shape[1]
        + 3
    )
    if not num_active:
        return torch.cat((basic, basic.new_zeros(context_dim)))

    satellite_ids = active.nonzero().flatten()
    task_ids = actions[active]
    current_task_data = task_static_data[task_ids].float().clone()
    current_task_data[:, :2] -= time_index
    pair_context = torch.cat((
        constellation_static_data[satellite_ids].float().mean(0),
        constellation_data[satellite_ids].float().mean(0),
        current_task_data.mean(0),
        progress_ratio[task_ids].mean().reshape(1),
        sensor_enabled[satellite_ids].float().mean().reshape(1),
        (
            constellation_sensor_type[satellite_ids]
            == task_sensor_type[task_ids]
        ).float().mean().reshape(1),
    ))
    return torch.cat((basic, pair_context))


def build_dense_reward_targets(
    trajectory: Mapping[str, object],
    *,
    task_durations: torch.Tensor,
    task_release_times: torch.Tensor,
    satellite_sensor_power: torch.Tensor,
    episode_cost: float,
) -> DenseRewardTargets:
    """构造动作级奖励，并用终点校正确保累计值等于 ``-CS_paper``。

    局部奖励为完成质量增量，减去任务完成时延和传感器功耗。最后一个转移加入
    校正项，使 ``reward.sum() == -episode_cost``，因此 dense reward 不会改变
    最终优化目标。
    """

    taskset = trajectory['taskset']
    actions_container = trajectory['actions']
    if not isinstance(taskset, Mapping):
        raise TypeError('trajectory.taskset must be a mapping')
    if not isinstance(actions_container, Mapping):
        raise TypeError('trajectory.actions must be a mapping')
    progress = torch.as_tensor(taskset['progress']).float()
    actions = torch.as_tensor(actions_container['task_id']).long()
    if progress.ndim != 2 or actions.ndim != 2:
        raise ValueError('progress and actions must be time-major matrices')
    if progress.shape[0] < 2 or actions.shape[0] != progress.shape[0]:
        raise ValueError('trajectory must contain aligned time steps')
    if not math.isfinite(float(episode_cost)) or episode_cost <= 0:
        raise ValueError('episode cost must be finite and positive')

    task_durations = task_durations.float()
    task_release_times = task_release_times.float()
    satellite_sensor_power = satellite_sensor_power.float()
    if task_durations.shape != progress.shape[1:]:
        raise ValueError('task durations must match the number of tasks')
    if task_release_times.shape != task_durations.shape:
        raise ValueError('task release times must match task durations')
    if satellite_sensor_power.shape != actions.shape[1:]:
        raise ValueError('sensor power must match the number of satellites')
    if (task_durations <= 0).any() or (satellite_sensor_power < 0).any():
        raise ValueError('durations must be positive and power non-negative')

    progress_ratio = (progress / task_durations).clamp(0, 1)
    max_progress_ratio = progress_ratio.cummax(0).values
    completed = max_progress_ratio >= 1.0
    cr = completed.float().mean(-1)
    pcr = max_progress_ratio.mean(-1)
    wcr = (
        (completed.float() * task_durations).sum(-1)
        / task_durations.sum()
    )
    quality = 0.6 * cr + 0.2 * pcr + 0.2 * wcr
    quality_delta = quality[1:] - quality[:-1]

    tat_cost = torch.zeros_like(quality_delta)
    final_completed = completed[-1]
    num_succeeded = int(final_completed.sum().item())
    if num_succeeded:
        for task_id in final_completed.nonzero().flatten().tolist():
            completion_time = int(
                completed[:, task_id].nonzero().flatten()[0].item(),
            )
            # time 0 之前的收益无法归因到已保存动作，由终点校正统一吸收。
            if completion_time > 0:
                tat_cost[completion_time - 1] += (
                    completion_time - task_release_times[task_id]
                ) / (700.0 * num_succeeded)

    # PowerUsageEvaluator 对每个非空 assignment 累加 sensor.power，PC_Wh/100
    # 因而对应除以 3600*100。最后一个无 next_state 的动作由终点校正吸收。
    power_cost = (
        (actions[:-1] >= 0).float() * satellite_sensor_power
    ).sum(-1) / 360000.0
    local_reward = quality_delta - tat_cost - power_cost
    terminal_correction = local_reward.new_tensor(
        -float(episode_cost),
    ) - local_reward.sum()
    reward = local_reward.clone()
    reward[-1] += terminal_correction
    return_to_go = reward.flip(0).cumsum(0).flip(0)
    return DenseRewardTargets(
        reward=reward,
        return_to_go=return_to_go,
        quality_delta=quality_delta,
        tat_cost=tat_cost,
        power_cost=power_cost,
        terminal_correction=terminal_correction,
    )


def build_transition_tensors(
    trajectory: Mapping[str, object],
    *,
    task_durations: torch.Tensor,
    episode_cost: float,
    time_indices: list[int],
    task_static_data: torch.Tensor | None = None,
    constellation_static_data: torch.Tensor | None = None,
    task_sensor_type: torch.Tensor | None = None,
    constellation_sensor_type: torch.Tensor | None = None,
    dense_reward_targets: DenseRewardTargets | None = None,
) -> TransitionTensors:
    """从一条已保存轨迹抽取确定性的离线转移样本。

    默认采用 ``gamma=1`` 的终止回报：中间奖励为 0，最后一个可用
    转移奖励为 ``-CS_paper``。传入 ``dense_reward_targets`` 时改用已校正的
    局部奖励和 cost-to-go。两种模式都保留正的 ``episode_cost`` 供对照。
    """

    constellation = trajectory['constellation']
    taskset = trajectory['taskset']
    actions_container = trajectory['actions']
    if not isinstance(constellation, Mapping):
        raise TypeError('trajectory.constellation must be a mapping')
    if not isinstance(taskset, Mapping):
        raise TypeError('trajectory.taskset must be a mapping')
    if not isinstance(actions_container, Mapping):
        raise TypeError('trajectory.actions must be a mapping')

    sensor_enabled = torch.as_tensor(constellation['sensor_enabled'])
    constellation_data = torch.as_tensor(constellation['data'])
    progress = torch.as_tensor(taskset['progress']).float()
    actions = torch.as_tensor(actions_container['task_id']).long()
    if progress.ndim != 2 or actions.ndim != 2:
        raise ValueError('progress and actions must be time-major matrices')
    if progress.shape[0] < 2:
        raise ValueError('trajectory must contain at least two time steps')
    if task_durations.shape != progress.shape[1:]:
        raise ValueError('task durations must match the number of tasks')
    if any(index < 0 or index >= progress.shape[0] - 1
           for index in time_indices):
        raise ValueError('time indices must identify valid transitions')
    if (
        dense_reward_targets is not None
        and dense_reward_targets.reward.shape != (progress.shape[0] - 1,)
    ):
        raise ValueError('dense rewards must match all trajectory transitions')
    context = (
        task_static_data,
        constellation_static_data,
        task_sensor_type,
        constellation_sensor_type,
    )
    if any(item is None for item in context) and not all(
        item is None for item in context
    ):
        raise ValueError('all static pair-context tensors must be provided')
    if task_static_data is not None:
        if task_static_data.shape[0] != progress.shape[1]:
            raise ValueError('task static data must match the number of tasks')
        if constellation_static_data is None or (
            constellation_static_data.shape[0] != actions.shape[1]
        ):
            raise ValueError(
                'constellation static data must match the number of satellites',
            )

    task_durations = task_durations.float().clamp_min(1.0)
    progress_ratio = (progress / task_durations).clamp(0, 1)
    max_progress_ratio = progress_ratio.cummax(0).values

    states = []
    action_features = []
    next_states = []
    rewards = []
    returns = []
    dones = []
    last_transition = progress.shape[0] - 2
    for index in time_indices:
        states.append(_state_features(
            time_index=index,
            num_time_steps=progress.shape[0],
            sensor_enabled=sensor_enabled[index],
            constellation_data=constellation_data[index],
            progress_ratio=progress_ratio[index],
            max_progress_ratio=max_progress_ratio[index],
            task_durations=task_durations,
            task_static_data=task_static_data,
            constellation_static_data=constellation_static_data,
            task_sensor_type=task_sensor_type,
            constellation_sensor_type=constellation_sensor_type,
        ))
        action_features.append(_action_features(
            actions=actions[index],
            previous_actions=None if index == 0 else actions[index - 1],
            progress_ratio=progress_ratio[index],
            time_index=index,
            sensor_enabled=sensor_enabled[index],
            constellation_data=constellation_data[index],
            task_static_data=task_static_data,
            constellation_static_data=constellation_static_data,
            task_sensor_type=task_sensor_type,
            constellation_sensor_type=constellation_sensor_type,
        ))
        next_states.append(_state_features(
            time_index=index + 1,
            num_time_steps=progress.shape[0],
            sensor_enabled=sensor_enabled[index + 1],
            constellation_data=constellation_data[index + 1],
            progress_ratio=progress_ratio[index + 1],
            max_progress_ratio=max_progress_ratio[index + 1],
            task_durations=task_durations,
            task_static_data=task_static_data,
            constellation_static_data=constellation_static_data,
            task_sensor_type=task_sensor_type,
            constellation_sensor_type=constellation_sensor_type,
        ))
        done = index == last_transition
        if dense_reward_targets is None:
            rewards.append(-float(episode_cost) if done else 0.0)
            returns.append(-float(episode_cost))
        else:
            rewards.append(float(dense_reward_targets.reward[index].item()))
            returns.append(float(
                dense_reward_targets.return_to_go[index].item(),
            ))
        dones.append(done)

    return TransitionTensors(
        state=torch.stack(states),
        action=torch.stack(action_features),
        reward=torch.tensor(rewards, dtype=torch.float32),
        next_state=torch.stack(next_states),
        done=torch.tensor(dones, dtype=torch.bool),
        episode_cost=torch.full(
            (len(time_indices),),
            float(episode_cost),
            dtype=torch.float32,
        ),
        return_to_go=torch.tensor(returns, dtype=torch.float32),
    )


def split_records_by_scene(
    records: list[TrajectoryRecord],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[TrajectoryRecord], list[TrajectoryRecord]]:
    """按 scene id 划分，防止同场景轨迹泄漏到两侧。"""

    if not 0 < val_fraction < 1:
        raise ValueError('val_fraction must be in (0, 1)')
    scene_ids = sorted({record.scene_id for record in records})
    if len(scene_ids) < 2:
        raise ValueError('at least two scenes are required')
    random.Random(seed).shuffle(scene_ids)
    num_val = min(
        len(scene_ids) - 1,
        max(1, round(len(scene_ids) * val_fraction)),
    )
    val_ids = set(scene_ids[:num_val])
    train = [record for record in records if record.scene_id not in val_ids]
    val = [record for record in records if record.scene_id in val_ids]
    return train, val


class _MLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self._layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._layers(inputs).squeeze(-1)


class StateValueBaseline(_MLP):
    """只看状态的场景/进度难度基线。"""

    def __init__(self, *, state_dim: int, hidden_dim: int = 64) -> None:
        super().__init__(state_dim, hidden_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return super().forward(state)


class ActionConditionedCritic(_MLP):
    """输入当前状态和已执行联合动作的紧凑 Critic。"""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__(state_dim + action_dim, hidden_dim)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(torch.cat((state, action), -1))


@dataclass
class DiagnosticCriticBundle:
    """Critic、基线及训练集归一化统计。"""

    baseline: StateValueBaseline
    critic: ActionConditionedCritic
    state_mean: torch.Tensor
    state_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor
    cost_mean: torch.Tensor
    cost_std: torch.Tensor

    def predict(
        self,
        dataset: OfflineDatasetTensors,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.baseline.to(device).eval()
        self.critic.to(device).eval()
        state = (
            (dataset.state.to(device) - self.state_mean.to(device))
            / self.state_std.to(device)
        )
        action = (
            (dataset.action.to(device) - self.action_mean.to(device))
            / self.action_std.to(device)
        )
        with torch.inference_mode():
            baseline = self.baseline(state)
            critic = self.critic(state, action)
        cost_mean = self.cost_mean.to(device)
        cost_std = self.cost_std.to(device)
        return (
            (baseline * cost_std + cost_mean).cpu(),
            (critic * cost_std + cost_mean).cpu(),
        )


def _feature_statistics(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = tensor.float().mean(0)
    std = tensor.float().std(0, unbiased=False).clamp_min(1e-6)
    return mean, std


def _ranking_for_subset(
    dataset: OfflineDatasetTensors,
    *,
    target_cost: torch.Tensor,
    baseline_cost: torch.Tensor,
    critic_cost: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float | bool]:
    trajectory_ids, target, baseline = aggregate_by_trajectory(
        trajectory_ids=dataset.trajectory_ids[mask],
        target_cost=target_cost[mask],
        predicted_cost=baseline_cost[mask],
    )
    critic_ids, critic_target, critic = aggregate_by_trajectory(
        trajectory_ids=dataset.trajectory_ids[mask],
        target_cost=target_cost[mask],
        predicted_cost=critic_cost[mask],
    )
    if not torch.equal(trajectory_ids, critic_ids):
        raise RuntimeError('baseline and critic trajectory ids differ')
    if not torch.allclose(target, critic_target):
        raise RuntimeError('baseline and critic targets differ')
    return evaluate_ranking(
        target_cost=target,
        critic_cost=critic,
        baseline_cost=baseline,
    )


def fit_diagnostic_critics(
    train: OfflineDatasetTensors,
    val: OfflineDatasetTensors,
    *,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    target_mode: str = 'episode_cost',
) -> tuple[DiagnosticCriticBundle, dict[str, object]]:
    """训练 state baseline 与 action Critic，并执行轨迹级排序验收。"""

    if epochs <= 0 or batch_size <= 0:
        raise ValueError('epochs and batch_size must be positive')
    if target_mode == 'episode_cost':
        train_target_cost = train.episode_cost
        val_target_cost = val.episode_cost
    elif target_mode == 'dense_cost_to_go':
        train_target_cost = -train.return_to_go
        val_target_cost = -val.return_to_go
    else:
        raise ValueError(f'unsupported target mode: {target_mode}')
    torch.manual_seed(seed)
    state_mean, state_std = _feature_statistics(train.state)
    action_mean, action_std = _feature_statistics(train.action)
    cost_mean, cost_std = _feature_statistics(train_target_cost[:, None])
    cost_mean = cost_mean.squeeze(0)
    cost_std = cost_std.squeeze(0)

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
    loss_fn = nn.SmoothL1Loss()

    state = ((train.state - state_mean) / state_std).to(device)
    action = ((train.action - action_mean) / action_std).to(device)
    target = ((train_target_cost - cost_mean) / cost_std).to(device)
    generator = torch.Generator().manual_seed(seed)
    final_loss = float('nan')
    for _ in range(epochs):
        permutation = torch.randperm(len(target), generator=generator)
        for start in range(0, len(target), batch_size):
            indices = permutation[start:start + batch_size].to(device)
            baseline_prediction = baseline(state[indices])
            critic_prediction = critic(state[indices], action[indices])
            loss = (
                loss_fn(baseline_prediction, target[indices])
                + loss_fn(critic_prediction, target[indices])
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().item())

    bundle = DiagnosticCriticBundle(
        baseline=baseline.cpu(),
        critic=critic.cpu(),
        state_mean=state_mean.cpu(),
        state_std=state_std.cpu(),
        action_mean=action_mean.cpu(),
        action_std=action_std.cpu(),
        cost_mean=cost_mean.cpu(),
        cost_std=cost_std.cpu(),
    )
    baseline_cost, critic_cost = bundle.predict(val, device=device)
    all_mask = torch.ones(len(val.state), dtype=torch.bool)
    # 第 0 个 state 特征是归一化前的 time fraction。
    early_mask = val.state[:, 0] <= 0.5
    if early_mask.sum() < 2:
        raise ValueError('validation set has fewer than two early transitions')
    all_ranking = _ranking_for_subset(
        val,
        target_cost=val_target_cost,
        baseline_cost=baseline_cost,
        critic_cost=critic_cost,
        mask=all_mask,
    )
    early_ranking = _ranking_for_subset(
        val,
        target_cost=val_target_cost,
        baseline_cost=baseline_cost,
        critic_cost=critic_cost,
        mask=early_mask,
    )
    summary: dict[str, object] = {
        'target_mode': target_mode,
        'train_final_loss': final_loss,
        'num_train_transitions': len(train.state),
        'num_val_transitions': len(val.state),
        'all': all_ranking,
        'early': early_ranking,
        'accepted': bool(
            all_ranking['accepted'] and early_ranking['accepted']
        ),
    }
    return bundle, summary


def _rank(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().float().flatten()
    order = values.argsort(stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(values)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(target: torch.Tensor, prediction: torch.Tensor) -> float:
    target_rank = _rank(target)
    prediction_rank = _rank(prediction)
    target_rank -= target_rank.mean()
    prediction_rank -= prediction_rank.mean()
    denominator = target_rank.norm() * prediction_rank.norm()
    if denominator == 0:
        return 0.0
    return float((target_rank @ prediction_rank / denominator).item())


def _pairwise_accuracy(
    target: torch.Tensor,
    prediction: torch.Tensor,
) -> float:
    target = target.detach().float().flatten()
    prediction = prediction.detach().float().flatten()
    target_delta = target[:, None] - target[None, :]
    prediction_delta = prediction[:, None] - prediction[None, :]
    pairs = torch.triu(target_delta != 0, diagonal=1)
    if not pairs.any():
        return 0.0
    correct = torch.sign(target_delta[pairs]) == torch.sign(
        prediction_delta[pairs],
    )
    return float(correct.float().mean().item())


def evaluate_ranking(
    *,
    target_cost: torch.Tensor,
    critic_cost: torch.Tensor,
    baseline_cost: torch.Tensor,
    min_spearman: float = 0.5,
    min_pairwise_accuracy: float = 0.6,
    min_spearman_gain: float = 0.05,
) -> dict[str, float | bool]:
    """比较 action Critic 与 state-only baseline，给出进入 Actor 阶段的门槛。"""

    if not (
        target_cost.numel() == critic_cost.numel()
        == baseline_cost.numel()
    ):
        raise ValueError('ranking tensors must have the same number of items')
    if target_cost.numel() < 2:
        raise ValueError('at least two items are required for ranking')

    critic_spearman = _spearman(target_cost, critic_cost)
    baseline_spearman = _spearman(target_cost, baseline_cost)
    critic_pairwise = _pairwise_accuracy(target_cost, critic_cost)
    result: dict[str, float | bool] = {
        'critic_mae': float(
            (critic_cost.float() - target_cost.float()).abs().mean().item(),
        ),
        'baseline_mae': float(
            (baseline_cost.float() - target_cost.float()).abs().mean().item(),
        ),
        'critic_spearman': critic_spearman,
        'baseline_spearman': baseline_spearman,
        'critic_pairwise_accuracy': critic_pairwise,
        'baseline_pairwise_accuracy': _pairwise_accuracy(
            target_cost,
            baseline_cost,
        ),
        'spearman_gain': critic_spearman - baseline_spearman,
    }
    result['accepted'] = bool(
        math.isfinite(critic_spearman)
        and critic_spearman >= min_spearman
        and critic_pairwise >= min_pairwise_accuracy
        and result['spearman_gain'] >= min_spearman_gain
    )
    return result
