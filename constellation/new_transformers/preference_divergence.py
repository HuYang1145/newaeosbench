"""从同场景候选轨迹提取第一动作分歧偏好。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def first_divergence_index(
    left_actions: torch.Tensor,
    right_actions: torch.Tensor,
) -> int | None:
    """返回两条联合动作序列第一次不同的时间下标。"""

    left_actions = torch.as_tensor(left_actions)
    right_actions = torch.as_tensor(right_actions)
    if left_actions.shape != right_actions.shape:
        raise ValueError('joint action sequences must have the same shape')
    if left_actions.ndim != 2:
        raise ValueError('joint action sequences must be time-major matrices')
    different = (left_actions != right_actions).any(-1).nonzero().flatten()
    if not different.numel():
        return None
    return int(different[0].item())


def _action_summary(action: torch.Tensor) -> dict[str, int]:
    action = action.long().flatten()
    selected = action[action >= 0]
    active = int(selected.numel())
    unique = int(selected.unique().numel()) if active else 0
    return {
        'active_satellites': active,
        'null_satellites': int(action.numel()) - active,
        'unique_tasks': unique,
        'duplicate_assignments': active - unique,
    }


def _trajectory_state(
    trajectory: Mapping[str, Any],
    index: int,
    *,
    sensor_enabled_index: int | None,
) -> dict[str, torch.Tensor | None]:
    constellation = trajectory['constellation']
    taskset = trajectory['taskset']
    if not isinstance(constellation, Mapping) or not isinstance(taskset, Mapping):
        raise TypeError('trajectory state sections must be mappings')
    return {
        'task_progress': torch.as_tensor(taskset['progress'])[index],
        'sensor_enabled': (
            None if sensor_enabled_index is None
            else torch.as_tensor(
                constellation['sensor_enabled'],
            )[sensor_enabled_index]
        ),
        'constellation_data': torch.as_tensor(constellation['data'])[index],
    }


def build_first_divergence_record(
    *,
    scene_id: int,
    better_candidate: str,
    worse_candidate: str,
    better_cost: float,
    worse_cost: float,
    better_trajectory_path: str,
    worse_trajectory_path: str,
    better_trajectory: Mapping[str, Any],
    worse_trajectory: Mapping[str, Any],
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, object] | None:
    """构造当前状态可用、且不含 ``is_visible`` 的分歧点记录。"""

    if not better_cost < worse_cost:
        raise ValueError('better candidate cost must be lower than worse cost')
    better_actions = torch.as_tensor(
        better_trajectory['actions']['task_id'],
    ).long()
    worse_actions = torch.as_tensor(
        worse_trajectory['actions']['task_id'],
    ).long()
    divergence = first_divergence_index(better_actions, worse_actions)
    if divergence is None:
        return None

    # TrajectoryLogger 在 take_actions 之后记录 sensor_enabled，因此日志的
    # sensor_enabled[t] 已受 action[t] 影响；决策前状态应取上一帧。t=0
    # 没有保存初始传感器状态，不能把它伪装成共享状态样本。
    sensor_enabled_index = divergence - 1 if divergence > 0 else None
    better_state = _trajectory_state(
        better_trajectory,
        divergence,
        sensor_enabled_index=sensor_enabled_index,
    )
    worse_state = _trajectory_state(
        worse_trajectory,
        divergence,
        sensor_enabled_index=sensor_enabled_index,
    )
    state_match = {
        'task_progress': bool(torch.equal(
            better_state['task_progress'],
            worse_state['task_progress'],
        )),
        'sensor_enabled': (
            None if sensor_enabled_index is None
            else bool(torch.equal(
                better_state['sensor_enabled'],
                worse_state['sensor_enabled'],
            ))
        ),
        'constellation_data': bool(torch.allclose(
            better_state['constellation_data'].float(),
            worse_state['constellation_data'].float(),
            atol=atol,
            rtol=rtol,
        )),
    }
    better_action = better_actions[divergence]
    worse_action = worse_actions[divergence]
    return {
        'scene_id': int(scene_id),
        'better_candidate': better_candidate,
        'worse_candidate': worse_candidate,
        'better_cost': float(better_cost),
        'worse_cost': float(worse_cost),
        'cost_margin': float(worse_cost - better_cost),
        'divergence_index': divergence,
        'divergence_fraction': divergence / max(better_actions.shape[0] - 1, 1),
        'current_state_reconstructable': sensor_enabled_index is not None,
        'sensor_enabled_source_index': sensor_enabled_index,
        'shared_state_match': (
            sensor_enabled_index is not None
            and all(value is True for value in state_match.values())
        ),
        'state_match': state_match,
        'better_action': better_action.tolist(),
        'worse_action': worse_action.tolist(),
        'changed_satellites': int(
            (better_action != worse_action).sum().item(),
        ),
        'better_action_summary': _action_summary(better_action),
        'worse_action_summary': _action_summary(worse_action),
        'better_trajectory_path': better_trajectory_path,
        'worse_trajectory_path': worse_trajectory_path,
    }
