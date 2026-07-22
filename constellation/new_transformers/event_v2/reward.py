"""第一阶段完成质量 reward 与半马尔可夫 GAE。"""

from collections.abc import Sequence
from typing import NamedTuple

import torch


class GAEOutput(NamedTuple):
    advantages: torch.Tensor
    returns: torch.Tensor


def completion_task_weights(
    required_duration: torch.Tensor,
) -> torch.Tensor:
    """构造 dense potential 的逐任务权重。

    CR 与 PCR 的均匀权重合计为 ``0.8/N``，WCR 的时长权重为
    ``0.2*duration/sum(duration)``。这些权重只定义 potential；精确终点 Q 仍直接按
    Evaluator 公式重建。
    """

    if required_duration.ndim == 0 or required_duration.shape[-1] == 0:
        raise ValueError('required duration needs a non-empty task dimension')
    if (
        not torch.isfinite(required_duration).all()
        or (required_duration <= 0).any()
    ):
        raise ValueError('required duration must be finite and positive')
    num_tasks = required_duration.shape[-1]
    return (
        required_duration.new_full(required_duration.shape, 0.8 / num_tasks)
        + 0.2 * required_duration
        / required_duration.sum(dim=-1, keepdim=True)
    )


def _validate_completion_inputs(
    progress: torch.Tensor,
    required_duration: torch.Tensor,
    task_weights: torch.Tensor,
) -> None:
    if (
        progress.shape != required_duration.shape
        or progress.shape != task_weights.shape
    ):
        raise ValueError(
            'progress, required duration and task weights must share shape'
        )
    if progress.ndim == 0:
        raise ValueError('completion inputs must include a task dimension')
    if not all(torch.isfinite(value).all() for value in (
        progress,
        required_duration,
        task_weights,
    )):
        raise ValueError('completion inputs must contain finite values')
    if (required_duration <= 0).any():
        raise ValueError('required duration must be positive')
    if (task_weights < 0).any():
        raise ValueError('task weight must be non-negative')


def completion_potential(
    progress: torch.Tensor,
    required_duration: torch.Tensor,
    task_weights: torch.Tensor,
) -> torch.Tensor:
    """计算 ``Phi(s)=sum_i omega_i*progress_ratio_i``。"""

    _validate_completion_inputs(progress, required_duration, task_weights)
    progress_ratio = (progress / required_duration).clamp(0, 1)
    return (task_weights * progress_ratio).sum(dim=-1)


def terminal_completion_quality(
    progress: torch.Tensor,
    required_duration: torch.Tensor,
    completed: torch.Tensor,
) -> torch.Tensor:
    """严格按 ``0.6CR + 0.2PCR + 0.2WCR`` 重建终点 Q。"""

    if (
        completed.shape != progress.shape
        or completed.shape != required_duration.shape
    ):
        raise ValueError('terminal completion tensors must share shape')
    if completed.ndim == 0:
        raise ValueError('completion inputs must include a task dimension')
    if completed.dtype != torch.bool:
        raise ValueError('completed flags must use bool dtype')
    if (
        not torch.isfinite(progress).all()
        or not torch.isfinite(required_duration).all()
    ):
        raise ValueError('terminal completion tensors must be finite')
    if (required_duration <= 0).any():
        raise ValueError('required duration must be positive')
    dtype = required_duration.dtype
    completed_float = completed.to(dtype)
    progress_ratio = (progress / required_duration).clamp(0, 1)
    completion_rate = completed_float.mean(dim=-1)
    partial_completion_rate = progress_ratio.mean(dim=-1)
    weighted_completion_rate = (
        completed_float * required_duration
    ).sum(dim=-1) / required_duration.sum(dim=-1)
    return (
        0.6 * completion_rate
        + 0.2 * partial_completion_rate
        + 0.2 * weighted_completion_rate
    )


def build_completion_event_rewards(
    progress: Sequence[torch.Tensor],
    required_duration: torch.Tensor,
    task_weights: torch.Tensor,
    completed: torch.Tensor,
) -> list[torch.Tensor]:
    """构造 telescoping event reward，并在末事件精确校正到终点 Q。"""

    if len(progress) < 2:
        raise ValueError('event rewards require at least two states')
    potentials = [
        completion_potential(value, required_duration, task_weights)
        for value in progress
    ]
    rewards = [
        next_potential - potential
        for potential, next_potential in zip(potentials, potentials[1:])
    ]
    terminal_quality = terminal_completion_quality(
        progress[-1],
        required_duration,
        completed,
    )
    rewards[-1] = rewards[-1] + terminal_quality - potentials[-1]
    return rewards


def time_aware_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    delta_t: torch.Tensor,
    done: torch.Tensor,
    *,
    lambda_base: float = 0.95,
    reference_seconds: float = 5.0,
) -> GAEOutput:
    """以真实物理时间计算 ``gamma=1`` 的 GAE。"""

    expected_shape = rewards.shape
    for name, value in (
        ('values', values),
        ('next_values', next_values),
        ('delta_t', delta_t),
        ('done', done),
    ):
        if value.shape != expected_shape:
            raise ValueError(f'{name} must match rewards shape')
    if rewards.ndim == 0 or rewards.shape[0] == 0:
        raise ValueError('GAE inputs must include a non-empty event dimension')
    if done.dtype != torch.bool:
        raise ValueError('done must use bool dtype')
    numeric = (rewards, values, next_values, delta_t)
    if not all(torch.isfinite(value).all() for value in numeric):
        raise ValueError('GAE inputs must contain finite values')
    if (delta_t <= 0).any():
        raise ValueError('delta_t must be positive')
    if not 0 <= lambda_base <= 1:
        raise ValueError('lambda_base must be in [0, 1]')
    if reference_seconds <= 0:
        raise ValueError('reference_seconds must be positive')

    not_done = (~done).to(rewards.dtype)
    deltas = rewards + not_done * next_values - values
    lambdas = lambda_base ** (delta_t / reference_seconds)
    advantages = torch.empty_like(deltas)
    running = torch.zeros_like(deltas[-1])
    for index in range(deltas.shape[0] - 1, -1, -1):
        running = (
            deltas[index]
            + not_done[index] * lambdas[index] * running
        )
        advantages[index] = running
    return GAEOutput(advantages=advantages, returns=advantages + values)
