"""基于策略 logits 的轻量全局任务 owner 分配。"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment


def resolve_owner_assignments(
    actor_logits: npt.ArrayLike,
    *,
    task_ids: Sequence[int],
    num_satellites: int,
    previous_owner_task_ids: Sequence[int] | None = None,
    continuation_bonus: float = 0.0,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """求解单个环境当前时间步的卫星—任务唯一 owner。

    第 0 列是空动作，后续列依次对应 ``task_ids``。每颗卫星最多选择一个任务，
    每个任务最多分配给一颗卫星；额外的虚拟空动作列允许任意卫星保持空闲。
    """
    if continuation_bonus < 0:
        raise ValueError('continuation_bonus must be non-negative')

    logits = np.asarray(actor_logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError('actor_logits must be a 2D array')
    if not 0 <= num_satellites <= logits.shape[0]:
        raise ValueError('num_satellites is outside actor_logits')

    num_tasks = len(task_ids)
    if logits.shape[1] < num_tasks + 1:
        raise ValueError('actor_logits does not contain every task')

    actions = np.zeros(logits.shape[0], dtype=np.int64)
    owner_task_ids = np.full(logits.shape[0], -1, dtype=np.int64)
    if num_satellites == 0 or num_tasks == 0:
        return actions, owner_task_ids

    active_logits = logits[:num_satellites, :num_tasks + 1]
    null_logits = active_logits[:, 0]
    if not np.isfinite(null_logits).all():
        raise ValueError('null action logits must be finite')

    task_advantages = active_logits[:, 1:] - null_logits[:, None]
    if previous_owner_task_ids is not None:
        if len(previous_owner_task_ids) < num_satellites:
            raise ValueError('previous owner list is shorter than satellites')
        task_to_column = {
            int(task_id): column
            for column, task_id in enumerate(task_ids)
        }
        for satellite_id, previous_task_id in enumerate(
            previous_owner_task_ids[:num_satellites]
        ):
            column = task_to_column.get(int(previous_task_id))
            if column is not None:
                task_advantages[satellite_id, column] += continuation_bonus

    task_advantages = np.nan_to_num(
        task_advantages,
        nan=-1e12,
        neginf=-1e12,
        posinf=1e12,
    )
    scores = np.concatenate([
        task_advantages,
        np.zeros((num_satellites, num_satellites), dtype=np.float64),
    ], axis=1)
    row_ids, column_ids = linear_sum_assignment(scores, maximize=True)

    for satellite_id, column in zip(row_ids.tolist(), column_ids.tolist()):
        if column >= num_tasks or scores[satellite_id, column] <= 0:
            continue
        actions[satellite_id] = column + 1
        owner_task_ids[satellite_id] = int(task_ids[column])

    return actions, owner_task_ids
