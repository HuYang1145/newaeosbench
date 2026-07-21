"""M3 事件候选、稳健偏好标签与数据门槛统计。"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch


ALLOWED_EVENT_COMMITMENTS = (1, 5, 15, 30, 60)

__all__ = [
    'ALLOWED_EVENT_COMMITMENTS',
    'EventCandidateSpec',
    'EventDecisionPoint',
    'PreferenceAudit',
    'audit_preference_pair',
    'build_event_candidate_specs',
    'find_event_decisions',
    'summarize_preference_audits',
]


@dataclasses.dataclass(frozen=True)
class EventCandidateSpec:
    """同一事件状态下的一个任务与承诺时长组合。"""

    name: str
    task_id: int
    commitment_seconds: int
    action_kind: Literal['stay', 'switch']

    def to_dict(self) -> dict[str, int | str]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class EventDecisionPoint:
    """参考轨迹中一个需要比较 stay/switch 的事件点。"""

    decision_time: int
    satellite_index: int
    stay_task_id: int
    switch_task_id: int
    pattern: str

    def to_dict(self) -> dict[str, int | str]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class PreferenceAudit:
    """一个候选 pair 在稳健标签规则下的审计结果。"""

    first_branch: str
    second_branch: str
    accepted: bool
    reason: str
    better_branch: str | None
    worse_branch: str | None
    margin_300: float | None
    direction_agrees: bool | None

    def to_dict(self) -> dict[str, bool | float | str | None]:
        return dataclasses.asdict(self)


def _candidate_name(
    action_kind: Literal['stay', 'switch'],
    task_id: int,
    commitment_seconds: int,
) -> str:
    return f'{action_kind}_task{task_id}_d{commitment_seconds}'


def build_event_candidate_specs(
    *,
    stay_task_id: int,
    switch_task_id: int,
    commitments: Sequence[int] = ALLOWED_EVENT_COMMITMENTS,
) -> list[EventCandidateSpec]:
    """构造唯一 stay/switch-duration 候选；idle 只允许一秒。"""
    if stay_task_id < -1 or switch_task_id < -1:
        raise ValueError('task ids must be -1 or non-negative')
    if stay_task_id == switch_task_id:
        raise ValueError('stay and switch task ids must differ')
    normalized = tuple(int(value) for value in commitments)
    if (
        not normalized
        or normalized[0] != 1
        or any(value <= 0 for value in normalized)
        or any(a >= b for a, b in zip(normalized, normalized[1:]))
    ):
        raise ValueError(
            'commitments must be strictly increasing and start at one'
        )

    output = []
    for action_kind, task_id in (
        ('stay', stay_task_id),
        ('switch', switch_task_id),
    ):
        durations = (1,) if task_id == -1 else normalized
        for duration in durations:
            output.append(EventCandidateSpec(
                name=_candidate_name(action_kind, task_id, duration),
                task_id=task_id,
                commitment_seconds=duration,
                action_kind=action_kind,
            ))
    return output


def _prefix_metrics(
    branch: Mapping[str, Any],
    horizon: int,
) -> Mapping[str, Any] | None:
    horizons = branch.get('horizons')
    if not isinstance(horizons, Mapping):
        return None
    item = horizons.get(str(horizon))
    if not isinstance(item, Mapping):
        return None
    metrics = item.get('prefix_metrics')
    return metrics if isinstance(metrics, Mapping) else None


def _rejected(
    first: str,
    second: str,
    reason: str,
    *,
    margin_300: float | None = None,
    direction_agrees: bool | None = None,
) -> PreferenceAudit:
    return PreferenceAudit(
        first_branch=first,
        second_branch=second,
        accepted=False,
        reason=reason,
        better_branch=None,
        worse_branch=None,
        margin_300=margin_300,
        direction_agrees=direction_agrees,
    )


def audit_preference_pair(
    first: str,
    second: str,
    branches: Mapping[str, Mapping[str, Any]],
    *,
    min_margin: float = 0.01,
) -> PreferenceAudit:
    """只接受 180/300 秒方向一致且有质量保护的候选 pair。"""
    if min_margin < 0:
        raise ValueError('min_margin must be non-negative')
    if first not in branches or second not in branches:
        raise KeyError('preference branch is missing')
    first_branch = branches[first]
    second_branch = branches[second]
    first_identity = (
        int(first_branch['applied_task_id']),
        int(first_branch['requested_commitment_seconds']),
    )
    second_identity = (
        int(second_branch['applied_task_id']),
        int(second_branch['requested_commitment_seconds']),
    )
    if first_identity == second_identity:
        return _rejected(first, second, 'identical_candidate')

    first_180 = _prefix_metrics(first_branch, 180)
    second_180 = _prefix_metrics(second_branch, 180)
    first_300 = _prefix_metrics(first_branch, 300)
    second_300 = _prefix_metrics(second_branch, 300)
    metrics = (first_180, second_180, first_300, second_300)
    if any(item is None for item in metrics):
        return _rejected(first, second, 'missing_window')
    assert all(item is not None for item in metrics)
    costs = tuple(item.get('prefix_cost') for item in metrics)
    if any(value is None for value in costs):
        return _rejected(first, second, 'missing_window')
    cost_first_180, cost_second_180, cost_first_300, cost_second_300 = (
        float(value) for value in costs
    )
    delta_180 = cost_first_180 - cost_second_180
    delta_300 = cost_first_300 - cost_second_300
    direction_agrees = (
        delta_180 != 0
        and delta_300 != 0
        and (delta_180 < 0) == (delta_300 < 0)
    )
    margin_300 = abs(delta_300)
    if not direction_agrees:
        return _rejected(
            first,
            second,
            'horizon_reversal',
            margin_300=margin_300,
            direction_agrees=False,
        )
    if margin_300 < min_margin:
        return _rejected(
            first,
            second,
            'small_margin',
            margin_300=margin_300,
            direction_agrees=True,
        )

    if delta_300 < 0:
        better_name, worse_name = first, second
        better_metrics, worse_metrics = first_300, second_300
    else:
        better_name, worse_name = second, first
        better_metrics, worse_metrics = second_300, first_300
    quality_keys = ('cr', 'pcr', 'wcr')
    if all(
        float(better_metrics[key]) < float(worse_metrics[key])
        for key in quality_keys
    ):
        return _rejected(
            first,
            second,
            'quality_protection',
            margin_300=margin_300,
            direction_agrees=True,
        )
    return PreferenceAudit(
        first_branch=first,
        second_branch=second,
        accepted=True,
        reason='accepted',
        better_branch=better_name,
        worse_branch=worse_name,
        margin_300=margin_300,
        direction_agrees=True,
    )


def _transition_pattern(previous: int, current: int) -> str:
    if previous == -1 and current >= 0:
        return 'idle_to_task'
    if previous >= 0 and current == -1:
        return 'task_to_idle'
    return 'task_to_task'


def find_event_decisions(
    actions: torch.Tensor,
    *,
    max_decisions: int,
    latest_decision_time: int,
    bin_seconds: int = 300,
) -> list[EventDecisionPoint]:
    """从动作变化点中按时间桶选择分散的事件状态。"""
    if actions.ndim != 2:
        raise ValueError('actions must have shape (time, satellites)')
    if max_decisions <= 0 or bin_seconds <= 0:
        raise ValueError('max_decisions and bin_seconds must be positive')
    last_time = min(int(latest_decision_time), actions.shape[0] - 1)
    if last_time < 1:
        return []
    first_by_bin: dict[int, EventDecisionPoint] = {}
    for time in range(1, last_time + 1):
        for satellite in range(actions.shape[1]):
            previous = int(actions[time - 1, satellite])
            current = int(actions[time, satellite])
            if previous == current:
                continue
            item = EventDecisionPoint(
                decision_time=time,
                satellite_index=satellite,
                stay_task_id=previous,
                switch_task_id=current,
                pattern=_transition_pattern(previous, current),
            )
            first_by_bin.setdefault(time // bin_seconds, item)
    representatives = list(first_by_bin.values())
    if len(representatives) <= max_decisions:
        return representatives
    if max_decisions == 1:
        return [representatives[len(representatives) // 2]]
    indices = [
        round(index * (len(representatives) - 1) / (max_decisions - 1))
        for index in range(max_decisions)
    ]
    return [representatives[index] for index in indices]


def summarize_preference_audits(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """汇总 M3-B 标签门槛需要的原始统计，不在此处放宽门槛。"""
    accepted_scenes: set[int] = set()
    stable_pairs = 0
    comparable_pairs = 0
    agreeing_pairs = 0
    duration_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for record in records:
        branches = record['branches']
        for raw in record['pair_audits']:
            audit = raw if isinstance(raw, PreferenceAudit) else PreferenceAudit(
                **raw
            )
            reason_counts[audit.reason] = reason_counts.get(
                audit.reason, 0
            ) + 1
            if audit.direction_agrees is not None:
                comparable_pairs += 1
                agreeing_pairs += int(audit.direction_agrees)
            if not audit.accepted:
                continue
            stable_pairs += 1
            accepted_scenes.add(int(record['scene_id']))
            assert audit.better_branch is not None
            duration = str(int(
                branches[audit.better_branch][
                    'requested_commitment_seconds'
                ]
            ))
            duration_counts[duration] = duration_counts.get(duration, 0) + 1
    max_fraction = (
        0.0
        if stable_pairs == 0
        else max(duration_counts.values(), default=0) / stable_pairs
    )
    return {
        'accepted_scene_count': len(accepted_scenes),
        'stable_pair_count': stable_pairs,
        'comparable_pair_count': comparable_pairs,
        'agreeing_pair_count': agreeing_pairs,
        'horizon_agreement': (
            None
            if comparable_pairs == 0
            else agreeing_pairs / comparable_pairs
        ),
        'winning_duration_counts': duration_counts,
        'winning_duration_class_count': len(duration_counts),
        'max_winning_duration_fraction': max_fraction,
        'reason_counts': reason_counts,
    }
