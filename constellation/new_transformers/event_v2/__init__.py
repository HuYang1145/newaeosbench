"""事件级联合 Transformer V2 的独立实现。"""

from .state import (
    COMMITMENT_SECONDS,
    MAX_TASK_OWNERS,
    EventStateTensors,
    build_commitment_mask,
    build_replan_order,
)
from .reward import (
    GAEOutput,
    build_completion_event_rewards,
    completion_potential,
    terminal_completion_quality,
    time_aware_gae,
)

__all__ = [
    'COMMITMENT_SECONDS',
    'MAX_TASK_OWNERS',
    'EventStateTensors',
    'build_commitment_mask',
    'build_replan_order',
    'GAEOutput',
    'build_completion_event_rewards',
    'completion_potential',
    'terminal_completion_quality',
    'time_aware_gae',
]
