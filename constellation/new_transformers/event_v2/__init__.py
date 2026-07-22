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
    completion_task_weights,
    terminal_completion_quality,
    time_aware_gae,
)
from .transition import (
    TRANSITION_SCHEMA_VERSION,
    ActionTrace,
    EventTransition,
    JointEventAction,
    transition_schema_definition,
    transition_schema_fingerprint,
)
from .backbone import Stage3BackboneOutput, Stage3FeatureBackbone
from .critic import (
    CentralizedValueCritic,
    EventStateEncoder,
    EventStateEncoding,
)
from .actor import (
    ActionEvaluation,
    ActorOutput,
    AutoregressiveJointActor,
)
from .model import EventActorCriticOutput, EventJointActorCritic
from .dataset import (
    EventV2OfflineDataset,
    OfflineEventBatch,
    OfflineEventTargets,
    build_capped_owner_counts,
    build_commitment_targets,
    compress_expert_actions_to_events,
)
from .offline import OfflineLosses, event_v2_offline_loss

__all__ = [
    'COMMITMENT_SECONDS',
    'MAX_TASK_OWNERS',
    'EventStateTensors',
    'build_commitment_mask',
    'build_replan_order',
    'GAEOutput',
    'build_completion_event_rewards',
    'completion_potential',
    'completion_task_weights',
    'terminal_completion_quality',
    'time_aware_gae',
    'TRANSITION_SCHEMA_VERSION',
    'ActionTrace',
    'EventTransition',
    'JointEventAction',
    'transition_schema_definition',
    'transition_schema_fingerprint',
    'Stage3BackboneOutput',
    'Stage3FeatureBackbone',
    'CentralizedValueCritic',
    'EventStateEncoder',
    'EventStateEncoding',
    'ActionEvaluation',
    'ActorOutput',
    'AutoregressiveJointActor',
    'EventActorCriticOutput',
    'EventJointActorCritic',
    'EventV2OfflineDataset',
    'OfflineEventBatch',
    'OfflineEventTargets',
    'build_capped_owner_counts',
    'build_commitment_targets',
    'compress_expert_actions_to_events',
    'OfflineLosses',
    'event_v2_offline_loss',
]
