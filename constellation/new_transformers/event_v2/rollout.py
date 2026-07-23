"""V2 同步事件 rollout、完整行为 trace 与 learner 重放。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol

import torch

from .basilisk_runtime import RuntimeStep
from .model import EventJointActorCritic
from .observation import EventPolicyObservation
from .state import EventStateTensors
from .transition import ActionTrace, JointEventAction


class EventRuntime(Protocol):
    def reset(self) -> EventPolicyObservation:
        raise NotImplementedError

    def step(self, action: JointEventAction) -> RuntimeStep:
        raise NotImplementedError


@dataclass
class SynchronousRuntimeSlot:
    environment_index: int
    episode_id: int
    observation: EventPolicyObservation
    runtime: EventRuntime
    event_index: int = 0
    finished: bool = False

    def __post_init__(self) -> None:
        if self.environment_index < 0 or self.episode_id < 0:
            raise ValueError('runtime identifiers must be non-negative')
        self.observation.validate()


class StoredEventStep(NamedTuple):
    environment_index: int
    episode_id: int
    event_index: int
    observation: EventPolicyObservation
    action: JointEventAction
    trace: ActionTrace
    behavior_log_prob: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor
    delta_t: torch.Tensor
    next_observation: EventPolicyObservation | None
    next_value: torch.Tensor
    done: torch.Tensor
    policy_version: int

    def validate(self) -> None:
        if min(
            self.environment_index,
            self.episode_id,
            self.event_index,
            self.policy_version,
        ) < 0:
            raise ValueError('rollout identifiers must be non-negative')
        self.observation.validate()
        if self.observation.batch_size != 1:
            raise ValueError('stored rollout observations must contain one scene')
        satellite_shape = (1, self.observation.num_satellites)
        for name, value in self.action._asdict().items():
            if value.shape != satellite_shape:
                raise ValueError(f'rollout action {name} has an invalid shape')
        trace_shapes = {
            'action_order': satellite_shape,
            'termination_mask': satellite_shape,
            'task_masks': (
                1,
                self.observation.num_satellites,
                self.observation.num_tasks + 1,
            ),
            'commitment_masks': (
                1,
                self.observation.num_satellites,
                5,
            ),
            'owner_state': (
                1,
                self.observation.num_satellites,
                self.observation.num_tasks,
            ),
        }
        for name, shape in trace_shapes.items():
            if getattr(self.trace, name).shape != shape:
                raise ValueError(f'rollout trace {name} has an invalid shape')
        for name, value in (
            ('behavior log-prob', self.behavior_log_prob),
            ('value', self.value),
            ('reward', self.reward),
            ('delta_t', self.delta_t),
            ('next value', self.next_value),
        ):
            if value.ndim != 0 or not torch.isfinite(value):
                raise ValueError(f'{name} must be a finite scalar')
        if self.delta_t <= 0:
            raise ValueError('delta_t must be positive')
        if self.done.ndim != 0 or self.done.dtype != torch.bool:
            raise ValueError('done must be a boolean scalar')
        if bool(self.done):
            if self.next_observation is not None or self.next_value != 0:
                raise ValueError('terminal transition must use zero bootstrap')
        elif self.next_observation is None:
            raise ValueError('non-terminal transition needs a next observation')


def _map_namedtuple(value: Any, function) -> Any:
    if isinstance(value, torch.Tensor):
        return function(value)
    if isinstance(value, tuple) and hasattr(value, '_fields'):
        return type(value)(*(_map_namedtuple(item, function) for item in value))
    return value


def _to_device(value: Any, device: torch.device) -> Any:
    return _map_namedtuple(
        value,
        lambda tensor: tensor.to(
            device,
            non_blocking=device.type == 'cuda',
        ).clone(),
    )


def _detach_cpu(value: Any) -> Any:
    return _map_namedtuple(
        value,
        lambda tensor: tensor.detach().to('cpu').clone(),
    )


def _value_only(
    model: EventJointActorCritic,
    observation: EventPolicyObservation,
) -> torch.Tensor:
    encoding = model.encode(
        *observation.model_args(),
        event_state=observation.event_state,
    )
    return model.critic(
        encoding,
        observation.constellation_mask,
        observation.tasks_mask,
    )


def collect_synchronous_rollout(
    model: EventJointActorCritic,
    slots: Sequence[SynchronousRuntimeSlot],
    *,
    target_events: int,
    policy_version: int,
    device: torch.device,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> list[StoredEventStep]:
    """轮询同步环境，达到事件数或所有 scene 终止时返回。"""

    if target_events <= 0:
        raise ValueError('target events must be positive')
    if policy_version < 0:
        raise ValueError('policy version must be non-negative')
    if not slots or not any(not slot.finished for slot in slots):
        raise ValueError('at least one active runtime is required')

    collected: list[StoredEventStep] = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            while len(collected) < target_events:
                progressed = False
                for slot in slots:
                    if slot.finished:
                        continue
                    progressed = True
                    observation_device = slot.observation.to(device)
                    with torch.autocast(
                        device_type=device.type,
                        enabled=amp_enabled,
                        dtype=amp_dtype,
                    ):
                        output = model.act(
                            *observation_device.model_args(),
                            event_state=observation_device.event_state,
                            deterministic=False,
                        )
                    if (
                        not torch.isfinite(output.actor.log_prob).all()
                        or not torch.isfinite(output.value).all()
                    ):
                        raise RuntimeError('policy produced a non-finite rollout value')
                    action_cpu = _detach_cpu(output.actor.action)
                    result = slot.runtime.step(action_cpu)
                    if result.delta_t <= 0:
                        raise RuntimeError('rollout delta_t must be positive')
                    if result.invalid_action_count != 0:
                        raise RuntimeError('runtime reported an invalid action')
                    if not torch.isfinite(torch.tensor(result.reward)):
                        raise RuntimeError('runtime returned a non-finite reward')
                    if not result.done and result.observation is None:
                        raise RuntimeError('non-terminal runtime step has no observation')

                    next_value = torch.tensor(0.)
                    if not result.done:
                        assert result.observation is not None
                        result.observation.validate()
                        next_observation_device = result.observation.to(device)
                        with torch.autocast(
                            device_type=device.type,
                            enabled=amp_enabled,
                            dtype=amp_dtype,
                        ):
                            next_value_device = _value_only(
                                model,
                                next_observation_device,
                            )
                        if not torch.isfinite(next_value_device).all():
                            raise RuntimeError('policy produced a non-finite bootstrap')
                        next_value = _detach_cpu(next_value_device[0])

                    step = StoredEventStep(
                        environment_index=slot.environment_index,
                        episode_id=slot.episode_id,
                        event_index=slot.event_index,
                        observation=_detach_cpu(slot.observation),
                        action=action_cpu,
                        trace=_detach_cpu(output.actor.trace),
                        behavior_log_prob=_detach_cpu(output.actor.log_prob[0]),
                        value=_detach_cpu(output.value[0]),
                        reward=torch.tensor(float(result.reward)),
                        delta_t=torch.tensor(float(result.delta_t)),
                        next_observation=(
                            None
                            if result.observation is None
                            else _detach_cpu(result.observation)
                        ),
                        next_value=next_value,
                        done=torch.tensor(bool(result.done)),
                        policy_version=policy_version,
                    )
                    step.validate()
                    collected.append(step)
                    slot.event_index += 1
                    if result.done:
                        slot.finished = True
                    else:
                        assert result.observation is not None
                        slot.observation = result.observation
                    if len(collected) >= target_events:
                        break
                if not progressed:
                    break
    finally:
        model.train(was_training)
    return collected


def replay_rollout_log_probs(
    model: EventJointActorCritic,
    steps: Sequence[StoredEventStep],
    *,
    device: torch.device,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """使用行为时 trace 重放联合 log-prob，保持原事件顺序。"""

    if not steps:
        raise ValueError('at least one rollout step is required')
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type,
                enabled=amp_enabled,
                dtype=amp_dtype,
            ):
                replayed, _, _ = evaluate_rollout_steps(
                    model,
                    steps,
                    device=device,
                )
    finally:
        model.train(was_training)
    result = replayed.detach().cpu()
    if not torch.isfinite(result).all():
        raise RuntimeError('learner replay produced non-finite log-prob')
    return result


def evaluate_rollout_steps(
    model: EventJointActorCritic,
    steps: Sequence[StoredEventStep],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """逐事件以 ``batch=1`` 重放，保持行为策略的数值执行路径。"""

    if not steps:
        raise ValueError('at least one rollout step is required')
    outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for step in steps:
        step.validate()
        # rollout 在 inference_mode 中采集；这里 clone 成普通张量，确保
        # learner 反向传播不会保存 inference tensor。
        observation = _to_device(step.observation, device)
        action = _to_device(step.action, device)
        trace = _to_device(step.trace, device)
        evaluation, values = model.evaluate_actions(
            *observation.model_args(),
            event_state=observation.event_state,
            action=action,
            trace=trace,
        )
        outputs.append((
            evaluation.log_prob[0],
            evaluation.entropy[0],
            values[0],
        ))
    return tuple(
        torch.stack([values[field] for values in outputs])
        for field in range(3)
    )
