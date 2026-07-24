"""V2 同步 PPO：半马尔可夫优势、联合概率和原子回滚。"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import NamedTuple

import torch

from .model import EventJointActorCritic
from .reward import time_aware_gae
from .rollout import (
    StoredEventStep,
    evaluate_rollout_steps,
    replay_rollout_log_probs,
)


class PPOUpdateRejected(RuntimeError):
    """当前更新违反预注册稳定性门槛，模型已回滚。"""


@dataclass(frozen=True)
class PPOConfig:
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 1.0
    max_kl: float = 0.03
    ppo_epochs: int = 4
    minibatch_events: int = 64
    lambda_base: float = 0.95
    reference_seconds: float = 5.0
    replay_atol: float = 1e-6
    normalize_advantages: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.clip_ratio < 1:
            raise ValueError('PPO clip ratio must be in (0, 1)')
        if self.value_coefficient < 0 or self.entropy_coefficient < 0:
            raise ValueError('PPO loss coefficients must be non-negative')
        if self.max_grad_norm <= 0 or self.max_kl <= 0:
            raise ValueError('PPO stability limits must be positive')
        if self.ppo_epochs <= 0 or self.minibatch_events <= 0:
            raise ValueError('PPO epochs and minibatch size must be positive')
        if not 0 <= self.lambda_base <= 1:
            raise ValueError('GAE lambda base must be in [0, 1]')
        if self.reference_seconds <= 0 or self.replay_atol < 0:
            raise ValueError('PPO time scale and replay tolerance are invalid')


class ClippedPPOObjective(NamedTuple):
    policy_loss: torch.Tensor
    ratio: torch.Tensor
    clip_fraction: torch.Tensor
    approx_kl: torch.Tensor


class RolloutTargets(NamedTuple):
    advantages: torch.Tensor
    returns: torch.Tensor


@dataclass(frozen=True)
class PPOUpdateMetrics:
    total_loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    gradient_norm: float
    advantage_mean: float
    advantage_std: float
    frozen_parameter_changes: int
    policy_version: int
    completed_epochs: int
    early_stopped: bool


def clipped_ppo_objective(
    new_log_prob: torch.Tensor,
    behavior_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_ratio: float,
) -> ClippedPPOObjective:
    if not (
        new_log_prob.shape
        == behavior_log_prob.shape
        == advantages.shape
    ):
        raise ValueError('PPO policy tensors must share shape')
    if not all(torch.isfinite(value).all() for value in (
        new_log_prob,
        behavior_log_prob,
        advantages,
    )):
        raise ValueError('PPO policy tensors must be finite')
    if not 0 < clip_ratio < 1:
        raise ValueError('PPO clip ratio must be in (0, 1)')
    log_ratio = new_log_prob - behavior_log_prob
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped_ratio = ratio.clamp(1 - clip_ratio, 1 + clip_ratio)
    clipped = clipped_ratio * advantages
    return ClippedPPOObjective(
        policy_loss=-torch.minimum(unclipped, clipped).mean(),
        ratio=ratio,
        clip_fraction=((ratio - 1).abs() > clip_ratio).to(
            ratio.dtype
        ).mean(),
        approx_kl=((ratio - 1) - log_ratio).mean(),
    )


def compute_rollout_targets(
    steps: list[StoredEventStep],
    *,
    lambda_base: float,
    reference_seconds: float,
    normalize_advantages: bool,
) -> RolloutTargets:
    if not steps:
        raise ValueError('rollout targets require at least one event')
    for step in steps:
        step.validate()
    advantages = torch.empty(len(steps), dtype=torch.float32)
    returns = torch.empty_like(advantages)
    groups: dict[tuple[int, int], list[int]] = {}
    for index, step in enumerate(steps):
        groups.setdefault(
            (step.environment_index, step.episode_id),
            [],
        ).append(index)
    for indices in groups.values():
        indices.sort(key=lambda index: steps[index].event_index)
        event_indices = [steps[index].event_index for index in indices]
        if len(event_indices) != len(set(event_indices)):
            raise ValueError('rollout contains duplicate event indices')
        output = time_aware_gae(
            rewards=torch.stack([steps[index].reward for index in indices]),
            values=torch.stack([steps[index].value for index in indices]),
            next_values=torch.stack([
                steps[index].next_value for index in indices
            ]),
            delta_t=torch.stack([steps[index].delta_t for index in indices]),
            done=torch.stack([steps[index].done for index in indices]),
            lambda_base=lambda_base,
            reference_seconds=reference_seconds,
        )
        advantages[indices] = output.advantages
        returns[indices] = output.returns
    if normalize_advantages:
        standard_deviation = advantages.std(unbiased=False)
        if standard_deviation > 1e-8:
            advantages = (
                advantages - advantages.mean()
            ) / (standard_deviation + 1e-8)
        else:
            advantages = advantages - advantages.mean()
    if not torch.isfinite(advantages).all() or not torch.isfinite(returns).all():
        raise ValueError('rollout targets must be finite')
    return RolloutTargets(advantages=advantages, returns=returns)


def event_action_component_counts(
    steps: list[StoredEventStep],
) -> torch.Tensor:
    """返回每个联合事件实际进入 entropy 的条件动作数量。"""

    counts = torch.tensor([
        int(step.trace.termination_mask.sum().item())
        + int((step.trace.action_order >= 0).sum().item())
        + int((step.action.commitment_indices >= 0).sum().item())
        for step in steps
    ], dtype=torch.float32)
    if (counts <= 0).any():
        raise ValueError('each policy event needs at least one action component')
    return counts


class SynchronousPPOTrainer:
    def __init__(
        self,
        *,
        model: EventJointActorCritic,
        optimizer: torch.optim.Optimizer,
        config: PPOConfig,
        device: torch.device,
        amp_enabled: bool = False,
        amp_dtype: torch.dtype = torch.bfloat16,
        scaler: torch.amp.GradScaler | None = None,
        require_fully_frozen_backbone: bool = True,
        verify_behavior_replay: bool = True,
    ) -> None:
        if require_fully_frozen_backbone and not model.backbone_is_frozen:
            raise ValueError('V2-1 requires a fully frozen Stage3 backbone')
        if require_fully_frozen_backbone and any(
            parameter.requires_grad
            for parameter in model.backbone.transformer.parameters()
        ):
            raise ValueError('Stage3 transformer parameters must be frozen')
        self.model = model.to(device)
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.amp_enabled = amp_enabled
        self.amp_dtype = amp_dtype
        self.verify_behavior_replay = verify_behavior_replay
        self.scaler = scaler or torch.amp.GradScaler(
            device.type,
            enabled=False,
        )
        self.policy_version = 0
        self._frozen_reference = {
            name: parameter.detach().cpu().clone()
            for name, parameter in (
                model.backbone.transformer.named_parameters()
            )
            if not parameter.requires_grad
        }

    def _trainable_snapshot(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }

    def _restore_trainable(
        self,
        snapshot: dict[str, torch.Tensor],
    ) -> None:
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(snapshot[name].to(parameter.device))

    def _frozen_parameter_changes(self) -> int:
        changes = 0
        current = dict(
            self.model.backbone.transformer.named_parameters(),
        )
        for name, reference in self._frozen_reference.items():
            if not torch.equal(current[name].detach().cpu(), reference):
                changes += 1
        return changes

    def _reject(self, message: str) -> None:
        raise PPOUpdateRejected(message)

    def update(self, steps: list[StoredEventStep]) -> PPOUpdateMetrics:
        if not steps:
            raise ValueError('PPO update requires rollout events')
        for step in steps:
            step.validate()
        behavior = torch.stack([
            step.behavior_log_prob for step in steps
        ]).to(torch.float32)
        if self.verify_behavior_replay:
            replay = replay_rollout_log_probs(
                self.model,
                steps,
                device=self.device,
                amp_enabled=self.amp_enabled,
                amp_dtype=self.amp_dtype,
            )
            replay_error = (replay - behavior).abs().max().item()
            if replay_error > self.config.replay_atol:
                self._reject(
                    f'behavior log-prob replay mismatch: {replay_error:.8g}'
                )

        targets = compute_rollout_targets(
            steps,
            lambda_base=self.config.lambda_base,
            reference_seconds=self.config.reference_seconds,
            normalize_advantages=self.config.normalize_advantages,
        )
        model_snapshot = self._trainable_snapshot()
        optimizer_snapshot = copy.deepcopy(self.optimizer.state_dict())
        was_training = self.model.training
        self.model.eval()
        totals: list[float] = []
        policies: list[float] = []
        values: list[float] = []
        entropies: list[float] = []
        kls: list[float] = []
        clips: list[float] = []
        gradients: list[float] = []
        completed_epochs = 0
        early_stopped = False
        post_kl = torch.zeros((), dtype=torch.float32)
        try:
            for _ in range(self.config.ppo_epochs):
                epoch_model_snapshot = self._trainable_snapshot()
                epoch_optimizer_snapshot = copy.deepcopy(
                    self.optimizer.state_dict()
                )
                epoch_totals: list[float] = []
                epoch_policies: list[float] = []
                epoch_values: list[float] = []
                epoch_entropies: list[float] = []
                epoch_kls: list[float] = []
                epoch_clips: list[float] = []
                epoch_gradients: list[float] = []
                epoch_kl_exceeded = False
                permutation = torch.randperm(len(steps))
                for start in range(0, len(steps), self.config.minibatch_events):
                    indices = permutation[
                        start:start + self.config.minibatch_events
                    ]
                    minibatch = [steps[index] for index in indices.tolist()]
                    with torch.autocast(
                        device_type=self.device.type,
                        enabled=self.amp_enabled,
                        dtype=self.amp_dtype,
                    ):
                        new_log_prob, entropy, new_value = evaluate_rollout_steps(
                            self.model,
                            minibatch,
                            device=self.device,
                        )
                        behavior_batch = behavior[indices].to(self.device)
                        advantage_batch = targets.advantages[indices].to(
                            self.device
                        )
                        return_batch = targets.returns[indices].to(self.device)
                        objective = clipped_ppo_objective(
                            new_log_prob,
                            behavior_batch,
                            advantage_batch,
                            clip_ratio=self.config.clip_ratio,
                        )
                        value_loss = 0.5 * (
                            new_value - return_batch
                        ).square().mean()
                        entropy_counts = event_action_component_counts(
                            minibatch
                        ).to(self.device)
                        entropy_mean = (entropy / entropy_counts).mean()
                        total_loss = (
                            objective.policy_loss
                            + self.config.value_coefficient * value_loss
                            - self.config.entropy_coefficient * entropy_mean
                        )
                    if not all(torch.isfinite(value).all() for value in (
                        total_loss,
                        objective.approx_kl,
                        entropy_mean,
                        value_loss,
                    )):
                        self._reject('PPO loss contains a non-finite value')
                    if objective.approx_kl > self.config.max_kl:
                        epoch_kl_exceeded = True
                        break

                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    trainable_parameters = [
                        parameter
                        for parameter in self.model.parameters()
                        if parameter.requires_grad and parameter.grad is not None
                    ]
                    if not trainable_parameters or any(
                        not torch.isfinite(parameter.grad).all()
                        for parameter in trainable_parameters
                    ):
                        self._reject('PPO gradient contains a non-finite value')
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        self.config.max_grad_norm,
                    )
                    if not torch.isfinite(gradient_norm):
                        self._reject('PPO gradient norm is non-finite')
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if any(
                        not torch.isfinite(parameter).all()
                        for parameter in trainable_parameters
                    ):
                        self._reject('PPO update produced a non-finite parameter')

                    epoch_totals.append(float(total_loss.detach().cpu()))
                    epoch_policies.append(
                        float(objective.policy_loss.detach().cpu())
                    )
                    epoch_values.append(float(value_loss.detach().cpu()))
                    epoch_entropies.append(float(entropy_mean.detach().cpu()))
                    epoch_kls.append(float(objective.approx_kl.detach().cpu()))
                    epoch_clips.append(
                        float(objective.clip_fraction.detach().cpu())
                    )
                    epoch_gradients.append(float(gradient_norm.detach().cpu()))

                # 局部 minibatch KL 只负责停止后续梯度步；是否回滚必须由
                # 完整 rollout 的联合 KL 决定，避免因单个高方差 minibatch
                # 丢弃此前仍处于全局门槛内的安全更新。
                epoch_post_log_prob = replay_rollout_log_probs(
                    self.model,
                    steps,
                    device=self.device,
                    amp_enabled=self.amp_enabled,
                    amp_dtype=self.amp_dtype,
                )
                epoch_log_ratio = epoch_post_log_prob - behavior
                epoch_post_ratio = torch.exp(epoch_log_ratio)
                epoch_post_kl = (
                    (epoch_post_ratio - 1) - epoch_log_ratio
                ).mean()
                if not torch.isfinite(epoch_post_kl):
                    self._reject('PPO post-epoch KL is non-finite')
                global_kl_exceeded = bool(
                    epoch_post_kl > self.config.max_kl
                )

                if global_kl_exceeded:
                    self._restore_trainable(epoch_model_snapshot)
                    self.optimizer.load_state_dict(epoch_optimizer_snapshot)
                    early_stopped = True
                    break

                if epoch_totals:
                    totals.extend(epoch_totals)
                    policies.extend(epoch_policies)
                    values.extend(epoch_values)
                    entropies.extend(epoch_entropies)
                    kls.extend(epoch_kls)
                    clips.extend(epoch_clips)
                    gradients.extend(epoch_gradients)
                    completed_epochs += 1
                    post_kl = epoch_post_kl.detach().cpu()
                if epoch_kl_exceeded:
                    early_stopped = True
                    break

            if completed_epochs == 0:
                self._reject(
                    'first PPO epoch exceeded the registered KL limit: '
                    f'full_kl={float(epoch_post_kl):.8g}, '
                    f'applied_minibatches={len(epoch_totals)}'
                )
            frozen_changes = self._frozen_parameter_changes()
            if frozen_changes:
                self._reject('frozen Stage3 parameters changed during PPO')
        except Exception:
            self._restore_trainable(model_snapshot)
            self.optimizer.load_state_dict(optimizer_snapshot)
            self.model.train(was_training)
            raise
        self.model.train(was_training)
        self.policy_version += 1
        return PPOUpdateMetrics(
            total_loss=sum(totals) / len(totals),
            policy_loss=sum(policies) / len(policies),
            value_loss=sum(values) / len(values),
            entropy=sum(entropies) / len(entropies),
            approx_kl=max(kls + [float(post_kl)]),
            clip_fraction=sum(clips) / len(clips),
            gradient_norm=max(gradients),
            advantage_mean=float(targets.advantages.mean()),
            advantage_std=float(targets.advantages.std(unbiased=False)),
            frozen_parameter_changes=0,
            policy_version=self.policy_version,
            completed_epochs=completed_epochs,
            early_stopped=early_stopped,
        )
