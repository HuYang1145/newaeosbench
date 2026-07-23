"""Stage3 热启动的事件级联合 Transformer Actor-Critic。"""

from collections.abc import Iterable, Mapping
import pathlib
from typing import Any, NamedTuple

import torch
from torch import nn

from .actor import (
    ActionEvaluation,
    ActorOutput,
    AutoregressiveJointActor,
)
from .backbone import Stage3FeatureBackbone
from .critic import (
    CentralizedValueCritic,
    EventStateEncoder,
    EventStateEncoding,
)
from .state import EventStateTensors
from .transition import ActionTrace, JointEventAction


class EventActorCriticOutput(NamedTuple):
    actor: ActorOutput
    value: torch.Tensor


class EventJointActorCritic(nn.Module):
    """V2 foundation 模型，不覆盖现有 ``JointModel``。"""

    def __init__(
        self,
        *,
        event_width: int = 256,
        freeze_backbone: bool = True,
        encoder_width: int = 512,
        decoder_width: int = 512,
        num_termination_reasons: int = 8,
        num_event_types: int = 8,
        **stage3_model_kwargs: Any,
    ) -> None:
        super().__init__()
        self.backbone = Stage3FeatureBackbone(
            edge_width=event_width,
            encoder_width=encoder_width,
            decoder_width=decoder_width,
            **stage3_model_kwargs,
        )
        self.state_encoder = EventStateEncoder(
            satellite_width=decoder_width,
            task_width=encoder_width,
            edge_width=event_width,
            event_width=event_width,
            num_termination_reasons=num_termination_reasons,
            num_event_types=num_event_types,
        )
        self.actor = AutoregressiveJointActor(event_width=event_width)
        self.critic = CentralizedValueCritic(event_width=event_width)
        self.backbone_is_frozen = freeze_backbone
        if freeze_backbone:
            self.backbone.freeze()

    def encode(
        self,
        time_steps: torch.Tensor | Iterable[int],
        constellation_sensor_type: torch.Tensor,
        constellation_sensor_enabled: torch.Tensor,
        constellation_data: torch.Tensor,
        constellation_mask: torch.Tensor,
        tasks_sensor_type: torch.Tensor,
        tasks_data: torch.Tensor,
        tasks_mask: torch.Tensor,
        *,
        event_state: EventStateTensors,
    ) -> EventStateEncoding:
        backbone_output = self.backbone(
            time_steps,
            constellation_sensor_type,
            constellation_sensor_enabled,
            constellation_data,
            constellation_mask,
            tasks_sensor_type,
            tasks_data,
            tasks_mask,
        )
        return self.state_encoder(
            backbone_output,
            event_state,
            constellation_mask,
            tasks_mask,
        )

    def act(
        self,
        time_steps: torch.Tensor | Iterable[int],
        constellation_sensor_type: torch.Tensor,
        constellation_sensor_enabled: torch.Tensor,
        constellation_data: torch.Tensor,
        constellation_mask: torch.Tensor,
        tasks_sensor_type: torch.Tensor,
        tasks_data: torch.Tensor,
        tasks_mask: torch.Tensor,
        *,
        event_state: EventStateTensors,
        deterministic: bool = False,
    ) -> EventActorCriticOutput:
        encoding = self.encode(
            time_steps,
            constellation_sensor_type,
            constellation_sensor_enabled,
            constellation_data,
            constellation_mask,
            tasks_sensor_type,
            tasks_data,
            tasks_mask,
            event_state=event_state,
        )
        return EventActorCriticOutput(
            actor=self.actor.sample_actions(
                encoding,
                event_state,
                constellation_mask,
                tasks_mask,
                deterministic=deterministic,
                task_compatibility=(
                    constellation_sensor_type.unsqueeze(-1)
                    == tasks_sensor_type.unsqueeze(1)
                ),
            ),
            value=self.critic(encoding, constellation_mask, tasks_mask),
        )

    def forward(
        self,
        time_steps: torch.Tensor | Iterable[int],
        constellation_sensor_type: torch.Tensor,
        constellation_sensor_enabled: torch.Tensor,
        constellation_data: torch.Tensor,
        constellation_mask: torch.Tensor,
        tasks_sensor_type: torch.Tensor,
        tasks_data: torch.Tensor,
        tasks_mask: torch.Tensor,
        *,
        event_state: EventStateTensors,
        deterministic: bool = False,
    ) -> EventActorCriticOutput:
        return self.act(
            time_steps,
            constellation_sensor_type,
            constellation_sensor_enabled,
            constellation_data,
            constellation_mask,
            tasks_sensor_type,
            tasks_data,
            tasks_mask,
            event_state=event_state,
            deterministic=deterministic,
        )

    def evaluate_actions(
        self,
        time_steps: torch.Tensor | Iterable[int],
        constellation_sensor_type: torch.Tensor,
        constellation_sensor_enabled: torch.Tensor,
        constellation_data: torch.Tensor,
        constellation_mask: torch.Tensor,
        tasks_sensor_type: torch.Tensor,
        tasks_data: torch.Tensor,
        tasks_mask: torch.Tensor,
        *,
        event_state: EventStateTensors,
        action: JointEventAction,
        trace: ActionTrace,
    ) -> tuple[ActionEvaluation, torch.Tensor]:
        encoding = self.encode(
            time_steps,
            constellation_sensor_type,
            constellation_sensor_enabled,
            constellation_data,
            constellation_mask,
            tasks_sensor_type,
            tasks_data,
            tasks_mask,
            event_state=event_state,
        )
        evaluation = self.actor.evaluate_actions(
            encoding,
            event_state,
            constellation_mask,
            tasks_mask,
            action,
            trace,
        )
        value = self.critic(encoding, constellation_mask, tasks_mask)
        return evaluation, value

    @staticmethod
    def _unwrap_checkpoint(
        checkpoint: Any,
    ) -> Mapping[str, torch.Tensor]:
        if not isinstance(checkpoint, Mapping):
            raise ValueError('Stage3 checkpoint must contain a state dict')
        for key in ('model', 'state_dict'):
            candidate = checkpoint.get(key)
            if isinstance(candidate, Mapping):
                return candidate
        if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return checkpoint
        raise ValueError('Stage3 checkpoint has no model state dict')

    def load_stage3_checkpoint(
        self,
        path: str | pathlib.Path,
    ) -> None:
        checkpoint = torch.load(
            pathlib.Path(path),
            map_location='cpu',
            weights_only=False,
        )
        self.backbone.load_stage3_state_dict(
            self._unwrap_checkpoint(checkpoint),
        )

    def unfreeze_last_layers(
        self,
        encoder_layers: int,
        decoder_layers: int,
    ) -> None:
        self.backbone.unfreeze_last_layers(
            encoder_layers,
            decoder_layers,
        )
        self.backbone_is_frozen = not (encoder_layers or decoder_layers)

    def parameter_groups(
        self,
        new_module_lr: float,
        backbone_lr_scale: float = 0.1,
    ) -> list[dict[str, Any]]:
        if new_module_lr <= 0:
            raise ValueError('new module learning rate must be positive')
        if not 0 < backbone_lr_scale <= 1:
            raise ValueError('backbone lr scale must be in (0, 1]')
        backbone_parameters = [
            parameter
            for parameter in self.backbone.transformer.parameters()
            if parameter.requires_grad
        ]
        backbone_ids = {id(parameter) for parameter in backbone_parameters}
        new_parameters = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in backbone_ids
        ]
        groups: list[dict[str, Any]] = []
        if new_parameters:
            groups.append({'params': new_parameters, 'lr': new_module_lr})
        if backbone_parameters:
            groups.append({
                'params': backbone_parameters,
                'lr': new_module_lr * backbone_lr_scale,
            })
        if not groups:
            raise ValueError('model has no trainable parameters')
        return groups
