"""从 Stage3 checkpoint 提取 V2 所需的任务、卫星与边 token。"""

from collections.abc import Iterable, Mapping
from typing import NamedTuple

import einops
import torch
from torch import nn

from ..feasibility import (
    apply_feasibility_penalty,
    apply_feasibility_threshold,
)
from ..model import Transformer


class Stage3BackboneOutput(NamedTuple):
    task_tokens: torch.Tensor
    satellite_tokens: torch.Tensor
    edge_features: torch.Tensor
    teacher_null_logits: torch.Tensor
    teacher_task_logits: torch.Tensor
    feasibility_logits: torch.Tensor | None


class Stage3FeatureBackbone(nn.Module):
    """复用 Stage3 表征并保留独立可训练的 V2 edge projection。"""

    def __init__(
        self,
        *,
        edge_width: int,
        time_embedding_dim: int = 64,
        sensor_type_embedding_dim: int = 128,
        tasks_data_embedding_dim: int = 128,
        encoder_width: int = 512,
        encoder_depth: int = 12,
        encoder_num_heads: int = 16,
        sensor_enabled_embedding_dim: int = 128,
        constellation_data_embedding_dim: int = 128,
        decoder_width: int = 512,
        decoder_depth: int = 12,
        decoder_num_heads: int = 16,
        use_constraint_module: bool = True,
        use_sdpa: bool = False,
        feasibility_threshold: float | None = None,
        feasibility_penalty_threshold: float | None = None,
        feasibility_penalty_strength: float | None = None,
        use_assignment_head: bool = False,
        assignment_head_hidden_width: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()
        if edge_width <= 0:
            raise ValueError('edge_width must be positive')
        unsupported = {
            name: value
            for name, value in kwargs.items()
            if name in {'use_temporal_adapter'} and value
        }
        if unsupported:
            raise ValueError(
                'Stage3FeatureBackbone does not consume temporal adapters'
            )
        self.transformer = Transformer(
            time_embedding_dim=time_embedding_dim,
            sensor_type_embedding_dim=sensor_type_embedding_dim,
            tasks_data_embedding_dim=tasks_data_embedding_dim,
            encoder_width=encoder_width,
            encoder_depth=encoder_depth,
            encoder_num_heads=encoder_num_heads,
            sensor_enabled_embedding_dim=sensor_enabled_embedding_dim,
            constellation_data_embedding_dim=constellation_data_embedding_dim,
            decoder_width=decoder_width,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            use_constraint_module=use_constraint_module,
            use_sdpa=use_sdpa,
            feasibility_threshold=feasibility_threshold,
            feasibility_penalty_threshold=feasibility_penalty_threshold,
            feasibility_penalty_strength=feasibility_penalty_strength,
            use_assignment_head=use_assignment_head,
            assignment_head_hidden_width=assignment_head_hidden_width,
        )
        self.satellite_edge_projection = nn.Linear(
            decoder_width,
            edge_width,
        )
        self.task_edge_projection = nn.Linear(encoder_width, edge_width)

    def freeze(self) -> None:
        """冻结 Stage3 checkpoint 参数，不冻结 V2 新 edge projection。"""

        self.transformer.requires_grad_(False)
        self.satellite_edge_projection.requires_grad_(True)
        self.task_edge_projection.requires_grad_(True)

    def unfreeze_last_layers(
        self,
        encoder_layers: int,
        decoder_layers: int,
    ) -> None:
        """仅解冻 Stage3 Encoder/Decoder 尾部层及其最终 LayerNorm。"""

        encoder_blocks = list(self.transformer._encoder._blocks.children())
        decoder_blocks = list(self.transformer._decoder._blocks.children())
        if (
            not 0 <= encoder_layers <= len(encoder_blocks)
            or not 0 <= decoder_layers <= len(decoder_blocks)
        ):
            raise ValueError('unfreeze layer count is outside backbone depth')
        self.transformer.requires_grad_(False)
        for block in encoder_blocks[len(encoder_blocks) - encoder_layers:]:
            block.requires_grad_(True)
        for block in decoder_blocks[len(decoder_blocks) - decoder_layers:]:
            block.requires_grad_(True)
        if encoder_layers:
            self.transformer._encoder._norm.requires_grad_(True)
        if decoder_layers:
            self.transformer._decoder._norm.requires_grad_(True)
        self.satellite_edge_projection.requires_grad_(True)
        self.task_edge_projection.requires_grad_(True)

    def load_stage3_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
    ) -> None:
        """严格加载 `Model`、DDP 或裸 `Transformer` 的 Stage3 参数。"""

        expected = set(self.transformer.state_dict())
        normalized: dict[str, torch.Tensor] = {}
        unexpected: list[str] = []
        for original_key, value in state_dict.items():
            key = original_key
            if key.startswith('module.'):
                key = key[len('module.'):]
            scoped_to_transformer = False
            if key.startswith('_transformer.'):
                key = key[len('_transformer.'):]
                scoped_to_transformer = True
            elif key.startswith('transformer.'):
                key = key[len('transformer.'):]
                scoped_to_transformer = True
            if key not in expected:
                if scoped_to_transformer:
                    unexpected.append(original_key)
                continue
            normalized[key] = value

        missing = sorted(expected - set(normalized))
        if missing:
            raise ValueError(
                'missing Stage3 backbone keys: ' + ', '.join(missing[:8])
            )
        if unexpected:
            raise ValueError(
                'unexpected Stage3 backbone keys: '
                + ', '.join(sorted(unexpected)[:8])
            )
        self.transformer.load_state_dict(normalized, strict=True)

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
    ) -> Stage3BackboneOutput:
        """复现 Stage3 forward，并额外返回中间 token。"""

        if isinstance(time_steps, torch.Tensor):
            time_steps = time_steps.flatten().tolist()
        else:
            time_steps = list(time_steps)

        if self.transformer._use_constraint_module:
            _, raw_time_mask = self.transformer._time_model.predict(
                time_steps,
                constellation_data,
                constellation_mask,
                tasks_data,
                tasks_mask,
            )
            feasibility_logits: torch.Tensor | None = raw_time_mask
            time_mask = raw_time_mask.clamp_min(-100)
            time_mask = self.transformer._time_projection(
                einops.rearrange(time_mask, 'b ns nt -> b ns nt 1')
            )
            time_mask = einops.rearrange(
                time_mask,
                'b ns nt 1 -> b ns nt',
            )
        else:
            feasibility_logits = None
            time_mask = constellation_data.new_zeros(
                constellation_data.shape[0],
                constellation_data.shape[1],
                tasks_data.shape[1],
            )

        time_embedding = self.transformer._time_embedding[time_steps]
        task_sensor_embedding = self.transformer._sensor_type_embedding(
            tasks_sensor_type,
        )
        task_tokens = self.transformer._encoder(
            time_embedding,
            task_sensor_embedding,
            tasks_data,
            tasks_mask,
        )
        satellite_sensor_embedding = self.transformer._sensor_type_embedding(
            constellation_sensor_type,
        )
        decoder_output = self.transformer._decoder(
            time_embedding,
            satellite_sensor_embedding,
            constellation_sensor_enabled,
            constellation_data,
            constellation_mask,
            task_tokens,
            tasks_mask,
            time_mask,
        )
        if not isinstance(decoder_output, tuple):
            raise RuntimeError('Stage3 decoder must return logits and tokens')
        null_logits, task_logits, satellite_tokens = decoder_output
        task_logits = apply_feasibility_threshold(
            task_logits,
            feasibility_logits,
            self.transformer._feasibility_threshold,
        )
        task_logits = apply_feasibility_penalty(
            task_logits,
            feasibility_logits,
            threshold=self.transformer._feasibility_penalty_threshold,
            strength=self.transformer._feasibility_penalty_strength,
        )
        if self.transformer._assignment_head is not None:
            task_logits = self.transformer._assignment_head(
                einops.rearrange(null_logits, 'b ns -> b ns 1'),
                task_logits,
                satellite_tokens,
                task_tokens,
                constellation_mask,
                tasks_mask,
            )

        edge_features = (
            self.satellite_edge_projection(satellite_tokens).unsqueeze(2)
            + self.task_edge_projection(task_tokens).unsqueeze(1)
        )
        return Stage3BackboneOutput(
            task_tokens=task_tokens,
            satellite_tokens=satellite_tokens,
            edge_features=edge_features,
            teacher_null_logits=null_logits,
            teacher_task_logits=task_logits,
            feasibility_logits=feasibility_logits,
        )
