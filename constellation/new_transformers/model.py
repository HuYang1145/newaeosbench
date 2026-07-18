from asyncio import tasks
from typing import Any, Iterable
import einops
import todd
import torch
from torch import nn

from todd.models.modules.transformer import Block
from todd.models.modules import sinusoidal_position_embedding
from todd.patches.torch import Sequential
from todd.models.losses import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from todd.registries import InitWeightsMixin
from constellation.data import SensorType
from .dataset import Batch, JointBatch
from .feasibility import (
    apply_feasibility_penalty,
    apply_feasibility_threshold,
)
from .registries import ConstellationModelRegistry
from .time_model import TimeModel
from todd.runners import Memo, BaseRunner
from todd.registries import InitWeightsMixin
from todd.runners.callbacks import TensorBoardCallback
from constellation import MAX_TIME_STEP
from torch.distributions import Categorical
from .assignment import AssignmentAuxiliaryLoss, BipartiteAssignmentHead
from .constants import SATELLITE_DIM, TASK_DIM, TIME_SCALE
from .temporal_adapter import (
    TemporalAdapter,
    TemporalAdapterOutput,
    TemporalHistoryTensors,
    TemporalOutcomePositiveWeights,
    temporal_outcome_loss,
)

GLOBALS = dict()


class Encoder(nn.Module):

    def __init__(
        self,
        *args,
        time_embedding_dim: int,
        sensor_type_embedding_dim: int,
        data_dim: int = TASK_DIM,
        data_embedding_dim: int,
        width: int,
        depth: int,
        num_heads: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._num_heads = num_heads

        self._data_embedding = nn.Linear(data_dim, data_embedding_dim)
        self._in_projector = nn.Linear(
            time_embedding_dim + sensor_type_embedding_dim
            + data_embedding_dim,
            width,
        )
        self._blocks = Sequential(
            *[Block(width=width, num_heads=num_heads) for _ in range(depth)],
        )
        self._norm = nn.LayerNorm(width)

    def forward(
        self,
        time_embedding: torch.Tensor,
        sensor_type_embedding: torch.Tensor,
        data: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        time_embedding = einops.repeat(
            time_embedding,
            'b d -> b nt d',
            nt=data.shape[1],
        )
        data_embedding = self._data_embedding(data)
        embedding = torch.cat((
            time_embedding,
            sensor_type_embedding,
            data_embedding,
        ), -1)
        x = self._in_projector(embedding)
        attention_mask = (
            einops.rearrange(attention_mask, 'b nt -> b nt 1')
            & einops.rearrange(attention_mask, 'b nt -> b 1 nt')
        )
        attention_mask = einops.repeat(
            attention_mask,
            'b nt nt_prime -> (b nh) nt nt_prime',
            nh=self._num_heads,
        )
        attention_mask = torch.where(attention_mask, 0, float('-inf'))
        x = self._blocks(x, attention_mask=attention_mask)
        x = self._norm(x)
        return x


class DecoderBlock(Block):

    def __init__(
        self,
        *args,
        width: int,
        num_heads: int,
        use_sdpa: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, width=width, num_heads=num_heads, **kwargs)
        self._use_sdpa = use_sdpa
        self._norm3 = nn.LayerNorm(width, 1e-6)
        self._cross_attention = nn.MultiheadAttention(
            width,
            num_heads,
            batch_first=True,
        )

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        hidden_states: torch.Tensor,
        cross_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = super().forward(x, attention_mask)

        norm = self._norm3(x)
        if self._use_sdpa:
            sdpa_backends = [
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                torch.nn.attention.SDPBackend.MATH,
            ]
            with torch.nn.attention.sdpa_kernel(sdpa_backends):
                cross_attention, _ = self._cross_attention(
                    norm,
                    hidden_states,
                    hidden_states,
                    need_weights=False,
                    attn_mask=cross_attention_mask,
                )
        else:
            cross_attention, _ = self._cross_attention(
                norm,
                hidden_states,
                hidden_states,
                need_weights=False,
                attn_mask=cross_attention_mask,
            )
        x = x + cross_attention

        return x


class Decoder(InitWeightsMixin, nn.Module):

    def __init__(
        self,
        *args,
        time_embedding_dim: int,
        sensor_type_embedding_dim: int,
        sensor_enabled_embedding_dim: int,
        data_dim: int = SATELLITE_DIM,
        data_embedding_dim: int,
        width: int,
        depth: int,
        num_heads: int,
        return_logits: bool,
        use_sdpa: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._num_heads = num_heads

        self._sensor_enabled_embedding = nn.Embedding(
            2,
            sensor_enabled_embedding_dim,
        )
        self._data_embedding = nn.Linear(data_dim, data_embedding_dim)

        self._in_projector = nn.Linear(
            time_embedding_dim + sensor_type_embedding_dim
            + sensor_enabled_embedding_dim + data_embedding_dim,
            width,
        )

        self._blocks = Sequential(
            *[
                DecoderBlock(
                    width=width,
                    num_heads=num_heads,
                    use_sdpa=use_sdpa,
                )
                for _ in range(depth)
            ],
        )
        self._norm = nn.LayerNorm(width)

        if return_logits:
            self._null_task = nn.Parameter(torch.empty(width))

    @property
    def return_logits(self) -> bool:
        return hasattr(self, '_null_task')

    def init_weights(self, config: todd.Config) -> bool:
        if self.return_logits:
            self._null_task.data.zero_()
        return super().init_weights(config)

    def forward(
        self,
        time_embedding: torch.Tensor,
        sensor_type_embedding: torch.Tensor,
        sensor_enabled: torch.Tensor,
        data: torch.Tensor,
        mask: torch.Tensor,
        hidden_states: torch.Tensor,
        tasks_mask: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_embedding = einops.repeat(
            time_embedding,
            'b d -> b ns d',
            ns=data.shape[1],
        )
        sensor_enabled_embedding = self._sensor_enabled_embedding(
            sensor_enabled,
        )
        data_embedding = self._data_embedding(data)
        embedding = torch.cat((
            time_embedding,
            sensor_type_embedding,
            sensor_enabled_embedding,
            data_embedding,
        ), -1)
        x = self._in_projector(embedding)

        mask = torch.where(mask, 0, float('-inf'))
        attention_mask = einops.repeat(
            mask,
            'b ns -> (b nh) ns ns_prime',
            nh=self._num_heads,
            ns_prime=embedding.shape[1],
        )
        cross_attention_mask = einops.repeat(
            tasks_mask,
            'b nt -> b ns nt',
            ns=data.shape[1],
        )
        cross_attention_mask = torch.where(
            cross_attention_mask,
            time_mask,
            # 0,
            float('-inf'),
        )
        cross_attention_mask = einops.repeat(
            cross_attention_mask,
            'b ns nt -> (b nh) ns nt',
            nh=self._num_heads,
        )

        x = self._blocks(
            x,
            attention_mask=attention_mask,
            hidden_states=hidden_states,
            cross_attention_mask=cross_attention_mask,
        )
        x = self._norm(x)

        if not self.return_logits:
            return x

        null_logits = torch.einsum('b s d, d -> b s', x, self._null_task)

        logits_mask = einops.rearrange(tasks_mask, 'b nt -> b 1 nt')
        logits = torch.einsum('b s d, b t d -> b s t', x, hidden_states)
        logits = logits + logits_mask

        return null_logits, logits, x
        # x = self._out_projector(x)
        # return x


class Transformer(nn.Module):

    def __init__(
        self,
        *args,
        time_embedding_dim: int = 64,
        sensor_type_embedding_dim: int,
        tasks_data_embedding_dim: int,
        encoder_width: int,
        encoder_depth: int,
        encoder_num_heads: int,
        sensor_enabled_embedding_dim: int,
        constellation_data_embedding_dim: int,
        decoder_width: int,
        decoder_depth: int,
        decoder_num_heads: int,
        return_logits: bool = True,
        use_constraint_module: bool = True,
        use_sdpa: bool = False,
        feasibility_threshold: float | None = None,
        feasibility_penalty_threshold: float | None = None,
        feasibility_penalty_strength: float | None = None,
        use_assignment_head: bool = False,
        assignment_head_hidden_width: int = 32,
        use_temporal_adapter: bool = False,
        temporal_adapter_hidden_width: int = 64,
        temporal_horizons: tuple[int, ...] = (5, 15, 30, 300),
        temporal_residual_scale: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        penalty_enabled = (
            feasibility_penalty_threshold is not None
            or feasibility_penalty_strength is not None
        )
        if feasibility_threshold is not None and penalty_enabled:
            raise ValueError(
                'hard threshold and soft penalty cannot be enabled '
                'simultaneously',
            )
        if feasibility_threshold is not None and not use_constraint_module:
            raise ValueError(
                'feasibility threshold requires the constraint module',
            )
        if (
            feasibility_threshold is not None
            and not 0 <= feasibility_threshold <= 1
        ):
            raise ValueError('feasibility threshold must be in [0, 1]')
        if penalty_enabled and not use_constraint_module:
            raise ValueError(
                'feasibility penalty requires the constraint module',
            )
        if (
            (feasibility_penalty_threshold is None)
            != (feasibility_penalty_strength is None)
        ):
            raise ValueError(
                'penalty threshold and strength must be set together',
            )
        if (
            feasibility_penalty_threshold is not None
            and not 0 < feasibility_penalty_threshold <= 1
        ):
            raise ValueError('penalty threshold must be in (0, 1]')
        if (
            feasibility_penalty_strength is not None
            and feasibility_penalty_strength < 0
        ):
            raise ValueError('penalty strength must be non-negative')
        self._return_logits = return_logits
        self._use_constraint_module = use_constraint_module
        self._feasibility_threshold = feasibility_threshold
        self._feasibility_penalty_threshold = feasibility_penalty_threshold
        self._feasibility_penalty_strength = feasibility_penalty_strength
        self._use_assignment_head = use_assignment_head
        if temporal_residual_scale < 0:
            raise ValueError('temporal_residual_scale must be non-negative')
        self._temporal_residual_scale = temporal_residual_scale
        if use_assignment_head and not return_logits:
            raise ValueError('assignment head requires task logits')

        time_embedding = sinusoidal_position_embedding(
            torch.arange(MAX_TIME_STEP),
            time_embedding_dim,
        )
        self._time_embedding = nn.Parameter(time_embedding)

        self._sensor_type_embedding = nn.Embedding(
            len(SensorType),
            sensor_type_embedding_dim,
        )
        self._encoder = Encoder(
            time_embedding_dim=time_embedding_dim,
            sensor_type_embedding_dim=sensor_type_embedding_dim,
            data_embedding_dim=tasks_data_embedding_dim,
            width=encoder_width,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
        )
        self._decoder = Decoder(
            time_embedding_dim=time_embedding_dim,
            sensor_type_embedding_dim=sensor_type_embedding_dim,
            sensor_enabled_embedding_dim=sensor_enabled_embedding_dim,
            data_embedding_dim=constellation_data_embedding_dim,
            width=decoder_width,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            return_logits=return_logits,
            use_sdpa=use_sdpa,
        )
        self._time_model = TimeModel()
        self._time_projection = nn.Linear(1, 1)
        self._assignment_head = (
            BipartiteAssignmentHead(
                satellite_width=decoder_width,
                task_width=encoder_width,
                hidden_width=assignment_head_hidden_width,
            )
            if use_assignment_head else None
        )
        self._temporal_adapter = (
            TemporalAdapter(
                satellite_width=decoder_width,
                task_width=encoder_width,
                hidden_width=temporal_adapter_hidden_width,
                horizons=temporal_horizons,
            )
            if use_temporal_adapter else None
        )

        self._time_model.requires_grad_(use_constraint_module)
        self._encoder.requires_grad_(True)
        self._decoder.requires_grad_(True)
        self._time_projection.requires_grad_(use_constraint_module)

        
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
        temporal_history: TemporalHistoryTensors | None = None,
        return_temporal_output: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[
            torch.Tensor,
            torch.Tensor,
            TemporalAdapterOutput | None,
        ]
    ):
        if isinstance(time_steps, torch.Tensor):
            time_steps = time_steps.flatten().tolist()
        else:
            time_steps = list(time_steps)

        if self._use_constraint_module:
            _, time_mask = self._time_model.predict(
                time_steps,
                constellation_data,
                constellation_mask,
                tasks_data,
                tasks_mask,
            )
            feasibility_logits = time_mask
            time_mask = time_mask.clamp_min(-100)

            time_mask = einops.rearrange(time_mask, 'b ns nt -> b ns nt 1')
            time_mask = self._time_projection(time_mask)
            time_mask = einops.rearrange(time_mask, 'b ns nt 1 -> b ns nt')
        else:
            feasibility_logits = None
            time_mask = constellation_data.new_zeros(
                constellation_data.shape[0],
                constellation_data.shape[1],
                tasks_data.shape[1],
            )

        time_embedding = self._time_embedding[time_steps]

        tasks_sensor_type_embedding = self._sensor_type_embedding(
            tasks_sensor_type,
        )
        hidden_states = self._encoder(
            time_embedding,
            tasks_sensor_type_embedding,
            tasks_data,
            tasks_mask,
        )

        constellation_sensor_type_embedding = self._sensor_type_embedding(
            constellation_sensor_type,
        )
        outputs = self._decoder(
            # return self._decoder(
            time_embedding,
            constellation_sensor_type_embedding,
            constellation_sensor_enabled,
            constellation_data,
            constellation_mask,
            hidden_states,
            tasks_mask,
            time_mask,
        )

        if not self._return_logits:
            x = outputs
            return x
        null_logits, logits, satellite_features = outputs

        logits = apply_feasibility_threshold(
            logits,
            feasibility_logits,
            self._feasibility_threshold,
        )
        logits = apply_feasibility_penalty(
            logits,
            feasibility_logits,
            threshold=self._feasibility_penalty_threshold,
            strength=self._feasibility_penalty_strength,
        )
        if self._assignment_head is not None:
            logits = self._assignment_head(
                einops.rearrange(null_logits, 'b ns -> b ns 1'),
                logits,
                satellite_features,
                hidden_states,
                constellation_mask,
                tasks_mask,
            )

        temporal_output = None
        if self._temporal_adapter is not None:
            if temporal_history is None:
                raise ValueError(
                    'temporal history is required when the adapter is enabled'
                )
            temporal_output = self._temporal_adapter(
                satellite_features=satellite_features,
                task_features=hidden_states,
                null_logits=null_logits,
                task_logits=logits,
                satellite_mask=constellation_mask,
                task_mask=tasks_mask,
                history=temporal_history,
            )
            null_logits = null_logits + self._temporal_residual_scale * (
                temporal_output.null_delta.tanh()
            )
            logits = logits + self._temporal_residual_scale * (
                temporal_output.task_delta.tanh()
            )

        if return_temporal_output:
            return null_logits, logits, temporal_output
        return null_logits, logits


class DiversityLoss(MSELoss):

    def forward(  # type: ignore[override]
        self,
        logits: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        p = logits.softmax(-1)
        p = p[..., 1:]
        counts = p.sum(-2)
        counts = counts[counts > 1.]
        return super().forward(counts, torch.ones_like(counts))


@ConstellationModelRegistry.register_()
class Model(nn.Module):

    def __init__(
        self,
        *args,
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
        use_compile: bool = False,
        use_sdpa: bool = False,
        feasibility_threshold: float | None = None,
        feasibility_penalty_threshold: float | None = None,
        feasibility_penalty_strength: float | None = None,
        use_assignment_head: bool = False,
        assignment_head_hidden_width: int = 32,
        freeze_assignment_backbone: bool = False,
        use_temporal_adapter: bool = False,
        temporal_adapter_hidden_width: int = 64,
        temporal_horizons: tuple[int, ...] = (5, 15, 30, 300),
        temporal_residual_scale: float = 1.0,
        freeze_temporal_backbone: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._transformer = Transformer(
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
            use_temporal_adapter=use_temporal_adapter,
            temporal_adapter_hidden_width=temporal_adapter_hidden_width,
            temporal_horizons=temporal_horizons,
            temporal_residual_scale=temporal_residual_scale,
        )
        if freeze_assignment_backbone and freeze_temporal_backbone:
            raise ValueError(
                'assignment and temporal backbone freeze modes are exclusive'
            )
        if freeze_assignment_backbone:
            if self._transformer._assignment_head is None:
                raise ValueError(
                    'freezing the backbone requires the assignment head',
                )
            self._transformer.requires_grad_(False)
            self._transformer._assignment_head.requires_grad_(True)
        if freeze_temporal_backbone:
            if self._transformer._temporal_adapter is None:
                raise ValueError(
                    'freezing the backbone requires the temporal adapter'
                )
            self._transformer.requires_grad_(False)
            self._transformer._temporal_adapter.requires_grad_(True)
        self._ce_loss = CrossEntropyLoss()

    @staticmethod
    def _history_from_batch(batch) -> TemporalHistoryTensors | None:
        temporal = batch.temporal
        if temporal is None:
            return None
        return TemporalHistoryTensors(
            previous_task_indices=temporal.previous_task_indices,
            previous_task_available=temporal.previous_task_available,
            previous_was_idle=temporal.previous_was_idle,
            run_lengths=temporal.run_lengths,
            switch_count_30=temporal.switch_count_30,
            switch_count_60=temporal.switch_count_60,
        )

    def _predict_with_temporal_output(
        self,
        *args,
        temporal_history: TemporalHistoryTensors | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, TemporalAdapterOutput | None]:
        null_logits, logits, temporal_output = self._transformer(
            *args,
            temporal_history=temporal_history,
            return_temporal_output=True,
            **kwargs,
        )
        null_logits = einops.rearrange(null_logits, 'b ns -> b ns 1')
        logits = torch.cat((null_logits, logits), -1)
        return logits, temporal_output

    def predict(
        self,
        *args,
        temporal_history: TemporalHistoryTensors | None = None,
        **kwargs,
    ) -> torch.Tensor:
        logits, _ = self._predict_with_temporal_output(
            *args,
            temporal_history=temporal_history,
            **kwargs,
        )
        return logits

    def forward(
        self,
        runner: BaseRunner[nn.Module],
        batch: Batch,
        memo: Memo,
    ) -> Memo:
        log: Memo | None = memo.get('log')
        tensorboard: TensorBoardCallback | None = memo.get('tensorboard')

        batch = Batch(*batch)  # for PrefetchDataLoader
        memo['actions_task_id'] = einops.rearrange(
            batch.actions_task_id + 1,
            'b ns -> (b ns)',
        )

        logits = self.predict(
            batch.time_steps,
            batch.constellation_sensor_type,
            batch.constellation_sensor_enabled,
            batch.constellation_data,
            batch.constellation_mask,
            batch.tasks_sensor_type,
            batch.tasks_data,
            batch.tasks_mask,
            temporal_history=self._history_from_batch(batch),
        )
        memo['logits'] = einops.rearrange(logits, 'b ns nt -> (b ns) nt')

        ce_loss = self._ce_loss(
            einops.rearrange(logits, 'b ns nt -> (b ns) nt'),
            einops.rearrange(batch.actions_task_id + 1, 'b ns -> (b ns)'),
        )
        loss = ce_loss

        memo['loss'] = loss

        tensors: dict[str, torch.Tensor] = dict(loss=loss, ce_loss=ce_loss)
        if log is not None:
            log.update({k: f'{v:.3f}' for k, v in tensors.items()})
        if tensorboard is not None:
            for k, v in tensors.items():
                tensorboard.summary_writer.add_scalar(
                    tensorboard.tag(k),
                    v.float(),
                    runner.iter_,
                )

        return memo


@ConstellationModelRegistry.register_()
class JointModel(Model):

    def __init__(
        self,
        *args,
        feasibility_loss_weight: float = 1.0,
        time_loss_weight: float = 1.0,
        assignment_loss_weight: float = 1.0,
        collision_loss_weight: float = 0.0,
        coverage_loss_weight: float = 0.0,
        train_duration_head_only: bool = False,
        temporal_visible_loss_weight: float = 0.0,
        temporal_progress_loss_weight: float = 0.0,
        temporal_completion_loss_weight: float = 0.0,
        temporal_event_time_loss_weight: float = 0.0,
        temporal_visible_positive_weights: tuple[float, ...] | None = None,
        temporal_progress_positive_weights: tuple[float, ...] | None = None,
        temporal_completion_positive_weights: tuple[float, ...] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if train_duration_head_only and any((
            assignment_loss_weight,
            collision_loss_weight,
            coverage_loss_weight,
        )):
            raise ValueError(
                'duration-head-only training requires all action loss weights '
                'to be zero',
            )
        if train_duration_head_only and feasibility_loss_weight:
            raise ValueError(
                'duration-head-only training requires feasibility loss '
                'weight to be zero',
            )
        self._bce_loss = BCEWithLogitsLoss()
        self._mse_loss = MSELoss()
        self._feasibility_loss_weight = feasibility_loss_weight
        self._time_loss_weight = time_loss_weight
        self._assignment_loss_weight = assignment_loss_weight
        self._collision_loss_weight = collision_loss_weight
        self._coverage_loss_weight = coverage_loss_weight
        self._train_duration_head_only = train_duration_head_only
        self._temporal_visible_loss_weight = temporal_visible_loss_weight
        self._temporal_progress_loss_weight = temporal_progress_loss_weight
        self._temporal_completion_loss_weight = (
            temporal_completion_loss_weight
        )
        self._temporal_event_time_loss_weight = (
            temporal_event_time_loss_weight
        )
        temporal_adapter = self._transformer._temporal_adapter
        expected_positive_weights = (
            None
            if temporal_adapter is None
            else 1 + len(temporal_adapter.horizons)
        )
        positive_weight_groups = (
            ('temporal_visible_positive_weights',
             temporal_visible_positive_weights),
            ('temporal_progress_positive_weights',
             temporal_progress_positive_weights),
            ('temporal_completion_positive_weights',
             temporal_completion_positive_weights),
        )
        provided_positive_weights = [
            values is not None for _, values in positive_weight_groups
        ]
        if any(provided_positive_weights) and not all(
            provided_positive_weights
        ):
            raise ValueError(
                'temporal positive weights must provide visible, progress '
                'and completion together'
            )
        for name, values in positive_weight_groups:
            if values is not None and (
                expected_positive_weights is None
                or len(values) != expected_positive_weights
                or any(
                    not torch.isfinite(torch.tensor(value))
                    or value <= 0
                    for value in values
                )
            ):
                raise ValueError(
                    f'{name} must contain one positive value for next and '
                    'each temporal horizon'
                )
            self.register_buffer(
                f'_{name}',
                (
                    None
                    if values is None
                    else torch.tensor(values, dtype=torch.float)
                ),
                persistent=False,
            )
        self._assignment_auxiliary_loss = AssignmentAuxiliaryLoss()
        if train_duration_head_only:
            self.requires_grad_(False)
            self._transformer._time_model._duration_head.requires_grad_(True)
            self.register_load_state_dict_post_hook(
                self._initialize_duration_head_from_legacy_checkpoint,
            )

    def _initialize_duration_head_from_legacy_checkpoint(
        self,
        module: nn.Module,
        incompatible_keys: Any,
    ) -> None:
        del module
        prefix = '_transformer._time_model._duration_head.'
        required_missing = {f'{prefix}weight', f'{prefix}bias'}
        if not required_missing.issubset(incompatible_keys.missing_keys):
            return

        time_model = self._transformer._time_model
        legacy_head = time_model._mlp[-1]
        with torch.no_grad():
            # 旧 checkpoint 的 duration 数值尺度错误。先用残差精确抵消旧输出，
            # 再仅训练独立 head 学习修正后的归一化持续时间。
            time_model._duration_head.weight.copy_(
                -legacy_head.weight[0:1],
            )
            time_model._duration_head.bias.copy_(
                -legacy_head.bias[0:1],
            )

    def forward(
        self,
        runner: BaseRunner[nn.Module],
        batch: JointBatch,
        memo: Memo,
    ) -> Memo:
        log: Memo | None = memo.get('log')
        tensorboard: TensorBoardCallback | None = memo.get('tensorboard')

        batch = JointBatch(*batch)  # for PrefetchDataLoader
        if not self._train_duration_head_only:
            memo['actions_task_id'] = einops.rearrange(
                batch.actions_task_id + 1,
                'b ns -> (b ns)',
            )

            logits, temporal_output = self._predict_with_temporal_output(
                batch.time_steps,
                batch.constellation_sensor_type,
                batch.constellation_sensor_enabled,
                batch.constellation_data,
                batch.constellation_mask,
                batch.tasks_sensor_type,
                batch.tasks_data,
                batch.tasks_mask,
                temporal_history=self._history_from_batch(batch),
            )
            flat_logits = einops.rearrange(logits, 'b ns nt -> (b ns) nt')
            memo['logits'] = flat_logits

            la_loss = self._ce_loss(
                flat_logits,
                einops.rearrange(
                    batch.actions_task_id + 1,
                    'b ns -> (b ns)',
                ),
            )
            assignment_auxiliary = self._assignment_auxiliary_loss(
                logits,
                batch.actions_task_id,
                batch.constellation_mask,
                batch.tasks_mask,
            )
            if temporal_output is None:
                temporal_visible_loss = logits.new_zeros(())
                temporal_progress_loss = logits.new_zeros(())
                temporal_completion_loss = logits.new_zeros(())
                temporal_event_time_loss = logits.new_zeros(())
            else:
                if batch.temporal is None:
                    raise ValueError(
                        'temporal targets are required for adapter training'
                    )
                temporal_losses = temporal_outcome_loss(
                    temporal_output,
                    batch.temporal,
                    batch.actions_task_id,
                    positive_weights=(
                        None
                        if self._temporal_visible_positive_weights is None
                        else TemporalOutcomePositiveWeights(
                            visible=(
                                self._temporal_visible_positive_weights
                            ),
                            progress=(
                                self._temporal_progress_positive_weights
                            ),
                            completion=(
                                self._temporal_completion_positive_weights
                            ),
                        )
                    ),
                )
                temporal_visible_loss = temporal_losses.visible
                temporal_progress_loss = temporal_losses.progress
                temporal_completion_loss = temporal_losses.completion
                temporal_event_time_loss = temporal_losses.event_time

        pred_durations, pred_masks = self._transformer._time_model._predict(
            batch.constraint_time_steps,
            batch.constraint_constellation_data,
            batch.constraint_tasks_data,
        )
        gt_masks = batch.constraint_durations >= 0
        if gt_masks.any():
            lt_loss = self._mse_loss(
                pred_durations[gt_masks],
                batch.constraint_durations[gt_masks].float(),
            )
            duration_mae_s = (
                pred_durations[gt_masks]
                - batch.constraint_durations[gt_masks].float()
            ).abs().mean() * TIME_SCALE
        else:
            lt_loss = pred_durations.new_zeros(())
            duration_mae_s = pred_durations.new_zeros(())

        ls_loss = self._bce_loss(
            pred_masks,
            gt_masks.float(),
            mask=torch.where(
                gt_masks,
                gt_masks.sum() / (~gt_masks).sum().clamp_min(1),
                1.,
            ),
        )
        if self._train_duration_head_only:
            la_loss = pred_durations.new_zeros(())
            assignment_collision_loss = pred_durations.new_zeros(())
            assignment_coverage_loss = pred_durations.new_zeros(())
            temporal_visible_loss = pred_durations.new_zeros(())
            temporal_progress_loss = pred_durations.new_zeros(())
            temporal_completion_loss = pred_durations.new_zeros(())
            temporal_event_time_loss = pred_durations.new_zeros(())
        else:
            assignment_collision_loss = assignment_auxiliary.collision
            assignment_coverage_loss = assignment_auxiliary.coverage
        loss = (
            self._feasibility_loss_weight * ls_loss
            + self._time_loss_weight * lt_loss
            + self._assignment_loss_weight * la_loss
            + self._collision_loss_weight * assignment_collision_loss
            + self._coverage_loss_weight * assignment_coverage_loss
            + self._temporal_visible_loss_weight * temporal_visible_loss
            + self._temporal_progress_loss_weight * temporal_progress_loss
            + self._temporal_completion_loss_weight
            * temporal_completion_loss
            + self._temporal_event_time_loss_weight
            * temporal_event_time_loss
        )
        memo.update(
            loss=loss,
            ls_loss=ls_loss,
            lt_loss=lt_loss,
            duration_mae_s=duration_mae_s,
            pred_masks=pred_masks,
            gt_masks=gt_masks,
            assignment_loss=la_loss,
            assignment_collision_loss=assignment_collision_loss,
            assignment_coverage_loss=assignment_coverage_loss,
            temporal_visible_loss=temporal_visible_loss,
            temporal_progress_loss=temporal_progress_loss,
            temporal_completion_loss=temporal_completion_loss,
            temporal_event_time_loss=temporal_event_time_loss,
        )

        tensors: dict[str, torch.Tensor] = dict(
            loss=loss,
            ls_loss=ls_loss,
            lt_loss=lt_loss,
            duration_mae_s=duration_mae_s,
            la_loss=la_loss,
            ce_loss=la_loss,
            assignment_collision_loss=assignment_collision_loss,
            assignment_coverage_loss=assignment_coverage_loss,
            temporal_visible_loss=temporal_visible_loss,
            temporal_progress_loss=temporal_progress_loss,
            temporal_completion_loss=temporal_completion_loss,
            temporal_event_time_loss=temporal_event_time_loss,
        )
        if log is not None:
            log.update({k: f'{v:.3f}' for k, v in tensors.items()})
        if tensorboard is not None:
            for k, v in tensors.items():
                tensorboard.summary_writer.add_scalar(
                    tensorboard.tag(k),
                    v.float(),
                    runner.iter_,
                )

        return memo
