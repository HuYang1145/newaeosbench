"""M2 事件终止、持续时间与短窗口结果头的冻结主干训练配置。"""

from copy import deepcopy as _deepcopy

from constellation.new_transformers.config_paper_stage3_200k import (
    trainer as _stage3_trainer,
    validator as _stage3_validator,
)


iters = 10_000
model = {
    'type': 'ConstellationModelRegistry.JointModel',
    'use_constraint_module': True,
    'use_compile': False,
    'use_sdpa': True,
    'use_temporal_adapter': True,
    'temporal_adapter_hidden_width': 64,
    'temporal_horizons': (5, 15, 30, 60),
    'temporal_residual_scale': 0.0,
    'freeze_temporal_backbone': True,
    'feasibility_loss_weight': 0.0,
    'time_loss_weight': 0.0,
    'assignment_loss_weight': 0.0,
    'collision_loss_weight': 0.0,
    'coverage_loss_weight': 0.0,
    'temporal_visible_loss_weight': 1.0,
    'temporal_progress_loss_weight': 1.0,
    'temporal_completion_loss_weight': 1.0,
    'temporal_event_time_loss_weight': 1.0,
    'temporal_continue_loss_weight': 1.0,
    'temporal_duration_loss_weight': 1.0,
    # 2026-07-20 对 Stage3 annotation 前 256 场、15,402,489 条非空边
    # 的审计结果。continue 使用 stop/continue；duration 权重为归一化逆频率。
    'temporal_continue_positive_weight': 0.005588996,
    'temporal_duration_class_weights': (
        2.659873,
        1.090517,
        0.758177,
        0.407495,
        0.083938,
    ),
    # 顺序为 next、5s、15s、30s、60s 的 neg/pos。
    'temporal_visible_positive_weights': (
        12.598763,
        9.810389,
        7.202741,
        4.965879,
        2.764055,
    ),
    'temporal_progress_positive_weights': (
        5.173770,
        4.180250,
        3.068755,
        2.119655,
        1.188129,
    ),
    'temporal_completion_positive_weights': (
        294.956978,
        57.670361,
        18.304235,
        8.480025,
        3.602941,
    ),
}

trainer = _deepcopy(_stage3_trainer)
trainer['model'] = model
trainer['iters'] = iters
trainer['callbacks'][1]['lr_scheduler'] = {
    'type': 'SequentialLR',
    'schedulers': [{
        'type': 'LinearLR',
        'start_factor': 1e-3,
        'total_iters': 499,
    }, {
        'type': 'CosineAnnealingLR',
        'T_max': 9_500,
        'eta_min': 1e-5,
    }],
    'milestones': [500],
}
trainer['callbacks'][5]['interval'] = 1_000.0
trainer['dataset']['include_temporal_history'] = True
trainer['dataset']['temporal_horizons'] = (5, 15, 30, 60)
# local-10 的四张 4090 当前均被共享 VLLM 各占约 21.5 GiB。
# M2 第一轮只做资源受限 pilot，使用已通过真实 batch smoke 的小批量。
trainer['dataset']['batch_size'] = 8
trainer['dataset']['constraint_batch_size'] = 8
trainer['optimizer']['lr'] = 5e-4

validator = _deepcopy(_stage3_validator)
validator['model'] = model
validator['dataset']['include_temporal_history'] = True
validator['dataset']['temporal_horizons'] = (5, 15, 30, 60)

del _deepcopy, _stage3_trainer, _stage3_validator
