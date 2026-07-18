"""P0-B 因果历史 Temporal Adapter 冻结主干训练配置。"""

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
    'temporal_horizons': (5, 15, 30, 300),
    'temporal_residual_scale': 0.25,
    'freeze_temporal_backbone': True,
    'feasibility_loss_weight': 0.0,
    'time_loss_weight': 0.0,
    'assignment_loss_weight': 1.0,
    'collision_loss_weight': 0.0,
    'coverage_loss_weight': 0.0,
    'temporal_visible_loss_weight': 1.0,
    'temporal_progress_loss_weight': 1.0,
    'temporal_completion_loss_weight': 1.0,
    'temporal_event_time_loss_weight': 1.0,
    # 2026-07-18 Stage3 annotation 全量审计得到的 neg/pos。
    # 顺序：next, 5s, 15s, 30s, 300s。
    'temporal_visible_positive_weights': (
        12.876360,
        9.997053,
        7.347674,
        5.074503,
        0.451956,
    ),
    'temporal_progress_positive_weights': (
        5.161846,
        4.165305,
        3.057156,
        2.112248,
        0.184455,
    ),
    'temporal_completion_positive_weights': (
        297.349788,
        58.149510,
        18.465460,
        8.562731,
        0.376449,
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
trainer['dataset']['temporal_horizons'] = (5, 15, 30, 300)
trainer['dataset']['constraint_batch_size'] = 16
trainer['optimizer']['lr'] = 5e-4

validator = _deepcopy(_stage3_validator)
validator['model'] = model
validator['dataset']['include_temporal_history'] = True
validator['dataset']['temporal_horizons'] = (5, 15, 30, 300)

del _deepcopy, _stage3_trainer, _stage3_validator
