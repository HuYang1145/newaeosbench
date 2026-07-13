"""P0 二部图分配头第一阶段训练配置。"""

from copy import deepcopy

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
    'use_assignment_head': True,
    'assignment_head_hidden_width': 32,
    'freeze_assignment_backbone': True,
    'feasibility_loss_weight': 0.0,
    'time_loss_weight': 0.0,
    'assignment_loss_weight': 1.0,
    'collision_loss_weight': 0.2,
    'coverage_loss_weight': 0.1,
}

trainer = deepcopy(_stage3_trainer)
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
trainer['dataset']['constraint_batch_size'] = 16
trainer['optimizer']['lr'] = 5e-4

validator = deepcopy(_stage3_validator)
validator['model'] = model
