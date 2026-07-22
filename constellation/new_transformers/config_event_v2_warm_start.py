"""V2-0 事件级联合 Actor-Critic 离线 warm start 配置。"""

stage = 'V2-0'
max_hours = 4
seed = 3407
annotation_file = 'train_paper_stage3_tau_e_existing.json'
stage3_checkpoint = (
    'work_dirs/paper_joint_stage3_200k/checkpoints/iter_200000/model.pth'
)
output_dir = 'work_dirs/event_joint_transformer_v2/v2_0_warm_start'

max_steps = 10_000
event_batch_size = 8
num_workers = 4
log_interval = 50
checkpoint_interval = 1_000
amp = True
amp_dtype = 'bfloat16'

model = {
    'event_width': 256,
    'freeze_backbone': True,
    'use_constraint_module': True,
    'use_sdpa': True,
}
optimizer = {
    'lr': 3e-4,
    'betas': (0.9, 0.98),
    'weight_decay': 1e-4,
}
loss_weights = {
    'task': 1.0,
    'termination': 1.0,
    'commitment': 1.0,
    'value': 1.0,
}
