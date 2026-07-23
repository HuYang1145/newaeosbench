"""V2-1 同步 PPO 正确性阶段的预注册配置。"""

stage = 'V2-1'
split = 'train'
seed = 3407
scene_ids = (0, 1, 2, 3)
max_time_step = 3600
max_hours = 4
safety_review_seconds = 5

warm_start_checkpoint = (
    'work_dirs/event_joint_transformer_v2/v2_0_warm_start/'
    'checkpoint_step_010000.pth'
)
output_dir = 'work_dirs/event_joint_transformer_v2/v2_1_sync_ppo'

rollout_events_per_update = 64
max_updates = 64
checkpoint_interval = 4
log_interval = 1

gamma = 1.0
lambda_base = 0.95
reference_seconds = 5.0
clip_ratio = 0.2
value_coefficient = 0.5
entropy_coefficient = 0.01
max_grad_norm = 1.0
max_kl = 0.03
ppo_epochs = 4
minibatch_events = 16
logprob_replay_atol = 1e-6

freeze_backbone = True
amp = True
amp_dtype = 'bfloat16'

model = {
    'event_width': 256,
    'freeze_backbone': True,
    'use_constraint_module': True,
    'use_sdpa': True,
}
optimizer = {
    'lr': 2e-5,
    'betas': (0.9, 0.98),
    'weight_decay': 1e-4,
    'eps': 1e-8,
}
