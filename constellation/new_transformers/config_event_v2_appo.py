"""V2-3 异步 APPO 扩展阶段的预注册配置。"""

stage = 'V2-3'
split = 'train'
seed = 5407
scene_ids = tuple(range(205, 325))
max_time_step = 3600
max_hours = 28
safety_review_seconds = 5

bootstrap_checkpoint = (
    'work_dirs/event_joint_transformer_v2/v2_2_sync_ppo/replica_0/'
    'checkpoint_update_001046.pth'
)
output_dir = 'work_dirs/event_joint_transformer_v2/v2_3_appo'

actor_chunk_events = 32
learner_batch_events = 128
max_policy_lag = 2
max_updates = 5000
checkpoint_interval = 100
log_interval = 1

gamma = 1.0
lambda_base = 0.95
reference_seconds = 5.0
clip_ratio = 0.2
value_coefficient = 0.5
entropy_coefficient = 0.01
max_grad_norm = 1.0
max_kl = 0.03
ppo_epochs = 2
minibatch_events = 32
logprob_replay_atol = 1e-6

encoder_unfreeze_layers = 1
decoder_unfreeze_layers = 1
backbone_lr_scale = 0.1
amp = True
amp_dtype = 'bfloat16'

model = {
    'event_width': 256,
    'freeze_backbone': True,
    'use_constraint_module': True,
    'use_sdpa': True,
}
optimizer = {
    'lr': 1e-6,
    'betas': (0.9, 0.98),
    'weight_decay': 1e-4,
    'eps': 1e-8,
}
