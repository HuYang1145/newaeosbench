"""V2-2 四 GPU 同步 PPO 收益阶段的预注册配置。"""

stage = 'V2-2'
split = 'train'
seed = 4407
scene_shards = (
    tuple(range(4, 52)),
    tuple(range(52, 100)),
    tuple(range(100, 148)),
    tuple(range(148, 196)),
)
scene_ids = scene_shards[0]
heldout_train_scene_ids = tuple(range(196, 204))
max_time_step = 3600
max_hours = 16
safety_review_seconds = 5

bootstrap_checkpoint = (
    'work_dirs/event_joint_transformer_v2/v2_1_sync_ppo/'
    'checkpoint_update_000101.pth'
)
output_dir = 'work_dirs/event_joint_transformer_v2/v2_2_sync_ppo'

rollout_events_per_update = 64
max_updates = 1400
checkpoint_interval = 200
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
    'lr': 2e-6,
    'betas': (0.9, 0.98),
    'weight_decay': 1e-4,
    'eps': 1e-8,
}
