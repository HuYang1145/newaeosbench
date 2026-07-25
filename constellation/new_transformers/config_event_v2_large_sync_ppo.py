"""V2-2 大规模严格同步 PPO 的预注册单 seed 配置。"""

stage = 'V2-2-Large'
split = 'train'
seed = 5408
scene_ids = tuple(range(205, 325))
max_time_step = 3600
safety_review_seconds = 5

bootstrap_checkpoint = (
    'work_dirs/event_joint_transformer_v2/v2_2_sync_ppo/replica_0/'
    'checkpoint_update_001046.pth'
)
output_dir = (
    'work_dirs/event_joint_transformer_v2/v2_2_large_sync_ppo/seed_5408'
)

actor_count = 12
active_environments = 60
actor_devices = ('cuda:0', 'cuda:1')
events_per_actor_round = 8
min_update_events = 64
max_updates = 3000
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
