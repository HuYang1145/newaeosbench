"""修复持续时间尺度后的 duration-head-only 小规模微调配置。"""

iters = 2_000
warmup_iters = 100

strategy = dict(type='DDPStrategy')
model = dict(
    type='ConstellationModelRegistry.JointModel',
    use_constraint_module=True,
    use_compile=False,
    use_sdpa=True,
    feasibility_loss_weight=0.,
    time_loss_weight=1.,
    assignment_loss_weight=0.,
    collision_loss_weight=0.,
    coverage_loss_weight=0.,
    train_duration_head_only=True,
)

trainer = dict(
    type='IterBasedTrainer',
    model=model,
    strategy=strategy,
    callbacks=[
        dict(type='OptimizeCallback'),
        dict(
            type='LRScheduleCallback',
            lr_scheduler=dict(
                type='SequentialLR',
                schedulers=[
                    dict(
                        type='LinearLR',
                        start_factor=1e-4,
                        total_iters=warmup_iters - 1,
                    ),
                    dict(
                        type='CosineAnnealingLR',
                        T_max=iters - warmup_iters,
                        eta_min=1e-6,
                    ),
                ],
                milestones=[warmup_iters],
            ),
        ),
        dict(
            type='LogCallback',
            interval=20,
            collect_env=dict(),
            with_file_handler=True,
            eta=dict(type='EMA_ETA', ema=dict(decay=0.9)),
            priority=dict(init=-1),
        ),
        dict(type='GitCallback', diff='HEAD'),
        dict(
            type='TensorBoardCallback',
            interval=20,
            summary_writer=dict(),
            main_tag='train',
        ),
        dict(type='CheckpointCallback', interval=500),
    ],
    dataset=dict(
        type='ConstellationDatasetRegistry.JointDataset',
        annotation_file='train_paper_stage3_tau_e_existing.json',
        split='train',
        batch_size=8,
        constraint_batch_size=256,
    ),
    dataloader=dict(
        type='PrefetchDataLoader',
        batch_size=None,
        num_workers=4,
        sampler=dict(type='DistributedSampler', shuffle=True),
    ),
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        betas=(0.9, 0.98),
        weight_decay=1e-4,
        eps=1e-8,
        fused=True,
    ),
    iters=iters,
)
