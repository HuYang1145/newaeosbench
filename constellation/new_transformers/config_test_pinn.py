strategy = dict(type='DDPStrategy')
model = dict(
    type='ConstellationModelRegistry.Model',
    use_compile=False,
)

iters = 1  # 只运行1次迭代
warmup_iters = 0
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
                        start_factor=1e-8,
                        total_iters=1,
                    ),
                    dict(
                        type='CosineAnnealingLR',
                        T_max=1,
                        eta_min=5e-6,
                    ),
                ],
                milestones=[1],
            ),
        ),
        dict(
            type='LogCallback',
            interval=1,
            collect_env=dict(),
            with_file_handler=True,
            eta=dict(type='EMA_ETA', ema=dict(decay=0.9)),
            priority=dict(init=-1),
        ),
        dict(type='GitCallback', diff='HEAD'),
        dict(
            type='TensorBoardCallback',
            interval=1,
            summary_writer=dict(),
            main_tag='train',
        ),
        dict(type='CheckpointCallback', interval=1e4),
    ],
    dataset=dict(
        type='ConstellationDatasetRegistry.Dataset',
        annotation_file='train.json',
        split='train',
        batch_size=2,  # 小批量测试
    ),
    dataloader=dict(
        type='PrefetchDataLoader',
        batch_size=None,
        num_workers=2,
        sampler=dict(type='DistributedSampler', shuffle=True),
    ),
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        betas=(0.9, 0.98),
        weight_decay=1e-4,
        eps=1e-8,
    ),
    iters=iters,
)
validator = dict(
    type='Validator',
    model=model,
    strategy=strategy,
    callbacks=[
        dict(
            type='MetricCallback',
            metrics=dict(
                loss=dict(
                    type='Mean',
                    value='loss',
                ),
            ),
        ),
    ],
    dataset=dict(
        type='ConstellationDatasetRegistry.Dataset',
        annotation_file='val_seen.json',
        split='val_seen',
        batch_size=2,
    ),
    dataloader=dict(
        type='PrefetchDataLoader',
        batch_size=None,
        num_workers=2,
        sampler=dict(type='DistributedSampler', shuffle=False),
    ),
)
