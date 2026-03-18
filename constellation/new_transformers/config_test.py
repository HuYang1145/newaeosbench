strategy = dict(type='DDPStrategy')
model = dict(
    type='ConstellationModelRegistry.Model',
    use_compile=True,  # 测试 compile
)

iters = 10  # 只跑 10 个迭代测试
warmup_iters = 2
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
                    dict(type='LinearLR', start_factor=1e-8, total_iters=warmup_iters - 1),
                    dict(type='CosineAnnealingLR', T_max=iters - warmup_iters, eta_min=5e-6),
                ],
                milestones=[warmup_iters],
            ),
        ),
        dict(type='LogCallback', interval=1, collect_env=dict(), with_file_handler=True, eta=dict(type='EMA_ETA', ema=dict(decay=0.9)), priority=dict(init=-1)),
    ],
    dataset=dict(
        type='ConstellationDatasetRegistry.Dataset',
        annotation_file='train.json',
        split='train',
        batch_size=2,  # 小 batch 快速测试
    ),
    dataloader=dict(
        type='PrefetchDataLoader',
        batch_size=None,
        num_workers=0,  # 测试时不用多进程
        sampler=dict(type='DistributedSampler', shuffle=True),
    ),
    optimizer=dict(type='AdamW', lr=1e-4, betas=(0.9, 0.98), weight_decay=1e-4, eps=1e-8),
    iters=iters,
)
