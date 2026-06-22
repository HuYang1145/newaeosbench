iters = 200000
model = {
    'type': 'ConstellationModelRegistry.JointModel',
    'use_constraint_module': True,
    'use_compile': False,
    'use_sdpa': True,
    'feasibility_loss_weight': 1.0,
    'time_loss_weight': 1.0,
    'assignment_loss_weight': 1.0
}
strategy = {'type': 'DDPStrategy'}
trainer = {
    'type':
    'IterBasedTrainer',
    'model': {
        'type': 'ConstellationModelRegistry.JointModel',
        'use_constraint_module': True,
        'use_compile': False,
        'use_sdpa': True,
        'feasibility_loss_weight': 1.0,
        'time_loss_weight': 1.0,
        'assignment_loss_weight': 1.0
    },
    'strategy': {
        'type': 'DDPStrategy'
    },
    'callbacks': [{
        'type': 'OptimizeCallback'
    }, {
        'type': 'LRScheduleCallback',
        'lr_scheduler': {
            'type':
            'SequentialLR',
            'schedulers': [{
                'type': 'LinearLR',
                'start_factor': 1e-08,
                'total_iters': 9999
            }, {
                'type': 'CosineAnnealingLR',
                'T_max': 190000,
                'eta_min': 5e-06
            }],
            'milestones': [10000]
        }
    }, {
        'type': 'LogCallback',
        'interval': 50,
        'collect_env': {},
        'with_file_handler': True,
        'eta': {
            'type': 'EMA_ETA',
            'ema': {
                'decay': 0.9
            }
        },
        'priority': {
            'init': -1
        }
    }, {
        'type': 'GitCallback',
        'diff': 'HEAD'
    }, {
        'type': 'TensorBoardCallback',
        'interval': 50,
        'summary_writer': {},
        'main_tag': 'train'
    }, {
        'type': 'CheckpointCallback',
        'interval': 10000.0
    }],
    'dataset': {
        'type': 'ConstellationDatasetRegistry.JointDataset',
        'annotation_file': 'train_paper_stage3_tau_e_existing.json',
        'split': 'train',
        'batch_size': 48,
        'constraint_batch_size': 48
    },
    'dataloader': {
        'type': 'PrefetchDataLoader',
        'batch_size': None,
        'num_workers': 4,
        'sampler': {
            'type': 'DistributedSampler',
            'shuffle': True
        }
    },
    'optimizer': {
        'type': 'AdamW',
        'lr': 0.0001,
        'betas': (0.9, 0.98),
        'weight_decay': 0.0001,
        'eps': 1e-08,
        'fused': True
    },
    'iters':
    200000
}
validator = {
    'type':
    'Validator',
    'model': {
        'type': 'ConstellationModelRegistry.JointModel',
        'use_constraint_module': True,
        'use_compile': False,
        'use_sdpa': True,
        'feasibility_loss_weight': 1.0,
        'time_loss_weight': 1.0,
        'assignment_loss_weight': 1.0
    },
    'strategy': {
        'type': 'DDPStrategy'
    },
    'callbacks': [{
        'type': 'MetricCallback',
        'metrics': {
            'loss': {
                'type': 'ReadyMadeMetric',
                'attr': '["loss"]'
            },
            'accuracy': {
                'type': 'AccuracyMetric',
                'top_k': 1,
                'logits': '["logits"]',
                'target': '["actions_task_id"]'
            }
        }
    }, {
        'type': 'LogCallback',
        'interval': 50,
        'collect_env': {},
        'with_file_handler': True,
        'eta': {
            'type': 'EMA_ETA',
            'ema': {
                'decay': 0.9
            }
        }
    }],
    'dataset': {
        'type': 'ConstellationDatasetRegistry.JointDataset',
        'split': 'val_seen',
        'batch_size': 128,
        'constraint_batch_size': 128
    },
    'dataloader': {
        'type': 'PrefetchDataLoader',
        'batch_size': None,
        'num_workers': 0,
        'sampler': {
            'type': 'DistributedSampler',
            'shuffle': False
        }
    }
}
warmup_iters = 10000
