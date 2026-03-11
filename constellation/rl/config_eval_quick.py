# 快速评估配置 - val_unseen 数据集
environment = dict(world_size=4, split='val_unseen')  # 使用4个GPU

algorithm = dict(verbose=1)

log_interval = 50
