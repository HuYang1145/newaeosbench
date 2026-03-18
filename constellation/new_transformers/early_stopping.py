import torch
from todd.runners.callbacks import BaseCallback
from todd.runners.registries import CallbackRegistry


@CallbackRegistry.register_()
class EarlyStoppingCallback(BaseCallback):
    """早停回调：验证集Loss连续N轮不下降则停止训练"""

    def __init__(self, patience=5, delta=0.0, **kwargs):
        super().__init__(**kwargs)
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def after_val_epoch(self, runner, memo):
        """验证后检查是否早停"""
        val_loss = memo.get('loss')
        if val_loss is None:
            return

        if isinstance(val_loss, str):
            val_loss = float(val_loss)

        # 检查是否刷新最佳Loss
        if val_loss < self.best_loss - self.delta:
            print(f"🎉 验证Loss刷新: {self.best_loss:.4f} -> {val_loss:.4f}")
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"⚠️ 验证Loss未改善 ({self.counter}/{self.patience})")

            if self.counter >= self.patience:
                self.early_stop = True
                print(f"🛑 触发早停！最佳Loss: {self.best_loss:.4f}")
                runner.should_stop = True
