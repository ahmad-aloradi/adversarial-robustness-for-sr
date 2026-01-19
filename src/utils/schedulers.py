"""Custom learning rate schedulers for speaker verification training."""

from torch.optim.lr_scheduler import LRScheduler


class WarmupExponentialLR(LRScheduler):
    """Exponential LR scheduler with linear warmup.

    Implements the wespeaker-style learning rate schedule:
    1. Linear warmup from warmup_start_factor * lr to lr over warmup_epochs
    2. Exponential decay after warmup: lr = lr * gamma^(epoch - warmup_epochs)

    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of warmup epochs (default: 1, scaled for ~15 epoch training)
        warmup_start_factor: Starting LR factor during warmup (default: 0.01)
        gamma: Multiplicative factor for exponential decay (default: 0.80)
        last_epoch: The index of last epoch (default: -1)

    Note:
        Defaults are scaled for ~15 epoch training. For longer training (e.g., 150 epochs),
        use warmup_epochs=6 and gamma=0.97 to match wespeaker's original schedule.
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs: int = 1,
        warmup_start_factor: float = 0.01,
        gamma: float = 0.80,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.warmup_start_factor = warmup_start_factor
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            factor = self.warmup_start_factor + alpha * (1.0 - self.warmup_start_factor)
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            # Exponential decay after warmup
            exp_epoch = self.last_epoch - self.warmup_epochs
            return [base_lr * (self.gamma ** exp_epoch) for base_lr in self.base_lrs]
