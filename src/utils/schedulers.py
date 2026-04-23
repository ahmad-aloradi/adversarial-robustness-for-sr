"""Custom learning rate schedulers for speaker verification training."""

from typing import Optional

from torch.optim.lr_scheduler import LRScheduler


class WarmupExponentialLR(LRScheduler):
    """Exponential LR scheduler with linear warmup.

    Implements the wespeaker-style learning rate schedule:
    1. Linear warmup from warmup_start_factor * lr to lr over warmup_epochs
    2. Exponential decay after warmup: lr = lr * gamma^(epoch - warmup_epochs)

    When ``steps_per_epoch`` is provided, the scheduler updates per step
    (progress is measured in fractional epochs). When ``steps_per_epoch`` 
    is None, the scheduler is epoch-granular (call ``.step()`` once per epoch).

    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of warmup epochs
        warmup_start_factor: Starting LR factor during warmup
        gamma: Per-epoch multiplicative factor for exponential decay
        steps_per_epoch: If set, scheduler operates per step with epoch-level semantics
        last_epoch: The index of last epoch (default: -1)
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs: int = 1,
        warmup_start_factor: float = 0.01,
        gamma: float = 0.8,
        steps_per_epoch: Optional[int] = None,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.warmup_start_factor = warmup_start_factor
        self.gamma = gamma
        self.steps_per_epoch = steps_per_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.steps_per_epoch is not None and self.steps_per_epoch > 0:
            t = self.last_epoch / self.steps_per_epoch
        else:
            t = float(self.last_epoch)

        if t < self.warmup_epochs:
            alpha = t / max(1, self.warmup_epochs)
            factor = self.warmup_start_factor + alpha * (1.0 - self.warmup_start_factor)
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            exp_epochs = t - self.warmup_epochs
            return [base_lr * (self.gamma ** exp_epochs) for base_lr in self.base_lrs]
