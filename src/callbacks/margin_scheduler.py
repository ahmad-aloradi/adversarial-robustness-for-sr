"""Progressive margin scheduler callback for AAM-Softmax training."""

import math

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class ProgressiveMarginScheduler(Callback):
    """Callback to progressively increase the AAM margin during training.

    Implements wespeaker-style progressive margin scheduling:
    - Start with initial_margin (default 0.0) at start_epoch
    - Linearly increase to final_margin (default 0.2) over warmup_epochs
    - Hold at final_margin for the rest of training

    This allows easier learning in early epochs when the model hasn't learned
    good representations yet, then progressively increases discrimination.

    Args:
        initial_margin: Starting margin value (default: 0.0)
        final_margin: Target margin value (default: 0.2)
        warmup_epochs: Number of epochs to ramp margin (default: 2, scaled for ~15 epochs)
        start_epoch: Epoch to start increasing margin (default: 2, scaled for ~15 epochs)

    Note:
        Defaults are scaled for ~15 epoch training. For longer training (e.g., 150 epochs),
        use warmup_epochs=20 and start_epoch=20 to match wespeaker's original schedule.
    """

    def __init__(
        self,
        initial_margin: float = 0.0,
        final_margin: float = 0.2,
        warmup_epochs: int = 2,
        start_epoch: int = 2,
    ):
        super().__init__()
        self.initial_margin = initial_margin
        self.final_margin = final_margin
        self.warmup_epochs = warmup_epochs
        self.start_epoch = start_epoch

    def _get_margin(self, epoch: int) -> float:
        """Calculate margin for the given epoch."""
        if epoch < self.start_epoch:
            return self.initial_margin

        progress_epoch = epoch - self.start_epoch
        if progress_epoch >= self.warmup_epochs:
            return self.final_margin

        # Linear interpolation
        alpha = progress_epoch / self.warmup_epochs
        return self.initial_margin + alpha * (self.final_margin - self.initial_margin)

    def _update_margin(self, pl_module: pl.LightningModule, margin: float) -> None:
        """Update the margin in the AAM loss function."""
        # Access the loss function through train_criterion
        criterion = pl_module.train_criterion

        # Handle LogSoftmaxWrapper wrapping AdditiveAngularMargin
        if hasattr(criterion, "loss_fn"):
            loss_fn = criterion.loss_fn
        else:
            loss_fn = criterion

        # Update margin and derived values
        if hasattr(loss_fn, "margin"):
            loss_fn.margin = margin
            # Recompute derived values used by AdditiveAngularMargin
            if hasattr(loss_fn, "cos_m"):
                loss_fn.cos_m = math.cos(margin)
            if hasattr(loss_fn, "sin_m"):
                loss_fn.sin_m = math.sin(margin)
            if hasattr(loss_fn, "th"):
                loss_fn.th = math.cos(math.pi - margin)
            if hasattr(loss_fn, "mm"):
                loss_fn.mm = math.sin(math.pi - margin) * margin

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Update margin at the start of each epoch."""
        current_epoch = trainer.current_epoch
        new_margin = self._get_margin(current_epoch)
        self._update_margin(pl_module, new_margin)

        # Log the current margin
        pl_module.log("train/margin", new_margin, prog_bar=False)
