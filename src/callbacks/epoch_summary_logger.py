from pathlib import Path

from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_only

from src.callbacks.pruning.shared_prune_utils import compute_sparsity


class EpochSummaryLogger(Callback):
    """Logs average train/validation loss, monitor metric, and model sparsity
    to a file each epoch.

    Minimal and self-contained: accumulates batch losses (weighted by batch size)
    for train and validation, computes average losses at epoch end, computes
    sparsity (fraction of parameters near-zero) and appends a CSV line to
    `train_log.txt` in the experiment directory (Trainer.default_root_dir).
    """

    def __init__(
        self,
        monitor: str,
        filename: str = "train_log.txt",
        sparsity_threshold: float = 1e-12,
    ):
        self.filename = filename
        self.monitor = monitor

        # train accumulators
        self.train_loss_sum = 0.0
        self.train_samples = 0

        # val accumulators
        self.val_loss_sum = 0.0
        self.val_samples = 0

    # --- helpers ---
    def _get_batch_size(self, batch) -> int:
        # Try common container shapes used in modules
        try:
            if hasattr(batch, "audio"):
                return int(batch.audio.shape[0])
            if isinstance(batch, (list, tuple)) and len(batch) > 0:
                first = batch[0]
                if hasattr(first, "shape"):
                    return int(first.shape[0])
                if isinstance(first, (list, tuple)) and hasattr(
                    first[0], "shape"
                ):
                    return int(first[0].shape[0])
            if isinstance(batch, dict):
                first = next(iter(batch.values()))
                if hasattr(first, "shape"):
                    return int(first.shape[0])
        except Exception:
            pass
        return 1

    # --- training hooks ---
    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.train_loss_sum = 0.0
        self.train_samples = 0

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if outputs is None:
            return
        loss = (
            outputs.get("loss")
            if isinstance(outputs, dict)
            else getattr(outputs, "loss", None)
        )
        if loss is None:
            return
        try:
            batch_loss = float(loss.detach().cpu().item())
        except Exception:
            return
        batch_size = self._get_batch_size(batch)
        self.train_loss_sum += batch_loss * batch_size
        self.train_samples += batch_size

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.val_loss_sum = 0.0
        self.val_samples = 0

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if outputs is None:
            return
        loss = (
            outputs.get("loss")
            if isinstance(outputs, dict)
            else getattr(outputs, "loss", None)
        )
        if loss is None:
            return
        try:
            batch_loss = float(loss.detach().cpu().item())
        except Exception:
            return
        batch_size = self._get_batch_size(batch)
        self.val_loss_sum += batch_loss * batch_size
        self.val_samples += batch_size

    @rank_zero_only
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        # Compute averages
        train_avg = (
            self.train_loss_sum / self.train_samples
            if self.train_samples > 0
            else float("nan")
        )
        # Prefer the val accumulators (validation usually runs at epoch end)
        val_avg = (
            self.val_loss_sum / self.val_samples
            if self.val_samples > 0
            else float("nan")
        )

        # Compute sparsity using shared utility
        sparsity = compute_sparsity(
            list(pl_module.parameters()), threshold=1e-12
        )

        cb_metrics = dict(trainer.callback_metrics)
        if self.monitor not in cb_metrics:
            raise KeyError(
                f"EpochSummaryLogger: monitored metric '{self.monitor}' not found in trainer.callback_metrics. Available: {list(cb_metrics.keys())}"
            )

        monitor_val = (
            float(cb_metrics[self.monitor].item())
            if hasattr(cb_metrics[self.monitor], "item")
            else float(cb_metrics[self.monitor])
        )

        # Write to file
        out_dir = Path(trainer.default_root_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / self.filename

        with open(path, "a") as f:
            f.write(
                f"epoch: {trainer.current_epoch}, "
                f"train_loss: {train_avg:.4f}, "
                f"valid_loss: {val_avg:.4f}, "
                f"{self.monitor}: {monitor_val:.4f}, "
                f"sparsity: {sparsity:.4f}\n"
            )
