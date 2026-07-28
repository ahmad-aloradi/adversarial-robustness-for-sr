"""Lightning-native sparsity gating for pruning runs.

A sparsity ramp must not select on off-target epochs: an epoch whose sparsity
falls outside ``tolerance`` as a fraction of the target should not win a
``save_top_k`` slot, trip early stopping, or even spend the validation pass.

Three self-contained pieces, each gating on the sparsity the pruner injects
into ``trainer.callback_metrics`` at the epoch's last training batch:

- ``SparsityGatedModelCheckpoint``: skips top-k saving out of band; ``last.ckpt``
  is a separate path and keeps saving (resume stays intact).
- ``SparsityGatedEarlyStopping``: skips the early-stopping check out of band, so
  patience never accrues on off-target metrics.
- ``RampValidationGate``: zeroes ``limit_val_batches`` out of band to skip the
  validation forward pass, and relaxes plateau LR schedulers so a skipped epoch
  does not crash on the absent monitor.

The pruner publishes the gate metric each epoch (Bregman: overall ``sparsity``;
magnitude: ``pruning/sparsity``); these callbacks read it and gate.
"""

from typing import Optional

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.utils import get_pylogger

logger = get_pylogger(__name__)


def sparsity_in_band(
    trainer: Trainer,
    target_sparsity: float,
    tolerance: float,
    metric_key: str = "sparsity",
) -> bool:
    """True when the sparsity is within a ``tolerance`` relative difference.

    In band iff ``(1 - tolerance) * target <= sparsity <= (1 + tolerance) *
    target``, so the band is the same fraction of the target at any target.

    The pruner must have injected ``metric_key`` into ``callback_metrics``; a
    missing key is a wiring bug (fail loud rather than silently pass/skip).
    """
    if metric_key not in trainer.callback_metrics:
        raise KeyError(
            f"sparsity gate metric '{metric_key}' not in callback_metrics; "
            "the pruner must inject it each epoch."
        )
    current = float(trainer.callback_metrics[metric_key])
    low = (1 - tolerance) * target_sparsity - 1e-6
    high = (1 + tolerance) * target_sparsity + 1e-6
    return low <= current <= high


class SparsityGatedModelCheckpoint(ModelCheckpoint):
    """``ModelCheckpoint`` that saves top-k only when sparsity is in band.

    Out of band, ``_save_topk_checkpoint`` is a no-op, so no off-target epoch
    enters ``best_k_models`` (and downstream checkpoint averaging). The separate
    ``_save_last_checkpoint`` path is untouched.
    """

    def __init__(
        self,
        target_sparsity: float,
        tolerance: float = 0.01,
        sparsity_metric: str = "sparsity",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_sparsity = target_sparsity
        self.tolerance = tolerance
        self.sparsity_metric = sparsity_metric

    def _save_topk_checkpoint(self, trainer, monitor_candidates):
        if not sparsity_in_band(
            trainer,
            self.target_sparsity,
            self.tolerance,
            self.sparsity_metric,
        ):
            return
        super()._save_topk_checkpoint(trainer, monitor_candidates)


class SparsityGatedEarlyStopping(EarlyStopping):
    """``EarlyStopping`` that checks only when sparsity is in band.

    Out of band the check returns early, so patience never advances on an
    off-target metric; in band it resumes with fresh patience (nothing stale to
    reset). Set ``check_on_train_epoch_end=False`` so the check runs at
    validation end on the fresh metric.
    """

    def __init__(
        self,
        target_sparsity: float,
        tolerance: float = 0.01,
        sparsity_metric: str = "sparsity",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_sparsity = target_sparsity
        self.tolerance = tolerance
        self.sparsity_metric = sparsity_metric

    def _run_early_stopping_check(self, trainer):
        if not sparsity_in_band(
            trainer,
            self.target_sparsity,
            self.tolerance,
            self.sparsity_metric,
        ):
            return
        super()._run_early_stopping_check(trainer)


class RampValidationGate(Callback):
    """Skip the validation forward pass while sparsity is out of band.

    Lightning has no per-epoch "skip validation" hook, so the only lever is
    ``trainer.limit_val_batches``. It is read just after the last training
    batch, so the decision is made there from the freshly-injected sparsity
    (the pruner must run before this callback). Plateau LR schedulers are set
    non-strict in ``on_fit_start`` so a skipped epoch (absent val monitor) does
    not raise; they simply do not step, freezing LR across the ramp.
    """

    def __init__(
        self,
        target_sparsity: float,
        tolerance: float = 0.01,
        sparsity_metric: str = "sparsity",
    ):
        super().__init__()
        self.target_sparsity = target_sparsity
        self.tolerance = tolerance
        self.sparsity_metric = sparsity_metric
        # Validation budget to restore in band (the user's limit_val_batches).
        self.restore_limit: float = 1.0
        # Throttles the log to suppress/restore transitions.
        self._suppressed: Optional[bool] = None

    def on_fit_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        value = trainer.limit_val_batches
        if isinstance(value, (int, float)) and value > 0:
            self.restore_limit = value
        for c in trainer.lr_scheduler_configs:
            if c.reduce_on_plateau:
                c.strict = False
        # Start suppressed; on_train_batch_end reopens once in band.
        trainer.limit_val_batches = 0

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if not trainer.is_last_batch:
            return
        in_band = sparsity_in_band(
            trainer,
            self.target_sparsity,
            self.tolerance,
            self.sparsity_metric,
        )
        trainer.limit_val_batches = self.restore_limit if in_band else 0
        suppress = not in_band
        if suppress != self._suppressed:  # log transitions only
            self._suppressed = suppress
            logger.info(
                "Validation %s (sparsity gate, band %.2f%%-%.2f%%)",
                "suppressed" if suppress else "restored",
                (1 - self.tolerance) * self.target_sparsity * 100,
                (1 + self.tolerance) * self.target_sparsity * 100,
            )


if __name__ == "__main__":
    from types import SimpleNamespace

    fake = SimpleNamespace(callback_metrics={"sparsity": torch.tensor(0.897)})
    print(
        "in band (0.897 vs 0.90 ± 0.5%):", sparsity_in_band(fake, 0.90, 0.005)
    )
    fake.callback_metrics["sparsity"] = torch.tensor(0.80)
    print(
        "in band (0.80 vs 0.90 ± 0.5%): ", sparsity_in_band(fake, 0.90, 0.005)
    )
