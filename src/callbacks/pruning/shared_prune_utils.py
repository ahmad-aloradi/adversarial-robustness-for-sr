"""Shared validation-gating and sparsity utilities for pruners.

ValidationSuppressor: stateless gate that toggles `trainer.limit_val_batches`
based on whether current sparsity matches the target.

compute_sparsity: Unified sparsity computation for both magnitude and Bregman
pruning (handles raw Parameter lists and (Module, name) pairs).
"""

from typing import List, Tuple, Union

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch.nn as nn

from src.utils import get_pylogger

logger = get_pylogger(__name__)


# ---------------------------------------------------------------------------
# Shared sparsity computation
# ---------------------------------------------------------------------------


def compute_sparsity(
    params: Union[
        List[nn.Parameter],
        List[Tuple[nn.Module, str]],
    ],
    threshold: float = 1e-12,
) -> float:
    """Compute fraction of near-zero elements across parameters.

    Handles two calling conventions:
    - List[Parameter]: raw parameter tensors (Bregman style)
    - List[Tuple[Module, str]]: (module, param_name) pairs (magnitude style)
    """
    if not params:
        return 0.0

    total = 0
    zeros = 0

    first = params[0]
    is_tuple = isinstance(first, (tuple, list))

    if is_tuple:
        for module, name in params:
            param = getattr(module, name, None)
            if param is not None:
                total += param.numel()
                zeros += (param.abs() <= threshold).sum().item()
    else:
        for p in params:
            if not p.requires_grad:
                continue
            total += p.numel()
            zeros += (p.abs() <= threshold).sum().item()

    return zeros / max(1, total)


# ---------------------------------------------------------------------------
# Validation suppression
# ---------------------------------------------------------------------------


class ValidationSuppressor:
    """Stateless validation gate driven by sparsity.

    Usage (from a pruner callback):

      # Once, in on_fit_start — prevents sanity check and makes the
      # val-monitoring callbacks tolerant of skipped validations:
      ValidationSuppressor.prepare(trainer)

      # Each time you want to gate validation (e.g. on_validation_epoch_start
      # or on_train_epoch_start, whichever fires first for your pruner):
      suppressor.gate(trainer, current_sparsity, target_sparsity)

    ``gate`` sets ``trainer.limit_val_batches`` to 0 if sparsity is outside
    tolerance of target, otherwise to ``restore_limit``. No save/restore, no
    transitions, no state — the truth lives on the trainer. Safe to call on
    every hook without special-casing epoch 0, resume, or oscillation.
    """

    def __init__(self, tolerance: float = 1e-2, restore_limit: float = 1.0):
        self.tolerance = tolerance
        self.restore_limit = restore_limit
        self._was_suppressed = True  # only used for log throttling

    def gate(
        self,
        trainer: Trainer,
        current_sparsity: float,
        target_sparsity: float,
    ) -> None:
        # +1e-9 absorbs IEEE-754 rounding at the exact boundary, e.g.
        # abs(0.91 - 0.90) == 0.010000000000000009 > 0.01 without the slack.
        suppress = (
            abs(current_sparsity - target_sparsity) > self.tolerance + 1e-9
        )
        trainer.limit_val_batches = (
            0 if suppress else self.restore_limit
        )
        if suppress != self._was_suppressed:
            self._was_suppressed = suppress
            if suppress:
                logger.info(
                    f"Validation suppressed (sparsity {current_sparsity:.2%}"
                    f" outside target {target_sparsity:.2%}"
                    f" ± {self.tolerance:.1%})"
                )
            else:
                logger.info(
                    f"Validation restored (sparsity {current_sparsity:.2%}"
                    f" reached target {target_sparsity:.2%})"
                )

    @staticmethod
    def prepare(trainer: Trainer) -> None:
        """One-time setup in ``on_fit_start``.

        Configures sibling callbacks to tolerate skipped validations:
        - ``num_sanity_val_steps = 0``: skip the pre-training sanity check.
        - ``EarlyStopping._check_on_train_epoch_end = False``: prevent ES
          from firing on training metrics while val is suppressed.
        - ``ModelCheckpoint.save_on_train_epoch_end = False``: force saves to
          occur at end of validation, so suppressed epochs simply skip saving
          rather than checkpointing on a missing monitor.
        - ``ReduceLROnPlateau.strict = False``: tolerate the missing val
          monitor metric during suppressed epochs.
        """
        trainer.num_sanity_val_steps = 0
        for cb in trainer.callbacks:
            if isinstance(cb, EarlyStopping):
                cb._check_on_train_epoch_end = False
            elif isinstance(cb, ModelCheckpoint):
                cb.save_on_train_epoch_end = False
        for c in trainer.lr_scheduler_configs:
            if c.reduce_on_plateau:
                c.strict = False
