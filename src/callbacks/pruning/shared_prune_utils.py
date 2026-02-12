"""Shared validation/checkpoint suppression and sparsity utilities for pruners.

ValidationSuppressor: Composition-based class that suppresses validation,
ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau during sparsity ramp-up.

compute_sparsity: Unified sparsity computation for both magnitude and Bregman
pruning (handles raw Parameter lists and (Module, name) pairs).
"""

from typing import List, Tuple, Union

import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.types import LRSchedulerConfig

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

    # Detect calling convention from first element
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
    """Suppresses validation, ModelCheckpoint, EarlyStopping, and
    ReduceLROnPlateau during sparsity ramp-up. Restores and resets
    trackers when target is reached.

    Intended to be instantiated internally by each pruner (composition).
    """

    def __init__(self, tolerance: float = 1e-2):
        self.tolerance = tolerance
        self._suppressed = False
        self._original_limit_val_batches = None
        # Keyed by monitor name for cross-process stability
        self._original_save_top_k: dict[str, int] = {}
        self._original_es_check_on_train: dict[str, bool] = {}
        self._original_lr_strict: dict[str, bool] = {}

    def check(
        self,
        trainer: Trainer,
        current_sparsity: float,
        target_sparsity: float,
    ) -> None:
        """Check sparsity vs target and suppress/restore accordingly."""
        target_reached = (
            abs(current_sparsity - target_sparsity) <= self.tolerance
        )

        if not target_reached and not self._suppressed:
            # Transition: unsuppressed -> suppressed
            self._suppress(trainer, current_sparsity, target_sparsity)
        elif not target_reached and self._suppressed:
            # Still suppressed — reset EarlyStopping each epoch
            self._reset_early_stoppings(trainer)
            logger.info(
                f"Validation still suppressed (sparsity {current_sparsity:.2%}"
                f" outside target {target_sparsity:.2%}"
                f" ± {self.tolerance:.1%})"
            )
        elif target_reached and self._suppressed:
            # Transition: suppressed -> restored
            self._restore(trainer, current_sparsity, target_sparsity)
        # else: target_reached and not suppressed — no-op

    # -- State persistence ---------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "suppressed": self._suppressed,
            "original_limit_val_batches": self._original_limit_val_batches,
            "original_save_top_k": dict(self._original_save_top_k),
            "original_es_check_on_train": dict(
                self._original_es_check_on_train
            ),
            "original_lr_strict": dict(self._original_lr_strict),
        }

    def load_state_dict(self, state: dict) -> None:
        self._suppressed = state["suppressed"]
        self._original_limit_val_batches = state["original_limit_val_batches"]
        self._original_save_top_k = dict(state["original_save_top_k"])
        self._original_es_check_on_train = dict(
            state.get("original_es_check_on_train", {})
        )
        self._original_lr_strict = dict(
            state.get("original_lr_strict", {})
        )

    # -- Internal helpers ----------------------------------------------------

    def _suppress(
        self,
        trainer: Trainer,
        current_sparsity: float,
        target_sparsity: float,
    ) -> None:
        # Save originals
        self._original_limit_val_batches = trainer.limit_val_batches
        trainer.limit_val_batches = 0

        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                key = cb.monitor or ""
                self._original_save_top_k[key] = cb.save_top_k
                cb.save_top_k = 0
            if isinstance(cb, EarlyStopping):
                key = cb.monitor or ""
                self._original_es_check_on_train[key] = (
                    cb._check_on_train_epoch_end
                )
                # Disable: limit_val_batches=0 already prevents
                # on_validation_end, so setting this to False prevents
                # on_train_epoch_end check too — fully disabling ES.
                cb._check_on_train_epoch_end = False

        # Disable strict metric enforcement on ReduceLROnPlateau so
        # Lightning skips the step (instead of crashing) when the
        # monitored validation metric is absent.
        for config in trainer.lr_scheduler_configs:
            if config.reduce_on_plateau:
                key = config.monitor or ""
                self._original_lr_strict[key] = config.strict
                config.strict = False

        self._suppressed = True
        logger.info(
            f"Validation suppressed (sparsity {current_sparsity:.2%}"
            f" outside target {target_sparsity:.2%}"
            f" ± {self.tolerance:.1%})"
        )

    def _restore(
        self,
        trainer: Trainer,
        current_sparsity: float,
        target_sparsity: float,
    ) -> None:
        # Restore limit_val_batches
        if self._original_limit_val_batches is not None:
            trainer.limit_val_batches = self._original_limit_val_batches

        # Restore and reset ModelCheckpoint / EarlyStopping
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                key = cb.monitor or ""
                if key in self._original_save_top_k:
                    cb.save_top_k = self._original_save_top_k[key]
                _reset_model_checkpoint(cb)
            if isinstance(cb, EarlyStopping):
                key = cb.monitor or ""
                if key in self._original_es_check_on_train:
                    cb._check_on_train_epoch_end = (
                        self._original_es_check_on_train[key]
                    )
                _reset_early_stopping(cb)

        # Restore strict metric enforcement on ReduceLROnPlateau
        for config in trainer.lr_scheduler_configs:
            if config.reduce_on_plateau:
                key = config.monitor or ""
                if key in self._original_lr_strict:
                    config.strict = self._original_lr_strict[key]

        self._suppressed = False
        self._original_limit_val_batches = None
        self._original_save_top_k = {}
        self._original_es_check_on_train = {}
        self._original_lr_strict = {}
        logger.info(
            f"Validation restored (sparsity {current_sparsity:.2%}"
            f" reached target {target_sparsity:.2%})"
        )

    def _reset_early_stoppings(self, trainer: Trainer) -> None:
        for cb in trainer.callbacks:
            if isinstance(cb, EarlyStopping):
                _reset_early_stopping(cb)


# ---------------------------------------------------------------------------
# Reset helpers (moved from MagnitudePruner)
# ---------------------------------------------------------------------------


def _get_extreme_value(mode: str) -> float:
    return float("inf") if mode == "min" else float("-inf")


def _reset_early_stopping(cb: EarlyStopping) -> None:
    cb.wait_count = 0
    cb.stopped_epoch = 0
    extreme_val = _get_extreme_value(cb.mode)
    if isinstance(cb.best_score, torch.Tensor):
        cb.best_score = torch.tensor(extreme_val, device=cb.best_score.device)
    else:
        cb.best_score = torch.tensor(extreme_val)


def _reset_model_checkpoint(cb: ModelCheckpoint) -> None:
    cb.best_k_models = {}
    cb.kth_best_model_path = ""
    extreme_val = _get_extreme_value(cb.mode)
    if isinstance(cb.best_model_score, torch.Tensor):
        cb.kth_value = torch.tensor(
            extreme_val, device=cb.best_model_score.device
        )
    else:
        cb.kth_value = torch.tensor(extreme_val)
    cb.best_model_score = cb.kth_value
