"""Unit tests for the Lightning-native sparsity-gated callbacks.

Mirrors the band logic proven in the standalone POC: top-k checkpointing,
early stopping, and the validation forward pass open only when the injected
sparsity is within tolerance of the target. The pruners publish the sparsity
metric; here it is driven directly via a mocked callback_metrics dict.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.callbacks.pruning.sparsity_gating import (
    RampValidationGate,
    SparsityGatedEarlyStopping,
    SparsityGatedModelCheckpoint,
    sparsity_in_band,
)


def _trainer(sparsity=None, metric_key="sparsity"):
    trainer = MagicMock()
    trainer.callback_metrics = (
        {} if sparsity is None else {metric_key: torch.tensor(sparsity)}
    )
    return trainer


# --------------------------------------------------------------------------- #
# sparsity_in_band
# --------------------------------------------------------------------------- #
def test_in_band_within_tolerance():
    assert sparsity_in_band(_trainer(0.895), 0.90, 0.01)


def test_in_band_at_exact_boundary():
    # |0.91 - 0.90| == 0.01 must count as in band (+1e-9 absorbs rounding).
    assert sparsity_in_band(_trainer(0.91), 0.90, 0.01)


def test_out_of_band_beyond_tolerance():
    assert not sparsity_in_band(_trainer(0.88), 0.90, 0.01)


def test_missing_metric_raises():
    with pytest.raises(KeyError):
        sparsity_in_band(_trainer(None), 0.90, 0.01)


# --------------------------------------------------------------------------- #
# SparsityGatedModelCheckpoint
# --------------------------------------------------------------------------- #
def test_checkpoint_skips_topk_out_of_band():
    mc = SparsityGatedModelCheckpoint(target_sparsity=0.90, tolerance=0.01)
    with patch.object(ModelCheckpoint, "_save_topk_checkpoint") as parent:
        mc._save_topk_checkpoint(_trainer(0.50), {})
        parent.assert_not_called()


def test_checkpoint_saves_topk_in_band():
    mc = SparsityGatedModelCheckpoint(target_sparsity=0.90, tolerance=0.01)
    with patch.object(ModelCheckpoint, "_save_topk_checkpoint") as parent:
        mc._save_topk_checkpoint(_trainer(0.895), {})
        parent.assert_called_once()


def test_checkpoint_last_path_not_overridden():
    # last.ckpt must keep saving -> the subclass only gates the top-k path.
    assert (
        SparsityGatedModelCheckpoint._save_last_checkpoint
        is ModelCheckpoint._save_last_checkpoint
    )


def test_checkpoint_respects_custom_metric_key():
    mc = SparsityGatedModelCheckpoint(
        target_sparsity=0.90,
        tolerance=0.01,
        sparsity_metric="pruning/sparsity",
    )
    trainer = _trainer(0.895, metric_key="pruning/sparsity")
    with patch.object(ModelCheckpoint, "_save_topk_checkpoint") as parent:
        mc._save_topk_checkpoint(trainer, {})
        parent.assert_called_once()


# --------------------------------------------------------------------------- #
# SparsityGatedEarlyStopping
# --------------------------------------------------------------------------- #
def test_early_stopping_skips_check_out_of_band():
    es = SparsityGatedEarlyStopping(
        target_sparsity=0.90, tolerance=0.01, monitor="val_loss", mode="min"
    )
    with patch.object(EarlyStopping, "_run_early_stopping_check") as parent:
        es._run_early_stopping_check(_trainer(0.50))
        parent.assert_not_called()


def test_early_stopping_runs_check_in_band():
    es = SparsityGatedEarlyStopping(
        target_sparsity=0.90, tolerance=0.01, monitor="val_loss", mode="min"
    )
    with patch.object(EarlyStopping, "_run_early_stopping_check") as parent:
        es._run_early_stopping_check(_trainer(0.90))
        parent.assert_called_once()


# --------------------------------------------------------------------------- #
# RampValidationGate
# --------------------------------------------------------------------------- #
def _last_batch_trainer(sparsity, limit=1.0):
    trainer = _trainer(sparsity)
    trainer.is_last_batch = True
    trainer.limit_val_batches = limit
    return trainer


def test_gate_suppresses_out_of_band():
    gate = RampValidationGate(target_sparsity=0.90, tolerance=0.01)
    gate.restore_limit = 1.0
    trainer = _last_batch_trainer(0.50)
    gate.on_train_batch_end(trainer, MagicMock(), None, None, 0)
    assert trainer.limit_val_batches == 0


def test_gate_restores_in_band():
    gate = RampValidationGate(target_sparsity=0.90, tolerance=0.01)
    gate.restore_limit = 0.5
    trainer = _last_batch_trainer(0.895)
    gate.on_train_batch_end(trainer, MagicMock(), None, None, 0)
    assert trainer.limit_val_batches == 0.5


def test_gate_only_acts_on_last_batch():
    gate = RampValidationGate(target_sparsity=0.90, tolerance=0.01)
    trainer = _trainer(0.50)
    trainer.is_last_batch = False
    trainer.limit_val_batches = 1.0
    gate.on_train_batch_end(trainer, MagicMock(), None, None, 3)
    assert trainer.limit_val_batches == 1.0  # untouched


def test_gate_on_fit_start_captures_relaxes_and_suppresses():
    gate = RampValidationGate(target_sparsity=0.90, tolerance=0.01)
    plateau = MagicMock()
    plateau.reduce_on_plateau = True
    plateau.strict = True
    trainer = MagicMock()
    trainer.limit_val_batches = 0.5
    trainer.lr_scheduler_configs = [plateau]
    gate.on_fit_start(trainer, MagicMock())
    assert gate.restore_limit == 0.5  # captured the user's budget
    assert plateau.strict is False  # plateau relaxed for skipped epochs
    assert trainer.limit_val_batches == 0  # starts suppressed
