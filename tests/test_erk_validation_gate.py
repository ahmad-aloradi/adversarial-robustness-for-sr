"""Validation gating in ERK mode keys on PRUNED sparsity, not overall.

ERK targets are defined over the prunable weight set; BN/bias stay dense, so
overall-model sparsity is diluted below the target. The gate must therefore use
the regularized-only sparsity, else validation would never reopen.
"""
from unittest.mock import MagicMock

from pytorch_lightning import LightningModule, Trainer

from src.callbacks.pruning.bregman.bregman_pruner import BregmanPruner


def _make_erk_pruner(final=0.9):
    pruner = BregmanPruner(
        lambda_scheduler=MagicMock(), target_sparsity=final, verbose=0
    )
    pruner._initialized = True
    pruner._erk_mode = True
    pruner.manager = MagicMock()

    trainer = MagicMock(spec=Trainer)
    trainer.limit_val_batches = 1.0
    trainer.current_epoch = 0
    trainer.is_last_batch = True
    pl_module = MagicMock(spec=LightningModule)

    measured = {"overall": 0.0, "pruned": 0.0}
    pruner._overall_sparsity = lambda: measured["overall"]
    pruner._pruned_sparsity = lambda: measured["pruned"]
    return pruner, trainer, pl_module, measured


def _fire_last_batch(pruner, trainer, pl_module):
    pruner._step_lambda_scheduler = lambda tr: None
    pruner._step_wp_annealer = lambda tr: None
    pruner._wp_anneal_active = lambda: False
    pruner._log_metrics = lambda pl, tr: None
    trainer.is_last_batch = True
    pruner.on_train_batch_end(trainer, pl_module, None, None, 0)


def test_gate_opens_on_pruned_target_despite_diluted_overall():
    pruner, trainer, pl_module, measured = _make_erk_pruner(final=0.9)
    # Prunable weights at target, overall diluted below it by dense BN/bias.
    measured["pruned"] = 0.9
    measured["overall"] = 0.62
    _fire_last_batch(pruner, trainer, pl_module)
    assert trainer.limit_val_batches == 1.0


def test_gate_suppresses_when_pruned_below_target():
    pruner, trainer, pl_module, measured = _make_erk_pruner(final=0.9)
    measured["pruned"] = 0.6
    measured["overall"] = 0.6
    _fire_last_batch(pruner, trainer, pl_module)
    assert trainer.limit_val_batches == 0


def test_gate_ignores_high_overall_when_pruned_low():
    # Defensive: a high overall reading must not open the gate in ERK mode.
    pruner, trainer, pl_module, measured = _make_erk_pruner(final=0.9)
    measured["pruned"] = 0.5
    measured["overall"] = 0.95
    _fire_last_batch(pruner, trainer, pl_module)
    assert trainer.limit_val_batches == 0
