"""Tests for ValidationSuppressor and compute_sparsity.

The suppressor is a stateless gate: ``gate(trainer, current, target)`` sets
``trainer.limit_val_batches`` to ``restore_limit`` if target is met,
otherwise to 0. ``prepare(trainer)`` runs once at fit start to make sibling
callbacks (EarlyStopping, ReduceLROnPlateau) tolerant of suppressed epochs.

The 9-case coverage table documented in the conversation is implemented as
``test_case_N_*`` methods in ``TestGateTransitions``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.callbacks.pruning.shared_prune_utils import (
    ValidationSuppressor,
    compute_sparsity,
)


# =============================================================================
# compute_sparsity tests
# =============================================================================


class TestComputeSparsity:
    def test_raw_params_format(self):
        p = nn.Parameter(torch.zeros(10))
        assert compute_sparsity([p]) == 1.0

    def test_module_name_pairs_format(self):
        m = nn.Linear(10, 5)
        m.weight.data.zero_()
        assert compute_sparsity([(m, "weight")]) == 1.0

    def test_threshold_based_detection(self):
        p = nn.Parameter(torch.tensor([0.0, 1e-7, 1e-5, 1.0]))
        sparsity = compute_sparsity([p], threshold=1e-6)
        assert abs(sparsity - 0.5) < 1e-6

    def test_custom_threshold(self):
        p = nn.Parameter(torch.tensor([0.0, 0.001, 0.01, 1.0]))
        sparsity = compute_sparsity([p], threshold=0.005)
        assert abs(sparsity - 0.5) < 1e-6

    def test_empty_list(self):
        assert compute_sparsity([]) == 0.0

    def test_all_zero(self):
        p = nn.Parameter(torch.zeros(100))
        assert compute_sparsity([p]) == 1.0

    def test_no_zero(self):
        p = nn.Parameter(torch.ones(100))
        assert compute_sparsity([p]) == 0.0

    def test_skips_non_grad_params(self):
        p = nn.Parameter(torch.zeros(10), requires_grad=False)
        assert compute_sparsity([p]) == 0.0

    def test_module_name_pairs_missing_attr(self):
        m = nn.Linear(10, 5)
        assert compute_sparsity([(m, "nonexistent")]) == 0.0

    def test_multiple_params(self):
        p1 = nn.Parameter(torch.zeros(50))
        p2 = nn.Parameter(torch.ones(50))
        sparsity = compute_sparsity([p1, p2])
        assert abs(sparsity - 0.5) < 1e-6


# =============================================================================
# Trainer fixture helpers
# =============================================================================


def _make_trainer(
    limit_val_batches: float = 1.0,
    num_sanity_val_steps: int = 2,
    callbacks=None,
    lr_configs=None,
) -> MagicMock:
    """Build a MagicMock trainer shaped enough to drive gate() / prepare()."""
    trainer = MagicMock()
    trainer.limit_val_batches = limit_val_batches
    trainer.num_sanity_val_steps = num_sanity_val_steps
    trainer.callbacks = callbacks or []
    trainer.lr_scheduler_configs = lr_configs or []
    return trainer


def _fresh_suppressor_with_prepared_trainer(
    tolerance: float = 1e-2, restore_limit: float = 1.0
):
    """Mirror the pruner's on_fit_start sequence: prepare + start suppressed."""
    trainer = _make_trainer()
    ValidationSuppressor.prepare(trainer)
    trainer.limit_val_batches = 0
    return ValidationSuppressor(
        tolerance=tolerance, restore_limit=restore_limit
    ), trainer


# =============================================================================
# prepare() — one-time fit-start setup
# =============================================================================


class TestPrepare:
    def test_zeros_sanity_check_steps(self):
        trainer = _make_trainer(num_sanity_val_steps=2)
        ValidationSuppressor.prepare(trainer)
        assert trainer.num_sanity_val_steps == 0

    def test_disables_early_stopping_train_epoch_end_check(self):
        es = EarlyStopping(monitor="val/loss", mode="min")
        es._check_on_train_epoch_end = True
        trainer = _make_trainer(callbacks=[es])
        ValidationSuppressor.prepare(trainer)
        assert es._check_on_train_epoch_end is False

    def test_relaxes_reduce_on_plateau_strict(self):
        cfg = SimpleNamespace(
            reduce_on_plateau=True, strict=True, monitor="val/loss"
        )
        trainer = _make_trainer(lr_configs=[cfg])
        ValidationSuppressor.prepare(trainer)
        assert cfg.strict is False

    def test_leaves_non_plateau_schedulers_alone(self):
        cfg = SimpleNamespace(
            reduce_on_plateau=False, strict=True, monitor=None
        )
        trainer = _make_trainer(lr_configs=[cfg])
        ValidationSuppressor.prepare(trainer)
        assert cfg.strict is True

    def test_forces_model_checkpoint_save_on_validation(self):
        mc = ModelCheckpoint(monitor="val/loss", save_top_k=3)
        mc.save_on_train_epoch_end = True
        trainer = _make_trainer(callbacks=[mc])
        ValidationSuppressor.prepare(trainer)
        # save_on_train_epoch_end -> False so suppressed epochs (no val run)
        # don't trigger a save against a stale/missing val/loss.
        assert mc.save_on_train_epoch_end is False
        # save_top_k must remain whatever the user configured.
        assert mc.save_top_k == 3


# =============================================================================
# gate() — the 9 documented transition cases
# =============================================================================


class TestGateTransitions:
    """Each ``test_case_N_*`` maps to a row in the coverage table."""

    # ---- Case 1: fresh start, epoch 0, target not met → stay suppressed ----
    def test_case_1_fresh_epoch_0_target_not_met_stays_suppressed(self):
        sup, trainer = _fresh_suppressor_with_prepared_trainer()
        sup.gate(trainer, current_sparsity=0.10, target_sparsity=0.90)
        assert trainer.limit_val_batches == 0

    # ---- Case 2: end of epoch 0, target met → restore ----
    def test_case_2_end_epoch_0_target_met_restores(self):
        sup, trainer = _fresh_suppressor_with_prepared_trainer(
            restore_limit=1.0
        )
        sup.gate(trainer, current_sparsity=0.90, target_sparsity=0.90)
        assert trainer.limit_val_batches == 1.0

    # ---- Case 3: end of epoch 0, target not met → stay suppressed ----
    def test_case_3_end_epoch_0_target_not_met_stays_suppressed(self):
        sup, trainer = _fresh_suppressor_with_prepared_trainer()
        sup.gate(trainer, current_sparsity=0.50, target_sparsity=0.90)
        assert trainer.limit_val_batches == 0

    # ---- Case 4: epoch ≥ 1, previously suppressed, target met → restore ----
    def test_case_4_previously_suppressed_target_met_restores(self):
        sup, trainer = _fresh_suppressor_with_prepared_trainer(
            restore_limit=0.5
        )
        sup.gate(trainer, 0.30, 0.90)
        assert trainer.limit_val_batches == 0
        sup.gate(trainer, 0.90, 0.90)
        assert trainer.limit_val_batches == 0.5

    # ---- Case 5: epoch ≥ 1, previously suppressed, target not met → stay ----
    def test_case_5_previously_suppressed_target_not_met_stays(self):
        sup, trainer = _fresh_suppressor_with_prepared_trainer()
        sup.gate(trainer, 0.30, 0.90)
        sup.gate(trainer, 0.40, 0.90)
        assert trainer.limit_val_batches == 0

    # ---- Case 6: epoch ≥ 1, previously restored, target still met → no-op --
    def test_case_6_previously_restored_target_still_met_no_op(self):
        sup, trainer = _fresh_suppressor_with_prepared_trainer(
            restore_limit=1.0
        )
        sup.gate(trainer, 0.90, 0.90)
        assert trainer.limit_val_batches == 1.0
        sup.gate(trainer, 0.895, 0.90)
        assert trainer.limit_val_batches == 1.0

    # ---- Case 7: epoch ≥ 1, previously restored, target drifts → re-suppr --
    def test_case_7_previously_restored_target_drifts_re_suppresses(self):
        sup, trainer = _fresh_suppressor_with_prepared_trainer(
            restore_limit=1.0
        )
        sup.gate(trainer, 0.90, 0.90)
        assert trainer.limit_val_batches == 1.0
        sup.gate(trainer, 0.80, 0.90)
        assert trainer.limit_val_batches == 0

    # ---- Case 8: resume from suppressed state → first gate call settles ----
    def test_case_8_resume_from_suppressed_state(self):
        # New session: fresh trainer, pruner's on_fit_start calls prepare()
        # and sets limit_val_batches=0. The suppressor instance is fresh too
        # — no state carries across sessions in the stateless design.
        sup, trainer = _fresh_suppressor_with_prepared_trainer()
        sup.gate(trainer, 0.50, 0.90)
        assert trainer.limit_val_batches == 0

    # ---- Case 9: resume from restored state → first gate call settles ----
    def test_case_9_resume_from_restored_state(self):
        sup, trainer = _fresh_suppressor_with_prepared_trainer()
        sup.gate(trainer, 0.90, 0.90)
        assert trainer.limit_val_batches == 1.0


# =============================================================================
# gate() — tolerance boundary and configuration
# =============================================================================


class TestGateBoundaries:
    def test_just_inside_tolerance_counts_as_met(self):
        sup, trainer = _fresh_suppressor_with_prepared_trainer(
            tolerance=0.01, restore_limit=1.0
        )
        # Safely inside tolerance; avoids FP rounding at the exact boundary
        # where 0.01 - 0.01 is sometimes reported as 1e-17 > 0.
        sup.gate(trainer, 0.895, 0.90)
        assert trainer.limit_val_batches == 1.0

    def test_exact_tolerance_boundary_counts_as_met(self):
        # Locks in the IEEE-754 boundary fix: abs(0.91 - 0.90) is
        # 0.010000000000000009, which used to fail the > tolerance check.
        sup, trainer = _fresh_suppressor_with_prepared_trainer(
            tolerance=0.01, restore_limit=1.0
        )
        sup.gate(trainer, 0.91, 0.90)
        assert trainer.limit_val_batches == 1.0
        sup.gate(trainer, 0.89, 0.90)
        assert trainer.limit_val_batches == 1.0

    def test_just_outside_tolerance_is_suppressed(self):
        sup, trainer = _fresh_suppressor_with_prepared_trainer(
            tolerance=0.01
        )
        sup.gate(trainer, 0.88, 0.90)
        assert trainer.limit_val_batches == 0

    def test_overshoot_within_tolerance_is_met(self):
        sup, trainer = _fresh_suppressor_with_prepared_trainer(
            tolerance=0.01, restore_limit=1.0
        )
        sup.gate(trainer, 0.905, 0.90)
        assert trainer.limit_val_batches == 1.0

    def test_custom_restore_limit_is_honored(self):
        sup, trainer = _fresh_suppressor_with_prepared_trainer(
            restore_limit=0.25
        )
        sup.gate(trainer, 0.90, 0.90)
        assert trainer.limit_val_batches == 0.25

    def test_tighter_tolerance_suppresses_previously_accepted_value(self):
        sup, trainer = _fresh_suppressor_with_prepared_trainer(
            tolerance=1e-4, restore_limit=1.0
        )
        sup.gate(trainer, 0.89, 0.90)
        assert trainer.limit_val_batches == 0


# =============================================================================
# Idempotence and oscillation handling
# =============================================================================


class TestGateIdempotence:
    def test_repeated_suppress_calls_leave_state_unchanged(self):
        sup, trainer = _fresh_suppressor_with_prepared_trainer()
        for _ in range(5):
            sup.gate(trainer, 0.30, 0.90)
            assert trainer.limit_val_batches == 0

    def test_repeated_restore_calls_leave_state_unchanged(self):
        sup, trainer = _fresh_suppressor_with_prepared_trainer(
            restore_limit=1.0
        )
        for _ in range(5):
            sup.gate(trainer, 0.90, 0.90)
            assert trainer.limit_val_batches == 1.0

    def test_oscillation_tracks_target_each_call(self):
        sup, trainer = _fresh_suppressor_with_prepared_trainer(
            restore_limit=1.0
        )
        pattern = [
            (0.900, 1.0),
            (0.800, 0),
            (0.905, 1.0),
            (0.700, 0),
            (0.895, 1.0),
        ]
        for current, expected in pattern:
            sup.gate(trainer, current, 0.90)
            assert trainer.limit_val_batches == expected
