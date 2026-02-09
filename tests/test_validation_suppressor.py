"""Tests for ValidationSuppressor and compute_sparsity from suppression.py."""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
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
        """Works with List[Parameter] input (Bregman style)."""
        p = nn.Parameter(torch.zeros(10))
        assert compute_sparsity([p]) == 1.0

    def test_module_name_pairs_format(self):
        """Works with List[Tuple[Module, str]] input (magnitude style)."""
        m = nn.Linear(10, 5)
        m.weight.data.zero_()
        assert compute_sparsity([(m, "weight")]) == 1.0

    def test_threshold_based_detection(self):
        """Values at or below threshold count as zero."""
        p = nn.Parameter(torch.tensor([0.0, 1e-7, 1e-5, 1.0]))
        # threshold=1e-6: 0.0 and 1e-7 are <= 1e-6
        sparsity = compute_sparsity([p], threshold=1e-6)
        assert abs(sparsity - 0.5) < 1e-6

    def test_custom_threshold(self):
        """Custom threshold changes detection."""
        p = nn.Parameter(torch.tensor([0.0, 0.001, 0.01, 1.0]))
        # threshold=0.005 → 0.0, 0.001 are below
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
        """Raw param format skips params with requires_grad=False."""
        p = nn.Parameter(torch.zeros(10), requires_grad=False)
        assert compute_sparsity([p]) == 0.0  # Skipped → 0 total

    def test_module_name_pairs_missing_attr(self):
        """Gracefully handles missing attribute in (module, name) format."""
        m = nn.Linear(10, 5)
        # 'nonexistent' doesn't exist → skipped
        assert compute_sparsity([(m, "nonexistent")]) == 0.0

    def test_multiple_params(self):
        """Aggregates across multiple parameters."""
        p1 = nn.Parameter(torch.zeros(50))  # 50 zeros
        p2 = nn.Parameter(torch.ones(50))  # 0 zeros
        sparsity = compute_sparsity([p1, p2])
        assert abs(sparsity - 0.5) < 1e-6


# =============================================================================
# ValidationSuppressor tests
# =============================================================================


def _make_trainer(**kwargs):
    """Create a mock trainer with callbacks."""
    trainer = MagicMock(spec=Trainer)
    trainer.limit_val_batches = kwargs.get("limit_val_batches", 1.0)
    trainer.callbacks = kwargs.get("callbacks", [])
    return trainer


class TestValidationSuppressor:
    def test_suppresses_when_below_target(self):
        """Suppresses validation when sparsity < target."""
        sup = ValidationSuppressor(tolerance=1e-3)
        trainer = _make_trainer()

        sup.check(trainer, current_sparsity=0.5, target_sparsity=0.9)

        assert trainer.limit_val_batches == 0
        assert sup._suppressed

    def test_suppresses_model_checkpoint(self):
        """Sets save_top_k=0 on ModelCheckpoint during suppression."""
        sup = ValidationSuppressor()
        ckpt = ModelCheckpoint(
            monitor="val_loss", save_top_k=3, mode="min", dirpath="/tmp"
        )
        trainer = _make_trainer(callbacks=[ckpt])

        sup.check(trainer, current_sparsity=0.1, target_sparsity=0.9)

        assert ckpt.save_top_k == 0

    def test_restores_when_target_reached(self):
        """Restores original values when target is reached."""
        sup = ValidationSuppressor()
        ckpt = ModelCheckpoint(
            monitor="val_loss", save_top_k=3, mode="min", dirpath="/tmp"
        )
        trainer = _make_trainer(limit_val_batches=0.5, callbacks=[ckpt])

        # Suppress
        sup.check(trainer, current_sparsity=0.1, target_sparsity=0.9)
        assert trainer.limit_val_batches == 0
        assert ckpt.save_top_k == 0

        # Restore
        sup.check(trainer, current_sparsity=0.9, target_sparsity=0.9)
        assert trainer.limit_val_batches == 0.5
        assert ckpt.save_top_k == 3
        assert not sup._suppressed

    def test_resets_early_stopping_on_restore(self):
        """Resets EarlyStopping state when validation is restored."""
        sup = ValidationSuppressor()
        es = EarlyStopping(monitor="val_loss", patience=3, mode="min")
        es.wait_count = 2
        es.best_score = torch.tensor(0.5)
        trainer = _make_trainer(callbacks=[es])

        # Suppress then restore
        sup.check(trainer, current_sparsity=0.1, target_sparsity=0.9)
        sup.check(trainer, current_sparsity=0.9, target_sparsity=0.9)

        assert es.wait_count == 0
        assert es.best_score == torch.tensor(float("inf"))

    def test_resets_early_stopping_max_mode(self):
        """Resets EarlyStopping best_score to -inf in max mode."""
        sup = ValidationSuppressor()
        es = EarlyStopping(monitor="val_acc", patience=3, mode="max")
        es.best_score = torch.tensor(0.9)
        trainer = _make_trainer(callbacks=[es])

        sup.check(trainer, current_sparsity=0.1, target_sparsity=0.9)
        sup.check(trainer, current_sparsity=0.9, target_sparsity=0.9)

        assert es.best_score == torch.tensor(float("-inf"))

    def test_resets_model_checkpoint_on_restore(self):
        """Resets ModelCheckpoint trackers on restore."""
        sup = ValidationSuppressor()
        ckpt = ModelCheckpoint(
            monitor="val_loss", save_top_k=3, mode="min", dirpath="/tmp"
        )
        # Simulate some recorded state
        ckpt.best_k_models = {"/tmp/a.ckpt": torch.tensor(0.5)}
        ckpt.kth_best_model_path = "/tmp/a.ckpt"
        trainer = _make_trainer(callbacks=[ckpt])

        sup.check(trainer, current_sparsity=0.1, target_sparsity=0.9)
        sup.check(trainer, current_sparsity=0.9, target_sparsity=0.9)

        assert ckpt.best_k_models == {}
        assert ckpt.kth_best_model_path == ""

    def test_noop_when_always_at_target(self):
        """No suppression when sparsity is within tolerance of target."""
        sup = ValidationSuppressor()
        trainer = _make_trainer()

        for _ in range(5):
            sup.check(trainer, current_sparsity=0.9005, target_sparsity=0.9)

        assert trainer.limit_val_batches == 1.0
        assert not sup._suppressed

    def test_maintains_suppression_across_epochs(self):
        """Stays suppressed across multiple calls below target."""
        sup = ValidationSuppressor()
        ckpt = ModelCheckpoint(
            monitor="val_loss", save_top_k=3, mode="min", dirpath="/tmp"
        )
        trainer = _make_trainer(callbacks=[ckpt])

        for sparsity in [0.1, 0.3, 0.5, 0.7]:
            sup.check(trainer, current_sparsity=sparsity, target_sparsity=0.9)
            assert trainer.limit_val_batches == 0
            assert ckpt.save_top_k == 0
            assert sup._suppressed

    def test_resets_early_stopping_each_suppressed_epoch(self):
        """EarlyStopping is reset every epoch while suppressed."""
        sup = ValidationSuppressor()
        es = EarlyStopping(monitor="val_loss", patience=3, mode="min")
        trainer = _make_trainer(callbacks=[es])

        # First call: suppress
        sup.check(trainer, current_sparsity=0.1, target_sparsity=0.9)

        # Simulate EarlyStopping accumulating state
        es.wait_count = 2
        es.best_score = torch.tensor(0.3)

        # Second call: still suppressed, should reset ES
        sup.check(trainer, current_sparsity=0.3, target_sparsity=0.9)
        assert es.wait_count == 0
        assert es.best_score == torch.tensor(float("inf"))

    def test_tolerance_works_correctly(self):
        """Sparsity within tolerance band of target counts as reached."""
        # Below target, within band: |0.8995 - 0.9| ≈ 0.0005 ≤ 0.001 → reached
        sup = ValidationSuppressor(tolerance=1e-3)
        trainer = _make_trainer()
        sup.check(trainer, current_sparsity=0.8995, target_sparsity=0.9)
        assert trainer.limit_val_batches == 1.0  # Not suppressed

        # Below target, outside band: |0.898 - 0.9| = 0.002 > 0.001 → not reached
        sup2 = ValidationSuppressor(tolerance=1e-3)
        trainer2 = _make_trainer()
        sup2.check(trainer2, current_sparsity=0.898, target_sparsity=0.9)
        assert trainer2.limit_val_batches == 0  # Suppressed

        # Above target, within band: |0.9005 - 0.9| ≈ 0.0005 ≤ 0.001 → reached
        sup3 = ValidationSuppressor(tolerance=1e-3)
        trainer3 = _make_trainer()
        sup3.check(trainer3, current_sparsity=0.9005, target_sparsity=0.9)
        assert trainer3.limit_val_batches == 1.0  # Not suppressed

        # Above target, outside band: |0.902 - 0.9| = 0.002 > 0.001 → not reached
        sup4 = ValidationSuppressor(tolerance=1e-3)
        trainer4 = _make_trainer()
        sup4.check(trainer4, current_sparsity=0.902, target_sparsity=0.9)
        assert trainer4.limit_val_batches == 0  # Suppressed

    def test_state_dict_roundtrip(self):
        """state_dict / load_state_dict preserves state."""
        sup = ValidationSuppressor()
        trainer = _make_trainer(
            callbacks=[
                ModelCheckpoint(
                    monitor="val_loss",
                    save_top_k=3,
                    mode="min",
                    dirpath="/tmp",
                )
            ]
        )

        # Suppress to populate state
        sup.check(trainer, current_sparsity=0.1, target_sparsity=0.9)

        state = sup.state_dict()
        assert state["suppressed"] is True
        assert state["original_limit_val_batches"] == 1.0
        assert "val_loss" in state["original_save_top_k"]

        # Load into new instance
        sup2 = ValidationSuppressor()
        sup2.load_state_dict(state)

        assert sup2._suppressed is True
        assert sup2._original_limit_val_batches == 1.0
        assert sup2._original_save_top_k == {"val_loss": 3}

    def test_strict_load_fails_on_missing_keys(self):
        """load_state_dict raises KeyError on missing keys."""
        sup = ValidationSuppressor()
        with pytest.raises(KeyError):
            sup.load_state_dict({"suppressed": True})

    def test_multiple_model_checkpoints(self):
        """Handles multiple ModelCheckpoint callbacks correctly."""
        sup = ValidationSuppressor()
        ckpt1 = ModelCheckpoint(
            monitor="val_loss", save_top_k=3, mode="min", dirpath="/tmp"
        )
        ckpt2 = ModelCheckpoint(
            monitor="val_eer", save_top_k=1, mode="min", dirpath="/tmp"
        )
        trainer = _make_trainer(callbacks=[ckpt1, ckpt2])

        # Suppress
        sup.check(trainer, current_sparsity=0.1, target_sparsity=0.9)
        assert ckpt1.save_top_k == 0
        assert ckpt2.save_top_k == 0

        # Restore
        sup.check(trainer, current_sparsity=0.9, target_sparsity=0.9)
        assert ckpt1.save_top_k == 3
        assert ckpt2.save_top_k == 1
