---
phase: 02-pruning-verification
plan: 01
subsystem: pruning-verification
tags: [pruning, magnitude, cnceleb, testing, verification]
dependency_graph:
  requires: [phase-01-cnceleb-baseline]
  provides: [cnceleb-pruning-config, pruning-verification-tests]
  affects: [pruning-training, pruning-evaluation]
tech_stack:
  added: []
  patterns: [hydra-config-composition, pytest-mock-pattern, synthetic-model-testing]
key_files:
  created:
    - configs/experiment/sv/sv_pruning_mag_struct_cnceleb.yaml
    - tests/test_pruning_verification.py
  modified: []
decisions:
  - choice: Use WarmupExponentialLR instead of ReduceLROnPlateau for pruning
    rationale: ReduceLROnPlateau relies on validation metric monitoring which is suppressed during pruning ramp, making it non-functional. WarmupExponentialLR is epoch-based and works regardless of validation suppression.
    impact: Ensures learning rate scheduling works correctly throughout pruning ramp and fine-tuning phases
  - choice: Extend max_epochs to 30 (10 ramp + 20 fine-tuning)
    rationale: Provides sufficient time for scheduled pruning ramp-up and subsequent fine-tuning at target sparsity
    impact: Longer training time but better convergence at target sparsity
  - choice: Keep all sv_wespeaker augmentation and margin scheduling
    rationale: Phase 1 baseline (8.86% EER) used this recipe, so keeping it ensures apples-to-apples comparison
    impact: Pruned models directly comparable to baseline
metrics:
  duration_minutes: 3.1
  tasks_completed: 2
  tests_added: 6
  config_files_added: 1
  commits: 2
  completed_date: 2026-02-09
---

# Phase 02 Plan 01: CNCeleb Pruning Config and Verification Tests Summary

**One-liner:** CNCeleb pruning experiment config combining sv_wespeaker recipe with magnitude pruning, plus 6 verification tests covering sparsity accuracy, mask binary property, layer collapse prevention, and PyTorch API consistency.

## Objective

Create a CNCeleb-specific magnitude pruning experiment config that combines the proven Phase 1 baseline training recipe (SGD + WarmupExponentialLR, 8.86% EER) with scheduled ln_structured pruning. Add unit tests verifying pruning correctness without requiring full training runs.

## What Was Built

### 1. CNCeleb Pruning Experiment Config

**File:** `configs/experiment/sv/sv_pruning_mag_struct_cnceleb.yaml`

**Purpose:** Enables apples-to-apples comparison between baseline (8.86% EER) and pruned models by using identical training recipe.

**Key Properties:**
- Dataset: CNCeleb (not VoxCeleb like sv_pruning_mag_struct.yaml)
- Optimizer: SGD (lr=0.1, momentum=0.9, nesterov, wd=1e-4)
- LR Scheduler: WarmupExponentialLR (epoch-based, works with validation suppression)
- Pruning: ln_structured, scheduled from 0% to 90% over 10 epochs
- Max epochs: 30 (10 ramp + 20 fine-tuning)
- Augmentations: Speed perturbation with virtual speakers, RIR/noise mutually exclusive
- Margin scheduling: Progressive exponential (0.0 → 0.2)

**Verified:** Config composes correctly via Hydra, all fields resolve properly.

**Usage:**
```bash
python src/train.py experiment=sv/sv_pruning_mag_struct_cnceleb
python src/train.py experiment=sv/sv_pruning_mag_struct_cnceleb callbacks.model_pruning.final_amount=0.70
```

### 2. Pruning Verification Test Suite

**File:** `tests/test_pruning_verification.py`

**Purpose:** Verify pruning correctness using synthetic models (no real training data needed).

**Tests Implemented (6 total):**

1. **test_sparsity_calculation_known_values**
   - Creates models with exactly known sparsity (25%, 50%, 75%)
   - Verifies compute_sparsity returns correct values within 1e-6 tolerance
   - Uses manually zeroed tensor elements for ground truth

2. **test_sparsity_calculation_threshold**
   - Tests the 1e-12 threshold boundary handling
   - Sets values to 1e-13 (below), 1e-12 (at), 1e-11 (above)
   - Verifies below/at threshold counted as zero, above counted as non-zero

3. **test_masks_are_binary_after_pruning**
   - Applies ln_structured pruning at 30%, 70%, 90%
   - Verifies all mask values are exactly 0.0 or 1.0
   - Ensures no floating-point artifacts in masks

4. **test_no_fully_collapsed_layers**
   - Applies 90% global pruning to 3-layer model
   - Verifies no individual layer exceeds 99% sparsity
   - Prevents pathological cases where one layer is entirely zeroed

5. **test_pruner_matches_pytorch_reference**
   - Creates two identical models (same seed)
   - Applies pruning via MagnitudePruner vs direct torch.nn.utils.prune.ln_structured
   - Verifies resulting masks are identical
   - Ensures MagnitudePruner implements PyTorch API correctly

6. **test_sparsity_monotonically_increases_during_ramp**
   - Simulates full scheduled ramp (10 epochs) plus continuation
   - Verifies sparsity is non-decreasing at each epoch
   - Confirms scheduler produces monotonic sparsity targets

**Test Results:** All 6 tests pass.

## Deviations from Plan

None — plan executed exactly as written.

## Technical Details

### Config Composition Strategy

The config uses Hydra's defaults system to compose:
- Base modules: wespeaker_ecapa_tdnn, as_norm scoring, CNCeleb dataset
- Callbacks: Default + margin_scheduler + model_pruning
- Auto-scaled parameters: num_classes (virtual speakers), warmup/margin epochs

### WarmupExponentialLR vs ReduceLROnPlateau

**Critical difference:** ReduceLROnPlateau monitors validation metrics to decide when to reduce LR. During pruning ramp-up, validation is suppressed (trainer.limit_val_batches=0) to avoid misleading checkpoints during rapid sparsity increase. This makes ReduceLROnPlateau non-functional — it never receives validation metrics and never reduces LR.

WarmupExponentialLR is epoch-based and doesn't rely on validation metrics, so it works correctly throughout the entire training run.

### Test Design Patterns

All tests use synthetic Conv1d models to avoid dependencies on real data:
- Mock trainer with MagicMock (spec=Trainer)
- Conv1d layers (ndim=3) for structured pruning (pruning_dim=1)
- Deterministic seeding for reproducibility
- Direct parameter manipulation for known-sparsity tests

### Verification Philosophy

Tests verify correctness properties, not performance:
- Sparsity calculation accuracy (PRUNE-01)
- Mask binary property (PRUNE-03)
- No layer collapse (PRUNE-03)
- PyTorch API consistency (PRUNE-03)
- Monotonic ramp behavior (PRUNE-03)

Performance (EER) will be verified in subsequent Phase 2 plans via actual training runs.

## Impact on Project

### Enables Phase 2 Experiments

This plan provides the foundation for Phase 2 pruning experiments:
- Config ready for multiple sparsity targets (50%, 70%, 90%)
- Tests verify pruning behaves correctly before expensive training runs
- Direct comparison to Phase 1 baseline (same training recipe)

### Ensures Correctness

Unit tests catch pruning bugs early:
- Sparsity miscalculation would be caught immediately
- Layer collapse would be detected before training
- PyTorch API misuse would fail tests
- Scheduler bugs would be caught by monotonicity test

### Reduces Risk

By verifying correctness with synthetic models:
- Avoid wasting GPU hours on broken pruning
- Catch configuration errors before training
- Verify assumptions about PyTorch pruning API

## Next Steps

**Immediate (Phase 2):**
1. Run training with sv_pruning_mag_struct_cnceleb at 50%, 70%, 90% sparsity
2. Evaluate pruned models on CNCeleb test set
3. Compare EER to baseline (8.86%) and document accuracy-sparsity tradeoff
4. Verify sparsity metrics match expected targets

**Future (Phase 3):**
1. Create analogous Bregman pruning config for CNCeleb
2. Add Bregman-specific verification tests
3. Compare Bregman vs magnitude pruning at equivalent sparsity

## Files Changed

### Created (2 files)
- `configs/experiment/sv/sv_pruning_mag_struct_cnceleb.yaml` (191 lines)
- `tests/test_pruning_verification.py` (273 lines)

### Modified
None

## Commits

1. **cf160ef** - feat(02-01): create CNCeleb magnitude pruning experiment config
2. **696749e** - test(02-01): add pruning verification test suite

## Self-Check: PASSED

**Created files exist:**
```
FOUND: configs/experiment/sv/sv_pruning_mag_struct_cnceleb.yaml
FOUND: tests/test_pruning_verification.py
```

**Commits exist:**
```
FOUND: cf160ef
FOUND: 696749e
```

**Tests pass:**
```
6 passed in 0.07s
```

**Config composes:**
```
✓ Config composes successfully
✓ Dataset: CNCeleb
✓ Optimizer: SGD
✓ LR Scheduler: WarmupExponentialLR
✓ Pruning callback: MagnitudePruner
✓ Pruning method: ln_structured
```
