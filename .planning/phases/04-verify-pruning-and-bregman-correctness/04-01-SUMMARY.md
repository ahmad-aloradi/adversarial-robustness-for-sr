---
phase: 04
plan: 01
subsystem: pruning
tags: [pruning, validation-suppression, callbacks, testing]
requires: [magnitude-pruner, pytorch-lightning]
provides: [validation-suppression-fix, verification-tests]
affects: [pruning-training-efficiency]
tech-stack:
  added: []
  patterns: [validation-suppression-via-limit_val_batches]
key-files:
  created:
    - tests/test_pruning_validation_suppression.py
  modified:
    - src/callbacks/pruning/prune.py
key-decisions:
  - id: validation-suppression-mechanism
    choice: Use trainer.limit_val_batches=0 instead of callback state manipulation
    rationale: Direct trainer control prevents validation execution entirely, avoiding compute waste and misleading logs
  - id: restoration-timing
    choice: Restore validation when new_sparsity >= (final_amount - 1e-4)
    rationale: Ensures validation resumes exactly when target sparsity is reached
duration: 2.3 min
completed: 2026-02-06
---

# Phase 04 Plan 01: Fix Pruning Validation Suppression Summary

**One-liner:** Fixed MagnitudePruner to properly suppress validation during sparsity ramp-up using `trainer.limit_val_batches=0`, eliminating wasted compute and spurious checkpoint messages.

## Performance

- **Duration:** 2.3 minutes
- **Started:** 2026-02-06T17:35:57Z
- **Completed:** 2026-02-06T17:38:14Z
- **Tasks completed:** 2/2
- **Files modified:** 2 (1 created, 1 modified)

## Accomplishments

### Fixed Validation Suppression Mechanism

**Problem:** The original MagnitudePruner attempted to suppress validation during sparsity ramp-up by manipulating EarlyStopping and ModelCheckpoint state (resetting counters, setting `save_top_k=0`). However, validation still executed on every epoch, wasting compute and producing misleading "New best score" log messages at low sparsity levels where model performance is poor.

**Solution:** Implemented proper validation suppression by directly controlling `trainer.limit_val_batches`:
- Set `trainer.limit_val_batches = 0` when `scheduled_pruning=True` and current sparsity < target
- Saved original `limit_val_batches` value on first suppression
- Restored original value when target sparsity reached (`new_sparsity >= final_amount - 1e-4`)
- Added log message "Target sparsity reached. Restoring validation." for visibility

**Impact:**
- Validation completely skipped during ramp-up (no compute wasted)
- No misleading checkpoint messages during low-sparsity epochs
- Clean validation resume when model is at target sparsity
- Existing tracker management (EarlyStopping/ModelCheckpoint reset) preserved as secondary safety net

### Comprehensive Verification Tests

Created `tests/test_pruning_validation_suppression.py` with 6 tests covering all edge cases:

1. **test_validation_disabled_during_ramp** - Confirms `limit_val_batches=0` during epochs 0-3 of ramp
2. **test_validation_restored_after_target_reached** - Confirms restoration to original value after ramp completes
3. **test_original_limit_val_batches_preserved** - Verifies custom values (e.g., 0.5) are preserved and restored correctly
4. **test_no_validation_suppression_without_scheduled_pruning** - Ensures no modification when `scheduled_pruning=False`
5. **test_early_stopping_reset_on_target_reached** - Validates EarlyStopping state reset when validation resumes
6. **test_model_checkpoint_save_top_k_restored** - Validates ModelCheckpoint `save_top_k` restoration

All tests pass, confirming the fix works correctly across all scenarios.

## Task Commits

| Task | Description | Commit | Files |
|------|-------------|--------|-------|
| 1 | Fix validation suppression in MagnitudePruner | d039b47 | src/callbacks/pruning/prune.py |
| 2 | Write verification tests for pruning validation suppression | 08f1834 | tests/test_pruning_validation_suppression.py |

## Files Created/Modified

### Created
- `tests/test_pruning_validation_suppression.py` (212 lines)
  - Helper function `_make_pruner_and_mocks()` for test setup
  - 6 comprehensive test functions covering all edge cases
  - Uses mocking (MagicMock) for trainer/pl_module, real nn.Module for model

### Modified
- `src/callbacks/pruning/prune.py`
  - Added `_original_limit_val_batches` state variable in `__init__`
  - Added validation suppression logic in `on_train_epoch_start` (after pruning verification)
  - Preserved existing `_manage_metric_trackers` method for state reset functionality
  - Added log message for validation restoration

## Decisions Made

### 1. Validation Suppression Mechanism

**Decision:** Use `trainer.limit_val_batches = 0` for validation suppression during ramp-up.

**Alternatives considered:**
- Override `on_validation_epoch_start` hook - Rejected: fights Lightning framework, complex
- Use `trainer.should_stop` - Rejected: nuclear option, stops training entirely
- Keep only tracker state manipulation - Rejected: doesn't prevent validation execution

**Rationale:** Direct trainer control is the cleanest approach. It prevents validation from executing entirely (no compute waste), works seamlessly with Lightning 2.x callback system, and doesn't interfere with framework internals. The existing tracker management remains as a secondary safety net for state reset when validation resumes.

### 2. Restoration Timing

**Decision:** Restore validation when `new_sparsity >= (final_amount - 1e-4)` in `on_train_epoch_start`.

**Rationale:** This ensures validation resumes exactly when the model reaches target sparsity. The tolerance of `1e-4` accounts for floating-point precision. Restoration happens in `on_train_epoch_start` (after pruning is applied and verified) so that validation runs immediately on that epoch.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

### Test Failures (Resolved)

**Issue 1:** `test_no_validation_suppression_without_scheduled_pruning` failed with AssertionError: "epochs_to_ramp should be None when scheduled_pruning is False."

**Resolution:** Added `epochs_to_ramp=0` parameter to MagnitudePruner constructor in test. The pruner enforces that `epochs_to_ramp` must be 0 or None when `scheduled_pruning=False`.

**Issue 2:** `test_early_stopping_reset_on_target_reached` failed - wait_count was not reset to 0.

**Resolution:** The test was checking state after epoch 4, but target sparsity is only reached at epoch 5 (when `current_epoch >= epochs_to_ramp`). Updated test to run through epoch 5 and check state after `on_train_epoch_end` at that point.

## Next Phase Readiness

### Ready for Phase 04-02: Verify Bregman Lambda Update Frequency

**Prerequisites met:**
- Pruning callback validation behavior is now correct
- Tests verify the fix works across all scenarios
- No regressions in existing checkpoint save/load logic

**Blockers:** None

**Recommendations:**
1. Apply similar verification approach to Bregman Learning callbacks
2. Consider adding integration tests that run actual training with MagnitudePruner to verify end-to-end behavior
3. Monitor training logs in next pruning experiments to confirm no spurious checkpoint messages appear

## Self-Check: PASSED

All claimed files created and all commit hashes verified:
- tests/test_pruning_validation_suppression.py: EXISTS
- src/callbacks/pruning/prune.py: MODIFIED (verified)
- Commit d039b47: EXISTS
- Commit 08f1834: EXISTS
