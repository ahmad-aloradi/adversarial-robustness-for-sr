# Phase 4: Verify Pruning & Bregman Correctness - Context

**Gathered:** 2026-02-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Centralize and fix validation/checkpoint suppression during sparsity ramp-up in both MagnitudePruner and BregmanPruner. The current approach in MagnitudePruner fails to fully suppress validation — logs show "New best score" during ramp-up despite `limit_val_batches=0`. BregmanPruner has no tracker management at all. This phase extracts shared suppression logic into a centralized module and unifies sparsity computation.

**Evidence of bug (from training logs):**
```
Metric valid/MulticlassAccuracy improved. New best score: 0.823
[Pruning Monitor] Epoch 10: Target=71.82% | Result=71.83% | Status: Pruned
Metric valid/MulticlassAccuracy improved. New best score: 0.813
```

</domain>

<decisions>
## Implementation Decisions

### Suppression strategy
- Claude decides hook timing and whether to use belt-and-suspenders (disable trackers explicitly in addition to bypassing validation)
- Log transitions only: log once when suppression starts and once when restored, no per-epoch noise
- Single unified tolerance of 1e-3 (0.1%) for both pruners when checking if target sparsity is reached
- Handle both scheduled and non-scheduled pruning (non-scheduled could benefit from a single validation skip at epoch 0)

### Tracker reset behavior
- Full reset to fresh state when target sparsity is reached: best_score to inf/-inf, wait_count=0, best_k_models={}
- Ideally no checkpoints should exist from suppression period. If validation was somehow run (not recommended), clear checkpoint files too
- EarlyStopping gets full patience window from target reached (wait_count=0 at restoration)

### Code centralization
- New file: `src/callbacks/pruning/suppression.py`
- Sparsity-agnostic: suppression logic only needs (current_sparsity, target_sparsity), doesn't care about pruning method
- Pruner computes sparsity and passes values to suppressor (suppressor is a gatekeeper, not a sparsity calculator)
- Explicit call from pruner: `self.suppressor.check(current, target)` — clear dependency, no magic
- Internal instantiation by pruner (not Hydra config injection) — simpler
- Prioritize simplicity and robustness in the centralization approach

### Checkpoint save/load
- Save suppression state to checkpoint under its own top-level key (`validation_suppression_state`)
- Strict loading: old checkpoints without the key fail loudly — no backward compatibility
- On resume mid-ramp, suppression continues seamlessly from saved state

### Parameter management (sparsity computation)
- Centralize sparsity computation only — parameter collection stays separate per pruner
- Shared function uses threshold-based zero detection (abs <= threshold) with sensible default (e.g., 1e-6)
- Support both global sparsity ratio and optional per-layer breakdown for debugging
- Location: `src/callbacks/pruning/utils/shared_pruners_utils.py` alongside the suppressor utilities

### Claude's Discretion
- Hook timing for suppression (on_train_epoch_start vs other hooks)
- Whether to also explicitly disable EarlyStopping/ModelCheckpoint as defense-in-depth
- Centralization approach (mixin vs base class vs composition)
- Sparsity threshold default value

</decisions>

<specifics>
## Specific Ideas

- The root cause of the bug is likely that ModelCheckpoint fires on_validation_end BEFORE the pruner's on_train_epoch_end resets trackers — validation suppression and tracker management need to happen at epoch start, not end
- BregmanPruner's `_compute_sparsity` uses `p.abs() <= 1e-30` while MagnitudePruner uses `param == 0` — the unified function should handle both via configurable threshold
- BregmanPruner has a defensive `on_validation_epoch_start` hook that logs a CRITICAL warning but doesn't functionally suppress — this should be replaced by the centralized approach

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-verify-pruning-and-bregman-correctness*
*Context gathered: 2026-02-08*
