---
phase: 04-verify-pruning-and-bregman-correctness
plan: 02
subsystem: testing
tags: [bregman, lambda-scheduler, verification, unit-tests, integration-tests]

requires:
  - 04-01 # Phase 4 research/documentation

provides:
  - comprehensive-lambda-scheduler-verification
  - bregman-pruner-integration-tests
  - lambda-update-frequency-validation
  - ema-smoothing-validation
  - checkpoint-state-validation

affects:
  - 04-03 # Future Bregman implementation work
  - training-reliability # Confidence in lambda scheduling behavior

tech-stack:
  added: []
  patterns:
    - pytest-mocking-for-pytorch-lightning-callbacks
    - unit-test-isolation-for-schedulers

key-files:
  created:
    - tests/test_bregman_lambda_verification.py
  modified: []

key-decisions:
  - id: lambda-volatility-metric
    decision: "Measure EMA effectiveness via direction change count, not variance"
    rationale: "When sparsity oscillates around target, EMA prevents lambda direction reversals (increase/decrease flip-flopping). This is more meaningful than overall variance when both sparsity values are on same side of target."
    alternatives: ["variance of lambda values", "step-to-step absolute differences"]
    impact: "Test properly validates EMA smoothing benefit for training stability"

duration: "4m"
completed: 2026-02-06
---

# Phase 04 Plan 02: Bregman Lambda Verification Tests Summary

**One-liner:** 12 comprehensive tests verifying lambda scheduler updates once per batch, moves in correct direction, respects bounds, EMA smooths oscillations, and checkpoint state is preserved.

## Performance

- **Duration:** 4 minutes
- **Started:** 2026-02-06 17:36:57 UTC
- **Completed:** 2026-02-06 17:40:47 UTC
- **Tasks:** 2/2 completed
- **Files modified:** 1 created

## Accomplishments

### Lambda Scheduler Unit Tests (9 tests)
1. **Update frequency:** Confirmed lambda updates exactly once per `step()` call (300 calls = 300 updates)
2. **Direction correctness:**
   - Lambda increases monotonically when sparsity < target
   - Lambda decreases monotonically when sparsity > target
   - Lambda unchanged when sparsity == target
3. **Bounds enforcement:** Lambda never exceeds [min_lambda, max_lambda] range
4. **EMA smoothing:** Reduces direction reversals when sparsity oscillates around target (fewer sign changes in lambda updates)
5. **Checkpoint save/restore:** Full state preservation (lambda, EMA, hyperparameters)
6. **Resume path:** `last_sparsity` parameter correctly initializes EMA on first post-resume step
7. **Input validation:** Rejects invalid sparsity values (≤0, >1, NaN, Inf)

### BregmanPruner Integration Tests (3 tests)
8. **Per-batch updates:** Lambda updates once per `on_train_batch_end` call (10 calls = 10 updates)
9. **Lambda propagation:** Scheduler lambda correctly propagated to `reg.lamda` in optimizer param groups
10. **Lambda scaling:** `lambda_scale` parameter correctly multiplies scheduler lambda

## Task Commits

| Task | Description | Commit | Files |
|------|-------------|--------|-------|
| 1+2  | Write LambdaScheduler unit tests + BregmanPruner integration tests | add720f | tests/test_bregman_lambda_verification.py |

## Files Created

```
tests/test_bregman_lambda_verification.py  (485 lines)
├── 9 LambdaScheduler unit tests
└── 3 BregmanPruner integration tests
```

## Files Modified

None (read-only verification plan)

## Decisions Made

### Lambda Volatility Metric
**Decision:** Measure EMA effectiveness via direction change count, not variance

**Context:** Initial test used variance of lambda values, which failed because when sparsity alternates on the same side of target (e.g., 0.5 and 0.7 both below 0.9), lambda moves in one direction regardless of EMA. Variance increased with EMA due to lag effect.

**Chosen approach:** Count sign changes in lambda updates (direction reversals). When sparsity oscillates *around* target (some below, some above), raw sparsity causes rapid lambda direction changes (increase ↔ decrease), while EMA smoothing prevents these reversals.

**Impact:** Test now properly validates EMA's benefit for training stability: fewer lambda direction changes means smoother convergence toward target sparsity.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

### EMA Test Design Challenge
**Issue:** Initial test failed because measuring variance of lambda values didn't properly capture EMA's benefit.

**Root cause:** When alternating sparsity values are both below (or both above) target, lambda moves in one direction. EMA lag causes different magnitude of updates but not necessarily lower variance.

**Resolution:** Changed test to measure direction reversals (sign changes in lambda deltas) with sparsity oscillating around target. This properly demonstrates EMA's smoothing effect.

**Learning:** EMA smoothing's primary benefit is preventing rapid policy reversals (increase ↔ decrease flip-flopping), not reducing overall lambda range.

## Next Phase Readiness

### Blockers
None

### Concerns
None - all tests pass, lambda scheduler behavior confirmed correct

### Dependencies
- Phase 04 Plan 03 (if exists): Can confidently use lambda scheduler in future Bregman training experiments

### Recommendations
1. **Add slow integration test:** Full training loop with BregmanPruner + LambdaScheduler (mark with `@pytest.mark.slow`)
2. **Monitor production behavior:** Compare lambda traces in actual training to test behavior
3. **Extend to adaptive schedules:** If adaptive target sparsity is added, verify dynamic target updates work correctly

## Self-Check: PASSED

All created files verified:
- tests/test_bregman_lambda_verification.py ✓

All commits verified:
- add720f ✓
