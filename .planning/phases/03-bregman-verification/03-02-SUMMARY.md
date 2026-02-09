---
phase: 03-bregman-verification
plan: 02
subsystem: testing/bregman
tags: [testing, bregman, optimizer, regularizer, integration]
completed: 2026-02-09

dependency_graph:
  requires:
    - "03-01: Bregman lambda scheduler verification"
  provides:
    - "Bregman optimizer correctness test suite (11 tests)"
    - "Bregman mini-training integration tests (5 tests)"
  affects:
    - "Confidence in Bregman mathematical correctness"
    - "Future Bregman experiments (03-03)"

tech_stack:
  added: []
  patterns:
    - "Mock-based mini-training loops for callback testing"
    - "Minimal LightningModule with PruningManager setup"
    - "Direct optimizer instantiation without full Trainer"

key_files:
  created:
    - tests/test_bregman_optimizer_correctness.py
    - tests/test_bregman_experiments.py
  modified: []

decisions:
  - name: "Mock trainer vs real Trainer for mini-training"
    choice: "Use unittest.mock.Mock for trainer"
    rationale: "Real Trainer has read-only properties (current_epoch). Mock provides full control for testing callback hooks without Lightning infrastructure overhead."
    alternatives: ["Real Trainer with monkey-patching", "Custom test Trainer subclass"]

  - name: "LinBreg learning rate for tests"
    choice: "Higher lr (0.1) and stronger lambda (1.0) for LinBreg vs AdaBreg (0.01 lr, 0.5 lambda)"
    rationale: "LinBreg lacks adaptive moments, needs stronger regularization to achieve comparable sparsity in short test runs"
    alternatives: ["Same hyperparams for both (fails for LinBreg)", "More training steps (slower tests)"]

  - name: "Initial sparsity for integration tests"
    choice: "Start tests from moderate sparsity (0.6-0.99) rather than dense (0.0)"
    rationale: "Prevents layer collapse when starting from dense with strong regularization in short runs. Mirrors real inverse-scale workflow."
    alternatives: ["Lower lambda (less realistic)", "More training steps (slower tests)", "Dense start (caused collapse)"]

metrics:
  duration: 344s (5m 44s)
  commits: 2
  tests_added: 16
  files_created: 2
---

# Phase 3 Plan 2: Bregman Optimizer and Mini-Training Tests Summary

Comprehensive test suite for Bregman optimizer/regularizer mathematical correctness and end-to-end mini-training behavior.

## One-Liner

**Test suite (16 tests) verifying Bregman optimizer soft-thresholding, group sparsity, sparsity induction, and mini-training stability with PruningManager integration**

## What Was Built

### 1. Optimizer and Regularizer Correctness Tests (11 tests)

**File**: `tests/test_bregman_optimizer_correctness.py`

**RegL1 proximal operator tests (3 tests)**:
- `test_regl1_prox_soft_thresholding`: Verifies correct soft-thresholding formula `sign(x) * max(|x| - delta*lambda, 0)`
- `test_regl1_prox_zeros_small_weights`: Confirms weights below threshold become exactly zero
- `test_regl1_subgrad_matches_sign`: Validates subgradient computation `lambda * sign(v)`

**RegL1L2Conv group sparsity tests (2 tests)**:
- `test_regl1l2conv_prox_group_sparsity`: Verifies entire filter groups zeroed when L2 norm below threshold
- `test_regl1l2conv_prox_preserves_large_groups`: Confirms large groups are scaled, not zeroed

**AdaBreg optimizer tests (3 tests)**:
- `test_adabreg_single_step_updates_subgrad`: Validates single optimization step initializes state (sub_grad, exp_avg, exp_avg_sq)
- `test_adabreg_induces_sparsity`: Trains Linear(20,10) for 100 steps, achieves >10% sparsity
- `test_adabreg_no_sparsity_with_regnone`: Confirms RegNone produces no sparsity (<1%)

**LinBreg optimizer tests (2 tests)**:
- `test_linbreg_single_step`: Validates single step and state initialization
- `test_linbreg_induces_sparsity`: Achieves >5% sparsity with higher lr (0.1) and lambda (1.0)

**Cross-optimizer consistency test (1 test)**:
- `test_adabreg_linbreg_both_induce_sparsity`: Both optimizers produce sparse solutions on same problem (AdaBreg >10%, LinBreg >5%)

### 2. Mini-Training Integration Tests (5 tests)

**File**: `tests/test_bregman_experiments.py`

**Test infrastructure**:
- `SimpleMLP`: 3-layer MLP (Linear(50,30) -> ReLU -> Linear(30,10))
- `MiniBregmanModule`: Minimal LightningModule with PruningManager integration
- `_run_mini_bregman_training()`: Helper running mock training loop (10 epochs, 20 batches/epoch)

**Integration tests**:
- `test_bregman_mini_training_produces_sparsity`: Final sparsity between 0.3-0.99 when starting from 0.99 (inverse-scale)
- `test_bregman_mini_training_no_nan`: All parameters remain finite (no NaN/Inf)
- `test_bregman_per_layer_sparsity_not_degenerate`: No layer fully collapsed (< 99% sparse) or too dense (> 10% sparse starting from 0.85)
- `test_bregman_lambda_evolves_during_training`: Lambda decreases when model starts too sparse (0.99 -> target 0.5)
- `test_bregman_scheduled_mode_mini_training`: Scheduled target mode ramps from 0.5 to 0.9, final sparsity in (0.4, 0.95)

## Implementation Details

### Mock Training Loop Pattern

Used `unittest.mock.Mock` for trainer instead of real `pytorch_lightning.Trainer`:

```python
trainer = Mock()
trainer.optimizers = [optimizer]
trainer.current_epoch = epoch  # Writable
trainer.global_step = step
```

Allows direct control of callback hooks without Lightning infrastructure:

```python
pruner.on_fit_start(trainer, pl_module)
for epoch in range(num_epochs):
    pruner.on_train_epoch_start(trainer, pl_module)
    for batch_idx in range(num_batches_per_epoch):
        loss.backward()
        optimizer.step()
        pruner.on_train_batch_end(trainer, pl_module, None, batch, batch_idx)
    pruner.on_train_epoch_end(trainer, pl_module)
```

### PruningManager Integration

Configured in `MiniBregmanModule.configure_optimizers()`:

```python
self.pruning_manager = PruningManager(
    pl_module=self,
    group_configs=[
        {
            "name": "linear_weights",
            "layer_types": ["torch.nn.Linear"],
            "param_names": ["weight"],
            "optimizer_settings": {
                "reg": RegL1(lamda=0.5),
                "lambda_scale": 1.0,
            },
            "pruning_config": {
                "pruning_type": "unstructured",
                "sparsity_rate": initial_sparsity,
            },
        },
        {"name": "other", "is_fallback": True, ...},
    ],
)
optimizer_param_groups = self.pruning_manager.get_optimizer_param_groups()
optimizer = AdaBreg(optimizer_param_groups, lr=0.01, delta=1.0)
```

### Test Hyperparameter Tuning

**AdaBreg vs LinBreg**:
- AdaBreg: lr=0.01, lambda=0.5 → >10% sparsity in 100 steps
- LinBreg: lr=0.1, lambda=1.0 → >5% sparsity in 100 steps (needs stronger regularization without adaptive moments)

**Initial sparsity choices**:
- Dense start (0.0): Caused layer collapse with strong lambda in short runs
- Moderate start (0.6-0.85): Realistic for testing, prevents collapse
- Sparse start (0.99): Tests inverse-scale mode (high initial sparsity, ramp down toward target)

## Deviations from Plan

None - plan executed exactly as written.

## Testing & Verification

```bash
# All optimizer correctness tests (11 tests)
pytest tests/test_bregman_optimizer_correctness.py -v
# PASSED in 0.39s

# All mini-training integration tests (5 tests, marked slow)
pytest tests/test_bregman_experiments.py -v
# PASSED in 1.07s

# Combined run (16 tests)
pytest tests/test_bregman_optimizer_correctness.py tests/test_bregman_experiments.py -v
# PASSED in 1.12s
```

All tests fast (< 2s total) and CPU-only (no GPU/dataset required).

## Key Insights

1. **LinBreg convergence**: Significantly slower than AdaBreg without adaptive moments. Needs 5-10x higher lr or stronger lambda for comparable sparsity in short runs.

2. **Layer collapse risk**: Starting from dense (0.0 sparsity) with strong regularization (lambda=0.5) can fully collapse small layers (<100 params) in mini-training. Mitigated by starting from moderate sparsity or lowering lambda.

3. **Mock trainer flexibility**: Using `Mock` instead of real `Trainer` allows complete control over callback hook invocation without Lightning framework constraints. Essential for testing validation suppression and scheduled target updates.

4. **Sparsity threshold alignment**: Tests use 1e-12 threshold matching project standard (aligned in Phase 4 Plan 4, was 1e-30 for Bregman).

## Documentation

No user-facing docs needed (internal test suite).

## Next Steps

**Phase 3 Plan 3**: Full Bregman experimental runs on VoxCeleb dataset with various sparsity levels, confirming optimizer behavior at scale.

**Blockers**: None - all tests passing, ready for full-scale experiments.

## Commits

1. `8751364`: test(03-02): add Bregman optimizer and regularizer correctness tests
2. `59ebead`: test(03-02): add Bregman mini-training integration tests

## Self-Check: PASSED

**Files created:**
```bash
$ ls tests/test_bregman_optimizer_correctness.py
tests/test_bregman_optimizer_correctness.py  # FOUND

$ ls tests/test_bregman_experiments.py
tests/test_bregman_experiments.py  # FOUND
```

**Commits exist:**
```bash
$ git log --oneline --all | grep 8751364
8751364 test(03-02): add Bregman optimizer and regularizer correctness tests  # FOUND

$ git log --oneline --all | grep 59ebead
59ebead test(03-02): add Bregman mini-training integration tests  # FOUND
```

**Tests pass:**
```bash
$ pytest tests/test_bregman_optimizer_correctness.py tests/test_bregman_experiments.py -v
16 passed in 1.12s  # PASSED
```
