# Roadmap: Compression Robustness v0.1 Foundations

## Overview

This milestone establishes verified baselines before compression experiments. We first fix the CNCeleb training recipe to match WeSpeaker reference performance, then verify that magnitude-based pruning and Bregman learning implementations behave correctly. Each phase depends on the previous — there is no point testing compression if the baseline is broken.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [ ] **Phase 1: CNCeleb Baseline** - Fix training config, achieve EER within 1% of WeSpeaker reference
- [x] **Phase 1.1: Eval Auto-Config (INSERTED)** - Complete eval.py checkpoint averaging implementation ✓
- [ ] **Phase 2: Pruning Verification** - Verify magnitude pruning at 30-90% sparsity levels (parallel with Phase 3)
- [ ] **Phase 3: Bregman Verification** - Compare Bregman implementation against reference (parallel with Phase 2)
- [ ] **Phase 4: Verify Pruning & Bregman Correctness** - Fix pruning validation-during-ramp bugs, verify Bregman lambda update frequency

## Phase Details

### Phase 1: CNCeleb Baseline
**Goal**: CNCeleb baseline achieves competitive EER with WeSpeaker reference
**Depends on**: Nothing (first phase)
**Requirements**: BASE-01, BASE-02, BASE-03
**Success Criteria** (what must be TRUE):
  1. Trained model achieves EER of 8.5% or lower on CNCeleb eval set
  2. Training config uses correct epoch count, LR schedule reaches ~0.0001 final, margin timing is proportional
  3. Single reproducible command trains the model from scratch to target EER
**Plans**: 2 plans

Plans:
- [ ] 01-01-PLAN.md — Fix hyperparameters and validate EER target (Wave 1)
- [ ] 01-02-PLAN.md — Create checkpoint averaging script (Wave 2)

### Phase 1.1: Eval Auto-Config from Experiment Directory (INSERTED)
**Goal**: Complete eval.py implementation to automatically parse and apply configs from experiment directories
**Depends on**: Phase 1 (uses trained models from baseline)
**Requirements**: EVAL-01
**Success Criteria** (what must be TRUE):
  1. Running `python src/eval.py exp_dir=logs/train/runs/2026-01-27_16-00-10` automatically loads all training configs
  2. Checkpoint averaging (`use_avg_ckpt=True`) works correctly with the TODO implementation completed
  3. All eval-specific overrides (predict, ckpt_path, etc.) properly merge with experiment configs
**Plans**: 1 plan

Plans:
- [x] 01.1-01-PLAN.md — Extract checkpoint averaging utility and wire into eval.py ✓

### Phase 2: Pruning Verification
**Goal**: Magnitude-based pruning produces expected sparsity and accuracy behavior
**Depends on**: Phase 1 (need working baseline to prune)
**Requirements**: PRUNE-01, PRUNE-02, PRUNE-03
**Success Criteria** (what must be TRUE):
  1. Pruned models at 30%, 50%, 70%, 90% sparsity have actual sparsity within 1% of target
  2. Model at 60-70% sparsity shows EER degradation under 0.25% compared to dense baseline
  3. All verification tests pass: sparsity calculation correct, masks are binary (0 or 1), no layer fully collapsed (all zeros)
  4. Pruning behavior matches PyTorch torch.nn.utils.prune reference
**Plans**: TBD

Plans:
- [ ] 02-01: TBD

### Phase 3: Bregman Verification
**Goal**: Bregman learning produces expected sparsity patterns matching reference implementation
**Depends on**: Phase 1 (need working baseline; parallel with Phase 2)
**Requirements**: BREG-01, BREG-02, BREG-03
**Success Criteria** (what must be TRUE):
  1. Bregman pruning produces sparsity patterns at target levels
  2. Results compared against BregmanLearning reference (github.com/TimRoith/BregmanLearning) show equivalent behavior
  3. All verification tests pass for Bregman-specific behavior
**Plans**: TBD

Plans:
- [ ] 03-01: TBD

### Phase 4: Verify Pruning & Bregman Correctness
**Goal**: Fix known bugs in pruning callback and verify Bregman lambda update correctness
**Depends on**: Phase 1 (need working baseline)
**Requirements**: PRUNE-01, BREG-01
**Success Criteria** (what must be TRUE):
  1. Pruning callback suppresses validation (EarlyStopping + ModelCheckpoint) while sparsity < target, even when `check_val_every_n_epoch` triggers validation
  2. No spurious "New best score" recorded during sparsity ramp-up phase
  3. Bregman lambda updates occur at correct frequency matching reference implementation
  4. All verification tests pass for both pruning and Bregman behaviors
**Plans**: 2 plans

Plans:
- [ ] 04-01-PLAN.md — Fix pruning validation suppression via limit_val_batches + tests (Wave 1)
- [ ] 04-02-PLAN.md — Verify Bregman lambda update correctness + tests (Wave 1)

**Known Issues (from logs):**
1. With `check_val_every_n_epoch=5`, validation still runs every epoch during pruning ramp — "New best score" recorded at low sparsity (e.g., 0.001 accuracy at epoch 3, then 0.005 at epoch 18)
2. Bregman lambda update frequency needs verification against reference (github.com/TimRoith/BregmanLearning)

## Progress

**Execution Order:**
Phase 1 first, then Phase 1.1, then Phases 2 and 3 in parallel, then Phase 4: 1 -> 1.1 -> (2 || 3) -> 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. CNCeleb Baseline | 0/2 | Planned | - |
| 1.1 Eval Auto-Config (INSERTED) | 1/1 | ✓ Complete | 2026-01-31 |
| 2. Pruning Verification | 0/TBD | Not started | - |
| 3. Bregman Verification | 0/TBD | Not started | - |
| 4. Verify Pruning & Bregman Correctness | 0/2 | Planned | - |

---
*Roadmap created: 2026-01-25*
*Last updated: 2026-02-06 - Added Phase 4 (Verify Pruning & Bregman Correctness)*
*Milestone: v0.1 Foundations*
