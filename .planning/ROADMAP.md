# Roadmap: Compression Robustness v0.1 Foundations

## Overview

This milestone establishes verified baselines before compression experiments. We first fix the CNCeleb training recipe to match WeSpeaker reference performance, then verify that magnitude-based pruning and Bregman learning implementations behave correctly. Each phase depends on the previous — there is no point testing compression if the baseline is broken.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [ ] **Phase 1: CNCeleb Baseline** - Fix training config, achieve EER within 1% of WeSpeaker reference
- [ ] **Phase 2: Pruning Verification** - Verify magnitude pruning at 30-90% sparsity levels (parallel with Phase 3)
- [ ] **Phase 3: Bregman Verification** - Compare Bregman implementation against reference (parallel with Phase 2)

## Phase Details

### Phase 1: CNCeleb Baseline
**Goal**: CNCeleb baseline achieves competitive EER with WeSpeaker reference
**Depends on**: Nothing (first phase)
**Requirements**: BASE-01, BASE-02, BASE-03
**Success Criteria** (what must be TRUE):
  1. Trained model achieves EER of 8.5% or lower on CNCeleb eval set
  2. Training config uses correct epoch count, LR schedule reaches ~0.0001 final, margin timing is proportional
  3. Single reproducible command trains the model from scratch to target EER
**Plans**: 1 plan

Plans:
- [ ] 01-01-PLAN.md — Fix hyperparameters and validate EER target

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

## Progress

**Execution Order:**
Phase 1 first, then Phases 2 and 3 in parallel: 1 -> (2 || 3)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. CNCeleb Baseline | 0/1 | Planned | - |
| 2. Pruning Verification | 0/TBD | Not started | - |
| 3. Bregman Verification | 0/TBD | Not started | - |

---
*Roadmap created: 2026-01-25*
*Milestone: v0.1 Foundations*
