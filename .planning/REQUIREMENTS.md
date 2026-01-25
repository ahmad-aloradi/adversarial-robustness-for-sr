# Requirements: Compression Robustness for SR

**Defined:** 2026-01-25
**Core Value:** Determine whether compressed speaker recognition models retain cross-domain robustness

## v0.1 Requirements

Requirements for foundations milestone. Establishes working baselines before compression experiments.

### Baseline

- [ ] **BASE-01**: CNCeleb baseline achieves ≤8.5% EER (within 1% of WeSpeaker's 7.8%)
- [ ] **BASE-02**: Training config verified correct (epochs scaled appropriately, LR schedule reaches ~0.0001 final, margin timing proportional)
- [ ] **BASE-03**: Reproducible training command documented

### Pruning Verification

- [ ] **PRUNE-01**: Pruning produces correct sparsity at target levels (30%, 50%, 70%, 90%)
- [ ] **PRUNE-02**: Pruned model EER degradation matches literature expectations (<0.25% at 60-70% sparsity)
- [ ] **PRUNE-03**: Verification tests pass (sparsity calculation, mask binary, no layer collapse)

### Bregman Verification

- [ ] **BREG-01**: Bregman implementation produces expected sparsity patterns
- [ ] **BREG-02**: Bregman results compared against reference implementation (https://github.com/TimRoith/BregmanLearning)
- [ ] **BREG-03**: Verification tests pass

## v1.0+ Requirements

Deferred to future milestone. Tracked but not in current roadmap.

### Systematic Experiments

- **EXP-01**: Multi-architecture support (ECAPA-TDNN, ResNet, Transformer models)
- **EXP-02**: Systematic experiment pipeline (all combinations of model × sparsity × dataset)
- **EXP-03**: Fixed sparsity level configurations (50%, 70%, 90%, 95%)
- **EXP-04**: Cross-domain evaluation matrix (train on VoxCeleb, eval on CNCeleb and vice versa)

### Paper Deliverables

- **PAPER-01**: Paper-ready results generation (tables, figures)
- **PAPER-02**: Reproducible experiment commands for all results
- **PAPER-03**: Statistical significance analysis

## Out of Scope

| Feature | Reason |
|---------|--------|
| VoicePrivacy attacker work | Already complete |
| New model architectures | Use existing WeSpeaker/SpeechBrain backends |
| Real-time inference optimization | Focus is compression-robustness tradeoff |
| Quantization | Focus on pruning and Bregman only for this study |
| VoxCeleb baseline tuning | CNCeleb is the cross-domain target |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| BASE-01 | Phase 1 | Pending |
| BASE-02 | Phase 1 | Pending |
| BASE-03 | Phase 1 | Pending |
| PRUNE-01 | Phase 2 | Pending |
| PRUNE-02 | Phase 2 | Pending |
| PRUNE-03 | Phase 2 | Pending |
| BREG-01 | Phase 3 | Pending |
| BREG-02 | Phase 3 | Pending |
| BREG-03 | Phase 3 | Pending |

**Coverage:**
- v0.1 requirements: 9 total
- Mapped to phases: 9
- Unmapped: 0 ✓

---
*Requirements defined: 2026-01-25*
*Last updated: 2026-01-25 after initial definition*
