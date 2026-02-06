# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-25)

**Core value:** Determine whether compressed speaker recognition models retain cross-domain robustness
**Current focus:** Phase 4 - Verify Pruning & Bregman Correctness

## Current Position

Phase: 4 of 4 (Verify Pruning & Bregman Correctness)
Plan: 2 of 2 executed
Status: Phase complete
Last activity: 2026-02-06 - Completed 04-02-PLAN.md (Bregman Lambda Verification Tests)

Progress: [███.......] 30% (Phase 1 nearly done, Phase 4 complete)

## Performance Metrics

**Velocity:**
- Total plans completed (GSD): 3
- Plans completed outside GSD: ~4 major changes
- Average duration: 3.1 min
- Total execution time: 9.3 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1.1 | 1 | 3.0 min | 3.0 min |
| 4 | 2 | 6.3 min | 3.2 min |

*Updated after each plan completion*

## Accumulated Context

### Recent Work (outside GSD)

1. **Checkpoint averaging** — Done. Extracted utility, wired into eval.py callback (`src/callbacks/checkpoint_averaging.py`)
2. **Auto-parse configs** — Done. Eval auto-config works for basic cases. Metadata auto-parse implemented: eval.py now uses snapshotted `metadata/src/` and `metadata/configs/` from experiment dirs for backward-compatible evaluation.
3. **[Major] Scoring normalization fix** — Fixed normalization bugs in `src/modules/scoring.py` (L187-219). Embeddings now properly L2-normalized after centering and after aggregation. Best model achieves **9.5% EER** (target: 8.5%)
4. **[Important] Augmentation overhaul** — Done. Updated augmentations to use mutually exclusive RIR/noise and virtual speakers after resampling (WeSpeaker style). Virtual speaker label mismatch bug fixed (1.0 speed moved to index 0). Verified correct behavior.

### Best Model Configuration (9.5% EER)

```
VAD + sv_wespeaker + all augs (+ specaugment)
optimizer: SGD (lr=0.1, momentum=0.9, nesterov, wd=1e-4)
lr_scheduler: WarmupExponentialLR (warmup_epochs=1, warmup_start=0.01, gamma=0.8)
margin_scheduler: ProgressiveMarginScheduler (init=0.0, final=0.2, warmup=2, start=2)
trainer: max_epochs=20, gradient_clip_val=1.0
```

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

| Phase | Decision | Choice | Impact |
|-------|----------|--------|--------|
| 01.1-01 | Checkpoint averaging utility | Extract from callback into standalone function | Enables eval.py checkpoint averaging without callback infrastructure |
| 01.1-01 | Import location | Use package-level import (utils.average_checkpoints) | Cleaner code following Python conventions |
| 01 (manual) | Scoring normalization | L2-normalize after centering AND after aggregation | Fixed multi-enrollment scoring, contributed to 9.5% EER |
| 01 (manual) | Augmentation strategy | Mutually exclusive RIR/noise + virtual speakers post-resampling | Matches WeSpeaker recipe more closely |
| 04-01 | Validation suppression mechanism | Use trainer.limit_val_batches=0 instead of callback state manipulation | Direct trainer control prevents validation execution entirely, avoiding compute waste and misleading logs |
| 04-01 | Restoration timing | Restore validation when new_sparsity >= (final_amount - 1e-4) | Ensures validation resumes exactly when target sparsity is reached |
| 04-02 | Lambda volatility metric | Measure EMA effectiveness via direction change count, not variance | When sparsity oscillates around target, EMA prevents lambda direction reversals (increase/decrease flip-flopping) |

### Roadmap Evolution

- Phase 1.1 inserted after Phase 1: Eval Auto-Config from Experiment Directory (URGENT) - Complete partial eval.py implementation for automatic config parsing
- Phase 4 added: Verify Pruning & Bregman Correctness — fix pruning validation-during-ramp bugs, verify Bregman lambda update frequency

### Pending Todos

- None

### Blockers/Concerns

- EER at 9.5% vs target 8.5% — 1% gap remaining. May need longer training (currently 20 epochs, WeSpeaker uses 50) or further hyperparameter tuning

## Session Continuity

Last session: 2026-02-06T17:40:47Z
Stopped at: Completed 04-02-PLAN.md - Bregman lambda verification tests (Phase 4 complete)
Resume file: None
