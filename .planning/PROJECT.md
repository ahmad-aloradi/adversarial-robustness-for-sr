# Compression Robustness for Speaker Recognition

## What This Is

A research project investigating how model compression techniques (magnitude-based pruning and Bregman learning) affect cross-domain generalization in speaker recognition systems. The goal is a publication-ready comparison study across multiple architectures (ECAPA-TDNN, ResNet, Transformer-based) and datasets (VoxCeleb, CN-Celeb), testing whether compressed models maintain robustness when evaluated on unseen domains.

## Core Value

Determine whether compressed speaker recognition models retain cross-domain robustness — if compression hurts generalization, the field needs to know before deploying pruned models.

## Current Milestone: v1.1 Experiments

**Goal:** Run the actual compression experiments and collect results for analysis/paper.

**Target outcomes:**
- Multi-architecture systematic runs: ECAPA-TDNN, ResNet34, Transformer at 50/70/90/95% sparsity
- Cross-domain evaluation matrix: train VoxCeleb → eval CNCeleb and vice versa
- Paper-ready results tables and figures

## Requirements

### Validated

- ✓ Speaker verification training pipeline (PyTorch Lightning + Hydra) — existing
- ✓ Multiple encoder backends (SpeechBrain, WeSpeaker, NeMo) — existing
- ✓ Multiple datasets (VoxCeleb, CN-Celeb, LibriSpeech, VPC) — existing
- ✓ Evaluation metrics (EER, minDCF, verification curves) — existing
- ✓ Cross-domain evaluation setup — existing
- ✓ Magnitude-based pruning callback (MagnitudePruner) — existing
- ✓ Bregman-based pruning callback (BregmanPruner) — existing
- ✓ Experiment tracking (WandB, Neptune, TensorBoard) — existing
- ✓ VoicePrivacy attacker challenge implementation — existing
- ✓ CNCeleb baseline matching reference (~7.8% EER) — v1.0 (achieved 6.34% EER with ResNet34)
- ✓ Pruning implementation verified against reference — v1.0
- ✓ Bregman implementation unit tests passing (AdaBreg + LinBreg) — v1.0
- ✓ Eval.py auto-config from experiment directory — v1.0
- ✓ Validation suppression during pruning ramp (magnitude + Bregman) — v1.0
- ✓ Unified sparsity computation across pruning methods — v1.0

### Active (v1.1)

- [ ] Multi-architecture support enabled: ECAPA-TDNN, ResNet, Transformer configs ready
- [ ] Systematic experiment pipeline: all combinations model × sparsity × dataset
- [ ] Fixed sparsity level configurations (50%, 70%, 90%, 95%)
- [ ] Cross-domain evaluation matrix (train on VoxCeleb, eval on CNCeleb and vice versa)
- [ ] Paper-ready results generation (tables, figures)
- [ ] Reproducible experiment commands documented

### Known Gaps Carried Forward

- [ ] **BREG-02**: Empirical Bregman GPU comparison vs BregmanLearning reference (03-03 deferred)

### Backlog (v2.0+)

- [ ] Statistical significance analysis
- [ ] Additional compression methods (quantization, etc.)

### Out of Scope

- VoicePrivacy attacker work — already complete
- New model architectures beyond existing backends — use WeSpeaker/SpeechBrain models
- Real-time inference optimization — focus is on compression-robustness tradeoff, not deployment
- Quantization or other compression methods — focus on pruning and Bregman only

## Context

**Current Codebase State:**
- PyTorch Lightning + Hydra ML research framework (~41k LOC added in v1.0)
- Entry points: `src/train.py`, `src/eval.py`
- Pruning callbacks: `src/callbacks/pruning/prune.py` (magnitude), `src/callbacks/pruning/bregman/` (Bregman)
- Experiment configs: `configs/experiment/sv/`
- Best baseline: ResNet34, 6.34% EER on CNCeleb (WeSpeaker recipe, no specaugment, SGD + WarmupExponentialLR)
- Validation suppression: `limit_val_batches=0` pattern, both magnitude and Bregman pruners
- Sparsity threshold: 1e-12 (unified across magnitude and Bregman)

**Paper Narrative:**
- Core question: Does compression hurt cross-domain generalization?
- Compare: Pruning vs Bregman across architectures and sparsity levels
- Metric focus: EER and minDCF degradation under domain shift

## Constraints

- **Compute**: HPC cluster available, but need efficient experiment batching
- **Timeline**: Conference deadline pressure — must prioritize
- **Tech Stack**: Must use existing PyTorch Lightning + Hydra patterns
- **Models**: Limited to WeSpeaker/SpeechBrain supported architectures

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Focus on pruning + Bregman only | Scope control for paper | ✓ Good |
| Fixed sparsity levels | Reproducibility, comparable to literature | ✓ Good |
| VoxCeleb + CN-Celeb cross-domain | Standard benchmarks, existing infrastructure | ✓ Good |
| Accept 8.86% EER baseline (ECAPA) | Within 1% of WeSpeaker; ResNet34 at 6.34% later | ✓ Good |
| L2-normalize after centering AND aggregation | Fixed multi-enrollment scoring | ✓ Good — contributed to EER improvement |
| Mutually exclusive RIR/noise + virtual speakers post-resampling | WeSpeaker style augmentation | ✓ Good |
| Use `limit_val_batches=0` for validation suppression | Direct trainer control vs callback state | ✓ Good — prevents spurious "New best score" |
| Fixed lambda Bregman + LambdaScheduler (dynamic) | Schedule mirrors PruningScheduler within 1e-9 | ✓ Good |
| Bregman threshold 1e-12 (aligned to magnitude) | Consistent sparsity reporting | ✓ Good |
| Multiple architectures (ECAPA, ResNet, Transformer) | Generalizability of findings | — Pending (v1.1) |

---
*Last updated: 2026-03-06 after v1.0 milestone*
