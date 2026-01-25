# Compression Robustness for Speaker Recognition

## What This Is

A research project investigating how model compression techniques (magnitude-based pruning and Bregman learning) affect cross-domain generalization in speaker recognition systems. The goal is a publication-ready comparison study across multiple architectures (ECAPA-TDNN, ResNet, Transformer-based) and datasets (VoxCeleb, CN-Celeb), testing whether compressed models maintain robustness when evaluated on unseen domains.

## Core Value

Determine whether compressed speaker recognition models retain cross-domain robustness — if compression hurts generalization, the field needs to know before deploying pruned models.

## Current Milestone: v0.1 Foundations

**Goal:** Get baselines and compression methods working correctly before running the main experiments.

**Target outcomes:**
- CNCeleb baseline matches WeSpeaker reference (~7.8% EER, currently 10.8%)
- Pruning implementation verified against reference (correct sparsity/accuracy tradeoff)
- Bregman implementation verified against reference (correct behavior)

**Approach:**
- Focus on CNCeleb baseline first — systematic investigation of training recipe differences
- Once baseline works, verify pruning and Bregman sequentially

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

### Active (v0.1)

- [ ] CNCeleb baseline matching WeSpeaker reference (~7.8% EER)
- [ ] Pruning implementation verified against reference
- [ ] Bregman implementation verified against reference

### Backlog (v1.0+)

- [ ] Multi-architecture support (ECAPA-TDNN, ResNet, Transformer models)
- [ ] Systematic experiment pipeline (all combinations of model × sparsity × dataset)
- [ ] Fixed sparsity level configurations (e.g., 50%, 70%, 90%, 95%)
- [ ] Cross-domain evaluation matrix (train on X, eval on Y)
- [ ] Paper-ready results generation (tables, figures)
- [ ] Reproducible experiment commands

### Out of Scope

- VoicePrivacy attacker work — already complete
- New model architectures beyond existing backends — use WeSpeaker/SpeechBrain models
- Real-time inference optimization — focus is on compression-robustness tradeoff, not deployment
- Quantization or other compression methods — focus on pruning and Bregman only

## Context

**Existing Codebase:**
- PyTorch Lightning + Hydra ML research framework
- Entry points: `src/train.py`, `src/eval.py`
- Pruning callbacks: `src/callbacks/pruning/prune.py` (magnitude), `src/callbacks/pruning/bregman/` (Bregman)
- Experiment configs: `configs/experiment/sv/`
- Current issues:
  - CNCeleb baseline at 10.8% EER vs WeSpeaker's 7.8% (3% gap)
  - Gap likely from training recipe differences (front-end, preprocessing, augmentation, loss scaling)
  - Pruning and Bregman implementations need verification against reference implementations

**Paper Narrative:**
- Core question: Does compression hurt cross-domain generalization?
- Compare: Pruning vs Bregman across architectures and sparsity levels
- Metric focus: EER and minDCF degradation under domain shift

**Timeline:**
- Conference/milestone deadline pressure — prioritize working experiments over perfection

## Constraints

- **Compute**: HPC cluster available, but need efficient experiment batching
- **Timeline**: Conference deadline pressure — must prioritize
- **Tech Stack**: Must use existing PyTorch Lightning + Hydra patterns
- **Models**: Limited to WeSpeaker/SpeechBrain supported architectures

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Focus on pruning + Bregman only | Scope control for paper, other compression methods future work | — Pending |
| Fixed sparsity levels | Reproducibility, comparable to literature | — Pending |
| VoxCeleb + CN-Celeb cross-domain | Standard benchmarks, existing infrastructure | — Pending |
| Multiple architectures (ECAPA, ResNet, Transformer) | Generalizability of findings | — Pending |

---
*Last updated: 2026-01-25 after v0.1 milestone initialization*
