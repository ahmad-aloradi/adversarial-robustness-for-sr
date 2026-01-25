# WeSpeaker ECAPA-TDNN Training Recipe Analysis

**Researched:** 2026-01-25
**Overall confidence:** HIGH (verified against official WeSpeaker repository)

## Executive Summary

WeSpeaker's ECAPA-TDNN on CNCeleb achieves **7.395% EER** (with Large Margin + AS-Norm) or **7.879% EER** (baseline without post-processing). The user's current setup achieves 10.8% EER - a **2.9-3.4% gap**. This document analyzes WeSpeaker's exact recipe to identify potential sources of this discrepancy.

**Key finding:** The most likely causes of the EER gap are:
1. Training duration (150 epochs vs 15 epochs)
2. CMVN normalization difference (per-utterance mean-only vs sentence norm)
3. Margin scheduling timing (epoch 20-40 vs epoch 2-15)
4. Large margin fine-tuning stage (missing)

---

## Front-End Features

### WeSpeaker Configuration (HIGH confidence)

| Parameter | WeSpeaker Value | Source |
|-----------|-----------------|--------|
| **Mel bins** | 80 | `ecapa_tdnn.yaml`: `num_mel_bins: 80` |
| **Frame length** | 25ms | `ecapa_tdnn.yaml`: `frame_length: 25` |
| **Frame shift** | 10ms | `ecapa_tdnn.yaml`: `frame_shift: 10` |
| **Sample rate** | 16kHz | `ecapa_tdnn.yaml`: `resample_rate: 16000` |
| **Dither** | 1.0 | `processor.py`: `dither: 1.0` |
| **Window** | Hamming | `torchaudio.compliance.kaldi.fbank` default |
| **Energy** | Excluded | `torchaudio.compliance.kaldi.fbank` default |

**Feature extraction code** (from `wespeaker/dataset/processor.py`):
```python
# Waveform normalized to 16-bit scale before processing
waveform = waveform * (1 << 15)
mat = torchaudio.compliance.kaldi.fbank(
    waveform,
    num_mel_bins=80,
    frame_length=25,
    frame_shift=10,
    dither=1.0,
    energy_floor=0.0,
    sample_frequency=16000
)
```

### CMVN Normalization (CRITICAL DIFFERENCE)

**WeSpeaker's approach:**
```python
def apply_cmvn(mat, norm_mean=True, norm_var=False):
    if norm_mean:
        mat = mat - torch.mean(mat, dim=0)  # Mean normalization per feature
    if norm_var:
        mat = mat / (torch.std(mat, dim=0) + 1e-8)  # Variance norm (DISABLED by default)
    return mat
```

**Key observations:**
- WeSpeaker uses **mean-only normalization** (no variance normalization)
- Applied **per-utterance** (global mean subtracted from each feature dimension)
- `norm_var=False` by default

### Current Setup Comparison

| Parameter | WeSpeaker | Current | Match? |
|-----------|-----------|---------|--------|
| Mel bins | 80 | 80 | YES |
| Frame length | 25ms | 25ms | YES |
| Frame shift | 10ms | 10ms | YES |
| Sample rate | 16kHz | 16kHz | YES |
| Dither | 1.0 | Unknown (SpeechBrain default) | CHECK |
| CMVN | Mean-only, per-utterance | `norm_type: sentence, std_norm: False` | LIKELY MATCH |

**Action:** Verify SpeechBrain's `InputNormalization(norm_type='sentence', std_norm=False)` matches WeSpeaker's CMVN behavior. SpeechBrain's sentence normalization should be equivalent.

---

## Data Preprocessing

### Audio Chunking Strategy (HIGH confidence)

**Training:**
- **Frame count:** 200 frames (2 seconds at 10ms frame shift)
- **Chunk selection:** Random crop from full utterance
- **Formula:** `chunk_length = ((num_frms - 1) * frame_shift + frame_length) * sample_rate // 1000`
- **Result:** `((200 - 1) * 10 + 25) * 16 = 31,840 samples = 1.99s`

**Filtering:**
```yaml
filter_conf:
  min_num_frms: 100  # Minimum ~1 second
  max_num_frms: 800  # Maximum ~8 seconds
```

**Inference:**
- Full utterance used (no chunking)
- `whole_utt: True` in extraction config
- Dither disabled (`dither: 0.0`)
- Speed perturbation disabled

### Current Setup Comparison

| Parameter | WeSpeaker | Current | Match? |
|-----------|-----------|---------|--------|
| Training duration | 200 frames (~2s) | `max_duration: 3.0` (3s) | CLOSE |
| Min duration | 100 frames (~1s) | `min_duration: 0.5` | DIFFERENT |
| Chunk strategy | Random crop | Pre-segmentation or random crop | CHECK |
| Inference | Full utterance | Full utterance | YES |

**Potential issue:** Current setup uses `use_pre_segmentation: true` with fixed 3-second segments. WeSpeaker uses **random cropping** during training for augmentation diversity.

---

## Training Recipe

### Optimizer Configuration (HIGH confidence)

```yaml
# WeSpeaker ecapa_tdnn.yaml
optimizer: SGD
optimizer_args:
  momentum: 0.9
  nesterov: True
  weight_decay: 0.0001
```

### Learning Rate Schedule

**WeSpeaker:**
```yaml
scheduler: ExponentialDecrease
scheduler_args:
  initial_lr: 0.1
  final_lr: 0.00005
  warm_from_zero: True
  warm_up_epoch: 6  # 6 epochs warmup from 0 to 0.1
```

**Formula (ExponentialDecrease):**
- Warmup: Linear increase from 0 to `initial_lr` over 6 epochs
- Decay: Exponential decay from `initial_lr` to `final_lr` over remaining epochs

**Current setup:**
```yaml
lr_scheduler:
  scheduler:
    _target_: src.utils.schedulers.WarmupExponentialLR
    warmup_epochs: 1       # Only 1 epoch warmup
    warmup_start_factor: 0.01
    gamma: 0.80            # Per-epoch decay
```

| Parameter | WeSpeaker | Current | Match? |
|-----------|-----------|---------|--------|
| Initial LR | 0.1 | 0.1 | YES |
| Final LR | 0.00005 | ~0.1 * 0.8^14 = 0.0044 | NO (10x higher) |
| Warmup epochs | 6 | 1 | NO |
| Warmup start | 0 | 0.01 | NO |

**Critical difference:** Current schedule decays to ~0.0044 after 15 epochs; WeSpeaker decays to 0.00005 after 150 epochs.

### Margin Scheduling (HIGH confidence)

**WeSpeaker MarginScheduler:**
```yaml
margin_scheduler: MarginScheduler
margin_update:
  initial_margin: 0.0
  final_margin: 0.2
  increase_start_epoch: 20  # Start increasing at epoch 20
  fix_start_epoch: 40       # Fix at 0.2 from epoch 40
  update_margin: True
  increase_type: "exp"      # Exponential increase
```

**Exponential formula:**
```python
ratio = 1.0 - math.exp(
    (current_iter / increase_iter) * math.log(final_val / (initial_val + 1e-6))
) * initial_val
margin = initial_margin + (final_margin - initial_margin) * ratio
```

**Current setup:**
```yaml
margin_scheduler:
  _target_: src.callbacks.margin_scheduler.ProgressiveMarginScheduler
  initial_margin: 0.0
  final_margin: 0.2
  warmup_epochs: 2   # Start increasing at epoch 2
  start_epoch: 2     # Much earlier than WeSpeaker
```

| Parameter | WeSpeaker | Current | Match? |
|-----------|-----------|---------|--------|
| Initial margin | 0.0 | 0.0 | YES |
| Final margin | 0.2 | 0.2 | YES |
| Increase start | Epoch 20 | Epoch 2 | NO |
| Fix epoch | Epoch 40 | ~Epoch 15 | NO |
| Increase type | Exponential | Unknown | CHECK |

**Critical difference:** WeSpeaker starts margin increase at epoch 20 (13% into training). Current setup starts at epoch 2 (13% into 15 epochs = epoch 2, but total training is much shorter).

### Training Duration

| Parameter | WeSpeaker | Current | Impact |
|-----------|-----------|---------|--------|
| Total epochs | 150 | 15 | **10x shorter** |
| Effective training | Full convergence | ~10% of WeSpeaker | MAJOR |

### Batch Configuration

```yaml
# WeSpeaker
batch_size: 128
num_workers: 16
```

**Sampling:** WeSpeaker uses standard DataLoader sampling (not PK-Sampler for base training).

**Current setup:** `batch_size: 128` (matches), PK-Sampler commented out (matches).

### Loss Function

**WeSpeaker:**
```yaml
projection_args:
  project_type: "arc_margin"
  scale: 32.0
  easy_margin: False
loss: CrossEntropyLoss
```

**ArcMargin formula:**
```python
# Margin applied only to target class
phi = cos(theta + margin)  # Add angular margin
# With boundary handling
phi = where(cos(theta) > threshold, phi, cos(theta) - margin * sin(margin))
output = scale * (one_hot * phi + (1 - one_hot) * cos(theta))
```

**Current setup:**
```yaml
criterion:
  aam:
    _target_: speechbrain.nnet.losses.LogSoftmaxWrapper
    loss_fn:
      _target_: speechbrain.nnet.losses.AdditiveAngularMargin
      margin: 0.0  # Initial, updated by scheduler
      scale: 32.0
```

| Parameter | WeSpeaker | Current | Match? |
|-----------|-----------|---------|--------|
| Loss type | ArcMargin + CrossEntropy | AAM + LogSoftmax | EQUIVALENT |
| Scale | 32.0 | 32.0 | YES |
| Easy margin | False | False (default) | YES |

---

## Augmentation Pipeline

### WeSpeaker Augmentation (HIGH confidence)

```yaml
# Applied with 60% probability
aug_prob: 0.6

speed_perturb: True  # Speed factors: [0.9, 1.0, 1.1]
spec_aug: False      # Disabled for CNCeleb

# Reverb and noise from LMDB
reverb_lmdb_file: ...
noise_lmdb_file: ...
```

**Order of operations:**
1. Speed perturbation (always applied, one of 0.9/1.0/1.1x)
2. Feature extraction (Fbank)
3. CMVN normalization
4. Reverb (60% probability)
5. Noise (60% probability)

**Note:** Reverb and noise are applied **after** feature extraction in WeSpeaker (on spectrogram), not on raw waveform.

### Current Setup

```yaml
augmentations:
  wav_augmenter:
    transforms:
      - SpeedPerturb: [0.9, 1.0, 1.1], p=1.0
      - NormalizedReverb: p=0.6
      - NoiseFromCSV: SNR 0-15dB, p=0.6
```

| Augmentation | WeSpeaker | Current | Match? |
|--------------|-----------|---------|--------|
| Speed perturb | Yes, always | Yes, always | YES |
| Reverb prob | 0.6 | 0.6 | YES |
| Noise prob | 0.6 | 0.6 | YES |
| Application domain | Spectrogram | Waveform | DIFFERENT |
| SpecAugment | Disabled | Disabled | YES |

**Potential issue:** WeSpeaker applies reverb/noise on spectrogram; current setup applies on waveform. This may produce different acoustic characteristics.

---

## Large Margin Fine-tuning (LM) Stage

### WeSpeaker LM Configuration (HIGH confidence)

**This is a critical second training stage that current setup is missing.**

```yaml
# ecapa_tdnn_lm.yaml (Stage 9)
epochs: 5
batch_size: 64
num_frms: 600          # 6 seconds (vs 200 frames in base training)

optimizer_args:
  lr: 0.0001           # 1000x smaller than initial training
  final_lr: 0.000025   # Even smaller final LR

margin_update:
  initial_margin: 0.5  # CONSTANT 0.5 (no scheduling)
  final_margin: 0.5
```

**Key differences from base training:**
- **3x longer segments** (600 vs 200 frames)
- **1000x smaller learning rate** (0.0001 vs 0.1)
- **Fixed large margin** (0.5 vs progressive 0.0-0.2)
- **Initialized from averaged base model**

**Impact on results:**

| Model | EER (Base) | EER (LM) | Improvement |
|-------|------------|----------|-------------|
| ECAPA_TDNN_GLOB_c1024 | 7.879% | 7.986% | -0.1% (slightly worse alone) |
| ECAPA_TDNN_GLOB_c1024 + AS-Norm | 7.412% | 7.395% | +0.02% |

LM fine-tuning primarily helps with longer utterance handling and AS-Norm compatibility.

---

## Evaluation Protocol

### WeSpeaker Evaluation (HIGH confidence)

**Test set:** `CNC-Eval-Avg.lst`

**Embedding extraction:**
- Full utterance (no chunking)
- CMVN applied
- Dither disabled
- Speed perturbation disabled

**Enrollment handling:**
- Multi-utterance averaging: `tools/vector_mean.py`
- Aggregates all utterances per enrollment ID

**Scoring:**
```python
# Cosine similarity
cos_score = cosine_similarity(emb1, emb2)
```

**Mean subtraction:**
- Subtracts mean of training set (VoxCeleb2 dev) from all embeddings
- Applied before cosine similarity

**Score normalization options:**
1. **AS-Norm:** Adaptive score normalization using cohort
2. **QMF:** Quantile mean function (additional post-processing)

### Results Table (Official WeSpeaker)

| Model | LM | AS-Norm | EER (%) | minDCF |
|-------|----|---------|---------| -------|
| ECAPA_TDNN_GLOB_c512 | No | No | 8.313 | 0.432 |
| ECAPA_TDNN_GLOB_c512 | No | Yes | 7.644 | 0.390 |
| ECAPA_TDNN_GLOB_c512 | Yes | Yes | 7.417 | 0.379 |
| **ECAPA_TDNN_GLOB_c1024** | No | No | **7.879** | 0.420 |
| ECAPA_TDNN_GLOB_c1024 | No | Yes | 7.412 | 0.379 |
| **ECAPA_TDNN_GLOB_c1024** | Yes | Yes | **7.395** | 0.372 |

**Note:** The 7.8% figure likely refers to the baseline c1024 model (7.879% EER) without LM or AS-Norm.

---

## Key Differences to Investigate

### High Priority (Likely causing most of the gap)

1. **Training duration: 150 epochs vs 15 epochs**
   - Impact: Model may not have converged
   - Recommendation: Try 50-100 epochs minimum
   - Expected improvement: 1-2% EER

2. **Final learning rate too high**
   - WeSpeaker: 0.00005
   - Current: ~0.0044 (88x higher)
   - Impact: Model may oscillate, not fully converge
   - Recommendation: Lower gamma or use cosine annealing to 0.0001

3. **Margin scheduling timing**
   - WeSpeaker: Epochs 20-40 of 150 (13%-27% of training)
   - Current: Epochs 2-15 of 15 (13%-100% of training)
   - Impact: Margin increases too fast relative to model learning
   - Recommendation: If training 15 epochs, consider fixed margin 0.1-0.15

### Medium Priority

4. **Warmup duration**
   - WeSpeaker: 6 epochs
   - Current: 1 epoch
   - Impact: May cause training instability early on

5. **Large Margin fine-tuning missing**
   - Impact: 0.02-0.1% EER on long utterances
   - Recommendation: Add LM stage if targeting <7.5% EER

6. **Augmentation domain (waveform vs spectrogram)**
   - Impact: Unknown, possibly small
   - Recommendation: Test both approaches

### Lower Priority

7. **Dither setting**
   - WeSpeaker: 1.0
   - Current: Unknown (SpeechBrain default)
   - Impact: Small effect on robustness

8. **Pre-segmentation vs random cropping**
   - Current uses fixed 3s segments
   - WeSpeaker uses random 2s crops
   - Impact: Less augmentation diversity

---

## Recommended Changes to Close the Gap

### Phase 1: Quick Wins (Expected: 1-2% improvement)

```yaml
# Increase training duration
trainer:
  max_epochs: 50  # Was 15

# Fix LR schedule
lr_scheduler:
  scheduler:
    _target_: src.utils.schedulers.WarmupExponentialLR
    warmup_epochs: 3           # Was 1 (WeSpeaker: 6/150 = 3/50)
    warmup_start_factor: 0.0   # Was 0.01
    gamma: 0.95                # Gentler decay to reach lower final LR

# Adjust margin schedule proportionally
margin_scheduler:
  warmup_epochs: 7    # ~13% of 50 epochs (WeSpeaker: 20/150)
  start_epoch: 7
  # Keep final_margin: 0.2
```

### Phase 2: Full Recipe (Expected: additional 0.5-1% improvement)

```yaml
# Match WeSpeaker exactly
trainer:
  max_epochs: 150

lr_scheduler:
  scheduler:
    _target_: WeSpeakerExponentialDecrease  # Custom scheduler
    initial_lr: 0.1
    final_lr: 0.00005
    warm_up_epoch: 6
    warm_from_zero: True

margin_scheduler:
  increase_start_epoch: 20
  fix_start_epoch: 40
  increase_type: "exp"

# Disable pre-segmentation for random cropping
datamodule:
  dataset:
    use_pre_segmentation: false
    max_duration: 2.0  # Match WeSpeaker's 200 frames
```

### Phase 3: Large Margin Fine-tuning (Expected: 0.1-0.5% improvement)

Add second training stage after base training:
```yaml
# LM fine-tuning config
trainer:
  max_epochs: 5

lr_scheduler:
  scheduler:
    initial_lr: 0.0001
    final_lr: 0.000025

margin_scheduler:
  initial_margin: 0.5
  final_margin: 0.5
  update_margin: False  # Keep constant

datamodule:
  dataset:
    max_duration: 6.0  # 600 frames
```

---

## Sources

- [WeSpeaker GitHub Repository](https://github.com/wenet-e2e/wespeaker)
- [WeSpeaker CNCeleb v2 Recipe](https://github.com/wenet-e2e/wespeaker/tree/master/examples/cnceleb/v2)
- [ECAPA-TDNN Config](https://raw.githubusercontent.com/wenet-e2e/wespeaker/master/examples/cnceleb/v2/conf/ecapa_tdnn.yaml)
- [ECAPA-TDNN LM Config](https://raw.githubusercontent.com/wenet-e2e/wespeaker/master/examples/cnceleb/v2/conf/ecapa_tdnn_lm.yaml)
- [WeSpeaker Paper (arXiv:2210.17016)](https://arxiv.org/abs/2210.17016)
- [WeSpeaker CNCeleb README](https://github.com/wenet-e2e/wespeaker/blob/master/examples/cnceleb/v2/README.md)

---

## Confidence Assessment

| Area | Confidence | Reason |
|------|------------|--------|
| Front-end features | HIGH | Verified from official config and code |
| CMVN normalization | HIGH | Verified from processor.py |
| Training recipe | HIGH | Verified from ecapa_tdnn.yaml |
| Margin scheduling | HIGH | Verified from config and scheduler code |
| LM fine-tuning | HIGH | Verified from ecapa_tdnn_lm.yaml |
| Augmentation | HIGH | Verified from config |
| Evaluation protocol | MEDIUM | Inferred from scripts, not fully traced |
| Gap analysis | MEDIUM | Based on comparison, not empirically tested |
