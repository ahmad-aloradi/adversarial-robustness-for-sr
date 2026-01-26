# Phase 1: CNCeleb Baseline - Research Addendum

**Researched:** 2026-01-26
**Domain:** Advanced techniques for bridging CNCeleb baseline gap (AS-norm, multi-enrollment, features, model averaging)
**Confidence:** HIGH

## Summary

This addendum addresses four additional techniques identified by user feedback that bridge the gap between the current implementation and the CNSRC 2022 winning recipe. While the main 01-RESEARCH.md focused on hyperparameters (epochs, LR, margin timing), these techniques provide additional performance gains:

1. **AS-norm (Adaptive S-norm)**: WeSpeaker uses `top_n=300` from the CNCeleb training cohort, providing ~0.4-0.5% EER improvement
2. **Multi-enrollment embedding averaging**: The codebase already implements this correctly via mean pooling
3. **Fbank and CMVN normalization**: Current implementation matches WeSpeaker (80 mel bins, sentence-level mean normalization, no variance normalization)
4. **Checkpoint averaging**: WeSpeaker averages 10 checkpoints for final model; codebase needs this added

**Primary recommendation:** The codebase is already correctly configured for AS-norm, multi-enrollment, and feature normalization. The main gap is checkpoint averaging, which should be added as a post-training step.

---

## 1. AS-norm (Adaptive Symmetric Normalization)

### What WeSpeaker Does

WeSpeaker applies AS-norm during evaluation with these parameters (from `run.sh`):

```bash
# From WeSpeaker examples/cnceleb/v2/run.sh
top_n=300          # Number of top cohort scores for normalization
cohort=cnceleb_train  # Cohort embeddings from training set
```

**AS-norm Algorithm (from WeSpeaker `score_norm.py`):**

```python
# 1. Normalize embeddings to unit vectors
# 2. Compute enrollment vs cohort similarity scores
enroll_cohort_scores = np.matmul(enroll_emb, cohort.T)
# 3. Select top-n scores for statistics
enroll_cohort_top = enroll_cohort_scores[:, :top_n]  # After sorting
enroll_mean = np.mean(enroll_cohort_top, axis=1)
enroll_std = np.std(enroll_cohort_top, axis=1)

# 4. Same for test embedding
test_cohort_scores = np.matmul(test_emb, cohort.T)
test_cohort_top = test_cohort_scores[:, :top_n]
test_mean = np.mean(test_cohort_top, axis=1)
test_std = np.std(test_cohort_top, axis=1)

# 5. Symmetric normalization
normalized_score = 0.5 * (
    (raw_score - enroll_mean) / enroll_std +
    (raw_score - test_mean) / test_std
)
```

### What the Codebase Currently Does

The codebase has AS-norm implemented in `src/modules/metrics/metrics.py`:

```python
def AS_norm(score: float,
            enroll_embedding: torch.Tensor,
            test_embedding: torch.Tensor,
            cohort_embeddings: torch.Tensor,
            topk: int = 3000,          # DEFAULT: 3000 (different from WeSpeaker!)
            min_cohort_size: int = 300):
```

**Current config in `sv.yaml`:**
```yaml
scores_norm:
  embeds_metric_params:
    num_speakers_in_cohort: 600
    min_utts_per_speaker: 10
  scores_norm_params:
    topk: 600           # Currently 600
    min_cohort_size: 1
```

**Current config in `sv_wespeaker.yaml`:**
```yaml
module:
  normalize_test_scores: True
  subtract_mean: True
```

### Gap Analysis

| Parameter | WeSpeaker | Current Codebase | Gap |
|-----------|-----------|------------------|-----|
| AS-norm enabled | Yes | Yes | None |
| top_n/topk | 300 | 600 | Minor (600 is more conservative) |
| Cohort source | cnceleb_train | Training set | None |
| Mean subtraction | Yes | Yes | None |

**Verdict:** AS-norm is already correctly implemented. The `topk=600` is actually more conservative than WeSpeaker's 300 and should work well. No changes needed.

### Expected EER Impact

- **Without AS-norm**: 7.879% EER (WeSpeaker baseline)
- **With AS-norm**: 7.412% EER (WeSpeaker)
- **Improvement**: ~0.47% absolute EER reduction

**Confidence:** HIGH - Verified against WeSpeaker source code

---

## 2. Multi-Enrollment: Embedding Averaging vs Utterance Concatenation

### What WeSpeaker Does

WeSpeaker uses **embedding averaging** for multi-enrollment:

> "Embeddings for all enrollment sessions for each speaker are extracted and **averaged** to obtain the final enrollment embedding, which brings considerable performance improvement in our experiments."

The process:
1. Extract individual embeddings for each enrollment utterance
2. Average (mean pool) all embeddings for the same enrollment ID
3. Use averaged embedding for scoring

**NOT concatenation** - WeSpeaker does not concatenate utterances before feature extraction.

### What the Codebase Currently Does

The codebase correctly implements embedding averaging in `sv.py`:

```python
# From sv.py _compute_embeddings()
if mode == 'enrollment' and hasattr(batch, 'utt_counts') and batch.utt_counts is not None:
    # Aggregate embeddings per enroll_id using mean pooling
    idx = 0
    for enroll_id, count in zip(batch.enroll_id, batch.utt_counts):
        utt_embeds = embed[idx:idx + count]  # [count, embed_dim]
        aggregated_embed = utt_embeds.mean(dim=0)  # [embed_dim]  <-- AVERAGING
        embeddings_dict[enroll_id] = aggregated_embed
        idx += count
```

The multi-enrollment infrastructure in `cnceleb_dataset.py`:
- `CNCelebEnrollMulti`: Loads all utterances per enrollment ID
- `EnrollCollateMulti`: Flattens utterances and tracks `utt_counts`
- `CNCelebDataModule`: Supports `enrollment_mode: 'single'`, `'multi'`, or `'both'`

### Gap Analysis

| Aspect | WeSpeaker | Current Codebase | Gap |
|--------|-----------|------------------|-----|
| Method | Embedding averaging | Embedding averaging | None |
| Multi-utt support | Yes | Yes (via `CNCelebEnrollMulti`) | None |
| Config option | N/A | `enrollment_mode: single/multi/both` | None |

**Verdict:** The codebase correctly implements embedding averaging for multi-enrollment. The default is `enrollment_mode: single` for faster iteration, but switching to `multi` for final evaluation matches WeSpeaker.

### Expected EER Impact

Multi-enrollment averaging typically provides 0.1-0.3% EER improvement over single-enrollment.

**Confidence:** HIGH - Code verified to match WeSpeaker approach

---

## 3. Fbank and Feature Normalization

### What WeSpeaker Does

**Fbank Extraction:**
```python
# From WeSpeaker processor.py
def compute_fbank(data, num_mel_bins=80, frame_length=25,
                  frame_shift=10, dither=1.0):
    # Uses Kaldi-style fbank with:
    # - 80 mel bins
    # - 25ms frame length
    # - 10ms frame shift
    # - Hamming window
    # - No energy feature
```

**CMVN Normalization:**
```python
# From WeSpeaker processor.py
def apply_cmvn(data, norm_mean=True, norm_var=False):
    """Apply CMVN per utterance"""
    mat = mat - torch.mean(mat, dim=0)  # Mean normalization only
    # norm_var=False by default - no variance normalization
```

**Key: Sentence-level mean normalization, NO variance normalization.**

### What the Codebase Currently Does

**Fbank config (`wespeaker_ecapa_tdnn.yaml`):**
```yaml
audio_processor:
  _target_: speechbrain.lobes.features.Fbank
  sample_rate: 16000
  n_mels: 80           # Matches WeSpeaker
  n_fft: 512
  win_length: 25       # 25ms - matches WeSpeaker
  hop_length: 10       # 10ms - matches WeSpeaker
  deltas: False
  f_min: 0
  f_max: null
```

**CMVN config:**
```yaml
audio_processor_normalizer:
  _target_: speechbrain.processing.features.InputNormalization
  norm_type: sentence   # Per-utterance normalization
  std_norm: False       # NO variance normalization - matches WeSpeaker!
```

### Gap Analysis

| Parameter | WeSpeaker | Current Codebase | Gap |
|-----------|-----------|------------------|-----|
| Mel bins | 80 | 80 | None |
| Frame length | 25ms | 25ms | None |
| Frame shift | 10ms | 10ms | None |
| Mean normalization | Yes (sentence) | Yes (sentence) | None |
| Variance normalization | No | No (`std_norm: False`) | None |
| Dithering | 1.0 | Handled by SpeechBrain | Minor |

**Verdict:** Feature extraction and normalization are correctly configured to match WeSpeaker. The `std_norm: False` setting is critical and correctly set.

**Confidence:** HIGH - Config verified against WeSpeaker source

---

## 4. Checkpoint Averaging

### What WeSpeaker Does

WeSpeaker averages the **last N checkpoints** to create the final model:

```bash
# From WeSpeaker run.sh
num_avg=10  # Average last 10 checkpoints
```

**Algorithm (`average_model.py`):**
```python
# 1. Find checkpoint files, excluding avg/final/convert
path_list = glob.glob('{}/[!avg][!final][!convert]*.pt'.format(src_path))

# 2. Sort by epoch number (descending) and take last N
path_list = path_list[-num_avg:]

# 3. Average parameters
for path in path_list:
    states = torch.load(path)
    for k in avg.keys():
        avg[k] += states[k]

# 4. Divide by count
for k in avg.keys():
    avg[k] = torch.true_divide(avg[k], num_avg)

# 5. Save as avg_model.pt
torch.save(avg, dst_model)
```

### What the Codebase Currently Does

**No checkpoint averaging is implemented.** The codebase uses the best checkpoint from `ModelCheckpoint` callback directly.

### Gap Analysis

| Aspect | WeSpeaker | Current Codebase | Gap |
|--------|-----------|------------------|-----|
| Checkpoint averaging | Yes (N=10) | No | **MISSING** |
| Selection method | Last N epochs | Best by metric | Different |
| Output format | `avg_model.pt` | `best.ckpt` | Different |

**Verdict:** This is a missing feature that needs implementation.

### Implementation Approach

**Option A: Post-training script (Recommended)**

Create `scripts/average_checkpoints.py`:
```python
import torch
import glob
from pathlib import Path
import argparse

def average_checkpoints(src_dir: str, output_path: str, num_avg: int = 10):
    """Average the last N checkpoints."""
    # Find checkpoints
    ckpt_files = sorted(
        glob.glob(f"{src_dir}/epoch=*.ckpt"),
        key=lambda x: int(x.split('epoch=')[1].split('-')[0])
    )

    # Take last N
    ckpt_files = ckpt_files[-num_avg:]
    print(f"Averaging {len(ckpt_files)} checkpoints")

    # Load and average
    avg_state = None
    for path in ckpt_files:
        state = torch.load(path, map_location='cpu')['state_dict']
        if avg_state is None:
            avg_state = {k: v.clone() for k, v in state.items()}
        else:
            for k in avg_state:
                avg_state[k] += state[k]

    # Divide
    for k in avg_state:
        avg_state[k] = avg_state[k] / len(ckpt_files)

    # Save
    torch.save({'state_dict': avg_state}, output_path)
    print(f"Saved averaged model to {output_path}")
```

**Option B: Lightning Callback**

Add a callback that saves checkpoints for averaging and produces averaged model at training end.

**Recommendation:** Option A (post-training script) is simpler and matches WeSpeaker's approach.

### Expected EER Impact

Checkpoint averaging typically provides 0.1-0.3% EER improvement by smoothing over training noise.

**Confidence:** HIGH - Algorithm verified from WeSpeaker source

---

## Summary of Gaps

| Feature | Status | Action Required | Priority |
|---------|--------|-----------------|----------|
| AS-norm | Implemented | Verify `topk` matches (currently 600, WeSpeaker uses 300) | Low |
| Multi-enrollment averaging | Implemented | Use `enrollment_mode: multi` for final eval | Low |
| Fbank extraction | Correct | None | None |
| CMVN normalization | Correct (`std_norm: False`) | None | None |
| Mean subtraction | Implemented | Already enabled in sv_wespeaker.yaml | None |
| Checkpoint averaging | **Missing** | Implement post-training script | **High** |

## Recommendations

### Immediate (for 01-01-PLAN.md)

1. **Checkpoint averaging**: Add a post-training step to average last N checkpoints
   - Create `scripts/average_checkpoints.py`
   - Document usage in training workflow
   - Default to N=10 (WeSpeaker default)

2. **Verify AS-norm topk**: Consider changing from 600 to 300 to match WeSpeaker exactly, though 600 should work fine

### For Final Evaluation

1. Use `enrollment_mode: multi` in cnceleb.yaml for proper CNCeleb protocol
2. Use averaged checkpoint instead of best checkpoint
3. Ensure `normalize_test_scores: True` and `subtract_mean: True`

## Code Examples

### Checkpoint Averaging Script

```python
#!/usr/bin/env python
"""Average model checkpoints for improved speaker verification performance.

Usage:
    python scripts/average_checkpoints.py \
        --src_dir logs/train/runs/2026-01-26_12-00-00/checkpoints \
        --output_path logs/train/runs/2026-01-26_12-00-00/avg_model.ckpt \
        --num_avg 10
"""
import torch
import glob
import argparse
from pathlib import Path


def average_checkpoints(src_dir: str, output_path: str, num_avg: int = 10):
    """Average the last N PyTorch Lightning checkpoints.

    Args:
        src_dir: Directory containing checkpoint files
        output_path: Path to save averaged checkpoint
        num_avg: Number of checkpoints to average (default: 10)
    """
    # Find checkpoint files (Lightning format: epoch=X-step=Y.ckpt)
    ckpt_files = list(Path(src_dir).glob("epoch=*.ckpt"))

    # Sort by epoch number
    def get_epoch(p):
        return int(p.stem.split('epoch=')[1].split('-')[0])

    ckpt_files = sorted(ckpt_files, key=get_epoch)

    # Take last N
    ckpt_files = ckpt_files[-num_avg:]

    if len(ckpt_files) < num_avg:
        print(f"Warning: Only found {len(ckpt_files)} checkpoints, averaging all")

    print(f"Averaging {len(ckpt_files)} checkpoints:")
    for f in ckpt_files:
        print(f"  - {f.name}")

    # Load and accumulate state dicts
    avg_state = None
    for path in ckpt_files:
        checkpoint = torch.load(path, map_location='cpu')
        state = checkpoint['state_dict']

        if avg_state is None:
            avg_state = {k: v.float().clone() for k, v in state.items()}
        else:
            for k in avg_state:
                avg_state[k] += state[k].float()

    # Divide by count
    for k in avg_state:
        avg_state[k] = avg_state[k] / len(ckpt_files)

    # Save in Lightning format
    output_checkpoint = {
        'state_dict': avg_state,
        'averaged_from': [str(f) for f in ckpt_files],
        'num_averaged': len(ckpt_files),
    }

    torch.save(output_checkpoint, output_path)
    print(f"Saved averaged model to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", required=True, help="Checkpoint directory")
    parser.add_argument("--output_path", required=True, help="Output path")
    parser.add_argument("--num_avg", type=int, default=10, help="Number to average")
    args = parser.parse_args()

    average_checkpoints(args.src_dir, args.output_path, args.num_avg)
```

### Evaluation with Averaged Model

```bash
# Training (saves multiple checkpoints)
python src/train.py experiment=sv/sv_wespeaker datamodule=datasets/cnceleb

# Post-training: average checkpoints
python scripts/average_checkpoints.py \
    --src_dir logs/train/runs/LATEST/checkpoints \
    --output_path logs/train/runs/LATEST/avg_model.ckpt \
    --num_avg 10

# Evaluation with averaged model
python src/eval.py \
    ckpt_path=logs/train/runs/LATEST/avg_model.ckpt \
    datamodule=datasets/cnceleb \
    datamodule.dataset.enrollment_mode=multi
```

---

## Sources

### Primary (HIGH confidence)
- [WeSpeaker score_norm.py](https://raw.githubusercontent.com/wenet-e2e/wespeaker/master/wespeaker/bin/score_norm.py) - AS-norm implementation
- [WeSpeaker average_model.py](https://raw.githubusercontent.com/wenet-e2e/wespeaker/master/wespeaker/bin/average_model.py) - Checkpoint averaging
- [WeSpeaker processor.py](https://raw.githubusercontent.com/wenet-e2e/wespeaker/master/wespeaker/dataset/processor.py) - Fbank and CMVN
- [WeSpeaker cnceleb run.sh](https://raw.githubusercontent.com/wenet-e2e/wespeaker/master/examples/cnceleb/v2/run.sh) - Configuration values
- Codebase: `src/modules/sv.py`, `src/modules/metrics/metrics.py`

### Secondary (MEDIUM confidence)
- [WeSpeaker CNCeleb README](https://raw.githubusercontent.com/wenet-e2e/wespeaker/master/examples/cnceleb/v2/README.md) - Results table
- [WeSpeaker Paper (arXiv)](https://arxiv.org/abs/2210.17016) - Multi-enrollment methodology

---

## Metadata

**Confidence breakdown:**
- AS-norm parameters: HIGH - Verified against WeSpeaker source code
- Multi-enrollment: HIGH - Code verified to implement embedding averaging correctly
- Feature normalization: HIGH - Config verified against WeSpeaker source
- Checkpoint averaging: HIGH - Algorithm verified from WeSpeaker source

**Research date:** 2026-01-26
**Valid until:** Indefinite (WeSpeaker recipe is stable)
