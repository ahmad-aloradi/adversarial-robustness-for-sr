# Neural Network Quantization

A Brevitas-backed quantization stack for speaker recognition: post-training quantization (PTQ), quantization-aware training (QAT), and FP16 inference cast. Mirrors the pruning compression stack ([`docs/pruning.md`](./pruning.md)) and integrates with the same `train.py` / `eval.py` entry points.

---

## Overview

| Spec name    | Weights | Activations | Path used                                       |
|--------------|---------|-------------|-------------------------------------------------|
| `none`       | FP32    | FP32        | sentinel; eval/PTQ short-circuits               |
| `fp16`       | FP16    | FP16        | `.half()` cast — no layer substitution          |
| `int8_w`     | INT8    | FP32        | Brevitas `QuantConv*` / `QuantLinear` substitution |
| `int8_wa`    | INT8    | INT8        | substitution + activation calibration           |
| `int4_w`     | INT4    | FP32        | substitution                                    |
| `int4_wa`    | INT4    | INT4        | substitution + calibration                      |
| `int2_w`     | INT2    | FP32        | substitution (first/last layer protection recommended) |
| `int2_wa`    | INT2    | INT2        | substitution + calibration                      |
| `int1_w`     | INT1    | FP32        | substitution (first/last layer protection strongly recommended) |
| `int1_wa`    | INT1    | INT1        | substitution + calibration                      |

**What is substituted:** the encoder's `Conv1d` / `Conv2d` / `Linear` leaves (`model.audio_encoder.encoder`), optionally the classifier head (`quantize_classifier=true`).

**What stays FP32 by design:** the Fbank front-end (`model.audio_encoder.audio_processor`), the input normalizer (`audio_processor_normalizer`), and any layer matching a `skip_patterns` regex.

---

## Quickstart

**PTQ** — quantize a finished FP32 training run:
```bash
python scripts/quantize_ptq.py \
    exp_dir=/path/to/fp32/run \
    quantization=int8_wa \
    quantization.calibration.num_batches=32
```

**QAT** — train a quantized model from scratch (or from an FP32 seed):
```bash
python src/train.py \
    experiment=quant/sv_qat_ecapa_int8_wa \
    callbacks.qat.pretrained_ckpt_path=/path/to/fp32/best.ckpt
```

**Visualize** a bit-width sweep:
```bash
python scripts/visualize_quantization.py \
    --input_dir /path/to/runs/ \
    --output_dir /path/to/runs/plots
```
One PDF is written per ``(test_set, metric_kind)`` pair; the metric kinds
(``EER``, ``minDCF``) are discovered from the JSON payloads, not selected.

---

## PTQ vs QAT vs FP16 — when to use each

| Method | Use when                                                            | Cost                        |
|--------|---------------------------------------------------------------------|-----------------------------|
| **FP16** | You just want a free 2× inference speedup, no accuracy budget pain. Skip if you need INT acceleration. | Zero — single `.half()` cast |
| **PTQ**  | You have a finished FP32 ckpt and want INT8 / INT4 in minutes. Works well down to INT8; INT4 starts to lose accuracy. | One calibration pass (~32 batches) |
| **QAT**  | You need INT4 or lower and PTQ collapses. The model trains *with* the quantizer in the loop and learns to compensate. | Full training run (typically 5-10 epochs from an FP32 seed) |

---

## Configuration

The stack composes through Hydra config groups:

```
configs/
├── quantization/
│   ├── int8_w.yaml      # spec, calibration.num_batches, skip_patterns
│   ├── int8_wa.yaml
│   ├── ...
│   └── none.yaml        # sentinel: bypass quantization
├── callbacks/
│   └── qat.yaml         # QATCallback + QuantizedCheckpointHandler
└── experiment/quant/
    ├── sv_qat_ecapa_int8_wa.yaml      # ECAPA × INT8 W+A
    └── sv_qat_resnet34_int4_w.yaml    # ResNet34 × INT4 weight-only
```

**Override `skip_patterns` from the CLI** to protect specific layers (regex on the qualified module name relative to the encoder root):

```bash
# ECAPA: keep first conv + final embedding linear in FP32 at INT2
python scripts/quantize_ptq.py \
    exp_dir=/path/to/run \
    quantization=int2_w \
    'quantization.skip_patterns=["^0\.layer1\.conv$","^0\.linear$"]'
```

**Restrict `target_layers`** to one class:

```bash
python src/train.py experiment=quant/sv_qat_resnet34_int4_w \
    callbacks.qat.target_layers='[Conv2d]'  # leave Linear FP32
```

The wespeaker model lives at index 0 of the encoder `nn.Sequential`, hence the `^0\.` prefix in regex patterns. Run a small inspection to dump real layer names:

```python
from src.utils.hf_utils import load_wespeaker_model
import torch.nn as nn
m = load_wespeaker_model("resnet34", model_args={"feat_dim": 80, "embed_dim": 256, "pooling_func": "TSTP"})
for n, c in m.named_modules():
    if isinstance(c, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        print(n)
```

---

## Artifact layout

**QAT** (writes through Lightning's `ModelCheckpoint`):

```
logs/train/runs/<timestamp>/
└── checkpoints/
    ├── epoch005-loss_valid…-qint8_wa.ckpt   # filename carries spec via interpolation
    └── last.ckpt
```

The ckpt's payload includes `qat_callback_state = {"spec": ..., "skip_patterns": ..., "substituted": [...]}` (see `src/callbacks/quantization/qat.py:139-145`).

**PTQ** (`scripts/quantize_ptq.py`):

```
<exp_dir>/
└── quantized/
    └── int8_wa/
        ├── state_dict.pth        # model.state_dict() with Brevitas quant buffers
        ├── quantization.yaml     # the resolved cfg.quantization
        └── source_ckpt.txt       # path of the FP32 ckpt PTQ started from
```

**Silent-overwrite convention:** rerunning PTQ with the same `spec` against the same `exp_dir` overwrites the prior `state_dict.pth` and friends. A `log.warning` is emitted on overwrite. If you need parallel sweeps, run them under distinct `exp_dir`s.

---

## Loading a quantized checkpoint for eval

`src/eval.py` reads two config keys:

- `cfg.quantization` — the spec dict (same as used at training/PTQ time).
- `cfg.quantized_ckpt_path` — path to a PTQ `state_dict.pth` (optional; QAT ckpts go through `cfg.ckpt_path` as usual).

The eval-time order is fixed at `src/eval.py:184-203`:

1. `quantize_model(model, cfg.quantization)` — substitute layers, so the key names match.
2. `load_quantized_state_dict(model, cfg.quantized_ckpt_path)` — overlay weights + scales.

`quantize_model` always runs **before** `load_quantized_state_dict` so the keys align. There is no code path that loads a quantized ckpt into an un-substituted model.

---

## Visualization

`scripts/visualize_quantization.py` produces a bar plot per (test_set, metric_kind):

```
input_dir/
├── run_fp32/test_artifacts/.../*_metrics.json
├── run_int8_w/test_artifacts/.../*_metrics.json
├── run_int4_wa/test_artifacts/.../*_metrics.json
└── plots/
    ├── <test_set>_<metric>_norm.pdf
    └── <test_set>_<metric>_raw.pdf
```

- **Input discovery**: walks `--input_dir` for `*_metrics.json` files. The spec name is parsed from a path component using the `SPEC_RE` regex (`scripts/visualize_quantization.py:64-65`).
- **X-axis order**: hard-coded `SPEC_ORDER` list places FP32 leftmost, then FP16, then descending bit-widths × {w, wa} (see lines 38-49).
- **Note** vs. the pruning visualizers, which read pre-aggregated `eer_leaderboard.csv`: this is intentional. Quant sweeps do not yet have a leaderboard CSV; the per-run JSONs are sufficient.

---

## Compatibility caveats

- **SWA + QAT is hard-blocked.** Averaging QAT iterates miscalibrates quantizer scales (averaged FP32 weights vs. per-ckpt-calibrated scales). The guard at `src/train.py:153-167` raises a `ValueError` if both `callbacks.qat` and `callbacks.checkpoint_averaging` are wired live. The shipped QAT experiment configs set `callbacks.checkpoint_averaging: null` explicitly.
- **Pruning + Quantization simultaneously is untested.** The two compression stacks have not been validated as concurrent. Run them sequentially: prune first, then PTQ the pruned ckpt.
- **FP16 + mixed-precision training is separate.** The `fp16.yaml` quantization spec only casts the encoder's weights to half precision; it does **not** set `trainer.precision`. If you want mixed-precision training, pass `trainer.precision=16-mixed` explicitly.
- **`save_state_dicts` has no quant finalizer.** Unlike pruning (`saving_utils._finalize_pruned_state_dict`), there is no equivalent step for quantization. Brevitas stores scales/zero-points as buffers inside the layer itself, so the standard `state_dict()` round-trips correctly without surgery.
- **Spec mismatch at eval raises.** Brevitas reuses the same state-dict key layout across bit widths, so a plain `strict=True` load cannot detect a mismatch (e.g. int4_w state dict loaded under an int8_w cfg). `load_quantized_state_dict` enforces parity by reading the sibling `quantization.yaml` (written by `scripts/quantize_ptq.py`) and comparing its `spec` against the caller's cfg; both a missing sidecar and a spec disagreement raise. See `tests/test_quantization_integration.py::test_ptq_spec_mismatch_raises`.

---

## Technical details

### Brevitas observer semantics

Activation specs (`*_wa`) wrap each layer with an activation observer that tracks running statistics. During training, the model's `.train()` flag enables observer updates; during eval, observers freeze. PTQ explicitly enters `calibration_mode()` via `src/quantization/calibrate.py:36-129` to fit observers from a handful of representative batches without any gradient updates.

`load_quantized_state_dict` defaults to `strict=True` (`src/quantization/wrap.py`). Quantized state dicts include the Brevitas quant buffers (`*.weight_quant.scale`, `*.input_quant.zero_point`, observer running stats), so a strict load is the right contract once the model has been substituted. The legacy "load a vanilla FP32 ckpt into a quantized model" flow goes through `QuantizedCheckpointHandler` (training-resume only); the eval path loads the FP32 ckpt *before* substitution and is strict on both sides.

### Round-trip identity

The spec name is persisted in four places that must agree:

1. `model._quantization_spec: str` — set at `wrap.py:87` (int specs) and `wrap.py:130` (FP16); read by `is_quantized` and the checkpoint handler.
2. `checkpoint["qat_callback_state"]["spec"]` — saved by `QATCallback.on_save_checkpoint` (qat.py:139-145).
3. `<exp_dir>/quantized/<spec>/quantization.yaml` — saved by `scripts/quantize_ptq.py:128`.
4. The ckpt filename via `q${oc.select:callbacks.qat.spec,fp32}` in `configs/callbacks/model_checkpoint.yaml:17-19` — resolves to `qint8_wa` / `qfp32` at Hydra-compose time.

### Substitution algorithm

`QuantizationManager.substitute()` (`src/quantization/manager.py:72-114`) walks `module.named_modules()`, matches `target_layers` (class-name filter), checks `skip_patterns` (regex `search` on the qualified name), then `setattr`-swaps the leaf with a Brevitas equivalent. Weights and bias are copied tensor-wise (`manager.py:199-202`), preserving the pretrained init.

`_resolve_encoder` in `wrap.py:19-32` is what keeps the Fbank front-end out of substitution: it walks down to `model.audio_encoder.encoder` and ignores the sibling `audio_processor` / `audio_processor_normalizer`.

### FP16 asymmetry

FP16 deliberately does **not** go through `QuantizationManager`. `_apply_fp16` in `wrap.py:115-132` calls `.half()` on the encoder (and optionally the classifier). The substitution path is skipped because FP16 weights are storage-format quantization, not numeric-format quantization — no scales, no observers, no calibration.

If you call `QuantizationManager.substitute()` on an FP16 spec, it raises (manager.py:78-82). The dispatch is gated by `spec.is_fp16` at `wrap.py:68-69`.

### Bit-budget accounting

`src/quantization/utils.py` exports two helpers:

- `compute_model_bits(model, spec=None)` → `{"weights_bits": int, "buffers_bits": int, "total_bits": int}`. When a `spec` is passed, INT weights inside Brevitas quant layers are counted at `spec.bit_width`; other tensors are counted by their actual dtype. Quantizer proxy sub-modules (scale / zero-point) are deliberately excluded so the compression ratio compares like with like.
- `compression_ratio(fp32_model, quant_model, spec)` → `total_bits(fp32) / total_bits(quant)`.

### Calibration data

`scripts/quantize_ptq.py` uses `datamodule.train_dataloader()` to source calibration batches. No overlap with eval test loaders by construction — the datamodule keeps these splits disjoint at setup time. The default is 32 batches (configurable via `quantization.calibration.num_batches`); weight-only specs skip calibration entirely (`quantize_ptq.py:104-115`).

---

## Troubleshooting

**Many missing / unexpected keys when loading a quant ckpt.**
The most common cause is a spec mismatch between `cfg.quantization` and the ckpt's actual spec. Check `model._quantization_spec` after `quantize_model`, and the `qat_callback_state["spec"]` (or `quantization.yaml`) in the source artifact. They must agree.

**INT2 / INT1 accuracy collapses.**
You are quantizing the first conv (raw signal path) or the final embedding linear. Add a `skip_patterns` regex to protect them — see the per-backbone defaults in `configs/experiment/quant/sv_qat_*.yaml` and the override example in Configuration above.

**`ValueError: callbacks.qat and callbacks.checkpoint_averaging cannot be enabled together`.**
Working as intended. SWA over QAT iterates miscalibrates scales (see Compatibility caveats). Either set `callbacks.checkpoint_averaging: null` in your experiment config, or run SWA on an FP32 baseline and quantize the average via `scripts/quantize_ptq.py`.

**PTQ overwrote my prior run.**
By design — `<exp_dir>/quantized/<spec>/` is keyed only by spec name. A `log.warning` fires on overwrite. Run parallel sweeps under distinct `exp_dir`s, or hand-copy artifacts before rerunning.

**`brevitas` import errors.**
The stack lazy-imports Brevitas. Install with `pip install brevitas==0.11.0` to match the pinned version (`requirements.txt`). Other versions may produce different key names for quant buffers and break the round-trip.

---

## File Structure

```
src/quantization/
├── __init__.py              # Re-exports: resolve_spec, quantize_model, …
├── bit_specs.py             # BitSpec registry + weight/act quantizer factories
├── manager.py               # QuantizationManager.substitute()
├── wrap.py                  # quantize_model, load_quantized_state_dict, is_quantized
├── calibrate.py             # PTQ calibration loop
├── checkpoint_handler.py    # QuantizedCheckpointHandler Lightning callback
└── utils.py                 # compute_model_bits, compression_ratio, describe_spec

src/callbacks/quantization/
├── __init__.py
└── qat.py                   # QATCallback

configs/quantization/
├── int{1,2,4,8}_{w,wa}.yaml # Int spec configs
├── fp16.yaml                # FP16 cast
└── none.yaml                # Bypass sentinel

configs/callbacks/
├── qat.yaml                 # Wires QATCallback + QuantizedCheckpointHandler
└── model_checkpoint.yaml    # Filename includes q${oc.select:callbacks.qat.spec,fp32}

configs/experiment/quant/
├── sv_qat_ecapa_int8_wa.yaml
└── sv_qat_resnet34_int4_w.yaml

scripts/
├── quantize_ptq.py          # PTQ entry point
└── visualize_quantization.py # Bit-width sweep visualizer

tests/
├── test_quantization.py             # Component-level (TinyEncoder)
└── test_quantization_integration.py # Save/load round-trip
```
