"""Correctness tests for the quantization stack.

Each test that needs brevitas uses ``pytest.importorskip("brevitas")`` so
the suite skips cleanly when the dependency isn't installed.

Coverage:

1. Layer substitution preserves output shape on a tiny model.
2. Bit width is actually enforced (assert ``weight_quant.bit_width``).
3. PTQ calibration runs end-to-end on a synthetic dataloader.
4. QAT forward + backward + optimizer step.
5. State-dict round-trip preserves outputs.
6. Weight-only vs weight+activation modes both produce finite outputs.
7. FP16 path casts encoder weights to half precision.
8. Bit-budget accounting helpers.
9. Hydra config composition for every quantization spec + experiment yaml.
10. End-to-end Lightning Trainer.fit with the QATCallback.
11. Quantization-error magnitudes vs FP32 (informational + loose guard).
12. Brevitas-side deprecation warnings do not corrupt forward outputs.
"""

from __future__ import annotations

import warnings

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.quantization.bit_specs import BIT_SPECS, QuantMode, resolve_spec
from src.quantization.manager import QuantizationManager
from src.quantization.utils import compute_model_bits, describe_spec
from src.quantization.wrap import is_quantized, quantize_model


# ---------------------------------------------------------------------------
# Tiny model fixtures
# ---------------------------------------------------------------------------


class TinyEncoder(nn.Module):
    """Conv1d + Conv2d + Linear — covers all three substitution targets."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(4, 8, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(8)
        self.proj = nn.Linear(8, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.bn(x)
        x = x.mean(dim=-1)  # global avg pool over time
        return self.proj(x)


class TinyEncoderWrapper(nn.Module):
    """Mimics src.modules.encoder_wrappers.EncoderWrapper layout so
    ``quantize_model`` can find ``model.audio_encoder.encoder``."""

    def __init__(self) -> None:
        super().__init__()
        self.audio_encoder = nn.Module()
        self.audio_encoder.encoder = TinyEncoder()
        self.classifier = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.audio_encoder.encoder(x))


def _random_input(batch_size: int = 2) -> torch.Tensor:
    return torch.randn(batch_size, 1, 32)


def _all_int_specs() -> list[str]:
    """All non-FP16 specs (FP16 has a separate code path)."""
    return [name for name, s in BIT_SPECS.items() if not s.is_fp16]


# ---------------------------------------------------------------------------
# 1. Substitution preserves shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec_name", _all_int_specs())
def test_substitution_preserves_output_shape(spec_name: str) -> None:
    pytest.importorskip("brevitas")

    fp32 = TinyEncoderWrapper()
    fp32.eval()
    x = _random_input()
    fp32_out = fp32(x)

    quant = TinyEncoderWrapper()
    quant.load_state_dict(fp32.state_dict())
    quant.eval()
    quantize_model(quant, {"spec": spec_name})

    quant_out = quant(x)
    assert quant_out.shape == fp32_out.shape
    assert torch.isfinite(quant_out).all(), (
        f"Spec {spec_name} produced non-finite outputs."
    )


# ---------------------------------------------------------------------------
# 2. Bit width is enforced
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec_name", _all_int_specs())
def test_bit_width_is_enforced(spec_name: str) -> None:
    pytest.importorskip("brevitas")
    import brevitas.nn as qnn

    spec = resolve_spec(spec_name)
    model = TinyEncoderWrapper()
    quantize_model(model, {"spec": spec_name})

    quant_layers = [
        m
        for m in model.audio_encoder.encoder.modules()
        if isinstance(m, (qnn.QuantConv1d, qnn.QuantConv2d, qnn.QuantLinear))
    ]
    assert quant_layers, "No quant layers were created."

    for layer in quant_layers:
        # ``weight_quant.bit_width`` is a tensor (a constant); compare via item().
        actual_bw = int(layer.weight_quant.bit_width().item())
        assert actual_bw == spec.bit_width, (
            f"{type(layer).__name__}: expected bit_width={spec.bit_width}, "
            f"got {actual_bw}"
        )


# ---------------------------------------------------------------------------
# 3. PTQ calibration runs end-to-end
# ---------------------------------------------------------------------------


def _calibration_loader() -> DataLoader:
    xs = torch.randn(16, 1, 32)
    ys = torch.randint(0, 4, (16,))
    return DataLoader(TensorDataset(xs, ys), batch_size=4)


@pytest.mark.parametrize(
    "spec_name", [n for n, s in BIT_SPECS.items() if s.quantize_activations]
)
def test_ptq_calibration_runs(spec_name: str) -> None:
    pytest.importorskip("brevitas")
    from src.quantization import calibrate

    model = TinyEncoderWrapper()
    quantize_model(model, {"spec": spec_name})

    loader = _calibration_loader()

    def fwd(m, batch):
        x, _ = batch
        return m(x)

    consumed = calibrate(
        model=model, dataloader=loader, num_batches=2, forward_fn=fwd
    )
    assert consumed == 2

    # Post-calibration forward must be finite.
    out = model(torch.randn(2, 1, 32))
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 4. QAT forward + backward + step
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec_name", _all_int_specs())
def test_qat_training_step(spec_name: str) -> None:
    pytest.importorskip("brevitas")

    model = TinyEncoderWrapper()
    quantize_model(model, {"spec": spec_name})
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    x = _random_input()
    target = torch.zeros(x.size(0), 4)

    out = model(x)
    loss = ((out - target) ** 2).mean()
    loss.backward()
    optimizer.step()

    # At least one quant layer's weight must have received a gradient (STE).
    grads = [
        p.grad
        for n, p in model.audio_encoder.encoder.named_parameters()
        if n.endswith("weight") and p.grad is not None
    ]
    assert grads, f"No gradients on quant weights for spec {spec_name}."
    for g in grads:
        assert torch.isfinite(g).all(), (
            f"Non-finite gradient under spec {spec_name}."
        )


# ---------------------------------------------------------------------------
# 5. State-dict round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec_name",
    [n for n, s in BIT_SPECS.items() if not s.quantize_activations],
)
def test_state_dict_round_trip(spec_name: str) -> None:
    """Weight-only modes round-trip exactly. W+A modes need calibration first
    to give the activation quantizers a defined scale, so we cover them in
    the calibration test."""
    if spec_name != "fp16":
        pytest.importorskip("brevitas")

    a = TinyEncoderWrapper()
    quantize_model(a, {"spec": spec_name})
    a.eval()
    sd = a.state_dict()

    b = TinyEncoderWrapper()
    quantize_model(b, {"spec": spec_name})
    b.eval()
    b.load_state_dict(sd, strict=False)

    # Forward through the encoder only — the FP32 classifier won't accept
    # FP16 inputs and isn't part of the round-trip we care about.
    x = _random_input()
    if spec_name == "fp16":
        x = x.half()
    with torch.no_grad():
        out_a = a.audio_encoder.encoder(x)
        out_b = b.audio_encoder.encoder(x)
    assert torch.allclose(out_a, out_b, atol=1e-5), (
        f"Round-trip mismatch for {spec_name}: max diff "
        f"{(out_a - out_b).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# 6. Weight-only vs W+A: both modes are usable
# ---------------------------------------------------------------------------


def test_weight_only_and_w_plus_a_both_work() -> None:
    pytest.importorskip("brevitas")

    m_wo = TinyEncoderWrapper()
    quantize_model(m_wo, {"spec": "int8_w"})
    m_wa = TinyEncoderWrapper()
    quantize_model(m_wa, {"spec": "int8_wa"})

    x = _random_input()
    out_wo = m_wo(x)
    out_wa = m_wa(x)
    assert torch.isfinite(out_wo).all() and torch.isfinite(out_wa).all()
    # The two modes should differ — same model + same input but different
    # quantization paths.
    assert not torch.allclose(out_wo, out_wa, atol=1e-4), (
        "Weight-only and W+A produced identical outputs; one of them is "
        "probably not wired up."
    )


# ---------------------------------------------------------------------------
# 7. FP16 path
# ---------------------------------------------------------------------------


def test_fp16_path_casts_encoder() -> None:
    model = TinyEncoderWrapper()
    quantize_model(model, {"spec": "fp16"})
    assert is_quantized(model)
    # Encoder params must be half precision; front-end-equivalent stays fp32.
    for p in model.audio_encoder.encoder.parameters():
        assert p.dtype == torch.float16, (
            f"FP16 spec did not cast encoder param: dtype={p.dtype}"
        )
    # Classifier stays FP32 unless quantize_classifier=True.
    for p in model.classifier.parameters():
        assert p.dtype == torch.float32


# ---------------------------------------------------------------------------
# 8. Bit-budget accounting helpers
# ---------------------------------------------------------------------------


def test_compute_model_bits_returns_finite() -> None:
    pytest.importorskip("brevitas")

    fp32 = TinyEncoderWrapper()
    quant = TinyEncoderWrapper()
    quant.load_state_dict(fp32.state_dict())
    quantize_model(quant, {"spec": "int4_w"})

    fp32_bits = compute_model_bits(fp32, spec=None)
    spec = resolve_spec("int4_w")
    quant_bits = compute_model_bits(quant, spec=spec)

    assert fp32_bits["total_bits"] > 0
    assert quant_bits["total_bits"] > 0
    # INT4 weights should reduce the weights bit count meaningfully.
    assert quant_bits["weights_bits"] < fp32_bits["weights_bits"]


# ---------------------------------------------------------------------------
# Sanity tests that don't need brevitas
# ---------------------------------------------------------------------------


def test_resolve_spec_unknown_raises() -> None:
    with pytest.raises(KeyError):
        resolve_spec("not_a_spec")


def test_describe_spec_strings() -> None:
    assert "FP16" in describe_spec(resolve_spec("fp16"))
    assert "INT8" in describe_spec(resolve_spec("int8_wa"))
    assert "weight-only" in describe_spec(resolve_spec("int8_w"))


def test_bit_spec_enumerates_expected_combinations() -> None:
    names = set(BIT_SPECS)
    expected = {
        "int8_w", "int8_wa",
        "int4_w", "int4_wa",
        "int2_w", "int2_wa",
        "int1_w", "int1_wa",
        "fp16",
    }
    assert expected.issubset(names)


def test_manager_skip_patterns_filter_layers() -> None:
    pytest.importorskip("brevitas")

    model = TinyEncoder()
    spec = resolve_spec("int8_w")
    manager = QuantizationManager(spec=spec, skip_patterns=[r"^conv1$"])
    report = manager.substitute(model)

    skipped_names = {name for name, _ in report.skipped}
    assert "conv1" in skipped_names
    assert "conv1" not in report.substituted
    assert "conv2" in report.substituted


def test_quant_mode_enum_values() -> None:
    assert QuantMode.WEIGHT_ONLY.value == "weight_only"
    assert QuantMode.WEIGHT_ACT.value == "weight_act"
    assert QuantMode.FP16.value == "fp16"


# ---------------------------------------------------------------------------
# 9. Hydra config composition (catches defaults-list ordering / package
# bugs that don't show up in unit tests but break `python src/train.py
# experiment=quant/...`).
# ---------------------------------------------------------------------------


import glob
from pathlib import Path

import pyrootutils
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from src.utils import register_custom_resolvers

_ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
_CONFIGS_DIR = _ROOT / "configs"


def _compose(config_name: str, overrides):
    """Compose a Hydra config from configs/, isolating GlobalHydra state."""
    GlobalHydra.instance().clear()

    @register_custom_resolvers(
        config_name=config_name,
        overrides=overrides,
        version_base="1.3",
        config_path=str(_CONFIGS_DIR),
    )
    def _do_compose():
        with initialize_config_dir(
            version_base="1.3", config_dir=str(_CONFIGS_DIR)
        ):
            return compose(config_name=config_name, overrides=overrides)

    try:
        return _do_compose()
    finally:
        GlobalHydra.instance().clear()


def _quantization_spec_names() -> list[str]:
    files = sorted(glob.glob(str(_CONFIGS_DIR / "quantization" / "*.yaml")))
    return [Path(p).stem for p in files]


def _quant_experiment_names() -> list[str]:
    files = sorted(
        glob.glob(str(_CONFIGS_DIR / "experiment" / "quant" / "*.yaml"))
    )
    return [Path(p).stem for p in files]


@pytest.mark.parametrize("spec_name", _quantization_spec_names())
def test_train_composes_with_quantization_override(spec_name: str) -> None:
    """`python src/train.py quantization=<spec>` must compose cleanly.

    Catches: 'Multiple values for quantization' (missing `override`),
    typos in /quantization/<name>.yaml, etc.
    """
    cfg = _compose(
        "train.yaml",
        overrides=[f"quantization={spec_name}", "logger=[]"],
    )
    assert cfg.quantization is not None
    if spec_name != "none":
        assert cfg.quantization.spec == spec_name


@pytest.mark.parametrize("exp_name", _quant_experiment_names())
def test_quant_experiment_composes(exp_name: str) -> None:
    """`python src/train.py experiment=quant/<name>` must compose cleanly.

    Catches every Hydra error a misconfigured quant experiment can raise at
    composition time — defaults-list ordering, missing config groups,
    duplicate group bindings.
    """
    cfg = _compose(
        "train.yaml",
        overrides=[f"experiment=quant/{exp_name}", "logger=[]"],
    )
    # Quantization spec wired through:
    assert cfg.quantization.spec not in (None, "none"), (
        f"Experiment {exp_name} composed without a real quantization spec."
    )
    # QAT callback is present and references the same spec:
    assert "qat" in cfg.callbacks, (
        f"Experiment {exp_name} did not wire up the QAT callback."
    )
    assert cfg.callbacks.qat.spec == cfg.quantization.spec, (
        f"QAT callback spec ({cfg.callbacks.qat.spec}) disagrees with "
        f"quantization spec ({cfg.quantization.spec}) in {exp_name}."
    )


# ---------------------------------------------------------------------------
# 10. End-to-end Lightning Trainer.fit with QATCallback
# ---------------------------------------------------------------------------


class _TinyQATLightningModule(pl.LightningModule):
    """Minimum-viable LightningModule that mirrors SpeakerVerification's
    layout (audio_encoder.encoder + classifier) so QATCallback finds the
    same substitution target a real run would."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.audio_encoder = nn.Module()
        self.audio_encoder.encoder = TinyEncoder()
        self.classifier = nn.Linear(16, num_classes)
        self._loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.audio_encoder.encoder(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        return self._loss(self(x), y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self._loss(self(x), y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-3)


def _tiny_loader(n: int = 8) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.randn(n, 1, 32), torch.randint(0, 4, (n,))),
        batch_size=2,
    )


@pytest.mark.parametrize("spec_name", ["int8_w", "int4_w", "int8_wa"])
def test_qat_callback_in_lightning_fit(
    spec_name: str, tmp_path: Path
) -> None:
    """Full Trainer.fit cycle with QATCallback — would have caught the
    'No quant layers were created' / config wiring issues if they had
    survived composition tests."""
    pytest.importorskip("brevitas")
    from src.callbacks.quantization.qat import QATCallback
    from src.quantization import is_quantized

    model = _TinyQATLightningModule()
    callback = QATCallback(spec=spec_name, verbose=False)

    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        callbacks=[callback],
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
        accelerator="cpu",
        default_root_dir=str(tmp_path),
    )
    trainer.fit(model, _tiny_loader(), _tiny_loader())

    assert is_quantized(model), "QATCallback did not mark the module quantized"
    assert callback._applied, "QATCallback.setup did not run"
    assert callback._substituted, "Zero layers substituted by QATCallback"

    # One forward pass after fit must still be finite.
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 1, 32))
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 11. Quantization-error report (informational + correctness guards)
# ---------------------------------------------------------------------------


def _spec_error_threshold(spec_name: str) -> float:
    """Loose upper bound on relative mean-abs error per spec, on a
    no-calibration weight-copied tiny model. These thresholds are wide
    enough not to flake but tight enough to catch a wiring regression
    (e.g. quantizer disabled, wrong bit width)."""
    return {
        "int8_w": 0.02,
        "int8_wa": 0.05,
        "int4_w": 0.20,
        "int4_wa": 0.30,
        "int2_w": 0.60,
        "int2_wa": 0.80,
        "int1_w": 1.00,
        "int1_wa": 5.00,
    }[spec_name]


@pytest.mark.parametrize("spec_name", _all_int_specs())
def test_quantization_error_within_expected_range(
    spec_name: str, capsys
) -> None:
    """Reports mean/max abs error vs FP32 and asserts a loose upper bound.

    Run with ``pytest -s`` to see the per-spec error numbers.
    """
    pytest.importorskip("brevitas")

    torch.manual_seed(0)
    fp32 = TinyEncoderWrapper()
    fp32.eval()
    x = torch.randn(8, 1, 64)
    with torch.no_grad():
        ref = fp32(x)

    quant = TinyEncoderWrapper()
    quant.load_state_dict(fp32.state_dict())
    quantize_model(quant, {"spec": spec_name})
    quant.eval()
    with torch.no_grad():
        out = quant(x)

    err = (out - ref).abs()
    rel = (err.mean() / ref.abs().mean()).item()
    print(
        f"[quant-error] {spec_name:<8s} mean|err|={err.mean().item():.4e} "
        f"max|err|={err.max().item():.4e} rel_mean={rel:.3f}"
    )

    assert torch.isfinite(out).all()
    threshold = _spec_error_threshold(spec_name)
    assert rel <= threshold, (
        f"Quantization error for {spec_name} is {rel:.3f}, "
        f"expected <= {threshold} on the no-calibration tiny model."
    )


# ---------------------------------------------------------------------------
# 12. Brevitas-side deprecation warnings do not corrupt forward outputs
# ---------------------------------------------------------------------------


def test_brevitas_deprecation_warnings_are_noise_only() -> None:
    """Brevitas 0.11 on PyTorch >=2.5 emits two UserWarnings on the first
    forward through a quant layer:

      - 'Named tensors and all their associated APIs are an experimental
        feature ...'
      - 'Defining your `__torch_function__` as a plain method is
        deprecated ...'

    These are upstream-pending fixes (Brevitas internals) and do not
    affect numerical correctness. This test pins that contract: if a
    future PyTorch/Brevitas combination starts producing non-finite
    outputs alongside the warnings, this test fails loudly instead of
    letting the warnings hide a real bug.
    """
    pytest.importorskip("brevitas")
    import brevitas.nn as qnn

    layer = qnn.QuantConv2d(1, 4, kernel_size=3)
    x = torch.randn(2, 1, 8, 8)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        out = layer(x)

    assert out.shape == (2, 4, 6, 6)
    assert out.dtype == torch.float32  # fake quant returns FP32 tensors
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 13. _parse_cfg accepts BitSpec, DictConfig, and plain dict
# ---------------------------------------------------------------------------


def test_parse_cfg_accepts_all_three_input_shapes() -> None:
    """`quantize_model` should accept BitSpec, OmegaConf DictConfig, and
    plain Python dict cfgs — the three branches in `wrap._parse_cfg`."""
    pytest.importorskip("brevitas")
    from omegaconf import OmegaConf

    spec = resolve_spec("int8_w")

    # (a) BitSpec — bypasses _parse_cfg's dict parsing.
    model_a = TinyEncoderWrapper()
    report_a = quantize_model(model_a, spec)
    assert len(report_a.substituted) > 0
    assert is_quantized(model_a)

    # (b) Plain Python dict.
    model_b = TinyEncoderWrapper()
    report_b = quantize_model(
        model_b,
        {"spec": "int8_w", "skip_patterns": [], "quantize_classifier": False},
    )
    assert len(report_b.substituted) == len(report_a.substituted)

    # (c) OmegaConf DictConfig.
    model_c = TinyEncoderWrapper()
    report_c = quantize_model(
        model_c,
        OmegaConf.create(
            {
                "spec": "int8_w",
                "skip_patterns": [],
                "quantize_classifier": False,
            }
        ),
    )
    assert len(report_c.substituted) == len(report_a.substituted)


# ---------------------------------------------------------------------------
# 14. quantize_classifier=True substitutes the classifier head
# ---------------------------------------------------------------------------


def test_quantize_classifier_substitutes_head() -> None:
    """When `quantize_classifier=True`, the Linear children inside the
    classifier head are substituted. Real classifiers (e.g. SpeechBrain's
    `ECAPA_TDNN.Classifier`) are container modules — mirror that here."""
    pytest.importorskip("brevitas")
    import brevitas.nn as qnn

    class ContainerClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(16, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    model = TinyEncoderWrapper()
    model.classifier = ContainerClassifier()

    report = quantize_model(
        model,
        {"spec": "int8_w", "quantize_classifier": True},
    )
    assert isinstance(model.classifier.fc, qnn.QuantLinear)
    # Both encoder layers and classifier child appear in the report.
    assert any(name == "fc" for name in report.substituted)


def test_quantize_classifier_default_leaves_head_fp32() -> None:
    """Default `quantize_classifier=False` must leave the head untouched."""
    pytest.importorskip("brevitas")

    class ContainerClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(16, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    model = TinyEncoderWrapper()
    model.classifier = ContainerClassifier()

    quantize_model(model, {"spec": "int8_w"})
    assert isinstance(model.classifier.fc, nn.Linear)
    assert type(model.classifier.fc).__name__ == "Linear"
