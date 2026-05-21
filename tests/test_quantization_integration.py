"""End-to-end round-trip tests for the quantization stack.

The unit tests in ``tests/test_quantization.py`` verify individual
operations (substitute, calibrate, fit) but never close the loop
*save -> reload -> compare forward outputs*. These integration tests
exercise that loop for both QAT and PTQ paths, on the same
``audio_encoder.encoder + classifier`` layout used by real
``SpeakerVerification`` / ``Countermeasure`` modules.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Fixtures: tiny model + datamodule mirroring the production layout
# ---------------------------------------------------------------------------


class _TinyEncoder(nn.Module):
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
        x = x.mean(dim=-1)
        return self.proj(x)


class _TinyQATModule(pl.LightningModule):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.audio_encoder = nn.Module()
        self.audio_encoder.encoder = _TinyEncoder()
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


def _loader(n: int = 8, seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(
            torch.randn(n, 1, 32, generator=g),
            torch.randint(0, 4, (n,), generator=g),
        ),
        batch_size=2,
    )


# ---------------------------------------------------------------------------
# QAT round-trip: fit -> save -> reload via QATCallback -> outputs match
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec_name", ["int8_w", "int4_w"])
def test_qat_save_reload_inference_matches(
    spec_name: str, tmp_path: Path
) -> None:
    """QAT fit one epoch, save the Lightning checkpoint, reload into a fresh
    module via QATCallback's `on_load_checkpoint`, and confirm forward
    outputs are bit-identical between trained-in-memory and reloaded models.

    Catches: substitution mismatch on reload, missing/unexpected key blow-up,
    silent dtype drift through the checkpoint round-trip.
    """
    pytest.importorskip("brevitas")
    from src.callbacks.quantization.qat import QATCallback
    from src.quantization import is_quantized

    torch.manual_seed(0)
    model = _TinyQATModule()
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
    trainer.fit(model, _loader(seed=1), _loader(seed=2))
    assert is_quantized(model)

    # Save via Lightning so the callback's on_save_checkpoint runs.
    ckpt_path = tmp_path / "qat.ckpt"
    trainer.save_checkpoint(str(ckpt_path))

    # Reference forward — quantized observers are now locked in `.eval()`.
    model.eval()
    fixed_input = torch.randn(2, 1, 32, generator=torch.Generator().manual_seed(99))
    with torch.no_grad():
        ref_out = model(fixed_input)

    # Reload into a fresh module + fresh callback.
    torch.manual_seed(0)
    fresh = _TinyQATModule()
    fresh_callback = QATCallback(spec=spec_name, verbose=False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    fresh_callback.on_load_checkpoint(trainer=None, pl_module=fresh, checkpoint=ckpt)
    fresh.load_state_dict(ckpt["state_dict"], strict=False)
    fresh.eval()

    with torch.no_grad():
        reloaded_out = fresh(fixed_input)

    assert torch.allclose(ref_out, reloaded_out, atol=1e-5), (
        f"QAT round-trip diverged for {spec_name}: "
        f"max|diff|={(ref_out - reloaded_out).abs().max().item():.3e}"
    )
    assert is_quantized(fresh), "QATCallback.on_load_checkpoint did not re-tag the model"


# ---------------------------------------------------------------------------
# PTQ round-trip: quantize -> save state_dict.pth -> load into fresh model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec_name", ["int8_w", "int4_w"])
def test_ptq_save_reload_inference_matches(
    spec_name: str, tmp_path: Path
) -> None:
    """Quantize a model, save just the state dict (as `scripts/quantize_ptq.py`
    does), load via `load_quantized_state_dict` into a fresh model, and
    confirm forward outputs match bit-for-bit.

    Catches: a mismatch between the keys produced by `quantize_model` and
    those expected by `load_quantized_state_dict`; a stale
    `_quantization_spec` attribute after reload; missing scales when the
    callback handler is bypassed.
    """
    pytest.importorskip("brevitas")
    import yaml

    from src.quantization import (
        is_quantized,
        load_quantized_state_dict,
        quantize_model,
    )

    torch.manual_seed(0)
    model = _TinyQATModule()
    quantize_model(model, {"spec": spec_name})
    model.eval()

    fixed_input = torch.randn(2, 1, 32, generator=torch.Generator().manual_seed(99))
    with torch.no_grad():
        ref_out = model(fixed_input)

    state_path = tmp_path / "state_dict.pth"
    torch.save(model.state_dict(), state_path)
    # PTQ writes a sidecar quantization.yaml next to the state dict so
    # load-time spec parity can be enforced; mirror that here.
    (tmp_path / "quantization.yaml").write_text(yaml.safe_dump({"spec": spec_name}))

    torch.manual_seed(0)
    fresh = _TinyQATModule()
    load_quantized_state_dict(fresh, state_path, cfg={"spec": spec_name})
    fresh.eval()

    with torch.no_grad():
        reloaded_out = fresh(fixed_input)

    assert is_quantized(fresh)
    assert torch.allclose(ref_out, reloaded_out, atol=1e-6), (
        f"PTQ round-trip diverged for {spec_name}: "
        f"max|diff|={(ref_out - reloaded_out).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# Negative: spec mismatch between saved ckpt and reload cfg is detectable
# ---------------------------------------------------------------------------


def test_ptq_spec_mismatch_raises(tmp_path: Path) -> None:
    """Loading an int4_w PTQ artifact while requesting int8_w must raise.

    Brevitas reuses the same state-dict key layout across bit widths, so a
    plain `strict=True` load *cannot* detect the spec mismatch (empty
    missing/unexpected sets). `load_quantized_state_dict` enforces parity
    by reading the sibling `quantization.yaml` (written by
    `scripts/quantize_ptq.py`) and comparing it against the requested cfg.
    """
    pytest.importorskip("brevitas")
    import yaml

    from src.quantization import load_quantized_state_dict, quantize_model

    torch.manual_seed(0)
    src = _TinyQATModule()
    quantize_model(src, {"spec": "int4_w"})
    state_path = tmp_path / "state_dict.pth"
    torch.save(src.state_dict(), state_path)
    # PTQ writes a sidecar quantization.yaml; mimic that here.
    (tmp_path / "quantization.yaml").write_text(yaml.safe_dump({"spec": "int4_w"}))

    fresh = _TinyQATModule()
    with pytest.raises(ValueError, match="spec mismatch"):
        load_quantized_state_dict(fresh, state_path, cfg={"spec": "int8_w"})


def test_load_quantized_state_dict_requires_sidecar(tmp_path: Path) -> None:
    """Without a sibling quantization.yaml, the spec cannot be verified —
    the loader must refuse to proceed rather than silently load."""
    pytest.importorskip("brevitas")
    from src.quantization import load_quantized_state_dict, quantize_model

    torch.manual_seed(0)
    src = _TinyQATModule()
    quantize_model(src, {"spec": "int4_w"})
    state_path = tmp_path / "state_dict.pth"
    torch.save(src.state_dict(), state_path)
    # No sibling quantization.yaml on purpose.

    fresh = _TinyQATModule()
    with pytest.raises(FileNotFoundError, match="quantization.yaml"):
        load_quantized_state_dict(fresh, state_path, cfg={"spec": "int4_w"})
