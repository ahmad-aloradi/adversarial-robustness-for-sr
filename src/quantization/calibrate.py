"""Post-training calibration loop.

Pushes a handful of representative batches through the quantized model so
Brevitas activation-quantizer observers can fit their scale/zero-point.
Uses ``brevitas.graph.calibrate.calibration_mode`` — the canonical PTQ
calibration context manager.
"""

from __future__ import annotations

if __name__ == "__main__":
    # Make `src.*` imports below resolve when running this file directly
    # (`python src/quantization/calibrate.py`) for the smoke block at EOF.
    import pyrootutils

    pyrootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from typing import Any, Callable, Optional, Union

import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.quantization.bit_specs import resolve_spec
from src.utils import get_pylogger

log = get_pylogger(__name__)


def _default_forward(model: nn.Module, batch: Any) -> torch.Tensor:
    """Default forward: assume the batch has ``audio`` and ``audio_length``
    attributes (the SpeakerVerification convention) and call ``model.forward``.
    """
    if hasattr(model, "forward") and hasattr(batch, "audio"):
        out = model(batch)
        if isinstance(out, dict):
            result = out.get("embeds", next(iter(out.values())))
        else:
            result = out
    else:
        result = model(batch)
    # The activation observers only update when fed a Tensor — anything else
    # (e.g. a python int from a loss-only return) silently no-ops the forward.
    assert isinstance(result, torch.Tensor), (
        f"_default_forward must return a Tensor for calibration observers "
        f"to fit; got {type(result).__name__}. Pass a custom `forward_fn=` "
        f"to calibrate() if your model returns a non-tensor."
    )
    return result


def calibrate(
    model: nn.Module,
    dataloader: DataLoader,
    num_batches: int = 32,  # Brevitas-recommended PTQ default; overridable
    forward_fn: Optional[Callable[[nn.Module, Any], torch.Tensor]] = None,
    device: Optional[Union[str, torch.device]] = None,
    cfg: Optional[Union[DictConfig, dict]] = None,
) -> int:
    """Run PTQ calibration.

    Parameters
    ----------
    model : nn.Module
        Already-quantized model (call ``quantize_model`` first).
    dataloader : DataLoader
        Source of calibration batches. Typically the training set; a small
        ``Subset`` works fine.
    num_batches : int
        Number of batches to feed through.
    forward_fn : callable, optional
        How to invoke the model on a batch. Defaults to ``_default_forward``,
        which works for SV/CM modules.
    device : str | torch.device, optional
        Move the model and inputs to this device before calibration. If
        ``None``, leaves things where they are.
    cfg : DictConfig | dict, optional
        Read ``num_batches`` from here when set (overrides the argument).

    Returns
    -------
    int
        Number of batches actually consumed.
    """
    if cfg is not None and "num_batches" in cfg:
        num_batches = int(cfg["num_batches"])

    assert num_batches > 0, (
        f"PTQ calibration needs at least 1 batch, got num_batches="
        f"{num_batches}. Set quantization.calibration.num_batches in config."
    )

    try:
        from brevitas.graph.calibrate import calibration_mode
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise ImportError(
            "brevitas is required for PTQ calibration. Install with "
            "`pip install brevitas==0.11.0`."
        ) from exc

    fwd = forward_fn or _default_forward
    if device is not None:
        model.to(device)
    model.eval()

    # Const-scale quantizers (binary / ternary, i.e. bit_width <= 2) have no
    # tunable observers. brevitas's `calibration_mode` context raises
    # `AttributeError` when entered against a const quantizer. Detect this
    # up front from the model's spec tag rather than via a try/except — the
    # old broad catch also absorbed unrelated AttributeErrors from `fwd`,
    # which silently masked genuine bugs in the forward pass.
    spec_name = getattr(model, "_quantization_spec", None)
    if spec_name is None:
        raise RuntimeError(
            "calibrate: model is not quantized. Call `quantize_model` first."
        )
    spec = resolve_spec(spec_name)
    use_ctx = spec.bit_width > 2

    consumed = 0
    log.info(
        f"PTQ calibration: pushing up to {num_batches} batches through the model "
        f"({'with' if use_ctx else 'without'} calibration_mode — "
        f"spec={spec_name!r}, bit_width={spec.bit_width})."
    )

    ctx = calibration_mode(model) if use_ctx else None
    with torch.no_grad():
        if ctx is not None:
            ctx.__enter__()
        try:
            for batch_idx, batch in enumerate(
                tqdm(
                    dataloader,
                    total=num_batches,
                    desc="Calibrating" if use_ctx else "Calibrating (const)",
                    leave=False,
                )
            ):
                if batch_idx >= num_batches:
                    break
                if device is not None:
                    batch = _move_batch(batch, device)
                fwd(model, batch)
                consumed = batch_idx + 1
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)

    assert consumed > 0, (
        "Dataloader yielded no batches; activation observers were not fit. "
        "Check that the calibration dataset is non-empty."
    )
    log.info(f"PTQ calibration done after {consumed} batches.")
    return consumed


def _move_batch(batch: Any, device: Union[str, torch.device]) -> Any:
    """Best-effort move of a batch's tensor fields to ``device``."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, (list, tuple)):
        return type(batch)(_move_batch(b, device) for b in batch)
    if isinstance(batch, dict):
        return {k: _move_batch(v, device) for k, v in batch.items()}
    # Dataclass / NamedTuple-style batch (e.g. VoxcelebItem): move each tensor field.
    for attr in ("audio", "audio_length", "class_id"):
        if hasattr(batch, attr):
            val = getattr(batch, attr)
            if isinstance(val, torch.Tensor):
                setattr(batch, attr, val.to(device))
    return batch


if __name__ == "__main__":
    # Smoke: substitute a tiny model and run a 1-batch calibration loop.
    from src.quantization.manager import QuantizationManager

    model = nn.Conv1d(4, 8, 3, padding=1)
    QuantizationManager(resolve_spec("int8_wa")).substitute(model)
    model._quantization_spec = "int8_wa"  # tag the calibrate() check expects

    batch = {"audio": torch.randn(2, 4, 16)}
    fwd = lambda m, b: m(b["audio"])  # noqa: E731

    consumed = calibrate(model, [batch], num_batches=1, forward_fn=fwd)
    assert consumed == 1, f"expected consumed=1, got {consumed}"
    print(f"smoke ok: consumed={consumed} batch on int8_wa Conv1d")
