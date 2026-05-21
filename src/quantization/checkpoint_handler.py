"""Checkpoint compatibility for quantized models.

Patches ``LightningModule.load_state_dict`` so that an FP32 baseline ckpt
can be seeded into a Brevitas-wrapped module during training-resume:
Brevitas adds quant-state buffers (``*.weight_quant.scale``,
``*.input_quant.zero_point``, observer running stats) that aren't in the
FP32 ckpt, so a strict load would explode. The patch filters strict→False
when (a) the module has been quantized and (b) the keys missing from the
ckpt are *only* Brevitas quant buffers. Any other missing key still raises.

Mirrors ``src/callbacks/pruning/checkpoint_handler.py`` in spirit.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping

from pytorch_lightning import Callback, LightningModule, Trainer

from src.utils import get_pylogger

log = get_pylogger(__name__)


# Brevitas key infixes — every quantizer proxy lives under one of these
# names on the parent layer. A missing key matching any of these is a
# benign FP32-seed scenario; anything else is a real mismatch.
_BREVITAS_QUANT_INFIXES: tuple[str, ...] = (
    ".weight_quant.",
    ".input_quant.",
    ".output_quant.",
    ".bias_quant.",
)


def _split_quant_keys(keys: Iterable[str]) -> tuple[List[str], List[str]]:
    """Partition ``keys`` into (brevitas_quant_only, everything_else)."""
    quant, other = [], []
    for k in keys:
        if any(infix in k for infix in _BREVITAS_QUANT_INFIXES):
            quant.append(k)
        else:
            other.append(k)
    return quant, other


class QuantizedCheckpointHandler(Callback):
    """Lightning callback for FP32-seeded loads into a quantized module.

    Patches ``pl_module.load_state_dict`` once at setup. When the module
    carries a ``_quantization_spec`` tag and the caller requested
    ``strict=True``, the patched method runs ``strict=False`` *only* so it
    can inspect the diff; any non-Brevitas missing/unexpected key triggers
    a ``RuntimeError``. This means the FP32-seed case works while a real
    architecture mismatch still fails loudly.
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self._patched: bool = False

    def setup(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        if self._patched:
            return

        original_load = pl_module.load_state_dict

        def patched_load(
            state_dict: Mapping[str, Any],
            strict: bool = True,
            *args: Any,
            **kwargs: Any,
        ):
            is_quant = getattr(pl_module, "_quantization_spec", None) is not None
            if not (is_quant and strict):
                return original_load(
                    state_dict, strict=strict, *args, **kwargs
                )

            # Quantized model + caller wants strict: inspect the diff.
            result = original_load(
                state_dict, strict=False, *args, **kwargs
            )
            missing = list(getattr(result, "missing_keys", []) or [])
            unexpected = list(getattr(result, "unexpected_keys", []) or [])

            quant_missing, real_missing = _split_quant_keys(missing)
            quant_unexpected, real_unexpected = _split_quant_keys(unexpected)

            if real_missing or real_unexpected:
                raise RuntimeError(
                    "QuantizedCheckpointHandler: refusing FP32-seed load — "
                    f"{len(real_missing)} non-Brevitas missing key(s) "
                    f"(first 5: {real_missing[:5]}) and "
                    f"{len(real_unexpected)} non-Brevitas unexpected key(s) "
                    f"(first 5: {real_unexpected[:5]}). This is not a valid "
                    "FP32→quant seed; check that the ckpt and module agree."
                )

            log.warning(
                "QuantizedCheckpointHandler: filtered strict=False for "
                f"quantized module — {len(quant_missing)} Brevitas key(s) "
                f"missing from the source ckpt (expected for FP32 seed). "
                f"unexpected={len(quant_unexpected)}."
            )
            return result

        pl_module.load_state_dict = patched_load  # type: ignore[method-assign]
        self._patched = True
