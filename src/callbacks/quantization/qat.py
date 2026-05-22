"""Quantization-aware training Lightning callback.

Substitutes target layers in the encoder with Brevitas quantized layers at
``setup`` time, then lets Lightning handle the training loop as usual. The
quantizer observers update during ``train``/``val`` and lock during ``test``.

Mirrors the structure of ``src/callbacks/pruning/prune.py:MagnitudePruner``
so the two compression stacks feel symmetric.
"""

from __future__ import annotations

if __name__ == "__main__":
    # Make `src.*` imports below resolve when running this file directly
    # (`python src/callbacks/quantization/qat.py`) for the smoke block at EOF.
    import pyrootutils

    pyrootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from typing import Any, Dict, List, Optional, Sequence

import torch
from pytorch_lightning import Callback, LightningModule, Trainer

from src.quantization.bit_specs import resolve_spec
from src.quantization.wrap import is_quantized, quantize_model
from src.utils import get_pylogger

log = get_pylogger(__name__)


class QATCallback(Callback):
    """Wrap the module in Brevitas quant layers before training begins.

    Parameters
    ----------
    spec : str
        Name of a registered ``BitSpec`` (e.g. ``"int4_w"``, ``"int1_wa"``).
    skip_patterns : sequence of regex strings
        Module qualified names matching any pattern stay FP32. Useful for
        first/last layer protection, or to skip non-traceable ops (SincConv,
        graph attention).
    target_layers : sequence of str, optional
        Restrict substitution to these layer class names. Defaults to all of
        ``Conv1d``/``Conv2d``/``Linear``.
    quantize_classifier : bool
        If True, the classifier head is also wrapped.
    pretrained_ckpt_path : str, optional
        If given, load this FP32 state dict into the module *before*
        substitution — seeds QAT from a pretrained baseline.
    verbose : bool
        Whether to log per-layer substitution decisions.
    """

    def __init__(
        self,
        spec: str,
        skip_patterns: Sequence[str] = (),
        target_layers: Optional[Sequence[str]] = None,
        quantize_classifier: bool = False,
        pretrained_ckpt_path: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        self.spec_name = spec
        self.spec = resolve_spec(spec)
        self.skip_patterns = list(skip_patterns)
        self.target_layers = (
            list(target_layers) if target_layers is not None else None
        )
        self.quantize_classifier = quantize_classifier
        self.pretrained_ckpt_path = pretrained_ckpt_path
        self.verbose = verbose

        # State populated at setup() time.
        self._applied: bool = False
        self._substituted: List[str] = []

    # ---- Lightning hooks -------------------------------------------------

    def setup(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        log.info(
            f"QATCallback init: spec={self.spec_name}, "
            f"target_layers="
            f"{self.target_layers or 'all (Conv1d/Conv2d/Linear)'}, "
            f"skip_patterns={self.skip_patterns}, "
            f"quantize_classifier={self.quantize_classifier}, "
            f"pretrained_ckpt={self.pretrained_ckpt_path or 'none'}, "
            f"stage={stage}"
        )

        if self._applied:
            log.info(
                f"QATCallback[{self.spec_name}]: already applied "
                "(setup() re-entered, e.g. fit→validate); skipping."
            )
            return
        if is_quantized(pl_module):
            log.info(
                f"QATCallback[{self.spec_name}]: module already quantized; "
                "skipping substitution."
            )
            self._applied = True
            return

        if self.pretrained_ckpt_path:
            log.info(
                f"QATCallback[{self.spec_name}]: seeding from FP32 ckpt "
                f"{self.pretrained_ckpt_path}"
            )
            state = torch.load(self.pretrained_ckpt_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            # strict=True: substitution has not happened yet (it runs below
            # in `quantize_model`), so the module is still vanilla FP32 and
            # the ckpt's keys must match exactly. Any mismatch indicates a
            # real architecture drift between the FP32 baseline and the
            # current config — refusing to start QAT here is correct.
            pl_module.load_state_dict(state, strict=True)

        report = quantize_model(
            pl_module,
            cfg={
                "spec": self.spec_name,
                "skip_patterns": self.skip_patterns,
                "target_layers": self.target_layers,
                "quantize_classifier": self.quantize_classifier,
            },
        )
        self._substituted = list(report.substituted)
        self._applied = True
        log.info(
            f"QATCallback[{self.spec_name}]: substituted "
            f"{len(self._substituted)} layers."
        )

    def on_save_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        checkpoint["qat_callback_state"] = {
            "spec": self.spec_name,
            "skip_patterns": self.skip_patterns,
            "target_layers": self.target_layers,
            "quantize_classifier": self.quantize_classifier,
            "substituted": list(self._substituted),
        }

    def on_load_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        """Restore QAT substitution before Lightning loads the state dict."""
        state = checkpoint.get("qat_callback_state")
        if state is None:
            if self._applied:
                raise RuntimeError(
                    f"QATCallback[{self.spec_name}]: resumed checkpoint "
                    "lacks `qat_callback_state`. The callback already "
                    "substituted layers at setup-time, so reloading FP32 "
                    "weights here would produce a half-quantized model. "
                    "Either resume from a QAT-trained ckpt, or seed from "
                    "FP32 via `pretrained_ckpt_path=` and start a fresh run."
                )
            log.info(
                f"QATCallback[{self.spec_name}] resume: checkpoint has no "
                "`qat_callback_state` and setup() hasn't run — treating as "
                "fresh FP32 load."
            )
            return

        saved_spec = state["spec"]
        if saved_spec != self.spec_name:
            raise ValueError(
                f"QATCallback spec mismatch: this callback is configured "
                f"with spec={self.spec_name!r}, but the resumed checkpoint "
                f"was trained with spec={saved_spec!r}. Reconfigure the "
                "callback to match the checkpoint (or retrain from scratch)."
            )

        if is_quantized(pl_module):
            # `setup` already substituted with the matching spec — nothing
            # more to do; the upcoming state-dict load will overlay weights.
            log.info(
                f"QATCallback[{self.spec_name}] resume: module already "
                "quantized with matching spec; Lightning will overlay "
                f"weights ({len(state.get('substituted', []))} layers in "
                "saved state)."
            )
            return

        # Fresh module (no setup): substitute using the saved cfg so the
        # state dict's quant-buffer keys have matching targets.
        log.info(
            f"QATCallback[{self.spec_name}] resume: re-substituting from "
            f"saved cfg (substituted={len(state.get('substituted', []))} "
            "layers in saved state)."
        )
        quantize_model(
            pl_module,
            cfg={
                "spec": saved_spec,
                "skip_patterns": state["skip_patterns"],
                "target_layers": state["target_layers"],
                "quantize_classifier": state["quantize_classifier"],
            },
        )
        self._applied = True
        self._substituted = list(state.get("substituted", []))


if __name__ == "__main__":
    # Smoke: instantiate QATCallback and run setup() on a tiny module that
    # exposes the `audio_encoder.encoder` path quantize_model expects.
    from torch import nn as _nn

    class _AE(_nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = _nn.Sequential(_nn.Conv1d(4, 8, 3, padding=1))

    class _LM(_nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.audio_encoder = _AE()

    cb = QATCallback("int8_w")
    cb.setup(trainer=None, pl_module=_LM(), stage="fit")
    assert cb._applied and cb._substituted, "setup did not apply substitution"
    print(f"smoke ok: _applied={cb._applied}, substituted={cb._substituted}")
