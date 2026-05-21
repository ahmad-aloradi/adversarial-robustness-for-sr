"""Quantization-aware training Lightning callback.

Substitutes target layers in the encoder with Brevitas quantized layers at
``setup`` time, then lets Lightning handle the training loop as usual. The
quantizer observers update during ``train``/``val`` and lock during ``test``.

Mirrors the structure of ``src/callbacks/pruning/prune.py:MagnitudePruner``
so the two compression stacks feel symmetric.
"""

from __future__ import annotations

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
        if self._applied:
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
        # Apply substitution before Lightning loads the state dict so the
        # quantizer-internal buffers in the checkpoint have matching targets.
        state = checkpoint.get("qat_callback_state")
        if state is None:
            return
        if is_quantized(pl_module):
            return
        quantize_model(
            pl_module,
            cfg={
                "spec": state["spec"],
                "skip_patterns": state["skip_patterns"],
                "target_layers": state["target_layers"],
                "quantize_classifier": state["quantize_classifier"],
            },
        )
        self._applied = True
        self._substituted = list(state.get("substituted", []))
