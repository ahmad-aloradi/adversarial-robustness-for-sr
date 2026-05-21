"""Layer-substitution manager.

Walks an ``nn.Module`` and replaces leaf ``Conv1d``/``Conv2d``/``Linear``
layers with their Brevitas counterparts, while respecting ``skip_patterns``
(regex on the qualified module name) and an optional ``target_layers`` filter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn

from src.quantization.bit_specs import BitSpec, act_quantizer, weight_quantizer
from src.utils import get_pylogger

log = get_pylogger(__name__)


# Layer classes that the manager knows how to substitute.
_SUPPORTED_LAYERS: tuple[type[nn.Module], ...] = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Linear,
)


@dataclass
class SubstitutionReport:
    """Records what was (and wasn't) substituted, for logging and tests."""

    substituted: List[str] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)  # (name, reason)

    def __len__(self) -> int:
        return len(self.substituted)


class QuantizationManager:
    """Substitutes torch layers with Brevitas quantized layers.

    Parameters
    ----------
    spec : BitSpec
        Bit-width × mode configuration to apply.
    skip_patterns : sequence of regex strings
        Module qualified names matching any pattern are left as FP32. Useful
        for first/last layer protection or for skipping non-traceable ops
        (SincConv in RawNet, graph attention in AASIST).
    target_layers : sequence of class names, optional
        If set, only these layer classes are substituted (e.g.
        ``["Conv1d", "Linear"]``). Defaults to all of Conv1d/Conv2d/Linear.
    """

    def __init__(
        self,
        spec: BitSpec,
        skip_patterns: Sequence[str] = (),
        target_layers: Optional[Sequence[str]] = None,
    ) -> None:
        self.spec = spec
        self._skip_regexes = [re.compile(p) for p in skip_patterns]
        self._target_layers = (
            tuple(target_layers) if target_layers is not None else None
        )

    # ---- Public API ------------------------------------------------------

    def substitute(self, module: nn.Module) -> SubstitutionReport:
        """In-place substitute layers inside ``module``.

        Walks the module tree, builds a list of (parent, attr_name, child)
        triples for matching leaves, then performs ``setattr`` swaps.
        """
        if self.spec.is_fp16:
            raise ValueError(
                "FP16 spec does not use layer substitution; call "
                "quantize_model() which dispatches to .half() instead."
            )

        report = SubstitutionReport()
        replacements: list[tuple[nn.Module, str, str, nn.Module]] = []

        for name, child in module.named_modules():
            parent, attr_name = self._parent_and_attr(module, name)
            if parent is None:
                continue  # the module itself
            if not isinstance(child, _SUPPORTED_LAYERS):
                continue
            if self._target_layers is not None and (
                type(child).__name__ not in self._target_layers
            ):
                report.skipped.append((name, "type-filter"))
                continue
            if self._matches_skip(name):
                report.skipped.append((name, "skip-pattern"))
                continue

            new_layer = self._build_quant_layer(child)
            replacements.append((parent, attr_name, name, new_layer))

        for parent, attr_name, qualname, new_layer in replacements:
            setattr(parent, attr_name, new_layer)
            report.substituted.append(qualname)

        log.info(
            f"QuantizationManager[{self.spec.name}]: "
            f"substituted {len(report.substituted)} layers, "
            f"skipped {len(report.skipped)}."
        )
        return report

    # ---- Internals -------------------------------------------------------

    @staticmethod
    def _parent_and_attr(
        root: nn.Module, qualname: str
    ) -> tuple[Optional[nn.Module], str]:
        """Return (parent_module, attr_name) for ``root.<qualname>``.

        For the root itself, returns ``(None, "")``.
        """
        if qualname == "":
            return None, ""
        parts = qualname.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        return parent, parts[-1]

    def _matches_skip(self, qualname: str) -> bool:
        return any(rx.search(qualname) for rx in self._skip_regexes)

    def _build_quant_layer(self, layer: nn.Module) -> nn.Module:
        """Construct a Brevitas equivalent of ``layer`` and copy weights."""
        try:
            import brevitas.nn as qnn
        except ImportError as exc:  # pragma: no cover - runtime guard only
            raise ImportError(
                "brevitas is required for layer substitution. "
                "Install with `pip install brevitas==0.11.0`."
            ) from exc

        w_q = weight_quantizer(self.spec)
        a_q = act_quantizer(self.spec)

        if isinstance(layer, nn.Conv1d):
            new_layer = qnn.QuantConv1d(
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=layer.groups,
                bias=layer.bias is not None,
                padding_mode=layer.padding_mode,
                weight_quant=w_q,
                input_quant=a_q,
                output_quant=None,
                return_quant_tensor=False,
            )
        elif isinstance(layer, nn.Conv2d):
            new_layer = qnn.QuantConv2d(
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=layer.groups,
                bias=layer.bias is not None,
                padding_mode=layer.padding_mode,
                weight_quant=w_q,
                input_quant=a_q,
                output_quant=None,
                return_quant_tensor=False,
            )
        elif isinstance(layer, nn.Linear):
            new_layer = qnn.QuantLinear(
                in_features=layer.in_features,
                out_features=layer.out_features,
                bias=layer.bias is not None,
                weight_quant=w_q,
                input_quant=a_q,
                output_quant=None,
                return_quant_tensor=False,
            )
        else:
            raise TypeError(
                f"Unsupported layer type for substitution: {type(layer)}"
            )

        # Copy pretrained weights so PTQ starts from the FP32 minimum and QAT
        # can fine-tune from the pretrained init.
        with torch.no_grad():
            new_layer.weight.copy_(layer.weight)
            if layer.bias is not None and new_layer.bias is not None:
                new_layer.bias.copy_(layer.bias)

        return new_layer

    # ---- Convenience -----------------------------------------------------

    def count_quant_layers(self, module: nn.Module) -> int:
        """Count Brevitas quant layers currently in ``module`` (post-sub).

        Brevitas is required by every code path that constructs a
        ``QuantizationManager``; we re-import here only to access the layer
        types. Missing-brevitas should never be silently absorbed into a
        zero count — that hides "quantization didn't actually happen" bugs.
        """
        import brevitas.nn as qnn

        quant_types = (qnn.QuantConv1d, qnn.QuantConv2d, qnn.QuantLinear)
        return sum(1 for m in module.modules() if isinstance(m, quant_types))


def iter_supported_layers(
    module: nn.Module,
) -> Iterable[tuple[str, nn.Module]]:
    """Yield ``(qualname, layer)`` for every supported leaf layer."""
    for name, child in module.named_modules():
        if isinstance(child, _SUPPORTED_LAYERS):
            yield name, child
