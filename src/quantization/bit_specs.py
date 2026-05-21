"""Bit-width specifications for the quantization stack.

Each ``BitSpec`` describes a (bit_width, mode) configuration. The
``resolve_spec`` helper materializes Brevitas quantizer classes lazily —
this lets the module be imported even when brevitas is not installed,
failing only when an actual quantization request is made.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Optional, Type


class QuantMode(str, enum.Enum):
    """Whether to quantize weights only or weights + activations.

    ``FP16`` is a special pseudo-mode that maps to half precision rather than
    integer quantization; it lives here for API uniformity.
    """

    WEIGHT_ONLY = "weight_only"
    WEIGHT_ACT = "weight_act"
    FP16 = "fp16"


@dataclass(frozen=True)
class BitSpec:
    """A bit-width × mode combination.

    Attributes
    ----------
    name : str
        Identifier used in configs (e.g. ``"int4_w"`` or ``"int1_wa"``).
    bit_width : int
        Number of bits per weight (and per activation, if ``mode == WEIGHT_ACT``).
        Use ``16`` for FP16.
    mode : QuantMode
        Weight-only, weight+activation, or FP16.
    """

    name: str
    bit_width: int
    mode: QuantMode

    @property
    def is_fp16(self) -> bool:
        return self.mode is QuantMode.FP16

    @property
    def quantize_activations(self) -> bool:
        return self.mode is QuantMode.WEIGHT_ACT


def _make_int_specs() -> dict[str, BitSpec]:
    """Pre-register all integer bit-width × mode combinations."""
    specs: dict[str, BitSpec] = {}
    for bits in (8, 4, 2, 1):
        specs[f"int{bits}_w"] = BitSpec(
            name=f"int{bits}_w",
            bit_width=bits,
            mode=QuantMode.WEIGHT_ONLY,
        )
        specs[f"int{bits}_wa"] = BitSpec(
            name=f"int{bits}_wa",
            bit_width=bits,
            mode=QuantMode.WEIGHT_ACT,
        )
    specs["fp16"] = BitSpec(name="fp16", bit_width=16, mode=QuantMode.FP16)
    return specs


BIT_SPECS: dict[str, BitSpec] = _make_int_specs()


def resolve_spec(name: str) -> BitSpec:
    """Look up a spec by config name."""
    if name not in BIT_SPECS:
        raise KeyError(
            f"Unknown quantization spec '{name}'. "
            f"Available: {sorted(BIT_SPECS)}"
        )
    return BIT_SPECS[name]


# ---------------------------------------------------------------------------
# Brevitas quantizer factories (lazy: imported only when actually used)
# ---------------------------------------------------------------------------


def _brevitas_modules() -> dict[str, Any]:
    """Import brevitas submodules. Raises ImportError on missing install."""
    try:
        import brevitas.quant as bq
        import brevitas.quant.binary as bq_bin
        import brevitas.quant.ternary as bq_tern
    except ImportError as exc:  # pragma: no cover - exercised at runtime only
        raise ImportError(
            "brevitas is required for quantization. Install with "
            "`pip install brevitas==0.11.0`."
        ) from exc
    return {"q": bq, "bin": bq_bin, "tern": bq_tern}


def weight_quantizer(spec: BitSpec) -> Optional[Type[Any]]:
    """Return the Brevitas weight quantizer class for ``spec``.

    Returns ``None`` for FP16 (no Brevitas quantizer needed; ``.half()`` is
    used at the wrap level instead).
    """
    if spec.is_fp16:
        return None

    mods = _brevitas_modules()
    bq = mods["q"]
    bq_bin = mods["bin"]
    bq_tern = mods["tern"]

    if spec.bit_width == 8:
        return bq.Int8WeightPerTensorFloat

    if spec.bit_width == 4:
        # Subclass to override bit_width — Brevitas reads the class attribute.
        return type(
            "Int4WeightPerTensorFloat",
            (bq.Int8WeightPerTensorFloat,),
            {"bit_width": 4},
        )

    if spec.bit_width == 2:
        return bq_tern.SignedTernaryWeightPerTensorConst

    if spec.bit_width == 1:
        return bq_bin.SignedBinaryWeightPerTensorConst

    raise ValueError(f"Unsupported bit_width: {spec.bit_width}")


def act_quantizer(spec: BitSpec) -> Optional[Type[Any]]:
    """Return the Brevitas activation quantizer class for ``spec``.

    Returns ``None`` for weight-only and FP16 modes (no activation quant).
    """
    if not spec.quantize_activations:
        return None

    mods = _brevitas_modules()
    bq = mods["q"]

    if spec.bit_width == 8:
        return bq.Int8ActPerTensorFloat

    if spec.bit_width == 4:
        return type(
            "Int4ActPerTensorFloat",
            (bq.Int8ActPerTensorFloat,),
            {"bit_width": 4},
        )

    if spec.bit_width == 2:
        return type(
            "Int2ActPerTensorFloat",
            (bq.Int8ActPerTensorFloat,),
            {"bit_width": 2},
        )

    if spec.bit_width == 1:
        # Brevitas exposes a signed binary activation; we keep symmetric int1
        # for parity with weights.
        from brevitas.quant.binary import SignedBinaryActPerTensorConst

        return SignedBinaryActPerTensorConst

    raise ValueError(f"Unsupported bit_width: {spec.bit_width}")
