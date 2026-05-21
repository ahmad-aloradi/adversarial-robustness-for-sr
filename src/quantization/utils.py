"""Bit-budget accounting and reporting helpers."""

from __future__ import annotations

from typing import Optional

from torch import nn

from src.quantization.bit_specs import BitSpec, QuantMode


def _dtype_bits(dtype_str: str) -> int:
    return {
        "torch.float32": 32,
        "torch.float16": 16,
        "torch.bfloat16": 16,
        "torch.int8": 8,
        "torch.int32": 32,
    }.get(dtype_str, 32)


def compute_model_bits(
    model: nn.Module, spec: Optional[BitSpec] = None
) -> dict[str, int]:
    """Estimate the bit footprint of a model's weights.

    If ``spec`` is given, all weight tensors inside Brevitas quant layers are
    counted at ``spec.bit_width``; everything else is counted by its actual
    ``dtype``. If ``spec`` is None, falls back to per-tensor dtype counting
    (useful for the FP32 baseline).

    Returns a dict with::

        {"weights_bits": int, "buffers_bits": int, "total_bits": int}
    """
    # Brevitas is required by every path that calls `compute_model_bits`
    # with a spec (the model itself was substituted by Brevitas). Letting a
    # silent ImportError collapse `quant_layer_types` to () would count INT
    # weights as FP32 and silently produce wrong compression ratios.
    import brevitas.nn as qnn

    quant_layer_types: tuple[type, ...] = (
        qnn.QuantConv1d,
        qnn.QuantConv2d,
        qnn.QuantLinear,
    )

    weights_bits = 0
    buffers_bits = 0

    quant_layer_ids = {
        id(m)
        for m in model.modules()
        if isinstance(m, quant_layer_types)
    }

    # Collect IDs of every *child* module nested inside a quant layer
    # (i.e., quantizer proxies like weight_quant, act_quant, tensor_quant,
    # scaling_impl, etc.).  Their parameters are quantizer metadata — scale /
    # zero_point tensors — not model weights, and must not be counted here.
    quant_child_ids: set[int] = set()
    for module in model.modules():
        if id(module) in quant_layer_ids:
            for child in module.modules():
                if child is not module:
                    quant_child_ids.add(id(child))

    # Walk parameters with their owning modules so we can tell whether the
    # parameter is inside a quant layer.
    for module in model.modules():
        if id(module) in quant_child_ids:
            continue  # skip quantizer proxy sub-modules
        is_quant = id(module) in quant_layer_ids
        # ``recurse=False`` so we don't double-count nested children.
        for name, param in module.named_parameters(recurse=False):
            numel = param.numel()
            if (
                is_quant
                and name == "weight"
                and spec is not None
                and not spec.is_fp16
            ):
                weights_bits += numel * spec.bit_width
            else:
                weights_bits += numel * _dtype_bits(str(param.dtype))

    for buf in model.buffers():
        buffers_bits += buf.numel() * _dtype_bits(str(buf.dtype))

    return {
        "weights_bits": weights_bits,
        "buffers_bits": buffers_bits,
        "total_bits": weights_bits + buffers_bits,
    }


def compression_ratio(
    fp32_model: nn.Module,
    quant_model: nn.Module,
    spec: BitSpec,
) -> float:
    """Ratio of FP32 total bits over quantized total bits.

    Larger means more compression.
    """
    fp32 = compute_model_bits(fp32_model, spec=None)["total_bits"]
    quant = compute_model_bits(quant_model, spec=spec)["total_bits"]
    if quant == 0:
        return float("inf")
    return fp32 / quant


def describe_spec(spec: BitSpec) -> str:
    """Human-readable description used in logs and reports."""
    if spec.is_fp16:
        return "FP16 (half precision)"
    mode = "weight+activation" if spec.mode is QuantMode.WEIGHT_ACT else "weight-only"
    return f"INT{spec.bit_width} ({mode})"
