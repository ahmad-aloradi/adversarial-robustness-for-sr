"""Quantization stack for speaker-recognition models.

Built on top of Brevitas (Xilinx) — supports FP16/INT8/INT4/INT2/INT1, with
both PTQ (post-training quantization) and QAT (quantization-aware training)
across weight-only and weight+activation modes.

Scope: only the neural-network ``encoder`` inside ``EncoderWrapper`` is
quantized. The Fbank front-end, per-utterance normalizer, and (by default)
the classifier head remain in FP32.
"""

from src.quantization.bit_specs import (
    BIT_SPECS,
    BitSpec,
    QuantMode,
    resolve_spec,
)
from src.quantization.calibrate import calibrate
from src.quantization.manager import QuantizationManager
from src.quantization.utils import compute_model_bits, compression_ratio
from src.quantization.wrap import (
    is_quantized,
    load_quantized_state_dict,
    quantize_model,
)

__all__ = [
    "BIT_SPECS",
    "BitSpec",
    "QuantMode",
    "QuantizationManager",
    "calibrate",
    "compression_ratio",
    "compute_model_bits",
    "is_quantized",
    "load_quantized_state_dict",
    "quantize_model",
    "resolve_spec",
]
