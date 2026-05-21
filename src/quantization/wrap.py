"""Top-level entry points: quantize a model, load quantized state dicts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import torch
import yaml
from omegaconf import DictConfig
from torch import nn

from src.quantization.bit_specs import BitSpec, QuantMode, resolve_spec
from src.quantization.manager import QuantizationManager, SubstitutionReport
from src.utils import get_pylogger

log = get_pylogger(__name__)


def _resolve_encoder(model: nn.Module) -> nn.Module:
    """Find the actual neural-network encoder inside a SpeakerVerification /
    Countermeasure LightningModule.

    The framework wraps the encoder in ``EncoderWrapper`` so the front-end
    Fbank / normalizer can be applied uniformly. Quantization only targets
    the inner ``encoder`` submodule.

    Raises
    ------
    AttributeError
        When the model layout has neither an ``audio_encoder.encoder`` path
        nor explicit confirmation that the bare model *is* the encoder.
        Silent fallback would otherwise quantize the front-end Fbank /
        normalizer alongside the encoder — a real correctness hazard.
    """
    if hasattr(model, "audio_encoder") and hasattr(
        model.audio_encoder, "encoder"
    ):
        return model.audio_encoder.encoder
    raise AttributeError(
        "quantize_model: model has no `audio_encoder.encoder` path. "
        "The quantization stack only targets that submodule by design "
        "(Fbank / normalizer must stay FP32). If you really intend to "
        "quantize a bare `nn.Module`, call `QuantizationManager.substitute` "
        "directly instead of `quantize_model`."
    )


def _maybe_resolve_classifier(model: nn.Module) -> Optional[nn.Module]:
    """Return ``model.classifier`` if present, else None."""
    return getattr(model, "classifier", None)


def quantize_model(
    model: nn.Module,
    cfg: Union[BitSpec, DictConfig, dict],
) -> SubstitutionReport:
    """Apply quantization to ``model`` in place.

    Parameters
    ----------
    model : nn.Module
        Typically a ``SpeakerVerification`` or ``Countermeasure`` LightningModule.
        Substitution is applied to ``model.audio_encoder.encoder`` and (optionally)
        ``model.classifier``.
    cfg : BitSpec | DictConfig | dict
        Either a ``BitSpec`` directly, or a config with keys::

            spec: <name in BIT_SPECS, e.g. "int4_w">
            skip_patterns: [<regex>, ...]            # optional
            target_layers: [Conv1d, Conv2d, Linear]  # optional
            quantize_classifier: false               # optional, default False

    Returns
    -------
    SubstitutionReport
        Aggregated report (encoder + classifier).
    """
    spec, skip_patterns, target_layers, quantize_classifier = _parse_cfg(cfg)

    # FP16 is a special case: no layer substitution, just half().
    if spec.is_fp16:
        return _apply_fp16(model)

    encoder = _resolve_encoder(model)
    manager = QuantizationManager(
        spec=spec,
        skip_patterns=skip_patterns,
        target_layers=target_layers,
    )
    report = manager.substitute(encoder)

    if quantize_classifier:
        classifier = _maybe_resolve_classifier(model)
        if classifier is not None:
            cls_report = manager.substitute(classifier)
            report.substituted.extend(cls_report.substituted)
            report.skipped.extend(cls_report.skipped)

    # Tag the model so downstream tooling can detect quantization.
    model._quantization_spec = spec.name  # type: ignore[attr-defined]
    return report


def _parse_cfg(
    cfg: Union[BitSpec, DictConfig, dict],
) -> tuple[BitSpec, Sequence[str], Optional[Sequence[str]], bool]:
    """Normalise the various accepted ``cfg`` shapes."""
    if isinstance(cfg, BitSpec):
        return cfg, (), None, False

    # Allow plain dicts / OmegaConf nodes.
    if "spec" not in cfg:  # type: ignore[operator]
        raise ValueError(
            "Quantization config must contain a 'spec' key naming a BitSpec."
        )
    spec = resolve_spec(str(cfg["spec"]))  # type: ignore[index]

    skip_patterns = list(cfg.get("skip_patterns", []) or [])  # type: ignore[union-attr]
    target_layers_raw = cfg.get("target_layers", None)  # type: ignore[union-attr]
    target_layers = (
        list(target_layers_raw) if target_layers_raw is not None else None
    )
    quantize_classifier = bool(cfg.get("quantize_classifier", False))  # type: ignore[union-attr]
    return spec, skip_patterns, target_layers, quantize_classifier


def _apply_fp16(model: nn.Module) -> SubstitutionReport:
    """Cast model parameters to FP16 (inference-mode).

    Only the encoder (and the classifier, if configured) is cast — the
    front-end Fbank and normalizer stay FP32 because they are deterministic
    signal processors whose numerical precision matters.
    """
    encoder = _resolve_encoder(model)
    encoder.half()
    report = SubstitutionReport()
    report.substituted = [
        name
        for name, p in encoder.named_parameters()
        if p.dtype == torch.float16
    ]
    model._quantization_spec = "fp16"  # type: ignore[attr-defined]
    log.info(f"FP16 quantization: cast {len(report.substituted)} params.")
    return report


def is_quantized(model: nn.Module) -> bool:
    return getattr(model, "_quantization_spec", None) is not None


def load_quantized_state_dict(
    model: nn.Module,
    path: Union[str, Path],
    cfg: Union[BitSpec, DictConfig, dict, None] = None,
    strict: bool = True,
) -> nn.Module:
    """Load a quantized state dict into ``model``.

    If the model has not already been quantized (no ``_quantization_spec``
    attribute) and ``cfg`` is provided, applies quantization first so the
    state dict's quant-parameter keys can be matched.

    Spec parity check
    -----------------
    Brevitas uses the same state-dict key layout for every bit width within
    the same quantizer family, so a `strict=True` load *cannot* detect a
    spec mismatch on its own (e.g. int4_w into an int8_w model). We
    therefore require a sibling ``quantization.yaml`` next to ``path`` —
    that file is written by ``scripts/quantize_ptq.py`` — and compare its
    ``spec`` against the requested ``cfg`` (and the model's runtime tag).
    Any disagreement raises ``ValueError``.
    """
    path = Path(path)

    requested_spec_name = _requested_spec_name(cfg)
    saved_spec_name = _read_saved_spec_name(path)
    _validate_spec_parity(
        path=path,
        requested=requested_spec_name,
        saved=saved_spec_name,
    )

    if not is_quantized(model) and cfg is not None:
        quantize_model(model, cfg)

    # Cross-check the model's runtime tag too — guards against a caller
    # that pre-quantized the model with a different spec than `cfg`.
    runtime_spec = getattr(model, "_quantization_spec", None)
    if runtime_spec is not None and runtime_spec != saved_spec_name:
        raise ValueError(
            "load_quantized_state_dict: spec mismatch between model "
            f"(_quantization_spec={runtime_spec!r}) and saved state dict "
            f"(spec={saved_spec_name!r} from {path.parent / 'quantization.yaml'}). "
            "Re-quantize the model with the correct spec before loading."
        )

    state_dict = torch.load(str(path), map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        log.warning(
            f"load_quantized_state_dict: {len(missing)} missing keys "
            f"(first 5: {missing[:5]})"
        )
    if unexpected:
        log.warning(
            f"load_quantized_state_dict: {len(unexpected)} unexpected keys "
            f"(first 5: {unexpected[:5]})"
        )
    return model


def _requested_spec_name(
    cfg: Union[BitSpec, DictConfig, dict, None],
) -> Optional[str]:
    if cfg is None:
        return None
    if isinstance(cfg, BitSpec):
        return cfg.name
    if "spec" in cfg:  # type: ignore[operator]
        return str(cfg["spec"])  # type: ignore[index]
    return None


def _read_saved_spec_name(state_path: Path) -> Optional[str]:
    """Read ``spec`` from the sibling ``quantization.yaml`` of a PTQ artifact.

    Returns ``None`` if no sibling yaml exists (callers decide whether that
    is acceptable). Raises if the yaml is present but malformed.
    """
    yaml_path = state_path.parent / "quantization.yaml"
    if not yaml_path.exists():
        return None
    with yaml_path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"{yaml_path}: expected a YAML mapping, got "
            f"{type(data).__name__}"
        )
    if "spec" not in data:
        raise ValueError(f"{yaml_path}: missing required 'spec' key.")
    return str(data["spec"])


def _validate_spec_parity(
    *,
    path: Path,
    requested: Optional[str],
    saved: Optional[str],
) -> None:
    if saved is None:
        raise FileNotFoundError(
            f"load_quantized_state_dict: no sibling quantization.yaml next "
            f"to {path}. Spec parity cannot be verified — Brevitas reuses "
            "the same key layout across bit widths, so silent mismatches "
            "would otherwise go undetected. Re-run scripts/quantize_ptq.py "
            "to produce the sidecar, or write quantization.yaml manually."
        )
    if requested is not None and saved != requested:
        raise ValueError(
            f"Quantization spec mismatch: state dict at {path} was saved "
            f"with spec={saved!r}, but the requested cfg.spec={requested!r}. "
            "Brevitas's identical key layout across bit widths makes this "
            "undetectable by strict-load; refusing to proceed."
        )
