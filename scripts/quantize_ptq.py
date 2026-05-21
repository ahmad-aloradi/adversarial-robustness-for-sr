#!/usr/bin/env python3
"""Post-training quantization (PTQ) entry point.

Loads an existing Lightning training run, applies layer substitution from a
``quantization=<spec>`` config, calibrates on a handful of train batches, and
saves the quantized state dict.

Usage::

    python scripts/quantize_ptq.py \\
        exp_dir=/path/to/run \\
        quantization=int8_wa \\
        quantization.calibration.num_batches=32

The script reuses the eval-side machinery for loading the training-time
config and resolving the checkpoint, so the artifact layout matches.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import pyrootutils
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule, seed_everything

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".env", "setup.py", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": str(root / "configs"),
    "config_name": "eval.yaml",
}

# Reuse shared exp_dir preprocessing (the same helpers eval.py uses).
from src import utils  # noqa: E402
from src.utils.exp_dir import (  # noqa: E402
    apply_eval_adjustments,
    ensure_use_avg_ckpt_default,
    prepare_argv_for_exp_dir,
    resolve_checkpoint,
)
from src.quantization import calibrate, quantize_model, resolve_spec  # noqa: E402

log = utils.get_pylogger(__name__)


def _quant_output_dir(exp_dir: Path, spec_name: str) -> Path:
    return exp_dir / "quantized" / spec_name


@utils.task_wrapper
def ptq(cfg: DictConfig):
    """Run the PTQ pipeline end-to-end."""
    if not cfg.get("exp_dir"):
        raise ValueError("ptq requires exp_dir=<training run path>")
    if cfg.get("quantization") in (None, "none"):
        raise ValueError(
            "ptq requires a quantization config (e.g. quantization=int8_wa)"
        )

    ensure_use_avg_ckpt_default(cfg)
    apply_eval_adjustments(cfg)
    exp_dir = Path(str(cfg.exp_dir)).expanduser().resolve()
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.datamodule, _recursive_=False
    )
    datamodule.setup(stage="fit")

    log.info(f"Instantiating model <{cfg.module._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        cfg.module, _recursive_=False
    )

    # Resolve and load the FP32 checkpoint that PTQ starts from.
    cfg.ckpt_path = resolve_checkpoint(cfg)
    log.info(f"Loading FP32 checkpoint: {cfg.ckpt_path}")
    state = torch.load(cfg.ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # strict=True: the FP32 ckpt was saved against the same module we just
    # instantiated; any mismatch is a real config/architecture drift.
    model.load_state_dict(state, strict=True)

    # Substitute layers.
    spec = resolve_spec(cfg.quantization.spec)
    log.info(f"Applying quantization spec: {spec.name}")
    report = quantize_model(model, cfg.quantization)
    log.info(
        f"Substituted {len(report.substituted)} layers, "
        f"skipped {len(report.skipped)}."
    )

    # Run calibration (skipped for weight-only and FP16).
    if spec.quantize_activations:
        # Hard-require the calibration block; a typo or missing key here
        # would otherwise silently fall back to a default num_batches.
        calib_cfg = cfg.quantization.get("calibration")
        if calib_cfg is None or "num_batches" not in calib_cfg:
            raise ValueError(
                f"Activation-quantizing spec '{spec.name}' requires "
                "`quantization.calibration.num_batches` to be set."
            )
        num_batches = int(calib_cfg.num_batches)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader = datamodule.train_dataloader()
        calibrate(
            model=model,
            dataloader=train_loader,
            num_batches=num_batches,
            device=device,
        )
    else:
        log.info(
            f"Skipping calibration: spec '{spec.name}' is "
            "weight-only or FP16."
        )

    # Save quantized state dict + spec metadata.
    out_dir = _quant_output_dir(exp_dir, spec.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state_dict.pth"
    if state_path.exists():
        log.warning(
            f"Overwriting existing PTQ artifact at {state_path}. "
            "Prior state_dict / quantization.yaml / source_ckpt.txt will be replaced."
        )
    torch.save(model.state_dict(), state_path)
    spec_path = out_dir / "quantization.yaml"
    OmegaConf.save(cfg.quantization, spec_path)
    source_path = out_dir / "source_ckpt.txt"
    source_path.write_text(str(cfg.ckpt_path) + "\n")
    log.info(f"Saved quantized state dict to: {state_path}")
    log.info(f"Saved quantization config to: {spec_path}")
    log.info(f"Recorded FP32 source ckpt to: {source_path}")

    return {"ckpt": str(state_path), "spec": spec.name}


# Pre-Hydra hook: parse `exp_dir=` from CLI, swap Hydra's config search
# path to the training snapshot, inject sanitized training overrides.
# Idempotent (no-op if already prepared).
prepare_argv_for_exp_dir(_HYDRA_PARAMS)


@utils.register_custom_resolvers(**_HYDRA_PARAMS | {"overrides": sys.argv[1:]})
@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> None:
    ptq(cfg)


if __name__ == "__main__":
    main()
