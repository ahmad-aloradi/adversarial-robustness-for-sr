import sys
from pathlib import Path
from typing import List, Tuple

import hydra
import pyrootutils
import torch
from omegaconf import DictConfig
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers.logger import Logger as PLLogger

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
from src import utils  # noqa: E402
from src.utils.exp_dir import (  # noqa: E402
    apply_eval_adjustments,
    ensure_use_avg_ckpt_default,
    prepare_argv_for_exp_dir,
    resolve_checkpoint,
)

log = utils.get_pylogger(__name__)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluate a checkpoint on a datamodule's test set."""

    ckpt_path_val = cfg.get("ckpt_path")
    exp_dir_val = cfg.get("exp_dir")

    if not ckpt_path_val and not exp_dir_val:
        raise ValueError(
            "Either 'ckpt_path' or 'exp_dir' must be provided.\n"
            "  ckpt_path: path to a specific checkpoint\n"
            "  exp_dir:   path to a training run directory"
        )

    # Derive exp_dir from ckpt_path if not provided.
    if ckpt_path_val and not exp_dir_val:
        ckpt = Path(str(ckpt_path_val)).expanduser().resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        derived = ckpt.parents[1]
        if not (derived / ".hydra" / "config.yaml").exists():
            raise ValueError(f"No training config at {derived / '.hydra'}")
        cfg.exp_dir = str(derived)
        log.info(f"Derived exp_dir from ckpt_path: {cfg.exp_dir}")

    ensure_use_avg_ckpt_default(cfg)
    apply_eval_adjustments(cfg)
    utils.log_gpu_memory_metadata()

    if cfg.get("seed"):
        log.info(f"Seed: {cfg.seed}")
        seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.datamodule, _recursive_=False
    )

    log.info(f"Instantiating model <{cfg.module._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        cfg.module, _recursive_=False
    )

    log.info("Instantiating loggers...")
    logger: List[PLLogger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    cfg.ckpt_path = resolve_checkpoint(cfg)
    log.info(f"Checkpoint: {cfg.ckpt_path}")

    # If quantization is requested, we must (1) load the FP32 weights into
    # the in-memory model, (2) substitute layers, (3) optionally overlay a
    # pre-quantized state dict, and (4) call trainer.test with ckpt_path=None
    # so Lightning does not reload (and undo) the substitution.
    quant_cfg = cfg.get("quantization")
    quant_active = (
        quant_cfg is not None and quant_cfg.get("spec") not in (None, "none")
    )
    if quant_active:
        from src.quantization import load_quantized_state_dict, quantize_model

        log.info(f"Pre-loading FP32 weights from: {cfg.ckpt_path}")
        fp32_state = torch.load(cfg.ckpt_path, map_location="cpu")
        if isinstance(fp32_state, dict) and "state_dict" in fp32_state:
            fp32_state = fp32_state["state_dict"]
        # strict=True: the FP32 ckpt was saved against the same in-memory
        # Lightning module we just instantiated. Any mismatch is a real bug.
        model.load_state_dict(fp32_state, strict=True)

        log.info(f"Applying quantization spec: {quant_cfg.spec}")
        quantize_model(model, quant_cfg)

        quantized_ckpt = cfg.get("quantized_ckpt_path")
        if quantized_ckpt:
            log.info(f"Overlaying quantized state dict from: {quantized_ckpt}")
            # strict=True: post-substitution the key layout is fixed; the
            # quantized ckpt must match. load_quantized_state_dict also
            # validates spec parity via the sibling quantization.yaml.
            load_quantized_state_dict(
                model, quantized_ckpt, cfg=quant_cfg, strict=True
            )

    if cfg.get("predict"):
        log.info("Starting prediction!")
        predictions = trainer.predict(
            model=model,
            datamodule=datamodule,
            ckpt_path=None if quant_active else cfg.ckpt_path,
        )
        utils.save_predictions(
            predictions=predictions,
            dirname=cfg.paths.output_dir,
            **cfg.extras.predictions_saving_params,
        )
    else:
        log.info("Starting testing!")
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=None if quant_active else cfg.ckpt_path,
        )

    return trainer.callback_metrics, object_dict


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# Pre-Hydra hook: parse `exp_dir=` from CLI, swap Hydra's config search
# path to the training snapshot, inject sanitized training overrides.
# Must run before the decorator stack so custom resolvers see the right
# config_path. The hook is idempotent (no-op on second call).
prepare_argv_for_exp_dir(_HYDRA_PARAMS)


@utils.register_custom_resolvers(**_HYDRA_PARAMS | {"overrides": sys.argv[1:]})
@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
