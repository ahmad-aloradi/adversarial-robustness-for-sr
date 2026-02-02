import sys
from pathlib import Path
from typing import List, Tuple

import hydra
import pyrootutils
import yaml
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers.logger import Logger as PLLogger

# --------------------------------------------------------------------------- #
# `pyrootutils.setup_root(...)` above is optional line to make environment more
# convenient should be placed at the top of each entry file
#
# main advantages:
# - allows you to keep all entry files in "src/" without installing project as
#   a package
# - launching python file works no matter where is your current work dir
# - automatically loads environment variables from ".env" if exists
#
# how it works:
# - `setup_root()` above recursively searches for either ".git" or
#   "pyproject.toml" in present and parent dirs, to determine the project root
#   dir
# - adds root dir to the PYTHONPATH (if `pythonpath=True`), so this file can
#   be run from any place without installing project as a package
# - sets PROJECT_ROOT environment variable which is used in
#   "configs/paths/default.yaml" to make all paths always relative to project
#   root
# - loads environment variables from ".env" in root dir (if `dotenv=True`)
#
# you can remove `pyrootutils.setup_root(...)` if you:
# 1. either install project as a package or move each entry file to the project
#    root dir
# 2. remove PROJECT_ROOT variable from paths in "configs/paths/default.yaml"
#
# https://github.com/ashleve/pyrootutils
# --------------------------------------------------------------------------- #


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
from src import utils  # noqa: E501

log = utils.get_pylogger(__name__)


def _apply_exp_dir_config(cfg: DictConfig) -> DictConfig:
    exp_dir_value = cfg.get("exp_dir")
    if not exp_dir_value:
        return cfg

    exp_dir = Path(str(exp_dir_value)).expanduser().resolve()
    exp_cfg_path = exp_dir / ".hydra" / "config.yaml"
    if not exp_cfg_path.exists():
        raise FileNotFoundError(f"Missing experiment config: {exp_cfg_path}")

    exp_cfg = OmegaConf.load(exp_cfg_path)

    # Preserve eval-specific settings
    for key in [
        "predict",
        "use_avg_ckpt",
        "ckpt_path",
        "ckpt_dir",
        "ckpt_avg_num",
        "ckpt_avg_min",
        "extras",
        "paths",
        "tags",
    ]:
        if cfg.get(key) is not None:
            OmegaConf.update(exp_cfg, key, cfg.get(key), merge=True)

    # Preserve any CLI overrides (e.g., batch_size, num_workers)
    for arg in sys.argv[1:]:
        if "=" not in arg or arg.startswith("exp_dir="):
            continue
        key = arg.lstrip("+~").split("=")[0]
        # Use OmegaConf.select to get nested keys like "datamodule.loaders.train.batch_size"
        value = OmegaConf.select(cfg, key)
        if value is not None:
            OmegaConf.update(exp_cfg, key, value, merge=True)

    exp_cfg.task_name = "eval"

    if not cfg.get("keep_logger"):
        exp_cfg.logger = None

    if exp_cfg.get("use_avg_ckpt") and not exp_cfg.get("ckpt_dir"):
        exp_cfg.ckpt_dir = str(exp_dir / "checkpoints")

    artifacts_dirs = sorted(exp_dir.glob("*_artifacts"))
    if artifacts_dirs:
        OmegaConf.update(
            exp_cfg,
            "datamodule.dataset.artifacts_dir",
            artifacts_dirs[0].as_posix(),
            merge=True,
        )

    noise_csv = exp_dir / "noise.csv"
    if noise_csv.exists():
        OmegaConf.update(
            exp_cfg,
            "module.data_augmentation.noise_annotation",
            noise_csv.as_posix(),
            merge=True,
        )

    reverb_csv = exp_dir / "reverb.csv"
    if reverb_csv.exists():
        OmegaConf.update(
            exp_cfg,
            "module.data_augmentation.rir_annotation",
            reverb_csv.as_posix(),
            merge=True,
        )

    return exp_cfg


def _resolve_checkpoint(cfg: DictConfig) -> str:
    """Resolve checkpoint path from config.

    Supports two modes:
    1. use_avg_ckpt=True: Average top N checkpoints from ckpt_dir
       - Reuses existing averaged checkpoint if available
    2. use_avg_ckpt=False: Use explicit ckpt_path
    """
    if cfg.get("use_avg_ckpt"):
        return _resolve_averaged_checkpoint(cfg)

    ckpt_path = cfg.get("ckpt_path")
    if not ckpt_path:
        raise ValueError("ckpt_path is required when use_avg_ckpt is False")

    ckpt_path = Path(str(ckpt_path)).expanduser()
    if not ckpt_path.exists():
        raise ValueError(f"Checkpoint not found: {ckpt_path}")

    # Temporary safeguard against using last.ckpt 
    # -> later: delete this field since it is inherited from resming exps in training
    if 'last' in str(ckpt_path):
        raise ValueError(f"Invalid checkpoint file: {ckpt_path}")
    
    return str(ckpt_path)


def _extract_metric_from_checkpoint(ckpt_path: Path) -> float | None:
    """Extract the metric value from a checkpoint filename.

    Expected format: epochxxx-loss_validx.xxx-metric_validx.xxx.ckpt
    Returns the metric_valid value, or None if parsing fails.
    """
    import re

    match = re.search(r"metric_valid([0-9]+(?:\.[0-9]+)?)", ckpt_path.name)
    if match:
        return float(match.group(1))
    return None


def _resolve_averaged_checkpoint(cfg: DictConfig) -> str:
    """Average top N checkpoints or reuse existing averaged checkpoint."""
    ckpt_dir = cfg.get("ckpt_dir")
    if not ckpt_dir:
        raise ValueError("use_avg_ckpt=True requires ckpt_dir to be set")

    ckpt_dir = Path(str(ckpt_dir)).expanduser()
    if not ckpt_dir.exists():
        raise ValueError(f"Checkpoint directory not found: {ckpt_dir}")

    avg_num = cfg.get("ckpt_avg_num", 10)
    avg_min = cfg.get("ckpt_avg_min", 2)

    # Check for existing averaged checkpoint
    existing_avg = ckpt_dir / f"averaged_top{avg_num}.ckpt"
    if existing_avg.exists():
        log.info(f"Reusing existing averaged checkpoint: {existing_avg}")
        return str(existing_avg)

    else:
        # Find candidate checkpoints (exclude averaged and last.ckpt)
        log.warning(
            f"No averaged checkpoint found at {ckpt_dir}. Creating a new averaged ckpt."
        )
        all_candidates = [
            p
            for p in ckpt_dir.glob("*.ckpt")
            if not p.name.startswith("averaged_") and p.name != "last.ckpt"
        ]

        # Determine checkpoint mode from config (max = higher is better, min = lower is better)
        # Default to "max" for metric (accuracy-like metrics)
        assert cfg.callbacks.get("model_checkpoint") is not None
        ckpt_mode = cfg.callbacks.model_checkpoint.get("mode")
        assert ckpt_mode in ["min", "max"], "model_checkpoint.mode must be 'min' or 'max'"
        
        # Try to sort by metric value extracted from filename
        candidates_with_metric = [
            (p, _extract_metric_from_checkpoint(p)) for p in all_candidates
        ]
        candidates_with_valid_metric = [
            (p, m) for p, m in candidates_with_metric if m is not None
        ]

        if candidates_with_valid_metric:
            # Sort by metric value: descending for "max" mode, ascending for "min" mode
            reverse_sort = ckpt_mode == "max"
            candidates_with_valid_metric.sort(key=lambda x: x[1], reverse=reverse_sort)
            candidates = [p for p, _ in candidates_with_valid_metric]
            log.info(
                f"Sorting checkpoints by metric value (mode={ckpt_mode}), "
                f"best metric: {candidates_with_valid_metric[0][1]}"
            )
        else:
            # Fallback to modification time if no metric values can be extracted
            log.warning(
                "Could not extract metric values from checkpoint filenames, "
                "falling back to modification time"
            )
            candidates = sorted(
                all_candidates,
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )

        if len(candidates) < avg_min:
            raise ValueError(
                f"Not enough checkpoints to average: found {len(candidates)}, "
                f"need at least {avg_min}"
            )

        candidates = candidates[:avg_num]
        output_path = ckpt_dir / f"averaged_top{len(candidates)}.ckpt"

        log.info(f"Averaging {len(candidates)} checkpoints: {[p.name for p in candidates]}")
        averaged = utils.average_checkpoints(candidates, output_path)

        if not averaged:
            raise ValueError("Failed to create averaged checkpoint")

        return str(averaged)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies
    extra utilities before and after the call.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated
        objects.
    """

    cfg = _apply_exp_dir_config(cfg)

    utils.log_gpu_memory_metadata()

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        log.info(f"Seed everything with <{cfg.seed}>")
        seed_everything(cfg.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.datamodule, _recursive_=False
    )

    # Init lightning model
    log.info(f"Instantiating lightning model <{cfg.module._target_}>")
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

    cfg.ckpt_path = _resolve_checkpoint(cfg)
    log.info(f"Test ckpt path: {cfg.ckpt_path}")

    if cfg.get("predict"):
        log.info("Starting predicting!")
        predictions = trainer.predict(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.ckpt_path,
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
            ckpt_path=cfg.ckpt_path,
        )

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


def _inject_exp_dir_overrides() -> None:
    """Inject a training run's Hydra overrides into sys.argv so that
    configuration composition (done during resolver registration) sees the same
    overrides the run used.

    This prevents missing/interpolation errors
    when composing `eval.yaml` together with the experiment overrides.
    """
    exp_arg = None
    for arg in sys.argv[1:]:
        if arg.startswith("exp_dir="):
            exp_arg = arg
            break

    if not exp_arg:
        return

    exp_dir = Path(exp_arg.split("=", 1)[1]).expanduser().resolve()
    overrides_path = exp_dir / ".hydra" / "overrides.yaml"

    if not overrides_path.exists():
        return

    with overrides_path.open("r") as handle:
        data = yaml.safe_load(handle) or []
    if not isinstance(data, list):
        return

    # If keep_logger was explicitly passed on CLI, keep logger overrides
    keep_logger = any(
        (
            a.startswith("keep_logger=")
            and a.split("=", 1)[1].lower() in ["1", "true"]
        )
        for a in sys.argv[1:]
    )

    new_overrides = []
    for item in data:
        s = str(item)
        if not keep_logger and (
            s.startswith("logger=") or s.startswith("logger.")
        ):
            continue
        if s.startswith("experiment="):
            new_overrides.append("+" + s)
        else:
            new_overrides.append(s)

    artifacts_dirs = sorted(exp_dir.glob("*_artifacts"))
    if artifacts_dirs:
        new_overrides.append(
            f"datamodule.dataset.artifacts_dir={artifacts_dirs[0].as_posix()}"
        )

    noise_csv = exp_dir / "noise.csv"
    if noise_csv.exists():
        new_overrides.append(
            f"module.data_augmentation.noise_annotation={noise_csv.as_posix()}"
        )

    reverb_csv = exp_dir / "reverb.csv"
    if reverb_csv.exists():
        new_overrides.append(
            f"module.data_augmentation.rir_annotation={reverb_csv.as_posix()}"
        )

    for ov in new_overrides:
        # Extract key from override (strip leading +/~ and get part before =)
        key = ov.lstrip("+~").split("=")[0]
        # Check if user already specified this key on CLI
        user_has_key = any(
            arg.lstrip("+~").split("=")[0] == key
            for arg in sys.argv[1:]
            if "=" in arg
        )
        if not user_has_key:
            sys.argv.append(ov)


_inject_exp_dir_overrides()


@utils.register_custom_resolvers(**_HYDRA_PARAMS | {"overrides": sys.argv[1:]})
@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
