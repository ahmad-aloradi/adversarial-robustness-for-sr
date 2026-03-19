import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import hydra
import pyrootutils
import yaml
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
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
    "config_name": "train.yaml",
}

from src import utils  # noqa: E501
from src.utils.utils import _resolve_test_ckpt_path  # noqa: E501

log = utils.get_pylogger(__name__)

from src.utils.augmentation_utils import prepare_speechbrain_augmentation


def inject_expdir_overrides() -> None:
    """Pre-Hydra hook: if expdir= is provided, resume training from that run.

    Loads .hydra/overrides.yaml from the given experiment directory, injects
    them into sys.argv (user CLI args take precedence), and appends
    ckpt_path pointing to the last checkpoint.  The expdir= arg is removed
    before Hydra sees it.
    """
    exp_dir: Path | None = None
    for arg in sys.argv[1:]:
        if arg.startswith("expdir="):
            exp_dir = Path(arg.split("=", 1)[1]).expanduser().resolve()
            break

    if exp_dir is None:
        return

    if not exp_dir.exists():
        raise FileNotFoundError(f"expdir does not exist: {exp_dir}")

    # Remove expdir= so Hydra doesn't see it as a config key.
    sys.argv = [a for a in sys.argv if not a.startswith("expdir=")]

    # Load saved training overrides.
    overrides_path = exp_dir / ".hydra" / "overrides.yaml"
    if overrides_path.exists():
        with overrides_path.open() as f:
            data = yaml.safe_load(f) or []
        overrides = (
            [str(item) for item in data] if isinstance(data, list) else []
        )
    else:
        log.warning(
            f"No overrides.yaml found at {overrides_path}; continuing without saved overrides."
        )
        overrides = []

    # Inject overrides; explicit CLI args take precedence.
    user_keys = {
        arg.lstrip("+~").split("=")[0] for arg in sys.argv[1:] if "=" in arg
    }
    for ov in overrides:
        key = ov.lstrip("+~").split("=")[0]
        if key not in user_keys:
            sys.argv.append(ov)

    # Append ckpt_path unless the user already specified one.
    if not any(a.startswith("ckpt_path=") for a in sys.argv[1:]):
        last_ckpt = exp_dir / "checkpoints" / "last.ckpt"
        if (exp_dir / "checkpoints").exists():
            if not last_ckpt.exists():
                raise FileNotFoundError(
                    f"Last checkpoint not found: {last_ckpt}"
                )
            sys.argv.append(f"ckpt_path={last_ckpt.as_posix()}")
            log.info(f"Resuming from checkpoint: {last_ckpt}")
        else:
            log.warning(
                f"No checkpoints directory found in {exp_dir} -> resuming without checkpoint."
            )

    # Reuse the existing run directory so Hydra doesn't create a new one.
    if not any(a.startswith("hydra.run.dir=") for a in sys.argv[1:]):
        sys.argv.append(f"hydra.run.dir={exp_dir.as_posix()}")


inject_expdir_overrides()


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best
    weights obtained during training.

    This method is wrapped in optional @task_wrapper decorator which applies
    extra utilities before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated
        objects.
    """

    utils.log_gpu_memory_metadata()

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        log.info(f"Seed everything with <{cfg.seed}>")
        seed_everything(cfg.seed, workers=True)

    # Prepare augmentation assets using SpeechBrain
    prepare_speechbrain_augmentation(cfg)

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

    # Init callbacks
    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(
        cfg.get("callbacks")
    )

    # Init loggers
    log.info("Instantiating loggers...")
    logger: List[PLLogger] = utils.instantiate_loggers(cfg.get("logger"))

    # Init lightning ddp plugins
    log.info("Instantiating plugins...")
    plugins: Optional[List[Any]] = utils.instantiate_plugins(cfg)

    # Init lightning trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger, plugins=plugins
    )

    # Send parameters from cfg to all lightning loggers
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    # Log metadata
    log.info("Logging metadata!")
    utils.log_metadata(cfg)

    # Train the model
    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
        )

    train_metrics = trainer.callback_metrics

    # Test the model
    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = _resolve_test_ckpt_path(trainer)
        if not ckpt_path:
            log.warning(
                "No averaged/best ckpt found! Using current weights for testing..."
            )
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Test ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # Save state dicts for best and last checkpoints
    if cfg.get("save_state_dict"):
        log.info("Starting saving state dicts!")
        utils.save_state_dicts(
            trainer=trainer,
            model=model,
            dirname=cfg.paths.output_dir,
            **cfg.extras.state_dict_saving_params,
        )

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@utils.register_custom_resolvers(**_HYDRA_PARAMS | {"overrides": sys.argv[1:]})
@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> Optional[float]:

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
