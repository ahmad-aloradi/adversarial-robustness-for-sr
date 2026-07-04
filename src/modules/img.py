from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from src import utils
from src.callbacks.pruning.utils.pruning_manager import PruningManager

log = utils.get_pylogger(__name__)


class ImageClassification(pl.LightningModule):
    """Image classification for validating the Bregman pruning stack on
    standard benchmarks (CIFAR-10/100, MNIST, TinyImageNet) with ResNet-18.

    A plain per-image classifier: each batch is a torchvision
    ``(images, targets)`` tuple, scored with CrossEntropyLoss and top-1
    accuracy. Pruning/Bregman is orthogonal — it is wired only through
    ``configure_optimizers`` (identical to ``sv.py``), so the same
    ``pruning_groups`` config drives both tasks.
    """

    def __init__(
        self,
        model: DictConfig,
        criterion: DictConfig,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        logging_params: DictConfig,
        metrics: DictConfig,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.logging_params = logging_params

        self._setup_metrics(metrics)
        self._setup_model_components(model)
        self._setup_training_components(criterion, optimizer, lr_scheduler)

    # ------------------------------------------------------------------ #
    #  Setup helpers                                                       #
    # ------------------------------------------------------------------ #

    def _setup_metrics(self, metrics: DictConfig) -> None:
        self.train_metric = instantiate(metrics.train)
        self.valid_metric = instantiate(metrics.valid)
        self.test_metric = instantiate(metrics.test)
        self.valid_metric_best = instantiate(metrics.valid_best)

    def _setup_model_components(self, model: DictConfig) -> None:
        self.net = instantiate(model.net)
        self.in_channels = model.net.in_channels

    def _setup_training_components(
        self,
        criterion: DictConfig,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
    ) -> None:
        self.train_criterion = instantiate(criterion.train_criterion)
        self.optimizer = optimizer
        self.slr_params = lr_scheduler

    # ------------------------------------------------------------------ #
    #  Forward / model step                                                #
    # ------------------------------------------------------------------ #

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert images.ndim == 4 and images.shape[1] == self.in_channels, (
            f"expected (B,{self.in_channels},H,W) images, "
            f"got {tuple(images.shape)}"
        )
        return self.net(images)

    def model_step(self, batch, criterion: Any) -> Dict[str, Any]:
        images, targets = batch
        logits = self(images)
        loss = criterion(logits, targets)
        return {"loss": loss, "outputs": {"logits": logits}}

    def _log_step_metrics(
        self, results: Dict[str, Any], batch, stage: str
    ) -> None:
        _, targets = batch
        batch_size = targets.shape[0]

        logged_dict = {
            f"{stage}/{self.train_criterion.__class__.__name__}": results[
                "loss"
            ].item()
        }
        self.log_dict(
            logged_dict, batch_size=batch_size, **self.logging_params
        )

        metric = getattr(self, f"{stage}_metric")
        computed_metric = metric(results["outputs"]["logits"], targets)
        self.log(
            f"{stage}/{metric.__class__.__name__}",
            computed_metric,
            batch_size=batch_size,
            **self.logging_params,
        )

    # ------------------------------------------------------------------ #
    #  Lightning hooks                                                     #
    # ------------------------------------------------------------------ #

    def on_train_start(self) -> None:
        # Lightning runs a sanity validation pass before training; reset so it
        # does not leak into the best-metric tracker.
        self.valid_metric_best.reset()

    def training_step(self, batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        results = self.model_step(batch, self.train_criterion)
        self._log_step_metrics(results, batch, "train")
        return results

    def on_train_epoch_end(self) -> None:
        self.train_metric.reset()

    @torch.inference_mode()
    def validation_step(
        self, batch, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        results = self.model_step(batch, self.train_criterion)
        self._log_step_metrics(results, batch, "valid")
        return results

    def on_validation_epoch_end(self) -> None:
        self.valid_metric.reset()

    @torch.inference_mode()
    def test_step(self, batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        results = self.model_step(batch, self.train_criterion)
        self._log_step_metrics(results, batch, "test")
        return results

    # ------------------------------------------------------------------ #
    #  Optimizer (Bregman-aware — identical contract to sv.py)             #
    # ------------------------------------------------------------------ #

    def configure_optimizers(self) -> Dict[str, Any]:
        """Bregman optimizers (AdaBreg/LinBreg/ProxSGD) route parameters
        through the PruningManager using ``model.pruning_groups``; any standard
        optimizer (SGD, Adam) is applied to all parameters uniformly."""
        BREGMAN_OPTIMIZERS = {
            "AdaBreg",
            "AdaBregW",
            "AdaBregL2",
            "LinBreg",
            "ProxSGD",
        }
        optimizer_class_name = self.hparams.optimizer._target_.split(".")[-1]
        optimizer_partial = instantiate(self.hparams.optimizer)

        if optimizer_class_name in BREGMAN_OPTIMIZERS:
            self.pruning_manager = PruningManager(
                pl_module=self,
                group_configs=self.hparams.model.pruning_groups,
            )
            optimizer_param_groups = (
                self.pruning_manager.get_optimizer_param_groups()
            )
            for group in optimizer_param_groups:
                if "reg" in group and isinstance(
                    group.get("reg"), (dict, DictConfig)
                ):
                    group["reg"] = instantiate(group["reg"])
            optimizer = optimizer_partial(params=optimizer_param_groups)
        else:
            optimizer = optimizer_partial(params=self.parameters())

        if not self.hparams.get("lr_scheduler"):
            return {"optimizer": optimizer}

        cfg = self.hparams.lr_scheduler.scheduler
        extras = self.hparams.lr_scheduler.get("extras") or {}
        scheduler = instantiate(cfg, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, **dict(extras)},
        }


if __name__ == "__main__":
    import sys

    import hydra
    import pyrootutils
    from hydra import compose, initialize_config_dir

    from src.utils import register_custom_resolvers

    root = pyrootutils.setup_root(
        search_from=__file__,
        indicator=[".git", "pyproject.toml"],
        pythonpath=True,
        dotenv=True,
    )
    cfgd = str(root / "configs")
    ov = ["experiment=img/cifar10_dense_sgd", "logger=[]"] + sys.argv[1:]

    @register_custom_resolvers(
        config_name="train.yaml",
        overrides=ov,
        version_base="1.3",
        config_path=cfgd,
    )
    def _smoke():
        with initialize_config_dir(version_base="1.3", config_dir=cfgd):
            cfg = compose(config_name="train.yaml", overrides=ov)
        model = hydra.utils.instantiate(cfg.module, _recursive_=False)
        batch = (
            torch.randn(4, model.in_channels, 32, 32),
            torch.randint(0, 10, (4,)),
        )
        out = model.model_step(batch, model.train_criterion)
        print(
            f"{type(model).__name__}: loss={out['loss'].item():.4f} "
            f"logits={tuple(out['outputs']['logits'].shape)}"
        )

    _smoke()
