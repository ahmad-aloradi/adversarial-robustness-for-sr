"""Config + wiring tests for the image-benchmark experiments.

Covers all `configs/experiment/img/*.yaml`: they compose, resolve the fragile
monitor key, carry the right per-dataset shape, and (for Bregman) wire the
right optimizer / lambda regime / ResNet-18 pruning groups. A slow MNIST
`fast_dev_run` exercises the full train loop for dense + both optimizers.
"""

import glob
from pathlib import Path

import hydra
import pyrootutils
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from src.utils import register_custom_resolvers

_ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
_CONFIGS_DIR = _ROOT / "configs"

# (num_classes, in_channels) keyed by the dataset prefix of the experiment name
_EXPECT_SHAPE = {
    "cifar10": (10, 3),
    "cifar100": (100, 3),
    "mnist": (10, 1),
    "tinyimagenet": (200, 3),
}


def _compose(overrides, return_hydra_config=False):
    """Compose train.yaml from configs/, isolating GlobalHydra state."""
    GlobalHydra.instance().clear()

    @register_custom_resolvers(
        config_name="train.yaml",
        overrides=overrides,
        version_base="1.3",
        config_path=str(_CONFIGS_DIR),
    )
    def _do_compose():
        with initialize_config_dir(
            version_base="1.3", config_dir=str(_CONFIGS_DIR)
        ):
            return compose(
                config_name="train.yaml",
                overrides=overrides,
                return_hydra_config=return_hydra_config,
            )

    try:
        return _do_compose()
    finally:
        GlobalHydra.instance().clear()


def _img_experiments():
    files = sorted(
        glob.glob(str(_CONFIGS_DIR / "experiment" / "img" / "*.yaml"))
    )
    return [Path(p).stem for p in files]


def _bregman_experiments():
    return [n for n in _img_experiments() if "bregman" in n]


def _instantiate_module(exp):
    cfg = _compose([f"experiment=img/{exp}", "logger=[]"])
    return hydra.utils.instantiate(cfg.module, _recursive_=False)


@pytest.mark.parametrize("exp", _img_experiments())
def test_img_experiment_composes(exp):
    cfg = _compose([f"experiment=img/{exp}", "logger=[]"])
    assert cfg.module._target_ == "src.modules.img.ImageClassification"
    # Most fragile contract: replace resolver -> instantiated metric class.
    assert cfg.callbacks.model_checkpoint.monitor == "valid/MulticlassAccuracy"
    num_classes, in_channels = _EXPECT_SHAPE[exp.split("_")[0]]
    assert cfg.datamodule.num_classes == num_classes
    assert cfg.datamodule.in_channels == in_channels
    assert cfg.module.model.net.num_classes == cfg.datamodule.num_classes
    assert cfg.module.model.net.in_channels == cfg.datamodule.in_channels


@pytest.mark.parametrize("exp", _bregman_experiments())
def test_bregman_wiring(exp):
    cfg = _compose([f"experiment=img/{exp}", "logger=[]"])

    optimizer = cfg.module.optimizer._target_.split(".")[-1]
    assert optimizer == ("LinBreg" if "linbreg" in exp else "AdaBreg")

    groups = [g.name for g in cfg.module.model.pruning_groups]
    assert groups[-1] == "fallback"

    gates = ("model_checkpoint", "early_stopping", "ramp_validation_gate")
    scheduler = cfg.callbacks.model_pruning.lambda_scheduler
    if "fixed" in exp:
        assert scheduler is None
        assert all(cfg.callbacks[g].tolerance == 1.0 for g in gates)
    else:
        assert scheduler is not None
        assert all(cfg.callbacks[g].tolerance == 0.01 for g in gates)


def _param_group_map(model):
    """Map each parameter name to the pruning group it landed in."""
    id_to_group = {}
    for group in model.pruning_manager.processed_groups:
        for param in group["params"]:
            id_to_group[id(param)] = group["config"]["name"]
    return {
        name: id_to_group.get(id(param))
        for name, param in model.named_parameters()
    }


def test_configure_optimizers_dense():
    model = _instantiate_module("cifar10_dense_sgd")
    out = model.configure_optimizers()
    assert out["optimizer"].__class__.__name__ == "SGD"
    assert (
        out["lr_scheduler"]["scheduler"].__class__.__name__
        == "CosineAnnealingLR"
    )
    assert not hasattr(model, "pruning_manager")


@pytest.mark.parametrize(
    "exp,optimizer",
    [
        ("cifar10_bregman_adabreg", "AdaBreg"),
        ("cifar10_bregman_linbreg", "LinBreg"),
    ],
)
def test_configure_optimizers_bregman(exp, optimizer):
    model = _instantiate_module(exp)
    out = model.configure_optimizers()
    assert out["optimizer"].__class__.__name__ == optimizer
    assert hasattr(model, "pruning_manager")

    groups = model.pruning_manager.processed_groups
    total = sum(len(g["params"]) for g in groups)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert total == len(trainable)

    # torchvision ResNet-18: type-only routing keeps downsample BN gammas out
    # of the RegL1 fallback (which is empty and therefore dropped).
    name_to_group = _param_group_map(model)
    assert name_to_group["net.layer2.0.downsample.0.weight"] == "conv_layers"
    assert name_to_group["net.layer2.0.downsample.1.weight"] == "norm_params"
    assert name_to_group["net.fc.weight"] == "linear_layers"
    assert "fallback" not in {g["config"]["name"] for g in groups}


@pytest.mark.slow
@pytest.mark.parametrize(
    "exp",
    ["mnist_dense_sgd", "mnist_bregman_adabreg", "mnist_bregman_linbreg"],
)
def test_mnist_fast_dev_run(exp, tmp_path):
    from src.train import train

    cfg = _compose([f"experiment=img/{exp}"], return_hydra_config=True)
    HydraConfig().set_config(cfg)
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.trainer.fast_dev_run = True
        cfg.trainer.accelerator = "cpu"
        cfg.trainer.deterministic = False
        cfg.test = False
        cfg.save_state_dict = False
        cfg.extras.print_config = False
        cfg.extras.enforce_tags = False
    train(cfg)
