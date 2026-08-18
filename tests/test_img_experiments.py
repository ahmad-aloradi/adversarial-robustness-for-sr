"""Config + wiring tests for the image-benchmark experiments.

Covers every `configs/experiment/img/*.yaml` (one file per method) crossed
with every image dataset config: they compose, resolve the fragile monitor
key, pull the epoch budget from the dataset, and keep one fixed 3-channel
backbone. Bregman files additionally wire the right optimizer / lambda
regime / ResNet-18 pruning groups. A slow MNIST `fast_dev_run` exercises the
full train loop for dense + both optimizers.
"""

import glob
from pathlib import Path

import hydra
import pyrootutils
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from src.utils import register_custom_resolvers

_ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
_CONFIGS_DIR = _ROOT / "configs"

# num_classes keyed by dataset config name (the only per-dataset model knob).
_DATASETS = {
    "cifar10": 10,
    "cifar100": 100,
    "mnist": 10,
    "tinyimagenet": 200,
    "imagenet": 1000,
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


@pytest.mark.parametrize("dataset", sorted(_DATASETS))
@pytest.mark.parametrize("exp", _img_experiments())
def test_img_experiment_composes(exp, dataset):
    cfg = _compose(
        [
            f"experiment=img/{exp}",
            f"datamodule=datasets/{dataset}",
            "logger=[]",
        ]
    )
    assert cfg.module._target_ == "src.modules.img.ImageClassification"
    # Most fragile contract: replace resolver -> instantiated metric class.
    assert cfg.callbacks.model_checkpoint.monitor == "valid/MulticlassAccuracy"
    assert cfg.datamodule.num_classes == _DATASETS[dataset]
    assert cfg.module.model.net.num_classes == cfg.datamodule.num_classes
    # One fixed backbone: 3-channel input regardless of dataset.
    assert cfg.module.model.net.in_channels == 3
    # Epoch budget flows from the dataset config.
    assert cfg.trainer.max_epochs == cfg.datamodule.max_epochs
    tags = OmegaConf.to_container(cfg.tags, resolve=True)
    assert dataset in tags


# The ramp knob each recipe holds the lr for; every other experiment anneals from epoch 0.
_RAMP_KNOB = {
    "pruning_mag_struct": "callbacks.model_pruning.epochs_to_ramp",
    "pruning_mag_unstruct": "callbacks.model_pruning.epochs_to_ramp",
    "pruning_granet": "callbacks.model_pruning.final_prune_epoch",
    "bregman_adabreg_progressive": "_bregman_ramp_epochs",
    "bregman_linbreg_progressive": "_bregman_ramp_epochs",
    "bregman_adabreg_quantile_progressive": "_bregman_ramp_epochs",
    "bregman_linbreg_quantile_progressive": "_bregman_ramp_epochs",
}
_BASE_LR = 0.1


def _lr_trajectory(cfg):
    """The per-epoch lr an experiment's configured scheduler produces."""
    optimizer = torch.optim.SGD(
        [torch.nn.Parameter(torch.zeros(1))], lr=_BASE_LR
    )
    scheduler = hydra.utils.instantiate(
        cfg.module.lr_scheduler.scheduler, optimizer=optimizer
    )
    values = []
    for _ in range(cfg.trainer.max_epochs):
        values.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    return scheduler, values


@pytest.mark.parametrize("dataset", sorted(_DATASETS))
@pytest.mark.parametrize("exp", sorted(_RAMP_KNOB))
def test_lr_is_held_flat_for_exactly_the_sparsity_ramp(exp, dataset):
    """Ramped recipes hold the base lr over their own ramp, then anneal.

    Parametrized over datasets because the knob is a fraction of the budget.
    """
    cfg = _compose(
        [
            f"experiment=img/{exp}",
            f"datamodule=datasets/{dataset}",
            "logger=[]",
        ]
    )
    _, values = _lr_trajectory(cfg)

    hold = OmegaConf.select(cfg, _RAMP_KNOB[exp], throw_on_missing=True)
    assert cfg.module.lr_scheduler.scheduler.constant_epochs == hold
    assert values[: hold + 1] == pytest.approx([_BASE_LR] * (hold + 1))
    assert values[hold + 1] < _BASE_LR
    assert values[-1] < 0.05 * _BASE_LR


@pytest.mark.parametrize(
    "exp", [e for e in _img_experiments() if e not in _RAMP_KNOB]
)
def test_lr_anneals_from_epoch_zero_when_unramped(exp):
    """Every recipe outside ``_RAMP_KNOB`` anneals from epoch 0."""
    cfg = _compose(
        [f"experiment=img/{exp}", "datamodule=datasets/cifar100", "logger=[]"]
    )
    scheduler, values = _lr_trajectory(cfg)
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert values[1] < _BASE_LR


@pytest.mark.parametrize("exp", sorted(_RAMP_KNOB))
def test_zero_hold_is_a_plain_cosine(exp):
    """``_lr_constant_epochs=0`` composes to a bare ``CosineAnnealingLR``.

    The trajectory equivalence itself is proven once, without Hydra, by
    test_lr_schedulers.py::test_zero_constant_epochs_matches_plain_cosine_trajectory.
    """
    cfg = _compose(
        [
            f"experiment=img/{exp}",
            "datamodule=datasets/cifar100",
            "logger=[]",
            "_lr_constant_epochs=0",
        ]
    )
    assert cfg.module.lr_scheduler.scheduler.constant_epochs == 0
    scheduler, values = _lr_trajectory(cfg)
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert values[1] < _BASE_LR


_SELECTION_GATES = ("model_checkpoint", "early_stopping")


@pytest.mark.parametrize("dataset", ["cifar100", "imagenet"])
@pytest.mark.parametrize("exp", _img_experiments())
def test_selection_is_banded_and_validation_is_not(exp, dataset):
    """Checkpointing and early stopping band the target; validation does not.

    Why: docs/pruning.md §3. ImageNet bands validation too, and a *_fixed run
    has no setpoint to band, so it opens all three.
    """
    cfg = _compose(
        [
            f"experiment=img/{exp}",
            f"datamodule=datasets/{dataset}",
            "logger=[]",
        ]
    )
    if cfg.callbacks.get("ramp_validation_gate") is None:
        pytest.skip(f"{exp} controls no sparsity to gate on")

    band = 1.0 if "fixed" in exp else 0.005
    assert all(cfg.callbacks[g].tolerance == band for g in _SELECTION_GATES)
    assert cfg.callbacks.ramp_validation_gate.tolerance == (
        band if dataset == "imagenet" else 1.0
    )


def test_mnist_transform_contract():
    """MNIST transforms pad 28->32 and replicate to 3 channels so it meets the
    same input contract as CIFAR (no download needed)."""
    from PIL import Image
    from torchvision.transforms import Compose

    cfg = _compose(
        ["experiment=img/dense_sgd", "datamodule=datasets/mnist", "logger=[]"]
    )
    datamodule = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)
    for specs in (datamodule._train_specs(), cfg.datamodule.transforms.eval):
        pipe = Compose([hydra.utils.instantiate(t) for t in specs])
        out = pipe(Image.new("L", (28, 28)))
        assert tuple(out.shape) == (3, 32, 32)


@pytest.mark.parametrize("exp", _bregman_experiments())
def test_bregman_wiring(exp):
    cfg = _compose([f"experiment=img/{exp}", "logger=[]"])

    optimizer = cfg.module.optimizer._target_.split(".")[-1]

    groups = [g.name for g in cfg.module.model.pruning_groups]
    assert groups[-1] == "fallback"

    scheduler = cfg.callbacks.model_pruning.lambda_scheduler
    if "fixed" in exp:
        # Static lambda -> sparsity is uncontrolled, so nothing gates on it.
        assert scheduler is None
    else:
        assert scheduler is not None


@pytest.mark.parametrize("exp", _bregman_experiments())
def test_regularizer_key_switches_every_thresholding_group(exp):
    """``_bregman_regularizer`` picks the readout; RegL1 is the default.

    The unregularized groups keep RegNone: only groups that threshold weights
    follow the key.
    """
    for override, expected in (
        ([], "RegL1"),
        (["_bregman_regularizer=RegSoftBernoulli"], "RegSoftBernoulli"),
    ):
        cfg = _compose([f"experiment=img/{exp}", "logger=[]"] + override)
        got = {
            g.name: type(
                hydra.utils.instantiate(g.optimizer_settings.reg)
            ).__name__
            for g in cfg.module.model.pruning_groups
        }
        assert got == {
            "conv_layers": expected,
            "linear_layers": expected,
            "norm_params": "RegNone",
            "bias_params": "RegNone",
            "fallback": expected,
        }


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


def test_top5_metrics_tracked():
    """Top-1 stays the monitored metric; top-5 runs alongside it per stage."""
    model = _instantiate_module("dense_sgd")
    for stage in ("train", "valid", "test"):
        assert getattr(model, f"{stage}_metric").top_k == 1
        assert getattr(model, f"{stage}_metric_top5").top_k == 5

    # Target ranked 2nd: wrong at top-1, correct within top-5.
    logits = torch.zeros(1, 10)
    logits[0, 3], logits[0, 7] = 2.0, 1.0
    targets = torch.tensor([7])
    assert float(model.valid_metric(logits, targets)) == 0.0
    assert float(model.valid_metric_top5(logits, targets)) == 1.0


def test_configure_optimizers_dense():
    model = _instantiate_module("dense_sgd")
    out = model.configure_optimizers()
    assert out["optimizer"].__class__.__name__ == "SGD"
    assert (
        out["lr_scheduler"]["scheduler"].__class__.__name__
        == "CosineAnnealingLR"
    )
    assert not hasattr(model, "pruning_manager")


def test_str_uses_one_weight_decay():
    """"s also has the same weight-decay parameter lambda" (Kusupati et al.,
    Sec 3), so w, bias, BatchNorm and the s scalars all sit in one group at
    module.optimizer.weight_decay, set to Table 10's lambda."""
    import hydra

    cfg = _compose(["experiment=img/soft_threshold"])
    model = hydra.utils.instantiate(cfg.module, _recursive_=False)
    hydra.utils.instantiate(cfg.callbacks.model_pruning).setup(
        None, model, stage="fit"
    )
    groups = model.configure_optimizers()["optimizer"].param_groups

    assert len(groups) == 1, f"expected one param group, got {len(groups)}"
    assert groups[0]["weight_decay"] == cfg.module.optimizer.weight_decay
    assert len(groups[0]["params"]) == len(list(model.parameters()))
    thresholds = [
        n
        for n, _ in model.named_parameters()
        if n.endswith("sparse_threshold")
    ]
    assert (
        thresholds
    ), "no STR thresholds; did callbacks.model_pruning get disabled?"


def test_dense_recipes_keep_one_parameter_group():
    import hydra

    cfg = _compose(["experiment=img/dense_sgd"])
    model = hydra.utils.instantiate(cfg.module, _recursive_=False)
    groups = model.configure_optimizers()["optimizer"].param_groups
    assert len(groups) == 1, "no STR layers means no split"


@pytest.mark.parametrize(
    "exp,optimizer",
    [
        ("bregman_adabreg", "AdaBreg"),
        ("bregman_linbreg", "LinBreg"),
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
    assert "fallback" not in {g["config"]["name"] for g in groups}

    # prune_first_layer=false: the stem sits out of RegL1, the head does not.
    assert name_to_group["net.conv1.weight"] == "first_dense"
    assert name_to_group["net.fc.weight"] == "linear_layers"


@pytest.mark.slow
@pytest.mark.parametrize(
    "exp",
    [
        "dense_sgd",
        "bregman_adabreg",
        "bregman_linbreg",
        "bregman_adabreg_quantile",
        "bregman_linbreg_quantile",
        "pruning_rigl",
        "pruning_set",
        "pruning_static",
        "pruning_snip",
        "pruning_granet",
        "soft_threshold",
    ],
)
def test_mnist_fast_dev_run(exp, tmp_path):
    from src.train import train

    cfg = _compose(
        [f"experiment=img/{exp}", "datamodule=datasets/mnist"],
        return_hydra_config=True,
    )
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
        if exp == "pruning_granet":
            # fast_dev_run is one step; the 1000-step cubic ramp has to fit it.
            cfg.callbacks.model_pruning.update_frequency = 1
            cfg.callbacks.model_pruning.final_prune_epoch = 1
    train(cfg)
