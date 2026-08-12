"""Tests for run-directory naming and the fixed-lambda retag script.

``run_subdir`` builds the name ``python src/train.py`` writes to;
``scripts/retag_fixed_lambda_runs.py`` migrates runs that finished under the
older spelling. Both must agree on how a token is spelled, or a finished run
becomes unfindable.
"""
import importlib.util
import pathlib

import pytest
import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig

from src.utils import register_custom_resolvers
from src.utils.run_naming import (
    initial_sparsity_token,
    is_fixed_lambda,
    lambda_token,
    run_subdir,
    sparsity_token,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIGS_DIR = str(ROOT / "configs")
_spec = importlib.util.spec_from_file_location(
    "retag_fixed_lambda_runs", ROOT / "scripts" / "retag_fixed_lambda_runs.py"
)
retag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(retag)


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sparsity, expected",
    [(0.9, "-sr90"), (0.99, "-sr99"), (0.29, "-sr29"), (None, "")],
)
def test_sparsity_token(sparsity, expected):
    # 0.29 * 100 is 28.999…, so truncation would spell it -sr28.
    assert sparsity_token(sparsity) == expected
    assert initial_sparsity_token(sparsity) == expected.replace("-sr", "-isr")


@pytest.mark.parametrize(
    "value, expected",
    [(18.0, "-lam18"), (0.025, "-lam0.025"), (1e-4, "-lam0.0001"), (None, "")],
)
def test_lambda_token(value, expected):
    assert lambda_token(value) == expected


# ---------------------------------------------------------------------------
# run_subdir
# ---------------------------------------------------------------------------


def test_fixed_runs_are_named_by_lambda_not_target():
    # A static lambda decides where the run lands, so a target would misdescribe it.
    assert run_subdir(
        "cifar100",
        "resnet18",
        False,
        "img/bregman_adabreg_fixed",
        0.99,
        0.5,
        18.0,
        None,
    ).endswith("bregman_adabreg_fixed-isr50-lam18-no_scheduler")


def test_adaptive_runs_keep_the_target():
    assert run_subdir(
        "cifar100",
        "resnet18",
        False,
        "img/bregman_adabreg",
        0.99,
        0.5,
        0.01,
        None,
    ).endswith("bregman_adabreg-isr50-sr99-no_scheduler")


def test_fixed_run_without_a_lambda_falls_back_to_the_target():
    # A non-Bregman *_fixed experiment has no lambda to name itself after.
    assert run_subdir(
        "cifar100",
        "resnet18",
        False,
        "img/something_fixed",
        0.9,
        None,
        None,
        None,
    ).endswith("something_fixed-sr90-no_scheduler")


def test_is_fixed_lambda():
    assert is_fixed_lambda("proxsgd_fixed")
    assert not is_fixed_lambda("proxsgd")
    assert not is_fixed_lambda("bregman_adabreg")


# ---------------------------------------------------------------------------
# End-to-end: the name train.yaml actually resolves to
# ---------------------------------------------------------------------------


def _resolved_name(experiment):
    """The run-dir stem ``python src/train.py experiment=<experiment>`` writes
    to."""
    overrides = [
        f"experiment={experiment}",
        "datamodule=datasets/cifar100",
        "datamodule.augmentation=true", # pinned: the stem under test is the method's, not the dataset's augmentation default
        "logger=[]",
    ]
    GlobalHydra.instance().clear()

    @register_custom_resolvers(
        config_name="train.yaml",
        overrides=overrides,
        version_base="1.3",
        config_path=CONFIGS_DIR,
    )
    def _compose():
        with initialize_config_dir(version_base="1.3", config_dir=CONFIGS_DIR):
            cfg = compose(
                config_name="train.yaml",
                overrides=overrides,
                return_hydra_config=True,
            )
            HydraConfig().set_config(
                cfg
            )  # the name interpolation reads hydra.runtime
            return cfg.name

    try:
        return _compose()
    finally:
        GlobalHydra.instance().clear()


@pytest.mark.parametrize(
    "experiment, stem",
    [
        (
            "img/bregman_adabreg_fixed",
            "bregman_adabreg_fixed-isr99-lam5-CosineAnnealing",
        ),
        (
            "img/bregman_linbreg_fixed",
            "bregman_linbreg_fixed-isr99-lam0.15-CosineAnnealing",
        ),
        ("img/proxsgd_fixed", "proxsgd_fixed-isr0-lam0.0001-CosineAnnealing"),
        ("img/bregman_adabreg", "bregman_adabreg-isr99-sr90-CosineAnnealing"),
        # lr is held flat over the sparsity ramp, so the tag names the hold;
        # isr reads model_pruning.initial_amount outside Bregman
        (
            "img/pruning_mag_struct",
            "pruning_mag_struct-isr50-sr90-Const150Cosine",
        ),
        ("img/pruning_granet", "pruning_granet-isr50-sr99-Const150Cosine"),
        (
            "img/bregman_adabreg_progressive",
            "bregman_adabreg_progressive-isr50-sr90-Const150Cosine",
        ),
        ("img/dense_sgd", "dense_sgd-CosineAnnealing"),
    ],
)
def test_train_yaml_resolves_the_expected_stem(experiment, stem):
    assert (
        _resolved_name(experiment) == f"cifar100/resnet18/augmentation/{stem}"
    )


# ---------------------------------------------------------------------------
# Retag script
# ---------------------------------------------------------------------------


def _fabricate_run(root, name, lambda_value, seeds=(42,)):
    """Write a run dir with the per-seed Hydra snapshot the script reads."""
    for seed in seeds:
        hydra_dir = root / name / f"seed_{seed}" / ".hydra"
        hydra_dir.mkdir(parents=True)
        (hydra_dir / "config.yaml").write_text(
            yaml.safe_dump({"_bregman_lambda": lambda_value})
        )
    return root / name


def test_plan_renames_only_fixed_runs(tmp_path):
    _fabricate_run(
        tmp_path, "bregman_adabreg_fixed-isr99-sr99-CosineAnnealing", 18.0
    )
    _fabricate_run(
        tmp_path, "bregman_adabreg-isr99-sr99-CosineAnnealing", 0.01
    )
    _fabricate_run(tmp_path, "pruning_mag_struct-sr95-CosineAnnealing", 0.0)

    actions = retag.plan(str(tmp_path))
    assert [(a, pathlib.Path(dst).name) for a, _, dst in actions] == [
        ("rename", "bregman_adabreg_fixed-isr99-lam18-CosineAnnealing")
    ]


def test_plan_skips_already_retagged_runs(tmp_path):
    _fabricate_run(tmp_path, "bregman_linbreg_fixed-isr50-lam0.025", 0.025)
    actions = retag.plan(str(tmp_path))
    assert [a for a, _, _ in actions] == ["skip"]


def test_apply_renames_the_run_dir(tmp_path):
    _fabricate_run(tmp_path, "proxsgd_fixed-isr0-sr90-CosineAnnealing", 0.01)
    retag.apply_actions(retag.plan(str(tmp_path)))
    assert (tmp_path / "proxsgd_fixed-isr0-lam0.01-CosineAnnealing").is_dir()
    assert not (tmp_path / "proxsgd_fixed-isr0-sr90-CosineAnnealing").exists()


def test_apply_leaves_the_tree_untouched_on_a_collision(tmp_path):
    # Every destination is checked before the first rename, so nothing half-moves.
    _fabricate_run(
        tmp_path, "bregman_adabreg_fixed-isr99-sr95-CosineAnnealing", 18.0
    )
    _fabricate_run(
        tmp_path, "bregman_adabreg_fixed-isr99-sr99-CosineAnnealing", 18.0
    )
    (tmp_path / "bregman_adabreg_fixed-isr99-lam18-CosineAnnealing").mkdir()

    with pytest.raises(AssertionError, match="target already exists"):
        retag.apply_actions(retag.plan(str(tmp_path)))
    assert (
        tmp_path / "bregman_adabreg_fixed-isr99-sr95-CosineAnnealing"
    ).is_dir()
    assert (
        tmp_path / "bregman_adabreg_fixed-isr99-sr99-CosineAnnealing"
    ).is_dir()


def test_read_lambda_rejects_seeds_that_disagree(tmp_path):
    run = _fabricate_run(
        tmp_path, "bregman_adabreg_fixed-sr99", 18.0, seeds=(1,)
    )
    other = run / "seed_2" / ".hydra"
    other.mkdir(parents=True)
    (other / "config.yaml").write_text(
        yaml.safe_dump({"_bregman_lambda": 10.0})
    )

    with pytest.raises(AssertionError, match="disagree on lambda"):
        retag.read_lambda(str(run))


def test_retagged_matches_run_subdir(tmp_path):
    # The migrated name must equal what src/train.py would write today.
    old = "bregman_adabreg_fixed-isr50-sr99-CosineAnnealing"
    expected = run_subdir(
        "cifar100",
        "resnet18",
        False,
        "img/bregman_adabreg_fixed",
        0.99,
        0.5,
        18.0,
        "torch.optim.lr_scheduler.CosineAnnealingLR",
    ).rsplit("/", 1)[-1]
    assert retag.retagged(old, 18.0) == expected
