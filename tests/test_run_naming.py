"""Tests for the shared run-name tokens and the fixed-lambda retag script.

``scripts/fabfile.py`` builds every submitted run's name from these tokens;
``scripts/retag_fixed_lambda_runs.py`` migrates runs that finished under the
older spelling. Both must agree on how a token is spelled, or a finished run
becomes unfindable.
"""
import importlib.util
import pathlib

import pytest
import yaml

from src.utils.run_naming import (
    initial_sparsity_token,
    is_fixed_lambda,
    lambda_token,
    sparsity_token,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
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


def test_is_fixed_lambda():
    assert is_fixed_lambda("proxsgd_fixed")
    assert not is_fixed_lambda("proxsgd")
    assert not is_fixed_lambda("bregman_adabreg")


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
