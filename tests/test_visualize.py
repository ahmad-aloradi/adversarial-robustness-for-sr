"""Tests for scripts/visualize.py — the shared visualization stack.

Covers the two directory layouts it must serve:
  * SV: ``<base>/<exp>`` (one run dir, no seed subdir)
  * Image: ``<base>/<dataset>/<model>/<augmentation>/<exp>/seed_<N>`` (grouped)

and the cross-seed accuracy aggregation (mean ± std).
"""

import importlib.util
import pathlib

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "visualize", ROOT / "scripts" / "visualize.py"
)
viz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(viz)


# ---------------------------------------------------------------------------
# Name parsing
# ---------------------------------------------------------------------------


def test_parse_image_pruning_name():
    info = viz.parse_experiment_name(
        "pruning_mag_unstruct-sr95-CosineAnnealing"
    )
    assert info["method_class"] == "pruning_unstruct"
    assert info["sparsity"] == 95
    assert info["scheduler"] == "CosineAnnealing"


@pytest.mark.parametrize(
    "name, method, variant, sparsity",
    [
        ("bregman_adabreg-sr90-CosineAnnealing", "adabreg", None, 90),
        ("bregman_adabreg_fixed-sr90-CosineAnnealing", "adabreg", "fixed", 90),
        (
            "bregman_adabreg_progressive-sr99-no_scheduler",
            "adabreg",
            "progressive",
            99,
        ),
        (
            "bregman_linbreg_progressive-sr99-ReduceLROnPlateau",
            "linbreg",
            "progressive",
            99,
        ),
        ("dense_sgd-CosineAnnealing", "dense", None, None),
    ],
)
def test_parse_image_method_variants(name, method, variant, sparsity):
    info = viz.parse_experiment_name(name)
    assert info["method_class"] == method
    assert info["variant"] == variant
    assert info["sparsity"] == sparsity


@pytest.mark.parametrize(
    "name, method, dataset, model, sparsity",
    [
        (
            "bregman_adabreg-resnet18-cifar10-bs128-sr99-classifier_10k-epshi",
            "adabreg",
            "cifar10",
            "resnet18",
            99,
        ),
        (
            "pruning_mag_struct-wrn28_10-cifar100-bs128-sr90-classifier_10k",
            "pruning_struct",
            "cifar100",
            "wrn28_10",
            90,
        ),
        (
            "bregman_adabreg-ramp80_constant-resnet18-cifar10-bs128-sr95",
            "adabreg",
            "cifar10",
            "resnet18",
            95,
        ),
        ("dense_sgd-resnet18-cifar10-bs128", "dense", "cifar10", "resnet18", None),
    ],
)
def test_parse_image_fabfile_name(name, method, dataset, model, sparsity):
    # Fabfile embeds model/dataset and puts -sr<NN> mid-name; the curated tree has no augmentation dir.
    info = viz.parse_experiment_name(name)
    assert info["method_class"] == method
    assert info["dataset"] == dataset
    assert info["model"] == model
    assert info["sparsity"] == sparsity


def test_parse_image_dotyaml_artifact():
    # Older runs kept the launcher's ".yaml" glued to the method token; the
    # substring match still classifies them (discovery strips it for labels).
    info = viz.parse_experiment_name("bregman_adabreg.yaml-sr90-CosineAnnealing")
    assert info["method_class"] == "adabreg"
    assert info["sparsity"] == 90


def test_dense_baseline_labels_with_optimizer():
    info = viz.parse_experiment_name("dense_sgd-CosineAnnealing")
    assert info["method_class"] == "dense"
    assert info["optimizer"] == "sgd"
    # Tick label shows the optimizer (group header already reads "Dense").
    assert viz.method_display_name(info) == "SGD"


def test_parse_sv_name_unchanged():
    info = viz.parse_experiment_name(
        "sv_bregman_adabreg-wespeaker_ecapa_tdnn-multi_sv-bs64-ep30-augFalse-sr99"
    )
    assert info["method_class"] == "adabreg"
    assert info["model"] == "wespeaker_ecapa_tdnn"
    assert info["sparsity"] == 99
    # The wespeaker backbone regex must still win over the image fallback.
    info2 = viz.parse_experiment_name(
        "sv_bregman_adabreg_progressive-wespeaker_resnet34-multi_sv-bs32-ep30-augFalse-sr99"
    )
    assert info2["model"] == "wespeaker_resnet34"
    assert info2["method_class"] == "adabreg"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _make_run(dir_path, acc_by_epoch, sparsity=0.9):
    """Write a minimal run dir (config_tree.log + train_log.txt)."""
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "config_tree.log").write_text("dummy\n")
    lines = [
        f"epoch: {e}, train_loss: 0.5, valid_loss: 0.5, "
        f"valid/MulticlassAccuracy: {a:.4f}, sparsity: {sparsity}"
        for e, a in acc_by_epoch
    ]
    (dir_path / "train_log.txt").write_text("\n".join(lines) + "\n")


def test_discover_groups_image_seeds(tmp_path):
    base = tmp_path / "train" / "runs"
    exp = (
        base
        / "cifar10"
        / "resnet18"
        / "augmentation"
        / "bregman_adabreg-sr90-CosineAnnealing"
    )
    _make_run(exp / "seed_42", [(1, 0.80), (2, 0.90)])
    _make_run(exp / "seed_43", [(1, 0.82), (2, 0.94)])

    found = viz.discover_experiments(
        [str(base)], ["bregman_adabreg-sr90-CosineAnnealing"]
    )
    assert len(found) == 1
    rep, info = found[0]
    assert info["seeds"] == [42, 43]
    assert len(info["seed_dirs"]) == 2
    assert rep == info["seed_dirs"][0]
    assert info["dataset"] == "cifar10"
    assert info["model"] == "resnet18"
    assert info["augmentation"] is True


def test_discover_separates_augmentation(tmp_path):
    # Same <exp> name under augmentation vs no_augmentation must stay distinct.
    base = tmp_path / "runs"
    for aug in ("augmentation", "no_augmentation"):
        _make_run(
            base / "cifar10" / "resnet18" / aug / "dense_sgd-CosineAnnealing"
            / "seed_42",
            [(1, 0.9)],
        )
    found = viz.discover_experiments([str(base)], ["dense_sgd-CosineAnnealing"])
    assert len(found) == 2
    assert sorted(info["augmentation"] for _, info in found) == [False, True]


def test_discover_strips_dotyaml_from_label(tmp_path):
    base = tmp_path / "runs"
    _make_run(
        base
        / "cifar10"
        / "resnet18"
        / "augmentation"
        / "bregman_adabreg.yaml-sr90-CosineAnnealing"
        / "seed_42",
        [(1, 0.9)],
    )
    # Clean pattern matches despite the ".yaml" on disk; dirname is normalized.
    found = viz.discover_experiments(
        [str(base)], ["bregman_adabreg-sr90-CosineAnnealing"]
    )
    assert len(found) == 1
    assert found[0][1]["dirname"] == "bregman_adabreg-sr90-CosineAnnealing"


def test_discover_sv_single_run(tmp_path):
    base = tmp_path / "logs" / "train" / "runs"
    exp = "sv_vanilla-wespeaker_ecapa_tdnn-multi_sv-bs64"
    _make_run(base / exp, [(1, 0.70), (2, 0.75)])

    found = viz.discover_experiments([str(base)], ["sv_vanilla*"])
    assert len(found) == 1
    rep, info = found[0]
    assert info["seeds"] == [None]
    assert info["seed_dirs"] == [rep]


def test_discover_ignores_nonmatching(tmp_path):
    base = tmp_path / "runs"
    root = base / "cifar10" / "resnet18" / "augmentation"
    _make_run(root / "dense_sgd-CosineAnnealing" / "seed_42", [(1, 0.9)])
    _make_run(
        root / "pruning_mag_unstruct-sr90-CosineAnnealing" / "seed_42",
        [(1, 0.8)],
    )
    found = viz.discover_experiments([str(base)], ["dense_sgd*"])
    assert len(found) == 1
    assert found[0][1]["method_class"] == "dense"


# ---------------------------------------------------------------------------
# Cross-seed aggregation
# ---------------------------------------------------------------------------


def test_aggregate_seed_series_mean_std():
    df1 = pd.DataFrame(
        {"epoch": [1, 2, 3], "valid/MulticlassAccuracy": [0.5, 0.7, 0.9]}
    )
    df2 = pd.DataFrame(
        {"epoch": [1, 2, 3], "valid/MulticlassAccuracy": [0.6, 0.8, 1.0]}
    )
    x, mean, std = viz.aggregate_seed_series(
        [df1, df2], "valid/MulticlassAccuracy", {}, "train_log"
    )
    assert list(x) == [1.0, 2.0, 3.0]
    assert mean == pytest.approx([0.55, 0.75, 0.95])
    # sample std (ddof=1) of {a, a+0.1} is 0.1/sqrt(2)
    assert std == pytest.approx([0.0707107] * 3, abs=1e-5)


def test_aggregate_single_seed_has_no_band():
    df = pd.DataFrame(
        {"epoch": [1, 2], "valid/MulticlassAccuracy": [0.5, 0.7]}
    )
    x, mean, std = viz.aggregate_seed_series(
        [df], "valid/MulticlassAccuracy", {}, "train_log"
    )
    assert std is None
    assert mean == pytest.approx([0.5, 0.7])


def test_aggregate_missing_metric_returns_none():
    df = pd.DataFrame({"epoch": [1, 2], "train_loss": [0.5, 0.4]})
    assert viz.aggregate_seed_series([df], "EER", {}, "train_log") is None


def test_per_seed_metric_reduce(tmp_path):
    base = tmp_path / "cifar10" / "resnet18" / "exp"
    _make_run(base / "seed_1", [(1, 0.80), (2, 0.92), (3, 0.88)])
    _make_run(base / "seed_2", [(1, 0.70), (2, 0.85), (3, 0.90)])
    seed_dirs = [str(base / "seed_1"), str(base / "seed_2")]

    best = viz.per_seed_metric(
        seed_dirs, "valid/MulticlassAccuracy", "train_log", reduce="max"
    )
    assert sorted(best) == pytest.approx([0.90, 0.92])
    last = viz.per_seed_metric(
        seed_dirs, "valid/MulticlassAccuracy", "train_log", reduce="last"
    )
    assert sorted(last) == pytest.approx([0.88, 0.90])
