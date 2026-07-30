"""Tests for scripts/visualize.py — the shared visualization stack.

Covers the two directory layouts it must serve:
  * SV: ``<base>/<exp>`` (one run dir, no seed subdir)
  * Image: ``<base>/<dataset>/<model>/<augmentation>/<exp>/seed_<N>`` (grouped)

and the cross-seed accuracy aggregation (mean ± std).
"""

import importlib.util
import json
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
        (
            "dense_sgd-resnet18-cifar10-bs128",
            "dense",
            "cifar10",
            "resnet18",
            None,
        ),
    ],
)
def test_parse_image_fabfile_name(name, method, dataset, model, sparsity):
    # Fabfile embeds model/dataset and puts -sr<NN> mid-name; the curated tree has no augmentation dir.
    info = viz.parse_experiment_name(name)
    assert info["method_class"] == method
    assert info["dataset"] == dataset
    assert info["model"] == model
    assert info["sparsity"] == sparsity


@pytest.mark.parametrize(
    "name, initial_sparsity, sparsity",
    [
        ("bregman_adabreg-isr99-sr90-CosineAnnealing", 99, 90),
        ("bregman_adabreg_fixed-isr0-sr99-CosineAnnealing", 0, 99),
        ("bregman_adabreg-resnet18-cifar10-bs128-isr50-sr99", 50, 99),
        ("bregman_adabreg-sr90-CosineAnnealing", None, 90),
    ],
)
def test_parse_image_initial_sparsity(name, initial_sparsity, sparsity):
    # -isr<NN> is the starting sparsity and never shadows the -sr<NN> target.
    info = viz.parse_experiment_name(name)
    assert info["initial_sparsity"] == initial_sparsity
    assert info["sparsity"] == sparsity
    assert info["method_class"] == "adabreg"


def _label(name, *parts):
    """Expected label, with the symbols spelled the way visualize.py spells
    them."""
    return f"{name} ({', '.join(parts)})" if parts else name


def _sp(value):
    return f"{viz.SPARSITY_SYM}={value}{viz.pct_sym()}"


def _isp(value):
    return f"{viz.INIT_SPARSITY_SYM}={value}{viz.pct_sym()}"


def _lam(value):
    return f"{viz.LAMBDA_SYM}={value}"


def test_initial_sparsity_labels_only_when_it_varies():
    infos = [
        viz.parse_experiment_name(
            f"bregman_adabreg-isr{i}-sr99-CosineAnnealing"
        )
        for i in (0, 99)
    ]
    viz.assign_label_visibility([(None, info) for info in infos])
    assert [viz.make_label(i) for i in infos] == [
        _label("AdaBreg", _sp(99), _isp(0)),
        _label("AdaBreg", _sp(99), _isp(99)),
    ]

    same = [
        viz.parse_experiment_name("bregman_adabreg-isr99-sr99-CosineAnnealing")
    ]
    viz.assign_label_visibility([(None, info) for info in same])
    assert viz.make_label(same[0]) == _label("AdaBreg", _sp(99))


def test_fixed_lambda_label_shows_the_value_next_to_init():
    # Lambda replaces the old "(fixed)" tag and must not swallow the sweep tag.
    infos = [
        viz.parse_experiment_name(f"bregman_adabreg_fixed-isr{i}-lam0.004")
        for i in (0, 99)
    ]
    viz.assign_label_visibility([(None, info) for info in infos])
    assert [viz.make_label(i) for i in infos] == [
        _label("AdaBreg", _isp(0), _lam("0.004")),
        _label("AdaBreg", _isp(99), _lam("0.004")),
    ]


def test_proxsgd_carries_the_bregman_knobs():
    # ProxSGD inherits the Bregman parent config, so its runs sweep isr too.
    infos = [
        viz.parse_experiment_name(f"proxsgd-isr{i}-sr99-CosineAnnealing")
        for i in (0, 99)
    ]
    viz.assign_label_visibility([(None, info) for info in infos])
    assert [viz.make_label(i) for i in infos] == [
        _label("ProxSGD", _sp(99), _isp(0)),
        _label("ProxSGD", _sp(99), _isp(99)),
    ]


@pytest.mark.parametrize(
    "name",
    [
        "bregman_adabreg_fixed-isr50-lam18-CosineAnnealing",
        "bregman_adabreg_fixed-resnet18-cifar100-bs128-isr50-lam18",
        "sv_bregman_adabreg_fixed-wespeaker_resnet34-vox2-bs128-isr50-lam18",
        # No scheduler tag: the lambda must not be read as one.
        "bregman_adabreg_fixed-isr50-lam18",
    ],
)
def test_parse_fixed_lambda_token(name):
    # A fixed-lambda run names itself by lambda; it never had a target sparsity.
    info = viz.parse_experiment_name(name)
    assert info["fixed_lambda"] == 18.0
    assert info["sparsity"] is None
    assert info["initial_sparsity"] == 50
    assert info["variant"] == "fixed"


def test_lambda_always_shows_even_for_a_lone_fixed_run():
    # Lambda identifies a fixed-lambda run, so it is never gated on variation:
    # without it the label is indistinguishable from the adaptive run's.
    infos = [
        viz.parse_experiment_name(
            f"bregman_adabreg_fixed-lam{v}-CosineAnnealing"
        )
        for v in ("10", "18")
    ]
    viz.assign_label_visibility([(None, info) for info in infos])
    assert [viz.make_label(i) for i in infos] == [
        _label("AdaBreg", _lam(10)),
        _label("AdaBreg", _lam(18)),
    ]

    lone = [
        viz.parse_experiment_name(
            "bregman_adabreg_fixed-lam18-CosineAnnealing"
        )
    ]
    viz.assign_label_visibility([(None, info) for info in lone])
    assert viz.make_label(lone[0]) == _label("AdaBreg", _lam(18))


def test_bar_ticks_carry_both_sparsity_and_lambda():
    # The group header names the tier, not the sparsity, so the tick shows both.
    info = viz.parse_experiment_name("bregman_linbreg_fixed-lam0.25")
    info["sparsity"] = 97
    viz.assign_label_visibility([(None, info)])
    assert viz.make_label(info) == _label("LinBreg", _sp(97), _lam("0.25"))


def test_resolve_sparsity_level_rounds_the_realized_value():
    # 89.5 belongs in the 90% bucket: that is who it should be compared against,
    # and the marker/hatch tables are keyed on the round numbers.
    df = pd.DataFrame(
        {
            "sparsity": [None, 90.0],
            "actual_sparsity": [0.8953, 0.9012],
            "method_class": ["linbreg", "linbreg"],
        }
    )
    viz.resolve_sparsity_level(df)
    assert list(df["sparsity"]) == [90.0, 90.0]


def test_resolve_sparsity_level_keeps_dense_rows_null():
    # A dense run has no sparsity level; that null is what marks it dense.
    df = pd.DataFrame(
        {
            "sparsity": [None, None],
            "actual_sparsity": [None, 0.0001],
            "method_class": ["linbreg", "dense"],
        }
    )
    viz.resolve_sparsity_level(df)
    assert df["sparsity"].isna().all()


def test_fixed_lambda_run_is_not_dense():
    # Regression: a *_fixed run carries no -sr token, and inferring "dense" from
    # that put it in the Dense bar group, compared against the wrong baseline.
    fixed = viz.parse_experiment_name("bregman_linbreg_fixed-isr99-lam0.25")
    assert fixed["sparsity"] is None
    assert not viz.is_dense(fixed)
    assert viz.is_dense(viz.parse_experiment_name("dense_sgd-CosineAnnealing"))


def test_method_tier_separates_the_four_families():
    tier = {
        "dense_sgd-CosineAnnealing": viz.TIER_DENSE,
        "pruning_mag_unstruct-sr99-CosineAnnealing": viz.TIER_SPARSE_BASELINE,
        "bregman_linbreg_fixed-isr99-lam0.25": viz.TIER_FIXED,
        "bregman_linbreg-isr99-sr99-CosineAnnealing": viz.TIER_ADAPTIVE,
    }
    for name, expected in tier.items():
        assert viz.method_tier(viz.parse_experiment_name(name)) == expected


def test_fixed_tier_orders_by_lambda_and_adaptive_by_sparsity():
    # Within a tier, LinBreg and AdaBreg stay in blocks; fixed runs then climb
    # by lambda and adaptive runs by sparsity, whatever order they arrive in.
    names = [
        "bregman_adabreg-sr99-CosineAnnealing",
        "bregman_linbreg_fixed-lam6",
        "bregman_adabreg-sr75-CosineAnnealing",
        "dense_sgd-CosineAnnealing",
        "bregman_linbreg_fixed-lam0.25",
        "pruning_mag_unstruct-sr99-CosineAnnealing",
    ]
    infos = [viz.parse_experiment_name(n) for n in names]
    ordered = sorted(infos, key=viz.experiment_sort_key)
    assert [i["dirname"] for i in ordered] == [
        "dense_sgd-CosineAnnealing",
        "pruning_mag_unstruct-sr99-CosineAnnealing",
        "bregman_linbreg_fixed-lam0.25",
        "bregman_linbreg_fixed-lam6",
        "bregman_adabreg-sr75-CosineAnnealing",
        "bregman_adabreg-sr99-CosineAnnealing",
    ]


# ---------------------------------------------------------------------------
# Sweep styling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method", ["bregman_adabreg", "bregman_adabreg_fixed"]
)
def test_initial_sparsity_sweep_varies_every_channel(method):
    # Labels alone don't separate a sweep, and neither does color alone: the
    # line shapes have to differ too, or the curves are unreadable in print.
    infos = [
        viz.parse_experiment_name(f"{method}-isr{i}-sr99-CosineAnnealing")
        for i in (0, 50, 75, 99)
    ]
    viz.assign_sweep_styles([(None, info) for info in infos])
    styles = [viz.get_style(info) for info in infos]
    assert len({c for c, _, _ in styles}) == 4
    assert len({m for _, m, _ in styles}) == 4
    assert len({str(ls) for _, _, ls in styles}) == 4


def test_lambda_sweep_varies_every_channel():
    # With no -sr token, lambda is the only thing telling fixed runs apart.
    infos = [
        viz.parse_experiment_name(
            f"bregman_adabreg_fixed-lam{v}-CosineAnnealing"
        )
        for v in ("2", "4", "10", "18")
    ]
    viz.assign_sweep_styles([(None, info) for info in infos])
    styles = [viz.get_style(info) for info in infos]
    assert len({c for c, _, _ in styles}) == 4
    assert len({m for _, m, _ in styles}) == 4
    assert len({str(ls) for _, _, ls in styles}) == 4


def test_marker_by_method_leaves_the_sweep_the_dash_pattern():
    # Where sparsity is the x-axis the method owns the marker, so the swept
    # field has to keep a channel of its own.
    infos = [
        viz.parse_experiment_name(f"bregman_linbreg-isr{i}-sr99")
        for i in (0, 99)
    ]
    viz.assign_sweep_styles([(None, info) for info in infos])
    styles = [viz.get_style(i, marker_by="method") for i in infos]
    assert {m for _, m, _ in styles} == {viz.METHOD_MARKERS["linbreg"]}
    assert len({str(ls) for _, _, ls in styles}) == 2


def test_alpha_sweep_still_excludes_fixed_runs():
    # Fixed runs carry the alpha default (1.0), so pooling them fakes a sweep.
    stem = "wespeaker_resnet34-vox2-bs128-sr90"
    swept = [
        viz.parse_experiment_name(f"sv_bregman_adabreg-{stem}-alpha{a}")
        for a in ("0.5", "2.0")
    ]
    fixed = viz.parse_experiment_name(f"sv_bregman_adabreg_fixed-{stem}")
    viz.assign_sweep_styles([(None, i) for i in swept + [fixed]])
    assert all(i.get("_sweep_color") is not None for i in swept)
    assert fixed.get("_sweep_color") is None
    # Without a sweep, a fixed run keeps the star that marks its family.
    assert viz.get_style(fixed)[1] == viz.VARIANT_MARKERS["fixed"]


def test_clear_sweep_styles_hands_the_channels_back():
    infos = [
        viz.parse_experiment_name(f"bregman_adabreg-isr{i}-sr99")
        for i in (0, 99)
    ]
    experiments = [(None, info) for info in infos]
    viz.assign_sweep_styles(experiments)
    viz.clear_sweep_styles(experiments)
    assert {viz.get_style(i)[0] for i in infos} == {
        viz.METHOD_CLASS_COLORS["adabreg"]
    }


def test_sweep_param_falls_through_the_candidates():
    members = [
        {"initial_sparsity": 0, "fixed_lambda": 10.0, "alpha": 1.0, "f": 50},
        {"initial_sparsity": 99, "fixed_lambda": 18.0, "alpha": 2.0, "f": 50},
    ]
    assert viz.sweep_param(members) == "initial_sparsity"
    for m in members:
        m["initial_sparsity"] = 99
    assert viz.sweep_param(members) == "fixed_lambda"
    for m in members:
        m["fixed_lambda"] = 18.0
    assert viz.sweep_param(members) == "alpha"


def test_parse_image_dotyaml_artifact():
    # Older runs kept the launcher's ".yaml" glued to the method token; the
    # substring match still classifies them (discovery strips it for labels).
    info = viz.parse_experiment_name(
        "bregman_adabreg.yaml-sr90-CosineAnnealing"
    )
    assert info["method_class"] == "adabreg"
    assert info["sparsity"] == 90


def test_dense_baseline_labels_with_optimizer():
    info = viz.parse_experiment_name("dense_sgd-CosineAnnealing")
    assert info["method_class"] == "dense"
    assert info["optimizer"] == "sgd"
    # The optimizer is the only thing telling two dense runs apart.
    assert viz.make_label(info) == "SGD"


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


def _make_run(
    dir_path, acc_by_epoch, sparsity=0.9, lam=None, best_sparsity=None
):
    """Write a minimal run dir (config_tree.log + train_log.txt).

    ``best_sparsity`` also writes the ``results.json`` a finished run leaves
    behind, which is where the best checkpoint's sparsity is read from.
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    tree = "dummy\n" if lam is None else f"_bregman_lambda\n{lam}\n"
    (dir_path / "config_tree.log").write_text(tree)
    lines = [
        f"epoch: {e}, train_loss: 0.5, valid_loss: 0.5, "
        f"valid/MulticlassAccuracy: {a:.4f}, sparsity: {sparsity}"
        for e, a in acc_by_epoch
    ]
    (dir_path / "train_log.txt").write_text("\n".join(lines) + "\n")
    if best_sparsity is not None:
        (dir_path / "results.json").write_text(
            json.dumps(
                {
                    "best_checkpoint": {
                        "epoch": acc_by_epoch[0][0],
                        "overall_sparsity": best_sparsity,
                    }
                }
            )
        )


def test_discover_places_a_fixed_run_at_its_best_ckpt_sparsity(tmp_path):
    # The name has no target, so discovery reads where the *selected* checkpoint
    # landed — the last epoch's 0.6 describes weights nobody reports on.
    base = tmp_path / "runs" / "cifar10" / "resnet18" / "augmentation"
    _make_run(
        base / "bregman_linbreg_fixed-isr99-lam0.25" / "seed_42",
        [(1, 0.9), (2, 0.93)],
        sparsity=0.6,
        best_sparsity=0.972,
        lam=0.25,
    )
    found = viz.discover_experiments([str(base.parents[2])], ["*_fixed-*"])
    assert len(found) == 1
    _, info = found[0]
    assert info["best_ckpt_sparsity"] == pytest.approx(0.972)
    assert info["sparsity"] == 97
    assert info["fixed_lambda"] == 0.25


def test_discover_leaves_the_dense_baseline_without_a_sparsity(tmp_path):
    base = tmp_path / "runs" / "cifar10" / "resnet18" / "augmentation"
    _make_run(
        base / "dense_sgd-CosineAnnealing" / "seed_42",
        [(1, 0.9)],
        sparsity=0.0001,
    )
    found = viz.discover_experiments([str(base.parents[2])], ["dense_sgd*"])
    _, info = found[0]
    assert info["sparsity"] is None
    assert viz.method_tier(info) == viz.TIER_DENSE


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
            base
            / "cifar10"
            / "resnet18"
            / aug
            / "dense_sgd-CosineAnnealing"
            / "seed_42",
            [(1, 0.9)],
        )
    found = viz.discover_experiments(
        [str(base)], ["dense_sgd-CosineAnnealing"]
    )
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
