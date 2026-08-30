"""Tests for the visualization stack under ``src/vis/``.

Covers the three tables every figure reads — the method registry, the variant
registry and the metric registry — plus the two directory layouts discovery must
serve:

  * SV: ``<base>/<exp>`` (one run dir, no seed subdir)
  * Image: ``<base>/<dataset>/<model>/<augmentation>/<exp>/seed_<N>`` (grouped)

and the cross-seed aggregation (mean ± std).
"""

import colorsys
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.vis.encoding as encoding
from src.vis.encoding import Encoding, flavor_for, method_for
from src.vis.metrics import METRICS, metric_for, short_name
from src.vis.pruning_compare import read_tested_ckpt
from src.vis.runs import RunGroup, discover, load_csv_metrics, read_landed_sparsity, resolve_sparsity_level


def _group(name):
    """One parsed run, with no directories behind it."""
    return RunGroup.from_name(name)


def _labels(*names):
    """Every label for a set of run names, encoded as one set."""
    groups = [_group(n) for n in names]
    enc = Encoding(groups)
    return [enc.label(g) for g in groups]


def _styles(names, **kwargs):
    groups = [_group(n) for n in names]
    enc = Encoding(groups)
    return [enc.style(g, **kwargs) for g in groups]


def _label(name, *parts):
    """Expected label, with the symbols spelled the way encoding.py spells them."""
    return f"{name} ({', '.join(parts)})" if parts else name


def _sp(value):
    return f"{encoding.SPARSITY_SYM}={value}{encoding.pct_sym()}"


def _isp(value):
    return f"{encoding.INIT_SPARSITY_SYM}={value}{encoding.pct_sym()}"


def _lam(value):
    return f"{encoding.LAMBDA_SYM}={value}"


def _hue(hex_color):
    """The hue of a hex color, rounded so a rounding step does not read as a shift."""
    r, g, b = (int(hex_color.lstrip("#")[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return round(colorsys.rgb_to_hls(r, g, b)[0], 2)


@pytest.fixture
def show_init_sparsity(monkeypatch):
    """Turn the starting-sparsity tag on, so its visibility rule stays testable."""
    monkeypatch.setattr(encoding, "SHOW_INIT_SPARSITY", True)


# ---------------------------------------------------------------------------
# The registries
# ---------------------------------------------------------------------------


def test_every_method_row_is_complete():
    # A row carries every channel, so no method can reach a figure with a color
    # but no marker. The drift this replaces left dense and proxsgd half-declared.
    for m in encoding.METHODS:
        assert m.key and m.tokens and m.display
        assert all(t for t in m.tokens)
        assert m.color.startswith("#") and len(m.color) == 7
        assert m.marker
        assert m.family in ("dense", "baseline", "bregman")
    assert len({m.marker for m in encoding.METHODS}) == len(encoding.METHODS)
    assert len({m.color for m in encoding.METHODS}) == len(encoding.METHODS)


def test_every_method_ranks_and_no_key_is_missing():
    # The tuple order is the sort rank, so a method cannot exist without one.
    assert set(encoding.METHOD_SORT_RANK) == set(encoding.METHOD_BY_KEY)
    assert sorted(encoding.METHOD_SORT_RANK.values()) == list(range(len(encoding.METHODS)))


@pytest.mark.parametrize(
    "token, key, flavor",
    [
        ("bregman_linbreg", "linbreg", None),
        ("bregman_linbreg_fixed", "linbreg", "fixed"),
        ("bregman_linbreg_quantile", "linbreg", "quantile"),
        ("bregman_linbreg_quantile_progressive", "linbreg", "quantile_progressive"),
        ("bregman_adabreg_progressive", "adabreg", "progressive"),
        ("pruning_snip_iter", "snip", "iter"),
        ("pruning_snip", "snip", None),
        ("pruning_mag_unstruct", "pruning_unstruct", None),
        ("pruning_mag_struct", "pruning_struct", None),
        ("soft_threshold", "str", None),
        ("dense_sgd", "dense", None),
        # A trailing name segment is not part of the flavor.
        ("bregman_linbreg_progressive-ramp100_cubic", "linbreg", "progressive"),
        ("pruning_granet-ramp100_cubic", "granet", None),
        # Older names glued their settings on with underscores; the flavor still reads.
        ("bregman_linbreg_fixed_lam0.15_noScheduler", "linbreg", "fixed"),
        ("bregman_linbreg_fixed_sr95", "linbreg", "fixed"),
        ("bregman_adabreg_progressive_quantile", "adabreg", "progressive"),
        # A flavor must end on a word boundary, so this is no flavor at all.
        ("bregman_adabreg_movement", "adabreg", None),
    ],
)
def test_the_method_token_resolves_to_its_row(token, key, flavor):
    assert method_for(token).key == key
    got = flavor_for(token)
    assert (got.key if got else None) == flavor


def test_an_unknown_token_raises():
    # A silent fallback drew a sparse run as a dense baseline: no sparsity, wrong
    # bar group. `pruning_str` is STR's name before the soft_threshold rename.
    with pytest.raises(ValueError, match="registered method"):
        method_for("pruning_str")


@pytest.mark.parametrize("name", [p.stem for p in sorted(Path("configs/experiment/img").glob("*.yaml"))])
def test_every_img_experiment_config_resolves_to_a_method(name):
    # scripts/fabfile.py puts the config's stem at the head of every run name.
    assert method_for(name).key


@pytest.mark.parametrize("name", [p.stem for p in sorted(Path("configs/experiment/sv").glob("*.yaml"))])
def test_every_sv_experiment_config_resolves_to_a_method(name):
    assert method_for(name).key


def test_an_unregistered_variant_prints_itself_and_takes_no_style():
    # An ad hoc name suffix must survive into the label without a row of its own.
    v = encoding.variant_for("cls_scale2")
    assert v.display == "cls_scale2"
    assert v.linestyle == "-"
    assert v.color_shift == (0.0, 0.0, 0.0)


def test_a_variant_with_no_shift_leaves_the_method_colour_alone():
    # The HLS round trip truncates, so passing a zero shift through it moves the
    # colour by a step and lifts ProxSGD's pure black off zero.
    for m in encoding.METHODS:
        assert encoding._variant_color(m.color, "cls_scale2") == m.color
        assert encoding._variant_color(m.color, "constant_lr") == m.color
        assert encoding._variant_color(m.color, None) == m.color
    # A variant that does ask for a shift still gets one.
    assert encoding._variant_color(encoding.METHOD_BY_KEY["linbreg"].color, "fixed") != encoding.METHOD_BY_KEY["linbreg"].color


def test_every_metric_row_declares_known_plots():
    for m in METRICS:
        assert set(m.plots) <= set(("curves", "summary"))
        assert m.stage
        assert (m.ylim is None) == (not m.yticks)


def test_an_unregistered_metric_still_draws_a_curve():
    spec = metric_for("bregman/new_thing")
    assert spec.plots == ("curves",)
    assert spec.stage == "other"
    assert short_name("valid/MulticlassAccuracy") == "valid_multiclassaccuracy"


def test_the_sparsity_panels_read_the_pruned_keys():
    # Each pruner writes the benchmark sparsity under its own key, and the
    # whole-model `sparsity` under another. Only the pruned pair has a row, so
    # no figure can plot a quantity no method sparsifies (docs/image_benchmarks.md).
    pruned = ("pruning/sparsity", "bregman/pruned_sparsity")
    for key in pruned:
        assert metric_for(key).ylim == (0.7, 1.005)
        assert metric_for(key).yticks
    assert {m.label for m in METRICS if m.key in pruned} == {r"$\mathsf{s}(\theta)$"}
    for whole_model in ("sparsity", "bregman/sparsity"):
        assert whole_model not in {m.key for m in METRICS}


def test_the_verification_metrics_draw_nothing_here():
    # They come from the score files, not the training logs, so routing them to a
    # curve produced an empty figure and a misleading "no data" line.
    assert metric_for("EER").plots == ()
    assert metric_for("minDCF").plots == ()


# ---------------------------------------------------------------------------
# Name parsing
# ---------------------------------------------------------------------------


def test_parse_image_pruning_name():
    g = _group("pruning_mag_unstruct-sr95-CosineAnnealing")
    assert g.method.key == "pruning_unstruct"
    assert g.sparsity == 95
    assert g.scheduler == "CosineAnnealing"


@pytest.mark.parametrize(
    "name, method, variant, sparsity",
    [
        ("bregman_adabreg-sr90-CosineAnnealing", "adabreg", None, 90),
        ("bregman_adabreg_fixed-lam18-CosineAnnealing", "adabreg", "fixed", None),
        ("bregman_adabreg_progressive-sr99-no_scheduler", "adabreg", "progressive", 99),
        ("bregman_linbreg_progressive-sr99-ReduceLROnPlateau", "linbreg", "progressive", 99),
        ("dense_sgd-CosineAnnealing", "dense", None, None),
    ],
)
def test_parse_image_method_variants(name, method, variant, sparsity):
    g = _group(name)
    assert g.method.key == method
    assert g.variant == variant
    assert g.sparsity == sparsity


@pytest.mark.parametrize(
    "name, method, dataset, model, sparsity",
    [
        ("bregman_adabreg-resnet18-cifar10-bs128-sr99-classifier_10k-epshi", "adabreg", "cifar10", "resnet18", 99),
        ("pruning_mag_struct-wrn28_10-cifar100-bs128-sr90-classifier_10k", "pruning_struct", "cifar100", "wrn28_10", 90),
        ("bregman_adabreg-ramp80_constant-resnet18-cifar10-bs128-sr95", "adabreg", "cifar10", "resnet18", 95),
        ("dense_sgd-resnet18-cifar10-bs128", "dense", "cifar10", "resnet18", None),
    ],
)
def test_parse_image_fabfile_name(name, method, dataset, model, sparsity):
    # Fabfile embeds model/dataset and puts -sr<NN> mid-name; the curated tree has no augmentation dir.
    g = _group(name)
    assert g.method.key == method
    assert g.dataset == dataset
    assert g.model == model
    assert g.sparsity == sparsity


@pytest.mark.parametrize(
    "name, initial_sparsity, sparsity",
    [
        ("bregman_adabreg-isr99-sr90-CosineAnnealing", 99, 90),
        ("bregman_adabreg_fixed-isr0-lam18-CosineAnnealing", 0, None),
        ("bregman_adabreg-resnet18-cifar10-bs128-isr50-sr99", 50, 99),
        ("bregman_adabreg-sr90-CosineAnnealing", None, 90),
    ],
)
def test_parse_image_initial_sparsity(name, initial_sparsity, sparsity):
    # -isr<NN> is the starting sparsity and never shadows the -sr<NN> target.
    g = _group(name)
    assert g.initial_sparsity == initial_sparsity
    assert g.sparsity == sparsity
    assert g.method.key == "adabreg"


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
    g = _group(name)
    assert g.fixed_lambda == 18.0
    assert g.sparsity is None
    assert g.initial_sparsity == 50
    assert g.variant == "fixed"
    assert g.is_fixed_lambda


@pytest.mark.parametrize(
    "name,lam",
    [
        # scripts/fabfile.py appends -tf<N> and its suffix after the lambda.
        ("bregman_linbreg_fixed-resnet50-cifar100-bs128-lam0.15-constant_lr", 0.15),
        ("bregman_linbreg_fixed-resnet50-cifar100-bs128-lam0.15-tf100", 0.15),
        ("sv_bregman_linbreg_fixed-wespeaker_resnet34-vox2-bs128-lam0.15-poor_init", 0.15),
        # run_naming.lambda_token spells a small lambda with "%g", so the value
        # itself carries a "-": the token must not stop inside the exponent.
        ("bregman_linbreg_fixed-resnet50-cifar100-bs128-lam1e-05", 1e-05),
        ("bregman_linbreg_fixed-isr50-lam1e-05", 1e-05),
        ("bregman_linbreg_fixed-isr50-lam1e-05-CosineAnnealing", 1e-05),
        ("sv_bregman_linbreg_fixed-wespeaker_resnet34-vox2-bs128-lam5e-05", 5e-05),
    ],
)
def test_a_lambda_token_ends_where_the_value_ends(name, lam):
    # The value stops at the segment boundary, and a trailing tag never reaches it.
    g = _group(name)
    assert g.fixed_lambda == lam
    assert g.is_fixed_lambda
    assert g.sparsity is None


@pytest.mark.parametrize(
    "name",
    [
        # The launcher spelled a fixed run by its target before run_naming did.
        "bregman_linbreg_fixed-resnet50-cifar100-bs128-isr99-sr90",
        "sv_bregman_adabreg_fixed-wespeaker_resnet34-vox2-bs128-sr90",
        # An older name glued the lambda on with underscores, and its value
        # contradicts the config, so the token is not read as one.
        "bregman_linbreg_fixed_lam0.15_noScheduler",
    ],
)
def test_a_fixed_lambda_name_without_its_lambda_raises(name):
    # Its sparsity is an outcome, so a target in the name states a number the run
    # never aimed at. scripts/retag_fixed_lambda_runs.py renames such runs.
    with pytest.raises(ValueError, match="retag_fixed_lambda_runs"):
        _group(name)


@pytest.mark.parametrize(
    "name, tag",
    [
        # scripts/fabfile.py appends the sweep suffix and the ramp end last.
        ("soft_threshold-resnet50-cifar100-bs128-wd5e-05", "wd5e-05"),
        ("pruning_granet-ramp100_cubic-resnet50-cifar100-bs128-sr90-tf50", "tf50"),
        ("dense_sgd-resnet50-cifar100-bs128-constant_lr", "constant_lr"),
        ("bregman_adabreg-resnet18-cifar10-bs128-sr99-classifier_10k", "classifier_10k"),
        ("bregman_linbreg-resnet50-cifar100-bs128-isr99-sr95", None),
    ],
)
def test_an_image_name_keeps_its_trailing_tag(name, tag):
    # Two STR runs differ in nothing but the weight decay; dropping the suffix
    # gave both one label and one style, so the pair collapsed to one curve.
    assert _group(name).variant == tag


def test_two_weight_decays_are_two_runs():
    names = [f"soft_threshold-resnet50-cifar100-bs128-wd{v}" for v in ("5e-05", "0.0001")]
    groups = [_group(n) for n in names]
    enc = Encoding(groups)
    assert len({enc.label(g) for g in groups}) == 2


def test_a_fabfile_image_name_states_its_model_and_dataset():
    with pytest.raises(AssertionError, match="<method>-<model>-<dataset>"):
        _group("dense_sgd-bs128-sr90")


def test_one_shot_pruning_is_its_own_variant():
    # sv_pruning_mag_unstruct_onetime and its repeated sibling differ in nothing
    # else, so without a row of its own the pair shared colour, dash and label.
    stem = "wespeaker_resnet34-vox2-bs128-sr90"
    groups = [_group(f"sv_pruning_mag_unstruct{f}-{stem}") for f in ("", "_onetime")]
    enc = Encoding(groups)
    assert len({enc.label(g) for g in groups}) == 2
    assert len({str(enc.style(g)) for g in groups}) == 2


def test_every_configured_fixed_lambda_round_trips():
    # The launcher writes the name; the parser reads it. Neither may drift, so
    # walk every value the table can hand to lambda_token.
    from src.utils.bregman_utils import BREGMAN_LAMBDA_CONFIGS
    from src.utils.run_naming import lambda_token

    for method in BREGMAN_LAMBDA_CONFIGS.values():
        for anchor in method.values():
            value = float(anchor["fixed_lambda"])
            token = lambda_token(value)
            for suffix in ("", "-constant_lr", "-tf100"):
                name = f"bregman_linbreg_fixed-resnet50-cifar100-bs128{token}{suffix}"
                assert _group(name).fixed_lambda == value, name


def test_parse_image_dotyaml_artifact():
    # Older runs kept the launcher's ".yaml" glued to the method token; the
    # substring match still classifies them (discovery strips it for labels).
    g = _group("bregman_adabreg.yaml-sr90-CosineAnnealing")
    assert g.method.key == "adabreg"
    assert g.sparsity == 90


def test_each_dense_baseline_reads_as_its_own_optimizer():
    # The optimizer is the only thing telling two dense runs apart, so each
    # spelling gets its own row rather than one shared "Dense".
    for name, expected in (
        ("dense_sgd-CosineAnnealing", "SGD"),
        ("sv_dense_adamw-wespeaker_ecapa_tdnn-cnceleb-bs256", "AdamW"),
        ("sv_dense_wespeaker-wespeaker_ecapa_tdnn-cnceleb-bs256", "SGD"),
        ("sv_vanilla-wespeaker_ecapa_tdnn-cnceleb-bs256", "AdamW"),
    ):
        g = _group(name)
        assert g.is_dense
        assert Encoding([g]).label(g) == expected


def test_parse_sv_name_unchanged():
    g = _group("sv_bregman_adabreg-wespeaker_ecapa_tdnn-multi_sv-bs64-ep30-augFalse-sr99")
    assert g.method.key == "adabreg"
    assert g.model == "wespeaker_ecapa_tdnn"
    assert g.sparsity == 99
    # The wespeaker backbone regex must still win over the image fallback.
    g2 = _group("sv_bregman_adabreg_progressive-wespeaker_resnet34-multi_sv-bs32-ep30-augFalse-sr99")
    assert g2.model == "wespeaker_resnet34"
    assert g2.method.key == "adabreg"


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def test_initial_sparsity_labels_only_when_it_varies(show_init_sparsity):
    assert _labels(*[f"bregman_adabreg-isr{i}-sr99-CosineAnnealing" for i in (0, 99)]) == [
        _label("AdaBregSGap", _isp(0)),
        _label("AdaBregSGap", _isp(99)),
    ]
    assert _labels("bregman_adabreg-isr99-sr99-CosineAnnealing") == ["AdaBregSGap"]


def test_initial_sparsity_is_off_by_default():
    # SHOW_INIT_SPARSITY drops the tag even where it varies. Two runs that differ
    # only in their start then read alike; color, marker and dash still part them.
    assert _labels(*[f"bregman_adabreg-isr{i}-sr99-CosineAnnealing" for i in (0, 99)]) == [
        "AdaBregSGap",
        "AdaBregSGap",
    ]


def test_fixed_lambda_label_shows_the_value_next_to_init(show_init_sparsity):
    # Lambda replaces the old "(fixed)" tag and must not swallow the sweep tag.
    assert _labels(*[f"bregman_adabreg_fixed-isr{i}-lam0.004" for i in (0, 99)]) == [
        _label("AdaBreg", _isp(0), _lam("0.004")),
        _label("AdaBreg", _isp(99), _lam("0.004")),
    ]


def test_proxsgd_carries_the_bregman_knobs(show_init_sparsity):
    # ProxSGD inherits the Bregman parent config, so its runs sweep isr too.
    assert _labels(*[f"proxsgd-isr{i}-sr99-CosineAnnealing" for i in (0, 99)]) == [
        _label("ProxSGD", _isp(0)),
        _label("ProxSGD", _isp(99)),
    ]


def test_lambda_always_shows_even_for_a_lone_fixed_run():
    # Lambda identifies a fixed-lambda run, so it is never gated on variation:
    # without it the label is indistinguishable from the adaptive run's.
    assert _labels(*[f"bregman_adabreg_fixed-lam{v}-CosineAnnealing" for v in ("10", "18")]) == [
        _label("AdaBreg", _lam(10)),
        _label("AdaBreg", _lam(18)),
    ]
    assert _labels("bregman_adabreg_fixed-lam18-CosineAnnealing") == [_label("AdaBreg", _lam(18))]


@pytest.mark.parametrize(
    "name, expected",
    [
        ("bregman_linbreg-isr99-sr99", "LinBregSGap"),
        ("bregman_linbreg_progressive-sr99", "LinBregSGap + Ramp"),
        ("bregman_linbreg_quantile-isr99-sr99", "LinBregTopK"),
        ("bregman_linbreg_quantile_progressive-sr99", "LinBregTopK + Ramp"),
        ("bregman_adabreg_quantile-isr99-sr99", "AdaBregTopK"),
        # A method with no pair entry is untouched.
        ("pruning_granet-sr99", "GraNet"),
    ],
)
def test_named_pairs_replace_the_variant_tag(name, expected):
    # The four Bregman flavors are methods, not ablations, so the name carries the
    # flavor and no "(Quant.)" or "(Prog.)" tag repeats it.
    assert _labels(name) == [expected]


def test_a_fixed_lambda_run_keeps_the_bare_method_name():
    # It drives neither controller, so neither SGap nor TopK applies to it.
    assert _labels("bregman_adabreg_fixed-lam18") == [_label("AdaBreg", _lam(18))]


def test_a_named_pair_keeps_an_ad_hoc_tag():
    # The name carries the flavor. A suffix tag is a different axis and must survive.
    g = _group("bregman_linbreg_quantile-isr99-sr99")
    g.variant = "constant_lr"
    assert Encoding([g]).label(g) == _label("LinBregTopK", encoding.VARIANT_BY_KEY["constant_lr"].display)


def test_one_target_hides_the_sparsity_tag():
    # Every run that holds a target sits at the same one, so naming it says
    # nothing. The dense baseline holds none and must not read as variation.
    assert _labels("bregman_linbreg-isr99-sr99", "bregman_linbreg_quantile-isr99-sr99", "dense_sgd") == [
        "LinBregSGap",
        "LinBregTopK",
        "SGD",
    ]


def test_two_targets_keep_the_sparsity_tag():
    assert _labels(*[f"bregman_linbreg-isr99-sr{s}" for s in (90, 99)]) == [
        _label("LinBregSGap", _sp(90)),
        _label("LinBregSGap", _sp(99)),
    ]


def test_a_fixed_lambda_run_always_shows_the_sparsity_it_reached():
    # That value is an outcome, not a setpoint. It stays out of the target set, so
    # it cannot force the tag onto the runs that do hold a target, and it always
    # shows: two lambdas that land apart must read apart.
    adaptive = _group("bregman_linbreg-isr99-sr99")
    fixed = _group("bregman_linbreg_fixed-isr99-lam0.9")
    fixed.sparsity = 95
    enc = Encoding([adaptive, fixed])
    assert enc.label(adaptive) == "LinBregSGap"
    assert enc.label(fixed) == _label("LinBreg", _sp(95), _lam("0.9"))


def test_bar_ticks_carry_both_sparsity_and_lambda():
    # The group header names the tier, not the sparsity, so the tick shows both.
    g = _group("bregman_linbreg_fixed-lam0.25")
    g.sparsity = 97
    assert Encoding([g]).label(g) == _label("LinBreg", _sp(97), _lam("0.25"))


def test_without_sparsity_drops_the_level_and_keeps_the_rest():
    # A figure with sparsity on its axis must not repeat it in every label.
    g = _group("bregman_linbreg_fixed-isr99-lam0.9")
    g.sparsity = 95
    bare = g.without_sparsity()
    assert bare.sparsity is None
    assert bare.fixed_lambda == 0.9
    assert Encoding([bare]).label(bare) == _label("LinBreg", _lam("0.9"))


# ---------------------------------------------------------------------------
# Tiers and ordering
# ---------------------------------------------------------------------------


def test_fixed_lambda_run_is_not_dense():
    # Regression: a *_fixed run carries no -sr token, and inferring "dense" from
    # that put it in the Dense bar group, compared against the wrong baseline.
    fixed = _group("bregman_linbreg_fixed-isr99-lam0.25")
    assert fixed.sparsity is None
    assert not fixed.is_dense
    assert _group("dense_sgd-CosineAnnealing").is_dense


def test_method_tier_separates_the_four_families():
    tier = {
        "dense_sgd-CosineAnnealing": encoding.TIER_DENSE,
        "pruning_mag_unstruct-sr99-CosineAnnealing": encoding.TIER_SPARSE_BASELINE,
        "bregman_linbreg_fixed-isr99-lam0.25": encoding.TIER_FIXED,
        "bregman_linbreg-isr99-sr99-CosineAnnealing": encoding.TIER_ADAPTIVE,
    }
    for name, expected in tier.items():
        assert _group(name).tier == expected


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
    ordered = sorted((_group(n) for n in names), key=lambda g: g.sort_key)
    assert [g.dirname for g in ordered] == [
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


@pytest.mark.parametrize("method", ["bregman_adabreg", "bregman_adabreg_fixed"])
def test_initial_sparsity_sweep_varies_every_channel(method):
    # Labels alone don't separate a sweep, and neither does color alone: the
    # line shapes have to differ too, or the curves are unreadable in print.
    level = "-lam18" if method.endswith("_fixed") else "-sr99"
    styles = _styles([f"{method}-isr{i}{level}-CosineAnnealing" for i in (0, 50, 75, 99)])
    assert len({c for c, _, _ in styles}) == 4
    assert len({m for _, m, _ in styles}) == 4
    assert len({str(ls) for _, _, ls in styles}) == 4


def test_lambda_sweep_varies_every_channel():
    # With no -sr token, lambda is the only thing telling fixed runs apart.
    styles = _styles([f"bregman_adabreg_fixed-lam{v}-CosineAnnealing" for v in ("2", "4", "10", "18")])
    assert len({c for c, _, _ in styles}) == 4
    assert len({m for _, m, _ in styles}) == 4
    assert len({str(ls) for _, _, ls in styles}) == 4


@pytest.mark.parametrize("method", ["linbreg", "adabreg"])
def test_the_sweep_ramp_spans_the_lightness_window(method):
    # The window is absolute. A base color that already sits light or dark cannot
    # push the first rank off the top of the scale or the last one to near black.
    light, dark = encoding.SWEEP_LIGHTNESS
    styles = _styles([f"bregman_{method}_fixed-isr99-lam{v}" for v in ("0.9", "1.1", "1.25")])
    levels = [encoding._lightness(c) for c, _, _ in styles]
    assert levels[0] == pytest.approx(light, abs=0.01)
    assert levels[-1] == pytest.approx(dark, abs=0.01)


def test_fixed_lambda_runs_sweep_over_lambda():
    # The realized sparsity differs per lambda. Grouping on it split the sweep into
    # three groups of one, and all three runs then drew in one color, marker and dash.
    groups = [_group(f"bregman_linbreg_fixed-isr99-lam{v}") for v in ("0.9", "1.1", "1.25")]
    for g, realized in zip(groups, (95, 96, 97)):
        g.sparsity = realized
    enc = Encoding(groups)
    colors, markers, dashes = zip(*(enc.style(g) for g in groups))
    assert len(set(colors)) == 3
    assert len(set(markers)) == 3
    assert len(set(dashes)) == 3
    # The ramp moves lightness only, so the fixed family keeps one hue.
    assert len({_hue(c) for c in colors}) == 1
    assert _hue(colors[0]) != _hue(encoding.METHOD_BY_KEY["linbreg"].color)


def test_the_starting_sparsity_wins_the_sweep_over_lambda():
    # Both fields vary. The starting sparsity claims the channels, so two runs
    # sharing a start share a color even where their lambdas differ.
    groups = [
        _group(f"bregman_linbreg_fixed-isr{i}-lam{v}")
        for i, v in ((0, "0.9"), (0, "1.1"), (99, "1.25"))
    ]
    enc = Encoding(groups)
    colors = [enc.style(g)[0] for g in groups]
    assert colors[0] == colors[1] != colors[2]


def test_lambda_takes_the_sweep_when_the_start_is_constant():
    groups = [_group(f"bregman_linbreg_fixed-isr99-lam{v}") for v in ("0.9", "1.1")]
    enc = Encoding(groups)
    assert enc.style(groups[0])[0] != enc.style(groups[1])[0]


def test_marker_by_method_leaves_the_sweep_the_dash_pattern():
    # Where sparsity is the x-axis the method owns the marker, so the swept
    # field has to keep a channel of its own.
    styles = _styles([f"bregman_linbreg-isr{i}-sr99" for i in (0, 99)], marker_by="method")
    assert {m for _, m, _ in styles} == {encoding.METHOD_BY_KEY["linbreg"].marker}
    assert len({str(ls) for _, _, ls in styles}) == 2


def test_marker_by_method_gives_the_dense_baseline_its_own_marker():
    # Regression: dense carried no marker row, fell back to "o" and collided with
    # AdaBreg on every trend plot that puts sparsity on the x-axis.
    styles = _styles(["dense_sgd", "bregman_adabreg-sr99"], marker_by="method")
    assert styles[0][1] != styles[1][1]


def test_alpha_sweep_still_excludes_fixed_runs():
    # Fixed runs carry the alpha default (1.0), so pooling them fakes a sweep.
    stem = "wespeaker_resnet34-vox2-bs128"
    groups = [_group(f"sv_bregman_adabreg-{stem}-sr90-alpha{a}") for a in ("0.5", "2.0")]
    fixed = _group(f"sv_bregman_adabreg_fixed-{stem}-lam18")
    enc = Encoding(groups + [fixed])
    assert len({enc.style(g)[0] for g in groups}) == 2
    # Without a sweep, a fixed run keeps the star that marks its family.
    assert enc.style(fixed)[1] == encoding.FIXED_LAMBDA_MARKER


def test_no_sweep_hands_the_channels_back():
    # A figure that encodes the swept field on an axis takes the channels back.
    groups = [_group(f"bregman_adabreg-isr{i}-sr99") for i in (0, 99)]
    enc = Encoding(groups, sweep=False)
    assert {enc.style(g)[0] for g in groups} == {encoding.METHOD_BY_KEY["adabreg"].color}


def test_an_encoding_refuses_a_run_it_never_saw():
    # Labels describe the set they were computed over. Answering for an outsider
    # would tag it against the wrong set instead of failing.
    enc = Encoding([_group("bregman_adabreg-sr90")])
    with pytest.raises(KeyError):
        enc.label(_group("bregman_linbreg-sr90"))


def test_style_rejects_an_unknown_marker_channel():
    g = _group("bregman_adabreg-sr90")
    with pytest.raises(AssertionError):
        Encoding([g]).style(g, marker_by="dataset")


# ---------------------------------------------------------------------------
# Realized sparsity on the CSV path
# ---------------------------------------------------------------------------


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
    resolve_sparsity_level(df)
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
    resolve_sparsity_level(df)
    assert df["sparsity"].isna().all()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


# Written to overall_sparsity. No test asserts it, so a reader that goes back to
# that key reads 10% and every assertion on the pruned value fails.
WRONG_SPARSITY = 0.10


def _make_run(dir_path, acc_by_epoch, sparsity=0.9, best_sparsity=None):
    """Write a minimal run dir (config_tree.log + train_log.txt).

    ``config_tree.log`` only marks the directory as a run; the lambda comes
    from the name. ``best_sparsity`` also writes the ``results.json`` a finished
    run leaves behind, which holds the best checkpoint's pruned sparsity.
    ``overall_sparsity`` sits next to it and must never reach a label.
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "config_tree.log").write_text("dummy\n")
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
                        "overall_sparsity": WRONG_SPARSITY,
                        "pruned_sparsity": best_sparsity,
                    }
                }
            )
        )


def test_discover_places_a_fixed_run_where_its_best_ckpt_landed(tmp_path):
    # The name has no target, so discovery reads where the *selected* checkpoint
    # landed — the last epoch's 0.6 describes weights nobody reports on. The
    # figure is the pruned one; overall_sparsity sits 5 pt below and must not win.
    base = tmp_path / "runs" / "cifar10" / "resnet18" / "augmentation"
    _make_run(
        base / "bregman_linbreg_fixed-isr99-lam0.25" / "seed_42",
        [(1, 0.9), (2, 0.93)],
        sparsity=0.6,
        best_sparsity=0.972,
    )
    found = discover([str(base.parents[2])], ["*_fixed-*"])
    assert len(found) == 1
    g = found[0]
    assert g.landed_sparsity == pytest.approx(0.972)
    assert g.landed_sparsity_std is None  # one seed carries no spread
    assert g.sparsity == 97
    assert g.sparsity_is_outcome
    assert g.fixed_lambda == 0.25


def test_discover_averages_the_landed_sparsity_over_seeds(tmp_path):
    # A fixed-lambda run lands where lambda takes it, and each seed lands
    # somewhere else. One seed is not the run, so the level is the seed mean.
    base = tmp_path / "runs" / "cifar10" / "resnet18" / "augmentation"
    exp = base / "bregman_linbreg_fixed-isr99-lam0.25"
    for seed, landed in ((42, 0.94), (1994, 0.96)):
        _make_run(exp / f"seed_{seed}", [(1, 0.9), (2, 0.93)], best_sparsity=landed)
    g = discover([str(base.parents[2])], ["*_fixed-*"])[0]
    assert g.seeds == [42, 1994]
    assert g.landed_sparsity == pytest.approx(0.95)
    assert g.landed_sparsity_std == pytest.approx(np.std([0.94, 0.96], ddof=1))
    assert g.sparsity == 95


def _fixed_lambda_label(tmp_path, landed_by_seed):
    """The legend of one fixed-lambda run, given where each seed landed."""
    base = tmp_path / "runs" / "cifar10" / "resnet18" / "augmentation"
    exp = base / "bregman_linbreg_fixed-isr99-lam0.25"
    for seed, landed in landed_by_seed:
        _make_run(exp / f"seed_{seed}", [(1, 0.9), (2, 0.93)], best_sparsity=landed)
    groups = discover([str(base.parents[2])], ["*_fixed-*"])
    return Encoding(groups).label(groups[0])


def test_a_fixed_lambda_legend_carries_the_seed_spread(tmp_path):
    # Lambda decides where the run lands, and the seeds disagree. A rounded 95%
    # hides that, so the legend prints the mean and the spread behind it.
    label = _fixed_lambda_label(tmp_path, [(42, 0.94), (1994, 0.96)])
    spread = 100 * np.std([0.94, 0.96], ddof=1)
    assert label == _label("LinBreg", _sp(rf"95.00$\pm${spread:.2f}"), _lam("0.25"))


def test_one_seed_prints_no_spread(tmp_path):
    # One seed carries no spread, so the run reports only where it landed.
    assert _fixed_lambda_label(tmp_path, [(42, 0.9532)]) == _label("LinBreg", _sp("95.32"), _lam("0.25"))


def test_a_target_run_keeps_the_bare_integer(tmp_path):
    # The target is the setpoint, and every seed reaches it. Two decimals and a
    # +/- 0.00 would only add noise, so the tag stays the level itself.
    base = tmp_path / "runs" / "cifar10" / "resnet18" / "augmentation"
    for seed in (42, 1994):
        _make_run(base / "bregman_linbreg-isr99-sr90" / f"seed_{seed}", [(1, 0.9), (2, 0.93)], best_sparsity=0.9)
        _make_run(base / "bregman_linbreg-isr99-sr99" / f"seed_{seed}", [(1, 0.9), (2, 0.93)], best_sparsity=0.99)
    found = discover([str(base.parents[2])], ["bregman_*"])
    enc = Encoding(found)
    assert [enc.label(g) for g in found] == [
        _label("LinBregSGap", _sp(90)),
        _label("LinBregSGap", _sp(99)),
    ]


def test_discover_averages_over_the_seeds_that_finished(tmp_path):
    # A seed still running has written no results.json. It must not drag the
    # mean, and the seeds that did finish still carry a spread.
    base = tmp_path / "runs" / "cifar10" / "resnet18" / "augmentation"
    exp = base / "bregman_linbreg_fixed-isr99-lam0.25"
    for seed, landed in ((42, 0.94), (1994, 0.96), (2026, None)):
        _make_run(exp / f"seed_{seed}", [(1, 0.9), (2, 0.93)], best_sparsity=landed)
    g = discover([str(base.parents[2])], ["*_fixed-*"])[0]
    assert g.seeds == [42, 1994, 2026]
    assert g.landed_sparsity == pytest.approx(0.95)
    assert g.landed_sparsity_std == pytest.approx(np.std([0.94, 0.96], ddof=1))


def test_discover_leaves_the_dense_baseline_without_a_sparsity(tmp_path):
    base = tmp_path / "runs" / "cifar10" / "resnet18" / "augmentation"
    _make_run(base / "dense_sgd-CosineAnnealing" / "seed_42", [(1, 0.9)], sparsity=0.0001)
    g = discover([str(base.parents[2])], ["dense_sgd*"])[0]
    assert g.sparsity is None
    assert g.tier == encoding.TIER_DENSE


def test_discover_groups_image_seeds(tmp_path):
    base = tmp_path / "train" / "runs"
    exp = base / "cifar10" / "resnet18" / "augmentation" / "bregman_adabreg-sr90-CosineAnnealing"
    _make_run(exp / "seed_42", [(1, 0.80), (2, 0.90)])
    _make_run(exp / "seed_43", [(1, 0.82), (2, 0.94)])

    found = discover([str(base)], ["bregman_adabreg-sr90-CosineAnnealing"])
    assert len(found) == 1
    g = found[0]
    assert g.seeds == [42, 43]
    assert len(g.dirs) == 2
    assert g.dataset == "cifar10"
    assert g.model == "resnet18"
    assert g.augmentation is True


def test_discover_separates_augmentation(tmp_path):
    # Same <exp> name under augmentation vs no_augmentation must stay distinct.
    base = tmp_path / "runs"
    for aug in ("augmentation", "no_augmentation"):
        _make_run(base / "cifar10" / "resnet18" / aug / "dense_sgd-CosineAnnealing" / "seed_42", [(1, 0.9)])
    found = discover([str(base)], ["dense_sgd-CosineAnnealing"])
    assert len(found) == 2
    assert sorted(g.augmentation for g in found) == [False, True]


def test_discover_strips_dotyaml_from_label(tmp_path):
    base = tmp_path / "runs"
    _make_run(
        base / "cifar10" / "resnet18" / "augmentation" / "bregman_adabreg.yaml-sr90-CosineAnnealing" / "seed_42",
        [(1, 0.9)],
    )
    # Clean pattern matches despite the ".yaml" on disk; dirname is normalized.
    found = discover([str(base)], ["bregman_adabreg-sr90-CosineAnnealing"])
    assert len(found) == 1
    assert found[0].dirname == "bregman_adabreg-sr90-CosineAnnealing"


def test_discover_sv_single_run(tmp_path):
    base = tmp_path / "logs" / "train" / "runs"
    exp = "sv_vanilla-wespeaker_ecapa_tdnn-multi_sv-bs64"
    _make_run(base / exp, [(1, 0.70), (2, 0.75)])

    found = discover([str(base)], ["sv_vanilla*"])
    assert len(found) == 1
    assert found[0].seeds == [None]
    assert found[0].dirs == [str(base / exp)]


def test_discover_ignores_nonmatching(tmp_path):
    base = tmp_path / "runs"
    root = base / "cifar10" / "resnet18" / "augmentation"
    _make_run(root / "dense_sgd-CosineAnnealing" / "seed_42", [(1, 0.9)])
    _make_run(root / "pruning_mag_unstruct-sr90-CosineAnnealing" / "seed_42", [(1, 0.8)])
    found = discover([str(base)], ["dense_sgd*"])
    assert len(found) == 1
    assert found[0].method.key == "dense"


# ---------------------------------------------------------------------------
# Cross-seed reduction
# ---------------------------------------------------------------------------


def _two_seed_group(tmp_path, seed_curves):
    base = tmp_path / "runs" / "cifar10" / "resnet18" / "augmentation"
    for seed, curve in seed_curves:
        _make_run(base / "bregman_adabreg-sr90" / f"seed_{seed}", curve)
    return discover([str(base.parents[2])], ["bregman_adabreg-sr90"])[0]


def test_curve_returns_the_seed_mean_and_spread(tmp_path):
    g = _two_seed_group(tmp_path, [(1, [(1, 0.5), (2, 0.7), (3, 0.9)]), (2, [(1, 0.6), (2, 0.8), (3, 1.0)])])
    x, mean, std = g.curve("valid/MulticlassAccuracy", "train_log")
    assert list(x) == [1.0, 2.0, 3.0]
    assert mean == pytest.approx([0.55, 0.75, 0.95])
    # sample std (ddof=1) of {a, a+0.1} is 0.1/sqrt(2)
    assert std == pytest.approx([0.0707107] * 3, abs=1e-5)


def test_a_single_seed_curve_has_no_band(tmp_path):
    g = _two_seed_group(tmp_path, [(1, [(1, 0.5), (2, 0.7)])])
    x, mean, std = g.curve("valid/MulticlassAccuracy", "train_log")
    assert std is None
    assert mean == pytest.approx([0.5, 0.7])


def test_a_missing_metric_yields_no_curve(tmp_path):
    g = _two_seed_group(tmp_path, [(1, [(1, 0.5), (2, 0.7)])])
    assert g.curve("EER", "train_log") is None
    assert g.scalar("EER", "train_log") is None


def test_scalar_reduces_each_seed_to_its_best_or_last_epoch(tmp_path):
    g = _two_seed_group(tmp_path, [(1, [(1, 0.80), (2, 0.92), (3, 0.88)]), (2, [(1, 0.70), (2, 0.85), (3, 0.90)])])
    mean, std, vals = g.scalar("valid/MulticlassAccuracy", "train_log", reduce="max")
    assert sorted(vals) == pytest.approx([0.90, 0.92])
    assert mean == pytest.approx(0.91)
    assert std == pytest.approx(np.std([0.90, 0.92], ddof=1))
    _, _, last = g.scalar("valid/MulticlassAccuracy", "train_log", reduce="last")
    assert sorted(last) == pytest.approx([0.88, 0.90])


def test_frames_are_read_once_per_source(tmp_path):
    # A multi-panel figure asks for one metric at a time. Without the cache every
    # panel re-reads every seed, and the cache must stay out of the record the
    # pruning summary serializes.
    from dataclasses import asdict, replace

    g = _two_seed_group(tmp_path, [(1, [(1, 0.8), (2, 0.9)]), (2, [(1, 0.7), (2, 0.85)])])
    first = g.frames("train_log")
    assert g.frames("train_log") is first
    assert "_frames" not in asdict(g)
    assert replace(g, sparsity=None)._frames == {}


def test_scalar_rejects_an_unknown_reduction(tmp_path):
    g = _two_seed_group(tmp_path, [(1, [(1, 0.8)])])
    with pytest.raises(AssertionError):
        g.scalar("valid/MulticlassAccuracy", "train_log", reduce="median")


# ---------------------------------------------------------------------------
# Reading a run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("criterion", ["CrossEntropyLoss", "LogSoftmaxWrapper"])
def test_the_loss_reaches_the_registry_whichever_criterion_wrote_it(tmp_path, criterion):
    # The image task logs CrossEntropyLoss and SV logs LogSoftmaxWrapper. Without
    # both aliases an image run drew an empty train_loss panel.
    version = tmp_path / "csv" / "version_0"
    version.mkdir(parents=True)
    (version / "metrics.csv").write_text(
        f"step,epoch,train/{criterion},valid/{criterion}\n0,0,2.5,2.4\n1,1,1.5,1.6\n"
    )
    df = load_csv_metrics(str(tmp_path))
    assert list(df["train_loss"]) == [2.5, 1.5]
    assert list(df["valid_loss"]) == [2.4, 1.6]


def _make_tested_run(run_dir, ckpt_name="epoch190-metric_valid0.73-sr0.988.ckpt", logged=True, on_disk=True):
    """A run dir holding the train.log line and the checkpoint it names."""
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "last.ckpt").write_text("x")
    if on_disk:
        (run_dir / "checkpoints" / ckpt_name).write_text("x")
    line = f"[INFO] - Test ckpt path: /vault/results/{run_dir.name}/checkpoints/{ckpt_name}\n"
    (run_dir / "train.log").write_text("[INFO] - Training done\n" + (line if logged else ""))
    return run_dir


def test_the_tested_checkpoint_comes_from_the_log(tmp_path):
    # The log states what the reported accuracy and mask describe. last.ckpt sits
    # right beside it and holds a different epoch's weights.
    run = _make_tested_run(tmp_path / "run")
    got = read_tested_ckpt(str(run))
    assert os.path.basename(got) == "epoch190-metric_valid0.73-sr0.988.ckpt"


def test_a_run_that_never_tested_raises(tmp_path):
    run = _make_tested_run(tmp_path / "run", logged=False)
    with pytest.raises(ValueError, match="never ran test"):
        read_tested_ckpt(str(run))


def test_a_cleaned_away_checkpoint_raises(tmp_path):
    # scripts/cleanup_checkpoints.sh keeps last.ckpt. Standing that in reported
    # the wrong epoch's mask under the tested checkpoint's label.
    run = _make_tested_run(tmp_path / "run", on_disk=False)
    with pytest.raises(FileNotFoundError, match="on disk"):
        read_tested_ckpt(str(run))


def test_a_run_without_a_train_log_raises(tmp_path):
    (tmp_path / "run" / "checkpoints").mkdir(parents=True)
    (tmp_path / "run" / "checkpoints" / "last.ckpt").write_text("x")
    with pytest.raises(FileNotFoundError, match="train.log"):
        read_tested_ckpt(str(tmp_path / "run"))


def test_the_legend_covers_every_panel(tmp_path, monkeypatch):
    # A run that logs only the second panel's metric draws there and nowhere
    # else. Reading the first panel alone left every such curve unlabelled.
    import importlib.util

    import matplotlib

    matplotlib.use("agg")
    spec = importlib.util.spec_from_file_location("_vis_entry", Path("scripts/visualize.py"))
    vis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vis)

    base = tmp_path / "runs" / "cifar100" / "resnet50" / "augmentation"
    _make_run(base / "pruning_rigl-resnet50-cifar100-bs128-sr99" / "seed_1", [(1, 0.7), (2, 0.72)])
    _make_run(base / "bregman_linbreg-resnet50-cifar100-bs128-sr99" / "seed_1", [(1, 0.8), (2, 0.82)])
    groups = discover(str(base), ["*"])
    enc = Encoding(groups)
    # Only the second panel's metric exists, so panel one draws nothing at all.
    drawn = {}
    monkeypatch.setattr(vis.plt.Figure, "legend", lambda self, handles, labels, **kw: drawn.update(zip(labels, handles)))
    vis.plot_curves(groups, enc, ["pruning/sparsity", "valid/MulticlassAccuracy"], str(tmp_path / "out.pdf"), "train_log")
    assert set(drawn) == {enc.label(g) for g in groups}


def test_an_sv_run_reads_its_landed_sparsity_from_the_tested_epoch(tmp_path):
    # src/modules/sv.py writes no results.json, so the value comes from the epoch
    # the tested checkpoint names. The filename's own sr tag is whole-model.
    run = _make_tested_run(tmp_path / "run", ckpt_name="epoch041-metric_valid0.93-sr0.897.ckpt")
    version = run / "csv" / "version_0"
    version.mkdir(parents=True)
    (version / "metrics.csv").write_text(
        "step,epoch,bregman/sparsity,bregman/pruned_sparsity\n"
        "10,40,0.80,0.85\n20,41,0.897,0.9123\n30,42,0.91,0.93\n"
    )
    assert read_landed_sparsity(str(run)) == pytest.approx(0.9123)


def test_results_json_wins_where_the_task_writes_one(tmp_path):
    # The image task computes it from the tested checkpoint's own weights, so it
    # is the record; the csv only samples inside the epoch.
    run = _make_tested_run(tmp_path / "run", ckpt_name="epoch041-metric_valid0.93-sr0.897.ckpt")
    version = run / "csv" / "version_0"
    version.mkdir(parents=True)
    (version / "metrics.csv").write_text("step,epoch,pruning/sparsity\n20,41,0.9123\n")
    (run / "results.json").write_text(
        json.dumps({"best_checkpoint": {"overall_sparsity": WRONG_SPARSITY, "pruned_sparsity": 0.99}})
    )
    assert read_landed_sparsity(str(run)) == 0.99


def test_a_run_with_neither_source_reports_no_landed_sparsity(tmp_path):
    run = _make_tested_run(tmp_path / "run", logged=False)
    assert read_landed_sparsity(str(run)) is None
