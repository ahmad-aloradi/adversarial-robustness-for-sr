#!/usr/bin/env python3
"""Visualization for experiment results: training curves, bar charts for final
metrics, and cross-seed accuracy summaries (mean ± std). The script picks the
valid plot types per metric automatically (e.g. EER → bar, accuracy → curve +
summary, train_loss → curve only).

Two directory layouts are handled transparently:
    * SV:    <base>/<exp>
    * Image: <base>/<dataset>/<model>/<augmentation>/<exp>/seed_<N>

Image ``<exp>`` names are either ``<method>[-sr<NN>]-<scheduler>`` (dataset,
model, augmentation read from the parent dirs) or the fabfile shape
``<method>-<model>-<dataset>-bs<NN>[-sr<NN>][-<suffix>]`` (dataset/model
embedded in the name). Seeds repeat across
``seed_<N>`` subdirs; all seeds of one experiment are grouped, curves show the
mean with a ±1 std band, and the accuracy summary reports mean ± std across
seeds (single-seed runs render plainly). Glob patterns match ``<exp>`` in
either layout.

Usage examples:
    # SV training curves
    python scripts/visualize.py \\
        --base_dirs logs/train/runs \\
        --experiments "sv_bregman_adabreg-*-sr90" "sv_dense_adamw-*" \\
        --metrics train_loss valid_loss sparsity \\
        --output results/figures/

    # Image accuracy over seeds (curves + mean±std summary bar + CSV)
    python scripts/visualize.py \\
        --base_dirs logs/train/runs/cifar10/resnet18/augmentation \\
        --experiments "dense_sgd-*" "pruning_mag_unstruct-sr90-*" "bregman_adabreg-sr90-*" \\
        --metrics valid/MulticlassAccuracy sparsity \\
        --source train_log --output results/img/cifar10/resnet18
"""

import argparse
import glob
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SHOW_ALPHA = False
SHOW_f = False
AUTO_RESCALE_METRICS =  ("sparsity", "bregman/sparsity")

# ---------------------------------------------------------------------------
# 1. LaTeX-style rendering setup
# ---------------------------------------------------------------------------

matplotlib.use("pdf")


def _latex_available():
    """Check if a usable LaTeX installation exists (with required packages)."""
    import shutil
    import subprocess  # nosec B404
    import tempfile

    if not shutil.which("pdflatex"):
        return False
    try:
        test_tex = (
            r"\documentclass{article}"
            r"\usepackage{type1cm}\usepackage{type1ec}"
            r"\begin{document}x\end{document}"
        )
        with tempfile.NamedTemporaryFile(
            suffix=".tex", mode="w", delete=False
        ) as f:
            f.write(test_tex)
            tmp = f.name
        result = subprocess.run(  # nosec B603 B607
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tmp],
            capture_output=True,
            timeout=10,
            cwd=os.path.dirname(tmp),
        )
        os.unlink(tmp)
        for ext in (".aux", ".log", ".pdf"):
            p = tmp.replace(".tex", ext)
            if os.path.exists(p):
                os.unlink(p)
        return result.returncode == 0
    except Exception:
        return False


def setup_matplotlib(font_size=10):
    """Configure matplotlib for publication-quality PDF output.

    Uses LaTeX if available, otherwise falls back to matplotlib's built-in
    Computer Modern mathtext (still serif, still looks good in papers).
    """
    use_latex = _latex_available()
    if not use_latex:
        print(
            "Note: Full LaTeX not available; using mathtext fallback (still serif)."
        )

    plt.rcParams.update(
        {
            "text.usetex": use_latex,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.3,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.transparent": True,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def export_standalone_legend(
    handles, labels, output_path, ncol, font_size=10, frameon=True
):
    """Save a tightly-cropped, legend-only PDF using the same handles/labels
    as the inline legend so side-by-side LaTeX figures can share one legend."""
    setup_matplotlib(font_size)
    fig = plt.figure(figsize=(0.01, 0.01))
    leg = fig.legend(
        handles,
        labels,
        loc="center",
        ncol=ncol,
        frameon=frameon,
        columnspacing=0.8,
        handletextpad=0.3,
    )
    fig.canvas.draw()
    bbox = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches=bbox.expanded(1.05, 1.10))
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# 2. Visual encoding — consistent across all plots
# ---------------------------------------------------------------------------

# Method class → color.  Bregman = cool tones, Pruning = warm tones, Baselines = neutral.
METHOD_CLASS_COLORS = {
    "linbreg": "#1f77b4",  # deep blue
    "adabreg": "#2A662B",  # vibrant cyan
    "pruning_struct": "#ed8d61",  # strong red
    "pruning_unstruct": "#ff7f0e",  # bright orange
    "vanilla": "#61291e",  # distinct brown (baseline)
    "wespeaker": "#9C4F4F",  # dark charcoal
    "dense": "#7f7f7f",  # neutral grey — image dense (SGD) baseline
    # forget about those other methods for now, just make them black so they stand out as "other"
    "proxsgd": "#000000",  # light gray
    "adabregw": "#000000",  # deep navy blue
}

METHOD_DISPLAY_NAMES = {
    "linbreg": "LinBreg",
    "linbreg_fixed": "LinBreg (Fixed $\lambda$)",
    "adabreg": "AdaBreg",
    "adabreg_fixed": "AdaBreg (Fixed $\lambda$)",
    "adabregw": "AdaBregW",
    "pruning_struct": "Str. Prun.",
    "pruning_unstruct": "Unst. Prun.",
    "proxsgd": "ProxSGD",
    "vanilla": "AdamW",
    "wespeaker": "SGD",
    "dense": "Dense",
}

# Sparsity → marker shape  (consistent everywhere)
SPARSITY_MARKERS = {
    None: "s",  # square  — dense / baseline
    0: "s",
    50: "D",  # diamond
    75: "^",  # triangle up
    90: "v",  # triangle down
    95: "o",  # circle
    99: "x",  # x-mark
}

# Variant → marker shape override.  Fixed-lambda runs land at an uncontrolled
# sparsity, so their marker should not encode a sparsity level.
VARIANT_MARKERS = {
    "fixed": "*",  # star — visually distinct from every sparsity marker
}

# Sparsity → line dash pattern
SPARSITY_LINESTYLES = {
    None: "-",
    0: "-",
    50: (0, (5, 3)),
    75: (0, (3, 1, 1, 1)),
    90: "--",
    95: ":",
    99: (0, (1, 1)),
}

# Variant → line dash pattern (used in sweep mode to keep same-alpha curves
# from different variants visually distinct).
VARIANT_LINESTYLES = {
    None: "-",
    "regl1_conv": "-", #(0, (5, 2)),
    "poor_init": (0, (3, 1, 1, 1)),
    "rescale_prox": (0, (1, 1)),
    "rescale_prox_v2": (0, (3, 1, 1, 1, 1, 1)),
    "subgrad_corr_v2": (0, (5, 1, 1, 1)),
    "subgrad_corr_v3": (0, (5, 2, 1, 2)),
    "subgrad_corr_v4": (0, (1, 2, 5, 2)),
    "fixed": "-", #(0, (4, 2, 1, 2, 1, 2)),
}

# Axis labels for known metric names
METRIC_LABELS = {
    "train_loss": "Train Loss",
    "valid_loss": "Valid. Loss",
    "train/MulticlassAccuracy": "Train Accuracy",
    "valid/MulticlassAccuracy": "Valid. Accuracy",
    "sparsity": r"$\mathsf{s}(\theta)$",
    "bregman/sparsity": r"$\mathsf{s}(\theta)$",
    "bregman/global_lambda": r"$\lambda$",
    "EER": "EER",
    "minDCF": "minDCF",
    "train/margin": "AAM margin $m$",
    "lr": "Learning rate",
}

# stage → metrics in that stage
METRIC_STAGES = {
    "train": ["train_loss", "train/MulticlassAccuracy"],
    "valid": ["valid_loss", "valid/MulticlassAccuracy"],
    "test": ["EER", "minDCF", "test/MulticlassAccuracy"],
    "internal": ["sparsity", "bregman/global_lambda", "bregman/sparsity"],
    "schedule": ["lr", "train/margin"],
}

# metric → set of valid plot types
METRIC_PLOT_TYPES = {
    # Convergence metrics — time-series curves
    "train_loss": {"curves"},
    "valid_loss": {"curves"},
    # Accuracy is the image-task headline metric: curves over epochs plus a
    # cross-seed mean±std summary bar.
    "train/MulticlassAccuracy": {"curves", "summary"},
    "valid/MulticlassAccuracy": {"curves", "summary"},
    # Test accuracy is a single post-training scalar, not a curve — a "curve"
    # of one point can't distinguish variants (see cross-seed summary bar).
    "test/MulticlassAccuracy": {"summary"},
    # Internal / regularizer metrics — curves only
    "sparsity": {"curves"},
    "bregman/global_lambda": {"curves"},
    "bregman/sparsity": {"curves"},
    # Schedule metrics (lr, margin) — curves only
    "lr": {"curves"},
    "train/margin": {"curves"},
    # Binary test metrics — bar (comparison) and scatter (correlation)
    "EER": {"bar", "scatter"},
    "minDCF": {"bar", "scatter"},
}
# Unknown metrics default to {"curves"}

# Metrics that use log-scale y-axis by default
METRIC_LOG_SCALE = {
    "bregman/global_lambda",
    "lr",
}

# Method-class → (preferred lr column regex, verify-against column regex).
# For Bregman methods we assume conv_layers is a proxy for the shared lr,
# and verify it matches linear_layers per-experiment before using it.
LR_COLUMN_RULES = {
    "vanilla":   (r"^lr-(AdamW|Adam|SGD)$", None),
    "wespeaker": (r"^lr-SGD$", None),
    "linbreg":   (r"^lr-LinBreg/conv_layers$", r"^lr-LinBreg/linear_layers$"),
    "adabreg":   (r"^lr-AdaBreg/conv_layers$", r"^lr-AdaBreg/linear_layers$"),
    "proxsgd":   (r"^lr-ProxSGD/conv_layers$", r"^lr-ProxSGD/linear_layers$"),
    "adabregw":  (r"^lr-AdaBregW?/conv_layers$", r"^lr-AdaBregW?/linear_layers$"),
}


def resolve_lr_column(df, info):
    """Resolve the virtual 'lr' metric to an actual column in df.

    For Bregman methods we use conv_layers as a proxy and verify it equals
    linear_layers (they share the same scheduler group). Returns (col, None)
    on success or (None, reason) if no suitable column is found.
    """
    method = info.get("method_class", "vanilla")
    preferred, verify = LR_COLUMN_RULES.get(
        method, LR_COLUMN_RULES["vanilla"]
    )
    pref_cols = [c for c in df.columns if re.match(preferred, c)]
    if not pref_cols:
        # Fallback: any lr-* column
        any_lr = [c for c in df.columns if c.startswith("lr-")]
        if not any_lr:
            return None, "no lr-* columns"
        return any_lr[0], None

    col = pref_cols[0]
    if verify is not None:
        ver_cols = [c for c in df.columns if re.match(verify, c)]
        if ver_cols:
            a = df[[col, ver_cols[0]]].dropna()
            if len(a) and not np.allclose(
                a[col].values, a[ver_cols[0]].values, rtol=0, atol=1e-12
            ):
                print(
                    f"  [warn] {info['dirname']}: {col} differs from "
                    f"{ver_cols[0]} — conv-layer proxy may be misleading."
                )
    return col, None


def _valid_plot_types(metric):
    return METRIC_PLOT_TYPES.get(metric, {"curves"})


def _metric_short_name(metric):
    return metric.replace("/", "_").replace(" ", "_").lower()


def _stage_of(metric):
    for stage, ms in METRIC_STAGES.items():
        if metric in ms:
            return stage
    return "other"


# ---------------------------------------------------------------------------
# 3. Experiment name parsing
# ---------------------------------------------------------------------------


# (name substring, method_class, variant). Order matters — most-specific first
# (adabregw before adabreg; *_fixed/_progressive before the bare method).
METHOD_PATTERNS = [
    ("adabregw", "adabregw", None),
    ("adabreg_fixed", "adabreg", "fixed"),
    ("adabreg_progressive", "adabreg", "progressive"),
    ("adabreg", "adabreg", None),
    ("linbreg_fixed", "linbreg", "fixed"),
    ("linbreg_progressive", "linbreg", "progressive"),
    ("linbreg", "linbreg", None),
    ("proxsgd_fixed", "proxsgd", "fixed"),
    ("proxsgd", "proxsgd", None),
    ("pruning_mag_struct", "pruning_struct", None),
    ("pruning_mag_unstruct", "pruning_unstruct", None),
    ("dense_sgd", "dense", None),
]


def _classify_method(token, info):
    """Set method_class / method_variant / optimizer from a method token via
    substring match (most-specific pattern first)."""
    for pattern, cls, variant in METHOD_PATTERNS:
        if pattern in token:
            info["method_class"] = cls
            info["method_variant"] = variant
            if variant and not info.get("variant"):
                info["variant"] = variant
            break
    else:
        info["method_class"] = (
            "wespeaker" if token.replace("sv_", "") == "wespeaker" else "vanilla"
        )
    # Dense baseline encodes its optimizer, e.g. "dense_sgd" -> SGD.
    if info["method_class"] == "dense":
        m = re.match(r"dense_(\w+)", token)
        if m:
            info["optimizer"] = m.group(1)


def _parse_image_name(name, info):
    """Image run name — two layouts, told apart by the ``-bs<NN>`` batch tag.

    * run_subdir ``<method>[-sr<NN>]-<scheduler>``: dataset/model come from the
      parent dirs (:func:`_path_metadata`); ``-sr<NN>`` precedes the scheduler.
    * fabfile ``<method>-<model>-<dataset>-bs<NN>[-sr<NN>][-<suffix>]``:
      dataset/model are embedded here (the curated tree has no augmentation dir
      to read them from); ``-sr<NN>`` sits mid-name and there is no scheduler.

    Method-flavor variants (fixed/progressive) live in the method token either
    way, so nothing after ``-sr<NN>`` is a variant.
    """
    m_bs = re.search(r"-bs\d+", name)
    if m_bs:  # fabfile layout: batch tag splits <method>-<model>-<dataset> from the rest
        head = name[: m_bs.start()].split("-")
        if len(head) >= 3:
            info["dataset"] = head[-1]
            info["model"] = head[-2]
            head = head[:-2]
        m = re.search(r"-sr(\d+)", name[m_bs.end():])
        if m:
            info["sparsity"] = int(m.group(1))
        _classify_method("-".join(head), info)
        return

    method, sep, sched = name.rpartition("-")
    if sep:  # trailing scheduler tag: CosineAnnealing, no_scheduler, …
        info["scheduler"] = sched
        name = method
    m = re.search(r"-sr(\d+)$", name)
    if m:
        info["sparsity"] = int(m.group(1))
        name = name[: m.start()]
    _classify_method(name, info)


def _parse_sv_name(name, info):
    """SV run name ``sv_<method>-wespeaker_<backbone>-<dataset>-…-sr<NN>[-variant]``.

    alpha/f/regl suffixes are stripped first so they don't leak into ``variant``.
    """
    m_f = re.search(r"-f(\d+)$", name)
    if m_f:
        info["f"] = int(m_f.group(1))
        name = name[: m_f.start()]
    m_alpha = re.search(r"-alpha([\d.]+)$", name)
    if m_alpha:
        info["alpha"] = float(m_alpha.group(1))
        name = name[: m_alpha.start()]
    name = re.sub(r"-regl[12]\w*$", "", name)  # drop reg-style suffix

    m = re.search(r"-(sr|sparsity)(\d+)(?:-(.+))?$", name)
    if m:
        info["sparsity"] = int(m.group(2))
        if m.group(3):
            info["variant"] = m.group(3)  # e.g. "cls_scale2", "poor_init"

    m_model = re.search(r"-(wespeaker_\w+)-", name)
    info["model"] = m_model.group(1) if m_model else "unknown"
    if m_model:
        m_ds = re.match(r"([^-]+)-bs\d+", name[m_model.end():])
        if m_ds:
            info["dataset"] = m_ds.group(1)

    _classify_method(name.split("-wespeaker")[0], info)


def parse_experiment_name(dirname):
    """Parse a run-directory name into a styling/label metadata dict.

    Two name shapes:
      * SV:    ``sv_<method>-wespeaker_<backbone>-<dataset>-…-sr<NN>[-variant]``
      * Image: ``<method>[-sr<NN>]-<scheduler>`` or the fabfile shape
               ``<method>-<model>-<dataset>-bs<NN>[-sr<NN>][-<suffix>]``
               (see :func:`_parse_image_name`).

    ``alpha``/``f`` default to 1.0/50 when absent so SV sweep runs still sort.
    """
    info = {
        "dirname": dirname,
        "sparsity": None,
        "variant": None,
        "method_variant": None,
        "dataset": None,
        "model": None,
        "optimizer": None,
        "scheduler": None,
        "alpha": 1.0,
        "f": 50,
    }
    if "-wespeaker" in dirname:
        _parse_sv_name(dirname, info)
    else:
        _parse_image_name(dirname, info)
    return info


VARIANT_DISPLAY_NAMES = {
    "poor_init": "poor init",
    "fixed": "Fixed $\lambda$",
    "progressive": "Prog.",
    "regl1_conv": "",
    "constant_lr": "Const. lr",
    "RoP": "RoP",
    "rescale_prox": "Rescale Prox.",
    "rescale_prox_v2": "Rescale Prox. V2",
    "rescale_prox_V2": "SubGrad Corr.",
    "subgrad_corr_v2": "SubGrad Corr. V2",
    "subgrad_corr_v3": "SubGrad Corr. V3",
    "subgrad_corr_v4": "SubGrad Corr. V4",
}


def _variant_display_parts(info):
    """Distinct variant display strings for info, method flavor (e.g.
    progressive/fixed) first, then any ad hoc suffix tag (e.g.
    constant_lr) — so a run tagged with both keeps both in its label instead
    of the suffix tag silently overwriting the method flavor."""
    parts = []
    for token in (info.get("method_variant"), info.get("variant")):
        if not token or token in parts:
            continue
        parts.append(token)
    return [VARIANT_DISPLAY_NAMES.get(p, p) for p in parts if VARIANT_DISPLAY_NAMES.get(p, p)]


# Variant color adjustments: (hue_shift, saturation_shift, lightness_shift)
# Hue rotates the color wheel (0-1 wraps), saturation/lightness are additive.
# This keeps variants visually related to their base method but clearly distinct.
VARIANT_COLOR_ADJUSTMENTS = {
    "poor_init": (-0.08, -0.15, -0.12),       # shift hue toward red, desaturate, darken
    "rescale_prox": (0.10, 0.05, 0.15),        # shift hue toward cyan, slightly brighter
    "rescale_prox_v2": (0.10, 0.05, 0.15),     # shift hue toward cyan, slightly brighter
    "rescale_prox_V2": (0.18, -0.10, -0.05),   # shift hue further, slightly muted
    "subgrad_corr_v2": (0.18, -0.10, -0.05),   # shift hue further, slightly muted,
    "subgrad_corr_v3": (0.36, -0.20, -0.1),   # shift hue further, slightly muted,
    "subgrad_corr_v4": (-0.18, 0.10, 0.05),
    "fixed": (0.12, -0.0, 0.18),              # shift hue toward purple, slightly darker
    "progressive": (0.05, 0.0, -0.12),        # slightly darker sibling of the base method
}


def _adjust_color(hex_color, hue_shift, sat_shift, light_shift):
    """Adjust a hex color in HLS space: shift hue, saturation, and lightness."""
    import colorsys

    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    h = (h + hue_shift) % 1.0
    s = max(0.0, min(1.0, s + sat_shift))
    l = max(0.05, min(0.95, l + light_shift))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def make_label(info, *, fixed_as_token: bool = True):
    """Create a concise, consistent legend label.

    Fixed-λ runs render as ``Method (fixed)`` by default to keep cross-model
    legends compact for paper figures. Pass ``fixed_as_token=False`` to render
    ``Method (λ=value)`` with the numeric λ instead.
    """
    name = METHOD_DISPLAY_NAMES.get(info["method_class"], info["method_class"])
    if info.get("variant") == "fixed" and info.get("fixed_lambda") is not None:
        if fixed_as_token:
            return f"{name} (fixed)"
        lam = info["fixed_lambda"]
        lam_sym = r"$\lambda$" if plt.rcParams.get("text.usetex") else "λ"
        return f"{name} ({lam_sym}={lam:g})"
    if info["sparsity"] is not None:
        pct = r"\%" if plt.rcParams.get("text.usetex") else "%"
        label = f"{name} {info['sparsity']}{pct}"
    else:
        label = name
    parts = _variant_display_parts(info)
    if parts:
        label += " (" + ", ".join(parts) + ")"
    extras = []
    if info.get("_show_alpha"):
        sym = r"$\alpha$" if plt.rcParams.get("text.usetex") else "α"
        extras.append(f"{sym}={info['alpha']:g}")
    if info.get("_show_f"):
        sym = r"$f$" if plt.rcParams.get("text.usetex") else "f"
        extras.append(f"{sym}={info['f']}")
    if extras:
        label += " " + ", ".join(extras)
    return label


def get_style(info):
    """Return (color, marker, linestyle) tuple — deterministic from
    metadata.

    In sweep mode (gradient color set), variant drives linestyle so that
    same-alpha curves from different variants remain visually distinct
    even though they share a color.
    """
    color = info.get("_gradient_color")
    if color is not None:
        variant = info.get("variant")
        marker = VARIANT_MARKERS.get(variant, SPARSITY_MARKERS.get(info["sparsity"], "x"))
        ls = VARIANT_LINESTYLES.get(variant, "-")
    else:
        color = METHOD_CLASS_COLORS.get(info["method_class"], "#333333")
        variant = info.get("variant")
        if variant:
            adj = VARIANT_COLOR_ADJUSTMENTS.get(variant)
            if adj:
                color = _adjust_color(color, *adj)
        # Fixed-lambda runs land at an uncontrolled sparsity, so encoding
        # marker/linestyle by sparsity target is misleading. Route through
        # VARIANT_MARKERS / VARIANT_LINESTYLES["fixed"] instead.
        if variant == "fixed":
            marker = VARIANT_MARKERS.get("fixed", SPARSITY_MARKERS.get(info["sparsity"], "x"))
            ls = VARIANT_LINESTYLES.get("fixed", SPARSITY_LINESTYLES.get(info["sparsity"], "-"))
        else:
            marker = SPARSITY_MARKERS.get(info["sparsity"], "x")
            ls = SPARSITY_LINESTYLES.get(info["sparsity"], "-")
    return color, marker, ls


def assign_gradient_colors(experiments):
    """Within each (method_class, sparsity), gradient-color by whichever of
    {alpha, f} varies. Color is keyed off the swept value (not the rank
    among experiments) so two runs with the same alpha share a color
    regardless of variant — variants are then distinguished by linestyle
    in :func:`get_style`. This avoids both color collisions across
    variants and silent overwrites.
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for _, info in experiments:
        # Fixed-lambda runs use default alpha/f (1.0/50) and aren't part of any
        # sweep; pooling them with non-fixed runs at the same sparsity creates a
        # spurious 2-point "sweep" that maps the fixed run to the darkest end of
        # the gradient (visually black) and overrides VARIANT_COLOR_ADJUSTMENTS
        # ["fixed"] in get_style. Skip them so they keep their fixed-variant color.
        if info.get("variant") == "fixed":
            continue
        groups[(info["method_class"], info.get("sparsity"))].append(info)

    for (method, _), members in groups.items():
        if len(members) < 2:
            continue
        param = None
        for cand in ("alpha", "f"):
            vals = {m.get(cand) for m in members if m.get(cand) is not None}
            if len(vals) >= 2:
                param = cand
                break
        if param is None:
            continue
        unique_vals = sorted(
            {m[param] for m in members if m.get(param) is not None}
        )
        n = len(unique_vals)
        if n < 2:
            continue
        base = METHOD_CLASS_COLORS.get(method, "#333333")
        for info in members:
            v = info.get(param)
            if v is None:
                continue
            t = unique_vals.index(v) / (n - 1)
            info["_gradient_color"] = _adjust_color(base, 0, 0, 0.30 - 0.55 * t)


def assign_label_visibility(experiments):
    """Set `_show_alpha`/`_show_f` per info based on whether the field
    takes >=2 distinct values across the full matched set. None counts
    as a distinct value (mixed presence is still variation). Only
    Bregman methods carry alpha/f — they are Bregman-only hyperparameters,
    so dense baselines and pruning runs never get them stamped on the
    label even when alpha varies elsewhere in the matched set.
    """
    if not experiments:
        return
    infos = [info for _, info in experiments]
    show_alpha = len({i.get("alpha") for i in infos}) >= 2
    show_f = len({i.get("f") for i in infos}) >= 2
    for info in infos:
        is_bregman = "breg" in info.get("method_class")
        info["_show_alpha"] = (
            show_alpha and info.get("alpha") is not None and is_bregman and SHOW_ALPHA
        )
        info["_show_f"] = (
            show_f and info.get("f") is not None and is_bregman and SHOW_f
        )


# ---------------------------------------------------------------------------
# 4. Data loading
# ---------------------------------------------------------------------------


def load_fixed_lambda(exp_dir):
    """Extract _bregman_lambda from config_tree.log, or None if not found."""
    path = os.path.join(exp_dir, "config_tree.log")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected config_tree.log not found in {exp_dir}")

    with open(path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "_bregman_lambda" in line and i + 1 < len(lines):
            m = re.search(r"[\d.]+", lines[i + 1])
            if m:
                return float(m.group())

    raise ValueError(f"_bregman_lambda value not found in config_tree.log of {exp_dir}")


def info_from_csv_row(exp_name, base_dirs=None):
    """Build an info dict for a CSV-leaderboard row.

    Mirrors what :func:`discover_experiments` does for a single dirname:
    parses the name and, when ``variant=='fixed'`` and ``base_dirs`` are
    provided, locates the experiment directory and loads the fixed lambda
    from ``config_tree.log``. Used by CSV-based downstream scripts so they
    produce labels identical to the directory-based pipeline.
    """
    info = parse_experiment_name(exp_name)
    if info.get("variant") == "fixed" and base_dirs:
        for bd in base_dirs:
            full = os.path.join(bd, exp_name)
            if os.path.isdir(full) and os.path.exists(
                os.path.join(full, "config_tree.log")
            ):
                info["fixed_lambda"] = load_fixed_lambda(full)
                break
    return info


def load_train_log(exp_dir):
    """Load epoch-level metrics from train_log.txt → DataFrame."""
    path = os.path.join(exp_dir, "train_log.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected train_log.txt not found in {exp_dir}")

    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = {}
            for pair in line.split(", "):
                k, v = pair.split(": ", 1)
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            rows.append(row)

    return pd.DataFrame(rows) if rows else None


CSV_COLUMN_ALIASES = {
    "train/LogSoftmaxWrapper": "train_loss",
    "valid/LogSoftmaxWrapper": "valid_loss",
}


def load_csv_metrics(exp_dir):
    """Load step-level metrics from csv/version_*/metrics.csv → DataFrame.

    Lightning's CSVLogger creates a fresh ``version_N`` directory on every run
    or resume. All non-empty versions are merged into a single continuous
    curve; where step ranges overlap, the highest version wins (it represents
    the most recent training run, which superseded the earlier one). Prints a
    clear warning if numeric metrics jump abruptly across a version boundary.
    """
    csv_root = os.path.join(exp_dir, "csv")
    if not os.path.isdir(csv_root):
        return None

    version_files = []
    for entry in sorted(os.listdir(csv_root)):
        m = re.match(r"version_(\d+)$", entry)
        if not m:
            continue
        path = os.path.join(csv_root, entry, "metrics.csv")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            version_files.append((int(m.group(1)), path))
    if not version_files:
        return None
    version_files.sort(key=lambda t: t[0])

    dfs = []
    for vidx, vpath in version_files:
        try:
            df_v = pd.read_csv(vpath)
        except pd.errors.EmptyDataError:
            continue
        if df_v.empty:
            continue
        df_v["__version__"] = vidx
        dfs.append(df_v)
    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True, sort=False)
    if "step" in df.columns:
        df = (
            df.sort_values(["step", "__version__"])
              .groupby("step", as_index=False)
              .last()
        )
        if len(dfs) > 1:
            _warn_on_version_discontinuity(df, exp_dir)

    df.drop(columns=["__version__"], errors="ignore", inplace=True)
    df.rename(columns=CSV_COLUMN_ALIASES, inplace=True)
    return df


def _warn_on_version_discontinuity(df, exp_dir, window=5, rel_tol=0.5):
    """Print a warning if metrics jump abruptly at a version boundary.

    For each step where ``__version__`` increases, compare the mean of the
    last ``window`` non-null samples before the boundary to the first
    ``window`` after; flag any numeric column whose relative jump exceeds
    ``rel_tol``.
    """
    if "step" not in df.columns or "__version__" not in df.columns:
        return
    sdf = df.sort_values("step").reset_index(drop=True)
    boundary_idxs = [
        i for i in range(1, len(sdf))
        if sdf.loc[i, "__version__"] > sdf.loc[i - 1, "__version__"]
    ]
    if not boundary_idxs:
        return

    skip = {"step", "epoch", "__version__"}
    metric_cols = [
        c for c in sdf.columns
        if c not in skip and pd.api.types.is_numeric_dtype(sdf[c])
    ]

    for b in boundary_idxs:
        v_prev = int(sdf.loc[b - 1, "__version__"])
        v_next = int(sdf.loc[b, "__version__"])
        boundary_step = int(sdf.loc[b, "step"])
        flagged = []
        for c in metric_cols:
            before = sdf.iloc[max(0, b - window):b][c].dropna()
            after = sdf.iloc[b:b + window][c].dropna()
            if len(before) < 2 or len(after) < 2:
                continue
            a, z = float(before.mean()), float(after.mean())
            if not (np.isfinite(a) and np.isfinite(z)):
                continue
            denom = max(abs(a), abs(z), 1e-9)
            if abs(z - a) / denom > rel_tol:
                flagged.append((c, a, z))
        if not flagged:
            continue
        print(
            f"Warning: discontinuity across version_{v_prev}→version_{v_next} "
            f"boundary (step ~{boundary_step}) in {exp_dir}"
        )
        for c, a, z in flagged[:8]:
            print(f"    {c}: {a:.4g} → {z:.4g}")
        if len(flagged) > 8:
            print(f"    ... and {len(flagged) - 8} more")


# Logical legend grouping (not alphabetical). Used wherever experiments need
# to be sorted in a way that matches how readers expect the legend to read:
#   Group 0: dense baselines (AdamW / SGD)
#   Group 1: pruning (structured, then unstructured)
#   Group 2: fixed-lambda Bregman runs
#   Group 3: adaptive Bregman runs, by sparsity sweep
EXPERIMENT_ORDER = [
    "dense",
    "vanilla",
    "wespeaker",
    "pruning_struct",
    "pruning_unstruct",
    "linbreg",
    "adabreg",
    "adabregw",
]


def experiment_sort_key(info):
    """Sort key turning an experiment ``info`` dict into a tuple ordered by
    the logical legend groups above. Reused by downstream scripts so they
    can re-sort their own legend handles to match this script's ordering.
    """
    mc = info.get("method_class", "")
    variant = info.get("variant")
    if mc in ("dense", "vanilla", "wespeaker"):
        group = 0
    elif mc in ("pruning_struct", "pruning_unstruct"):
        group = 1
    elif variant == "fixed":
        group = 2
    else:
        group = 3
    return (
        group,
        EXPERIMENT_ORDER.index(mc) if mc in EXPERIMENT_ORDER else 99,
        info.get("sparsity") or -1,
        info.get("variant") or "",
        info["alpha"] if info.get("alpha") is not None else -1.0,
        info["f"] if info.get("f") is not None else -1,
    )


# A run directory holds the training artifacts directly. Presence of either
# marker is enough to treat a directory as a leaf run (and stop descending).
RUN_ARTIFACT_MARKERS = ("config_tree.log", "train_log.txt")

# Artifact subdirs that never contain a nested run — pruned while walking.
_WALK_SKIP_DIRS = {
    "checkpoints", "csv", "metadata", "tensorboard", "test_artifacts", ".hydra",
}

# Per-seed run dirs are named "seed_<N>" (see scripts/fabfile.py:run_img).
SEED_RE = re.compile(r"^seed_?(\d+)$")


def _is_run_dir(path):
    return any(
        os.path.exists(os.path.join(path, m)) for m in RUN_ARTIFACT_MARKERS
    )


def _find_run_dirs(base_dir):
    """Yield leaf run directories under ``base_dir`` at any nesting depth.

    SV runs sit one level down (``<base>/<exp>``); image runs sit deeper
    (``<base>/<dataset>/<model>/<augmentation>/<exp>/seed_<N>``). Both bottom
    out in a dir holding the training artifacts, so we walk until we hit one.
    """
    for root, dirs, _ in os.walk(base_dir):
        if _is_run_dir(root):
            dirs[:] = []  # a run dir has no nested runs
            yield root
            continue
        dirs[:] = [
            d for d in dirs
            if d not in _WALK_SKIP_DIRS and not d.endswith("_artifacts")
        ]


def _path_metadata(exp_dir):
    """dataset/model/augmentation from an image run's parent dirs.

    Image layout: ``<dataset>/<model>/<augmentation>/<exp>[/seed_<N>]``. The
    augmentation dir has a fixed name, so anchor on it — this works no matter
    how deep ``base_dir`` points. Flat SV runs have no such dir → returns {}.
    """
    parts = os.path.normpath(exp_dir).split(os.sep)
    for i, name in enumerate(parts):
        if name in ("augmentation", "no_augmentation") and i >= 2:
            return {
                "dataset": parts[i - 2],
                "model": parts[i - 1],
                "augmentation": name == "augmentation",
            }
    return {}


def discover_experiments(base_dirs, patterns):
    """Find experiments matching glob patterns. Returns a sorted list of
    ``(representative_dir, info)`` tuples.

    An *experiment* is one run dir (SV) or one ``<exp>`` dir whose ``seed_<N>``
    subdirs are grouped (image). Grouping is keyed on the ``<exp>`` directory
    *path*, so same-named experiments under different dataset/model/augmentation
    parents stay distinct. ``info`` carries ``seed_dirs``/``seeds`` for
    cross-seed aggregation and, for image runs, dataset/model/augmentation read
    from the path. Patterns match the ``<exp>`` directory name in both layouts.

    Args:
        base_dirs: A single directory path (str) or a list of directory paths.
        patterns: Glob patterns for experiment directory names.
    """
    if isinstance(base_dirs, str):
        base_dirs = [base_dirs]

    groups = {}  # exp_dir -> {"name": str, "dirs": [...], "seeds": [...]}
    for base_dir in base_dirs:
        if not os.path.isdir(base_dir):
            raise ValueError(f"Warning: base dir does not exist: {base_dir}")

        for run_dir in _find_run_dirs(base_dir):
            m_seed = SEED_RE.match(os.path.basename(run_dir))
            seed = int(m_seed.group(1)) if m_seed else None
            exp_dir = os.path.dirname(run_dir) if m_seed else run_dir
            # Older runs kept the launcher's ".yaml" in the method token.
            exp_name = re.sub(r"\.yaml(?=-|$)", "", os.path.basename(exp_dir))
            if not any(glob.fnmatch.fnmatch(exp_name, p) for p in patterns):
                continue
            g = groups.setdefault(
                exp_dir, {"name": exp_name, "dirs": [], "seeds": []}
            )
            if run_dir in g["dirs"]:
                continue
            g["dirs"].append(run_dir)
            g["seeds"].append(seed)

    matched = []
    for exp_dir, g in groups.items():
        info = parse_experiment_name(g["name"])
        info.update(_path_metadata(exp_dir))
        # Order seed dirs by seed number (unseeded SV runs sort as -1).
        pairs = sorted(
            zip(g["seeds"], g["dirs"]),
            key=lambda t: (-1 if t[0] is None else t[0]),
        )
        info["seeds"] = [s for s, _ in pairs]
        info["seed_dirs"] = [d for _, d in pairs]
        rep = info["seed_dirs"][0]
        if info.get("variant") == "fixed":
            info["fixed_lambda"] = load_fixed_lambda(rep)
        matched.append((rep, info))

    matched.sort(key=lambda item: experiment_sort_key(item[1]))
    assign_gradient_colors(matched)
    assign_label_visibility(matched)
    return matched


# ---------------------------------------------------------------------------
# 5. Plotting
# ---------------------------------------------------------------------------


def _auto_ylim(ax, metric, margin=0.05):
    """Tighten y-axis to data range with smart zooming.

    For bounded [0,1] metrics like sparsity and accuracy, zooms to the region
    where the interesting data lives (ignores constant-zero baselines if other
    curves are far away).
    """
    lines = ax.get_lines()
    if not lines:
        return

    all_y = np.concatenate([ln.get_ydata() for ln in lines])
    all_y = all_y[np.isfinite(all_y)]
    if len(all_y) == 0:
        return

    ymin, ymax = all_y.min(), all_y.max()
    span = ymax - ymin if ymax > ymin else 1.0
    pad = span * margin

    if metric in AUTO_RESCALE_METRICS:
        # If the spread is large (e.g. 0 to 0.9), check if there's a cluster
        # of values far from the outliers — use IQR-based zoom
        if span > 0.3:
            q1, q3 = np.percentile(all_y, [10, 90])
            iqr = q3 - q1
            lo = max(0, q1 - 1.5 * max(iqr, 0.03))
            hi = min(1.0, q3 + 1.5 * max(iqr, 0.03))
        elif span < 0.15:
            lo = max(0, ymin - 0.02)
            hi = min(1.0, ymax + 0.02)
        else:
            lo = max(0, ymin - pad)
            hi = min(1.0, ymax + pad)
        ax.set_ylim(lo, hi)
    else:
        ax.set_ylim(ymin - pad, ymax + pad)


def load_run_df(run_dir, source):
    """Load one run's metrics frame from the chosen source."""
    return (
        load_train_log(run_dir)
        if source == "train_log"
        else load_csv_metrics(run_dir)
    )


def resolve_series(df, metric, info, source):
    """Return a ``pd.Series`` (index = x-axis, values = metric) for one run's
    frame, or None if the metric isn't available.

    Handles the virtual ``lr`` metric and the fixed-lambda constant synthesis
    exactly as the single-curve path did, so per-seed series stay consistent.
    """
    if df is None:
        return None
    x_col = "epoch" if source == "train_log" else "step"
    if x_col not in df.columns:
        return None

    if metric == "lr":
        col, _ = resolve_lr_column(df, info)
        if col is None:
            return None
    else:
        col = metric
    # Fixed-lambda Bregman runs never log bregman/global_lambda (the scheduler
    # doesn't run). Synthesize a constant column so the horizontal line appears.
    if (
        col == "bregman/global_lambda"
        and col not in df.columns
        and info.get("fixed_lambda") is not None
    ):
        df = df.copy()
        df[col] = info["fixed_lambda"]
    if col not in df.columns:
        return None

    x = df[x_col].astype(float)
    if source == "csv":
        x = x / 1000.0
    y = df[col]
    mask = y.notna()
    if mask.sum() == 0:
        return None
    s = pd.Series(y[mask].to_numpy(), index=x[mask].to_numpy())
    s = s[~s.index.duplicated(keep="last")]
    return s.sort_index()


def aggregate_seed_series(seed_dfs, metric, info, source):
    """Combine per-seed series for one metric into (x, mean, std).

    Series are outer-joined on their x-index; ``mean``/``std`` ignore missing
    seeds at a given x. ``std`` is None when only one seed contributes (so
    callers skip the band). Returns None if no seed has the metric.
    """
    series = [resolve_series(df, metric, info, source) for df in seed_dfs]
    series = [s for s in series if s is not None]
    if not series:
        return None
    frame = pd.concat(series, axis=1)
    mean = frame.mean(axis=1)
    x = mean.index.to_numpy()
    std = frame.std(axis=1).to_numpy() if frame.shape[1] > 1 else None
    return x, mean.to_numpy(), std


def plot_training_curves(
    experiments,
    metrics,
    output_path,
    font_size=10,
    fig_width=5.5,
    fig_height=None,
    source="train_log",
    log_scale=None,
    legend_mode="inline",
):
    """Plot training curves (one subplot per metric, shared x-axis).

    legend_mode:
        "inline" — embed the legend at the top of the figure (default).
        "split"  — omit the inline legend and write a separate
                   ``<output_path stem>_legend.pdf`` containing only the
                   legend, so two figures can share one legend in LaTeX.
    """
    if log_scale is None:
        log_scale = set()
    setup_matplotlib(font_size)

    n = len(metrics)
    if fig_height is None:
        fig_height = 2.3 * n + 0.4

    fig, axes = plt.subplots(
        n, 1, figsize=(fig_width, fig_height), sharex=True, squeeze=False
    )
    axes = axes.flatten()

    for exp_dir, info in experiments:
        seed_dfs = [
            load_run_df(d, source) for d in info.get("seed_dirs", [exp_dir])
        ]
        seed_dfs = [d for d in seed_dfs if d is not None]
        if not seed_dfs:
            print(f"  [skip] no {source} data: {info['dirname']}")
            continue

        color, marker, ls = get_style(info)
        label = make_label(info)

        for ax, metric in zip(axes, metrics):
            agg = aggregate_seed_series(seed_dfs, metric, info, source)
            if agg is None:
                continue
            x, mean, std = agg
            n_pts = len(x)
            # Skip constant-zero series on sparsity panel (e.g. baselines)
            if metric == "sparsity" and np.nanmax(mean) < 1e-6:
                continue
            ax.plot(
                x,
                mean,
                color=color,
                marker=marker,
                linestyle=ls,
                markersize=3.5,
                markevery=max(1, n_pts // 12),
                label=label,
            )
            # Shade ±1 std across seeds (only when >1 seed contributed).
            if std is not None:
                ax.fill_between(
                    x, mean - std, mean + std,
                    color=color, alpha=0.18, linewidth=0,
                )

    # Axis formatting
    for ax, metric in zip(axes, metrics):
        ax.set_ylabel(
            METRIC_LABELS.get(metric, metric.replace("_", " ").title())
        )
        if metric in log_scale:
            ax.set_yscale("log")
        elif metric in ("sparsity", "bregman/sparsity"):
            from matplotlib.ticker import FixedLocator
            ax.set_ylim(0.7, 1.005)
            ax.yaxis.set_major_locator(FixedLocator([0.75, 0.80, 0.85, 0.90, 0.95, 0.99]))
        else:
            _auto_ylim(ax, metric)
    axes[-1].set_xlabel("Epoch" if source == "train_log" else "iteration [K]")
    if source == "train_log":
        from matplotlib.ticker import MaxNLocator
        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Deduplicated legend at top
    handles, labels = [], []
    seen_labels = set()
    for h, l in zip(*axes[0].get_legend_handles_labels()):
        if l not in seen_labels:
            seen_labels.add(l)
            handles.append(h)
            labels.append(l)

    if handles:
        # ncol = min(5, len(labels))
        ncol = min(3 , len(labels))
        if legend_mode == "inline":
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=ncol,
                bbox_to_anchor=(0.5, 0.9),
                frameon=True,
                columnspacing=0.8,
                handletextpad=0.3,
            )
        elif legend_mode == "split":
            legend_path = os.path.splitext(output_path)[0] + "_legend.pdf"
            export_standalone_legend(
                handles, labels, legend_path, ncol, font_size=font_size
            )
        else:
            raise ValueError(
                f"legend_mode must be 'inline' or 'split', got {legend_mode!r}"
            )

    fig.align_ylabels(axes)
    fig.subplots_adjust(hspace=0.08)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _autosize_bar_axes(fig, ax, n_bars, base_rotation=25, max_width=20.0):
    """Grow a bar chart to fit its rotated x-tick labels without overlap.

    Starts from a per-bar width guess (more bars need more inches), then
    measures the actual rendered label extents and keeps widening — and, if
    widening alone hits ``max_width``, steepens the rotation — until adjacent
    labels no longer overlap. Measures the real render rather than guessing
    from character counts, since label length varies a lot across variants.
    """
    width, height = fig.get_size_inches()
    fig.set_size_inches(max(width, n_bars * 0.5), height)
    rotation = base_rotation
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(rotation)
    for _ in range(6):
        fig.canvas.draw()
        boxes = sorted(
            (lbl.get_window_extent() for lbl in ax.get_xticklabels()),
            key=lambda b: b.x0,
        )
        if all(a.x1 <= b.x0 for a, b in zip(boxes, boxes[1:])):
            break
        width, height = fig.get_size_inches()
        if width < max_width:
            fig.set_size_inches(min(width * 1.25, max_width), height)
        elif rotation < 60:
            rotation = min(rotation + 10, 60)
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(rotation)
        else:
            break


def plot_bar_comparison(
    experiments,
    metric,
    output_path,
    font_size=10,
    fig_width=5.5,
    fig_height=3.0,
    epoch=-1,
):
    """Grouped bar chart comparing a metric (last epoch), grouped by
    sparsity."""
    from collections import OrderedDict

    setup_matplotlib(font_size)

    entries = []
    for exp_dir, info in experiments:
        df = load_train_log(exp_dir)
        if df is None or metric not in df.columns:
            continue
        val = df[metric].iloc[epoch]
        entries.append((info, val))

    if not entries:
        print(f"No data for metric '{metric}'")
        return

    # Group by sparsity
    groups = OrderedDict()
    for info, val in entries:
        sp = info["sparsity"]
        groups.setdefault(sp, []).append((info, val))

    # Sort: None/0 (baselines) first, then ascending sparsity
    def _sp_key(sp):
        if sp is None or sp == 0:
            return -1
        return sp

    sorted_keys = sorted(groups.keys(), key=_sp_key)

    # Build bar positions with intra-group and inter-group gaps
    bar_width = 0.6
    intra_gap = 0.15
    group_gap = 1.5  # in bar-width units

    x_positions = []
    bar_infos = []  # (info, value) per bar
    group_centers = []  # (center_x, sparsity_label) per group
    pos = 0.0

    for sp in sorted_keys:
        group = groups[sp]
        group_xs = []
        for info, val in group:
            x_positions.append(pos)
            bar_infos.append((info, val))
            group_xs.append(pos)
            pos += bar_width + intra_gap
        # Remove trailing intra_gap, add group_gap
        pos -= intra_gap
        center = (group_xs[0] + group_xs[-1]) / 2
        if sp is None or sp == 0:
            sp_label = "Dense"
        else:
            pct = r"\%" if plt.rcParams.get("text.usetex") else "%"
            sp_label = f"{sp}{pct}"
        group_centers.append((center, sp_label))
        pos += group_gap * bar_width

    x_positions = np.array(x_positions)
    values = np.array([v for _, v in bar_infos])
    colors = [get_style(info)[0] for info, _ in bar_infos]
    tick_labels = [method_display_name(info) for info, _ in bar_infos]

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    bars = ax.bar(
        x_positions,
        values,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        width=bar_width,
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_labels, rotation=25, ha="right")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    _autosize_bar_axes(fig, ax, n_bars=len(tick_labels))

    # Value labels on top of each bar
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=font_size - 2,
        )

    # Sparsity group labels below x-axis
    for cx, sp_label in group_centers:
        ax.text(
            cx,
            -0.12,
            sp_label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontweight="bold",
            fontsize=font_size,
        )

    # Zoom y-axis if values are clustered (e.g. all near 0.95)
    vmin, vmax = values.min(), values.max()
    span = vmax - vmin
    if vmin > 0 and span / vmax < 0.3:  # values within 30% of each other
        lo = vmin - max(span * 1.5, vmax * 0.02)
        ax.set_ylim(max(0, lo), vmax + max(span * 0.8, vmax * 0.02))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    print(f"Saved: {output_path}")


def per_seed_metric(seed_dirs, metric, source, reduce="max"):
    """Reduce ``metric`` to one scalar per seed run.

    ``reduce="max"`` (best epoch, matches the checkpoint the run selects) or
    ``"last"`` (final epoch). Returns the list of per-seed scalars; seeds
    missing the metric are dropped.
    """
    assert reduce in ("max", "last"), f"unknown reduce: {reduce}"
    vals = []
    for d in seed_dirs:
        df = load_run_df(d, source)
        if df is None or metric not in df.columns:
            continue
        col = df[metric].dropna()
        if col.empty:
            continue
        vals.append(float(col.max() if reduce == "max" else col.iloc[-1]))
    return vals


def method_display_name(info):
    """Short method name for a bar/axis tick, including all variant tags
    (e.g. "AdaBreg (Prog., Const. lr)") so runs differing only by variant
    stay distinct. The dense baseline shows its optimizer (e.g. SGD) since
    the group header already reads 'Dense'."""
    if info["method_class"] == "dense" and info.get("optimizer"):
        name = info["optimizer"].upper()
    else:
        name = METHOD_DISPLAY_NAMES.get(info["method_class"], info["method_class"])
    parts = _variant_display_parts(info)
    if parts:
        name += " (" + ", ".join(parts) + ")"
    return name


def plot_accuracy_summary(
    experiments,
    metric,
    output_path,
    source="csv",
    reduce="max",
    font_size=10,
    fig_width=5.5,
    fig_height=3.0,
):
    """Grouped bar chart of a scalar metric per experiment, grouped by sparsity,
    with cross-seed mean and ±1 std error bars.

    Writes a companion ``<output>.csv`` with per-experiment n_seeds / mean / std
    and the raw per-seed values. Single-seed experiments render a plain bar (no
    error bar, std reported as 0).
    """
    from collections import OrderedDict

    setup_matplotlib(font_size)

    # (info, mean, std, seed_vals) per experiment that has the metric.
    entries = []
    for exp_dir, info in experiments:
        vals = per_seed_metric(
            info.get("seed_dirs", [exp_dir]), metric, source, reduce
        )
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        entries.append((info, float(arr.mean()), std, vals))

    if not entries:
        print(f"No data for metric '{metric}'")
        return

    # Group by sparsity; baselines (None/0) first, then ascending.
    groups = OrderedDict()
    for e in entries:
        groups.setdefault(e[0]["sparsity"], []).append(e)
    sorted_keys = sorted(
        groups.keys(), key=lambda sp: -1 if (sp is None or sp == 0) else sp
    )

    bar_width, intra_gap, group_gap = 0.6, 0.15, 1.5
    x_positions, bars_meta, group_centers = [], [], []
    pos = 0.0
    for sp in sorted_keys:
        group_xs = []
        for e in groups[sp]:
            x_positions.append(pos)
            bars_meta.append(e)
            group_xs.append(pos)
            pos += bar_width + intra_gap
        pos -= intra_gap
        center = (group_xs[0] + group_xs[-1]) / 2
        if sp is None or sp == 0:
            sp_label = "Dense"
        else:
            pct = r"\%" if plt.rcParams.get("text.usetex") else "%"
            sp_label = f"{sp}{pct}"
        group_centers.append((center, sp_label))
        pos += group_gap * bar_width

    x_positions = np.array(x_positions)
    means = np.array([m for _, m, _, _ in bars_meta])
    stds = np.array([s for _, _, s, _ in bars_meta])
    colors = [get_style(info)[0] for info, _, _, _ in bars_meta]
    tick_labels = [method_display_name(info) for info, _, _, _ in bars_meta]

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.bar(
        x_positions, means, yerr=stds, capsize=2.5,
        color=colors, edgecolor="white", linewidth=0.5, width=bar_width,
        error_kw={"elinewidth": 0.8},
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_labels, rotation=25, ha="right")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    _autosize_bar_axes(fig, ax, n_bars=len(tick_labels))

    for xp, m, s, (_, _, _, vals) in zip(x_positions, means, stds, bars_meta):
        txt = f"{m:.3f}" if len(vals) < 2 else f"{m:.3f}\n$\\pm${s:.3f}"
        ax.text(
            xp, m + s, txt, ha="center", va="bottom", fontsize=font_size - 2
        )

    # Place group labels below the rotated method labels; measure, don't guess.
    fig.canvas.draw()
    to_axes = ax.transAxes.inverted()
    labels_bottom = 0.0
    for lbl in ax.get_xticklabels():
        bb = lbl.get_window_extent().transformed(to_axes)
        labels_bottom = min(labels_bottom, bb.y0)
    group_y = labels_bottom - 0.04
    for cx, sp_label in group_centers:
        ax.text(
            cx, group_y, sp_label, transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontweight="bold", fontsize=font_size,
        )

    # Zoom y-axis when accuracies are clustered near the top.
    tops = means + stds
    vmin, vmax = means.min(), tops.max()
    if vmin > 0 and (vmax - vmin) / vmax < 0.3:
        lo = vmin - max((vmax - vmin) * 1.5, vmax * 0.02)
        ax.set_ylim(max(0, lo), vmax + max((vmax - vmin) * 0.8, vmax * 0.02))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    print(f"Saved: {output_path}")

    csv_path = os.path.splitext(output_path)[0] + ".csv"
    rows = []
    for info, mean, std, vals in entries:
        rows.append(
            {
                "experiment": info["dirname"],
                "label": make_label(info),
                "dataset": info.get("dataset"),
                "model": info.get("model"),
                "augmentation": info.get("augmentation"),
                "sparsity": info.get("sparsity"),
                "n_seeds": len(vals),
                f"{metric}_mean": mean,
                f"{metric}_std": std,
                "seed_values": ";".join(f"{v:.6f}" for v in vals),
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Publication-ready experiment visualization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--base_dirs",
        nargs="+",
        required=True,
        help="Root dir(s) containing experiment folders.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="Glob patterns for experiment directory names.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "train_loss",
            "train/MulticlassAccuracy",
            "valid/MulticlassAccuracy",
            "sparsity",
        ],
        help="Metrics to plot (column names).",
    )
    parser.add_argument(
        "--output",
        default="results/figures/",
        help="Output directory (default: figures/).",
    )
    parser.add_argument("--font_size", type=int, default=16)
    parser.add_argument(
        "--fig_width", type=float, default=5.5, help="Figure width in inches."
    )
    parser.add_argument(
        "--fig_height",
        type=float,
        default=None,
        help="Figure height in inches (auto if omitted).",
    )
    parser.add_argument(
        "--source",
        choices=["train_log", "csv"],
        default="csv",
        help="Data source: epoch-level or step-level.",
    )
    parser.add_argument(
        "--summary-reduce",
        dest="summary_reduce",
        choices=["max", "last"],
        default="max",
        help=(
            "How each seed's accuracy summary is reduced over epochs: "
            "max (best epoch, default) or last (final epoch)."
        ),
    )
    parser.add_argument(
        "--legend-mode",
        dest="legend_mode",
        choices=["inline", "split"],
        default="inline",
        help=(
            "inline: embed legend in figure (default). "
            "split: omit legend from figure and save it as a separate "
            "<metric>_legend.pdf for shared use in LaTeX side-by-side layouts."
        ),
    )
    args = parser.parse_args()

    # Resolve output directory (backward compat: if ends with .pdf, use dirname)
    out_dir = args.output
    if out_dir.endswith(".pdf"):
        out_dir = os.path.dirname(out_dir) or "figures"
    os.makedirs(out_dir, exist_ok=True)

    experiments = discover_experiments(args.base_dirs, args.experiments)
    if not experiments:
        print("No experiments matched the given patterns.")
        return

    print(f"Found {len(experiments)} experiments:")
    for _, info in experiments:
        print(f"  {info['dirname']}  ->  {make_label(info)}")

    # Route each metric to its valid plot types automatically
    from collections import defaultdict

    curve_metrics = []
    bar_metrics = []
    summary_metrics = []
    for m in args.metrics:
        types = _valid_plot_types(m)
        if "curves" in types:
            curve_metrics.append(m)
        if "bar" in types:
            bar_metrics.append(m)
        if "summary" in types:
            summary_metrics.append(m)

    # --- Curve plots ---
    if curve_metrics:
        # 1. One file per metric (single panel)
        for m in curve_metrics:
            plot_training_curves(
                experiments,
                [m],
                os.path.join(out_dir, f"{_metric_short_name(m)}.pdf"),
                font_size=args.font_size,
                fig_width=args.fig_width,
                fig_height=None,
                source=args.source,
                log_scale=METRIC_LOG_SCALE,
                legend_mode=args.legend_mode,
            )

        # 2. Per-stage groupings (only if stage has >1 metric)
        stage_metrics = defaultdict(list)
        for m in curve_metrics:
            stage_metrics[_stage_of(m)].append(m)
        for stage, ms in stage_metrics.items():
            if len(ms) > 1:
                plot_training_curves(
                    experiments,
                    ms,
                    os.path.join(out_dir, f"{stage}_curves.pdf"),
                    font_size=args.font_size,
                    fig_width=args.fig_width,
                    fig_height=None,
                    source=args.source,
                    log_scale=METRIC_LOG_SCALE,
                    legend_mode=args.legend_mode,
                )

    # --- Bar plots ---
    for m in bar_metrics:
        plot_bar_comparison(
            experiments,
            m,
            os.path.join(out_dir, f"{_metric_short_name(m)}_bar.pdf"),
            font_size=args.font_size,
            fig_width=args.fig_width,
        )

    # --- Cross-seed summary bars (mean±std) ---
    for m in summary_metrics:
        plot_accuracy_summary(
            experiments,
            m,
            os.path.join(out_dir, f"{_metric_short_name(m)}_summary.pdf"),
            source=args.source,
            reduce=args.summary_reduce,
            font_size=args.font_size,
            fig_width=args.fig_width,
        )

    if not curve_metrics and not bar_metrics and not summary_metrics:
        print("No plottable metrics found.")


if __name__ == "__main__":
    main()
