#!/usr/bin/env python3
"""Visualization for experiment results. The type of plots currently supported
are only training curves and bar charts for final metrics. The script
automatically determines which plot types are valid for each metric (e.g. EER
can be bar, but train_loss cannot).

Usage examples:
    # Plot training curves for specific experiments
    python scripts/visualize.py \\
        --base_dirs /dataHDD/ahmad/comfort26_sem/cnceleb /dataHDD/ahmad/21_03_2026/cnceleb \\
        --experiments "sv_bregman_adabreg-*-sr90" "sv_bregman_linbreg-*-sr95" "sv_vanilla-*" \\
        --metrics train_loss valid_loss sparsity \\
        --output figures/training_curves.pdf
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
    # forget about those other methods for now, just make them black so they stand out as "other"
    "proxsgd": "#000000",  # light gray
    "adabregw": "#000000",  # deep navy blue
    "adabregl2": "#000000",  # muted purple
}

METHOD_DISPLAY_NAMES = {
    "linbreg": "LinBreg",
    "linbreg_fixed": "LinBreg (Fixed $\lambda$)",
    "adabreg": "AdaBreg",
    "adabreg_fixed": "AdaBreg (Fixed $\lambda$)",
    "adabregw": "AdaBregW",
    "adabregl2": "AdaBregL2",
    "pruning_struct": "Str. Prun.",
    "pruning_unstruct": "Unst. Prun.",
    "proxsgd": "ProxSGD",
    "vanilla": "AdamW",
    "wespeaker": "SGD",
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
    "regl1_conv": (0, (5, 2)),
    "poor_init": (0, (3, 1, 1, 1)),
    "rescale_prox": (0, (1, 1)),
    "rescale_prox_v2": (0, (3, 1, 1, 1, 1, 1)),
    "subgrad_corr_v2": (0, (5, 1, 1, 1)),
    "subgrad_corr_v3": (0, (5, 2, 1, 2)),
    "subgrad_corr_v4": (0, (1, 2, 5, 2)),
    "fixed": (0, (4, 2, 1, 2, 1, 2)),
}

# Axis labels for known metric names
METRIC_LABELS = {
    "train_loss": "Train Loss",
    "valid_loss": "Valid. Loss",
    "train/MulticlassAccuracy": "Train Acc.",
    "valid/MulticlassAccuracy": "Valid. Acc.",
    "sparsity": r"$s(\theta)$", # "Sparsity"
    "bregman/sparsity": r"$s(\theta)$",
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
    "test": ["EER", "minDCF"],
    "internal": ["sparsity", "bregman/global_lambda", "bregman/sparsity"],
    "schedule": ["lr", "train/margin"],
}

# metric → set of valid plot types
METRIC_PLOT_TYPES = {
    # Convergence metrics — time-series curves
    "train_loss": {"curves"},
    "valid_loss": {"curves"},
    "train/MulticlassAccuracy": {"curves"},
    "valid/MulticlassAccuracy": {"curves"},
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
    "adabregl2": (r"^lr-AdaBreg(L2)?/conv_layers$", r"^lr-AdaBreg(L2)?/linear_layers$"),
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


def parse_experiment_name(dirname):
    """Parse experiment directory name into structured metadata dict.

    When ``-alpha<v>`` / ``-f<v>`` are absent we default to alpha=1.0 and
    f=50 so old runs sit naturally inside an alpha/f sweep.
    """
    info = {
        "dirname": dirname,
        "sparsity": None,
        "ramp_epochs": None,
        "schedule": None,
        "variant": None,
        "alpha": 1.0,
        "f": 50,
    }

    # Strip alpha/f hyperparameter suffixes (always at the end, e.g. "-alpha0.25-f50").
    # Order: -f<int> is innermost, then -alpha<float>. We strip both before
    # running the variant regex so they don't pollute the variant field.
    work = dirname
    m_f = re.search(r"-f(\d+)$", work)
    if m_f:
        info["f"] = int(m_f.group(1))
        work = work[: m_f.start()]
    m_alpha = re.search(r"-alpha([\d.]+)$", work)
    if m_alpha:
        info["alpha"] = float(m_alpha.group(1))
        work = work[: m_alpha.start()]

    # Sparsity: "-sr90" or "-sparsity90", possibly followed by a variant suffix
    m = re.search(r"-(sr|sparsity)(\d+)(?:-(.+))?$", work)
    if m:
        info["sparsity"] = int(m.group(2))
        if m.group(3):
            info["variant"] = m.group(3)  # e.g. "poor_init", "rescale_prox"

    # Ramp: "-ramp10_constant-"
    m = re.search(r"-ramp(\d+)_(\w+)-", work)
    if m:
        info["ramp_epochs"] = int(m.group(1))
        info["schedule"] = m.group(2)

    # Model backbone
    m = re.search(r"-(wespeaker_\w+)-", work)
    info["model"] = m.group(1) if m else "unknown"

    # Method class — order matters (adabregw before adabreg)
    prefix = (
        work.split("-wespeaker")[0] if "-wespeaker" in work else work
    )
    METHOD_PATTERNS = [
        ("adabregw", "adabregw"),
        ("adabregl2", "adabregl2"),
        ("adabreg_fixed", "adabreg"),   # must come before adabreg
        ("adabreg", "adabreg"),
        ("linbreg_fixed", "linbreg"),   # must come before linbreg
        ("linbreg", "linbreg"),
        ("proxsgd_fixed", "proxsgd"),
        ("proxsgd", "proxsgd"),
        ("pruning_mag_struct", "pruning_struct"),
        ("pruning_mag_unstruct", "pruning_unstruct"),
    ]
    info["method_class"] = "vanilla"  # default
    for pattern, cls in METHOD_PATTERNS:
        if pattern in prefix:
            info["method_class"] = cls
            if pattern.endswith("_fixed") and not info.get("variant"):
                info["variant"] = "fixed"
            break
    else:
        if prefix.replace("sv_", "") == "wespeaker":
            info["method_class"] = "wespeaker"

    return info


VARIANT_DISPLAY_NAMES = {
    "poor_init": "poor init",
    "fixed": "Fixed $\lambda$",
    "regl1_conv": "",
    "rescale_prox": "Rescale Prox.",
    "rescale_prox_v2": "Rescale Prox. V2",
    "rescale_prox_V2": "SubGrad Corr.",
    "subgrad_corr_v2": "SubGrad Corr. V2",
    "subgrad_corr_v3": "SubGrad Corr. V3",
    "subgrad_corr_v4": "SubGrad Corr. V4",
}

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
    "fixed": (0.0, -0.38, 0.12),              # slightly brighter/desaturated — same hue, visually distinct
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


def make_label(info):
    """Create a concise, consistent legend label."""
    name = METHOD_DISPLAY_NAMES.get(info["method_class"], info["method_class"])
    if info.get("variant") == "fixed" and info.get("fixed_lambda") is not None:
        lam = info["fixed_lambda"]
        lam_sym = r"$\lambda$" if plt.rcParams.get("text.usetex") else "λ"
        return f"{name} ({lam_sym}={lam:g})"
    if info["sparsity"] is not None:
        pct = r"\%" if plt.rcParams.get("text.usetex") else "%"
        label = f"{name} {info['sparsity']}{pct}"
    else:
        label = name
    if info.get("variant"):
        # Honor an explicit "" override as "suppress entirely"; fall back to
        # the raw name only for variants not listed in VARIANT_DISPLAY_NAMES,
        # so unknown variants still produce distinct labels.
        if info["variant"] in VARIANT_DISPLAY_NAMES:
            variant = VARIANT_DISPLAY_NAMES[info["variant"]]
        else:
            variant = info["variant"]
        if variant:
            label += f" ({variant})"
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
        marker = SPARSITY_MARKERS.get(info["sparsity"], "x")
        ls = VARIANT_LINESTYLES.get(info.get("variant"), "-")
    else:
        color = METHOD_CLASS_COLORS.get(info["method_class"], "#333333")
        variant = info.get("variant")
        if variant:
            adj = VARIANT_COLOR_ADJUSTMENTS.get(variant)
            if adj:
                color = _adjust_color(color, *adj)
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


def discover_experiments(base_dirs, patterns):
    """Find experiment directories matching glob patterns. Returns sorted list.

    Args:
        base_dirs: A single directory path (str) or a list of directory paths.
        patterns: Glob patterns for experiment directory names.
    """
    if isinstance(base_dirs, str):
        base_dirs = [base_dirs]

    seen = set()
    matched = []

    for base_dir in base_dirs:
        if not os.path.isdir(base_dir):
            raise ValueError(f"Warning: base dir does not exist: {base_dir}")

        all_dirs = sorted(os.listdir(base_dir))
        for pattern in patterns:
            for d in all_dirs:
                full = os.path.join(base_dir, d)
                if (
                    os.path.isdir(full)
                    and d not in seen
                    and glob.fnmatch.fnmatch(d, pattern)
                ):
                    seen.add(d)
                    info = parse_experiment_name(d)
                    if info.get("variant") == "fixed":
                        info["fixed_lambda"] = load_fixed_lambda(full)
                    matched.append((full, info))

    # Stable sort: method class order, then sparsity
    ORDER = [
        "vanilla",
        "wespeaker",
        "linbreg",
        "adabreg",
        "adabregw",
        "adabregl2",
        "pruning_struct",
        "pruning_unstruct",
    ]

    def key(item):
        mc = item[1]["method_class"]
        info = item[1]
        return (
            ORDER.index(mc) if mc in ORDER else 99,
            info["sparsity"] or -1,
            info.get("variant") or "",
            info["alpha"] if info.get("alpha") is not None else -1.0,
            info["f"] if info.get("f") is not None else -1,
        )

    matched.sort(key=key)
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

    if metric in ("sparsity", "valid/MulticlassAccuracy", "bregman/sparsity"):
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


def plot_training_curves(
    experiments,
    metrics,
    output_path,
    font_size=10,
    fig_width=5.5,
    fig_height=None,
    source="train_log",
    log_scale=None,
):
    """Plot training curves (one subplot per metric, shared x-axis)."""
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
        df = (
            load_train_log(exp_dir)
            if source == "train_log"
            else load_csv_metrics(exp_dir)
        )
        x_col = "epoch" if source == "train_log" else "step"

        if df is None:
            print(f"  [skip] no {source} data: {info['dirname']}")
            continue

        color, marker, ls = get_style(info)
        label = make_label(info)

        for ax, metric in zip(axes, metrics):
            # Virtual 'lr' metric → resolve per-experiment
            if metric == "lr":
                col, reason = resolve_lr_column(df, info)
                if col is None:
                    print(f"  [skip] {info['dirname']}: lr — {reason}")
                    continue
            else:
                col = metric
            if col not in df.columns:
                continue
            x = df[x_col].copy()
            if source == "csv":
                x = x / 1000.0
            y = df[col]
            mask = y.notna()
            n_pts = mask.sum()
            if n_pts == 0:
                continue
            # Skip constant-zero series on sparsity panel (e.g. baselines)
            if metric == "sparsity" and y[mask].max() < 1e-6:
                continue
            ax.plot(
                x[mask],
                y[mask],
                color=color,
                marker=marker,
                linestyle=ls,
                markersize=3.5,
                markevery=max(1, n_pts // 12),
                label=label,
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
        ncol = min(4, len(labels))
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

    fig.align_ylabels(axes)
    fig.subplots_adjust(hspace=0.08)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    print(f"Saved: {output_path}")


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
    tick_labels = [
        METHOD_DISPLAY_NAMES.get(info["method_class"], info["method_class"])
        for info, _ in bar_infos
    ]

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
    for m in args.metrics:
        types = _valid_plot_types(m)
        if "curves" in types:
            curve_metrics.append(m)
        if "bar" in types:
            bar_metrics.append(m)

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

    if not curve_metrics and not bar_metrics:
        print("No plottable metrics found.")


if __name__ == "__main__":
    main()
