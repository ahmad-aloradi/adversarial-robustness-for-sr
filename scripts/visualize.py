#!/usr/bin/env python3
"""Visualization for experiment results. The type of plots currently supported are only training curves and bar charts 
for final metrics. The script automatically determines which plot types are valid for each metric (e.g. EER can be bar, 
but train_loss cannot).

Usage examples:
    # Plot training curves for specific experiments
    python scripts/visualize.py \\
        --base_dir /dataHDD/ahmad/comfort26_sem/cnceleb \\
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

# ---------------------------------------------------------------------------
# 1. LaTeX-style rendering setup
# ---------------------------------------------------------------------------

matplotlib.use("pdf")


def _latex_available():
    """Check if a usable LaTeX installation exists (with required packages)."""
    import shutil
    import subprocess
    import tempfile

    if not shutil.which("pdflatex"):
        return False
    try:
        test_tex = (
            r"\documentclass{article}"
            r"\usepackage{type1cm}\usepackage{type1ec}"
            r"\begin{document}x\end{document}"
        )
        with tempfile.NamedTemporaryFile(suffix=".tex", mode="w", delete=False) as f:
            f.write(test_tex)
            tmp = f.name
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tmp],
            capture_output=True, timeout=10, cwd=os.path.dirname(tmp),
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
        print("Note: Full LaTeX not available; using mathtext fallback (still serif).")

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
    "linbreg": "#1f77b4",       # blue
    "adabreg": "#2ca02c",       # green
    "adabregw": "#006d5b",      # dark teal (distinct from blue)
    "pruning_struct": "#d62728",    # red
    "pruning_unstruct": "#ff7f0e",  # orange
    "vanilla": "#7f7f7f",       # gray
    "wespeaker": "#9467bd",     # purple
}

METHOD_DISPLAY_NAMES = {
    "linbreg": "LinBreg",
    "adabreg": "AdaBreg",
    "adabregw": "AdaBregW",
    "pruning_struct": "Struct. Pruning",
    "pruning_unstruct": "Unstruct. Pruning",
    "vanilla": "Baseline Adam",
    "wespeaker": "Baseline SGD",
}

# Sparsity → marker shape  (consistent everywhere)
SPARSITY_MARKERS = {
    None: "s",   # square  — dense / baseline
    0:    "s",
    50:   "D",   # diamond
    75:   "^",   # triangle up
    90:   "v",   # triangle down
    95:   "o",   # circle
}

# Sparsity → line dash pattern
SPARSITY_LINESTYLES = {
    None: "-",
    0:    "-",
    50:   (0, (5, 3)),
    75:   (0, (3, 1, 1, 1)),
    90:   "--",
    95:   ":",
}

# Axis labels for known metric names
METRIC_LABELS = {
    "train_loss": "Train Loss",
    "valid_loss": "Valid. Loss",
    "train/MulticlassAccuracy": "Train Acc.",
    "valid/MulticlassAccuracy": "Valid. Acc.",
    "sparsity": "Sparsity",
    "bregman/sparsity": "Sparsity",
    "bregman/global_lambda": r"$\lambda$",
    "EER": "EER",
    "minDCF": "minDCF",
}

# stage → metrics in that stage
METRIC_STAGES = {
    "train": ["train_loss", "train/MulticlassAccuracy"],
    "valid": ["valid_loss", "valid/MulticlassAccuracy"],
    "test":  ["EER", "minDCF"],
    "internal": ["sparsity", "bregman/global_lambda", "bregman/sparsity"],
}

# metric → set of valid plot types
METRIC_PLOT_TYPES = {
    # Convergence metrics — time-series curves
    "train_loss":               {"curves"},
    "valid_loss":               {"curves"},
    "train/MulticlassAccuracy": {"curves"},
    "valid/MulticlassAccuracy": {"curves"},
    # Internal / regularizer metrics — curves only
    "sparsity":                 {"curves"},
    "bregman/global_lambda":    {"curves"},
    "bregman/sparsity":         {"curves"},
    # Binary test metrics — bar (comparison) and scatter (correlation)
    "EER":                      {"bar", "scatter"},
    "minDCF":                   {"bar", "scatter"},
}
# Unknown metrics default to {"curves"}

# Metrics that use log-scale y-axis by default
METRIC_LOG_SCALE = {
    "bregman/global_lambda",
}


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
    """Parse experiment directory name into structured metadata dict."""
    info = {
        "dirname": dirname,
        "sparsity": None,
        "ramp_epochs": None,
        "schedule": None,
    }

    # Sparsity: new format "-sr90" or old format "-sparsity90"
    m = re.search(r"-(sr|sparsity)(\d+)$", dirname)
    if m:
        info["sparsity"] = int(m.group(2))

    # Ramp: "-ramp10_constant-"
    m = re.search(r"-ramp(\d+)_(\w+)-", dirname)
    if m:
        info["ramp_epochs"] = int(m.group(1))
        info["schedule"] = m.group(2)

    # Model backbone
    m = re.search(r"-(wespeaker_\w+)-", dirname)
    info["model"] = m.group(1) if m else "unknown"

    # Method class — order matters (adabregw before adabreg)
    prefix = dirname.split("-wespeaker")[0] if "-wespeaker" in dirname else dirname
    METHOD_PATTERNS = [
        ("adabregw",        "adabregw"),
        ("adabreg",         "adabreg"),
        ("linbreg",         "linbreg"),
        ("pruning_mag_struct",   "pruning_struct"),
        ("pruning_mag_unstruct", "pruning_unstruct"),
    ]
    info["method_class"] = "vanilla"  # default
    for pattern, cls in METHOD_PATTERNS:
        if pattern in prefix:
            info["method_class"] = cls
            break
    else:
        if prefix.replace("sv_", "") == "wespeaker":
            info["method_class"] = "wespeaker"

    return info


def make_label(info):
    """Create a concise, consistent legend label."""
    name = METHOD_DISPLAY_NAMES.get(info["method_class"], info["method_class"])
    if info["sparsity"] is not None:
        pct = r"\%" if plt.rcParams.get("text.usetex") else "%"
        return f"{name} {info['sparsity']}{pct}"
    return name


def get_style(info):
    """Return (color, marker, linestyle) tuple — deterministic from metadata."""
    color = METHOD_CLASS_COLORS.get(info["method_class"], "#333333")
    marker = SPARSITY_MARKERS.get(info["sparsity"], "x")
    ls = SPARSITY_LINESTYLES.get(info["sparsity"], "-")
    return color, marker, ls


# ---------------------------------------------------------------------------
# 4. Data loading
# ---------------------------------------------------------------------------

def load_train_log(exp_dir):
    """Load epoch-level metrics from train_log.txt → DataFrame."""
    path = os.path.join(exp_dir, "train_log.txt")
    if not os.path.exists(path):
        return None

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


def load_csv_metrics(exp_dir):
    """Load step-level metrics from csv/version_0/metrics.csv → DataFrame."""
    path = os.path.join(exp_dir, "csv", "version_0", "metrics.csv")
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    if "step" in df.columns:
        df = df.groupby("step").last().reset_index()
    return df


def discover_experiments(base_dir, patterns):
    """Find experiment directories matching glob patterns. Returns sorted list."""
    all_dirs = sorted(os.listdir(base_dir))
    seen = set()
    matched = []

    for pattern in patterns:
        for d in all_dirs:
            full = os.path.join(base_dir, d)
            if os.path.isdir(full) and d not in seen and glob.fnmatch.fnmatch(d, pattern):
                seen.add(d)
                matched.append((full, parse_experiment_name(d)))

    # Stable sort: method class order, then sparsity
    ORDER = ["vanilla", "wespeaker", "linbreg", "adabreg", "adabregw",
             "pruning_struct", "pruning_unstruct"]

    def key(item):
        mc = item[1]["method_class"]
        return (ORDER.index(mc) if mc in ORDER else 99,
                item[1]["sparsity"] or -1)

    matched.sort(key=key)
    return matched


# ---------------------------------------------------------------------------
# 5. Plotting
# ---------------------------------------------------------------------------

def _auto_ylim(ax, metric, margin=0.05):
    """Tighten y-axis to data range with smart zooming.

    For bounded [0,1] metrics like sparsity and accuracy, zooms to the
    region where the interesting data lives (ignores constant-zero baselines
    if other curves are far away).
    """
    lines = ax.get_lines()
    if not lines:
        return

    all_y = np.concatenate([l.get_ydata() for l in lines])
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


def plot_training_curves(experiments, metrics, output_path, font_size=10,
                         fig_width=5.5, fig_height=None, source="train_log",
                         log_scale=None):
    """Plot training curves (one subplot per metric, shared x-axis)."""
    if log_scale is None:
        log_scale = set()
    setup_matplotlib(font_size)

    n = len(metrics)
    if fig_height is None:
        fig_height = 2.4 * n + 0.4

    fig, axes = plt.subplots(n, 1, figsize=(fig_width, fig_height),
                             sharex=True, squeeze=False)
    axes = axes.flatten()

    for exp_dir, info in experiments:
        df = load_train_log(exp_dir) if source == "train_log" else load_csv_metrics(exp_dir)
        x_col = "epoch" if source == "train_log" else "step"

        if df is None:
            print(f"  [skip] no {source} data: {info['dirname']}")
            continue

        color, marker, ls = get_style(info)
        label = make_label(info)

        for ax, metric in zip(axes, metrics):
            if metric not in df.columns:
                continue
            x = df[x_col].copy()
            if source == "csv":
                x = x / 1000.0
            y = df[metric]
            mask = y.notna()
            n_pts = mask.sum()
            if n_pts == 0:
                continue
            # Skip constant-zero series on sparsity panel (e.g. baselines)
            if metric == "sparsity" and y[mask].max() < 1e-6:
                continue
            ax.plot(
                x[mask], y[mask],
                color=color, marker=marker, linestyle=ls,
                markersize=3.5, markevery=max(1, n_pts // 12),
                label=label,
            )

    # Axis formatting
    for ax, metric in zip(axes, metrics):
        ax.set_ylabel(METRIC_LABELS.get(metric, metric.replace("_", " ").title()))
        if metric in log_scale:
            ax.set_yscale("log")
        else:
            _auto_ylim(ax, metric)
    axes[-1].set_xlabel("Epoch" if source == "train_log" else "Steps [k]")

    # Deduplicated legend at top
    handles, labels = [], []
    seen_labels = set()
    for h, l in zip(*axes[0].get_legend_handles_labels()):
        if l not in seen_labels:
            seen_labels.add(l)
            handles.append(h)
            labels.append(l)

    if handles:
        # Choose ncol so rows are balanced (prefer 3 cols for 6 items, etc.)
        # ncol = min(3, len(labels))
        ncol = 1
        # if len(labels) <= 4:
        #     ncol = len(labels)
        fig.legend(handles, labels, loc="lower left",
                   ncol=ncol,
                   bbox_to_anchor=(0.65, 0.11), frameon=True,
                   columnspacing=0.8, handletextpad=0.1)

    fig.align_ylabels(axes)
    fig.subplots_adjust(hspace=0.08, top=0.93)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_bar_comparison(experiments, metric, output_path, font_size=10,
                        fig_width=5.5, fig_height=3.0, epoch=-1):
    """Grouped bar chart comparing a metric (last epoch), grouped by sparsity."""
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
    tick_labels = [METHOD_DISPLAY_NAMES.get(info["method_class"], info["method_class"])
                   for info, _ in bar_infos]

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    bars = ax.bar(x_positions, values, color=colors, edgecolor="white",
                  linewidth=0.5, width=bar_width)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_labels, rotation=25, ha="right")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))

    # Value labels on top of each bar
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=font_size - 2)

    # Sparsity group labels below x-axis
    for cx, sp_label in group_centers:
        ax.text(cx, -0.12, sp_label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontweight="bold", fontsize=font_size)

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
    parser.add_argument("--base_dir", required=True,
                        help="Root dir containing experiment folders.")
    parser.add_argument("--experiments", nargs="+", required=True,
                        help="Glob patterns for experiment directory names.")
    parser.add_argument("--metrics", nargs="+",
                        default=["train_loss", "train/MulticlassAccuracy", "valid/MulticlassAccuracy", "sparsity"],
                        help="Metrics to plot (column names).")
    parser.add_argument("--output", default="results/figures/",
                        help="Output directory (default: figures/).")
    parser.add_argument("--font_size", type=int, default=10)
    parser.add_argument("--fig_width", type=float, default=5.5,
                        help="Figure width in inches.")
    parser.add_argument("--fig_height", type=float, default=None,
                        help="Figure height in inches (auto if omitted).")
    parser.add_argument("--source", choices=["train_log", "csv"],
                        default="csv",
                        help="Data source: epoch-level or step-level.")
    args = parser.parse_args()

    # Resolve output directory (backward compat: if ends with .pdf, use dirname)
    out_dir = args.output
    if out_dir.endswith(".pdf"):
        out_dir = os.path.dirname(out_dir) or "figures"
    os.makedirs(out_dir, exist_ok=True)

    experiments = discover_experiments(args.base_dir, args.experiments)
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
                experiments, [m],
                os.path.join(out_dir, f"{_metric_short_name(m)}.pdf"),
                font_size=args.font_size, fig_width=args.fig_width,
                fig_height=None, source=args.source,
                log_scale=METRIC_LOG_SCALE,
            )

        # 2. Per-stage groupings (only if stage has >1 metric)
        stage_metrics = defaultdict(list)
        for m in curve_metrics:
            stage_metrics[_stage_of(m)].append(m)
        for stage, ms in stage_metrics.items():
            if len(ms) > 1:
                plot_training_curves(
                    experiments, ms,
                    os.path.join(out_dir, f"{stage}_curves.pdf"),
                    font_size=args.font_size, fig_width=args.fig_width,
                    fig_height=None, source=args.source,
                    log_scale=METRIC_LOG_SCALE,
                )

    # --- Bar plots ---
    for m in bar_metrics:
        plot_bar_comparison(
            experiments, m,
            os.path.join(out_dir, f"{_metric_short_name(m)}_bar.pdf"),
            font_size=args.font_size, fig_width=args.fig_width,
        )

    if not curve_metrics and not bar_metrics:
        print("No plottable metrics found.")


if __name__ == "__main__":
    main()
