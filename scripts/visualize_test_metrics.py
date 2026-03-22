#!/usr/bin/env python3
"""Visualize cross-experiment EER / minDCF scores from aggregated CSV leaderboards.

Reads eer_leaderboard.csv (produced by aggregate_json_scores.py) and generates
publication-quality grouped bar charts — one PDF per (dataset, metric) pair.

Datasets with multiple evaluation protocols (e.g. CNCeleb concatenated vs multi)
are shown as side-by-side subplots sharing a y-axis.

Usage:
    python scripts/visualize_test_metrics.py \\
        --input_dir results/cross_exp_comparison/test_metrics
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Import shared utilities from visualize.py (same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize import (
    setup_matplotlib,
    METHOD_CLASS_COLORS,
    METHOD_DISPLAY_NAMES,
    parse_experiment_name,
)

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Dataset / protocol parsing
# ---------------------------------------------------------------------------

DATASET_PROTOCOL_MAP = {
    "cnceleb_concatenated": ("CNCeleb", "Concatenated"),
    "cnceleb_multi":        ("CNCeleb", "Multi-enrollment"),
    "voxceleb1_O":          ("VoxCeleb", "VoxCeleb1-O"),
    "voxceleb1_E":          ("VoxCeleb", "VoxCeleb1-E"),
    "voxceleb1_H":          ("VoxCeleb", "VoxCeleb1-H"),
}

# Consistent method ordering for x-axis
METHOD_ORDER = ["linbreg", "adabreg", "adabregw", "adabregl2",
                "vanilla", "wespeaker", "pruning_struct", "pruning_unstruct"]

SPARSITY_HATCHES = {90: "", 95: "//"}


def parse_dataset_protocol(raw_name):
    """Map CSV dataset column to (dataset_name, protocol_name)."""
    if raw_name in DATASET_PROTOCOL_MAP:
        return DATASET_PROTOCOL_MAP[raw_name]
    # Fallback: try splitting on last underscore
    parts = raw_name.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0].replace("_", " ").title(), parts[1]
    return raw_name, "default"


def _method_sort_key(method_class):
    try:
        return METHOD_ORDER.index(method_class)
    except ValueError:
        return len(METHOD_ORDER)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_metric_for_dataset(df, dataset_name, protocols, metric, output_path,
                            font_size=10, fig_height=3.5):
    """Grouped bar chart for one dataset. One subplot per protocol."""
    setup_matplotlib(font_size)

    n_protocols = len(protocols)
    fig_width = max(4.0, 3.0 * n_protocols)
    fig, axes = plt.subplots(1, n_protocols, figsize=(fig_width, fig_height),
                             sharey=True, squeeze=False)
    axes = axes[0]

    # Discover sparsity levels present
    sparsities = sorted(df["sparsity"].dropna().unique())
    n_sparsity = len(sparsities)
    bar_width = 0.7 / max(n_sparsity, 1)

    # Determine y-axis limit: use IQR to detect outliers, cap y to show
    # the non-outlier range clearly while still reporting outlier values.
    all_vals = df[metric].dropna().values
    q1, q3 = np.percentile(all_vals, [25, 75])
    iqr = q3 - q1
    outlier_thresh = q3 + 2.5 * iqr
    non_outlier_max = all_vals[all_vals <= outlier_thresh].max()
    y_cap = non_outlier_max * 1.35  # headroom for annotations

    for ax_idx, protocol in enumerate(sorted(protocols)):
        ax = axes[ax_idx]
        sub = df[df["protocol"] == protocol].copy()

        # Best (lowest) score in this protocol for bold highlighting
        valid_vals = sub[metric].dropna()
        best_val = valid_vals[valid_vals > 0].min() if len(valid_vals) > 0 else None

        # Get unique methods present, sorted consistently
        methods = sorted(sub["method_class"].unique(), key=_method_sort_key)
        n_methods = len(methods)

        x = np.arange(n_methods)
        offsets = np.linspace(-(n_sparsity - 1) / 2 * bar_width,
                               (n_sparsity - 1) / 2 * bar_width,
                               n_sparsity) if n_sparsity > 1 else [0.0]

        for sp_idx, sp in enumerate(sparsities):
            vals = []
            colors = []
            for method in methods:
                row = sub[(sub["method_class"] == method) & (sub["sparsity"] == sp)]
                if len(row) > 0:
                    vals.append(row[metric].values[0])
                    colors.append(METHOD_CLASS_COLORS.get(method, "#333333"))
                else:
                    vals.append(0)
                    colors.append("#cccccc")

            vals = np.array(vals)
            # Clip display height for outliers; actual value shown in annotation
            clip_height = non_outlier_max * 1.12
            display_vals = np.clip(vals, 0, clip_height)
            hatch = SPARSITY_HATCHES.get(sp, "")
            bars = ax.bar(x + offsets[sp_idx], display_vals, bar_width,
                          color=colors, edgecolor="white", linewidth=0.5,
                          hatch=hatch)

            # Value annotations (in %, e.g. 11.2)
            use_latex = plt.rcParams.get("text.usetex", False)
            def _bold(s):
                return rf"\textbf{{{s}}}" if use_latex else s

            for bar, v, c in zip(bars, vals, colors):
                if v <= 0:
                    continue
                is_outlier = v > outlier_thresh
                is_best = best_val is not None and np.isclose(v, best_val)
                raw_text = f"{v * 100:.1f}"
                if is_outlier:
                    bx = bar.get_x()
                    bw = bar.get_width()
                    top = bar.get_height()
                    # White band across bar top to indicate truncation
                    band_h = y_cap * 0.015
                    ax.fill_between([bx, bx + bw], top - band_h, top + band_h,
                                    color="white", zorder=4)
                    # Small colored cap above the break to show bar continues
                    cap_h = y_cap * 0.025
                    cap_bot = top + band_h + y_cap * 0.005
                    ax.bar(bar.get_x() + bw / 2, cap_h, bw, bottom=cap_bot,
                           color=c, edgecolor="white", linewidth=0.5,
                           hatch=hatch, zorder=3)
                    # Label above the cap
                    ax.text(bx + bw / 2, cap_bot + cap_h,
                            raw_text, ha="center", va="bottom",
                            fontsize=font_size - 1.5, rotation=60)
                else:
                    bold = is_best
                    text = _bold(raw_text) if bold else raw_text
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            text, ha="center", va="bottom",
                            fontsize=font_size - (1.5 if bold else 2.5),
                            rotation=60,
                            fontweight="bold" if bold else "normal")

        tick_labels = [METHOD_DISPLAY_NAMES.get(m, m) for m in methods]
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=30, ha="right")
        ax.set_title(protocol)
        ax.set_ylim(0, y_cap)

        if ax_idx == 0:
            ax.set_ylabel(metric)

    # Legend: neutral gray patches distinguished by hatch pattern
    from matplotlib.patches import Patch
    pct = r"\%" if plt.rcParams.get("text.usetex") else "%"
    legend_handles = []
    for sp in sparsities:
        hatch = SPARSITY_HATCHES.get(sp, "")
        legend_handles.append(Patch(facecolor="#aaaaaa", edgecolor="white",
                                    hatch=hatch, label=f"{int(sp)}{pct} sparsity"))
    if legend_handles:
        fig.legend(handles=legend_handles, loc="upper right", framealpha=0.9)

    fig.suptitle(f"{dataset_name} — {metric}", y=1.02)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir",
                        default="results/cross_exp_comparison/test_metrics",
                        help="Directory containing eer_leaderboard.csv")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for figures (default: <input_dir>/figures)")
    parser.add_argument("--font_size", type=int, default=10)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.input_dir, "figures")

    # Load data
    csv_path = os.path.join(args.input_dir, "eer_leaderboard.csv")
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=["dataset", "exp"])

    # Parse experiment names to get method_class and sparsity
    parsed = df["exp"].apply(parse_experiment_name)
    df["method_class"] = parsed.apply(lambda x: x["method_class"])
    df["sparsity"] = parsed.apply(lambda x: x["sparsity"])

    # Parse dataset column into (dataset, protocol)
    dp = df["dataset"].apply(parse_dataset_protocol)
    df["dataset_name"] = dp.apply(lambda x: x[0])
    df["protocol"] = dp.apply(lambda x: x[1])

    # Ensure metrics are numeric
    for col in ["EER", "minDCF"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Generate one PDF per (dataset, metric)
    for metric in ["EER", "minDCF"]:
        if metric not in df.columns:
            continue
        for dataset_name, group in df.groupby("dataset_name"):
            protocols = group["protocol"].unique()
            safe_name = dataset_name.lower().replace(" ", "_")
            out_path = os.path.join(output_dir, f"{safe_name}_{metric.lower()}.pdf")
            plot_metric_for_dataset(group, dataset_name, protocols, metric,
                                   out_path, font_size=args.font_size)


if __name__ == "__main__":
    main()
