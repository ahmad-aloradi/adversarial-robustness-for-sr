#!/usr/bin/env python3
"""Line plots of EER / minDCF vs target sparsity for each method.

Reads eer_leaderboard.csv (produced by aggregate_json_scores.py) and generates
publication-quality line plots — one PDF per (train_dataset, metric) pair.

Each PDF is a 2×2 grid: Vox1-O, Vox1-E, Vox1-H, CNCeleb.
Each line represents one method class (AdaBreg, LinBreg, etc.) across sparsity
levels. Dense baselines (vanilla, wespeaker) appear as horizontal dashed lines.

Usage:
    python scripts/visualize_sparsity_trend.py \\
        --input_dir results/cross_exp_comparison/test_metrics
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Import shared utilities from visualize.py (same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from visualize import (
    METHOD_CLASS_COLORS,
    METHOD_DISPLAY_NAMES,
    parse_experiment_name,
    setup_matplotlib,
)
from visualize_test_metrics import (
    parse_dataset_protocol,
    parse_train_dataset_protocol,
)

# ---------------------------------------------------------------------------
# Subplot layout: (row, col) → (dataset_name, protocol)
# CNCeleb protocol is configurable; default = "Embeds Averaging" (cnceleb_multi)
# ---------------------------------------------------------------------------

VOXCELEB_SUBPLOTS = [
    (0, 0, "VoxCeleb", "Vox1-O"),
    (0, 1, "VoxCeleb", "Vox1-E"),
    (1, 0, "VoxCeleb", "Vox1-H"),
]
CNCELEB_POS = (1, 1)

# Methods treated as dense baselines (horizontal lines, not sparsity curves)
BASELINE_METHODS = {"vanilla", "wespeaker"}

# Y-axis cap: values above this are clipped and annotated at the top
Y_CAP_MIN = 15  # percent — ensures outliers don't blow the scale

# Method ordering for consistent legend
METHOD_ORDER = [
    "adabreg",
    "adabregw",
    "adabregl2",
    "linbreg",
    "pruning_struct",
    "pruning_unstruct",
]

# Per-method marker shapes (distinct from sparsity markers used elsewhere)
METHOD_MARKERS = {
    "adabreg": "o",  # circle
    "adabregw": "s",  # square
    "adabregl2": "^",  # triangle up
    "linbreg": "D",  # diamond
    "pruning_struct": "v",  # triangle down
    "pruning_unstruct": "P",  # plus (filled)
}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_sparsity_trends(
    df,
    metric_col,
    output_path,
    font_size=10,
    cnceleb_protocol="Embeds Averaging",
):
    """Line plot: metric vs sparsity, one subplot per protocol.

    Automatically determines layout from available data — up to 2×2 grid.
    """
    setup_matplotlib(font_size)

    all_specs = [
        ("VoxCeleb", "Vox1-O"),
        ("VoxCeleb", "Vox1-E"),
        ("VoxCeleb", "Vox1-H"),
        ("CNCeleb", cnceleb_protocol),
    ]

    # Filter to specs that have data
    active_specs = []
    for dataset_name, protocol in all_specs:
        sub = df[
            (df["dataset_name"] == dataset_name) & (df["protocol"] == protocol)
        ]
        if not sub.empty and sub[metric_col].notna().sum() > 0:
            active_specs.append((dataset_name, protocol))

    n = len(active_specs)
    if n == 0:
        return

    # Choose grid layout
    if n <= 2:
        nrows, ncols = 1, n
    else:
        nrows, ncols = 2, 2

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False
    )

    # Collect all legend handles across subplots (deduplicated)
    legend_entries = {}  # label → handle

    for idx, (dataset_name, protocol) in enumerate(active_specs):
        ax = axes[idx // ncols, idx % ncols]
        sub = df[
            (df["dataset_name"] == dataset_name) & (df["protocol"] == protocol)
        ]

        # --- Baselines: horizontal dashed lines ---
        for method in sorted(BASELINE_METHODS):
            bl = sub[sub["method_class"] == method]
            if bl.empty:
                continue
            val = bl[metric_col].mean() * 100
            color = METHOD_CLASS_COLORS.get(method, "#333333")
            label = METHOD_DISPLAY_NAMES.get(method, method)
            line = ax.axhline(
                val,
                color=color,
                linestyle="-.",
                linewidth=1.2,
                alpha=0.8,
                label=label,
            )
            if label not in legend_entries:
                legend_entries[label] = line

        # --- Sparsity curves: one line per method ---
        sparse_df = sub[~sub["method_class"].isin(BASELINE_METHODS)].copy()
        sparse_df = sparse_df.dropna(subset=[metric_col, "sparsity"])

        methods = sorted(
            sparse_df["method_class"].unique(),
            key=lambda m: METHOD_ORDER.index(m)
            if m in METHOD_ORDER
            else len(METHOD_ORDER),
        )

        y_cap = Y_CAP_MIN

        for method in methods:
            mdf = sparse_df[sparse_df["method_class"] == method].sort_values(
                "sparsity"
            )
            if mdf.empty:
                continue

            x = mdf["sparsity"].values
            y_raw = mdf[metric_col].values * 100
            color = METHOD_CLASS_COLORS.get(method, "#333333")
            marker = METHOD_MARKERS.get(method, "o")
            label = METHOD_DISPLAY_NAMES.get(method, method)

            # Split into in-range segments (connected) and capped points (isolated)
            in_range = y_raw <= y_cap

            # Plot connected segments for consecutive in-range points
            line_handle = None
            seg_x, seg_y = [], []
            for xi, yi, ok in zip(x, y_raw, in_range):
                if ok:
                    seg_x.append(xi)
                    seg_y.append(yi)
                else:
                    if seg_x:
                        (h,) = ax.plot(
                            seg_x,
                            seg_y,
                            color=color,
                            marker=marker,
                            markersize=6,
                            linewidth=1.5,
                            label=label,
                        )
                        if line_handle is None:
                            line_handle = h
                        label = None  # avoid duplicate legend entries
                        seg_x, seg_y = [], []
            if seg_x:
                (h,) = ax.plot(
                    seg_x,
                    seg_y,
                    color=color,
                    marker=marker,
                    markersize=6,
                    linewidth=1.5,
                    label=label,
                )
                if line_handle is None:
                    line_handle = h
                label = None

            # Plot capped points as isolated markers at the cap line
            use_latex = plt.rcParams.get("text.usetex", False)
            pct = r"\%" if use_latex else "%"
            for xi, yi, ok in zip(x, y_raw, in_range):
                if not ok:
                    (h,) = ax.plot(
                        xi,
                        y_cap,
                        color=color,
                        marker=marker,
                        markersize=6,
                        linewidth=0,
                        label=label,
                    )
                    if line_handle is None:
                        line_handle = h
                    label = None
                    short_name = METHOD_DISPLAY_NAMES.get(method, method)
                    ax.annotate(
                        f"{short_name}: {yi:.1f}{pct}",
                        xy=(xi, y_cap),
                        fontsize=font_size - 2,
                        alpha=0.85,
                        ha="center",
                        va="bottom",
                        textcoords="offset points",
                        xytext=(0, 4),
                    )

            if line_handle is not None:
                disp_label = METHOD_DISPLAY_NAMES.get(method, method)
                if disp_label not in legend_entries:
                    legend_entries[disp_label] = line_handle

        ax.set_ylim(top=y_cap * 1.008)  # small headroom for annotations

        # --- Axis formatting ---
        use_latex = plt.rcParams.get("text.usetex", False)
        pct_str = r"\%" if use_latex else "%"
        title = protocol if dataset_name == "VoxCeleb" else dataset_name
        ax.set_title(title)
        ax.set_xlabel(f"Target sparsity ({pct_str})")
        if idx % ncols == 0:
            metric_label = metric_col.replace("_raw", "").replace("_norm", "")
            ax.set_ylabel(f"{metric_label} ({pct_str})")

    # Hide unused axes (e.g. 3 specs in a 2×2 grid)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    # --- Shared legend at bottom ---
    if legend_entries:
        fig.legend(
            legend_entries.values(),
            legend_entries.keys(),
            loc="lower center",
            ncol=min(len(legend_entries), 4),
            framealpha=0.9,
            bbox_to_anchor=(0.5, -0.02),
        )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        default="results/cross_exp_comparison/test_metrics",
        help="Directory containing eer_leaderboard.csv",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for figures (default: <input_dir>/figures)",
    )
    parser.add_argument("--font_size", type=int, default=10)
    parser.add_argument(
        "--cnceleb_protocol",
        default="Embeds Averaging",
        help="CNCeleb protocol for the 4th subplot (default: Embeds Averaging)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.input_dir, "figures")

    # Load data
    csv_path = os.path.join(args.input_dir, "eer_leaderboard.csv")
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=["dataset", "exp"])

    # Parse experiment names
    parsed = df["exp"].apply(parse_experiment_name)
    df["method_class"] = parsed.apply(lambda x: x["method_class"])
    df["sparsity"] = parsed.apply(lambda x: x["sparsity"])

    # Parse dataset column
    dp = df["dataset"].apply(parse_dataset_protocol)
    df["dataset_name"] = dp.apply(lambda x: x[0])
    df["protocol"] = dp.apply(lambda x: x[1])
    df["train_dataset"] = df["exp"].apply(parse_train_dataset_protocol)

    # Ensure metric columns are numeric
    base_metrics = ["EER", "minDCF"]
    for base in base_metrics:
        for col in [base, f"{base}_raw", f"{base}_norm"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build metric variants list
    metric_variants = []
    for base in base_metrics:
        for suffix, subdir in [
            ("_raw", "raw"),
            ("_norm", "norm"),
            ("", "raw"),
        ]:
            col = f"{base}{suffix}"
            if col in df.columns and df[col].notna().any():
                metric_variants.append((col, base, subdir))

    # Generate one PDF per (train_dataset, metric_variant)
    for col, base_name, subdir in metric_variants:
        for train_ds, train_group in df.groupby("train_dataset"):
            if train_group[col].notna().sum() == 0:
                continue
            out_path = os.path.join(
                output_dir,
                train_ds,
                subdir,
                f"sparsity_trend_{base_name.lower()}.pdf",
            )
            plot_sparsity_trends(
                train_group,
                col,
                out_path,
                font_size=args.font_size,
                cnceleb_protocol=args.cnceleb_protocol,
            )


if __name__ == "__main__":
    main()
