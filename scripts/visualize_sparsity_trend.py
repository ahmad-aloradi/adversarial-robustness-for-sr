#!/usr/bin/env python3
"""Line plots of EER / minDCF vs target for each method.

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
from matplotlib.lines import Line2D
from visualize import (
    METHOD_CLASS_COLORS,
    METHOD_DISPLAY_NAMES,
    VARIANT_COLOR_ADJUSTMENTS,
    _adjust_color,
    assign_label_visibility,
    export_standalone_legend,
    info_from_csv_row,
    make_label,
    setup_matplotlib,
)
from visualize_test_metrics import (
    filter_by_exp_patterns,
    parse_dataset_protocol,
    parse_train_dataset_protocol,
    resolve_actual_sparsity,
)

# ---------------------------------------------------------------------------
# Subplot layout: (row, col) → (dataset_name, protocol)
# CNCeleb protocol is configurable; default = "Embeds Averaging" (cnceleb_multi)
# ---------------------------------------------------------------------------
VARIANT_LINESTYLES = {
    None: "-",
}


VOXCELEB_SUBPLOTS = [
    (0, 0, "VoxCeleb", "Vox1-O"),
    (0, 1, "VoxCeleb", "Vox1-E"),
    (1, 0, "VoxCeleb", "Vox1-H"),
]
CNCELEB_POS = (1, 1)

# Methods treated as dense baselines (horizontal lines, not sparsity curves)
BASELINE_METHODS = {"vanilla", "wespeaker"}

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
    sparsity_label="target",
    legend_mode="inline",
    layout="1x4",
):
    """Line plot: metric vs sparsity, one subplot per protocol.

    layout:
        "1x4" — single row of up to 4 subplots (default).
        "2x2" — 2×2 grid.

    legend_mode:
        "inline" — embed the shared legend at the bottom of the figure (default).
        "split"  — omit the inline legend and write a separate
                   ``<output_path stem>_legend.pdf`` containing only the
                   legend, so two figures can share one legend in LaTeX.
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
    if layout == "2x2":
        nrows, ncols = (1, n) if n <= 2 else (2, 2)
    else:  # "1x4" (default)
        nrows, ncols = 1, n

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False
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

        # --- Sparsity curves: one line per (method, alpha) combo ---
        sparse_df = sub[~sub["method_class"].isin(BASELINE_METHODS)].copy()
        sparse_df = sparse_df.dropna(subset=[metric_col, "sparsity"])

        methods = sorted(
            sparse_df["method_class"].unique(),
            key=lambda m: METHOD_ORDER.index(m)
            if m in METHOD_ORDER
            else len(METHOD_ORDER),
        )

        # Per-subplot cap: CNCeleb-E and the VoxCeleb protocols cover very
        # different metric ranges, so each panel scales independently via
        # IQR rather than sharing a fixed global cap.
        sub_vals = sub[metric_col].dropna().values * 100
        if len(sub_vals) == 0:
            continue
        q1, q3 = np.percentile(sub_vals, [25, 75])
        iqr = q3 - q1
        non_outliers = sub_vals[sub_vals <= q3 + 3.5 * iqr]
        non_outlier_max = (
            non_outliers.max() if len(non_outliers) > 0 else sub_vals.max()
        )
        y_cap = non_outlier_max * 1.25  # headroom for capped-point annotations
        use_latex = plt.rcParams.get("text.usetex", False)
        # Track next vertical offset (pts) for capped annotations per x position
        ann_next_offset: dict[float, int] = {}

        # Pick the swept hparam (alpha or f), build one curve per
        # (method, variant, sweep_value) so different variants at the same
        # alpha don't overwrite each other.
        sweep_param = None
        for cand in ("alpha", "f"):
            if cand in sparse_df.columns and sparse_df[cand].dropna().nunique() >= 2:
                sweep_param = cand
                break

        curve_units = []
        for method in methods:
            mdf = sparse_df[sparse_df["method_class"] == method]
            for var_key in sorted(mdf["variant"].fillna("__none__").unique()):
                vrows = mdf[mdf["variant"].fillna("__none__") == var_key]
                if sweep_param and vrows[sweep_param].notna().any():
                    vals = sorted(vrows[sweep_param].dropna().unique().tolist())
                    for v in vals:
                        curve_units.append((method, var_key, v, vals))
                else:
                    curve_units.append((method, var_key, None, []))

        # Build per-unit info dicts and route label generation through
        # visualize.make_label so legend strings match the other viz
        # scripts. Sparsity is stripped because it's the x-axis here.
        curve_unit_infos = []
        for method, var_key, sweep_val, _ in curve_units:
            cond = (
                (sparse_df["method_class"] == method)
                & (sparse_df["variant"].fillna("__none__") == var_key)
            )
            if sweep_param and sweep_val is not None:
                cond &= sparse_df[sweep_param] == sweep_val
            matching = sparse_df[cond]
            if matching.empty:
                info = {
                    "method_class": method,
                    "sparsity": None,
                    "variant": None if var_key == "__none__" else var_key,
                    "alpha": sweep_val if sweep_param == "alpha" else None,
                    "f": sweep_val if sweep_param == "f" else None,
                }
            else:
                info = dict(matching["info"].iloc[0])
                info["sparsity"] = None
            curve_unit_infos.append(info)
        assign_label_visibility([(None, info) for info in curve_unit_infos])
        curve_labels = [make_label(info) for info in curve_unit_infos]

        for unit_idx, (method, var_key, sweep_val, sweep_vals) in enumerate(
            curve_units
        ):
            base_mask = (
                (sparse_df["method_class"] == method)
                & (sparse_df["variant"].fillna("__none__") == var_key)
            )
            if sweep_param and sweep_val is not None:
                mdf = sparse_df[base_mask & (sparse_df[sweep_param] == sweep_val)]
            else:
                mdf = sparse_df[base_mask]
            mdf = mdf.sort_values("sparsity")
            if mdf.empty:
                continue

            # Fixed-lambda Bregman runs land off-target (e.g. 89.5 instead of
            # 90), so plot them at their realized sparsity. Same recipe as
            # visualize_test_metrics: actual_sparsity is resolved per-row in
            # main() via resolve_actual_sparsity. Falls back to target if
            # unresolved (e.g. --base_dirs not given).
            x_target = mdf["sparsity"].values.astype(float)
            is_fixed = var_key == "fixed"
            if "actual_sparsity" in mdf.columns and (
                is_fixed or sparsity_label == "actual"
            ):
                actual = pd.to_numeric(
                    mdf["actual_sparsity"], errors="coerce"
                ).values
                x = np.where(np.isnan(actual), x_target, actual * 100)
            else:
                x = x_target
            y_raw = mdf[metric_col].values * 100

            # Re-sort by x; actual sparsity can perturb the target ordering
            # enough to produce a non-monotonic line.
            if len(x) > 1 and not np.all(np.diff(x) >= 0):
                order = np.argsort(x)
                x = x[order]
                y_raw = y_raw[order]

            base_color = METHOD_CLASS_COLORS.get(method, "#333333")
            actual_variant = var_key if var_key != "__none__" else None
            variant_adj = VARIANT_COLOR_ADJUSTMENTS.get(actual_variant)
            if variant_adj:
                base_color = _adjust_color(base_color, *variant_adj)
            if sweep_val is not None and len(sweep_vals) >= 2:
                t = sweep_vals.index(sweep_val) / (len(sweep_vals) - 1)
                color = _adjust_color(base_color, 0, 0, 0.30 - 0.55 * t)
            else:
                color = base_color

            marker = METHOD_MARKERS.get(method, "o")
            linestyle = VARIANT_LINESTYLES.get(actual_variant, "-")
            label = curve_labels[unit_idx]

            # Capped points appear at y_cap; connections to them are dashed.
            y_plot = np.minimum(y_raw, y_cap)
            is_capped = y_raw > y_cap

            # Proxy artist for the legend (line + marker, no data plotted)
            line_handle = Line2D(
                [], [],
                color=color,
                marker=marker,
                markersize=6,
                linewidth=1.5,
                linestyle=linestyle,
                label=label,
            )
            label = None  # avoid duplicate legend entries

            # Markers at their (possibly capped) y positions
            ax.plot(x, y_plot, color=color, marker=marker, markersize=6, linewidth=0)

            # Segments: solid for in-range pairs, dashed when either endpoint is capped
            for i in range(len(x) - 1):
                seg_ls = "--" if (is_capped[i] or is_capped[i + 1]) else linestyle
                ax.plot(
                    [x[i], x[i + 1]],
                    [y_plot[i], y_plot[i + 1]],
                    color=color,
                    linewidth=1.5,
                    linestyle=seg_ls,
                )

            # Annotate capped points with their true value
            pct = r"\%" if use_latex else "%"
            for xi, yi, capped in zip(x, y_raw, is_capped):
                if capped:
                    # Stack annotations vertically so multiple capped methods
                    # at the same x don't overwrite each other.
                    x_key = round(float(xi), 4)
                    y_offset = ann_next_offset.get(x_key, 4)
                    ann_next_offset[x_key] = y_offset + 12
                    ax.annotate(
                        f"{yi:.1f}{pct}",
                        xy=(xi, y_cap),
                        fontsize=font_size - 6,
                        color=color,
                        alpha=0.9,
                        ha="center",
                        va="top",
                        textcoords="offset points",
                        xytext=(0, -2 * y_offset),
                    )

            if line_handle is not None:
                disp_label = curve_labels[unit_idx]
                if disp_label not in legend_entries:
                    legend_entries[disp_label] = line_handle

        ax.set_ylim(top=y_cap * 1.01)  # small headroom for annotations

        # --- Axis formatting ---
        pct_str = r"\%" if use_latex else "%"
        s_star = r"$\mathsf{s}^{\ast}$" if use_latex else "s*"
        title = protocol if dataset_name == "VoxCeleb" else "CNCeleb-E"
        ax.set_title(title)
        ax.set_xlabel(f"{s_star} [{pct_str}]")
        if idx % ncols == 0:
            metric_label = metric_col.replace("_raw", "").replace("_norm", "")
            ax.set_ylabel(f"{metric_label} [{pct_str}]")

        # X ticks at the actual sparsity levels present (e.g. 50, 75, 90, 95, 99)
        sparsity_levels = sorted(
            sparse_df["sparsity"].dropna().unique().tolist()
        )
        if sparsity_levels:
            ax.set_xticks(sparsity_levels)
            ax.set_xticklabels([f"{int(s)}" for s in sparsity_levels])

        # Place gridlines behind data
        ax.set_axisbelow(True)

    # Hide unused axes (e.g. 3 specs in a 2×2 grid)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    # --- Shared legend ---
    if legend_entries:
        handles = list(legend_entries.values())
        labels = list(legend_entries.keys())
        ncol = min(len(legend_entries), 10)
        if legend_mode == "inline":
            fig.legend(
                handles,
                labels,
                loc="upper center",
                fontsize=font_size - 2,
                ncol=ncol,
                framealpha=0.9,
                bbox_to_anchor=(0.5, -0.03) if n > 1 else (0.5, -0.1),
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
    parser.add_argument("--font_size", type=int, default=18)
    parser.add_argument(
        "--base_dirs",
        nargs="+",
        default=None,
        help=(
            "Optional experiment root dir(s); used only to resolve "
            "_bregman_lambda from config_tree.log so that fixed-lambda runs "
            "are labeled e.g. 'AdaBreg (λ=1e-3)' instead of '[fixed]'."
        ),
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help=(
            "Optional fnmatch glob patterns to keep only matching `exp` rows "
            "from the leaderboard CSV. Use the same patterns you pass to "
            "scripts/visualize.py so the trend plots match the convergence "
            "curves and bar charts."
        ),
    )
    parser.add_argument(
        "--cnceleb_protocol",
        default="Embeds Averaging",
        help="CNCeleb protocol for the 4th subplot (default: Embeds Averaging)",
    )
    parser.add_argument(
        "--sparsity_label",
        choices=["target", "actual"],
        default="target",
        help=(
            "X position of plotted points: 'target' uses the integer from "
            "the experiment name (e.g. 90); 'actual' uses the realized "
            "sparsity. Fixed-lambda Bregman runs always use the realized "
            "sparsity regardless of this toggle."
        ),
    )
    parser.add_argument(
        "--fixed_lambda_test_ckpt",
        choices=["best", "last"],
        default="best",
        help=(
            "For Bregman fixed-lambda runs only: how the test was performed. "
            "'best' parses the sr fraction from the test ckpt path in "
            "train.log (same as non-fixed runs). 'last' uses the last "
            "bregman/sparsity from csv/version_*/metrics.csv, since last.ckpt "
            "has no sr in its filename."
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
    parser.add_argument(
        "--layout",
        choices=["1x4", "2x2"],
        default="1x4",
        help=(
            "Subplot grid layout. '1x4' (default) places all protocols in a "
            "single row; '2x2' uses a 2×2 grid."
        ),
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.input_dir, "figures")

    # Load data
    csv_path = os.path.join(args.input_dir, "eer_leaderboard.csv")
    df = pd.read_csv(csv_path)
    df = filter_by_exp_patterns(df, args.experiments)
    if args.experiments:
        if df.empty:
            print(
                f"No experiments in {csv_path} matched any of "
                f"{args.experiments!r}; nothing to plot."
            )
            return
        kept = sorted(df["exp"].astype(str).unique())
        print(
            f"Filtering by --experiments kept {len(kept)} experiments "
            f"from {csv_path}:"
        )
        for name in kept:
            print(f"  - {name}")
    df = df.drop_duplicates(subset=["dataset", "exp"])

    # Parse experiment names. info_from_csv_row also loads _bregman_lambda
    # from config_tree.log when --base_dirs is provided, so fixed-lambda
    # runs get labels like "AdaBreg (λ=1e-3)" instead of "[fixed]".
    parsed = df["exp"].apply(lambda e: info_from_csv_row(e, args.base_dirs))
    df["info"] = parsed
    df["method_class"] = parsed.apply(lambda x: x["method_class"])
    df["sparsity"] = parsed.apply(lambda x: x["sparsity"])
    df["alpha"] = parsed.apply(lambda x: x.get("alpha"))
    df["f"] = parsed.apply(lambda x: x.get("f"))
    df["variant"] = parsed.apply(lambda x: x.get("variant"))

    # Realized sparsity per run (best ckpt's sr, or last logged for
    # fixed-lambda + last.ckpt). Falls back to None if base_dirs not given
    # or the source files aren't present; the curve then drops back to the
    # target sparsity on the x-axis for that point.
    df["actual_sparsity"] = df.apply(
        lambda r: resolve_actual_sparsity(
            r["exp"], args.base_dirs, r["info"], args.fixed_lambda_test_ckpt
        ),
        axis=1,
    )

    # Parse dataset column
    dp = df["dataset"].apply(parse_dataset_protocol)
    df["dataset_name"] = dp.apply(lambda x: x[0])
    df["protocol"] = dp.apply(lambda x: x[1])
    df["train_dataset"] = df["exp"].apply(parse_train_dataset_protocol)

    # Ensure metric columns are numeric. Norm-cohort variants are
    # intentionally ignored: trends only plot the raw metric.
    base_metrics = ["EER", "minDCF"]
    SCORES =  "norm" # "raw" "norm"
    for base in base_metrics:
        for col in [base, f"{base}_{SCORES}"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # One column per base metric: prefer SCORES when present, otherwise the
    # unqualified column. This avoids two iterations writing the same PDF.
    metric_variants = []

    for base in base_metrics:
        raw_col = f"{base}_{SCORES}"
        if raw_col in df.columns and df[raw_col].notna().any():
            metric_variants.append((raw_col, base))
        elif base in df.columns and df[base].notna().any():
            metric_variants.append((base, base))

    # Generate one PDF per (train_dataset, metric)
    for col, base_name in metric_variants:
        for train_ds, train_group in df.groupby("train_dataset"):
            if train_group[col].notna().sum() == 0:
                continue
            out_path = os.path.join(
                output_dir,
                train_ds,
                f"sparsity_trend_{base_name.lower()}.pdf",
            )
            plot_sparsity_trends(
                train_group,
                col,
                out_path,
                font_size=args.font_size,
                cnceleb_protocol=args.cnceleb_protocol,
                sparsity_label=args.sparsity_label,
                legend_mode=args.legend_mode,
                layout=args.layout,
            )


if __name__ == "__main__":
    main()
