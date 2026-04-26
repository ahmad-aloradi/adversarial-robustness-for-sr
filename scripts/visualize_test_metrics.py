#!/usr/bin/env python3
"""Visualize cross-experiment EER / minDCF scores from aggregated CSV
leaderboards.

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
import matplotlib.pyplot as plt
from visualize import (
    METHOD_CLASS_COLORS,
    VARIANT_COLOR_ADJUSTMENTS,
    _adjust_color,
    assign_label_visibility,
    info_from_csv_row,
    make_label,
    setup_matplotlib,
)

# ---------------------------------------------------------------------------
# Dataset / protocol parsing
# ---------------------------------------------------------------------------

TRAIN_DATASET_PROTOCOL_MAP = {
    "cnceleb": "cnceleb",
    "multi_sv_cnc": "multi_sv_cnc",  # must come before 'multi_sv'
    "multi_sv": "voxceleb",
    "voxceleb": "voxceleb",
}

DATASET_PROTOCOL_MAP = {
    "cnceleb_concatenated": ("CNCeleb", "Concat-Enroll Utterances"),
    "cnceleb_multi": ("CNCeleb", "Embeds Averaging"),
    "voxceleb_veri_test2": ("VoxCeleb", "Vox1-O"),
    "voxceleb_veri_test_extended2": ("VoxCeleb", "Vox1-E"),
    "voxceleb_veri_test_hard2": ("VoxCeleb", "Vox1-H"),
}

# Consistent method ordering for x-axis
METHOD_ORDER = [
    "linbreg",
    "adabreg",
    "adabregw",
    "adabregl2",
    "vanilla",
    "wespeaker",
    "pruning_struct",
    "pruning_unstruct",
]

PROTOCOL_ORDER = ["Vox1-O", "Vox1-E", "Vox1-H"]

SPARSITY_HATCHES = {50: ".", 70: "|", 90: "", 95: "/", 99: "x"}


def _gradient_color(method_class, variant, value, sorted_values):
    """Lightness gradient over a swept hyperparameter (e.g. alpha).

    Variant color adjustment is applied first so bars from different variants
    of the same method remain visually distinct even at the same sweep value.
    """
    base = METHOD_CLASS_COLORS.get(method_class, "#333333")
    adj = VARIANT_COLOR_ADJUSTMENTS.get(variant)
    if adj:
        base = _adjust_color(base, *adj)
    if value is None or len(sorted_values) < 2:
        return base
    rank = sorted_values.index(value)
    t = rank / (len(sorted_values) - 1)
    return _adjust_color(base, 0, 0, 0.30 - 0.55 * t)


def parse_train_dataset_protocol(exp_name):
    """Map CSV experiment name to (train_dataset, protocol_name)."""
    for key in TRAIN_DATASET_PROTOCOL_MAP:
        if key in exp_name:
            return TRAIN_DATASET_PROTOCOL_MAP[key]
    raise ValueError(f"Unknown train dataset in experiment name: {exp_name}")


def parse_dataset_protocol(raw_name):
    """Map CSV dataset column to (dataset_name, protocol_name) or None if
    unknown."""
    if raw_name in DATASET_PROTOCOL_MAP:
        return DATASET_PROTOCOL_MAP[raw_name]
    raise ValueError(f"Unknown dataset in experiment name: {raw_name}")


def _method_sort_key(method_class):
    try:
        return METHOD_ORDER.index(method_class)
    except ValueError:
        return len(METHOD_ORDER)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_metric_for_dataset(
    df,
    dataset_name,
    protocols,
    metric,
    output_path,
    font_size=10,
    fig_height=3.5,
):
    """Grouped bar chart for one dataset.

    One subplot per protocol.
    """
    setup_matplotlib(font_size)

    n_protocols = len(protocols)
    fig_width = max(4.0, 3.0 * n_protocols)
    fig, axes = plt.subplots(
        1,
        n_protocols,
        figsize=(fig_width, fig_height),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    # Discover sparsity levels present
    sparsities = sorted(df["sparsity"].dropna().unique())
    n_sparsity = len(sparsities)
    bar_width = 0.7 / max(n_sparsity, 1)

    # Determine y-axis limit: use IQR to detect outliers, cap y to show
    # the non-outlier range clearly while still reporting outlier values.
    all_vals = df[metric].dropna().values
    if len(all_vals) == 0:
        plt.close(fig)
        return
    q1, q3 = np.percentile(all_vals, [25, 75])
    iqr = q3 - q1
    outlier_thresh = q3 + 2.5 * iqr
    non_outlier_max = all_vals[all_vals <= outlier_thresh].max()
    y_cap = non_outlier_max * 1.35  # headroom for annotations

    def _protocol_sort_key(p):
        try:
            return PROTOCOL_ORDER.index(p)
        except ValueError:
            return len(PROTOCOL_ORDER)

    for ax_idx, protocol in enumerate(
        sorted(protocols, key=_protocol_sort_key)
    ):
        ax = axes[ax_idx]
        sub = df[df["protocol"] == protocol].copy()

        # Best (lowest) score in this protocol for bold highlighting
        valid_vals = sub[metric].dropna()
        positive_vals = valid_vals[valid_vals > 0]
        best_display = (
            f"{float(positive_vals.min()) * 100:.1f}"
            if len(positive_vals) > 0
            else None
        )

        # Pick the swept hparam (alpha or f, whichever varies); expand each
        # method into one bar per (variant, swept value) so different
        # variants at the same alpha don't overwrite each other.
        sweep_param = None
        for cand in ("alpha", "f"):
            if cand in sub.columns and sub[cand].dropna().nunique() >= 2:
                sweep_param = cand
                break

        units = []  # (method, variant, sweep_value)
        for method in sorted(sub["method_class"].unique(), key=_method_sort_key):
            mrows = sub[sub["method_class"] == method]
            variants = sorted(
                mrows["variant"].fillna("__none__").unique().tolist()
            )
            for var_key in variants:
                vrows = mrows[mrows["variant"].fillna("__none__") == var_key]
                if sweep_param and vrows[sweep_param].notna().any():
                    for v in sorted(vrows[sweep_param].dropna().unique()):
                        units.append((method, var_key, v))
                else:
                    units.append((method, var_key, None))

        n_methods = len(units)
        # Per-(method, variant) sorted sweep values, used for gradient ranking
        sweep_value_lookup = {}
        for method, var_key, _ in units:
            k = (method, var_key)
            if k not in sweep_value_lookup:
                vrows = sub[
                    (sub["method_class"] == method)
                    & (sub["variant"].fillna("__none__") == var_key)
                ]
                sweep_value_lookup[k] = (
                    sorted(vrows[sweep_param].dropna().unique().tolist())
                    if sweep_param
                    else []
                )

        x = np.arange(n_methods)
        offsets = (
            np.linspace(
                -(n_sparsity - 1) / 2 * bar_width,
                (n_sparsity - 1) / 2 * bar_width,
                n_sparsity,
            )
            if n_sparsity > 1
            else [0.0]
        )

        for sp_idx, sp in enumerate(sparsities):
            vals = []
            colors = []
            for method, var_key, sweep_val in units:
                cond = (
                    (sub["method_class"] == method)
                    & (sub["sparsity"] == sp)
                    & (sub["variant"].fillna("__none__") == var_key)
                )
                if sweep_param and sweep_val is not None:
                    cond &= sub[sweep_param] == sweep_val
                row = sub[cond]
                if len(row) > 0:
                    vals.append(row[metric].values[0])
                    actual_variant = var_key if var_key != "__none__" else None
                    colors.append(
                        _gradient_color(
                            method,
                            actual_variant,
                            sweep_val,
                            sweep_value_lookup.get((method, var_key), []),
                        )
                    )
                else:
                    vals.append(0)
                    colors.append("#cccccc")

            vals = np.array(vals)
            clip_height = non_outlier_max * 1.12
            display_vals = np.clip(vals, 0, clip_height)
            hatch = SPARSITY_HATCHES.get(sp, "")
            bars = ax.bar(
                x + offsets[sp_idx],
                display_vals,
                bar_width,
                color=colors,
                edgecolor="white",
                linewidth=0.5,
                hatch=hatch,
            )

            use_latex = plt.rcParams.get("text.usetex", False)

            def _bold(s):
                return rf"\textbf{{{s}}}" if use_latex else s

            for bar, v, c in zip(bars, vals, colors):
                if v <= 0:
                    continue
                raw_text = f"{v * 100:.1f}"
                is_outlier = v > outlier_thresh
                is_best = best_display is not None and raw_text == best_display
                if is_outlier:
                    bx = bar.get_x()
                    bw = bar.get_width()
                    top = bar.get_height()
                    band_h = y_cap * 0.015
                    ax.fill_between(
                        [bx, bx + bw],
                        top - band_h,
                        top + band_h,
                        color="white",
                        zorder=4,
                    )
                    cap_h = y_cap * 0.025
                    cap_bot = top + band_h + y_cap * 0.005
                    ax.bar(
                        bar.get_x() + bw / 2,
                        cap_h,
                        bw,
                        bottom=cap_bot,
                        color=c,
                        edgecolor="white",
                        linewidth=0.5,
                        hatch=hatch,
                        zorder=3,
                    )
                    ax.text(
                        bx + bw / 2,
                        cap_bot + cap_h,
                        raw_text,
                        ha="center",
                        va="bottom",
                        fontsize=font_size - 1.5,
                        rotation=60,
                    )
                else:
                    text = _bold(raw_text) if is_best else raw_text
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        text,
                        ha="center",
                        va="bottom",
                        fontsize=font_size - (1.5 if is_best else 2.5),
                        rotation=60,
                        fontweight="bold" if is_best else "normal",
                    )

        # Build per-unit info dicts and let visualize.make_label do the
        # heavy lifting — that keeps labels identical to convergence_curves
        # and test_artifacts. Sparsity is stripped from the info because
        # the bar chart shows sparsity via hatching, not via the tick.
        unit_infos = []
        for method, var_key, sweep_val in units:
            cond = (
                (sub["method_class"] == method)
                & (sub["variant"].fillna("__none__") == var_key)
            )
            if sweep_param and sweep_val is not None:
                cond &= sub[sweep_param] == sweep_val
            matching = sub[cond]
            if matching.empty:
                # No data for this unit at any sparsity — fall back to a
                # synthetic info just so the tick still gets a label.
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
            unit_infos.append(info)

        assign_label_visibility([(None, info) for info in unit_infos])
        tick_labels = [make_label(info) for info in unit_infos]
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
        legend_handles.append(
            Patch(
                facecolor="#aaaaaa",
                edgecolor="white",
                hatch=hatch,
                label=f"{int(sp)}{pct} sparsity",
            )
        )
    if legend_handles:
        fig.legend(handles=legend_handles, loc="upper right", framealpha=0.9)

    fig.suptitle(f"{dataset_name} — {metric}", y=1.02)
    fig.tight_layout()

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
        "--exclude_cnceleb_concatenated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Skip the cnceleb_concatenated protocol entirely and relabel "
            "cnceleb_multi as 'CNCeleb-E'. Use --no-exclude_cnceleb_concatenated "
            "to keep both CNCeleb protocols as side-by-side subplots."
        ),
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.input_dir, "figures")

    # Load data --> de-duplicate based on is_latest flag (keep only the latest)
    csv_path = os.path.join(args.input_dir, "eer_leaderboard.csv")
    df = pd.read_csv(csv_path)
    if args.exclude_cnceleb_concatenated:
        df = df[df["dataset"] != "cnceleb_concatenated"].copy()
    if "is_latest" in df.columns:
        stale = df[~df["is_latest"].astype(bool)]
        for _, row in stale.iterrows():
            print(
                f"  [warn] skipping older run: exp={row['exp']}, dataset={row['dataset']}, run_ts={row.get('run_ts', '?')}"
            )
        df = df[df["is_latest"].astype(bool)]

    dupes = df[df.duplicated(subset=["dataset", "exp"], keep="first")]
    for _, row in dupes.iterrows():
        print(
            f"  [warn] unexpected duplicate after is_latest filter: exp={row['exp']}, dataset={row['dataset']}"
        )
    df = df.drop_duplicates(subset=["dataset", "exp"], keep="first")

    # Parse experiment names to get method_class, sparsity, alpha, f, variant.
    # Use info_from_csv_row so fixed-lambda runs pick up _bregman_lambda from
    # config_tree.log when --base_dirs is provided. The scalar columns are
    # kept for downstream pandas filtering; the full info dict lives in the
    # "info" column and is the input to make_label.
    parsed = df["exp"].apply(lambda e: info_from_csv_row(e, args.base_dirs))
    df["info"] = parsed
    df["method_class"] = parsed.apply(lambda x: x["method_class"])
    df["sparsity"] = parsed.apply(lambda x: x["sparsity"])
    df["alpha"] = parsed.apply(lambda x: x.get("alpha"))
    df["f"] = parsed.apply(lambda x: x.get("f"))
    df["variant"] = parsed.apply(lambda x: x.get("variant"))

    # Parse dataset column into (dataset_name, protocol) — raises on unknown
    dp = df["dataset"].apply(parse_dataset_protocol)
    df["dataset_name"] = dp.apply(lambda x: x[0])
    df["protocol"] = dp.apply(lambda x: x[1])
    if args.exclude_cnceleb_concatenated:
        df.loc[df["dataset"] == "cnceleb_multi", "protocol"] = "CNCeleb-E"
    df["train_dataset"] = df["exp"].apply(parse_train_dataset_protocol)

    # Ensure all metric columns are numeric
    base_metrics = ["EER", "minDCF"]
    for base in base_metrics:
        for col in [base, f"{base}_raw", f"{base}_norm"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build list of (column_name, subdir_name) for each qualifier present
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

    # Generate one PDF per (train_dataset, dataset, metric_variant)
    for col, base_name, subdir in metric_variants:
        for train_ds, train_group in df.groupby("train_dataset"):
            for dataset_name, group in train_group.groupby("dataset_name"):
                # Skip if this group has no data for this column
                if group[col].notna().sum() == 0:
                    continue
                protocols = group["protocol"].unique()
                safe_name = dataset_name.lower().replace(" ", "_")
                out_path = os.path.join(
                    output_dir,
                    train_ds,
                    subdir,
                    f"{safe_name}_{base_name.lower()}.pdf",
                )
                plot_metric_for_dataset(
                    group,
                    dataset_name,
                    protocols,
                    col,
                    out_path,
                    font_size=args.font_size,
                )


if __name__ == "__main__":
    main()
