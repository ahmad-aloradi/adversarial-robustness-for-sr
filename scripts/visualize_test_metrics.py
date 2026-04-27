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
import fnmatch
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

SPARSITY_HATCHES = {75: "//", 90: "*", 95: "oo", 99: "\\"}

# Extra x-space inserted between consecutive (method, variant) groups so
# single-bar methods (e.g. AdamW, SGD baselines) don't visually merge with
# their neighbors when many sweep × sparsity bars sit next to them.
GROUP_GAP = 0.6
# SPARSITY_HATCHES = {75: "///", 90: "\\\\\\", 95: "xxx", 99: "..."}


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


def filter_by_exp_patterns(df, patterns):
    """Keep rows whose `exp` matches at least one fnmatch glob pattern."""
    if not patterns:
        return df
    mask = np.zeros(len(df), dtype=bool)
    exps = df["exp"].astype(str).values
    for i, name in enumerate(exps):
        for pat in patterns:
            if fnmatch.fnmatch(name, pat):
                mask[i] = True
                break
    return df[mask].copy()


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


def _protocol_sort_key(p):
    try:
        return PROTOCOL_ORDER.index(p)
    except ValueError:
        return len(PROTOCOL_ORDER)


def _build_units(sub):
    """Enumerate (method, variant, sweep_value, sparsity) bars for a slice.

    Lifted out of plot_metric_for_dataset so figure sizing can scale with
    the actual bar count instead of just n_protocols.
    """
    sweep_param = None
    for cand in ("alpha", "f"):
        if cand in sub.columns and sub[cand].dropna().nunique() >= 2:
            sweep_param = cand
            break
    units = []
    for method in sorted(sub["method_class"].unique(), key=_method_sort_key):
        mrows = sub[sub["method_class"] == method]
        for var_key in sorted(
            mrows["variant"].fillna("__none__").unique().tolist()
        ):
            vrows = mrows[mrows["variant"].fillna("__none__") == var_key]
            spars_levels = sorted(vrows["sparsity"].dropna().unique().tolist())
            if vrows["sparsity"].isna().any():
                spars_levels = [None] + spars_levels
            for sp in spars_levels:
                srows = (
                    vrows[vrows["sparsity"].isna()]
                    if sp is None
                    else vrows[vrows["sparsity"] == sp]
                )
                if sweep_param and srows[sweep_param].notna().any():
                    for v in sorted(srows[sweep_param].dropna().unique()):
                        units.append((method, var_key, v, sp))
                else:
                    units.append((method, var_key, None, sp))
    return units, sweep_param


def _unit_x_positions(units, gap=GROUP_GAP):
    """X-coordinates for each unit with `gap` inserted at group boundaries."""
    n = len(units)
    if n == 0:
        return np.zeros(0, dtype=float)
    x = np.zeros(n, dtype=float)
    cur = 0.0
    prev_key = (units[0][0], units[0][1])
    for i in range(1, n):
        cur_key = (units[i][0], units[i][1])
        cur += 1.0 + (gap if cur_key != prev_key else 0.0)
        x[i] = cur
        prev_key = cur_key
    return x


def _effective_width(units, gap=GROUP_GAP):
    """Total x-extent (in bar slots) for a list of units, including gaps."""
    if not units:
        return 1.0
    x = _unit_x_positions(units, gap=gap)
    return float(x[-1] - x[0] + 1.0)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_metric_for_dataset(
    df,
    dataset_name,
    protocols,
    metric,
    output_path,
    font_size=16,
    fig_height=5.0,
):
    """Grouped bar chart for one dataset.

    One subplot per protocol.
    """
    setup_matplotlib(font_size)

    # Resolve the bar layout per protocol upfront. Subplot widths scale
    # with the bar count via gridspec width_ratios, and figure width
    # scales with total bars, so dense methods don't get cramped when a
    # neighboring protocol only has a few bars.
    protocol_data = []  # (protocol, sub, units, sweep_param)
    for protocol in sorted(protocols, key=_protocol_sort_key):
        sub = df[df["protocol"] == protocol].copy()
        units, sweep_param = _build_units(sub)
        protocol_data.append((protocol, sub, units, sweep_param))

    n_protocols = len(protocol_data)
    width_ratios = [_effective_width(u) for _, _, u, _ in protocol_data]
    total_units = sum(width_ratios)
    # ~0.45" per bar (gap-inflated so single-bar methods get more horizontal
    # room) plus per-protocol padding for axis labels and titles.
    fig_width = max(6.0, 0.45 * total_units + 1.2 * n_protocols)
    fig, axes = plt.subplots(
        1,
        n_protocols,
        figsize=(fig_width, fig_width // len(protocol_data)),
        sharey=True,
        squeeze=False,
        gridspec_kw={"width_ratios": width_ratios},
    )
    axes = axes[0]

    # Sparsity buckets present anywhere in the figure — drives the legend.
    # Dense (NaN) is listed first so its legend entry leads.
    sparsity_levels = sorted(df["sparsity"].dropna().unique().tolist())
    has_dense = df["sparsity"].isna().any()
    sparsity_buckets = ([None] if has_dense else []) + sparsity_levels

    # Determine y-axis limit: use IQR to detect outliers, cap y to show
    # the non-outlier range clearly while still reporting outlier values.
    # Work in percent throughout so axis ticks line up with bar annotations.
    all_vals = df[metric].dropna().values * 100.0
    if len(all_vals) == 0:
        plt.close(fig)
        return
    q1, q3 = np.percentile(all_vals, [25, 75])
    iqr = q3 - q1
    non_outliers = all_vals[all_vals <= q3 + 3.0 * iqr]
    non_outlier_max = (
        non_outliers.max() if len(non_outliers) > 0 else all_vals.max()
    )
    y_cap = non_outlier_max * 1.35  # headroom for annotations
    # Outlier bars get clipped near the top of the chart so the broken-bar
    # break + small cap + value annotation render right next to y_cap and
    # clearly read as "this bar exceeds the chart" rather than appearing
    # mid-figure. The outlier criterion is v > y_cap, so the indicator only
    # fires when the bar would actually exceed the visible range.
    clip_height = y_cap * 0.93

    for ax_idx, (protocol, sub, units, sweep_param) in enumerate(
        protocol_data
    ):
        ax = axes[ax_idx]

        # Best (lowest) score in this protocol for bold highlighting
        valid_vals = sub[metric].dropna()
        positive_vals = valid_vals[valid_vals > 0]
        best_display = (
            f"{float(positive_vals.min()) * 100:.1f}"
            if len(positive_vals) > 0
            else None
        )

        # Light horizontal gridlines behind bars improve readability.
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, alpha=0.3, linewidth=0.4)
        ax.xaxis.grid(False)

        # Show the protocol name above each subplot only when there are
        # multiple side-by-side protocols (e.g. Vox1-O / Vox1-E / Vox1-H).
        # Single-protocol figures (e.g. CNCeleb-E alone) don't need it.
        if n_protocols > 1:
            ax.set_title(protocol, fontsize=font_size + 1, pad=6)

        n_units = len(units)
        # Per-(method, variant) sorted sweep values, used for gradient ranking
        sweep_value_lookup = {}
        for method, var_key, _, _ in units:
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

        # Resolve value, color, and hatch per unit
        bar_width = 0.7
        x = _unit_x_positions(units)
        vals = np.zeros(n_units)
        colors = ["#cccccc"] * n_units
        hatches = [""] * n_units
        for i, (method, var_key, sweep_val, sp) in enumerate(units):
            sparsity_match = (
                sub["sparsity"].isna() if sp is None else (sub["sparsity"] == sp)
            )
            cond = (
                (sub["method_class"] == method)
                & sparsity_match
                & (sub["variant"].fillna("__none__") == var_key)
            )
            if sweep_param and sweep_val is not None:
                cond &= sub[sweep_param] == sweep_val
            row = sub[cond]
            if len(row) > 0:
                vals[i] = float(row[metric].values[0]) * 100.0
                actual_variant = var_key if var_key != "__none__" else None
                colors[i] = _gradient_color(
                    method,
                    actual_variant,
                    sweep_val,
                    sweep_value_lookup.get((method, var_key), []),
                )
            hatches[i] = "" if sp is None else SPARSITY_HATCHES.get(sp, "")

        display_vals = np.where(vals > y_cap, clip_height, vals)

        # Render bars in hatch-grouped batches (matplotlib bar() takes one
        # hatch per call, so units sharing a hatch are drawn together).
        bars = [None] * n_units
        hatch_groups = {}
        for i, h in enumerate(hatches):
            hatch_groups.setdefault(h, []).append(i)
        for h, idxs in hatch_groups.items():
            sub_bars = ax.bar(
                x[idxs],
                display_vals[idxs],
                bar_width,
                color=[colors[i] for i in idxs],
                edgecolor="#222222",
                linewidth=0.7,
                hatch=h,
            )
            for j, bar in zip(idxs, sub_bars):
                bars[j] = bar

        use_latex = plt.rcParams.get("text.usetex", False)

        def _bold(s):
            return rf"\textbf{{{s}}}" if use_latex else s

        for i, (bar, v, c, h) in enumerate(zip(bars, vals, colors, hatches)):
            if v <= 0:
                continue
            raw_text = f"{v:.1f}"
            is_outlier = v > y_cap
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
                    edgecolor="#222222",
                    linewidth=0.7,
                    hatch=h,
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
                    fontsize=font_size - 1.5,
                    rotation=60,
                    fontweight="bold" if is_best else "normal",
                )

        # --- Two-tier x labels ---
        # Top tier (per-bar tick): sparsity only, e.g. "75%". Blank for
        # dense baselines so they get just a single method label below.
        # Bottom tier: method/variant name written ONCE, centered under
        # each consecutive run of bars sharing it (e.g. AdaBreg's four
        # sparsity bars get one "AdaBreg" label spanning them). Saves
        # horizontal space and avoids repeating the method name.
        use_latex_x = plt.rcParams.get("text.usetex", False)
        pct_str_tick = r"\%" if use_latex_x else "%"
        bar_tick_labels = [
            f"{int(sp)}{pct_str_tick}" if sp is not None else ""
            for _, _, _, sp in units
        ]

        # Build per-unit info with sparsity stripped — the per-bar tick
        # already shows it, so make_label only contributes method/variant.
        unit_infos = []
        for method, var_key, sweep_val, sp in units:
            cond = (
                (sub["method_class"] == method)
                & (sub["variant"].fillna("__none__") == var_key)
            )
            if sp is None:
                cond &= sub["sparsity"].isna()
            else:
                cond &= sub["sparsity"] == sp
            if sweep_param and sweep_val is not None:
                cond &= sub[sweep_param] == sweep_val
            matching = sub[cond]
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
            unit_infos.append(info)

        assign_label_visibility([(None, info) for info in unit_infos])
        group_labels = [make_label(info) for info in unit_infos]

        ax.set_xticks(x)
        ax.set_xticklabels(bar_tick_labels, rotation=0, ha="center")
        ax.set_ylim(0, y_cap)
        ax.tick_params(axis="x", which="both", length=0)

        # Render the method/variant label once per consecutive run of
        # units that share it. Position is in axis-fraction y (just below
        # the per-bar ticks) and data-x (run midpoint).
        runs = []
        if group_labels:
            run_start = 0
            for i in range(1, len(group_labels)):
                if group_labels[i] != group_labels[i - 1]:
                    runs.append((group_labels[i - 1], run_start, i - 1))
                    run_start = i
            runs.append((group_labels[-1], run_start, len(group_labels) - 1))

        any_sparsity_tick = any(bar_tick_labels)
        # Push the group label further down when there are sparsity ticks
        # above it, so the two tiers don't visually crowd each other.
        group_label_y = -0.05 if any_sparsity_tick else -0.05
        for label, run_start, run_end in runs:
            ax.text(
                (x[run_start] + x[run_end]) / 2.0,
                group_label_y,
                _bold(label),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=font_size + 2,
            )

        if ax_idx == 0:
            pct = r"\%" if plt.rcParams.get("text.usetex") else "%"
            metric_label = metric.replace("_raw", "").replace("_norm", "")
            ax.set_ylabel(f"{metric_label} [{pct}]")

    # Legend: neutral gray patches distinguished by hatch pattern
    from matplotlib.patches import Patch

    pct = r"\%" if plt.rcParams.get("text.usetex") else "%"
    legend_handles = []
    for sp in sparsity_buckets:
        if sp is None:
            legend_handles.append(
                Patch(
                    facecolor="#aaaaaa",
                    edgecolor="#222222",
                    linewidth=0.7,
                    label="Dense",
                )
            )
        else:
            hatch = SPARSITY_HATCHES.get(sp, "")
            legend_handles.append(
                Patch(
                    facecolor="#aaaaaa",
                    edgecolor="#222222",
                    linewidth=0.7,
                    hatch=hatch,
                    label=f"{int(sp)}{pct}",
                )
            )
    # if legend_handles:
    #     fig.legend(handles=legend_handles, loc="upper right", framealpha=0.9, ncols=len(legend_handles)//2)

    metric_clean = metric.replace("_raw", "").replace("_norm", "")
    # if 'vox' in dataset_name.lower():
    #     fig.suptitle(f"{dataset_name} — {metric_clean}", y=1.02, fontweight="bold")
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
        "--experiments",
        nargs="+",
        default=None,
        help=(
            "Optional fnmatch glob patterns to keep only matching `exp` rows "
            "from the leaderboard CSV. Use the same patterns you pass to "
            "scripts/visualize.py so the bar charts match the convergence "
            "curves."
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

    # Ensure all metric columns are numeric. Norm-cohort variants are
    # intentionally ignored: the bar charts only plot the raw metric.
    base_metrics = ["EER", "minDCF"]
    for base in base_metrics:
        for col in [base, f"{base}_raw"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # One column per base metric: prefer "_raw" when present, otherwise the
    # unqualified column. This avoids two iterations writing the same PDF.
    metric_variants = []
    for base in base_metrics:
        raw_col = f"{base}_raw"
        if raw_col in df.columns and df[raw_col].notna().any():
            metric_variants.append((raw_col, base))
        elif base in df.columns and df[base].notna().any():
            metric_variants.append((base, base))

    # Generate one PDF per (train_dataset, dataset, metric)
    for col, base_name in metric_variants:
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
