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
import re
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
    load_csv_metrics,
    make_label,
    setup_matplotlib,
)

SHOW_ALPHA = False
SHOW_f = False

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

# Consistent method ordering for x-axis. Bregman methods (the proposed work)
# come first, followed by the sparse pruning benchmarks, then the dense
# baselines.
METHOD_ORDER = [
    "linbreg",
    "adabreg",
    "linbreg_fixed",
    "adabreg_fixed",
    "adabregw",
    "adabregl2",
    "pruning_struct",
    "pruning_unstruct",
    "vanilla",
    "wespeaker",
]

# Method class → benchmark group. Used to render a labeled bracket below the
# x-axis under the comparison baselines. "main" (Bregman methods) gets no
# bracket — those are the proposed methods, not benchmarks.
METHOD_GROUPS = {
    "linbreg_fixed": "sparse_bench",
    "adabreg_fixed": "sparse_bench",
    "linbreg": "main",
    "adabreg": "main",
    "adabregw": "main",
    "adabregl2": "main",
    "proxsgd": "main",
    "pruning_struct": "sparse_bench",
    "pruning_unstruct": "sparse_bench",
    "vanilla": "dense_bench",
    "wespeaker": "dense_bench",
}

GROUP_LABELS = {
    "sparse_bench": "Sparse Baselines",
    "dense_bench": "Dense Baselines",
}

PROTOCOL_ORDER = ["Vox1-O", "Vox1-E", "Vox1-H"]

SPARSITY_HATCHES = {75: "/", 90: "|", 95: "-", 99: "\\"}

# Hatch lines are drawn in this color (less-saturated black) by overlaying a
# transparent-face bar so the bar's solid dark outline isn't lightened too.
HATCH_COLOR = "#3F3E3E"

# Extra x-space inserted between consecutive (method, variant) groups so
# single-bar methods (e.g. AdamW, SGD baselines) don't visually merge with
# their neighbors when many sweep × sparsity bars sit next to them.
GROUP_GAP = 0.9


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


# ---------------------------------------------------------------------------
# Actual-sparsity resolution
#
# Bar ticks should show the realized sparsity rather than the target written
# in the experiment name (e.g. an "sr90" run typically lands at 89.5–90.5%).
# Two sources, picked per-experiment:
#   - test ckpt path embedded in train.log → parse "sr0.900" from filename
#   - csv/version_*/metrics.csv → last recorded bregman/sparsity (used for
#     fixed-lambda runs whose test was run from last.ckpt rather than the
#     monitor-best ckpt, which doesn't carry sparsity in its filename)
# ---------------------------------------------------------------------------

CKPT_SR_PATTERN = re.compile(r"-sr([\d.]+)\.ckpt")


def _sparsity_from_test_ckpt_log(exp_dir):
    """Find 'Test ckpt path:' in train.log and parse the sr fraction from it.

    Handles the multiple-ckpts-with-same-val case (the log records exactly
    which file was loaded for testing) without us having to replay the
    monitor's tiebreak.
    """
    log_path = os.path.join(exp_dir, "train.log")
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        for line in f:
            if "Test ckpt path:" not in line:
                continue
            m = CKPT_SR_PATTERN.search(line)
            if m:
                return float(m.group(1))
    return None


def _last_sparsity_from_csv(exp_dir):
    """Last non-null bregman sparsity recorded across CSVLogger versions."""
    df = load_csv_metrics(exp_dir)
    if df is None:
        return None
    for col in ("bregman/sparsity", "bregman/pruned_sparsity"):
        if col in df.columns:
            series = df[col].dropna()
            if len(series) > 0:
                return float(series.iloc[-1])
    return None


def resolve_actual_sparsity(exp_name, base_dirs, info, fixed_lambda_test_ckpt):
    """Return realized sparsity (0–1) for one row, or None if undeterminable.

    Fixed-lambda runs evaluated from last.ckpt: read the last logged
    bregman/sparsity from CSV, since last.ckpt's filename has no sr suffix.
    Everything else (including fixed-lambda runs tested from the best ckpt):
    parse the sr value out of the test ckpt path in train.log.
    """
    if not base_dirs or info.get("sparsity") is None:
        return None
    is_fixed = info.get("variant") == "fixed"
    for bd in base_dirs:
        exp_dir = os.path.join(bd, exp_name)
        if not os.path.isdir(exp_dir):
            continue
        if is_fixed and fixed_lambda_test_ckpt == "last":
            return _last_sparsity_from_csv(exp_dir)
        return _sparsity_from_test_ckpt_log(exp_dir)
    return None


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


def _method_key(method_class, variant):
    """Effective bucket for ordering and group lookup.

    The shared parser collapses both regular and fixed-lambda Bregman runs
    into the same `method_class` (e.g. 'adabreg') and tags fixed runs with
    `variant='fixed'`. We re-expand here so METHOD_ORDER / METHOD_GROUPS
    can position fixed-lambda runs separately (typically alongside the
    sparse pruning baselines instead of with the proposed Bregman group).
    """
    if variant == "fixed":
        return f"{method_class}_fixed"
    return method_class


def _method_sort_key(method_class, variant=None):
    key = _method_key(method_class, variant)
    try:
        return METHOD_ORDER.index(key)
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
    # Iterate (method, variant) pairs sorted by their effective method_key
    # so that fixed-lambda runs ('adabreg' + variant='fixed' →
    # 'adabreg_fixed') land at the position METHOD_ORDER assigns them
    # rather than next to the regular Bregman runs.
    pairs = (
        sub[["method_class", "variant"]]
        .assign(variant=sub["variant"].fillna("__none__"))
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    pairs = sorted(
        pairs,
        key=lambda mv: (
            _method_sort_key(
                mv[0], None if mv[1] == "__none__" else mv[1]
            ),
            mv[1],
        ),
    )
    for method, var_key in pairs:
        vrows = sub[
            (sub["method_class"] == method)
            & (sub["variant"].fillna("__none__") == var_key)
        ]
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
    fig_height=8.0,
    sparsity_label="target",
    fixed_lambda_test_ckpt="best",
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
    fig_width = max(8.0, 0.45 * total_units + 1.2 * n_protocols)
    fig, axes = plt.subplots(
        1,
        n_protocols,
        figsize=(fig_width, fig_height),
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
            f"{float(positive_vals.min()) * 100:.2f}"
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
            ax.set_title(protocol, fontsize=font_size + 6, pad=6)

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

        # Resolve value, color, hatch, and realized sparsity per unit
        bar_width = 0.7
        x = _unit_x_positions(units)
        vals = np.zeros(n_units)
        colors = ["#cccccc"] * n_units
        hatches = [""] * n_units
        actual_sps = [None] * n_units  # realized sparsity (0–1) or None
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
                if "actual_sparsity" in row.columns:
                    asp = row["actual_sparsity"].values[0]
                    if pd.notna(asp):
                        actual_sps[i] = float(asp)
            hatches[i] = "" if sp is None else SPARSITY_HATCHES.get(sp, "")

        display_vals = np.where(vals > y_cap, clip_height, vals)

        # Draw all primary bars in a single batch (dark outline, no hatch).
        # Hatches are layered separately on top via a transparent overlay so
        # their lines render in HATCH_COLOR rather than the bar's outline
        # color (matplotlib < 3.10 ties hatch color to edgecolor).
        bars = list(
            ax.bar(
                x,
                display_vals,
                bar_width,
                color=colors,
                edgecolor="#222222",
                linewidth=0.7,
            )
        )
        for bar, h in zip(bars, hatches):
            if not h:
                continue
            ax.bar(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                bar.get_width(),
                color="none",
                edgecolor=HATCH_COLOR,
                linewidth=0.0,
                hatch=h,
            )

        use_latex = plt.rcParams.get("text.usetex", False)

        def _bold(s):
            return rf"\textbf{{{s}}}" if use_latex else s

        for i, (bar, v, c, h) in enumerate(zip(bars, vals, colors, hatches)):
            if v <= 0:
                continue
            raw_text = f"{v:.2f}"
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
                    fontsize=font_size + 1,
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
                    fontsize=font_size + 1,
                    rotation=60,
                    fontweight="bold" if is_best else "normal",
                )

        # --- Two-tier x labels ---
        # Top tier (per-bar tick): the target sparsity ("90") by default,
        # or the realized sparsity to one decimal ("89.5") when the user
        # opts in via --sparsity_label actual. Fixed-lambda runs follow
        # the same toggle, with one exception: when they were tested from
        # last.ckpt (--fixed_lambda_test_ckpt last), we always show the
        # realized sparsity, since the run wasn't snapped to a monitor
        # checkpoint near the target and the realized value is what the
        # comparison actually measures. Blank for dense baselines so they
        # get just a single method label below.
        # Bottom tier: method/variant name written ONCE, centered under
        # each consecutive run of bars sharing it (e.g. AdaBreg's four
        # sparsity bars get one "AdaBreg" label spanning them). Saves
        # horizontal space and avoids repeating the method name.
        use_latex_x = plt.rcParams.get("text.usetex", False)
        pct_str_tick = r"\%" if use_latex_x else "%"
        bar_tick_labels = []
        for (_, var_key, _, sp), asp in zip(units, actual_sps):
            if sp is None:
                bar_tick_labels.append("")
                continue
            is_fixed = var_key == "fixed"
            force_actual = is_fixed and fixed_lambda_test_ckpt == "last"
            want_actual = force_actual or sparsity_label == "actual"
            if want_actual and asp is not None:
                bar_tick_labels.append(f"{asp * 100:.1f}{pct_str_tick}")
            else:
                bar_tick_labels.append(f"{int(sp)}{pct_str_tick}")

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
                    "alpha": sweep_val if sweep_param == "alpha" and SHOW_ALPHA else None,
                    "f": sweep_val if sweep_param == "f" and SHOW_f else None,
                }
            else:
                info = dict(matching["info"].iloc[0])
                info["sparsity"] = None
            unit_infos.append(info)

        assign_label_visibility([(None, info) for info in unit_infos])
        group_labels = [make_label(info) for info in unit_infos]

        ax.set_xticks(x)
        ax.set_xticklabels(bar_tick_labels, rotation=0, ha="center", fontsize=font_size)
        ax.set_ylim(0, y_cap)
        ax.tick_params(axis="x", which="both", length=0)
        ax.tick_params(axis="y", labelsize=font_size + 6)

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
        group_label_y = -0.07 if any_sparsity_tick else -0.05
        # Slight tilt so longer method names ("Unst. Prun.", "AdaBregW")
        # don't bump into each other when many narrow groups sit side by
        # side. Centered rotation keeps each label roughly under its run.
        for label, run_start, run_end in runs:
            ax.text(
                (x[run_start] + x[run_end]) / 2.0,
                group_label_y,
                _bold(label),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                rotation=15,
                rotation_mode="anchor",
                fontsize=font_size - 0.5,
            )

        # Vertical separator lines between method groups (Bregman →
        # sparse benchmarks → dense benchmarks). The label for each new
        # group is rendered vertically just to the right of its separator.
        unit_groups = [
            METHOD_GROUPS.get(
                _method_key(info.get("method_class"), info.get("variant")),
                "main",
            )
            for info in unit_infos
        ]
        for i in range(1, len(unit_groups)):
            if unit_groups[i] == unit_groups[i - 1]:
                continue
            sep_x = (x[i - 1] + x[i]) / 2.0
            ax.axvline(
                x=sep_x,
                color="gray",
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                zorder=0.5,
            )
            label = GROUP_LABELS.get(unit_groups[i])
            if label:
                ax.text(
                    sep_x + 0.08,
                    0.97,
                    label,
                    transform=ax.get_xaxis_transform(),
                    rotation=90,
                    fontsize=font_size,
                    va="top",
                    ha="left",
                    color="gray",
                    fontstyle="italic",
                )

        if ax_idx == 0:
            pct = r"\%" if plt.rcParams.get("text.usetex") else "%"
            metric_label = metric.replace("_raw", "").replace("_norm", "")
            ax.set_ylabel(f"{metric_label} [{pct}]", fontsize=font_size + 6 if len(protocols) == 1 else font_size + 8)

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
    parser.add_argument("--font_size", type=int, default=14)
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
        "--sparsity_label",
        choices=["target", "actual"],
        default="target",
        help=(
            "Per-bar tick label for sparse runs: 'target' (default) prints "
            "the integer from the experiment name (e.g. '90'); 'actual' "
            "prints the realized sparsity to one decimal (e.g. '89.5'). "
            "Fixed-lambda runs follow this toggle too — except when their "
            "test ran from last.ckpt (--fixed_lambda_test_ckpt last), in "
            "which case the realized sparsity is always shown."
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

    # Realized sparsity per run (best ckpt's sr, or last logged for
    # fixed-lambda + last.ckpt). Falls back to None if base_dirs not given
    # or the source files aren't present; the plot then drops back to the
    # target sparsity for that bar.
    df["actual_sparsity"] = df.apply(
        lambda r: resolve_actual_sparsity(
            r["exp"], args.base_dirs, r["info"], args.fixed_lambda_test_ckpt
        ),
        axis=1,
    )

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
    SCORES = 'norm' # 'norm' 'raw'
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
                    font_size=args.font_size + 3 if len(protocols) > 1 else args.font_size,
                    sparsity_label=args.sparsity_label,
                    fixed_lambda_test_ckpt=args.fixed_lambda_test_ckpt,
                )


if __name__ == "__main__":
    main()
