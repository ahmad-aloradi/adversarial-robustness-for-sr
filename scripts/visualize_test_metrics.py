#!/usr/bin/env python3
"""Grouped bar charts of EER / minDCF, one PDF per (dataset, metric) pair.

Reads test_metrics.csv from scripts/aggregate_json_scores.py. A dataset with
several protocols draws them as side-by-side subplots on one y-axis.

Run it with::

    python scripts/visualize_test_metrics.py \\
        --input_dir results/test_eval/metrics/ecapa_tdnn
"""

import argparse
import fnmatch
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

from src.vis.common import setup_matplotlib  # noqa: E402
from src.vis.encoding import (  # noqa: E402
    METHOD_BY_KEY,
    SHOW_ALPHA,
    SHOW_f,
    Encoding,
    variant_for,
)
from src.vis.runs import (  # noqa: E402
    RunGroup,
    read_landed_sparsity,
    resolve_sparsity_level,
)

# Dense baselines carry no sparsity level, so they never join a per-rate group.
DENSE_METHOD_KEYS = frozenset(k for k, m in METHOD_BY_KEY.items() if m.family == "dense")

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


# ---------------------------------------------------------------------------
# Actual-sparsity resolution
#
# A bar tick shows the realized sparsity, not the target written in the name —
# an "sr90" run typically lands at 89.5–90.5%. It comes from results.json's
# best_checkpoint, so the tick, the legend and the sparsity curve all report the
# one pruned_sparsity (docs/image_benchmarks.md).
# ---------------------------------------------------------------------------


def resolve_actual_sparsity(exp_name, base_dirs, info):
    """Mean realized sparsity (0–1) over one experiment's seeds, or None.

    The ckpt filename's ``sr`` tag is whole-model sparsity, so it is not read
    here. A dense run states no level and returns None.
    """
    if not base_dirs or info.is_dense:
        return None
    for bd in base_dirs:
        exp_dir = os.path.join(bd, exp_name)
        if not os.path.isdir(exp_dir):
            continue
        seed_dirs = sorted(glob.glob(os.path.join(exp_dir, "seed_*"))) or [exp_dir]
        landed = [v for v in (read_landed_sparsity(d) for d in seed_dirs) if v is not None]
        return float(np.mean(landed)) if landed else None
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


def _method_key(method_class, is_fixed):
    """The bucket METHOD_ORDER and METHOD_GROUPS key on.

    The parser gives a fixed-lambda run the same ``method_class`` as its
    adaptive sibling, so this re-expands the two into separate buckets.
    """
    return f"{method_class}_fixed" if is_fixed else method_class


def _method_sort_key(method_class, is_fixed):
    key = _method_key(method_class, is_fixed)
    try:
        return METHOD_ORDER.index(key)
    except ValueError:
        return len(METHOD_ORDER)


def _protocol_sort_key(p):
    try:
        return PROTOCOL_ORDER.index(p)
    except ValueError:
        return len(PROTOCOL_ORDER)


def _arm_cond(sub, method, is_fixed, var_key):
    """Rows of one method arm. ``is_fixed`` parts a fixed-lambda run from its
    adaptive sibling, which shares the method class."""
    return (
        (sub["method_class"] == method)
        & (sub["is_fixed"] == is_fixed)
        & (sub["variant"].fillna("__none__") == var_key)
    )


def _build_units(sub):
    """Enumerate (method, is_fixed, variant, sweep_value, sparsity) bars.

    One unit is one bar. Figure sizing counts them before anything is drawn.
    """
    sweep_param = None
    for cand in ("initial_sparsity", "fixed_lambda", "alpha", "f"):
        if cand in sub.columns and sub[cand].dropna().nunique() >= 2:
            sweep_param = cand
            break
    units = []
    arms = (
        sub[["method_class", "is_fixed", "variant"]]
        .assign(variant=sub["variant"].fillna("__none__"))
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    arms = sorted(arms, key=lambda a: (_method_sort_key(a[0], a[1]), a[2]))
    for method, is_fixed, var_key in arms:
        vrows = sub[_arm_cond(sub, method, is_fixed, var_key)]
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
                    units.append((method, is_fixed, var_key, v, sp))
            else:
                units.append((method, is_fixed, var_key, None, sp))
    return units, sweep_param


def _unit_rows(sub, units, sweep_param):
    """The test-metric rows behind each unit, in unit order."""
    rows = []
    for method, is_fixed, var_key, sweep_val, sp in units:
        cond = _arm_cond(sub, method, is_fixed, var_key)
        cond &= sub["sparsity"].isna() if sp is None else sub["sparsity"] == sp
        if sweep_param and sweep_val is not None:
            cond &= sub[sweep_param] == sweep_val
        matched = sub[cond]
        if len(matched) > 1:
            print(f"  [warn] {len(matched)} rows share one bar; drawing the first: {sorted(matched['exp'])}")
        rows.append(matched)
    return rows


def _unit_infos(units, unit_rows, sweep_param):
    """One RunGroup per unit, plus the Encoding that labels them as a set.

    Sparsity is stripped because the per-bar tick already carries it, so the
    label contributes only the method and what varies within it.
    """
    infos = []
    for (method, is_fixed, var_key, sweep_val, _), rows in zip(units, unit_rows):
        if rows.empty:
            info = RunGroup(
                dirname=f"{method}-{var_key}",
                method=METHOD_BY_KEY[method],
                flavor=variant_for("fixed") if is_fixed else None,
                variant=None if var_key == "__none__" else var_key,
                alpha=None,
                f=None,
            )
            # SHOW_ALPHA/SHOW_f gate alpha/f display; the others always show.
            if sweep_param in ("initial_sparsity", "fixed_lambda"):
                setattr(info, sweep_param, sweep_val)
            elif sweep_param == "alpha" and SHOW_ALPHA:
                info.alpha = sweep_val
            elif sweep_param == "f" and SHOW_f:
                info.f = sweep_val
        else:
            info = rows["info"].iloc[0].without_sparsity()
        infos.append(info)
    return infos, Encoding(infos)


def _unit_x_positions(units, gap=GROUP_GAP):
    """X-coordinates for each unit, with ``gap`` between method arms."""
    n = len(units)
    if n == 0:
        return np.zeros(0, dtype=float)
    x = np.zeros(n, dtype=float)
    cur = 0.0
    prev_key = units[0][:3]
    for i in range(1, n):
        cur_key = units[i][:3]
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
    # Dense is listed first so its legend entry leads.
    sparsity_levels = sorted(df["sparsity"].dropna().unique().tolist())
    has_dense = df["method_class"].isin(DENSE_METHOD_KEYS).any()
    sparsity_buckets = ([None] if has_dense else []) + sparsity_levels

    # Fixed y-axis: always 0–20% with ticks at 0/5/10/15/20 so figures
    # are visually comparable across datasets/metrics. Work in percent
    # throughout so axis ticks line up with bar annotations.
    all_vals = df[metric].dropna().values * 100.0
    if len(all_vals) == 0:
        plt.close(fig)
        return
    y_cap = 20.0
    # Outlier bars (v > y_cap) get the broken-bar treatment: the primary
    # bar is drawn slightly past ylim and clipped at y_cap, then a white
    # break band + a small cap + the value annotation are rendered with
    # clip_on=False so they sit just above y_cap. That reads as "this bar
    # exceeds the chart" instead of silently topping out at y_cap.
    clip_height = y_cap * 1.05

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
        ax.yaxis.set_major_locator(FixedLocator([0.0, 5.0, 10.0, 15.0, 20.0]))

        # Show the protocol name above each subplot only when there are
        # multiple side-by-side protocols (e.g. Vox1-O / Vox1-E / Vox1-H).
        # Single-protocol figures (e.g. CNCeleb-E alone) don't need it.
        if n_protocols > 1:
            ax.set_title(protocol, fontsize=font_size + 6, pad=6)

        n_units = len(units)
        # One resolution of rows → infos per protocol, shared by the bar colors
        # and the labels below, so the two can never disagree about a unit.
        unit_rows = _unit_rows(sub, units, sweep_param)
        unit_infos, unit_enc = _unit_infos(units, unit_rows, sweep_param)

        bar_width = 0.7
        x = _unit_x_positions(units)
        vals = np.zeros(n_units)
        colors = ["#cccccc"] * n_units
        hatches = [""] * n_units
        actual_sps = [None] * n_units  # realized sparsity (0–1) or None
        for i, ((*_, sp), rows, info) in enumerate(
            zip(units, unit_rows, unit_infos)
        ):
            if not rows.empty:
                vals[i] = float(rows[metric].values[0]) * 100.0
                colors[i] = unit_enc.style(info)[0]
                if "actual_sparsity" in rows.columns:
                    asp = rows["actual_sparsity"].values[0]
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
                # Place the break right at y_cap and the "head" cap above
                # it, sticking out beyond the y-axis limit. clip_on=False
                # lets the cap + band + annotation render above ylim so
                # the bar visibly exceeds the chart range instead of
                # silently topping out at y_cap.
                top = y_cap
                band_h = y_cap * 0.020
                band = ax.fill_between(
                    [bx, bx + bw],
                    top - band_h,
                    top + band_h,
                    color="white",
                    zorder=4,
                )
                band.set_clip_on(False)
                cap_h = y_cap * 0.05
                cap_bot = top + band_h + y_cap * 0.005
                cap_bars = ax.bar(
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
                for cb in cap_bars:
                    cb.set_clip_on(False)
                ax.text(
                    bx + bw / 2,
                    cap_bot + cap_h,
                    raw_text,
                    ha="center",
                    va="bottom",
                    fontsize=font_size + 1,
                    rotation=60,
                    clip_on=False,
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
        # opts in via --sparsity_label actual. A fixed-lambda run always
        # shows the realized value: it aimed at no target. Blank for dense
        # baselines so they get just a single method label below.
        # Bottom tier: method/variant name written ONCE, centered under
        # each consecutive run of bars sharing it (e.g. AdaBreg's four
        # sparsity bars get one "AdaBreg" label spanning them). Saves
        # horizontal space and avoids repeating the method name.
        use_latex_x = plt.rcParams.get("text.usetex", False)
        pct_str_tick = r"\%" if use_latex_x else "%"
        bar_tick_labels = []
        for (_, is_fixed, _, _, sp), asp in zip(units, actual_sps):
            # A fixed-lambda run holds no target, so its realized value is the
            # only sparsity it has.
            want_actual = is_fixed or sparsity_label == "actual"
            if want_actual and asp is not None:
                bar_tick_labels.append(f"{asp * 100:.1f}{pct_str_tick}")
            elif sp is None:  # dense baseline — no sparsity to print
                bar_tick_labels.append("")
            else:
                bar_tick_labels.append(f"{int(sp)}{pct_str_tick}")

        group_labels = [unit_enc.label(info) for info in unit_infos]

        ax.set_xticks(x)
        ax.set_xticklabels(
            bar_tick_labels, rotation=0, ha="center", fontsize=font_size
        )
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
        # Slight tilt so longer method names ("Unst. Prun.", "Str. Prun.")
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
                _method_key(info.method.key, info.is_fixed_lambda),
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
            ax.set_ylabel(
                f"{metric_label} [{pct}]",
                fontsize=font_size + 6
                if len(protocols) == 1
                else font_size + 8,
            )

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

    # if 'vox' in dataset_name.lower():
    #     metric_clean = metric.replace("_raw", "").replace("_norm", "")
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
        help="Directory holding test_metrics.csv",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for figures (default: <input_dir>/figures)",
    )
    parser.add_argument("--font_size", type=int, default=16)
    parser.add_argument(
        "--base_dirs",
        nargs="+",
        default=None,
        help=(
            "Optional experiment root dir(s), used to resolve each run's "
            "realized sparsity. Without them, fixed-lambda runs have no "
            "sparsity level to be grouped at."
        ),
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help=(
            "Optional fnmatch glob patterns to keep only matching `exp` rows. "
            "Use the same patterns you pass to scripts/visualize.py so the bar "
            "charts match the convergence curves."
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
            "A fixed-lambda run always shows the realized value: it aimed "
            "at no target."
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

    csv_path = os.path.join(args.input_dir, "test_metrics.csv")
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

    # aggregate_json_scores.py writes one row per (exp, dataset). A second row
    # means the two disagree about one measurement, so say which.
    dupes = df[df.duplicated(subset=["dataset", "exp"], keep=False)]
    assert dupes.empty, f"one row per (exp, dataset) in {csv_path}, got {sorted(set(dupes['exp']))}"

    # The RunGroup lives in the "info" column and is what the Encoding labels.
    # The scalar columns beside it are what pandas groups and filters on.
    parsed = df["exp"].apply(RunGroup.from_name)
    df["info"] = parsed
    df["method_class"] = parsed.apply(lambda g: g.method.key)
    df["is_fixed"] = parsed.apply(lambda g: g.is_fixed_lambda)
    df["sparsity"] = parsed.apply(lambda g: g.sparsity)
    df["alpha"] = parsed.apply(lambda g: g.alpha)
    df["f"] = parsed.apply(lambda g: g.f)
    df["initial_sparsity"] = parsed.apply(lambda g: g.initial_sparsity)
    df["fixed_lambda"] = parsed.apply(lambda g: g.fixed_lambda)
    df["variant"] = parsed.apply(lambda g: g.variant)

    # Realized sparsity per run, the seed mean of the tested checkpoint's
    # pruned_sparsity. None without --base_dirs or without results.json; the
    # plot then drops back to the target sparsity for that bar.
    df["actual_sparsity"] = df.apply(
        lambda r: resolve_actual_sparsity(r["exp"], args.base_dirs, r["info"]),
        axis=1,
    )
    resolve_sparsity_level(df)

    # Parse dataset column into (dataset_name, protocol) — raises on unknown
    dp = df["dataset"].apply(parse_dataset_protocol)
    df["dataset_name"] = dp.apply(lambda x: x[0])
    df["protocol"] = dp.apply(lambda x: x[1])
    if args.exclude_cnceleb_concatenated:
        df.loc[df["dataset"] == "cnceleb_multi", "protocol"] = "CNCeleb-E"
    df["train_dataset"] = df["exp"].apply(parse_train_dataset_protocol)

    # Which score column every bar reads. src/vis/pruning_compare.py prefers
    # EER_raw, so the two figures do not read one column.
    base_metrics = ["EER", "minDCF"]
    SCORES = "norm"  # 'norm' or 'raw'
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
                    font_size=args.font_size + 3
                    if len(protocols) > 1
                    else args.font_size,
                    sparsity_label=args.sparsity_label,
                )


if __name__ == "__main__":
    main()
