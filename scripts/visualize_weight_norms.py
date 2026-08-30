#!/usr/bin/env python3
"""Per-experiment visualization of weight norm evolution across training.

Generates a 3-panel figure per experiment showing L2 norms, sparsity, and
BatchNorm gamma statistics by layer group. Follows the same per-experiment
output structure as visualize_test_artifacts.py.

Output structure:
    {output}/{experiment_dirname}/weight_norms.pdf

Usage:
    python scripts/visualize_weight_norms.py \\
        --base_dirs /dataHDD/ahmad/comfort26_sem/cnceleb /dataHDD/ahmad/21_03_2026/cnceleb \\
        --experiments "sv_bregman_*-sr90" "sv_bregman_*-sr95" \\
        --output results/weight_norms_vis/
"""

import argparse
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.vis.common import (  # noqa: E402
    MODEL_REGISTRY,
    export_standalone_legend,
    setup_matplotlib,
    ylim_for_rate,
)
from src.vis.encoding import Encoding  # noqa: E402
from src.vis.runs import discover  # noqa: E402

matplotlib.use("pdf")

# Layer group display config: (csv_prefix, display_name, color)
LAYER_GROUPS = [
    ("conv_layers", "Conv. Layers", "#050bb8"),
    ("linear_layers", "Linear Layers", "#ff7f0e"),
    ("classifier", "Classifier", "#5d361a"),
]

VISUALIZE_GAMMA = False
BN_GAMMA_COLOR = "#9467bd"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_weight_norms(exp_dir):
    """Load weight_norms.csv from an experiment directory.

    Returns DataFrame or None.
    """
    path = os.path.join(exp_dir, "weight_norms.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    if "epoch" not in df.columns:
        return None
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_weight_norms(df, output_path, title=None):
    """Generate a 2- or 3-panel figure: L2 norms, sparsity, and optionally BN gamma."""
    num_panels = 3 if VISUALIZE_GAMMA else 2
    fig_width = 6 * num_panels + 2
    fig, axes = plt.subplots(1, num_panels, figsize=(fig_width, 4))
    epochs = df["epoch"]
    int_locator = mticker.MaxNLocator(integer=True)

    # --- Panel A: L2 Norms ---
    ax = axes[0]
    for prefix, name, color in LAYER_GROUPS:
        col = f"{prefix}/l2_norm"
        if col in df.columns:
            ax.plot(epochs, df[col], label=name, color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$L_2$ Norm")
    # ax.set_title("A. Weight $L_2$ Norms")
    ax.xaxis.set_major_locator(int_locator)
    ax.legend(loc="best")

    # --- Panel B: Sparsity ---
    ax = axes[1]
    for prefix, name, color in LAYER_GROUPS:
        col = f"{prefix}/sparsity"
        if col in df.columns:
            ax.plot(epochs, df[col], label=name, color=color, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Sparsity")
    ax.set_ylim(-0.05, 1.05)
    # ax.set_title("B. Layer Sparsity")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(loc="best")

    # --- Panel C: BN Gamma ---
    if VISUALIZE_GAMMA:
        ax = axes[2]
        has_bn = all(
            c in df.columns
            for c in ["bn_gamma/geo_mean", "bn_gamma/min", "bn_gamma/max"]
        )
        if has_bn:
            ax.plot(
                epochs,
                df["bn_gamma/geo_mean"],
                label="Geometric Mean",
                color=BN_GAMMA_COLOR,
            )
            ax.fill_between(
                epochs,
                df["bn_gamma/min"],
                df["bn_gamma/max"],
                color=BN_GAMMA_COLOR,
                alpha=0.2,
                label="Min–Max Range",
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(r"$\gamma$ Value")
        ax.set_title(r"C. BatchNorm $\gamma$ Distribution")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend(loc="best")

    if title:
        fig.suptitle(title, y=1.02)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_weight_norms_comparison(all_data, output_path, legend_mode="inline"):
    """Overlay total L2 norm across experiments (single panel)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, style, df, _ in all_data:
        col = "model/total_l2_norm"
        if col not in df.columns:
            group_cols = [
                f"{p}/l2_norm"
                for p, _, _ in LAYER_GROUPS
                if f"{p}/l2_norm" in df.columns
            ]
            if not group_cols:
                print(f"WARNING: df.columns does not have {col}")
                continue
            vals = np.sqrt((df[group_cols] ** 2).sum(axis=1))
        else:
            vals = df[col]
        ax.plot(
            df["epoch"],
            vals,
            label=label,
            color=style[0],
            marker=style[1],
            linestyle=style[2],
            markersize=3.5,
            markevery=max(1, len(df) // 10),
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\|\theta\|_2$")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    handles, labels = ax.get_legend_handles_labels()
    if legend_mode == "inline":
        ax.legend(loc="upper center", fontsize=16, ncols=6)
    elif legend_mode == "split":
        if handles:
            legend_path = os.path.splitext(output_path)[0] + "_legend.pdf"
            export_standalone_legend(
                handles, labels, legend_path, ncol=6#min(6, len(labels))
            )
    else:
        raise ValueError(
            f"legend_mode must be 'inline' or 'split', got {legend_mode!r}"
        )
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_sparsity_comparison(all_data, output_path, legend_mode="inline"):
    """Per-layer-group sparsity across experiments, one panel per group.

    A dense run draws nothing: its curve is flat at zero. The shared y-axis
    comes from :func:`~src.vis.common.ylim_for_rate` at the lowest target in
    the set, so every panel reads on one scale.
    """
    FSIZE = 16
    in_percent = True

    sparse_data = [item for item in all_data if item[3].sparsity is not None]
    if not sparse_data:
        print(f"  [skip] {output_path}: no sparse experiments to compare")
        return
    targets = [item[3].sparsity for item in sparse_data]
    ylim_lo, ylim_hi = ylim_for_rate(min(targets), scale="fraction")

    n_groups = len(LAYER_GROUPS)
    fig, axes = plt.subplots(
        1, n_groups, figsize=(4 * n_groups, 4), sharey=True
    )

    for i, (ax, (prefix, group_name, _)) in enumerate(zip(axes, LAYER_GROUPS)):
        sparsity_col = f"{prefix}/sparsity"
        for label, style, df, _ in sparse_data:
            if sparsity_col not in df.columns:
                continue
            ax.plot(
                df["epoch"],
                df[sparsity_col] * 100 if in_percent else df[sparsity_col],
                label=label,
                color=style[0],
                marker=style[1],
                linestyle=style[2],
                markersize=3.5,
                markevery=max(1, len(df) // 10),
            )
        ax.set_xlabel("Epoch", fontsize=FSIZE)
        if i == 0:
            prct_str = r"$\mathsf{s}(\theta)$ $[\%]$" if in_percent else r"$\mathsf{s}(\theta)$"
            ax.set_ylabel(prct_str, fontsize=FSIZE)
        ax.set_title(group_name, fontsize=FSIZE)
        if in_percent:
            ax.set_ylim(ylim_lo * 100, ylim_hi * 100)
        else:
            ax.set_ylim(ylim_lo, ylim_hi)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.tick_params(axis="both", labelsize=FSIZE - 2)
        if legend_mode == "inline":
            ax.legend(loc="best", fontsize=FSIZE - 2, ncols=1)

    if legend_mode == "split":
        handles, labels = [], []
        seen = set()
        for ax in axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in seen:
                    seen.add(l)
                    handles.append(h)
                    labels.append(l)
        if handles:
            legend_path = os.path.splitext(output_path)[0] + "_legend.pdf"
            export_standalone_legend(
                handles, labels, legend_path, ncol=6#min(6, len(labels))
            )
    elif legend_mode != "inline":
        raise ValueError(
            f"legend_mode must be 'inline' or 'split', got {legend_mode!r}"
        )
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Visualize per-layer weight norm evolution across training.",
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
        "--output",
        default="results/weight_norms_vis/",
        help="Output root directory.",
    )
    parser.add_argument("--font_size", type=int, default=16)
    parser.add_argument(
        "--legend-mode",
        dest="legend_mode",
        choices=["inline", "split"],
        default="split",
        help=(
            "inline: draw legend on the comparison figures. "
            "split: omit inline legend and write a separate _legend.pdf "
            "next to each comparison figure."
        ),
    )
    args = parser.parse_args()

    setup_matplotlib(args.font_size)

    groups = discover(args.base_dirs, args.experiments)
    if not groups:
        print("No experiments matched the given patterns.")
        return

    distinct_models = {g.model for g in groups}
    cross_model = len(distinct_models) > 1
    if cross_model:
        print(f"\nCross-model mode: {sorted(distinct_models)}")
    # In cross-model mode the marker and the dash encode the backbone, so the
    # sweep may not also claim them. An alpha/f difference across backbones is a
    # config artifact, not a deliberate sweep, and the ramp would shade the
    # methods inconsistently: AdaBreg's already-dark green clamps to near-black
    # while LinBreg's brighter blue stays readable.
    enc = Encoding(groups, sweep=not cross_model)

    print(f"Found {len(groups)} experiments:")
    for g in groups:
        print(f"  {g.dirname}  ->  {enc.label(g)}")

    # Per-experiment plots + collect data for comparison
    comparison_data = []
    for g in groups:
        df = load_weight_norms(g.dirs[0])
        if df is None:
            print(f"\n[skip] {g.dirname}: no weight_norms.csv")
            continue

        print(f"\n--- {g.dirname} ---")
        exp_label = enc.label(g)
        out_dir = os.path.join(args.output, g.dirname)
        plot_weight_norms(
            df,
            os.path.join(out_dir, "weight_norms.pdf"),
            title=exp_label,
        )

        color, marker, ls = enc.style(g)
        if cross_model:
            entry = MODEL_REGISTRY.get(g.model, {})
            marker = entry.get("marker", marker)
            ls = entry.get("linestyle", ls)
            exp_label = f"{exp_label} [{entry.get('display_name', g.model)}]"
        comparison_data.append((exp_label, (color, marker, ls), df, g))

    # Cross-experiment comparisons (separate figures)
    if len(comparison_data) > 1:
        cmp_dir = os.path.join(args.output, "comparisons")
        plot_weight_norms_comparison(
            comparison_data,
            os.path.join(cmp_dir, "weight_norms_comparison.pdf"),
            legend_mode=args.legend_mode,
        )
        plot_sparsity_comparison(
            comparison_data,
            os.path.join(cmp_dir, "sparsity_comparison.pdf"),
            legend_mode=args.legend_mode,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
