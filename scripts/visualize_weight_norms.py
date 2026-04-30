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

# Reuse shared utilities from visualize.py
sys.path.insert(0, os.path.dirname(__file__))
from visualize import discover_experiments, make_label, setup_matplotlib

matplotlib.use("pdf")

# Layer group display config: (csv_prefix, display_name, color)
LAYER_GROUPS = [
    ("conv_layers", "Conv Layers", "#050bb8"),
    ("linear_layers", "Linear Layers", "#ff7f0e"),
    ("classifier", "Final Classifier", "#5d361a"),
]

VISUALIZE_GAMMA = False
VISUALIZE_MEAN_SPARSITY = False
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
    ax.set_title("A. Weight $L_2$ Norms")
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
    ax.set_title("B. Layer Sparsity")
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


def plot_weight_norms_comparison(all_data, output_path):
    """Overlay L2 norms and sparsity across experiments (2-panel
    comparison)."""
    num_subplots = 2 if VISUALIZE_MEAN_SPARSITY else 1
    fig, axes = plt.subplots(1, num_subplots, figsize=(11, 4) if num_subplots > 1 else (6, 4))

    # Panel A: Total model L2 norm
    ax = axes[0] if num_subplots > 1 else axes
    for label, style, df in all_data:
        col = "model/total_l2_norm"
        if col not in df.columns:
            # Fallback: sum of group norms
            group_cols = [
                f"{p}/l2_norm"
                for p, _, _ in LAYER_GROUPS
                if f"{p}/l2_norm" in df.columns
            ]
            if not group_cols:
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
            markersize=3.5,
            markevery=max(1, len(df) // 10),
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$L_2$ Norm")
    ax.set_title("Total Model $L_2$ Norm")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(loc="best", fontsize=16)

    # Panel B: Classifier sparsity
    if VISUALIZE_MEAN_SPARSITY:
        ax = axes[1]
        for label, style, df in all_data:
            sparsity_cols = [
                f"{p}/sparsity"
                for p, _, _ in LAYER_GROUPS
                if f"{p}/sparsity" in df.columns
            ]
            if not sparsity_cols:
                continue
            vals = df[sparsity_cols].mean(axis=1)
            ax.plot(
                df["epoch"],
                vals,
                label=label,
                color=style[0],
                marker=style[1],
                markersize=3.5,
                markevery=max(1, len(df) // 10),
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean Layer Sparsity")
        ax.set_title("Mean Sparsity Across Layer Groups")
        ax.set_ylim(-0.05, 1.05)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend(loc="best", fontsize=12)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, format="pdf")
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
    args = parser.parse_args()

    setup_matplotlib(args.font_size)

    experiments = discover_experiments(args.base_dirs, args.experiments)
    if not experiments:
        print("No experiments matched the given patterns.")
        return

    print(f"Found {len(experiments)} experiments:")
    for _, info in experiments:
        print(f"  {info['dirname']}  ->  {make_label(info)}")

    # Import styling for comparison plot
    from visualize import get_style

    # Per-experiment plots + collect data for comparison
    comparison_data = []
    for exp_dir, info in experiments:
        df = load_weight_norms(exp_dir)
        if df is None:
            print(f"\n[skip] {info['dirname']}: no weight_norms.csv")
            continue

        print(f"\n--- {info['dirname']} ---")
        exp_label = make_label(info)
        out_dir = os.path.join(args.output, info["dirname"])
        plot_weight_norms(
            df,
            os.path.join(out_dir, "weight_norms.pdf"),
            title=exp_label,
        )

        color, marker, _ = get_style(info)
        comparison_data.append((exp_label, (color, marker), df))

    # Cross-experiment comparison
    if len(comparison_data) > 1:
        cmp_dir = os.path.join(args.output, "comparisons")
        plot_weight_norms_comparison(
            comparison_data,
            os.path.join(cmp_dir, "weight_norms_comparison.pdf"),
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
