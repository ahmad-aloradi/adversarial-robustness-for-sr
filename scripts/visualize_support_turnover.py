#!/usr/bin/env python3
"""Compare support turnover across Bregman runs.

Answers one question: how much of its active weight set does each method
replace per epoch? The rates come from ``bregman/support_births`` and
``bregman/support_deaths`` in every run's ``metrics.csv``; the derivation lives
in :mod:`src.vis.support_turnover`.

Output structure:
    {output}/support_turnover.pdf          both panels, the default figure
    {output}/support_turnover_legend.pdf   the one legend every panel shares
    {output}/panels/<panel>.pdf            one file per panel, no panel letter
    {output}/support_turnover.csv          every derived value, per run and seed

Usage:
    python scripts/visualize_support_turnover.py \\
        --base_dirs /data/aloradad/results/cifar100/resnet50 \\
        --experiments "bregman_linbreg*" \\
        --output results/support_turnover
"""

import argparse
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.vis.common import export_standalone_legend, setup_matplotlib  # noqa: E402
from src.vis.encoding import Encoding  # noqa: E402
from src.vis.runs import discover, load_csv_metrics  # noqa: E402
from src.vis.support_turnover import (  # noqa: E402
    BIRTHS_COL,
    DERIVED_COLUMNS,
    PANELS,
    mean_over_seeds,
    read_support_turnover,
)

matplotlib.use("pdf")

PANEL_WIDTH = 5.0
PANEL_HEIGHT = 3.6
PANEL_LETTERS = "ABCDEFGH"

CSV_COLUMNS = ["epoch", *DERIVED_COLUMNS]


def collect_series(groups, enc):
    """Load every seed of every discovered run, derive turnover, average the seeds.

    Returns the plot series as ``(label, style, data)`` and the per-seed CSV
    rows. A seed whose metrics hold no turnover column drops out, because only
    the Bregman methods log it.
    """
    series, rows = [], []
    for g in groups:
        per_seed, seeds = [], []
        for seed, seed_dir in zip(g.seeds, g.dirs):
            df = load_csv_metrics(seed_dir)
            if df is None or BIRTHS_COL not in df.columns:
                continue
            data = read_support_turnover(df)
            per_seed.append(data)
            seeds.append(seed)
            frame = pd.DataFrame({c: data[c] for c in CSV_COLUMNS})
            frame.insert(0, "seed", seed)
            frame.insert(0, "run", g.dirname)
            rows.append(frame)
        if not per_seed:
            print(f"  [skip] {g.dirname}: no support turnover logged")
            continue
        data = mean_over_seeds(per_seed)
        series.append((enc.label(g), enc.style(g), data))
        print(
            f"  {g.dirname}: {len(seeds)} seed(s) {seeds}, {len(data['epoch'])} epochs, "
            f"median {100 * pd.Series(data['tau']).median():.2f}% of the active weights replaced per epoch, "
            f"{data['cumulative_tau'][-1]:.1f}x the active count in total, "
            f"median size change {100 * pd.Series(data['nu']).median():+.3f}%/epoch"
        )
    return series, rows


def save_figure(fig, output_path, legend_source, legend_mode):
    """Write one PDF. Inline mode draws the legend on the figure itself."""
    if legend_mode == "inline":
        legend_source.legend(loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_combined(series, panels, output_path, legend_mode):
    """Draw the selected panels as one grid, two columns wide.

    Split mode writes the one legend here. Every panel file shares it, so no
    panel writes a second copy.
    """
    ncols = min(2, len(panels))
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(PANEL_WIDTH * ncols, PANEL_HEIGHT * nrows),
        squeeze=False,
    )
    flat = axes.ravel()
    for i, key in enumerate(panels):
        renderer, title, _ = PANELS[key]
        renderer(series, flat[i])
        flat[i].set_title(f"{PANEL_LETTERS[i]}. {title}")
    for ax in flat[len(panels) :]:
        ax.set_visible(False)
    if legend_mode == "split":
        handles, labels = flat[0].get_legend_handles_labels()
        export_standalone_legend(handles, labels, os.path.splitext(output_path)[0] + "_legend.pdf", ncol=min(4, len(labels)))
    save_figure(fig, output_path, flat[0], legend_mode)


def plot_single(series, key, out_dir, legend_mode):
    """Draw one panel to its own PDF, with no panel letter and no title."""
    renderer, _, filename = PANELS[key]
    fig, ax = plt.subplots(figsize=(PANEL_WIDTH, PANEL_HEIGHT))
    renderer(series, ax)
    save_figure(fig, os.path.join(out_dir, f"{filename}.pdf"), ax, legend_mode)


def write_csv(rows, output_path):
    """Write every derived per-epoch value, one block per run and seed.

    The ``run`` column holds the experiment directory name, so a reader can join
    the CSV back to the run it came from.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare support turnover across Bregman runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--base_dirs", nargs="+", required=True, help="Root dir(s) holding experiment folders.")
    parser.add_argument("--experiments", nargs="+", required=True, help="Glob patterns for experiment directory names.")
    parser.add_argument("--output", default="results/support_turnover/", help="Output root directory.")
    parser.add_argument("--panels", nargs="+", choices=list(PANELS), default=list(PANELS), help="Panels to draw, in this order.")
    parser.add_argument("--font_size", type=int, default=10)
    parser.add_argument(
        "--legend-mode",
        dest="legend_mode",
        choices=["inline", "split"],
        default="split",
        help="inline: draw the legend on each figure. split: write a separate _legend.pdf next to it.",
    )
    args = parser.parse_args()

    setup_matplotlib(args.font_size)

    groups = discover(args.base_dirs, args.experiments)
    if not groups:
        print("No experiments matched the given patterns.")
        return

    print(f"Found {len(groups)} experiments:")
    series, rows = collect_series(groups, Encoding(groups))
    if not series:
        print("No experiment logged support turnover.")
        return

    print()
    plot_combined(series, args.panels, os.path.join(args.output, "support_turnover.pdf"), args.legend_mode)
    for key in args.panels:
        plot_single(series, key, os.path.join(args.output, "panels"), args.legend_mode)
    write_csv(rows, os.path.join(args.output, "support_turnover.csv"))

    print("\nDone.")


if __name__ == "__main__":
    main()
