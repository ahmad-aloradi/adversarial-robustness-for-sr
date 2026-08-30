#!/usr/bin/env python3
"""Training curves and cross-seed accuracy summaries for one matched set of runs.

Each metric routes to the plots it supports: accuracy draws curves and a summary
bar, a loss draws curves only, and the verification metrics draw nothing here (see
``scripts/visualize_test_metrics.py``). The registry in :mod:`src.vis.metrics`
decides.

All seeds of one experiment are one :class:`~src.vis.runs.RunGroup`. A curve is
the seed mean with a ±1 std band; a summary bar is the seed mean with a ±1 std
error bar. Labels and styling come from :mod:`src.vis.encoding`, so this script
and every sibling renderer describe a run identically.

Run it with::

    python scripts/visualize.py \\
        --base_dirs logs/train/runs/cifar10/resnet18/augmentation \\
        --experiments "dense_sgd-*" "bregman_adabreg-sr90-*" \\
        --metrics valid/MulticlassAccuracy bregman/pruned_sparsity \\
        --source csv --output results/img/cifar10/resnet18
"""

import argparse
import os
import sys
from collections import OrderedDict, defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, MaxNLocator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.vis.common import export_standalone_legend, setup_matplotlib  # noqa: E402
from src.vis.encoding import TIER_LABELS, Encoding  # noqa: E402
from src.vis.metrics import metric_for, short_name  # noqa: E402
from src.vis.runs import discover  # noqa: E402

matplotlib.use("pdf")


def _auto_ylim(ax, margin=0.05):
    """Tighten the y-axis to the drawn data, with a small margin."""
    ys = [ln.get_ydata() for ln in ax.get_lines()]
    if not ys:
        return
    all_y = np.concatenate(ys)
    all_y = all_y[np.isfinite(all_y)]
    if len(all_y) == 0:
        return
    ymin, ymax = all_y.min(), all_y.max()
    pad = (ymax - ymin if ymax > ymin else 1.0) * margin
    ax.set_ylim(ymin - pad, ymax + pad)


def plot_curves(groups, enc, metrics, output_path, source, font_size=10, fig_width=5.5, fig_height=None, legend_mode="inline"):
    """Draw one subplot per metric, sharing the x-axis and one legend.

    ``legend_mode="split"`` omits the inline legend and writes it beside the
    figure, so two figures can share one legend in a LaTeX side-by-side layout.
    """
    setup_matplotlib(font_size)
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(fig_width, fig_height or 2.3 * n + 0.4), sharex=True, squeeze=False)
    axes = axes.flatten()

    for g in groups:
        if not g.frames(source):
            print(f"  [skip] no {source} data: {g.dirname}")
            continue
        color, marker, ls = enc.style(g)
        label = enc.label(g)
        for ax, metric in zip(axes, metrics):
            drawn = g.curve(metric, source)
            if drawn is None:
                continue
            x, mean, std = drawn
            ax.plot(x, mean, color=color, marker=marker, linestyle=ls, markersize=3.5, markevery=max(1, len(x) // 12), label=label)
            if std is not None:
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)

    for ax, metric in zip(axes, metrics):
        spec = metric_for(metric)
        ax.set_ylabel(spec.label)
        if spec.log_scale:
            ax.set_yscale("log")
        elif spec.ylim is not None:
            ax.set_ylim(*spec.ylim)
            ax.yaxis.set_major_locator(FixedLocator(list(spec.yticks)))
        else:
            _auto_ylim(ax)
    axes[-1].set_xlabel("Epoch" if source == "train_log" else "iteration [K]")
    if source == "train_log":
        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Every panel contributes: a run that logs only the metric on the second
    # panel draws there, and reading the first panel alone would leave it out.
    # Two runs can share a label where the field that parts them is hidden; the
    # first handle wins, so the legend shows the one drawn first.
    handles = OrderedDict()
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    if handles:
        ncol = min(3, len(handles))
        if legend_mode == "split":
            export_standalone_legend(list(handles.values()), list(handles), os.path.splitext(output_path)[0] + "_legend.pdf", ncol, font_size=font_size)
        else:
            fig.legend(list(handles.values()), list(handles), loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, 0.9), frameon=True, columnspacing=0.8, handletextpad=0.3)

    fig.align_ylabels(axes)
    fig.subplots_adjust(hspace=0.08)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_summary(groups, enc, metric, output_path, source, reduce="max", font_size=10, fig_width=5.5, fig_height=3.0):
    """Draw one bar per experiment, grouped by method tier, with a ±1 std error bar.

    Writes a companion ``<output>.csv`` holding the per-seed values behind every
    bar, so a number in the figure can be traced to the seeds that produced it.
    """
    setup_matplotlib(font_size)
    entries = [(g, *r) for g in groups if (r := g.scalar(metric, source, reduce)) is not None]
    if not entries:
        print(f"  [skip] no {source} data for '{metric}'")
        return

    # Entries arrive in sort_key order, which already orders each tier internally.
    tiers = OrderedDict()
    for e in entries:
        tiers.setdefault(e[0].tier, []).append(e)

    bar_width, intra_gap, group_gap = 0.6, 0.15, 1.5
    x_positions, bars, group_centers, pos = [], [], [], 0.0
    for tier in sorted(tiers):
        first = pos
        for e in tiers[tier]:
            x_positions.append(pos)
            bars.append(e)
            pos += bar_width + intra_gap
        group_centers.append(((first + pos - bar_width - intra_gap) / 2, TIER_LABELS[tier]))
        pos += group_gap * bar_width - intra_gap

    x_positions = np.array(x_positions)
    means = np.array([m for _, m, _, _ in bars])
    stds = np.array([s for _, _, s, _ in bars])
    tick_labels = [enc.label(g) for g, _, _, _ in bars]

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.bar(x_positions, means, yerr=stds, capsize=2.5, color=[enc.style(g)[0] for g, _, _, _ in bars], edgecolor="white", linewidth=0.5, width=bar_width, error_kw={"elinewidth": 0.8})
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_labels, rotation=25, ha="right")
    ax.set_ylabel(metric_for(metric).label)

    # Widen until the rotated tick labels stop overlapping; measure the real render.
    fig.set_size_inches(max(fig_width, len(bars) * 0.5), fig_height)
    rotation = 25
    for _ in range(6):
        fig.canvas.draw()
        boxes = sorted((lbl.get_window_extent() for lbl in ax.get_xticklabels()), key=lambda b: b.x0)
        if all(a.x1 <= b.x0 for a, b in zip(boxes, boxes[1:])):
            break
        width, height = fig.get_size_inches()
        if width < 20.0:
            fig.set_size_inches(min(width * 1.25, 20.0), height)
        elif rotation < 60:
            rotation = min(rotation + 10, 60)
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(rotation)
        else:
            break

    for xp, m, s, (_, _, _, vals) in zip(x_positions, means, stds, bars):
        txt = f"{m:.3f}" if len(vals) < 2 else f"{m:.3f}\n$\\pm${s:.3f}"
        ax.text(xp, m + s, txt, ha="center", va="bottom", fontsize=font_size - 2)

    # Put the tier labels below the rotated method labels; measure, do not guess.
    fig.canvas.draw()
    to_axes = ax.transAxes.inverted()
    labels_bottom = min([0.0] + [lbl.get_window_extent().transformed(to_axes).y0 for lbl in ax.get_xticklabels()])
    for cx, tier_label in group_centers:
        ax.text(cx, labels_bottom - 0.04, tier_label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontweight="bold", fontsize=font_size)

    # Zoom in when the accuracies cluster near the top.
    tops = means + stds
    vmin, vmax = means.min(), tops.max()
    if vmin > 0 and (vmax - vmin) / vmax < 0.3:
        ax.set_ylim(max(0, vmin - max((vmax - vmin) * 1.5, vmax * 0.02)), vmax + max((vmax - vmin) * 0.8, vmax * 0.02))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    print(f"Saved: {output_path}")

    csv_path = os.path.splitext(output_path)[0] + ".csv"
    pd.DataFrame(
        [
            {
                "experiment": g.dirname,
                "label": enc.label(g),
                "dataset": g.dataset,
                "model": g.model,
                "augmentation": g.augmentation,
                "sparsity": g.sparsity,
                "pruned_sparsity_mean": g.landed_sparsity,
                "pruned_sparsity_std": g.landed_sparsity_std,
                "n_seeds": len(vals),
                f"{metric}_mean": mean,
                f"{metric}_std": std,
                "seed_values": ";".join(f"{v:.6f}" for v in vals),
            }
            for g, mean, std, vals in entries
        ]
    ).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Publication-ready experiment visualization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--base_dirs", nargs="+", required=True, help="Root dir(s) containing experiment folders.")
    parser.add_argument("--experiments", nargs="+", required=True, help="Glob patterns for experiment directory names.")
    parser.add_argument("--metrics", nargs="+", default=["train_loss", "train/MulticlassAccuracy", "valid/MulticlassAccuracy", "pruning/sparsity", "bregman/pruned_sparsity"], help="Metrics to plot (column names).")
    parser.add_argument("--output", default="results/figures/", help="Output directory.")
    parser.add_argument("--font_size", type=int, default=16)
    parser.add_argument("--fig_width", type=float, default=5.5, help="Figure width in inches.")
    parser.add_argument("--fig_height", type=float, default=None, help="Figure height in inches (auto if omitted).")
    parser.add_argument("--source", choices=["train_log", "csv"], default="csv", help="Data source: epoch-level or step-level.")
    parser.add_argument("--summary-reduce", dest="summary_reduce", choices=["max", "last"], default="max", help="How each seed's summary bar reduces over epochs: max (best epoch, default) or last.")
    parser.add_argument("--legend-mode", dest="legend_mode", choices=["inline", "split"], default="inline", help="inline: embed the legend. split: write it beside the figure for a shared LaTeX legend.")
    args = parser.parse_args()

    out_dir = args.output
    if out_dir.endswith(".pdf"):  # older callers passed a file path
        out_dir = os.path.dirname(out_dir) or "figures"
    os.makedirs(out_dir, exist_ok=True)

    groups = discover(args.base_dirs, args.experiments)
    if not groups:
        print("No experiments matched the given patterns.")
        return
    enc = Encoding(groups)

    print(f"Found {len(groups)} experiments:")
    for g in groups:
        # Where the selected checkpoint landed. The label rounds this to a percent.
        landed = ""
        if g.landed_sparsity is not None:
            landed = f"   [pruned sparsity {100 * g.landed_sparsity:.2f}%"
            landed += f" +/- {100 * g.landed_sparsity_std:.2f}%]" if g.landed_sparsity_std is not None else "]"
        print(f"  {g.dirname}  ->  {enc.label(g)}{landed}")

    by_plot = defaultdict(list)
    for m in args.metrics:
        spec = metric_for(m)
        if not spec.plots:
            print(f"  [skip] {m}: this script draws no {m} plot; see scripts/visualize_test_metrics.py")
        for kind in spec.plots:
            by_plot[kind].append(m)

    # One metric per figure. A stacked panel per stage said nothing the single
    # figures do not, and it put two panels of different methods under one legend.
    curve_kw = dict(source=args.source, font_size=args.font_size, fig_width=args.fig_width, legend_mode=args.legend_mode)
    for m in by_plot["curves"]:
        plot_curves(groups, enc, [m], os.path.join(out_dir, f"{short_name(m)}.pdf"), fig_height=args.fig_height, **curve_kw)

    for m in by_plot["summary"]:
        plot_summary(groups, enc, m, os.path.join(out_dir, f"{short_name(m)}_summary.pdf"), source=args.source, reduce=args.summary_reduce, font_size=args.font_size, fig_width=args.fig_width)


if __name__ == "__main__":
    main()
