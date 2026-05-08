#!/usr/bin/env python3
"""One-off legend generator: same content as
``train_multiclassaccuracy_legend.pdf`` but with the AdamW/SGD baselines
removed and fixed-lambda runs labelled '(fixed)' instead of '(λ=...)'.

Run from repo root:
    python scripts/_gen_custom_legend.py
"""

import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize import (  # noqa: E402
    METHOD_DISPLAY_NAMES,
    discover_experiments,
    experiment_sort_key,
    export_standalone_legend,
    get_style,
    setup_matplotlib,
)


def make_label_fixed_token(info):
    """Like ``visualize.make_label`` but renders fixed-lambda runs as
    ``Method (fixed)`` instead of ``Method (λ=value)``."""
    name = METHOD_DISPLAY_NAMES.get(info["method_class"], info["method_class"])
    if info.get("variant") == "fixed":
        return f"{name} (fixed)"
    if info["sparsity"] is not None:
        pct = r"\%" if plt.rcParams.get("text.usetex") else "%"
        return f"{name} {info['sparsity']}{pct}"
    return name


def main():
    base_dirs = ["/data/aloradad/results/cnceleb"]
    patterns = [
        "sv_bregman_*adabreg-wespeaker*ecapa_tdnn*cnceleb*sr[7-9][0-9]*",
        "sv_bregman_*linbreg-wespeaker*ecapa_tdnn*cnceleb*sr[7-9][0-9]*",
        "sv_bregman_*breg_fixed-wespeaker*ecapa_tdnn*cnceleb*sr[7-9][0-9]*",
    ]

    setup_matplotlib(font_size=16)
    experiments = discover_experiments(base_dirs, patterns)
    # Drop baselines just in case any pattern accidentally caught them.
    experiments = [
        (d, info) for d, info in experiments
        if info["method_class"] not in ("vanilla", "wespeaker")
    ]
    experiments.sort(key=lambda item: experiment_sort_key(item[1]))

    handles, labels, seen = [], [], set()
    for _, info in experiments:
        label = make_label_fixed_token(info)
        if label in seen:
            continue
        seen.add(label)
        color, marker, ls = get_style(info)
        h, = plt.plot(
            [0, 1], [0, 1],
            color=color, marker=marker, linestyle=ls,
            markersize=4, linewidth=1.3,
        )
        handles.append(h)
        labels.append(label)
    plt.close("all")

    out_path = (
        "results/cross_exp_comparison/convergence_curves/ecapa_tdnn/cnceleb/"
        "train_multiclassaccuracy_legend_nobaseline.pdf"
    )
    ncol = min(5, len(labels))
    export_standalone_legend(handles, labels, out_path, ncol, font_size=16)


if __name__ == "__main__":
    main()
