#!/usr/bin/env python3
"""One-off legend generator: same content as
``train_multiclassaccuracy_legend.pdf`` but with the AdamW/SGD baselines
removed, for LaTeX figures that share one legend across panels.

Selects on the method token only. Selecting on ``-sr<NN>`` would silently drop
the fixed-lambda runs, whose names carry ``-lam<value>`` instead.

Run from repo root:
    python scripts/_gen_custom_legend.py
"""

import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.vis.common import export_standalone_legend, setup_matplotlib  # noqa: E402
from src.vis.encoding import Encoding  # noqa: E402
from src.vis.runs import discover  # noqa: E402

BASE_DIRS = ["/data/aloradad/results/cnceleb"]
MODEL = "ecapa_tdnn"
DATASET = "cnceleb"
METHODS = ("adabreg", "linbreg", "adabreg_fixed", "linbreg_fixed")

OUT_PATH = (
    "results/cross_exp_comparison/convergence_curves/ecapa_tdnn/cnceleb/"
    "train_multiclassaccuracy_legend_nobaseline.pdf"
)


def main():
    patterns = [
        f"sv_bregman_*{m}-wespeaker*{MODEL}*{DATASET}*" for m in METHODS
    ]

    setup_matplotlib(font_size=16)
    groups = [g for g in discover(BASE_DIRS, patterns) if not g.is_dense]
    enc = Encoding(groups)

    handles, labels, seen = [], [], set()
    for g in groups:
        label = enc.label(g)
        if label in seen:
            continue
        seen.add(label)
        color, marker, ls = enc.style(g)
        (h,) = plt.plot(
            [0, 1],
            [0, 1],
            color=color,
            marker=marker,
            linestyle=ls,
            markersize=4,
            linewidth=1.3,
        )
        handles.append(h)
        labels.append(label)
    plt.close("all")

    ncol = min(5, len(labels))
    export_standalone_legend(handles, labels, OUT_PATH, ncol, font_size=16)


if __name__ == "__main__":
    main()
