#!/usr/bin/env python3
"""Plot per-layer trainable-scale evolution from a Bregman run.

A trainable-scales run (``sv/sv_bregman_*_trainable_scales``) writes
``trainable_scales_history.csv`` to its output dir. This renders one linear
scale-factor ``c`` line per layer over epochs, so the cross-layer allocation
drift is visible — unlike the scalar ``bregman/scale_min`` / ``scale_max`` the
tracker records.

Usage:
    python scripts/visualize_scale_evolution.py \\
        --run_dirs '/data/aloradad/results/cnceleb/sv_bregman_adabreg_trainable_scales*' \\
        --output results/scale_evolution
"""
import argparse
import glob
import logging
import os
import sys

import matplotlib.pyplot as plt

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPTS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.vis.common import setup_matplotlib  # noqa: E402
from src.vis.scale_evolution import find_history_csv  # noqa: E402
from src.vis.scale_evolution import read_scale_history, render_scale_evolution

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("scale_evolution")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--run_dirs",
        nargs="+",
        required=True,
        help="Run directories (globs ok); each is searched for the CSV.",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Output dir (default: write scale_evolution.pdf beside each CSV).",
    )
    args = ap.parse_args()

    setup_matplotlib(font_size=10)
    run_dirs = sorted({d for pat in args.run_dirs for d in glob.glob(pat)})
    if not run_dirs:
        log.error("No run dirs matched.")
        return 1
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    n_ok = 0
    for run_dir in run_dirs:
        csv_path = find_history_csv(run_dir)
        if csv_path is None:
            log.warning(f"{run_dir}: no {os.path.basename(run_dir)} CSV; skip")
            continue
        epochs, series = read_scale_history(csv_path)
        fig, ax = plt.subplots(figsize=(6.8, 4.0))
        render_scale_evolution(epochs, series, ax)
        ax.set_title(os.path.basename(run_dir.rstrip("/")), fontsize=9)
        fig.tight_layout()
        if args.output:
            name = os.path.basename(run_dir.rstrip("/")) or "run"
            out_path = os.path.join(args.output, f"{name}_scale_evolution.pdf")
        else:
            out_path = os.path.join(
                os.path.dirname(csv_path), "scale_evolution.pdf"
            )
        fig.savefig(out_path)
        plt.close(fig)
        log.info(f"{run_dir}: {len(series)} layers -> {out_path}")
        n_ok += 1

    if n_ok == 0:
        log.error("No run produced a plot (no history CSVs found).")
        return 1
    log.info(f"Done. {n_ok}/{len(run_dirs)} runs plotted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
