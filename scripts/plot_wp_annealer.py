"""Visualize how WpAnnealer controls the weights over training.

Renders a 4-panel figure that maps each WpAnnealer argument onto the w_p
schedule and shows what w_p means for the weights under the Bernoulli gate
(w_p = probability a weight keeps taking its exact reversible Bregman step,
else it is frozen at its current value).

    python scripts/plot_wp_annealer.py
    # writes docs/figures/wp_annealer_explained.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from src.callbacks.pruning.bregman.wp_scheduler import WpAnnealer

GREEN = "#2ca02c"
GRAY = "#bdbdbd"
OUT = Path("docs/figures/wp_annealer_explained.png")


def curve(ann: WpAnnealer, n: int = 400):
    prog = np.linspace(0.0, 1.0, n)
    return prog, np.array([ann.value_at(float(p)) for p in prog])


def panel_anatomy(ax):
    """Label every arg on a windowed example (start/end pulled inward so the
    held regions and breakpoints are visible)."""
    ann = WpAnnealer(
        w_p_init=1.0,
        w_p_final=0.0,
        start_fraction=0.25,
        end_fraction=0.85,
        schedule="cosine",
    )
    prog, wp = curve(ann)
    ax.plot(prog, wp, color="black", lw=2.5)
    ax.axvline(0.25, ls="--", color="gray", lw=1)
    ax.axvline(0.85, ls="--", color="gray", lw=1)

    ax.annotate(
        "w_p_init = 1.0\n(held before start)",
        xy=(0.10, 1.0),
        xytext=(0.02, 0.62),
        arrowprops=dict(arrowstyle="->"),
        fontsize=9,
    )
    ax.annotate(
        "start_fraction = 0.25\nanneal begins",
        xy=(0.25, 0.95),
        xytext=(0.30, 0.80),
        arrowprops=dict(arrowstyle="->"),
        fontsize=9,
    )
    ax.annotate(
        "schedule = 'cosine'\n(shape of descent)",
        xy=(0.55, ann.value_at(0.55)),
        xytext=(0.55, 0.70),
        arrowprops=dict(arrowstyle="->"),
        fontsize=9,
    )
    ax.annotate(
        "end_fraction = 0.85\nanneal complete",
        xy=(0.85, 0.05),
        xytext=(0.55, 0.22),
        arrowprops=dict(arrowstyle="->"),
        fontsize=9,
    )
    ax.annotate(
        "w_p_final = 0.0\n(held after end)",
        xy=(0.95, 0.0),
        xytext=(0.70, 0.08),
        arrowprops=dict(arrowstyle="->"),
        fontsize=9,
    )
    ax.set_title("(a) Anatomy of the schedule — what each arg does")
    ax.set_xlabel("training progress = global_step / total_steps")
    ax.set_ylabel("w_p")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.1)


def panel_your_config(ax):
    """Your config (start=0, end=1, cosine) + the Bernoulli reading."""
    ann = WpAnnealer(
        w_p_init=1.0,
        w_p_final=0.0,
        start_fraction=0.0,
        end_fraction=1.0,
        schedule="cosine",
    )
    prog, wp = curve(ann)
    ax.fill_between(
        prog,
        0,
        wp,
        color=GREEN,
        alpha=0.35,
        label="P(weight free / exploring) = w_p",
    )
    ax.fill_between(
        prog,
        wp,
        1,
        color=GRAY,
        alpha=0.55,
        label="P(weight latched / committed) = 1 - w_p",
    )
    ax.plot(prog, wp, color="black", lw=2.5)
    ax.set_title("(b) Your config: anneal across the entire run")
    ax.set_xlabel("training progress")
    ax.set_ylabel("w_p  =  fraction of support still exploring")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="center left", fontsize=8)


def panel_schedule_shape(ax):
    """linear vs cosine between the same endpoints."""
    for sched, color in (("linear", "#1f77b4"), ("cosine", "#d62728")):
        ann = WpAnnealer(schedule=sched)
        prog, wp = curve(ann)
        ax.plot(prog, wp, lw=2.5, color=color, label=sched)
    ax.set_title("(c) schedule: 'linear' (constant rate) vs 'cosine' (eased)")
    ax.set_xlabel("training progress")
    ax.set_ylabel("w_p")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.text(
        0.5,
        0.55,
        "cosine has zero slope at both ends:\nslow to start committing, slow to finish",
        fontsize=8,
        ha="center",
        color="#d62728",
    )


def panel_probabilistic_raster(ax):
    """Per-weight, per-step Bernoulli(w_p) gate (resampled every step)."""
    rng = np.random.default_rng(0)
    n_weights, n_steps = 1000, 220
    ann = WpAnnealer(schedule="cosine")
    prog = np.linspace(0.0, 1.0, n_steps)
    wp = np.array([ann.value_at(float(p)) for p in prog])
    gate = (rng.random((n_weights, n_steps)) < wp[None, :]).astype(int)

    ax.imshow(
        gate,
        aspect="auto",
        interpolation="nearest",
        extent=[0, 1, 0, n_weights],
        origin="lower",
        cmap=ListedColormap([GRAY, GREEN]),
    )
    # empirical exploring fraction per step tracks w_p
    ax.plot(prog, wp * n_weights, color="black", lw=2, label="w_p (target)")
    ax.set_title(
        "(d) Bernoulli gate: each cell = one weight at one step\n"
        "green = takes exact reversible step, gray = frozen"
    )
    ax.set_xlabel("training progress")
    ax.set_ylabel("weight index")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_weights)
    ax.legend(loc="upper right", fontsize=8)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    panel_anatomy(axes[0, 0])
    panel_your_config(axes[0, 1])
    panel_schedule_shape(axes[1, 0])
    panel_probabilistic_raster(axes[1, 1])
    fig.suptitle(
        "WpAnnealer: anneals w_p from w_p_init -> w_p_final over "
        "[start_fraction, end_fraction] of training",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
