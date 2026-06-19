"""Plot the evolution of trainable per-layer Bregman scale factors ``c``.

In trainable-scales mode the ``BregmanPruner`` writes one row per epoch to
``trainable_scales_history.csv`` in the run directory: columns ``epoch``,
``step``, then one column per regularized layer holding that layer's linear
scale factor ``c`` (neutral 1, floors at 0). The logger only carries the scalar
min/max of those scales; this module reads the full CSV and draws one line per
layer, so the cross-layer *allocation drift* (which layers gain pressure, which
shed it) is visible over training rather than collapsed to two extremes.

The producer (pruner) and this consumer share only the filename and column
schema below — no import coupling, so training never pulls matplotlib.

    python src/vis/scale_evolution.py   # smoke test on synthetic data
"""

import csv
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Contract with src/callbacks/pruning/bregman/bregman_pruner.py.
SCALE_HISTORY_CSV = "trainable_scales_history.csv"
_META_COLS = ("epoch", "step")


def find_history_csv(run_dir: str) -> Optional[str]:
    """Return the scale-history CSV under ``run_dir`` (recursive), or None."""
    direct = os.path.join(run_dir, SCALE_HISTORY_CSV)
    if os.path.isfile(direct):
        return direct
    for root, _dirs, files in os.walk(run_dir):
        if SCALE_HISTORY_CSV in files:
            return os.path.join(root, SCALE_HISTORY_CSV)
    return None


def read_scale_history(
    csv_path: str,
) -> Tuple[List[int], Dict[str, List[float]]]:
    """Read a history CSV into ``(epochs, {layer: [c per epoch]})``.

    Fails loud on a missing/empty/column-less file: a silent empty plot would
    hide a run that never logged scales (e.g. trainable scales were off).
    """
    assert os.path.isfile(csv_path), f"no scale history at {csv_path}"
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows, f"empty scale history at {csv_path}"
    layer_names = [c for c in rows[0] if c not in _META_COLS]
    assert layer_names, f"no layer columns in {csv_path}"
    epochs = [int(float(r["epoch"])) for r in rows]
    series = {n: [float(r[n]) for r in rows] for n in layer_names}
    return epochs, series


def _short(name: str, max_len: int = 30) -> str:
    """Drop the ``<group>__`` prefix and the encoder path, keep the layer tail."""
    name = re.sub(r"^[^.]*__", "", name)  # trainable-group prefix
    name = re.sub(r"^audio_encoder\.encoder\.\d*\.?", "", name)
    name = re.sub(r"\.weight$", "", name)
    return name if len(name) <= max_len else "…" + name[-(max_len - 1) :]


def render_scale_evolution(
    epochs: List[int],
    series: Dict[str, List[float]],
    ax: "plt.Axes",
    *,
    annotate_k: int = 4,
) -> None:
    """Draw one scale-factor ``c`` line per layer, colored by its final value.

    Args:
        epochs: x-axis epoch indices (length E).
        series: ``{layer_name: [c over E epochs]}``.
        ax: axes to draw on.
        annotate_k: label the k highest- and k lowest-final-scale layers at the
            right edge; a 30-line legend is unreadable.
    """
    assert series, "no layers to plot"
    finals = {name: vals[-1] for name, vals in series.items()}
    cmap = plt.get_cmap("viridis_r")
    norm = plt.Normalize(min(finals.values()), max(finals.values()))

    for name, vals in series.items():
        ax.plot(
            epochs, vals, lw=0.9, alpha=0.8, color=cmap(norm(finals[name]))
        )

    ax.axhline(1.0, color="0.6", lw=0.6, ls=":", zorder=0)  # neutral c=1

    # Label only the extreme allocators so the panel stays readable.
    ordered = sorted(finals.items(), key=lambda kv: kv[1])
    picked = {n for n, _ in ordered[:annotate_k] + ordered[-annotate_k:]}
    for name in picked:
        ax.annotate(
            _short(name),
            xy=(epochs[-1], series[name][-1]),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            fontsize=6,
            color=cmap(norm(finals[name])),
        )

    ax.set_xlabel("Epoch")
    usetex = plt.rcParams.get("text.usetex", False)
    ax.set_ylabel(r"Per-layer scale $c_g$" if usetex else "Per-layer scale c_g")
    ax.margins(x=0.12)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = ax.get_figure().colorbar(sm, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label("Final scale (high = more pruning pressure)", fontsize=8)


if __name__ == "__main__":
    # Smoke: synthetic climb / settle-at-floor / neutral layers.
    rng = np.random.default_rng(0)
    n_epochs = 20
    epochs = list(range(n_epochs))
    synth: Dict[str, List[float]] = {}
    for i in range(6):
        drift = (i - 2.5) * 0.04  # some climb, some shed toward the floor
        c = 1.0 + np.cumsum(rng.normal(drift, 0.03, n_epochs))
        synth[f"trainable_prunable__layer{i}.conv.weight"] = list(
            np.clip(c, 0.0, None)
        )

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    render_scale_evolution(epochs, synth, ax)
    fig.tight_layout()
    out = "/tmp/scale_evolution_smoke.pdf"
    fig.savefig(out)
    print(f"wrote {out} with {len(synth)} layer lines")
