"""Unit tests for src/vis/scale_evolution.py (reader + render + CSV contract)."""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from src.vis.scale_evolution import SCALE_HISTORY_CSV  # noqa: E402
from src.vis.scale_evolution import (
    find_history_csv,
    read_scale_history,
    render_scale_evolution,
)


def _write_csv(path, layer_names, n_epochs):
    fieldnames = ["epoch", "step", *layer_names]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for e in range(n_epochs):
            row = {"epoch": e, "step": e * 10}
            for i, name in enumerate(layer_names):
                row[name] = float(1.0 + 0.1 * i + 0.01 * e)
            w.writerow(row)


def test_read_round_trip(tmp_path):
    names = ["g__a.weight", "g__b.weight", "g__c.weight"]
    path = tmp_path / SCALE_HISTORY_CSV
    _write_csv(path, names, n_epochs=5)

    epochs, series = read_scale_history(str(path))
    assert epochs == [0, 1, 2, 3, 4]
    assert set(series) == set(names)
    assert all(len(v) == 5 for v in series.values())
    # "step" is metadata, never a layer series.
    assert "step" not in series


def test_find_history_csv_recursive(tmp_path):
    nested = tmp_path / "version_0"
    nested.mkdir()
    path = nested / SCALE_HISTORY_CSV
    _write_csv(path, ["g__a.weight"], n_epochs=2)
    assert find_history_csv(str(tmp_path)) == str(path)
    assert find_history_csv(str(tmp_path / "missing")) is None


def test_read_empty_fails_loud(tmp_path):
    path = tmp_path / SCALE_HISTORY_CSV
    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=["epoch", "step"]).writeheader()
    with pytest.raises(AssertionError):
        read_scale_history(str(path))


def test_render_draws_one_line_per_layer():
    names = [f"g__layer{i}.conv.weight" for i in range(6)]
    epochs = list(range(8))
    series = {n: [1.0 + 0.05 * e for e in epochs] for n in names}

    fig, ax = plt.subplots()
    render_scale_evolution(epochs, series, ax)
    # one Line2D per layer + the neutral c=1 axhline.
    assert len(ax.get_lines()) == len(names) + 1
    plt.close(fig)
