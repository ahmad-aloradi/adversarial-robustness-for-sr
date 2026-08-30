"""Unit tests for the support turnover derivation.

Pure math, no torch. Every test builds explicit index sets, counts births,
deaths and support sizes directly, then feeds only the two rates the pruner
logs back into the derivation. Covers:
- A growing, a shrinking and a constant support recover B, D and K_t/K_{t-1}.
- The replacement rate reduces to the DST drop fraction when the size holds.
- A random support walk recovers every support size it came from.
- The reader raises on a missing column.
- Seeds average per epoch, and a lone seed carries no band.
"""
import numpy as np
import pandas as pd
import pytest

from src.vis.support_turnover import (
    BIRTHS_COL,
    DEATHS_COL,
    derive_turnover,
    mean_over_seeds,
    read_support_turnover,
)

N_COORDS = 1000


def _rates(prev, now):
    """The two numbers BregmanPruner logs: B / K_t and D / K_{t-1}."""
    return len(now - prev) / len(now), len(prev - now) / len(prev)


# =============================================================================
# 1. The rates recover the counts they came from
# =============================================================================

CASES = {
    "growth": (set(range(0, 400)), set(range(200, 900))),
    "shrink": (set(range(0, 900)), set(range(700, 1000))),
    "constant": (set(range(0, 500)), set(range(100, 600))),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_derive_recovers_counts(name):
    prev, now = CASES[name]
    b, d = _rates(prev, now)
    out = derive_turnover([b], [d])

    births, deaths = len(now - prev), len(prev - now)
    assert out["ratio"][0] == pytest.approx(len(now) / len(prev))
    assert out["nu"][0] == pytest.approx((len(now) - len(prev)) / len(prev))
    assert out["tau"][0] == pytest.approx(
        (births + deaths) / (len(now) + len(prev))
    )


def test_replacement_rate_equals_drop_fraction_at_constant_size():
    prev, now = CASES["constant"]
    assert len(prev) == len(now)
    b, d = _rates(prev, now)
    out = derive_turnover([b], [d])
    assert out["nu"][0] == pytest.approx(0.0)
    assert out["tau"][0] == pytest.approx(len(now - prev) / len(now))


# =============================================================================
# 2. A whole trajectory
# =============================================================================


def test_chained_ratios_track_a_random_walk():
    """Chained ratios recover the true support size of every epoch."""
    rng = np.random.default_rng(0)
    coords = np.arange(N_COORDS)
    supports = [set(rng.choice(coords, size=300, replace=False).tolist())]
    for size in (320, 340, 300, 260, 260, 410):
        supports.append(set(rng.choice(coords, size=size, replace=False).tolist()))

    pairs = list(zip(supports[:-1], supports[1:]))
    births = [_rates(p, n)[0] for p, n in pairs]
    deaths = [_rates(p, n)[1] for p, n in pairs]

    out = derive_turnover(births, deaths)
    sizes = len(supports[0]) * np.cumprod(out["ratio"])
    assert sizes == pytest.approx([len(s) for s in supports[1:]])
    assert out["cumulative_tau"][-1] == pytest.approx(sum(out["tau"]))


# =============================================================================
# 3. The reader
# =============================================================================


def _frame():
    """Two logged epochs on a support that holds its size."""
    return pd.DataFrame(
        {
            "epoch": [0.0, 1.0, 2.0],
            BIRTHS_COL: [np.nan, 0.10, 0.08],
            DEATHS_COL: [np.nan, 0.10, 0.08],
        }
    )


def test_reader_derives_from_the_logged_columns():
    out = read_support_turnover(_frame())
    assert out["epoch"] == pytest.approx([1.0, 2.0])
    assert out["tau"] == pytest.approx([0.10, 0.08])
    assert out["nu"] == pytest.approx([0.0, 0.0])
    assert out["cumulative_tau"] == pytest.approx([0.10, 0.18])


def test_reader_raises_on_a_missing_column():
    with pytest.raises(KeyError, match=DEATHS_COL):
        read_support_turnover(_frame().drop(columns=[DEATHS_COL]))


# =============================================================================
# 4. Averaging over the seeds of one run
# =============================================================================


def _seed(rates, epochs):
    """One seed's read_support_turnover result, on a support that holds its size."""
    frame = pd.DataFrame({"epoch": epochs, BIRTHS_COL: rates, DEATHS_COL: rates})
    return read_support_turnover(frame)


def test_seeds_average_per_epoch():
    out = mean_over_seeds([_seed([0.10], [1]), _seed([0.30], [1])])
    assert out["epoch"] == pytest.approx([1.0])
    assert out["tau"] == pytest.approx([0.20])
    assert out["tau_std"] == pytest.approx([np.std([0.10, 0.30], ddof=1)])


def test_one_seed_carries_no_band():
    out = mean_over_seeds([_seed([0.10, 0.20], [1, 2])])
    assert out["tau"] == pytest.approx([0.10, 0.20])
    assert out["tau_std"] is None


def test_a_short_seed_leaves_the_later_epochs_to_the_long_one():
    out = mean_over_seeds([_seed([0.10, 0.30], [1, 2]), _seed([0.30], [1])])
    assert out["epoch"] == pytest.approx([1.0, 2.0])
    assert out["tau"] == pytest.approx([0.20, 0.30])
    assert not np.isnan(out["tau_std"][0])
    assert np.isnan(out["tau_std"][1])  # one seed reached epoch 2, so no spread
