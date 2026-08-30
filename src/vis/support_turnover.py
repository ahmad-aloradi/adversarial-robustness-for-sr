"""Reader and renderers for support turnover in Bregman runs.

Idea: the pruner logs two rates with different denominators — births over the
current support, deaths over the previous one. Counting the intersection from
both epochs gives ``K_t / K_{t-1} = (1 - d) / (1 - b)``. Every value here
follows from that identity, so no run needs a re-run and no metric needs a new
log.

Write ``m^k`` for the binary mask after epoch k. ``tau`` is
``||m^k - m^(k-1)||_1 / (||m^k||_1 + ||m^(k-1)||_1)``, and ``nu`` is the
relative change of ``||m^k||_1``.

**``tau`` only measures exploration while the mask size holds!** Read ``nu`` in
the CSV next to it. The figure draws ``tau`` alone.

The numerator is the flip rate of the N:M sparse-training papers. The
denominator is RigL's ``(1 - s_l) N_l``, which keeps a 90% run and a 99% run on
one axis. ``tau`` is therefore what RigL, SET and GraNet call the drop
fraction. Those methods set it per mask update, not per epoch. Multiply their
value by the mask updates per epoch before you compare it against ``tau``.

Two real checkpoints of the quantile run confirm the identity: the chained
ratios over epochs 187 to 199 reproduce the measured ``K_199 / K_186`` to a
relative error of 1.4e-10.

Reader functions load raw data and return plain dicts — no matplotlib.
Renderer functions draw to a caller-provided axis — no file I/O. They take what
:func:`mean_over_seeds` returns, one entry per run.

Run it with::

    python scripts/visualize_support_turnover.py \
        --base_dirs /data/aloradad/results/cifar100/resnet50 \
        --experiments "bregman_linbreg*" --output results/support_turnover
"""

import os
import sys
from typing import Any, Dict, List, Tuple

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_MODULE_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BIRTHS_COL = "bregman/support_births"
DEATHS_COL = "bregman/support_deaths"

# Every per-epoch value the reader derives, in CSV order. ``epoch`` indexes them.
DERIVED_COLUMNS = ("births", "deaths", "tau", "nu", "ratio", "cumulative_tau")

MARKER_SIZE = 3.5
MARKERS_PER_LINE = 10
BAND_ALPHA = 0.18  # the shade scripts/visualize.py gives its cross-seed band

# Axis labels, as the formula each panel plots. m^k is the binary mask after
# epoch k. The numerator is the flip rate's Hamming distance and the denominator
# is RigL's active count, so a DST reader needs no legend for either half.
# The form stays inline: a stacked fraction is unreadable rotated onto a y-axis.
_RATIO = r"\|m^{(k)}-m^{(k-1)}\|_{1}\;/\;\left(\|m^{(k)}\|_{1}+\|m^{(k-1)}\|_{1}\right)"
RATE_LABEL = rf"${_RATIO}$"
CUMULATIVE_LABEL = rf"$\sum_{{k}}\;{_RATIO}$"


def derive_turnover(births, deaths) -> Dict[str, Any]:
    """Per-epoch replacement rate and support-size change, on one denominator.

    ``births`` divides by the current support and ``deaths`` by the previous
    one, so a raw comparison of the two is meaningless; the ratio identity
    ``K_t / K_{t-1} = (1 - d) / (1 - b)`` puts every value below on one
    denominator.

    Args:
        births: per-epoch ``B / K_t``.
        deaths: per-epoch ``D / K_{t-1}``.

    Returns:
        ``tau`` (share of the support replaced in the epoch), ``nu``
        (relative change of the support size), ``ratio`` (``K_t / K_{t-1}``)
        and ``cumulative_tau``.

    >>> out = derive_turnover([0.1], [0.1])
    >>> round(float(out["tau"][0]), 6), round(float(out["nu"][0]), 6)
    (0.1, 0.0)
    """
    b = np.asarray(births, dtype=float)
    d = np.asarray(deaths, dtype=float)
    assert (
        b.shape == d.shape
    ), f"births and deaths must have one shape, got {b.shape} and {d.shape}"
    assert b.size > 0, "derive_turnover needs at least one epoch, got 0"
    assert np.all(b < 1.0) and np.all(
        d < 1.0
    ), f"a rate of 1.0 empties the support, got births<={b.max()} deaths<={d.max()}"

    ratio = (1.0 - d) / (1.0 - b)  # K_t / K_{t-1}
    tau = (b + d - 2.0 * b * d) / (2.0 - b - d)  # (B + D) / (K_t + K_{t-1})
    return {
        "tau": tau,
        "nu": ratio - 1.0,  # (K_t - K_{t-1}) / K_{t-1}
        "ratio": ratio,
        "cumulative_tau": np.cumsum(tau),
    }


def read_support_turnover(df: pd.DataFrame) -> Dict[str, Any]:
    """Per-epoch support turnover of one run, from its merged metrics table.

    The pruner writes the two rates once per epoch, so each column reduces to
    its last sample inside an epoch.

    Args:
        df: merged ``csv/version_*/metrics.csv`` of one run.

    Returns:
        ``epoch``, the two raw rates, and everything :func:`derive_turnover`
        returns.
    """
    missing = [
        c for c in ("epoch", BIRTHS_COL, DEATHS_COL) if c not in df.columns
    ]
    if missing:
        raise KeyError(
            f"support turnover needs columns epoch, {BIRTHS_COL}, {DEATHS_COL}, missing {missing}"
        )

    rates = df.dropna(subset=[BIRTHS_COL, DEATHS_COL]).groupby("epoch").last()
    assert (
        len(rates) > 0
    ), "support turnover needs at least one logged epoch, got 0"

    births = rates[BIRTHS_COL].to_numpy(dtype=float)
    deaths = rates[DEATHS_COL].to_numpy(dtype=float)
    out = derive_turnover(births, deaths)
    out.update(
        epoch=rates.index.to_numpy(dtype=float), births=births, deaths=deaths
    )
    return out


def mean_over_seeds(per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-epoch mean and spread of every derived value, over one run's seeds.

    The seeds outer-join on the epoch index, so a seed that stopped early leaves
    the later epochs to the seeds that reached them. One seed at an epoch gives
    no spread there, so ``<key>_std`` holds NaN and the band breaks.

    Args:
        per_seed: one :func:`read_support_turnover` result per seed.

    Returns:
        ``epoch``, the mean of every name in :data:`DERIVED_COLUMNS`, and a
        ``<name>_std``. Every ``_std`` is None for a single seed.

    >>> seed = lambda r: read_support_turnover(pd.DataFrame({"epoch": [1], BIRTHS_COL: [r], DEATHS_COL: [r]}))
    >>> out = mean_over_seeds([seed(0.1), seed(0.3)])
    >>> round(float(out["tau"][0]), 6), round(float(out["tau_std"][0]), 6)
    (0.2, 0.141421)
    """
    assert per_seed, "mean_over_seeds needs at least one seed, got 0"

    frames = {key: pd.concat([pd.Series(d[key], index=d["epoch"]) for d in per_seed], axis=1) for key in DERIVED_COLUMNS}
    out: Dict[str, Any] = {"epoch": frames["tau"].index.to_numpy(dtype=float)}
    for key, frame in frames.items():
        out[key] = frame.mean(axis=1).to_numpy()
        out[f"{key}_std"] = frame.std(axis=1).to_numpy() if len(per_seed) > 1 else None
    return out


def _plot(ax, data: Dict[str, Any], label: str, style: Tuple, key: str, scale: float = 1.0) -> None:
    """Draw one run's per-epoch curve with its shared color, marker and dash.

    The band is +/-1 standard deviation over the run's seeds. One seed carries
    no spread, so the band is absent.
    """
    color, marker, linestyle = style
    mean = scale * data[key]
    ax.plot(
        data["epoch"],
        mean,
        label=label,
        color=color,
        marker=marker,
        linestyle=linestyle,
        markersize=MARKER_SIZE,
        markevery=max(1, len(data["epoch"]) // MARKERS_PER_LINE),
    )
    std = data[f"{key}_std"]
    if std is not None:
        ax.fill_between(data["epoch"], mean - scale * std, mean + scale * std, color=color, alpha=BAND_ALPHA, linewidth=0)
    ax.set_xlabel("Training epoch")
    ax.set_xlim(left=0)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))


def render_replacement_rate(series: List[Tuple], ax) -> None:
    """Mask bits that flipped in epoch k, over the mask bits set on either side.

    The denominator is the run's own active count, so a 90% run and a 99% run
    read on one axis. 10% means the epoch swapped one active weight in ten.
    """
    for label, style, data in series:
        _plot(ax, data, label, style, "tau", 100.0)
    ax.set_ylabel(RATE_LABEL)
    ax.set_ylim(bottom=0.0)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))


def render_cumulative_replacement(series: List[Tuple], ax) -> None:
    """The same ratio, summed over every epoch up to k.

    A value of 20 means the run replaced twenty times as many weights as it
    holds active. A weight that leaves and comes back counts twice.
    """
    for label, style, data in series:
        _plot(ax, data, label, style, "cumulative_tau")
    ax.set_ylabel(CUMULATIVE_LABEL)
    ax.set_ylim(bottom=0.0)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))


# Panel key -> renderer, combined-figure title, single-panel filename.
PANELS: Dict[str, Tuple] = {
    "rate": (render_replacement_rate, "Replacement rate", "replacement_rate"),
    "cumulative": (
        render_cumulative_replacement,
        "Cumulative replacement",
        "cumulative_replacement",
    ),
}


if __name__ == "__main__":
    # Smoke: a support that holds its size, and one that grows every epoch.
    frames = {
        "constant support": pd.DataFrame({"epoch": [0, 1, 2], BIRTHS_COL: [np.nan, 0.10, 0.08], DEATHS_COL: [np.nan, 0.10, 0.08]}),
        "growing support": pd.DataFrame({"epoch": [0, 1, 2], BIRTHS_COL: [np.nan, 0.10, 0.10], DEATHS_COL: [np.nan, 0.05, 0.05]}),
    }
    for name, frame in frames.items():
        out = read_support_turnover(frame)
        print(f"{name}: tau={np.round(out['tau'], 4)} nu={np.round(out['nu'], 4)} total={out['cumulative_tau'][-1]:.3f}x")
    seeds = mean_over_seeds([read_support_turnover(f) for f in frames.values()])
    print(f"the two as one run's seeds: tau={np.round(seeds['tau'], 4)} +/- {np.round(seeds['tau_std'], 4)}")
