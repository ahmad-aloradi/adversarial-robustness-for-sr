"""The metric registry: one row per metric, for every figure that plots one.

A row says what a metric is called on an axis, which stage it belongs to, which
plots it supports and how its y-axis is scaled. One table replaces the five that
used to key on the metric name separately and could disagree about it.

Examples
--------
>>> metric_for("valid/MulticlassAccuracy").stage
'valid'
>>> metric_for("no/such/metric").plots
('curves',)
"""

import re
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Plot kinds: a time series over epochs or steps, and one cross-seed bar per run.
PLOT_KINDS = ("curves", "summary")


@dataclass(frozen=True)
class Metric:
    """One metric: its axis label, its stage, and the plots it supports.

    ``ylim``/``yticks`` pin an axis whose useful range is known in advance, so a
    sparsity panel does not rescale itself out of the region that matters.
    """

    key: str
    label: str
    stage: str
    plots: Tuple[str, ...] = ("curves",)
    log_scale: bool = False
    ylim: Optional[Tuple[float, float]] = None
    yticks: Tuple[float, ...] = ()


_SPARSITY_TICKS = (0.75, 0.80, 0.85, 0.90, 0.95, 0.99)

METRICS = (
    Metric("train_loss", "Train Loss", "train"),
    Metric("valid_loss", "Valid. Loss", "valid"),
    Metric("train/MulticlassAccuracy", "Train Top-1 Accuracy", "train", ("curves", "summary")),
    Metric("valid/MulticlassAccuracy", "Valid. Top-1 Accuracy", "valid", ("curves", "summary")),
    Metric("train/MulticlassAccuracy_top5", "Train Top-5 Accuracy", "train", ("curves", "summary")),
    Metric("valid/MulticlassAccuracy_top5", "Valid. Top-5 Accuracy", "valid", ("curves", "summary")),
    # Test accuracy is one scalar: a curve of one point cannot part two variants.
    Metric("test/MulticlassAccuracy", "Test Top-1 Accuracy", "test", ("summary",)),
    Metric("test/MulticlassAccuracy_top5", "Test Top-5 Accuracy", "test", ("summary",)),
    # Zeros over every weight tensor, norms and biases aside. Each pruner writes
    # its own key for that one quantity (docs/image_benchmarks.md). The whole-model
    # `sparsity` has no row: no method sparsifies BatchNorm or the biases.
    Metric("pruning/sparsity", r"$\mathsf{s}(\theta)$", "internal", ylim=(0.7, 1.005), yticks=_SPARSITY_TICKS),
    Metric("bregman/pruned_sparsity", r"$\mathsf{s}(\theta)$", "internal", ylim=(0.7, 1.005), yticks=_SPARSITY_TICKS),
    Metric("bregman/global_lambda", r"$\lambda$", "internal", log_scale=True),
    Metric("lr", "Learning rate", "schedule", log_scale=True),
    Metric("train/margin", "AAM margin $m$", "schedule"),
    # These come from the score files; scripts/visualize_test_metrics.py draws them.
    Metric("EER", "EER", "test", ()),
    Metric("minDCF", "minDCF", "test", ()),
)

METRIC_BY_KEY = {m.key: m for m in METRICS}
assert len(METRIC_BY_KEY) == len(METRICS), "every Metric needs a unique key"
assert all(set(m.plots) <= set(PLOT_KINDS) for m in METRICS), f"plots are drawn from {PLOT_KINDS}"


def metric_for(key):
    """The Metric a key names, or a curve-only row with a title-cased label.

    An unregistered column still plots as a curve, so a new logged metric needs
    no row before it can be looked at.
    """
    return METRIC_BY_KEY.get(key) or Metric(key, key.replace("_", " ").title(), "other")


def short_name(key):
    """The metric's filename stem."""
    return key.replace("/", "_").replace(" ", "_").lower()


# Method to (preferred lr column, verify-against column). One scheduler per method.
LR_COLUMN_RULES = {
    "linbreg": (r"^lr-LinBreg/conv_layers$", r"^lr-LinBreg/linear_layers$"),
    "adabreg": (r"^lr-AdaBreg/conv_layers$", r"^lr-AdaBreg/linear_layers$"),
    "proxsgd": (r"^lr-ProxSGD/conv_layers$", r"^lr-ProxSGD/linear_layers$"),
    "wespeaker": (r"^lr-SGD$", None),
    "vanilla": (r"^lr-(AdamW|Adam|SGD)$", None),
}


def resolve_lr_column(df, method_key, dirname):
    """The column in ``df`` that holds the learning rate, or None.

    ``lr`` is virtual: every method logs it under its own optimizer's name. Where
    a verify column exists and disagrees with the proxy, the groups are not on
    one scheduler and the curve would mislead, so this says so.
    """
    preferred, verify = LR_COLUMN_RULES.get(method_key, LR_COLUMN_RULES["vanilla"])
    pref = [c for c in df.columns if re.match(preferred, c)]
    if not pref:
        any_lr = [c for c in df.columns if c.startswith("lr-")]
        return any_lr[0] if any_lr else None

    col = pref[0]
    ver = [c for c in df.columns if verify and re.match(verify, c)]
    if ver:
        both = df[[col, ver[0]]].dropna()
        if len(both) and not np.allclose(both[col].values, both[ver[0]].values, rtol=0, atol=1e-12):
            print(f"  [warn] {dirname}: {col} differs from {ver[0]} — conv-layer proxy may be misleading.")
    return col


if __name__ == "__main__":
    for m in METRICS:
        print(f"{m.key:<34} {m.stage:<9} plots={m.plots} log={m.log_scale}")
