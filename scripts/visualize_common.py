"""Cross-script constants and helpers shared by the visualization stack.

Single source of truth for things that two or more scripts in ``scripts/``
need to agree on — backbone-specific styling, per-rate y-axis zooms, the
layerwise plot styling palette, and the canonical layerwise figure size.

Importers:
    visualize_structured_vs_unstructured.py
    visualize_weight_norms.py
"""

from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Backbone registry
# ---------------------------------------------------------------------------
# In cross-model figures, color is method-driven; marker + linestyle encode
# the model. ``panel_order`` is the left-to-right order in cross-model panel
# layouts; only models that actually appear in the matched data get a panel.

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "wespeaker_ecapa_tdnn": {
        "display_name": "ECAPA-TDNN",
        "marker": "^",
        "linestyle": (0, (5, 2)),
        "panel_order": 0,
    },
    "wespeaker_resnet34": {
        "display_name": "ResNet34",
        "marker": "o",
        "linestyle": "-",
        "panel_order": 1,
    },
}


def panel_models(by_model: Dict[str, Any]) -> List[str]:
    """Return models present in ``by_model``, sorted by registry panel_order.

    Unknown models (not in MODEL_REGISTRY) are appended after, in the
    insertion order of ``by_model``.
    """
    known = sorted(
        (m for m in by_model if m in MODEL_REGISTRY),
        key=lambda m: MODEL_REGISTRY[m]["panel_order"],
    )
    unknown = [m for m in by_model if m not in MODEL_REGISTRY]
    return known + unknown


# ---------------------------------------------------------------------------
# Per-rate y-axis zooms
# ---------------------------------------------------------------------------
# Canonical scale is *fraction* in [0, 1]. Convert at the call site for
# percent-scale plots via ``ylim_for_rate(rate, scale="percent")``.

YLIM_PER_RATE: Dict[int, Tuple[float, float]] = {
    75: (0.20, 1.005),
    90: (0.20, 1.005),
    95: (0.60, 1.005),
    99: (0.90, 1.005),
}
YLIM_DEFAULT: Tuple[float, float] = (0.40, 1.005)


def ylim_for_rate(
    rate: Optional[int], scale: str = "fraction",
) -> Tuple[float, float]:
    """Look up the per-rate y-axis limits.

    scale="fraction" → values in [0, 1] (default).
    scale="percent"  → values in [0, 100].
    """
    lo, hi = YLIM_PER_RATE.get(rate, YLIM_DEFAULT)
    if scale == "percent":
        return (lo * 100.0, hi * 100.0)
    if scale == "fraction":
        return (lo, hi)
    raise ValueError(f"scale must be 'fraction' or 'percent', got {scale!r}")


# ---------------------------------------------------------------------------
# Layerwise plot styling
# ---------------------------------------------------------------------------

PERFECT_LINE_KW = dict(
    color="#444444", linestyle="--", linewidth=0.8, alpha=0.6, zorder=0,
)

PARAM_BAR_COLOR = "#bcbcbc"
PARAM_BAR_ALPHA = 0.45

# Linestyle keyed by integer sparsity rate, used in the overlay-rates layerwise
# plot to distinguish curves at different targets (color/marker still encode
# method/variant).
RATE_LINESTYLES: Dict[int, Any] = {
    75: (0, (1, 1.5)),
    90: "--",
    95: (0, (3, 1)),
    99: "-",
}

# Single canonical per-panel size for every layerwise PDF. Width scales with
# panel count; height is fixed.
LAYERWISE_PANEL_WIDTH = 4.0
LAYERWISE_PANEL_HEIGHT = 3.4
LAYERWISE_FIG_PADDING_W = 0.4


def layerwise_figsize(n_panels: int) -> Tuple[float, float]:
    """Return ``(width, height)`` for a layerwise figure with ``n_panels``."""
    n = max(int(n_panels), 1)
    return (
        LAYERWISE_PANEL_WIDTH * n + LAYERWISE_FIG_PADDING_W,
        LAYERWISE_PANEL_HEIGHT,
    )
