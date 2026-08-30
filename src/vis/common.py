"""Shared matplotlib configuration, backbone registry, and layout helpers.

Every figure that needs rcParams, a legend file, the backbone registry or the
layerwise sizes reads them here. Colors, markers and legend text are not here —
they come from :mod:`src.vis.encoding`, the one place a run's appearance is set.

Examples
--------
>>> from src.vis.common import setup_matplotlib
>>> setup_matplotlib(font_size=10)
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# rcParams — only permitted module-level state in src/vis/
# ---------------------------------------------------------------------------


def _latex_available() -> bool:
    import shutil
    import subprocess  # nosec B404
    import tempfile

    if not shutil.which("pdflatex"):
        return False
    try:
        test_tex = (
            r"\documentclass{article}"
            r"\usepackage{type1cm}\usepackage{type1ec}"
            r"\begin{document}x\end{document}"
        )
        with tempfile.NamedTemporaryFile(
            suffix=".tex", mode="w", delete=False
        ) as f:
            f.write(test_tex)
            tmp = f.name
        result = subprocess.run(  # nosec B603 B607
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tmp],
            capture_output=True,
            timeout=10,
            cwd=os.path.dirname(tmp),
        )
        os.unlink(tmp)
        for ext in (".aux", ".log", ".pdf"):
            p = tmp.replace(".tex", ext)
            if os.path.exists(p):
                os.unlink(p)
        return result.returncode == 0
    except Exception:
        return False


def setup_matplotlib(font_size: int = 10) -> None:
    """Configure matplotlib for publication-quality PDF output.

    Falls back to Computer Modern mathtext when LaTeX is unavailable.

    Args:
        font_size: base font size applied to axes, labels, and ticks.
    """
    use_latex = _latex_available()
    if not use_latex:
        print("Note: LaTeX not available; using mathtext fallback (serif).")
    plt.rcParams.update(
        {
            "text.usetex": use_latex,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.3,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.transparent": True,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def export_standalone_legend(
    handles: List[Any],
    labels: List[str],
    output_path: str,
    ncol: int,
    font_size: int = 10,
    frameon: bool = True,
) -> None:
    """Save a tightly-cropped legend-only PDF.

    Args:
        handles: matplotlib artist handles.
        labels: legend text entries, one per handle.
        output_path: destination PDF path.
        ncol: number of legend columns.
    """
    setup_matplotlib(font_size)
    fig = plt.figure(figsize=(0.01, 0.01))
    leg = fig.legend(
        handles,
        labels,
        loc="center",
        ncol=ncol,
        frameon=frameon,
        columnspacing=0.8,
        handletextpad=0.3,
    )
    fig.canvas.draw()
    bbox = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(
        output_path, format="pdf", bbox_inches=bbox.expanded(1.05, 1.10)
    )
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Backbone registry — marker and dash encode the model in a cross-model figure
# ---------------------------------------------------------------------------

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
    """Return models sorted by registry panel_order, unknowns appended last.

    Args:
        by_model: mapping of model key to any value.

    Returns:
        Ordered list of model keys.
    """
    known = sorted(
        (m for m in by_model if m in MODEL_REGISTRY),
        key=lambda m: MODEL_REGISTRY[m]["panel_order"],
    )
    unknown = [m for m in by_model if m not in MODEL_REGISTRY]
    return known + unknown


# ---------------------------------------------------------------------------
# Layerwise plot constants
# ---------------------------------------------------------------------------

PERFECT_LINE_KW: Dict[str, Any] = dict(
    color="#444444",
    linestyle="--",
    linewidth=0.8,
    alpha=0.6,
    zorder=0,
)

PARAM_BAR_COLOR = "#bcbcbc"
PARAM_BAR_ALPHA = 0.45

RATE_LINESTYLES: Dict[int, Any] = {
    75: (0, (1, 1.5)),
    90: "--",
    95: (0, (3, 1)),
    99: "-",
}

YLIM_PER_RATE: Dict[int, Tuple[float, float]] = {
    75: (0.20, 1.005),
    90: (0.20, 1.005),
    95: (0.60, 1.005),
    99: (0.90, 1.005),
}
YLIM_DEFAULT: Tuple[float, float] = (0.40, 1.005)

LAYERWISE_PANEL_WIDTH = 4.0
LAYERWISE_PANEL_HEIGHT = 3.4
LAYERWISE_FIG_PADDING_W = 0.4


def ylim_for_rate(
    rate: Optional[int],
    scale: str = "fraction",
) -> Tuple[float, float]:
    """Look up per-rate y-axis limits.

    Args:
        rate: integer sparsity rate (e.g. 90 for 90%), or None for default.
        scale: ``"fraction"`` returns values in [0, 1]; ``"percent"`` in
            [0, 100].

    Returns:
        ``(lo, hi)`` y-axis limit tuple.
    """
    lo, hi = YLIM_PER_RATE.get(rate, YLIM_DEFAULT)
    if scale == "percent":
        return (lo * 100.0, hi * 100.0)
    if scale == "fraction":
        return (lo, hi)
    raise ValueError(f"scale must be 'fraction' or 'percent', got {scale!r}")


def layerwise_figsize(n_panels: int) -> Tuple[float, float]:
    """Return ``(width, height)`` for a layerwise figure.

    Args:
        n_panels: number of side-by-side panels.

    Returns:
        Figure size tuple in inches.
    """
    n = max(int(n_panels), 1)
    return (
        LAYERWISE_PANEL_WIDTH * n + LAYERWISE_FIG_PADDING_W,
        LAYERWISE_PANEL_HEIGHT,
    )
