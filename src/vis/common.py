"""Shared matplotlib configuration, style constants, and layout helpers.

Used by all renderer modules under ``src/vis/``.

Examples
--------
>>> from src.vis.common import setup_matplotlib, get_style
>>> setup_matplotlib(font_size=10)
>>> color, marker, ls = get_style(
...     {"method_class": "pruning_struct", "sparsity": 90, "variant": None}
... )
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
        print(
            "Note: LaTeX not available; using mathtext fallback (serif)."
        )
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
    bbox = leg.get_window_extent().transformed(
        fig.dpi_scale_trans.inverted()
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(
        output_path, format="pdf", bbox_inches=bbox.expanded(1.05, 1.10)
    )
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

METHOD_CLASS_COLORS: Dict[str, str] = {
    "linbreg": "#1f77b4",
    "adabreg": "#2A662B",
    "pruning_struct": "#ed8d61",
    "pruning_unstruct": "#ff7f0e",
    "vanilla": "#61291e",
    "wespeaker": "#9C4F4F",
    "proxsgd": "#000000",
    "adabregw": "#000000",
}

METHOD_DISPLAY_NAMES: Dict[str, str] = {
    "linbreg": "LinBreg",
    "linbreg_fixed": r"LinBreg (Fixed $\lambda$)",
    "adabreg": "AdaBreg",
    "adabreg_fixed": r"AdaBreg (Fixed $\lambda$)",
    "adabregw": "AdaBregW",
    "pruning_struct": "Str. Prun.",
    "pruning_unstruct": "Unst. Prun.",
    "proxsgd": "ProxSGD",
    "vanilla": "AdamW",
    "wespeaker": "SGD",
}

SPARSITY_MARKERS: Dict[Optional[int], str] = {
    None: "s",
    0: "s",
    50: "D",
    75: "^",
    90: "v",
    95: "o",
    99: "x",
}

VARIANT_MARKERS: Dict[str, str] = {
    "fixed": "*",
}

SPARSITY_LINESTYLES: Dict[Optional[int], Any] = {
    None: "-",
    0: "-",
    50: (0, (5, 3)),
    75: (0, (3, 1, 1, 1)),
    90: "--",
    95: ":",
    99: (0, (1, 1)),
}

VARIANT_LINESTYLES: Dict[Optional[str], Any] = {
    None: "-",
    "regl1_conv": "-",
    "poor_init": (0, (3, 1, 1, 1)),
    "rescale_prox": (0, (1, 1)),
    "rescale_prox_v2": (0, (3, 1, 1, 1, 1, 1)),
    "subgrad_corr_v2": (0, (5, 1, 1, 1)),
    "subgrad_corr_v3": (0, (5, 2, 1, 2)),
    "subgrad_corr_v4": (0, (1, 2, 5, 2)),
    "fixed": "-",
}

VARIANT_DISPLAY_NAMES: Dict[str, str] = {
    "poor_init": "poor init",
    "fixed": r"Fixed $\lambda$",
    "regl1_conv": "",
    "rescale_prox": "Rescale Prox.",
    "rescale_prox_v2": "Rescale Prox. V2",
    "rescale_prox_V2": "SubGrad Corr.",
    "subgrad_corr_v2": "SubGrad Corr. V2",
    "subgrad_corr_v3": "SubGrad Corr. V3",
    "subgrad_corr_v4": "SubGrad Corr. V4",
}

VARIANT_COLOR_ADJUSTMENTS: Dict[str, Tuple[float, float, float]] = {
    "poor_init": (-0.08, -0.15, -0.12),
    "rescale_prox": (0.10, 0.05, 0.15),
    "rescale_prox_v2": (0.10, 0.05, 0.15),
    "rescale_prox_V2": (0.18, -0.10, -0.05),
    "subgrad_corr_v2": (0.18, -0.10, -0.05),
    "subgrad_corr_v3": (0.36, -0.20, -0.1),
    "subgrad_corr_v4": (-0.18, 0.10, 0.05),
    "fixed": (0.12, -0.0, 0.18),
}

EXPERIMENT_ORDER: List[str] = [
    "vanilla",
    "wespeaker",
    "pruning_struct",
    "pruning_unstruct",
    "linbreg",
    "adabreg",
    "adabregw",
]


def _adjust_color(
    hex_color: str,
    hue_shift: float,
    sat_shift: float,
    light_shift: float,
) -> str:
    """Adjust a hex color in HLS space.

    Args:
        hex_color: RGB hex string, e.g. ``"#1f77b4"``.
        hue_shift: additive hue shift in [0, 1].
        sat_shift: additive saturation shift.
        light_shift: additive lightness shift.

    Returns:
        Adjusted RGB hex string.
    """
    import colorsys

    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i: i + 2], 16) / 255.0 for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    h = (h + hue_shift) % 1.0
    s = max(0.0, min(1.0, s + sat_shift))
    l = max(0.05, min(0.95, l + light_shift))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def get_style(
    info: Dict[str, Any],
) -> Tuple[str, str, Any]:
    """Return ``(color, marker, linestyle)`` for an experiment info dict.

    Args:
        info: experiment metadata dict from ``parse_experiment_name``.

    Returns:
        Tuple of (hex color, marker string, linestyle).
    """
    color = info.get("_gradient_color")
    if color is not None:
        variant = info.get("variant")
        marker = VARIANT_MARKERS.get(
            variant, SPARSITY_MARKERS.get(info.get("sparsity"), "x")
        )
        ls = VARIANT_LINESTYLES.get(variant, "-")
    else:
        color = METHOD_CLASS_COLORS.get(
            info.get("method_class", ""), "#333333"
        )
        variant = info.get("variant")
        if variant:
            adj = VARIANT_COLOR_ADJUSTMENTS.get(variant)
            if adj:
                color = _adjust_color(color, *adj)
        if variant == "fixed":
            marker = VARIANT_MARKERS.get(
                "fixed",
                SPARSITY_MARKERS.get(info.get("sparsity"), "x"),
            )
            ls = VARIANT_LINESTYLES.get(
                "fixed",
                SPARSITY_LINESTYLES.get(info.get("sparsity"), "-"),
            )
        else:
            marker = SPARSITY_MARKERS.get(info.get("sparsity"), "x")
            ls = SPARSITY_LINESTYLES.get(info.get("sparsity"), "-")
    return color, marker, ls


def make_label(
    info: Dict[str, Any],
    *,
    fixed_as_token: bool = True,
) -> str:
    """Create a concise legend label from experiment metadata.

    Args:
        info: experiment metadata dict.
        fixed_as_token: if True, render fixed-λ runs as ``"Method (fixed)"``
            instead of showing the numeric λ value.

    Returns:
        Human-readable label string.
    """
    name = METHOD_DISPLAY_NAMES.get(
        info.get("method_class", ""), info.get("method_class", "")
    )
    if (
        info.get("variant") == "fixed"
        and info.get("fixed_lambda") is not None
    ):
        if fixed_as_token:
            return f"{name} (fixed)"
        lam = info["fixed_lambda"]
        lam_sym = (
            r"$\lambda$" if plt.rcParams.get("text.usetex") else "λ"
        )
        return f"{name} ({lam_sym}={lam:g})"
    if info.get("sparsity") is not None:
        pct = r"\%" if plt.rcParams.get("text.usetex") else "%"
        label = f"{name} {info['sparsity']}{pct}"
    else:
        label = name
    if info.get("variant"):
        if info["variant"] in VARIANT_DISPLAY_NAMES:
            variant = VARIANT_DISPLAY_NAMES[info["variant"]]
        else:
            variant = info["variant"]
        if variant:
            label += f" ({variant})"
    return label


def experiment_sort_key(info: Dict[str, Any]) -> Tuple:
    """Sort key for consistent legend ordering across all plots.

    Args:
        info: experiment metadata dict.

    Returns:
        Comparable tuple used as a sort key.
    """
    mc = info.get("method_class", "")
    variant = info.get("variant")
    if mc in ("vanilla", "wespeaker"):
        group = 0
    elif mc in ("pruning_struct", "pruning_unstruct"):
        group = 1
    elif variant == "fixed":
        group = 2
    else:
        group = 3
    return (
        group,
        EXPERIMENT_ORDER.index(mc) if mc in EXPERIMENT_ORDER else 99,
        info.get("sparsity") or -1,
        info.get("variant") or "",
        info.get("alpha") if info.get("alpha") is not None else -1.0,
        info.get("f") if info.get("f") is not None else -1,
    )


# ---------------------------------------------------------------------------
# Backbone registry (from visualize_common.py)
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
# Layerwise plot constants (from visualize_common.py)
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
    raise ValueError(
        f"scale must be 'fraction' or 'percent', got {scale!r}"
    )


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
