"""The visual-encoding contract: one label and one style per run.

Every figure in the project reads its colors, markers, dash patterns and legend
text from here, so two plots of the same run can never disagree. Callers supply
the ``info`` dict that ``scripts/visualize.py:parse_experiment_name`` builds.

Channels, in the order they are claimed:
    hue        the method class
    lightness  the swept field, if the group sweeps one
    marker     the swept field, else the sparsity level (or the method, where
               sparsity is already an axis)
    dash       the swept field, else the variant, else the sparsity level

Examples
--------
>>> color, marker, ls = get_style(
...     {"method_class": "pruning_struct", "sparsity": 90, "variant": None}
... )
>>> make_label({"method_class": "linbreg", "sparsity": None, "variant": None})
'LinBreg'
"""

import colorsys
from collections import defaultdict

import matplotlib.pyplot as plt

# Alpha and f are swept rarely and clutter every other legend, so they stay off
# until a sweep actually needs them.
SHOW_ALPHA = False
SHOW_f = False

METHOD_CLASS_COLORS = {
    "linbreg": "#1f77b4",  # deep blue
    "adabreg": "#2A662B",  # vibrant cyan
    "pruning_struct": "#ed8d61",  # strong red
    "pruning_unstruct": "#ff7f0e",  # bright orange
    "vanilla": "#61291e",  # distinct brown (baseline)
    "wespeaker": "#9C4F4F",  # dark charcoal
    "dense": "#7f7f7f",  # neutral grey — image dense (SGD) baseline
    # forget about those other methods for now, just make them black so they stand out as "other"
    "proxsgd": "#000000",  # light gray
    "adabregw": "#000000",  # deep navy blue
}

METHOD_DISPLAY_NAMES = {
    "linbreg": "LinBreg",
    "adabreg": "AdaBreg",
    "adabregw": "AdaBregW",
    "pruning_struct": "Str. Prun.",
    "pruning_unstruct": "Unst. Prun.",
    "proxsgd": "ProxSGD",
    "vanilla": "AdamW",
    "wespeaker": "SGD",
    "dense": "Dense",
}

# Method classes that train dense. A run is dense because of its *method*, never
# because its name lacks a sparsity tag — fixed-lambda runs lack one too.
DENSE_METHOD_CLASSES = frozenset({"dense", "vanilla", "wespeaker"})

# Method classes carrying the Bregman knobs (alpha, f, initial sparsity).
# ProxSGD inherits the Bregman parent config, so it belongs here despite the name.
BREGMAN_METHOD_CLASSES = frozenset(
    {"adabreg", "adabregw", "linbreg", "proxsgd"}
)

# Notation shared by labels, group headers and axis titles. Math mode renders
# under both usetex and the mathtext fallback (see setup_matplotlib).
SPARSITY_SYM = r"$\mathsf{s}^{\ast}$"
INIT_SPARSITY_SYM = r"$\mathsf{s}^{0}$"
LAMBDA_SYM = r"$\lambda$"
ALPHA_SYM = r"$\alpha$"
F_SYM = r"$f$"


def is_dense(info):
    """Whether this run trains dense (no sparsity level to place it at)."""
    return info["method_class"] in DENSE_METHOD_CLASSES


def pct_sym():
    """``%``, escaped when LaTeX is doing the typesetting."""
    return r"\%" if plt.rcParams.get("text.usetex") else "%"


def is_fixed_lambda(info):
    """Whether this run holds lambda static, per its method token."""
    return "fixed" in (info.get("method_variant"), info.get("variant"))


# Bar groups, in reading order. Grouping on the method rather than the sparsity
# level keeps a fixed-lambda run out of the dense group and stops each realized
# sparsity from opening a group of its own.
TIER_DENSE, TIER_SPARSE_BASELINE, TIER_FIXED, TIER_ADAPTIVE = range(4)
TIER_LABELS = (
    "Dense",
    "Sparse baselines",
    f"Fixed {LAMBDA_SYM}",
    f"Adaptive {LAMBDA_SYM}",
)


def method_tier(info):
    """Which bar group this run belongs to (see :data:`TIER_LABELS`).

    Anything sparse that isn't Bregman is a baseline, so pruning — and RigL,
    SET and friends when they land — group together without needing to be
    listed.
    """
    if is_dense(info):
        return TIER_DENSE
    if info["method_class"] not in BREGMAN_METHOD_CLASSES:
        return TIER_SPARSE_BASELINE
    return TIER_FIXED if is_fixed_lambda(info) else TIER_ADAPTIVE


# Method class → marker, for plots where sparsity is an axis and so cannot also
# drive the marker (see get_style's ``marker_by``).
METHOD_MARKERS = {
    "adabreg": "o",
    "adabregw": "s",
    "linbreg": "D",
    "proxsgd": "^",
    "pruning_struct": "v",
    "pruning_unstruct": "P",
}

# Sparsity → marker shape  (consistent everywhere)
SPARSITY_MARKERS = {
    None: "s",  # square  — dense / baseline
    0: "s",
    50: "D",  # diamond
    75: "^",  # triangle up
    90: "v",  # triangle down
    95: "o",  # circle
    99: "x",  # x-mark
}

# Variant → marker shape override.  Fixed-lambda runs land at an uncontrolled
# sparsity, so their marker should not encode a sparsity level.
VARIANT_MARKERS = {
    "fixed": "*",  # star — visually distinct from every sparsity marker
}

# Sparsity → line dash pattern
SPARSITY_LINESTYLES = {
    None: "-",
    0: "-",
    50: (0, (5, 3)),
    75: (0, (3, 1, 1, 1)),
    90: "--",
    95: ":",
    99: (0, (1, 1)),
}

# Variant → line dash pattern (used in sweep mode to keep same-alpha curves
# from different variants visually distinct).
VARIANT_LINESTYLES = {
    None: "-",
    "regl1_conv": "-",  # (0, (5, 2)),
    "poor_init": (0, (3, 1, 1, 1)),
    "rescale_prox": (0, (1, 1)),
    "rescale_prox_v2": (0, (3, 1, 1, 1, 1, 1)),
    "subgrad_corr_v2": (0, (5, 1, 1, 1)),
    "subgrad_corr_v3": (0, (5, 2, 1, 2)),
    "subgrad_corr_v4": (0, (1, 2, 5, 2)),
    "fixed": "-",  # (0, (4, 2, 1, 2, 1, 2)),
}

# Sweep rank → marker / dash pattern. A group sweeping one field (initial
# sparsity, fixed lambda, alpha, f) varies all three channels over it, so runs
# differing only in that field stay apart in greyscale print too.
SWEEP_MARKERS = ("o", "^", "s", "D", "v", "P", "X", "*")
SWEEP_LINESTYLES = (
    "-",
    "--",
    (0, (1, 1)),
    (0, (3, 1, 1, 1)),
    (0, (5, 3)),
    (0, (5, 1, 1, 1)),
)

VARIANT_DISPLAY_NAMES = {
    "poor_init": "poor init",
    "fixed": "",  # the lambda value names these runs, so no redundant tag
    "progressive": "Prog.",
    "regl1_conv": "",
    "constant_lr": "Const. lr",
    "RoP": "RoP",
    "rescale_prox": "Rescale Prox.",
    "rescale_prox_v2": "Rescale Prox. V2",
    "rescale_prox_V2": "SubGrad Corr.",
    "subgrad_corr_v2": "SubGrad Corr. V2",
    "subgrad_corr_v3": "SubGrad Corr. V3",
    "subgrad_corr_v4": "SubGrad Corr. V4",
}


def _variant_display_parts(info):
    """Distinct variant display strings for info, method flavor (e.g.
    progressive/fixed) first, then any ad hoc suffix tag (e.g. constant_lr) —
    so a run tagged with both keeps both in its label instead of the suffix tag
    silently overwriting the method flavor."""
    parts = []
    for token in (info.get("method_variant"), info.get("variant")):
        if not token or token in parts:
            continue
        parts.append(token)
    return [
        VARIANT_DISPLAY_NAMES.get(p, p)
        for p in parts
        if VARIANT_DISPLAY_NAMES.get(p, p)
    ]


# Variant color adjustments: (hue_shift, saturation_shift, lightness_shift)
# Hue rotates the color wheel (0-1 wraps), saturation/lightness are additive.
# This keeps variants visually related to their base method but clearly distinct.
VARIANT_COLOR_ADJUSTMENTS = {
    "poor_init": (
        -0.08,
        -0.15,
        -0.12,
    ),  # shift hue toward red, desaturate, darken
    "rescale_prox": (
        0.10,
        0.05,
        0.15,
    ),  # shift hue toward cyan, slightly brighter
    "rescale_prox_v2": (
        0.10,
        0.05,
        0.15,
    ),  # shift hue toward cyan, slightly brighter
    "rescale_prox_V2": (
        0.18,
        -0.10,
        -0.05,
    ),  # shift hue further, slightly muted
    "subgrad_corr_v2": (
        0.18,
        -0.10,
        -0.05,
    ),  # shift hue further, slightly muted,
    "subgrad_corr_v3": (
        0.36,
        -0.20,
        -0.1,
    ),  # shift hue further, slightly muted,
    "subgrad_corr_v4": (-0.18, 0.10, 0.05),
    "fixed": (0.12, -0.0, 0.18),  # shift hue toward purple, slightly darker
    "progressive": (
        0.05,
        0.0,
        -0.12,
    ),  # slightly darker sibling of the base method
}


def _adjust_color(hex_color, hue_shift, sat_shift, light_shift):
    """Adjust a hex color in HLS space: shift hue, saturation, and lightness."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    h = (h + hue_shift) % 1.0
    s = max(0.0, min(1.0, s + sat_shift))
    l = max(0.05, min(0.95, l + light_shift))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def make_label(info):
    """The one legend/tick label builder: ``Method`` plus what varies, in
    parens.

    Reads e.g. ``LinBreg (s*=99%, s0=99%)`` — the sparsity level the run sits at,
    then its Bregman starting sparsity, then the static lambda of a fixed-lambda
    run, then alpha/f, then any variant tag. Fields absent or constant across the
    matched set are dropped (see :func:`assign_label_visibility`); lambda always
    shows, since it is what identifies a fixed-lambda run. A dense run names its
    optimizer, the only thing that tells two of them apart.
    """
    name = METHOD_DISPLAY_NAMES.get(info["method_class"], info["method_class"])
    if is_dense(info) and info.get("optimizer"):
        name = info["optimizer"].upper()

    parts = []
    if info["sparsity"] is not None:
        parts.append(f"{SPARSITY_SYM}={info['sparsity']:g}{pct_sym()}")
    if info.get("_show_init_sparsity"):
        parts.append(
            f"{INIT_SPARSITY_SYM}={info['initial_sparsity']:g}{pct_sym()}"
        )
    if info.get("fixed_lambda") is not None:
        parts.append(f"{LAMBDA_SYM}={info['fixed_lambda']:g}")
    if info.get("_show_alpha"):
        parts.append(f"{ALPHA_SYM}={info['alpha']:g}")
    if info.get("_show_f"):
        parts.append(f"{F_SYM}={info['f']}")
    parts += _variant_display_parts(info)
    return f"{name} ({', '.join(parts)})" if parts else name


def get_style(info, *, marker_by="sparsity"):
    """Return (color, marker, linestyle) — deterministic from metadata.

    Hue names the method. A swept field (see :func:`assign_sweep_styles`) claims
    lightness, marker *and* dash pattern, so runs differing only in their initial
    sparsity or their lambda never collapse onto one line shape. Without a sweep
    the sparsity level drives marker and dash, and fixed-lambda runs take the star
    so they read as their own family.

    ``marker_by="method"`` hands the marker to the method instead — for plots
    where sparsity is an axis and so cannot also encode it.
    """
    variant = info.get("variant")
    color = info.get("_sweep_color")
    if color is None:
        color = METHOD_CLASS_COLORS.get(info["method_class"], "#333333")
        adj = VARIANT_COLOR_ADJUSTMENTS.get(variant)
        if adj:
            color = _adjust_color(color, *adj)

    if marker_by == "method":
        marker = METHOD_MARKERS.get(info["method_class"], "o")
    elif info.get("_sweep_marker") is not None:
        marker = info["_sweep_marker"]
    elif is_fixed_lambda(info):
        marker = VARIANT_MARKERS["fixed"]
    else:
        marker = SPARSITY_MARKERS.get(info["sparsity"], "x")

    if info.get("_sweep_linestyle") is not None:
        ls = info["_sweep_linestyle"]
    elif variant is not None:
        ls = VARIANT_LINESTYLES.get(variant, "-")
    else:
        ls = SPARSITY_LINESTYLES.get(info["sparsity"], "-")
    return color, marker, ls


def _sweep_members(members, param):
    """The subset of ``members`` that legitimately carries ``param``.

    Fixed-lambda runs use default alpha/f (1.0/50) and aren't part of any such
    sweep; pooling them with non-fixed runs at the same sparsity creates a
    spurious 2-point "sweep" that maps the fixed run to the darkest end of the
    gradient (visually black) and overrides VARIANT_COLOR_ADJUSTMENTS["fixed"]
    in get_style. Initial sparsity and the fixed lambda are read from the run
    name, so they count for every variant that has them.

    >>> _sweep_members([{"variant": "fixed"}, {"variant": None}], "alpha")
    [{'variant': None}]
    """
    if param in ("initial_sparsity", "fixed_lambda"):
        return members
    return [m for m in members if not is_fixed_lambda(m)]


def sweep_param(members):
    """Which field this group sweeps, or None.

    Initial sparsity wins ties, then the fixed lambda.
    """
    for cand in ("initial_sparsity", "fixed_lambda", "alpha", "f"):
        pool = _sweep_members(members, cand)
        vals = {m.get(cand) for m in pool if m.get(cand) is not None}
        if len(vals) >= 2:
            return cand
    return None


def assign_sweep_styles(experiments):
    """Stamp ``_sweep_color``/``_sweep_marker``/``_sweep_linestyle`` per run.

    Within each (method_class, sparsity) group, whichever of {initial_sparsity,
    fixed_lambda, alpha, f} varies drives all three channels. Keying on the
    swept *value* rather than the position in the list keeps two runs sharing a
    value styled alike no matter what else differs between them.
    """
    groups = defaultdict(list)
    for _, info in experiments:
        groups[(info["method_class"], info.get("sparsity"))].append(info)

    for (method, _), members in groups.items():
        param = sweep_param(members)
        if param is None:
            continue
        members = _sweep_members(members, param)
        values = sorted(
            {m[param] for m in members if m.get(param) is not None}
        )
        if len(values) < 2:
            continue
        base = METHOD_CLASS_COLORS.get(method, "#333333")
        for info in members:
            v = info.get(param)
            if v is None:
                continue
            rank = values.index(v)
            t = rank / (len(values) - 1)
            info["_sweep_color"] = _adjust_color(base, 0, 0, 0.30 - 0.55 * t)
            info["_sweep_marker"] = SWEEP_MARKERS[rank % len(SWEEP_MARKERS)]
            info["_sweep_linestyle"] = SWEEP_LINESTYLES[
                rank % len(SWEEP_LINESTYLES)
            ]


def clear_sweep_styles(experiments):
    """Drop the stamped sweep styling so another encoding can own the
    channels."""
    for _, info in experiments:
        for key in ("_sweep_color", "_sweep_marker", "_sweep_linestyle"):
            info.pop(key, None)


def assign_label_visibility(experiments):
    """Set the ``_show_*`` flags per info based on whether the field takes >=2
    distinct values across the full matched set.

    Covers alpha, f and the initial sparsity. None counts as a distinct value
    (mixed presence is still variation). Only Bregman methods carry these — they
    are Bregman-only hyperparameters, so dense baselines and pruning runs never
    get them stamped on the label even when alpha varies elsewhere in the matched
    set. The fixed lambda has no flag: it identifies a fixed-lambda run, so
    :func:`make_label` always shows it.
    """
    if not experiments:
        return
    infos = [info for _, info in experiments]
    show_alpha = len({i.get("alpha") for i in infos}) >= 2
    show_f = len({i.get("f") for i in infos}) >= 2
    show_init = len({i.get("initial_sparsity") for i in infos}) >= 2
    for info in infos:
        is_bregman = info["method_class"] in BREGMAN_METHOD_CLASSES
        info["_show_alpha"] = (
            show_alpha
            and info.get("alpha") is not None
            and is_bregman
            and SHOW_ALPHA
        )
        info["_show_f"] = (
            show_f and info.get("f") is not None and is_bregman and SHOW_f
        )
        info["_show_init_sparsity"] = (
            show_init
            and info.get("initial_sparsity") is not None
            and is_bregman
        )


EXPERIMENT_ORDER = [
    "dense",
    "vanilla",
    "wespeaker",
    "pruning_struct",
    "pruning_unstruct",
    "linbreg",
    "adabreg",
    "adabregw",
]


def experiment_sort_key(info):
    """Sort key placing a run in its bar group and ordering it within.

    Groups read dense, sparse baselines, fixed lambda, adaptive lambda; inside
    a group runs sort by method, so LinBreg and AdaBreg stay in blocks. Lambda
    then orders the fixed-lambda runs and the sparsity level orders the rest —
    one key serves both, since lambda is constant where it is unset.

    Reused by downstream scripts so they can re-sort their own legend handles
    to match this ordering.
    """
    mc = info.get("method_class", "")
    lam, sparsity = info.get("fixed_lambda"), info.get("sparsity")
    return (
        method_tier(info),
        EXPERIMENT_ORDER.index(mc) if mc in EXPERIMENT_ORDER else 99,
        lam if lam is not None else -1.0,
        sparsity if sparsity is not None else -1,
        info.get("variant") or "",
        info["alpha"] if info.get("alpha") is not None else -1.0,
        info["f"] if info.get("f") is not None else -1,
    )
