"""The visual-encoding contract: one label and one style per run.

Every figure in the project reads its colors, markers, dash patterns and legend
text from here, so two plots of the same run can never disagree. A method is one
:class:`Method` row and a flavor is one :class:`Variant` row. A row carries every
field a figure needs, so no method can reach a plot with a color but no marker.

Channels, in the order they are claimed:
    hue        the method class
    lightness  the swept field, if the group sweeps one
    marker     the swept field, else the sparsity level (or the method, where
               sparsity is already an axis)
    dash       the swept field, else the variant, else the sparsity level

:class:`Encoding` decides what a legend shows and which field owns the sweep. It
reads the whole matched set once, and answers per run. Ask it about a run it was
not built from and it raises.

Examples
--------
>>> method_for("bregman_linbreg_quantile").key
'linbreg'
>>> flavor_for("bregman_linbreg_quantile").key
'quantile'
>>> method_for("pruning_str")
Traceback (most recent call last):
ValueError: a run name states one registered method (see METHODS), got 'pruning_str'
"""

import colorsys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Tuple

import matplotlib.pyplot as plt

# Alpha, f and the starting sparsity are swept rarely and clutter every other
# legend, so they stay off until a sweep actually needs them.
SHOW_ALPHA = False
SHOW_f = False
SHOW_INIT_SPARSITY = False

# Bar groups, in reading order. Grouping on the method rather than the sparsity
# level keeps a fixed-lambda run out of the dense group and stops each realized
# sparsity from opening a group of its own.
TIER_DENSE, TIER_SPARSE_BASELINE, TIER_FIXED, TIER_ADAPTIVE = range(4)

# Notation shared by labels, group headers and axis titles. Math mode renders
# under both usetex and the mathtext fallback (see setup_matplotlib).
SPARSITY_SYM = r"$\mathsf{s}(\theta^{\ast})$"
INIT_SPARSITY_SYM = r"$\mathsf{s}^{(0)}$"
LAMBDA_SYM = r"$\lambda$"
ALPHA_SYM = r"$\alpha$"
F_SYM = r"$f$"

TIER_LABELS = (
    "Dense",
    "Sparse baselines",
    f"Fixed {LAMBDA_SYM}",
    f"Adaptive {LAMBDA_SYM}",
)


# ---------------------------------------------------------------------------
# 1. The method registry — one row per method
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Method:
    """One method: how its runs are named, and how every figure draws it.

    ``tokens`` are the substrings of a run name that select this method. A
    method carries several where the launcher renamed it and finished runs keep
    the old spelling. What follows the matched token in the name is the flavor
    (see :class:`Variant`), so a new flavor costs no row here.

    ``family`` decides the bar group and which hyperparameters the run carries:
    ``dense`` trains dense, ``bregman`` carries alpha, f and lambda, ``baseline``
    is anything else that trains sparse.
    """

    key: str
    tokens: Tuple[str, ...]
    display: str
    color: str
    marker: str
    family: str


# Reading order everywhere. A row's position is its sort rank. Every token an
# experiment config or scripts/fabfile.py can put in a run name has a row.
METHODS = (
    Method("dense", ("dense_sgd",), "SGD", "#7f7f7f", "s", "dense"),
    Method("vanilla", ("dense_adamw", "vanilla"), "AdamW", "#61291e", ">", "dense"),
    Method("wespeaker", ("wespeaker",), "SGD", "#9C4F4F", "8", "dense"),
    Method("pruning_struct", ("pruning_mag_struct",), "Str. Prun.", "#ed8d61", "v", "baseline"),
    Method("pruning_unstruct", ("pruning_mag_unstruct",), "Unst. Prun.", "#ff7f0e", "P", "baseline"),
    Method("static", ("pruning_static",), "Static-ERK", "#bcbd22", "d", "baseline"),
    Method("snip", ("pruning_snip",), "SNIP", "#17becf", "p", "baseline"),
    Method("set", ("pruning_set",), "SET", "#c5b0d5", "X", "baseline"),
    Method("rigl", ("pruning_rigl",), "RigL", "#9467bd", "*", "baseline"),
    Method("granet", ("pruning_granet",), "GraNet", "#e377c2", "h", "baseline"),
    Method("str", ("soft_threshold",), "STR", "#d62728", "<", "baseline"),
    Method("linbreg", ("linbreg",), "LinBreg", "#1f77b4", "D", "bregman"),
    Method("adabreg", ("adabreg",), "AdaBreg", "#2A662B", "o", "bregman"),
    Method("proxsgd", ("proxsgd",), "ProxSGD", "#000000", "^", "bregman"),
)

METHOD_BY_KEY = {m.key: m for m in METHODS}
# Longest token first, so pruning_mag_unstruct wins over any shorter prefix.
_METHOD_TOKENS = sorted(((m, t) for m in METHODS for t in m.tokens), key=lambda mt: -len(mt[1]))

assert len(METHOD_BY_KEY) == len(METHODS), "every Method needs a unique key"
assert len({t for _, t in _METHOD_TOKENS}) == len(_METHOD_TOKENS), "every Method token needs a unique row"
assert len({m.color for m in METHODS}) == len(METHODS), "every Method needs a unique color"
assert len({m.marker for m in METHODS}) == len(METHODS), "every Method needs a unique marker"
assert {m.family for m in METHODS} <= {"dense", "baseline", "bregman"}, "family is dense, baseline or bregman"

METHOD_SORT_RANK = {m.key: i for i, m in enumerate(METHODS)}

# A (method, flavor) pair that is a method of its own, not an ablation of the
# base. SGap names the feedback controller on the sparsity gap. TopK names the
# K-th order statistic of the dual. A fixed-lambda run runs neither controller,
# so it keeps the bare method name.
METHOD_VARIANT_DISPLAY_NAMES = {
    ("linbreg", None): "LinBregSGap",
    ("linbreg", "progressive"): "LinBregSGap + Ramp",
    ("linbreg", "quantile"): "LinBregTopK",
    ("linbreg", "quantile_progressive"): "LinBregTopK + Ramp",
    ("adabreg", None): "AdaBregSGap",
    ("adabreg", "progressive"): "AdaBregSGap + Ramp",
    ("adabreg", "quantile"): "AdaBregTopK",
    ("adabreg", "quantile_progressive"): "AdaBregTopK + Ramp",
}


def _match_method(token):
    """The Method a run-name token selects, and the token text that selected it.

    Matches on the longest token, so ``bregman_linbreg_quantile`` resolves to
    LinBreg and everything after the token stays for :func:`flavor_for`. An
    unregistered name raises: a silent fallback drew a sparse run as a dense
    baseline, with no sparsity and in the wrong bar group.
    """
    for m, matched in _METHOD_TOKENS:
        if matched in token:
            return m, matched
    raise ValueError(f"a run name states one registered method (see METHODS), got {token!r}")


def method_for(token):
    """The Method a run-name token selects. Raises on an unregistered name."""
    return _match_method(token)[0]


# ---------------------------------------------------------------------------
# 2. The variant registry — one row per method flavor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    """One method flavor: its legend tag, its dash pattern and its hue shift.

    ``display`` is empty where the name already says it — a fixed-lambda run is
    named by its lambda, so a "fixed" tag would say it twice. ``color_shift`` is
    ``(hue, saturation, lightness)``; hue wraps the wheel, the other two add.
    """

    key: str
    display: str
    linestyle: Any
    color_shift: Tuple[float, float, float]


VARIANTS = (
    Variant("fixed", "", "-", (0.12, 0.0, 0.18)),
    Variant("progressive", "Prog.", (0, (6, 2)), (0.05, 0.0, -0.12)),
    Variant("quantile", "Quant.", (0, (6, 1, 1, 1)), (-0.10, 0.10, 0.05)),
    Variant("quantile_progressive", "Quant. Prog.", (0, (6, 1, 1, 1, 1, 1)), (-0.10, 0.10, -0.10)),
    Variant("iter", "Iter.", (0, (4, 2)), (0.05, 0.0, -0.12)),
    Variant("onetime", "One-shot", (0, (2, 2)), (-0.05, 0.0, 0.15)),
    Variant("constant_lr", "Const. lr", "-", (0.0, 0.0, 0.0)),
)

VARIANT_BY_KEY = {v.key: v for v in VARIANTS}
_VARIANT_KEYS_LONGEST_FIRST = sorted(VARIANT_BY_KEY, key=len, reverse=True)

assert len(VARIANT_BY_KEY) == len(VARIANTS), "every Variant needs a unique key"


def variant_for(key):
    """The Variant a key names, or an unstyled row carrying the key verbatim.

    An ad hoc name suffix (``cls_scale2``) has no row, so it prints as written
    and takes no dash or hue of its own.
    """
    if key is None:
        return None
    return VARIANT_BY_KEY.get(key) or Variant(key, key, "-", (0.0, 0.0, 0.0))


def flavor_for(token):
    """The Variant that follows the method token in a run name, or None.

    ``bregman_linbreg_quantile_progressive`` gives ``quantile_progressive``. The
    flavor ends at the next ``-``, so a following name segment such as the ramp
    tag in ``linbreg_progressive-ramp100_cubic`` stays out of it. It also ends at
    the next ``_``, so an older name that glued its settings on —
    ``linbreg_fixed_lam0.15_noScheduler`` — still reads as fixed. The longest key
    wins, so ``quantile_progressive`` never resolves to ``quantile``. A dense run
    names its optimizer in that slot instead, so it has no flavor.
    """
    method, matched = _match_method(token)
    if method.family == "dense":
        return None
    tail = token.split(matched, 1)[1].lstrip("_").split("-")[0]
    for key in _VARIANT_KEYS_LONGEST_FIRST:
        if tail == key or tail.startswith(key + "_"):
            return VARIANT_BY_KEY[key]
    return None


# ---------------------------------------------------------------------------
# 3. Sparsity and sweep channels
# ---------------------------------------------------------------------------

# Sparsity → marker shape (consistent everywhere)
SPARSITY_MARKERS = {
    None: "s",  # square  — dense / baseline
    0: "s",
    50: "D",  # diamond
    75: "^",  # triangle up
    90: "v",  # triangle down
    95: "o",  # circle
    99: "x",  # x-mark
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

# A fixed-lambda run lands at an uncontrolled sparsity, so its marker must not
# encode a level. The star is distinct from every sparsity marker.
FIXED_LAMBDA_MARKER = "*"

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

# Lightness of the first and the last sweep rank. Absolute, so a base color that
# already sits light or dark cannot push a rank off either end of the scale.
SWEEP_LIGHTNESS = (0.72, 0.20)

# Fields a group can sweep, in the order they win a tie.
SWEEP_FIELDS = ("initial_sparsity", "fixed_lambda", "alpha", "f")


def pct_sym():
    """``%``, escaped when LaTeX is doing the typesetting."""
    return r"\%" if plt.rcParams.get("text.usetex") else "%"


def _lightness(hex_color):
    """The HLS lightness of a hex color, so a caller can shift to an absolute one."""
    r, g, b = (int(hex_color.lstrip("#")[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return colorsys.rgb_to_hls(r, g, b)[1]


def _variant_color(base, variant_key):
    """``base`` shifted by the variant's hue, or ``base`` untouched.

    A variant that asks for no shift must not go through :func:`_adjust_color`:
    the HLS round trip truncates, so it moves a color by a step and lifts pure
    black off zero.
    """
    if variant_key is None:
        return base
    shift = variant_for(variant_key).color_shift
    return _adjust_color(base, *shift) if any(shift) else base


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


@dataclass(frozen=True)
class _Sweep:
    """The three channels one sweep rank claims."""

    color: str
    marker: str
    linestyle: Any


@dataclass(frozen=True)
class _Shown:
    """Which optional fields a label prints for one run."""

    sparsity: bool
    init_sparsity: bool
    alpha: bool
    f: bool


# ---------------------------------------------------------------------------
# 4. The encoding of one matched set
# ---------------------------------------------------------------------------


class Encoding:
    """Label and style for every run in one matched set.

    What varies across the set decides which fields a label prints, and which
    field owns the sweep channels. Both are properties of the set, so they are
    computed once here rather than stamped onto each run — a renderer that draws
    a subset builds its own Encoding over that subset and gets labels that
    describe it.

    ``sweep=False`` hands the marker and dash channels back to the sparsity
    level, for a figure that encodes the swept field on an axis instead.
    """

    def __init__(self, groups, *, sweep=True):
        self._groups = list(groups)
        self._shown = self._assign_labels(self._groups)
        self._sweep = self._assign_sweep(self._groups) if sweep else {}

    def __len__(self):
        return len(self._groups)

    def _of(self, group, table):
        """Look one run up, or raise — a run this Encoding never saw has no
        answer, and a silent default would label it against the wrong set."""
        key = id(group)
        if key not in table:
            raise KeyError(f"Encoding covers the runs it was built from, got {group.dirname}")
        return table[key]

    @staticmethod
    def _assign_labels(groups):
        """Print a field where it takes >=2 distinct values across the set.

        For alpha, f and the starting sparsity, None counts as a distinct value —
        mixed presence is still variation. Only Bregman methods carry those three,
        so a dense baseline never gets them stamped on its label.

        The target set holds only the runs that hold a target. A dense baseline
        has none, and a fixed-lambda run reached its sparsity instead of asking
        for it, so neither may decide the tag for the runs that did ask. A
        fixed-lambda run then always shows what it reached: that value is its
        identity, the way lambda is.
        """
        show_alpha = len({g.alpha for g in groups}) >= 2
        show_f = len({g.f for g in groups}) >= 2
        show_init = len({g.initial_sparsity for g in groups}) >= 2
        targets = {g.sparsity for g in groups if g.sparsity is not None and not g.is_fixed_lambda}
        show_sparsity = len(targets) >= 2

        shown = {}
        for g in groups:
            bregman = g.method.family == "bregman"
            shown[id(g)] = _Shown(
                sparsity=g.sparsity is not None and (show_sparsity or g.is_fixed_lambda),
                init_sparsity=show_init and g.initial_sparsity is not None and bregman and SHOW_INIT_SPARSITY,
                alpha=show_alpha and g.alpha is not None and bregman and SHOW_ALPHA,
                f=show_f and g.f is not None and bregman and SHOW_f,
            )
        return shown

    @staticmethod
    def _sweep_field(members):
        """Which field this group sweeps, or None.

        A fixed-lambda run uses default alpha/f and joins no such sweep; pooling
        it with the others would build a spurious two-point sweep and drag it to
        the dark end of the ramp. The starting sparsity and the lambda come from
        the run name, so they count for every variant that carries them.
        """
        for field in SWEEP_FIELDS:
            pool = Encoding._sweep_members(members, field)
            if len({getattr(m, field) for m in pool if getattr(m, field) is not None}) >= 2:
                return field
        return None

    @staticmethod
    def _sweep_members(members, field):
        """The subset of ``members`` that legitimately carries ``field``."""
        if field in ("initial_sparsity", "fixed_lambda"):
            return members
        return [m for m in members if not m.is_fixed_lambda]

    @staticmethod
    def _assign_sweep(groups):
        """Give the swept field all three channels, within each (method, level).

        Keying on the swept *value* rather than the position in the list styles
        two runs sharing a value alike, whatever else differs between them. The
        ramp keeps the variant's own hue and spans SWEEP_LIGHTNESS, so no rank
        lands too pale or too near black.
        """
        buckets = defaultdict(list)
        for g in groups:
            # A fixed-lambda run holds no target, so its level is an outcome and must not split the sweep.
            level = None if g.is_fixed_lambda else g.sparsity
            buckets[(g.method.key, level)].append(g)

        sweeps = {}
        for members in buckets.values():
            field = Encoding._sweep_field(members)
            if field is None:
                continue
            members = Encoding._sweep_members(members, field)
            values = sorted({getattr(m, field) for m in members if getattr(m, field) is not None})
            if len(values) < 2:
                continue
            for g in members:
                v = getattr(g, field)
                if v is None:
                    continue
                rank = values.index(v)
                t = rank / (len(values) - 1)
                base = _variant_color(g.method.color, g.style_variant)
                light, dark = SWEEP_LIGHTNESS
                sweeps[id(g)] = _Sweep(
                    color=_adjust_color(base, 0, 0, light + (dark - light) * t - _lightness(base)),
                    marker=SWEEP_MARKERS[rank % len(SWEEP_MARKERS)],
                    linestyle=SWEEP_LINESTYLES[rank % len(SWEEP_LINESTYLES)],
                )
        return sweeps

    def label(self, group):
        """The one legend/tick label: ``Method`` plus what varies, in parens.

        Reads e.g. ``LinBregTopK (s(θ*)=99%, λ=0.9)`` — the sparsity level it
        sits at, then its Bregman starting sparsity, then the static lambda of a
        fixed-lambda run, then alpha/f, then any variant tag. A dense run reads
        as its optimizer, the only thing that tells two of them apart.

        A run whose sparsity is an outcome rather than a target prints the seed
        mean and its spread. ``sparsity`` stays the rounded integer that keys the
        marker and dash tables, so the printed value drives no style lookup.
        """
        shown = self._of(group, self._shown)
        flavor = group.flavor.key if group.flavor else None
        name = METHOD_VARIANT_DISPLAY_NAMES.get((group.method.key, flavor)) or group.method.display

        parts = []
        if shown.sparsity:
            if group.sparsity_is_outcome:
                value = f"{100 * group.landed_sparsity:.2f}"
                if group.landed_sparsity_std is not None:
                    value += rf"$\pm${100 * group.landed_sparsity_std:.2f}"
            else:
                value = f"{group.sparsity:g}"
            parts.append(f"{SPARSITY_SYM}={value}{pct_sym()}")
        if shown.init_sparsity:
            parts.append(f"{INIT_SPARSITY_SYM}={group.initial_sparsity:g}{pct_sym()}")
        if group.fixed_lambda is not None:
            parts.append(f"{LAMBDA_SYM}={group.fixed_lambda:g}")
        if shown.alpha:
            parts.append(f"{ALPHA_SYM}={group.alpha:g}")
        if shown.f:
            parts.append(f"{F_SYM}={group.f}")
        parts += self._variant_tags(group)
        return f"{name} ({', '.join(parts)})" if parts else name

    @staticmethod
    def _variant_tags(group):
        """The variant tags a label appends: the method flavor, then any ad hoc
        suffix. A flavor that METHOD_VARIANT_DISPLAY_NAMES already names drops
        out, because the name carries it and the tag would say it twice."""
        flavor = group.flavor.key if group.flavor else None
        named = (group.method.key, flavor) in METHOD_VARIANT_DISPLAY_NAMES
        tags = []
        for key in (flavor, group.variant):
            if not key or key in tags or (named and key == flavor):
                continue
            tags.append(key)
        return [d for d in (variant_for(k).display for k in tags) if d]

    def style(self, group, *, marker_by="sparsity"):
        """Return (color, marker, linestyle) for one run.

        Hue names the method. A swept field claims lightness, marker *and* dash,
        so runs differing only in their starting sparsity or their lambda never
        collapse onto one line shape. Without a sweep the sparsity level drives
        marker and dash, and a fixed-lambda run takes the star.

        ``marker_by="method"`` hands the marker to the method instead — for plots
        where sparsity is an axis and so cannot also encode it.
        """
        assert marker_by in ("sparsity", "method"), f"marker_by is sparsity or method, got {marker_by!r}"
        sweep = self._sweep.get(id(group))
        self._of(group, self._shown)  # a run outside this set has no style either

        color = sweep.color if sweep else _variant_color(group.method.color, group.style_variant)

        if marker_by == "method":
            marker = group.method.marker
        elif sweep is not None:
            marker = sweep.marker
        elif group.is_fixed_lambda:
            marker = FIXED_LAMBDA_MARKER
        else:
            marker = SPARSITY_MARKERS.get(group.sparsity, "x")

        if sweep is not None:
            ls = sweep.linestyle
        elif group.style_variant is not None:
            ls = variant_for(group.style_variant).linestyle
        else:
            ls = SPARSITY_LINESTYLES.get(group.sparsity, "-")
        return color, marker, ls


if __name__ == "__main__":
    for token in ("bregman_linbreg_quantile_progressive", "pruning_snip_iter", "dense_sgd", "sv_dense_adamw", "soft_threshold"):
        m, v = method_for(token), flavor_for(token)
        print(f"{token:<40} -> {m.key:<18} {v.key if v else None}")
    try:
        method_for("pruning_str")
    except ValueError as e:
        print(f"unregistered name -> {e}")
