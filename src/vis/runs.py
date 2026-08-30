"""One experiment, all its seeds: discovery, name parsing and seed reduction.

A :class:`RunGroup` is one experiment. It holds every seed directory, what the
run name says about the method, and the readers that turn those directories into
numbers. Every cross-seed mean in the project goes through one of its three
reducers, so a curve, a summary bar and a printed table cannot disagree about
what "the mean over seeds" means.

Two directory layouts reach the same object::

    SV:    <base>/<exp>
    Image: <base>/<dataset>/<model>/<augmentation>/<exp>/seed_<N>

Run it with::

    python -m src.vis.runs <base_dir> '<glob>'
"""

import glob
import json
import os
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.vis.encoding import (
    METHOD_BY_KEY,
    METHOD_SORT_RANK,
    TIER_ADAPTIVE,
    TIER_DENSE,
    TIER_FIXED,
    TIER_SPARSE_BASELINE,
    Method,
    Variant,
    flavor_for,
    method_for,
)
from src.vis.metrics import resolve_lr_column

# A lambda value, exactly as run_naming.lambda_token spells it with "%g".
LAMBDA_VALUE_RE = r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"

# A name that ends in one of these tags carries no scheduler tag. The test reads
# the whole name, because "%g" puts a "-" inside an exponent: -lam1e-05.
TRAILING_TAG_RE = re.compile(rf"-(i?sr\d+|lam{LAMBDA_VALUE_RE})$")

# Per-seed run dirs are named "seed_<N>" (see scripts/fabfile.py:run_img).
SEED_RE = re.compile(r"^seed_?(\d+)$")

# Either marker makes a directory a leaf run, and the walk stops descending.
RUN_ARTIFACT_MARKERS = ("config_tree.log", "train_log.txt")

# Artifact subdirs that never contain a nested run — pruned while walking.
WALK_SKIP_DIRS = {"checkpoints", "csv", "metadata", "tensorboard", "test_artifacts", ".hydra"}

# The criterion names its own logged column, so the loss reaches the metric
# registry under one name whichever task wrote it. No run logs both criteria.
CSV_COLUMN_ALIASES = {
    "train/LogSoftmaxWrapper": "train_loss",
    "valid/LogSoftmaxWrapper": "valid_loss",
    "train/CrossEntropyLoss": "train_loss",
    "valid/CrossEntropyLoss": "valid_loss",
}


# ---------------------------------------------------------------------------
# 1. The experiment
# ---------------------------------------------------------------------------


@dataclass
class RunGroup:
    """One experiment and every seed of it.

    ``dirs`` holds one directory per seed, ordered by seed number; ``dirs[0]`` is
    the representative a single-run reader takes. A group parsed from a CSV
    leaderboard row has no directories, so it can be labelled and styled but not
    read.

    ``sparsity`` is the percent that keys the marker and dash tables. It is the
    target where the name states one. A fixed-lambda name states none, so the
    rounded ``landed_sparsity`` stands in and ``sparsity_is_outcome`` says so.
    """

    dirname: str
    method: Method
    flavor: Optional[Variant] = None
    variant: Optional[str] = None
    dirs: List[str] = field(default_factory=list)
    seeds: List[Optional[int]] = field(default_factory=list)
    sparsity: Optional[int] = None
    initial_sparsity: Optional[int] = None
    fixed_lambda: Optional[float] = None
    landed_sparsity: Optional[float] = None
    landed_sparsity_std: Optional[float] = None
    sparsity_is_outcome: bool = False
    dataset: Optional[str] = None
    model: Optional[str] = None
    augmentation: Optional[bool] = None
    scheduler: Optional[str] = None
    alpha: Optional[float] = 1.0
    f: Optional[int] = 50

    def __post_init__(self):
        """Open the per-source frame cache. It is not a field, so ``asdict`` and
        ``replace`` leave it out and a copy starts empty."""
        self._frames = {}

    # -- identity ----------------------------------------------------------

    @property
    def is_dense(self):
        """Whether this run trains dense. A run is dense because of its method,
        never because its name lacks a sparsity tag — a fixed-lambda name lacks
        one too."""
        return self.method.family == "dense"

    @property
    def is_fixed_lambda(self):
        """Whether this run holds lambda static, per its method token."""
        return "fixed" in ((self.flavor.key if self.flavor else None), self.variant)

    @property
    def style_variant(self):
        """The key that drives the dash pattern and the hue shift: the ad hoc
        name suffix where there is one, else the method flavor."""
        return self.variant or (self.flavor.key if self.flavor else None)

    @property
    def tier(self):
        """Which bar group this run belongs to (see encoding.TIER_LABELS).

        Anything sparse that is not Bregman is a baseline, so pruning — and RigL,
        SET and friends — group together without needing to be listed.
        """
        if self.is_dense:
            return TIER_DENSE
        if self.method.family != "bregman":
            return TIER_SPARSE_BASELINE
        return TIER_FIXED if self.is_fixed_lambda else TIER_ADAPTIVE

    @property
    def sort_key(self):
        """Place a run in its bar group and order it within.

        Groups read dense, sparse baselines, fixed lambda, adaptive lambda; inside
        a group runs sort by method, so LinBreg and AdaBreg stay in blocks. Lambda
        then orders the fixed-lambda runs and the sparsity level orders the rest —
        one key serves both, since lambda is constant where it is unset.
        """
        return (
            self.tier,
            METHOD_SORT_RANK[self.method.key],
            self.fixed_lambda if self.fixed_lambda is not None else -1.0,
            self.sparsity if self.sparsity is not None else -1,
            self.variant or "",
            self.alpha if self.alpha is not None else -1.0,
            self.f if self.f is not None else -1,
        )

    # -- reading the seeds -------------------------------------------------

    def frames(self, source):
        """Every seed's metrics frame, skipping seeds that wrote none.

        Cached per source. A multi-panel figure asks for one metric at a time, so
        without the cache every panel re-reads every seed's metrics.csv.
        """
        if source not in self._frames:
            self._frames[source] = [df for df in (load_run_df(d, source) for d in self.dirs) if df is not None]
        return self._frames[source]

    def per_seed(self, read):
        """``read(seed_dir)`` over every seed, dropping the seeds that return None.

        The one door for a reader that is not a metrics column — a checkpoint, a
        mask, a results file.
        """
        return [v for v in (read(d) for d in self.dirs) if v is not None]

    def curve(self, metric, source):
        """``(x, mean, std)`` over seeds for one metric, or None.

        Seeds are outer-joined on their x-index, so a seed missing an x sits out
        of that point rather than truncating the curve. ``std`` is None where one
        seed contributes, so the caller draws no band.
        """
        series = [s for s in (self.series(df, metric, source) for df in self.frames(source)) if s is not None]
        if not series:
            return None
        frame = pd.concat(series, axis=1)
        mean = frame.mean(axis=1)
        std = frame.std(axis=1).to_numpy() if frame.shape[1] > 1 else None
        return mean.index.to_numpy(), mean.to_numpy(), std

    def scalar(self, metric, source, reduce="max"):
        """``(mean, std, per_seed_values)`` for one metric, or None.

        ``reduce="max"`` takes each seed's best epoch. That is the selected
        checkpoint for the monitored metric only; another metric peaks at another
        epoch. ``"last"`` takes the final epoch. ``std`` is 0.0 for one seed.
        """
        assert reduce in ("max", "last"), f"reduce is max or last, got {reduce!r}"
        vals = []
        for df in self.frames(source):
            if metric not in df.columns:
                continue
            col = df[metric].dropna()
            if col.empty:
                continue
            vals.append(float(col.max() if reduce == "max" else col.iloc[-1]))
        if not vals:
            return None
        arr = np.array(vals, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, vals

    def series(self, df, metric, source):
        """One seed's metric as a Series indexed by epoch or step, or None.

        A fixed-lambda run never logs ``bregman/global_lambda`` — the scheduler
        does not run — so the constant it holds is synthesized here instead.
        """
        x_col = "epoch" if source == "train_log" else "step"
        if x_col not in df.columns:
            return None

        col = resolve_lr_column(df, self.method.key, self.dirname) if metric == "lr" else metric
        if col is None:
            return None
        if col == "bregman/global_lambda" and col not in df.columns and self.fixed_lambda is not None:
            df = df.assign(**{col: self.fixed_lambda})
        if col not in df.columns:
            return None

        x = df[x_col].astype(float)
        if source == "csv":
            x = x / 1000.0
        y = df[col]
        mask = y.notna()
        if mask.sum() == 0:
            return None
        s = pd.Series(y[mask].to_numpy(), index=x[mask].to_numpy())
        return s[~s.index.duplicated(keep="last")].sort_index()

    # -- construction ------------------------------------------------------

    @classmethod
    def from_name(cls, dirname):
        """Parse a run-directory name into a group with no directories attached.

        Two name shapes::

            SV:    sv_<method>-wespeaker_<backbone>-…[-isr<NN>][-sr<NN>|-lam<V>][-tag]
            Image: <method>[-isr<NN>][-sr<NN>|-lam<V>]-<scheduler>, or the fabfile
                   <method>-<model>-<dataset>-bs<NN>[-isr<NN>][-sr<NN>|-lam<V>][-tag]

        The name is the one source for the lambda. scripts/fabfile.py and the
        config both call bregman_utils.get_bregman_lambda, so the ``-lam`` token
        holds the value the run trained with. A fixed-lambda name that carries no
        ``-lam`` predates that spelling; scripts/retag_fixed_lambda_runs.py reads
        the lambda out of such a run's own Hydra snapshot and renames it.
        """
        fields = _parse_sv_name(dirname) if "-wespeaker" in dirname else _parse_image_name(dirname)
        token = fields.pop("token")
        flavor = flavor_for(token)
        group = cls(
            dirname=dirname,
            method=method_for(token),
            flavor=flavor,
            variant=fields.pop("tag", None) or (flavor.key if flavor else None),
            **fields,
        )
        if group.is_fixed_lambda and group.fixed_lambda is None:
            raise ValueError(
                f"a fixed-lambda run name carries -lam<value> (run scripts/retag_fixed_lambda_runs.py), got {dirname!r}"
            )
        return group

    def without_sparsity(self):
        """A copy that states no sparsity level, for a figure whose axis already
        carries it."""
        return replace(self, sparsity=None)


# ---------------------------------------------------------------------------
# 2. Name parsing
# ---------------------------------------------------------------------------


def _parse_image_name(name):
    """Image run name — two layouts, told apart by the ``-bs<NN>`` batch tag.

    The run_subdir layout reads dataset and model from the parent dirs; the
    fabfile layout embeds them, because the curated tree has no augmentation dir
    to read them from. ``isr`` is the starting sparsity, ``sr`` the target. A
    fixed-lambda run carries ``-lam<V>`` in place of a target: its sparsity is an
    outcome of that lambda, so only the realized value is meaningful. Whatever
    the fabfile appends after those tokens — a weight decay, a ramp end, a
    launcher suffix — is the ad hoc tag, and it is what parts two runs that
    differ in nothing else.
    """
    out = {}
    m_bs = re.search(r"-bs\d+", name)
    if m_bs:  # fabfile layout: the batch tag splits <method>-<model>-<dataset> from the rest
        head = name[: m_bs.start()].split("-")
        assert len(head) >= 3, f"a fabfile image name reads <method>-<model>-<dataset>-bs<NN>…, got {name!r}"
        out["dataset"], out["model"] = head[-1], head[-2]
        tail = name[m_bs.end() :]
        for key, pattern, cast in (
            ("initial_sparsity", r"-isr(\d+)", int),
            ("fixed_lambda", rf"-lam({LAMBDA_VALUE_RE})", float),
            ("sparsity", r"-sr(\d+)", int),
        ):
            m = re.search(pattern, tail)
            if m:
                out[key] = cast(m.group(1))
                tail = tail[: m.start()] + tail[m.end() :]
        if tail.lstrip("-"):
            out["tag"] = tail.lstrip("-")
        out["token"] = "-".join(head[:-2])
        return out

    method, sep, sched = name.rpartition("-")
    # A trailing scheduler tag: CosineAnnealing, no_scheduler, …
    if sep and not TRAILING_TAG_RE.search(name):
        out["scheduler"] = sched
        name = method
    for key, pattern, cast in (
        ("sparsity", r"-sr(\d+)$", int),
        ("fixed_lambda", rf"-lam({LAMBDA_VALUE_RE})$", float),
        ("initial_sparsity", r"-isr(\d+)$", int),
    ):
        m = re.search(pattern, name)
        if m:
            out[key] = cast(m.group(1))
            name = name[: m.start()]
    out["token"] = name
    return out


def _parse_sv_name(name):
    """SV run name ``sv_<method>-wespeaker_<backbone>-<dataset>-…-sr<NN>[-tag]``.

    The alpha, f and reg-style suffixes come off first, so they never leak into
    the ad hoc tag.
    """
    out = {}
    m_f = re.search(r"-f(\d+)$", name)
    if m_f:
        out["f"] = int(m_f.group(1))
        name = name[: m_f.start()]
    m_alpha = re.search(r"-alpha([\d.]+)$", name)
    if m_alpha:
        out["alpha"] = float(m_alpha.group(1))
        name = name[: m_alpha.start()]
    name = re.sub(r"-regl[12]\w*$", "", name)

    m_isr = re.search(r"-isr(\d+)", name)
    if m_isr:
        out["initial_sparsity"] = int(m_isr.group(1))
    m_lam = re.search(rf"-lam({LAMBDA_VALUE_RE})(?:-(.+))?$", name)
    if m_lam:
        out["fixed_lambda"] = float(m_lam.group(1))
        if m_lam.group(2):
            out["tag"] = m_lam.group(2)
    m_sr = re.search(r"-(sr|sparsity)(\d+)(?:-(.+))?$", name)
    if m_sr:
        out["sparsity"] = int(m_sr.group(2))
        if m_sr.group(3):
            out["tag"] = m_sr.group(3)  # e.g. "cls_scale2", "poor_init"

    m_model = re.search(r"-(wespeaker_\w+)-", name)
    out["model"] = m_model.group(1) if m_model else "unknown"
    if m_model:
        m_ds = re.match(r"([^-]+)-bs\d+", name[m_model.end() :])
        if m_ds:
            out["dataset"] = m_ds.group(1)
    out["token"] = name.split("-wespeaker")[0]
    return out


# ---------------------------------------------------------------------------
# 3. Readers
# ---------------------------------------------------------------------------


def read_landed_sparsity(seed_dir):
    """Pruned sparsity (fraction 0–1) at this seed's tested checkpoint, or None.

    ``pruned_sparsity`` covers every weight tensor, norms and biases aside. That is
    the figure the benchmark compares (``docs/image_benchmarks.md``).
    ``overall_sparsity`` dilutes it with the unprunable parameters and would place
    a run below where its method put it.

    Each task writes it against the checkpoint it tested. ``src/modules/img.py``
    writes ``results.json`` for the epoch the monitor selected;
    ``src/modules/sv.py`` writes it into every test set's metrics JSON. A run that
    never tested has neither file and reports no landed sparsity.
    """
    path = os.path.join(seed_dir, "results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["best_checkpoint"]["pruned_sparsity"]
    return _sparsity_from_test_metrics(seed_dir)


def _sparsity_from_test_metrics(seed_dir):
    """Pruned sparsity out of this seed's newest test-metrics JSON, or None.

    Every test set of one run scores the same weights, so the newest file answers
    for all of them. A file written before ``sv.py`` reported its sparsity raises:
    re-run the test rather than plot the wrong quantity.
    """
    paths = glob.glob(os.path.join(seed_dir, "test_artifacts", "*", "*", "*_metrics.json"))
    if not paths:
        return None
    newest = max(paths, key=lambda p: os.path.basename(os.path.dirname(p)))
    with open(newest) as f:
        metrics = json.load(f)
    if "pruned_sparsity" not in metrics:
        raise KeyError(f"a test-metrics JSON states pruned_sparsity (re-run test with +module.force_retest=true), missing in {newest}")
    return metrics["pruned_sparsity"]


def load_train_log(exp_dir):
    """Epoch-level metrics from train_log.txt → DataFrame."""
    path = os.path.join(exp_dir, "train_log.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected train_log.txt not found in {exp_dir}")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = {}
            for pair in line.split(", "):
                k, v = pair.split(": ", 1)
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            rows.append(row)
    return pd.DataFrame(rows) if rows else None


def load_csv_metrics(exp_dir):
    """Step-level metrics from csv/version_*/metrics.csv → DataFrame.

    Lightning's CSVLogger opens a fresh ``version_N`` on every run or resume. All
    non-empty versions merge into one continuous curve in index order, which is
    the order they were written; where step ranges overlap the highest version
    wins, because it superseded the earlier one. A name that states no index
    raises: it has no place in that order.
    """
    csv_root = os.path.join(exp_dir, "csv")
    if not os.path.isdir(csv_root):
        return None

    version_files = []
    for entry in sorted(os.listdir(csv_root)):
        m = re.match(r"version_(\d+)$", entry)
        if not m:
            raise ValueError(f"csv/ holds version_<N> directories only, got {entry!r} in {csv_root}")
        path = os.path.join(csv_root, entry, "metrics.csv")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            version_files.append((int(m.group(1)), path))
    if not version_files:
        return None
    version_files.sort(key=lambda t: t[0])

    dfs = []
    for vidx, vpath in version_files:
        df_v = pd.read_csv(vpath)
        if df_v.empty:
            continue
        df_v["__version__"] = vidx
        dfs.append(df_v)
    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True, sort=False)
    if "step" in df.columns:
        df = df.sort_values(["step", "__version__"]).groupby("step", as_index=False).last()
        if len(dfs) > 1:
            _warn_on_version_discontinuity(df, exp_dir)
    df.drop(columns=["__version__"], errors="ignore", inplace=True)
    df.rename(columns=CSV_COLUMN_ALIASES, inplace=True)
    return df


def _warn_on_version_discontinuity(df, exp_dir, window=5, rel_tol=0.5):
    """Report a metric that jumps at a version boundary. A resume that reloaded
    the wrong checkpoint shows up here and nowhere else."""
    if "step" not in df.columns or "__version__" not in df.columns:
        return
    sdf = df.sort_values("step").reset_index(drop=True)
    boundaries = [i for i in range(1, len(sdf)) if sdf.loc[i, "__version__"] > sdf.loc[i - 1, "__version__"]]
    skip = {"step", "epoch", "__version__"}
    cols = [c for c in sdf.columns if c not in skip and pd.api.types.is_numeric_dtype(sdf[c])]

    for b in boundaries:
        flagged = []
        for c in cols:
            before = sdf.iloc[max(0, b - window) : b][c].dropna()
            after = sdf.iloc[b : b + window][c].dropna()
            if len(before) < 2 or len(after) < 2:
                continue
            a, z = float(before.mean()), float(after.mean())
            if not (np.isfinite(a) and np.isfinite(z)):
                continue
            if abs(z - a) / max(abs(a), abs(z), 1e-9) > rel_tol:
                flagged.append((c, a, z))
        if not flagged:
            continue
        print(
            f"Warning: discontinuity across version_{int(sdf.loc[b - 1, '__version__'])}→"
            f"version_{int(sdf.loc[b, '__version__'])} boundary (step ~{int(sdf.loc[b, 'step'])}) in {exp_dir}"
        )
        for c, a, z in flagged[:8]:
            print(f"    {c}: {a:.4g} → {z:.4g}")
        if len(flagged) > 8:
            print(f"    ... and {len(flagged) - 8} more")


def load_run_df(run_dir, source):
    """One seed's metrics frame from the chosen source."""
    assert source in ("train_log", "csv"), f"source is train_log or csv, got {source!r}"
    return load_train_log(run_dir) if source == "train_log" else load_csv_metrics(run_dir)


# ---------------------------------------------------------------------------
# 4. Discovery
# ---------------------------------------------------------------------------


def _find_run_dirs(base_dir):
    """Every leaf run directory under ``base_dir``, at any nesting depth.

    SV runs sit one level down, image runs sit five. Both bottom out in a dir
    holding the training artifacts, so the walk stops at the first one it finds.
    """
    for root, dirs, _ in os.walk(base_dir):
        if any(os.path.exists(os.path.join(root, m)) for m in RUN_ARTIFACT_MARKERS):
            dirs[:] = []  # a run dir has no nested runs
            yield root
            continue
        dirs[:] = [d for d in dirs if d not in WALK_SKIP_DIRS and not d.endswith("_artifacts")]


def _path_metadata(exp_dir):
    """Dataset, model and augmentation from an image run's parent dirs.

    The augmentation dir has a fixed name, so anchor on it — this holds however
    deep ``base_dir`` points. A flat SV run has no such dir and returns {}.
    """
    parts = os.path.normpath(exp_dir).split(os.sep)
    for i, name in enumerate(parts):
        if name in ("augmentation", "no_augmentation") and i >= 2:
            return {"dataset": parts[i - 2], "model": parts[i - 1], "augmentation": name == "augmentation"}
    return {}


def discover(base_dirs, patterns):
    """Every experiment under ``base_dirs`` whose name matches a glob pattern.

    An experiment is one run dir (SV) or one ``<exp>`` dir whose ``seed_<N>``
    subdirs are grouped (image). Grouping keys on the ``<exp>`` directory *path*,
    so same-named experiments under different dataset/model/augmentation parents
    stay distinct. Patterns match the ``<exp>`` directory name in both layouts.

    ``landed_sparsity`` and its spread are the mean and sample deviation of the
    pruned sparsity each seed's selected checkpoint reached. Every other number in
    these figures is a cross-seed mean, so this one is too.
    """
    if isinstance(base_dirs, str):
        base_dirs = [base_dirs]

    buckets: Dict[str, Dict[str, Any]] = {}
    for base_dir in base_dirs:
        if not os.path.isdir(base_dir):
            raise ValueError(f"Warning: base dir does not exist: {base_dir}")
        for run_dir in _find_run_dirs(base_dir):
            m_seed = SEED_RE.match(os.path.basename(run_dir))
            seed = int(m_seed.group(1)) if m_seed else None
            exp_dir = os.path.dirname(run_dir) if m_seed else run_dir
            # Older runs kept the launcher's ".yaml" in the method token.
            exp_name = re.sub(r"\.yaml(?=-|$)", "", os.path.basename(exp_dir))
            if not any(glob.fnmatch.fnmatch(exp_name, p) for p in patterns):
                continue
            b = buckets.setdefault(exp_dir, {"name": exp_name, "dirs": [], "seeds": []})
            if run_dir in b["dirs"]:
                continue
            b["dirs"].append(run_dir)
            b["seeds"].append(seed)

    groups = []
    for exp_dir, b in buckets.items():
        g = RunGroup.from_name(b["name"])
        for key, value in _path_metadata(exp_dir).items():
            setattr(g, key, value)
        # Order seed dirs by seed number (an unseeded SV run sorts as -1).
        pairs = sorted(zip(b["seeds"], b["dirs"]), key=lambda t: (-1 if t[0] is None else t[0]))
        g.seeds = [s for s, _ in pairs]
        g.dirs = [d for _, d in pairs]
        landed = g.per_seed(read_landed_sparsity)
        g.landed_sparsity = float(np.mean(landed)) if landed else None
        g.landed_sparsity_std = float(np.std(landed, ddof=1)) if len(landed) > 1 else None
        # A fixed-lambda name carries no target, so where it landed is its level.
        if g.sparsity is None and not g.is_dense and g.landed_sparsity is not None:
            g.sparsity = round(g.landed_sparsity * 100)
            g.sparsity_is_outcome = True
        groups.append(g)

    groups.sort(key=lambda g: g.sort_key)
    return groups


def resolve_sparsity_level(df):
    """Fill each sparse row's ``sparsity`` (percent) from its realized value.

    A ``-lam<V>`` name carries no target, yet labels and the trend x-axis key on
    ``sparsity``. Rounding to the nearest percent keeps the marker and dash tables
    resolving. A dense row keeps a null sparsity — that is what marks it dense.
    """
    dense = df["method_class"].isin({k for k, m in METHOD_BY_KEY.items() if m.family == "dense"})
    missing = df["sparsity"].isna() & ~dense
    df.loc[missing, "sparsity"] = (pd.to_numeric(df.loc[missing, "actual_sparsity"], errors="coerce") * 100).round()
    unresolved = int((df["sparsity"].isna() & ~dense).sum())
    if unresolved:
        print(
            f"  [warn] {unresolved} sparse rows have neither a target nor a realized "
            "sparsity; pass --base_dirs or they drop out of sparsity-keyed plots"
        )
    return df


if __name__ == "__main__":
    import sys

    from src.vis.encoding import Encoding

    groups = discover(sys.argv[1], sys.argv[2:] or ["*"])
    enc = Encoding(groups)
    for g in groups:
        print(f"{g.dirname[:60]:<60} {len(g.dirs)} seed(s)  {enc.label(g)}")
