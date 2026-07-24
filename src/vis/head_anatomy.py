"""Reader and renderer functions for the classifier-head anatomy diagnostic.

Answers one question: why does an over-sized head trained with AdaBreg keep a
single bright column (a "stripe") across every phantom row? The stripe column
is traced back to the layers that feed the head — the last norm layer
(BatchNorm or LayerNorm), the adaptive average pool, and the head's own weight
and bias.

Reader functions load raw data from checkpoints (and optionally a forward pass)
and return plain dicts — no matplotlib. Renderer functions draw to
caller-provided axes — no file I/O.

Examples
--------
>>> import matplotlib.pyplot as plt
>>> from src.vis.head_anatomy import read_head_anatomy, render_norm_shift_vs_stripe
>>> anat = read_head_anatomy("/path/to/last.ckpt", n_real=10)  # doctest: +SKIP
>>> fig, axes = plt.subplots(3, 1)  # doctest: +SKIP
>>> render_norm_shift_vs_stripe(anat, axes)  # doctest: +SKIP
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_MODULE_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

log = logging.getLogger(__name__)

ZERO_TOL = 1e-12  # |w| below this → treated as pruned-to-zero
BN_EPS = 1e-5  # torch.nn.BatchNorm2d default

REAL_COLOR = "#1f77b4"
PHANTOM_COLOR = "#d62728"
STRIPE_COLOR = "#2A662B"
NEUTRAL_COLOR = "#8c8c8c"


# ---------------------------------------------------------------------------
# Checkpoint readers
# ---------------------------------------------------------------------------


def find_checkpoints(run_dir: str, limit: Optional[int] = None) -> List[str]:
    """Return the run's checkpoints, oldest first, truncated to the newest
    ``limit``.

    Args:
        run_dir: seed directory holding a ``checkpoints/`` subfolder.
        limit: keep only the ``limit`` most recent checkpoints; None keeps all.

    Returns:
        Checkpoint paths ordered by the epoch stored inside each file.
    """
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    assert os.path.isdir(ckpt_dir), f"no checkpoints/ under {run_dir}"
    files = [
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.endswith(".ckpt")
    ]
    assert files, f"no .ckpt files in {ckpt_dir}"
    files.sort(key=lambda p: _read_epoch(p))
    return files if limit is None else files[-limit:]


def _read_epoch(ckpt_path: str) -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "epoch" in ckpt, f"{ckpt_path} is not a Lightning checkpoint"
    return int(ckpt["epoch"])


def real_class_count(run_dir: str, repo_root: str) -> int:
    """Number of classes the dataset actually has, ignoring head inflation.

    The run config carries the *inflated* ``datamodule.num_classes``, so the
    true count is read from the dataset's own config in the repo, keyed on the
    dataset tag recorded in the run.

    Args:
        run_dir: seed directory holding ``.hydra/config.yaml``.
        repo_root: repository root holding ``configs/``.

    Returns:
        Number of real classes.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    name = cfg.datamodule.name
    ds_path = os.path.join(
        repo_root, "configs", "datamodule", "datasets", f"{name}.yaml"
    )
    assert os.path.exists(
        ds_path
    ), f"no dataset config for {name!r}: {ds_path}"
    n = OmegaConf.load(ds_path).num_classes
    assert isinstance(n, int), f"{ds_path}: num_classes must be a literal int"
    return n


def _dense_weight(sd: Dict[str, Any], base: str) -> torch.Tensor:
    """Materialize a weight whether or not a pruning hook is attached."""
    if base + "_orig" in sd:
        return sd[base + "_orig"].float() * sd[base + "_mask"].float()
    return sd[base].float()


def _fold_pruning_hooks(
    sd: Dict[str, Any], prefix: str
) -> Dict[str, torch.Tensor]:
    """Strip ``prefix`` and bake ``{p}_orig * {p}_mask`` pairs back into
    ``{p}``.

    A pruning-hooked checkpoint stores the mask separately; the bare network
    expects the masked weight, so fold before loading it.
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if not k.startswith(prefix):
            continue
        name = k[len(prefix) :]
        if name.endswith("_mask"):
            continue
        if name.endswith("_orig"):
            base = name[: -len("_orig")]
            out[base] = v.float() * sd[prefix + base + "_mask"].float()
        else:
            out[name] = v
    return out


def _find_head(sd: Dict[str, Any]) -> str:
    """Return the parameter base name of the last 2-D weight (the head)."""
    names = [
        k.replace("_orig", "")
        for k, v in sd.items()
        if isinstance(v, torch.Tensor)
        and k.endswith(("weight", "weight_orig"))
    ]
    two_d = [n for n in names if _dense_weight(sd, n).ndim == 2]
    assert two_d, "no 2-D weight in state_dict — cannot locate the head"
    return two_d[-1]


def _find_last_norm(sd: Dict[str, Any], n_features: int) -> str:
    """Return the module prefix of the last norm layer with ``n_features``.

    A norm layer is a 1-D ``weight``/``bias`` pair, which matches BatchNorm and
    LayerNorm alike; "last" in state-dict order is the one whose output feeds
    the pool and then the head.
    """
    hits = [
        k[: -len(".weight")]
        for k, v in sd.items()
        if k.endswith(".weight")
        and isinstance(v, torch.Tensor)
        and v.ndim == 1
        and v.numel() == n_features
        and k[: -len(".weight")] + ".bias" in sd
    ]
    assert hits, f"no norm layer with {n_features} features in state_dict"
    return hits[-1]


def _dead_offset(
    sd: Dict[str, Any], prefix: str, gamma: torch.Tensor, beta: torch.Tensor
) -> Optional[torch.Tensor]:
    """BatchNorm output for a channel whose input is exactly zero.

    LayerNorm normalizes across channels, so a channel's output depends on the
    others and no such per-channel constant exists — hence None there.
    """
    if prefix + ".running_mean" not in sd:
        return None
    mean = sd[prefix + ".running_mean"].float()
    var = sd[prefix + ".running_var"].float()
    return beta - gamma * mean / torch.sqrt(var + BN_EPS)


def read_head_anatomy(ckpt_path: str, n_real: int) -> Dict:
    """Head weight/bias structure plus the norm layer that feeds it.

    Rows ``[:n_real]`` are real classes; the rest are phantoms — classes the
    head can emit but the data never labels. The stripe column is the head
    column holding the largest share of surviving phantom weight.

    Args:
        ckpt_path: path to a Lightning ``.ckpt``.
        n_real: number of real classes.

    Returns:
        Dict with the head split (``real_*`` / ``phantom_*``), the stripe
        column, the feeding norm layer's per-channel ``beta``/``gamma``, and a
        per-norm-layer inventory.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]

    head = _find_head(sd)
    W = _dense_weight(sd, head)
    n_cls, dim = W.shape
    assert (
        n_real < n_cls
    ), f"n_real={n_real} but the head has only {n_cls} rows"
    bias_key = head.replace(".weight", ".bias")
    assert bias_key in sd, f"{head} has no bias — this diagnostic needs one"
    b = sd[bias_key].float()

    alive = W.abs() >= ZERO_TOL
    real, phantom = alive[:n_real], alive[n_real:]
    phantom_energy = (W[n_real:] ** 2).sum(0)
    total_energy = float(phantom_energy.sum())
    stripe_col = int(phantom_energy.argmax()) if total_energy > 0 else None

    norm = _find_last_norm(sd, dim)
    gamma = sd[norm + ".weight"].float()
    beta = sd[norm + ".bias"].float()
    dead_offset = _dead_offset(sd, norm, gamma, beta)

    return dict(
        ckpt=ckpt_path,
        epoch=int(ckpt["epoch"]),
        head_name=head,
        head_shape=(n_cls, dim),
        n_real=n_real,
        n_phantom=n_cls - n_real,
        real_nnz_per_row=real.sum(1).numpy(),
        phantom_nnz_per_row=phantom.sum(1).numpy(),
        real_cols_used=real.sum(0).numpy(),
        phantom_cols_used=phantom.sum(0).numpy(),
        phantom_col_energy=phantom_energy.numpy(),
        stripe_col=stripe_col,
        stripe_energy_share=(
            float(phantom_energy.max()) / total_energy
            if total_energy > 0
            else float("nan")
        ),
        stripe_weight_mean=(
            float(W[n_real:, stripe_col].mean())
            if stripe_col is not None
            else float("nan")
        ),
        dead_phantom_rows=int((phantom.sum(1) == 0).sum()),
        real_bias=b[:n_real].numpy(),
        phantom_bias=b[n_real:].numpy(),
        norm_name=norm,
        norm_gamma=gamma.numpy(),
        norm_beta=beta.numpy(),
        norm_dead_offset=(
            None if dead_offset is None else dead_offset.numpy()
        ),
        norm_inventory=_norm_inventory(sd),
        conv_dead_channels=_conv_dead_channels(sd),
    )


def _norm_inventory(sd: Dict[str, Any]) -> List[Dict]:
    """Per-norm-layer scale/shift norms, in model order."""
    out = []
    for k, v in sd.items():
        if not k.endswith(".weight") or not isinstance(v, torch.Tensor):
            continue
        pre = k[: -len(".weight")]
        if v.ndim != 1 or pre + ".bias" not in sd:
            continue
        gamma, beta = v.float(), sd[pre + ".bias"].float()
        out.append(
            dict(
                name=pre,
                n=int(gamma.numel()),
                gamma_l2=float(gamma.norm()),
                beta_l2=float(beta.norm()),
                gamma_absmax=float(gamma.abs().max()),
                beta_max=float(beta.max()),
            )
        )
    return out


def _conv_dead_channels(sd: Dict[str, Any]) -> List[Dict]:
    """Per-conv-layer count of output channels pruned entirely to zero."""
    out = []
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor) or not k.endswith(
            ("weight", "weight_orig")
        ):
            continue
        base = k.replace("_orig", "")
        w = _dense_weight(sd, base)
        if w.ndim != 4:
            continue
        nnz = (w.abs() >= ZERO_TOL).flatten(1).sum(1)
        out.append(
            dict(
                name=base,
                n_out=int(nnz.numel()),
                dead=int((nnz == 0).sum()),
                sparsity=float((w.abs() < ZERO_TOL).float().mean()),
            )
        )
    return out


def read_pooled_features(
    run_dir: str,
    ckpt_path: str,
    data_dir: str,
    *,
    n_batches: int = 16,
    device: str = "cpu",
) -> Dict:
    """Statistics of the adaptive-pool output — the vector the head consumes.

    Rebuilds the run's network and test loader from its saved Hydra config,
    then hooks the pooling layer. Adaptive average pooling has no parameters,
    so its per-channel output statistics are the only thing to look at.

    Args:
        run_dir: seed directory holding ``.hydra/config.yaml``.
        ckpt_path: checkpoint whose weights are loaded into the network.
        data_dir: local dataset root, replacing the training-time path.
        n_batches: number of test batches to accumulate.
        device: torch device string.

    Returns:
        Dict of per-channel arrays: ``mean``, ``std``, ``on_fraction``
        (share of samples where the feature is positive), ``sign_consistency``
        (mean divided by root-mean-square — 1 means the feature never varies),
        and the pre-pool spatial spread.
    """
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    assert os.path.isdir(data_dir), f"data_dir does not exist: {data_dir}"
    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    OmegaConf.update(cfg, "paths.data_dir", os.path.abspath(data_dir))

    dm = instantiate(
        OmegaConf.create(OmegaConf.to_container(cfg.datamodule, resolve=True)),
        _recursive_=False,
    )
    dm.setup("test")
    net = instantiate(
        OmegaConf.create(
            OmegaConf.to_container(cfg.module.model.net, resolve=True)
        )
    )
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)[
        "state_dict"
    ]
    net.load_state_dict(_fold_pruning_hooks(sd, "net."), strict=True)
    assert hasattr(net, "avgpool"), "network has no `avgpool` to hook"
    net.eval().to(device)

    pooled: List[torch.Tensor] = []
    spatial: List[torch.Tensor] = []

    def hook(module, inputs, output):
        pooled.append(output.flatten(1).cpu())
        spatial.append(inputs[0].flatten(2).std(dim=2).cpu())

    handle = net.avgpool.register_forward_hook(hook)
    with torch.no_grad():
        for i, (images, _) in enumerate(dm.test_dataloader()):
            if i >= n_batches:
                break
            net(images.to(device))
    handle.remove()

    X = torch.cat(pooled).float()
    S = torch.cat(spatial).float()
    mean = X.mean(0)
    rms = (X**2).mean(0).sqrt()
    return dict(
        n_samples=int(X.shape[0]),
        mean=mean.numpy(),
        std=X.std(0).numpy(),
        on_fraction=(X > 0).float().mean(0).numpy(),
        sign_consistency=(mean / rms.clamp_min(1e-12)).numpy(),
        spatial_std=S.mean(0).numpy(),
    )


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _rank_desc(values: np.ndarray, idx: int) -> int:
    """Number of entries strictly larger than ``values[idx]`` — 0 means top."""
    return int((values > values[idx]).sum())


def _ordinal(n: int) -> str:
    tail = (
        "th"
        if 10 <= n % 100 <= 20
        else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    )
    return f"{n}{tail}"


def _stripe_label(values: np.ndarray, idx: int, fmt: str = "{:.1f}") -> str:
    """``ch 356: 285.4, largest of 512`` — index, value, and where it ranks."""
    rank = _rank_desc(values, idx)
    place = "largest" if rank == 0 else f"{_ordinal(rank + 1)} largest"
    return f"ch {idx}: {fmt.format(values[idx])}, {place} of {len(values)}"


def _bar_with_stripe(
    ax: "plt.Axes",
    values: np.ndarray,
    col: Optional[int],
    *,
    fmt: str = "{:.1f}",
    base_color: str = NEUTRAL_COLOR,
) -> None:
    """One bar per channel, with the stripe channel recoloured and labelled.

    The stripe is marked on the bar itself rather than with a guide line, so
    nothing sits on top of the value the reader came to check.
    """
    ax.bar(np.arange(len(values)), values, width=1.0, color=base_color)
    if col is None:
        return
    ax.bar([col], [values[col]], width=1.0, color=STRIPE_COLOR, zorder=3)
    ax.plot(
        [col],
        [values[col]],
        marker="v",
        markersize=4,
        color=STRIPE_COLOR,
        zorder=4,
    )
    ax.annotate(
        _stripe_label(values, col, fmt),
        xy=(col, values[col]),
        xytext=(0.28 if col > len(values) / 2 else 0.72, 0.88),
        textcoords="axes fraction",
        fontsize=7,
        color=STRIPE_COLOR,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=STRIPE_COLOR, linewidth=0.8),
    )


def _mark_point(
    ax: "plt.Axes", x: float, y: float, text: str, text_xy: Tuple[float, float]
) -> None:
    """Star one scatter point and label it from a clear corner of the axes.

    ``text_xy`` is in axes fraction so the label can be parked away from the
    data; an arrow ties it back to the point.
    """
    ax.scatter([x], [y], s=60, marker="*", color=STRIPE_COLOR, zorder=4)
    ax.annotate(
        text,
        xy=(x, y),
        xytext=text_xy,
        textcoords="axes fraction",
        fontsize=7,
        color=STRIPE_COLOR,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=STRIPE_COLOR, linewidth=0.8),
    )


def render_norm_shift_vs_stripe(anat: Dict, axes: np.ndarray) -> None:
    """Align the feeding norm layer's per-channel affine with head-column
    usage.

    Three stacked panels over the same channel index: the norm shift added to
    the channel, the scale it is multiplied by, and how many phantom rows keep
    that head column. The claim to read off: one channel is the largest in the
    affine, and it is the one every phantom row keeps.

    Args:
        anat: dict from ``read_head_anatomy``.
        axes: 3 Axes, top to bottom.
    """
    beta, gamma = anat["norm_beta"], anat["norm_gamma"]
    used = anat["phantom_cols_used"]
    col = anat["stripe_col"]

    ax = axes[0]
    _bar_with_stripe(ax, beta, col)
    ax.set_ylabel("Shift added\n" + r"to the channel ($\beta$)")

    ax = axes[1]
    _bar_with_stripe(ax, np.abs(gamma), col)
    ax.set_ylabel("Scale the channel\n" + r"is multiplied by ($|\gamma|$)")

    ax = axes[2]
    _bar_with_stripe(
        ax, used.astype(float), col, fmt="{:.0f}", base_color=PHANTOM_COLOR
    )
    ax.set_ylabel("Phantom rows\nkeeping this column")
    ax.set_xlabel(
        f"Channel index — the {anat['norm_name']} output feeds the head"
    )
    ax.set_xlim(-1, len(beta))


def render_head_anatomy(anat: Dict, axes: np.ndarray) -> None:
    """Head weight and bias split by real versus phantom rows.

    Args:
        anat: dict from ``read_head_anatomy``.
        axes: 3 Axes — surviving weights per row, bias, per-column energy.
    """
    ax = axes[0]
    bins = np.arange(0, max(6, int(anat["real_nnz_per_row"].max()) + 2)) - 0.5
    ax.hist(
        anat["real_nnz_per_row"],
        bins=bins,
        color=REAL_COLOR,
        alpha=0.75,
        label=f"real ({anat['n_real']})",
        density=True,
    )
    ax.hist(
        anat["phantom_nnz_per_row"],
        bins=bins,
        color=PHANTOM_COLOR,
        alpha=0.6,
        label=f"phantom ({anat['n_phantom']})",
        density=True,
    )
    ax.set_xlabel("Surviving weights in the row")
    ax.set_ylabel("Share of rows")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    ax.hist(
        anat["real_bias"], bins=20, color=REAL_COLOR, alpha=0.75, density=True
    )
    ax.hist(
        anat["phantom_bias"],
        bins=40,
        color=PHANTOM_COLOR,
        alpha=0.6,
        density=True,
    )
    ax.axvline(0.0, **{"color": "#444444", "linestyle": ":", "linewidth": 0.8})
    ax.set_xlabel("Head bias")
    ax.set_ylabel("Share of rows")
    ax.set_title(
        f"phantom mean {anat['phantom_bias'].mean():.3f} vs "
        f"real {anat['real_bias'].mean():.3f}",
        fontsize=7,
    )

    # Disjoint axes here mean the stripe is not a real feature the phantoms hijacked.
    ax = axes[2]
    real_use, phantom_use = anat["real_cols_used"], anat["phantom_cols_used"]
    ax.scatter(
        real_use,
        phantom_use,
        s=8,
        color=NEUTRAL_COLOR,
        alpha=0.6,
        edgecolor="none",
    )
    col = anat["stripe_col"]
    if col is not None:
        _mark_point(
            ax,
            real_use[col],
            phantom_use[col],
            f"col {col}: kept by {phantom_use[col]} phantom rows,"
            f" {real_use[col]} real",
            (0.55, 0.55),
        )
    ax.set_xlabel("Real rows keeping the column")
    ax.set_ylabel("Phantom rows keeping the column")
    ax.set_title(
        f"top column holds {anat['stripe_energy_share']:.3f}"
        " of phantom weight",
        fontsize=7,
    )


def render_pooled_features(feat: Dict, anat: Dict, axes: np.ndarray) -> None:
    """Per-channel statistics of the vector the head actually sees.

    Separates the two candidate explanations for the stripe: feature *size*
    (left) versus how *consistently signed* the feature is (middle). Only the
    one that ranks the stripe column first explains it.

    Args:
        feat: dict from ``read_pooled_features``.
        anat: dict from ``read_head_anatomy``.
        axes: 3 Axes.
    """
    col = anat["stripe_col"]
    mean, r = feat["mean"], feat["sign_consistency"]
    idx = np.arange(len(mean))

    ax = axes[0]
    _bar_with_stripe(ax, mean, col)
    ax.set_ylabel("Average value\nentering the head")
    ax.set_xlabel("Channel index")

    ax = axes[1]
    ax.scatter(idx, r, s=4, color=NEUTRAL_COLOR, alpha=0.6, edgecolor="none")
    if col is not None:
        _mark_point(
            ax, col, r[col], _stripe_label(r, col, "{:.3f}"), (0.5, 0.12)
        )
    ax.set_ylabel("Sign consistency\n(1 = never changes sign)")
    ax.set_xlabel("Channel index")

    ax = axes[2]
    ax.scatter(
        feat["on_fraction"],
        np.clip(mean, 1e-3, None),
        s=6,
        color=NEUTRAL_COLOR,
        alpha=0.6,
        edgecolor="none",
    )
    if col is not None:
        _mark_point(
            ax,
            feat["on_fraction"][col],
            max(mean[col], 1e-3),
            f"ch {col}: above zero on {feat['on_fraction'][col]:.2f} of images",
            (0.4, 0.86),
        )
    ax.set_yscale("log")
    ax.set_xlabel("Share of test images where\nthe channel is above zero")
    ax.set_ylabel("Average value\nentering the head")


def render_norm_inventory(anat: Dict, axes: np.ndarray) -> None:
    """Where the norm-layer weight sits across the network, and conv channel
    death.

    Args:
        anat: dict from ``read_head_anatomy``.
        axes: 2 Axes.
    """
    inv = anat["norm_inventory"]
    names = [d["name"].replace("net.", "") for d in inv]
    pos = np.arange(len(inv))
    feeder = anat["norm_name"].replace("net.", "")

    ax = axes[0]
    ax.bar(
        pos - 0.2,
        [d["gamma_l2"] for d in inv],
        width=0.4,
        color=REAL_COLOR,
        label=r"$\|\gamma\|_2$",
    )
    ax.bar(
        pos + 0.2,
        [d["beta_l2"] for d in inv],
        width=0.4,
        color=PHANTOM_COLOR,
        label=r"$\|\beta\|_2$",
    )
    ax.set_yscale("log")
    ax.set_ylabel("Size of the layer's\nscale and shift (L2)")
    ax.set_xticks(pos)
    ax.set_xticklabels(names, rotation=90, fontsize=5)
    ax.legend(fontsize=7, frameon=False)
    # green tick label marks the layer whose output reaches the head
    ax.get_xticklabels()[names.index(feeder)].set_color(STRIPE_COLOR)

    conv = anat["conv_dead_channels"]
    ax = axes[1]
    cpos = np.arange(len(conv))
    ax.bar(
        cpos,
        [d["dead"] / d["n_out"] for d in conv],
        width=0.7,
        color=NEUTRAL_COLOR,
    )
    ax.set_ylabel("Share of output channels\nfully pruned")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(cpos)
    ax.set_xticklabels(
        [d["name"].replace("net.", "").replace(".weight", "") for d in conv],
        rotation=90,
        fontsize=5,
    )


def render_ckpt_evolution(series: List[Dict], axes: np.ndarray) -> None:
    """Track the stripe and the runaway norm channel across checkpoints.

    Args:
        series: list of ``read_head_anatomy`` dicts, oldest first.
        axes: 2 Axes.
    """
    epochs = [a["epoch"] for a in series]

    ax = axes[0]
    ax.plot(
        epochs,
        [a["norm_beta"].max() for a in series],
        marker="o",
        color=PHANTOM_COLOR,
        label=r"max $\beta$",
    )
    ax.set_ylabel(r"Largest norm shift $\beta$")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(
        epochs,
        [int(a["norm_beta"].argmax()) for a in series],
        marker="s",
        linestyle="--",
        color=STRIPE_COLOR,
        label="its channel",
    )
    # A run whose phantom rows all died has no stripe column to plot.
    stripe = [
        np.nan if a["stripe_col"] is None else a["stripe_col"] for a in series
    ]
    ax2.plot(
        epochs,
        stripe,
        marker="x",
        markersize=9,
        linestyle=":",
        color="#000000",
        label="stripe column",
    )
    ax2.set_ylabel("Channel index")
    ax2.grid(False)
    ax2.legend(fontsize=7, frameon=False, loc="lower right")

    ax = axes[1]
    ax.plot(
        epochs,
        [a["stripe_energy_share"] for a in series],
        marker="o",
        color=PHANTOM_COLOR,
        label="stripe share of phantom weight",
    )
    ax.plot(
        epochs,
        [a["dead_phantom_rows"] / a["n_phantom"] for a in series],
        marker="s",
        linestyle="--",
        color=REAL_COLOR,
        label="phantom rows fully dead",
    )
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("Share")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=7, frameon=False)


def summarize(anat: Dict, feat: Optional[Dict] = None) -> Dict:
    """Flatten the diagnostic to the scalars worth quoting in a doc.

    Args:
        anat: dict from ``read_head_anatomy``.
        feat: optional dict from ``read_pooled_features``.

    Returns:
        JSON-serializable dict of scalar findings.
    """
    col = anat["stripe_col"]
    beta, gamma = anat["norm_beta"], anat["norm_gamma"]
    out = dict(
        epoch=anat["epoch"],
        head=anat["head_name"],
        head_shape=list(anat["head_shape"]),
        n_real=anat["n_real"],
        n_phantom=anat["n_phantom"],
        dead_phantom_rows=anat["dead_phantom_rows"],
        phantom_nnz_per_row_mean=float(anat["phantom_nnz_per_row"].mean()),
        real_nnz_per_row_mean=float(anat["real_nnz_per_row"].mean()),
        phantom_bias_mean=float(anat["phantom_bias"].mean()),
        real_bias_mean=float(anat["real_bias"].mean()),
        stripe_col=col,
        stripe_energy_share=anat["stripe_energy_share"],
        stripe_weight_mean=anat["stripe_weight_mean"],
        norm_name=anat["norm_name"],
        norm_beta_argmax=int(beta.argmax()),
        norm_beta_max=float(beta.max()),
        norm_beta_median=float(np.median(beta)),
        norm_gamma_absmax=float(np.abs(gamma).max()),
        norm_gamma_absmedian=float(np.median(np.abs(gamma))),
    )
    if col is not None:
        out.update(
            stripe_beta=float(beta[col]),
            stripe_beta_rank=_rank_desc(beta, col),
            stripe_gamma=float(gamma[col]),
            stripe_gamma_rank=_rank_desc(np.abs(gamma), col),
            beta_argmax_is_stripe=bool(int(beta.argmax()) == col),
        )
        if anat["norm_dead_offset"] is not None:
            out["stripe_dead_offset"] = float(anat["norm_dead_offset"][col])
    if feat is not None:
        out.update(
            activation_samples=feat["n_samples"],
            pooled_mean_max=float(feat["mean"].max()),
            pooled_mean_median=float(np.median(feat["mean"])),
        )
        if col is not None:
            out.update(
                stripe_pooled_mean=float(feat["mean"][col]),
                stripe_pooled_mean_rank=_rank_desc(feat["mean"], col),
                stripe_on_fraction=float(feat["on_fraction"][col]),
                stripe_sign_consistency=float(feat["sign_consistency"][col]),
                stripe_sign_consistency_rank=_rank_desc(
                    feat["sign_consistency"], col
                ),
            )
    return out


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Print the head anatomy of one run."
    )
    ap.add_argument(
        "run_dir", help="Seed directory holding checkpoints/ and .hydra/"
    )
    args = ap.parse_args()

    ckpt = find_checkpoints(args.run_dir, limit=1)[0]
    n_real = real_class_count(args.run_dir, _PROJECT_ROOT)
    print(json.dumps(summarize(read_head_anatomy(ckpt, n_real)), indent=2))
