"""Reader and renderer functions for structured vs unstructured pruning.

Reader functions load raw data from experiment checkpoints and return plain
dicts — no matplotlib. Renderer functions draw to caller-provided axes —
no file I/O.

Examples
--------
>>> import matplotlib.pyplot as plt
>>> from src.vis.pruning_compare import (
...     read_pruning_experiment, render_layerwise_sparsity
... )
>>> data = read_pruning_experiment("/path/to/exp", info={})  # doctest: +SKIP
>>> fig, ax = plt.subplots()  # doctest: +SKIP
>>> render_layerwise_sparsity(data["layers_enriched"], ax)  # doctest: +SKIP
>>> fig.savefig("out.pdf")  # doctest: +SKIP
"""

import logging
import os
import re
import sys
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

# Allow direct execution: python src/vis/pruning_compare.py
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_MODULE_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate

from src.modules.encoder_wrappers import EncoderWrapper
from src.vis.common import (
    PARAM_BAR_ALPHA,
    PARAM_BAR_COLOR,
    PERFECT_LINE_KW,
    RATE_LINESTYLES,
    experiment_sort_key,
    export_standalone_legend,
    get_style,
    layerwise_figsize,
    make_label,
    panel_models,
    ylim_for_rate,
)

log = logging.getLogger(__name__)

ZERO_TOL = 1e-12  # |w| below this → treated as pruned-to-zero
_TEST_CKPT_LOG_RE = re.compile(r"Test ckpt path:\s*(\S+\.ckpt)")

# ---------------------------------------------------------------------------
# Checkpoint readers
# ---------------------------------------------------------------------------


def find_best_ckpt(
    exp_dir: str,
    info: Optional[Dict] = None,
) -> Optional[str]:
    """Return the checkpoint that was used for testing in this run.

    For non-pruning runs, reads ``train.log`` for the ``Test ckpt path:``
    line. Pruning runs always use ``last.ckpt`` because the final checkpoint
    carries the fully-converged mask state.

    Falls back to ``last.ckpt`` (or ``last-vN.ckpt``), then the
    highest-epoch file, when the log is absent or has no match.

    Args:
        exp_dir: path to the experiment directory.
        info: parsed experiment metadata dict; ``method_class`` is read
            to decide whether this is a pruning run.

    Returns:
        Absolute path to the chosen checkpoint, or ``None`` if not found.
    """
    is_pruning = "pruning" in (info or {}).get("method_class", "")

    if not is_pruning:
        log_path = os.path.join(exp_dir, "train.log")
        if os.path.exists(log_path):
            with open(log_path) as f:
                for line in f:
                    if "Test ckpt path:" not in line:
                        continue
                    m = _TEST_CKPT_LOG_RE.search(line)
                    if m:
                        local = os.path.join(
                            exp_dir,
                            "checkpoints",
                            os.path.basename(m.group(1)),
                        )
                        if os.path.exists(local):
                            return local
        log.warning(
            f"Best ckpt not found in train.log for {exp_dir} — falling back"
        )

    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None
    files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if not files:
        return None
    last = sorted(f for f in files if f.startswith("last"))
    if last:
        return os.path.join(ckpt_dir, last[-1])
    epoch_files = sorted(f for f in files if f.startswith("epoch"))
    if epoch_files:
        return os.path.join(ckpt_dir, epoch_files[-1])
    return os.path.join(ckpt_dir, files[0])


def extract_pruned_layers(
    ckpt_path: str,
    *,
    include_all: bool = False,
) -> List[Dict]:
    """Walk a Lightning checkpoint and return one record per prunable layer.

    Every conv/linear weight is reported, whatever its sparsity, so a dense
    baseline lists the same layers as a pruned run and the two line up
    layer for layer. Handles both representations — pruning-hook attached
    (``{name}_orig`` + ``{name}_mask`` pairs) and baked-in zeros
    (hook removed via ``prune.remove``).

    Each record contains: ``name`` (str), ``kind`` (str), ``shape``
    (tuple), ``weight`` (Tensor, float32), ``mask`` (bool Tensor).

    Args:
        ckpt_path: path to the ``.ckpt`` file.
        include_all: if True, also include 1-D tensors — norm scales and
            any pruning-hooked 1-D parameter.

    Returns:
        List of layer dicts in model topology order.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)

    orig_keys = {k[: -len("_orig")] for k in sd if k.endswith("_orig")}
    mask_keys = {k[: -len("_mask")] for k in sd if k.endswith("_mask")}
    paired = orig_keys & mask_keys
    skip_keys = (
        paired | {b + "_orig" for b in paired} | {b + "_mask" for b in paired}
    )

    layers: List[Dict] = []

    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue

        if k.endswith("_orig"):
            base = k[: -len("_orig")]
            if base not in paired:
                continue
            w = v.detach().to(torch.float32)
            m = sd[base + "_mask"].detach().to(torch.float32)
            if w.ndim < 2:
                if not include_all:
                    log.warning(
                        f"Skipping 1-D pruning-hooked param {base} "
                        f"(shape={tuple(w.shape)}); pass include_all=True to include."
                    )
                    continue
                log.info(
                    f"Including 1-D pruning-hooked param {base} (shape={tuple(w.shape)})"
                )
            layers.append(
                dict(
                    name=base,
                    kind=_kind_of(w),
                    shape=tuple(w.shape),
                    weight=(w * m),
                    mask=(m != 0),
                )
            )
            continue

        if k in skip_keys or k.endswith("_mask"):
            continue
        if not k.endswith(".weight"):
            continue
        if v.ndim == 0 or (v.ndim < 2 and not include_all):
            continue
        v = v.detach().to(torch.float32)
        layers.append(
            dict(
                name=k,
                kind=_kind_of(v),
                shape=tuple(v.shape),
                weight=v,
                mask=(v.abs() >= ZERO_TOL),
            )
        )

    return layers


def read_pruning_experiment(
    exp_dir: str,
    info: Dict,
    *,
    include_all_layers: bool = False,
) -> Optional[Dict]:
    """Load checkpoint and compute per-layer and aggregate pruning stats.

    Args:
        exp_dir: path to the experiment directory.
        info: parsed experiment metadata dict.
        include_all_layers: pass through to ``extract_pruned_layers``.

    Returns:
        Dict with keys: ``exp_dir``, ``info``, ``ckpt``, ``agg``,
        ``layers_enriched`` (full layers with weight/mask/stats),
        ``layers_lean`` (scalars only, for JSON/cross-exp plots).
        Returns ``None`` if no checkpoint is found.
    """
    ckpt = find_best_ckpt(exp_dir, info=info)
    if ckpt is None:
        log.warning(f"No checkpoint found in {exp_dir} — skipping")
        return None

    log.info(f"→ {info['dirname']}")
    log.info(f"   ckpt: {os.path.relpath(ckpt, exp_dir)}")

    raw_layers = extract_pruned_layers(ckpt, include_all=include_all_layers)
    if not raw_layers:
        log.warning(
            f"No prunable layers in {info['dirname']} — recording as dense baseline"
        )

    layers_enriched = []
    for layer in raw_layers:
        layer = dict(layer)
        layer["stats"] = layer_stats(layer["mask"])
        layers_enriched.append(layer)

    agg = aggregate_stats(layers_enriched)
    log.info(
        f"   nominal={agg['nominal_sparsity'] * 100:.1f}%"
        f" structural_density={agg['structural_density'] * 100:.1f}%"
        f" structuredness={agg['mean_structuredness']:.4f}"
        f" fully-zero row/col={agg['mean_fully_zero_row_frac'] * 100:.0f}%"
        f"/{agg['mean_fully_zero_col_frac'] * 100:.0f}%"
    )

    layers_lean = [
        dict(
            name=l["name"],
            kind=l["kind"],
            nominal_sparsity=l["stats"]["nominal_sparsity"],
            n_params=l["stats"]["n_params"],
        )
        for l in layers_enriched
    ]

    return dict(
        exp_dir=exp_dir,
        info={k: v for k, v in info.items() if not k.startswith("_")},
        ckpt=ckpt,
        agg=agg,
        layers_enriched=layers_enriched,
        layers_lean=layers_lean,
    )


def read_eer_lookup(
    model_name: Optional[str],
    test_set: Optional[str],
    repo_root: str,
) -> Dict[str, float]:
    """Return a mapping of experiment name → EER from the leaderboard CSV.

    Reads ``results/test_eval/metrics/{model}/eer_leaderboard.csv``
    produced by ``scripts/aggregate_json_scores.py``.

    Args:
        model_name: ``"ecapa_tdnn"`` or ``"resnet34"``; returns empty
            dict when ``None``.
        test_set: filter to this dataset name, e.g.
            ``"cnceleb_concatenated"``; ``None`` keeps all rows.
        repo_root: root of the repository.

    Returns:
        Dict mapping experiment dirname to float EER.
    """
    if not model_name:
        return {}
    csv_path = os.path.join(
        repo_root,
        "results",
        "test_eval",
        "metrics",
        model_name,
        "eer_leaderboard.csv",
    )
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    if "is_latest" in df.columns:
        df = df[df["is_latest"] == True]  # noqa: E712
    if test_set:
        df = df[df["dataset"] == test_set]
    if df.empty:
        return {}
    eer_col = "EER_raw" if "EER_raw" in df.columns else "EER"
    return {
        row["exp"]: float(row[eer_col])
        for _, row in df.iterrows()
        if pd.notna(row[eer_col])
    }


def read_rtf(
    exp_dir: str,
    ckpt_path: str,
    layers: List[Dict],
    device: str,
    audio_seconds: float,
    sample_rate: int,
    warmup: int,
    iters: int,
) -> Optional[Dict]:
    """Time the forward pass of the audio encoder on synthetic input.

    Bakes pruning masks into the encoder weights before timing so the
    multiply hook does not skew results.

    Args:
        exp_dir: used for log messages and hyper_parameters lookup.
        ckpt_path: path to the checkpoint to load encoder config from.
        layers: enriched layer records from ``read_pruning_experiment``.
        device: ``"cpu"`` or ``"cuda"``.
        audio_seconds: length of the synthetic test utterance.
        sample_rate: sample rate in Hz.
        warmup: number of warm-up forward passes before timing.
        iters: number of timed forward passes.

    Returns:
        Dict with ``rtf_median``, ``rtf_mean``, ``rtf_p25``,
        ``rtf_p75``, ``device``, ``iters``, ``audio_seconds``.
        Returns ``None`` if the checkpoint has no ``hyper_parameters``.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters")
    if not hp or "model" not in hp:
        log.warning(
            f"RTF skipped for {exp_dir}: no hyper_parameters in checkpoint"
        )
        return None
    model_cfg = hp["model"]

    audio_processor = instantiate(model_cfg["audio_processor"])
    audio_processor_normalizer = instantiate(
        model_cfg["audio_processor_normalizer"]
    )
    raw_encoder = instantiate(model_cfg["audio_encoder"])
    encoder = EncoderWrapper(
        encoder=raw_encoder,
        audio_processor=audio_processor,
        audio_processor_normalizer=audio_processor_normalizer,
    )

    enc_state = OrderedDict()
    sd = ckpt["state_dict"]
    for k, v in sd.items():
        if k.endswith("_mask") or k.endswith("_orig"):
            continue
        if k.startswith("audio_encoder."):
            enc_state[k[len("audio_encoder.") :]] = v
    for layer in layers:
        full = layer["name"]
        if not full.startswith("audio_encoder."):
            continue
        enc_state[full[len("audio_encoder.") :]] = layer["weight"]

    missing, _ = encoder.load_state_dict(enc_state, strict=False)
    if missing:
        log.debug(
            f"RTF: {len(missing)} missing keys when loading encoder for {exp_dir}"
        )

    encoder.eval()
    encoder.to(device)

    audio_len = int(sample_rate * audio_seconds)
    audio = torch.randn(1, audio_len, device=device)
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()

    timings: List[float] = []
    with torch.inference_mode():
        for _ in range(warmup):
            _ = encoder(
                audio,
                wav_lens=torch.tensor([audio_len], device=device),
            )
        if use_cuda:
            torch.cuda.synchronize()
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = encoder(
                audio,
                wav_lens=torch.tensor([audio_len], device=device),
            )
            if use_cuda:
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - t0)

    arr = np.asarray(timings)
    return dict(
        rtf_median=float(np.median(arr) / audio_seconds),
        rtf_mean=float(arr.mean() / audio_seconds),
        rtf_p25=float(np.percentile(arr, 25) / audio_seconds),
        rtf_p75=float(np.percentile(arr, 75) / audio_seconds),
        device=device,
        iters=iters,
        audio_seconds=audio_seconds,
    )


# ---------------------------------------------------------------------------
# Pure computation (no matplotlib, no I/O)
# ---------------------------------------------------------------------------


def layer_stats(mask: torch.Tensor) -> Dict:
    """Compute per-layer sparsity and structuredness metrics.

    Args:
        mask: boolean or float tensor where True/1 = retained.

    Returns:
        Dict with ``nominal_sparsity``, ``structural_density``,
        ``structuredness``, ``fully_zero_row_frac``,
        ``fully_zero_col_frac``, ``per_row_density`` (array),
        ``per_col_density`` (array), ``weight_density``,
        ``n_params``, ``row_dim``, ``col_dim``.
    """
    z = (~mask.bool()).float()
    if z.ndim == 1:
        z = z.unsqueeze(1)
    row_dim = z.shape[0]
    col_dim = z.shape[1]

    flat_row = z.reshape(row_dim, -1).mean(dim=1)
    flat_col = (
        z.permute(1, 0, *range(2, z.ndim)).reshape(col_dim, -1).mean(dim=1)
    )

    per_row_density = 1.0 - flat_row
    per_col_density = 1.0 - flat_col

    fully_zero_row = (flat_row == 1.0).sum().item()
    fully_zero_col = (flat_col == 1.0).sum().item()
    kept_row = row_dim - fully_zero_row
    kept_col = col_dim - fully_zero_col

    spatial = int(np.prod(z.shape[2:])) if z.ndim > 2 else 1
    total_elements = row_dim * col_dim * spatial
    structural_density = (kept_row * kept_col * spatial) / max(
        total_elements, 1
    )
    weight_density = 1.0 - z.mean().item()

    return dict(
        nominal_sparsity=z.mean().item(),
        per_row_density=per_row_density.cpu().numpy(),
        per_col_density=per_col_density.cpu().numpy(),
        structuredness=float(
            max(
                per_row_density.var().item(),
                per_col_density.var().item(),
            )
        ),
        fully_zero_row_frac=fully_zero_row / max(row_dim, 1),
        fully_zero_col_frac=fully_zero_col / max(col_dim, 1),
        structural_density=structural_density,
        weight_density=weight_density,
        n_params=total_elements,
        row_dim=row_dim,
        col_dim=col_dim,
    )


def aggregate_stats(layers: List[Dict]) -> Dict:
    """Aggregate per-layer stats into experiment-level summary numbers.

    Args:
        layers: enriched layer dicts, each with a ``stats`` sub-dict.

    Returns:
        Dict with experiment-level ``nominal_sparsity``,
        ``structural_density``, ``weight_density``,
        ``mean_structuredness``, ``mean_fully_zero_row_frac``,
        ``mean_fully_zero_col_frac``, ``total_params``,
        ``kept_params_structural``, ``n_layers``.
    """
    if not layers:
        return dict(
            n_layers=0,
            nominal_sparsity=0.0,
            structural_density=1.0,
            weight_density=1.0,
            mean_structuredness=0.0,
            mean_fully_zero_row_frac=0.0,
            mean_fully_zero_col_frac=0.0,
            total_params=0,
            kept_params_structural=0,
        )
    stats = [l["stats"] for l in layers]
    total = sum(s["n_params"] for s in stats)
    zeros = sum(s["nominal_sparsity"] * s["n_params"] for s in stats)
    kept_struct = sum(s["structural_density"] * s["n_params"] for s in stats)
    return dict(
        n_layers=len(layers),
        nominal_sparsity=zeros / max(total, 1),
        structural_density=kept_struct / max(total, 1),
        weight_density=1.0 - zeros / max(total, 1),
        mean_structuredness=float(
            np.mean([s["structuredness"] for s in stats])
        ),
        mean_fully_zero_row_frac=float(
            np.mean([s["fully_zero_row_frac"] for s in stats])
        ),
        mean_fully_zero_col_frac=float(
            np.mean([s["fully_zero_col_frac"] for s in stats])
        ),
        total_params=total,
        kept_params_structural=int(round(kept_struct)),
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _kind_of(t: torch.Tensor) -> str:
    return {1: "bn", 2: "linear", 3: "conv1d", 4: "conv2d", 5: "conv3d"}.get(
        t.ndim, "other"
    )


def _short_layer_name(name: str, max_len: int = 36) -> str:
    """Compress a deep parameter path to a compact display label."""
    s = re.sub(r"^audio_encoder\.encoder\.\d+\.", "", name)
    s = re.sub(r"\.weight$", "", s)
    s = re.sub(r"^layer(\d+)\.se_res2block\.", r"SERes2block\1.", s)
    s = re.sub(r"^layer(\d+)\.conv$", r"Conv\1", s)
    s = re.sub(r"^layer(\d+)\.", r"L\1.", s)
    s = re.sub(r"^mfa\.", "MFA.", s)
    s = re.sub(r"^asp\.", "ASP.", s)
    s = re.sub(r"^fc\.linear$", "FC", s)
    s = re.sub(r"^fc$", "FC", s)
    s = re.sub(r"^classifier$", "Classifier", s)
    s = re.sub(r"seg_1$", "Linear", s)
    s = re.sub(r"L(\d+)\.\d+\.shortcut\.\d+", r"L\1.residual", s)
    if len(s) > max_len:
        s = "…" + s[-(max_len - 1) :]
    return s


def _representative_layers(layers: List[Dict], k: int) -> List[Dict]:
    """Pick first/last conv, first/last linear, then the largest layers."""
    if len(layers) <= k:
        return layers
    convs = [l for l in layers if l["kind"].startswith("conv")]
    lins = [l for l in layers if l["kind"] == "linear"]
    chosen: List[Dict] = []

    def add(item):
        if item is not None and item not in chosen:
            chosen.append(item)

    if convs:
        add(convs[0])
        add(convs[-1])
    if lins:
        add(lins[0])
        add(lins[-1])
    by_size = sorted(layers, key=lambda l: -l["stats"]["n_params"])
    for l in by_size:
        if len(chosen) >= k:
            break
        add(l)
    chosen.sort(key=lambda l: layers.index(l))
    return chosen


def _fmt_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _infer_model_name(
    experiments: List[Tuple[str, Dict]],
) -> Optional[str]:
    names = [info["dirname"] for _, info in experiments]
    if any("ecapa_tdnn" in n for n in names):
        return "ecapa_tdnn"
    if any("resnet34" in n for n in names):
        return "resnet34"
    return None


def _json_safe(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, (np.ndarray, torch.Tensor)):
        return None
    raise TypeError(f"Not JSON-serializable: {type(o)}")


def _scatter_with_styles(
    ax: "plt.Axes",
    summaries: List[Dict],
    xfn: Any,
    yfn: Any,
) -> None:
    seen_labels: set = set()
    for s in summaries:
        info = s["info"]
        x, y = xfn(s), yfn(s)
        if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
            continue
        color, marker, _ = get_style(info)
        label = make_label(info)
        kwargs: Dict[str, Any] = dict(
            color=color, marker=marker, s=60, linewidth=0.8, zorder=3
        )
        if marker not in ("x", "+", "1", "2", "3", "4"):
            kwargs.update(edgecolor="black", linewidth=0.5)
        if label not in seen_labels:
            kwargs["label"] = label
            seen_labels.add(label)
        ax.scatter([x], [y], **kwargs)


# ---------------------------------------------------------------------------
# Per-experiment renderers
# ---------------------------------------------------------------------------


def render_layerwise_sparsity(
    layers: List[Dict],
    ax: "plt.Axes",
    *,
    annotate: bool = False,
) -> None:
    """Render per-layer sparsity bars colored by structural density.

    Args:
        layers: enriched layer dicts (each has ``name`` and ``stats``).
        ax: axes to draw on.
        annotate: if True, overlay the fraction of fully-zero rows/cols.
    """
    n = len(layers)
    if n == 0:
        return
    names = [_short_layer_name(s["name"]) for s in layers]
    sparsities = [s["stats"]["nominal_sparsity"] * 100 for s in layers]
    struct_densities = [s["stats"]["structural_density"] for s in layers]
    fz_row = [s["stats"]["fully_zero_row_frac"] * 100 for s in layers]
    fz_col = [s["stats"]["fully_zero_col_frac"] * 100 for s in layers]

    cmap = plt.get_cmap("viridis_r")
    norm = plt.Normalize(0.0, 1.0)
    colors = [cmap(norm(d)) for d in struct_densities]

    y = np.arange(n)
    ax.barh(y, sparsities, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    usetex = plt.rcParams.get("text.usetex")
    s_metric = r"$\mathsf{{s}}^*[\%]$" if usetex else "Sparsity (%)"
    ax.set_xlabel(s_metric)

    if annotate:
        for yi, sp, fz_r, fz_c in zip(y, sparsities, fz_row, fz_col):
            if (fz_r + fz_c) <= 1:
                continue
            annot = (
                f"row=0:{fz_r:.0f}\\%  col=0:{fz_c:.0f}\\%"
                if usetex
                else f"row=0:{fz_r:.0f}%  col=0:{fz_c:.0f}%"
            )
            ax.text(
                min(sp + 1, 99),
                yi,
                annot,
                va="center",
                fontsize=6,
                color="#333333",
            )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig = ax.get_figure()
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label(
        "Structural density (row×col surviving after pruning)",
        fontsize=8,
    )
    cb.ax.tick_params(labelsize=7)


def render_layerwise_sparsity_nparams(
    layers: List[Dict],
    ax: "plt.Axes",
) -> None:
    """Render per-layer sparsity bars colored by parameter count.

    Args:
        layers: enriched layer dicts (each has ``name`` and ``stats``).
        ax: axes to draw on.
    """
    n = len(layers)
    if n == 0:
        return
    names = [_short_layer_name(s["name"]) for s in layers]
    sparsities = [s["stats"]["nominal_sparsity"] * 100 for s in layers]
    n_params = [s["stats"]["n_params"] for s in layers]

    min_p, max_p = min(n_params), max(n_params)
    cmap = plt.get_cmap("plasma")
    norm = plt.Normalize(min_p, max_p)
    colors = [cmap(norm(p)) for p in n_params]

    y = np.arange(n)
    ax.barh(y, sparsities, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    usetex = plt.rcParams.get("text.usetex")
    s_metric = r"$\mathsf{{s}}^*[\%]$" if usetex else "Sparsity (%)"
    ax.set_xlabel(s_metric)

    for yi, sp, p in zip(y, sparsities, n_params):
        ax.text(
            min(sp + 1, 98),
            yi,
            _fmt_params(p),
            va="center",
            ha="left",
            fontsize=4,
            color="#454444",
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig = ax.get_figure()
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label(
        rf"\# parameters ({_fmt_params(min_p)} → {_fmt_params(max_p)})",
        fontsize=8,
    )
    cb.ax.tick_params(labelsize=7)
    cb.formatter = plt.FuncFormatter(lambda x, _: _fmt_params(int(x)))
    cb.update_ticks()


def render_per_filter_histograms(
    layers: List[Dict],
    axes: np.ndarray,
    max_panels: int = 12,
) -> None:
    """Render per-row / per-column density histograms.

    Args:
        layers: enriched layer dicts.
        axes: 1-D array of Axes with at least ``max_panels`` elements.
        max_panels: maximum number of layers to show.
    """
    pick = _representative_layers(layers, max_panels)
    if not pick:
        return
    flat_axes = np.asarray(axes).flatten()

    for ax, layer in zip(flat_axes, pick):
        s = layer["stats"]
        ax.hist(
            s["per_row_density"],
            bins=20,
            range=(0, 1),
            color="#1f77b4",
            alpha=0.65,
            label="row",
            edgecolor="white",
            linewidth=0.3,
        )
        ax.hist(
            s["per_col_density"],
            bins=20,
            range=(0, 1),
            color="#d62728",
            alpha=0.55,
            label="col",
            edgecolor="white",
            linewidth=0.3,
        )
        density_overall = 1 - s["nominal_sparsity"]
        ax.axvline(density_overall, color="#000", linestyle=":", linewidth=0.8)
        ax.set_title(_short_layer_name(layer["name"]), fontsize=7)
        ax.set_xlim(0, 1)
        ax.tick_params(axis="both", labelsize=6)

    for ax in flat_axes[len(pick) :]:
        ax.set_visible(False)

    flat_axes[0].legend(loc="upper center", fontsize=7, frameon=False, ncol=2)


def render_mask_heatmaps(
    layers: List[Dict],
    axes: np.ndarray,
    max_panels: int = 12,
) -> None:
    """Render binary mask presence maps (row × column).

    For Conv layers the kernel dims are reduced by mean so values are
    in [0, 1]: structured pruning → solid black rows or columns,
    unstructured → fine grey speckle.

    Args:
        layers: enriched layer dicts.
        axes: 1-D array of Axes with at least ``max_panels`` elements.
        max_panels: maximum number of layers to show.
    """
    pick = _representative_layers(layers, max_panels)
    if not pick:
        return
    flat_axes = np.asarray(axes).flatten()

    for ax, layer in zip(flat_axes, pick):
        m = layer["mask"].float()
        if m.ndim > 2:
            m2d = m.reshape(m.shape[0], m.shape[1], -1).mean(dim=2)
        else:
            m2d = m
        ax.imshow(
            m2d.cpu().numpy(),
            cmap="gray",
            aspect="auto",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        ax.set_title(
            _short_layer_name(layer["name"]) + f"  {tuple(layer['shape'])}",
            fontsize=7,
        )
        # ax.set_xlabel("column axis (input channels)", fontsize=7)
        # ax.set_ylabel("row axis (output filters)", fontsize=7)
        ax.set_xlabel("Column axis", fontsize=7)
        ax.set_ylabel("Row axis", fontsize=7)
        ax.tick_params(labelsize=6)

    for ax in flat_axes[len(pick) :]:
        ax.set_visible(False)


# ---------------------------------------------------------------------------
# Cross-experiment renderers
# ---------------------------------------------------------------------------


def render_flops_vs_sparsity(
    summaries: List[Dict],
    ax: "plt.Axes",
) -> None:
    """Render realizable compute reduction vs nominal sparsity scatter.

    Args:
        summaries: list of summary dicts, each with ``info`` and ``agg``.
        ax: axes to draw on.
    """
    ax.plot(
        [0, 100],
        [0, 100],
        **PERFECT_LINE_KW,
        label="perfect speedup ($y=x$)",
    )
    _scatter_with_styles(
        ax,
        summaries,
        xfn=lambda s: s["agg"]["nominal_sparsity"] * 100,
        yfn=lambda s: (1 - s["agg"]["structural_density"]) * 100,
    )
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    usetex = plt.rcParams.get("text.usetex")
    ax.set_xlabel(
        r"Nominal sparsity (\%)" if usetex else "Nominal sparsity (%)"
    )
    ax.set_ylabel(
        r"Realizable compute reduction (\%)"
        if usetex
        else "Realizable compute reduction (%)"
    )
    ax.set_title(
        "Structured sparsity hugs the diagonal; unstructured does not",
        fontsize=9,
    )
    ax.legend(fontsize=7, loc="upper left", frameon=True, framealpha=0.5)


def render_structural_density_vs_sparsity(
    summaries: List[Dict],
    ax: "plt.Axes",
) -> None:
    """Render realizable compute density vs nominal sparsity scatter.

    Args:
        summaries: list of summary dicts with ``info`` and ``agg``.
        ax: axes to draw on.
    """
    ax.plot(
        [0, 100],
        [100, 0],
        **PERFECT_LINE_KW,
        label=r"perfect structural reduction ($y=100-x$)",
    )
    _scatter_with_styles(
        ax,
        summaries,
        xfn=lambda s: s["agg"]["nominal_sparsity"] * 100,
        yfn=lambda s: s["agg"]["structural_density"] * 100,
    )
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    usetex = plt.rcParams.get("text.usetex")
    ax.set_xlabel(
        r"Nominal sparsity (\%)" if usetex else "Nominal sparsity (%)"
    )
    ax.set_ylabel(
        r"Structural density (\%)" if usetex else "Structural density (%)"
    )
    ax.set_title(
        "Realizable compute density after collapsing zero rows/cols",
        fontsize=9,
    )
    ax.legend(fontsize=7, loc="best", frameon=True, framealpha=0.5)


def render_eer_vs_effective_flops(
    summaries: List[Dict],
    ax: "plt.Axes",
) -> None:
    """Render EER Pareto over realizable compute density (log scale).

    Args:
        summaries: list of summary dicts; each may have ``eer`` key.
        ax: axes to draw on.
    """

    def _x(s):
        d = s["agg"]["structural_density"]
        return d if d > 0 else None

    def _y(s):
        eer = s.get("eer")
        return eer * 100 if eer is not None else None

    _scatter_with_styles(ax, summaries, xfn=_x, yfn=_y)

    dense = [
        s
        for s in summaries
        if s["agg"]["nominal_sparsity"] < 0.01 and s.get("eer") is not None
    ]
    if dense:
        eer_dense = min(s["eer"] for s in dense) * 100
        ax.axhline(
            eer_dense,
            color="#888",
            linestyle=":",
            linewidth=0.8,
            label=f"dense baseline ({eer_dense:.2f}%)",
        )

    ax.set_xscale("log")
    ax.set_xlim(0.005, 1.05)
    ax.set_xlabel("Realizable compute density (log scale)")
    usetex = plt.rcParams.get("text.usetex")
    ax.set_ylabel(r"EER (\%)" if usetex else "EER (%)")
    ax.set_title("Pareto: EER vs realizable compute", fontsize=9)
    ax.legend(fontsize=7, loc="best", frameon=True, framealpha=0.5)


def render_rtf_vs_sparsity(
    summaries: List[Dict],
    ax: "plt.Axes",
) -> bool:
    """Render median RTF vs nominal sparsity scatter.

    Args:
        summaries: list of summary dicts; those without ``rtf`` are
            skipped.
        ax: axes to draw on.

    Returns:
        ``True`` if any data was drawn, ``False`` if all runs lacked RTF.
    """
    has_rtf = [s for s in summaries if s.get("rtf") is not None]
    if not has_rtf:
        return False
    _scatter_with_styles(
        ax,
        has_rtf,
        xfn=lambda s: s["agg"]["nominal_sparsity"] * 100,
        yfn=lambda s: s["rtf"]["rtf_median"],
    )
    dense = [s for s in has_rtf if s["agg"]["nominal_sparsity"] < 0.01]
    if dense:
        ax.axhline(
            dense[0]["rtf"]["rtf_median"],
            color="#888",
            linestyle=":",
            linewidth=0.8,
            label="dense baseline",
        )
    device = has_rtf[0]["rtf"]["device"]
    usetex = plt.rcParams.get("text.usetex")
    ax.set_xlabel(
        r"Nominal sparsity (\%)" if usetex else "Nominal sparsity (%)"
    )
    ax.set_ylabel("RTF (median)")
    ax.set_title(f"Measured RTF on {device}", fontsize=9)
    ax.legend(fontsize=7, loc="best", frameon=True, framealpha=0.5)
    return True


# ---------------------------------------------------------------------------
# Layerwise cross-experiment panel renderer
# ---------------------------------------------------------------------------


def render_layerwise_panel(
    ax: "plt.Axes",
    members: List[Dict],
    *,
    rate_linestyles: Optional[Dict[int, Any]] = None,
    target_rates: Optional[List[int]] = None,
    show_param_bars: bool = True,
    show_target_lines: bool = True,
) -> Dict[str, Any]:
    """Render one layerwise-sparsity panel.

    Args:
        ax: axes to draw on.
        members: summary dicts for this panel (same backbone).
        rate_linestyles: if set, override per-curve linestyle by integer
            sparsity rate (overlay-rates mode).
        target_rates: horizontal reference lines at these integer rates;
            if ``None``, derived from ``members``.
        show_param_bars: whether to draw a log-scale twin-axis bar
            showing the parameter count per layer.
        show_target_lines: whether to draw target-rate reference lines.

    Returns:
        Dict with keys ``legend_handles`` (OrderedDict), ``bar_handle``,
        ``target_handle``, ``twin_ax``, ``canon_names``.
    """
    canon = max(members, key=lambda s: len(s["layers"]))
    canon_names = [l["name"] for l in canon["layers"]]
    canon_nparams = [
        max(int(l.get("n_params", 0)), 1) for l in canon["layers"]
    ]
    x = np.arange(len(canon_names))

    twin_ax = None
    bar_handle = None
    if show_param_bars:
        twin_ax = ax.twinx()
        twin_ax.set_yscale("log")
        bars = twin_ax.bar(
            x,
            canon_nparams,
            color=PARAM_BAR_COLOR,
            alpha=PARAM_BAR_ALPHA,
            width=0.85,
            zorder=0,
            edgecolor="none",
        )
        if len(bars) > 0:
            bar_handle = bars[0]
        ax.set_zorder(twin_ax.get_zorder() + 1)
        ax.patch.set_visible(False)

    legend_handles: "OrderedDict[str, Tuple[Any, Dict]]" = OrderedDict()
    for s in members:
        info_local = {
            k: v for k, v in s["info"].items() if k != "_gradient_color"
        }
        color, marker, ls = get_style(info_local)
        if rate_linestyles is not None:
            ls = rate_linestyles.get(info_local.get("sparsity"), ls)
        label = make_label(info_local)
        lookup = {l["name"]: l["nominal_sparsity"] * 100 for l in s["layers"]}
        ys = [lookup.get(name, np.nan) for name in canon_names]
        (line,) = ax.plot(
            x,
            ys,
            color=color,
            marker=marker,
            linestyle=ls,
            markersize=4,
            linewidth=1.0,
            alpha=0.9,
            zorder=3,
        )
        if label not in legend_handles:
            legend_handles[label] = (line, info_local)

    target_handle = None
    if show_target_lines:
        rates = (
            target_rates
            if target_rates is not None
            else sorted(
                {
                    s["info"].get("sparsity")
                    for s in members
                    if s["info"].get("sparsity") is not None
                }
            )
        )
        for r in rates:
            ax.axhline(r, **PERFECT_LINE_KW)
        if rates:
            target_handle = plt.Line2D(
                [0],
                [0],
                color=PERFECT_LINE_KW["color"],
                linestyle=PERFECT_LINE_KW["linestyle"],
                linewidth=PERFECT_LINE_KW["linewidth"],
                alpha=PERFECT_LINE_KW["alpha"],
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [_short_layer_name(n, max_len=24) for n in canon_names],
        rotation=60,
        ha="right",
        fontsize=6,
    )
    ax.set_xlabel("Layer (input → output)", fontsize=9)

    return dict(
        legend_handles=legend_handles,
        bar_handle=bar_handle,
        target_handle=target_handle,
        twin_ax=twin_ax,
        canon_names=canon_names,
    )


# ---------------------------------------------------------------------------
# Legend assembly helpers (called from entry point)
# ---------------------------------------------------------------------------


def assemble_layerwise_legend(
    legend_handles: "OrderedDict[str, Tuple[Any, Dict]]",
    *,
    target_handle: Optional[Any] = None,
    target_label: Optional[str] = None,
    bar_handle: Optional[Any] = None,
    bar_label: Optional[str] = None,
) -> Tuple[List[Any], List[str]]:
    """Sort method handles by experiment order, then append target/bar.

    Args:
        legend_handles: OrderedDict mapping label → (handle, info).
        target_handle: optional Line2D for the target-rate reference.
        target_label: legend text for target_handle.
        bar_handle: optional bar patch for the n_params twin axis.
        bar_label: legend text for bar_handle.

    Returns:
        ``(handles, labels)`` lists ready for ``fig.legend``.
    """
    sorted_items = sorted(
        legend_handles.items(),
        key=lambda kv: experiment_sort_key(kv[1][1]),
    )
    ordered_labels = [label for label, _ in sorted_items]
    ordered_handles = [h for _, (h, _) in sorted_items]
    if target_handle is not None and target_label is not None:
        ordered_labels.append(target_label)
        ordered_handles.append(target_handle)
    if bar_handle is not None and bar_label is not None:
        ordered_labels.append(bar_label)
        ordered_handles.append(bar_handle)
    return ordered_handles, ordered_labels


def emit_layerwise_legend(
    fig: "plt.Figure",
    out_path: str,
    handles: List[Any],
    labels: List[str],
    *,
    legend_mode: str,
) -> None:
    """Attach or export the layerwise legend, then tighten layout.

    Args:
        fig: the figure containing the layerwise axes.
        out_path: path used for the main figure PDF (the sidecar legend
            PDF is derived by appending ``_legend`` before ``.pdf``).
        handles: legend artist handles.
        labels: legend text entries.
        legend_mode: ``"inline"`` embeds the legend at the figure bottom;
            ``"split"`` writes a sidecar ``<out>_legend.pdf``.
    """
    ncol = min(6, max(2, len(labels)))
    if legend_mode == "inline":
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=ncol,
            fontsize=7,
            frameon=True,
            framealpha=0.8,
            bbox_to_anchor=(0.5, -0.02),
        )
        fig.tight_layout(rect=(0, 0.08, 1, 1))
    elif legend_mode == "split":
        legend_path = os.path.splitext(out_path)[0] + "_legend.pdf"
        export_standalone_legend(
            handles, labels, legend_path, ncol, font_size=8
        )
        fig.tight_layout()
    else:
        raise ValueError(
            f"legend_mode must be 'inline' or 'split', got {legend_mode!r}"
        )


if __name__ == "__main__":
    import tempfile

    # Smoke test: build synthetic layer list and render each per-exp plot.
    from src.vis.common import setup_matplotlib

    setup_matplotlib(font_size=9)

    n_layers = 6
    dummy_layers = []
    for i in range(n_layers):
        shape = (16, 8, 3)
        weight = torch.randn(*shape)
        if i % 2 == 0:  # every other layer: zero out full rows
            weight[:4] = 0.0
        mask = weight.abs() >= ZERO_TOL
        dummy_layers.append(
            dict(
                name=f"audio_encoder.encoder.0.layer{i}.conv.weight",
                kind="conv1d",
                shape=shape,
                weight=weight,
                mask=mask,
                stats=layer_stats(mask),
            )
        )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        out = f.name
    fig, ax = plt.subplots(figsize=(6.4, max(2.4, 0.18 * n_layers + 1.0)))
    render_layerwise_sparsity(dummy_layers, ax, annotate=True)
    ax.set_title("Smoke test — layerwise sparsity", fontsize=9)
    fig.savefig(out)
    plt.close(fig)
    import os as _os

    assert _os.path.getsize(out) > 0, "Output PDF is empty"
    print(f"smoke ok → {out}")
