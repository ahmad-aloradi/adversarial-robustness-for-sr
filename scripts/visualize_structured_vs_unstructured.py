#!/usr/bin/env python3
"""Diagnose structured vs unstructured sparsity in trained SV checkpoints.

Per-experiment plots (one folder per discovered run):
    layerwise_sparsity.pdf       — per-layer sparsity bars, colored by
                                   structuredness score (fixed scale
                                   [0, 0.25] = theoretical max p(1-p)),
                                   annotated with the fraction of fully-zero
                                   rows / columns
    per_filter_sparsity_hist.pdf — distribution of per-row / per-column
                                   density: bimodal ⇒ structured, narrow
                                   spike ⇒ unstructured
    mask_heatmap.pdf             — binary mask of representative layers,
                                   collapsed to a (row × column) presence
                                   map by averaging over the kernel dims
    summary.json                 — per-experiment metrics dump

Cross-experiment plots (in {output}/cross_exp/):
    flops_vs_sparsity.pdf                — realizable compute reduction vs
                                           nominal sparsity. Diagonal y=x is
                                           perfect speedup; structured runs hug
                                           it, unstructured runs sit far below.
                                           Headline structured-vs-unstructured
                                           figure.
    structural_density_vs_sparsity.pdf   — realizable compute density vs sparsity
    eer_vs_effective_flops.pdf           — EER Pareto over realizable FLOPs
    rtf_vs_sparsity.pdf              — measured RTF vs sparsity (--rtf only)

Usage:
    python scripts/visualize_structured_vs_unstructured.py \\
        --base_dirs /data/aloradad/results/cnceleb \\
        --experiments 'sv_pruning_mag_struct*ecapa_tdnn*sr90*' \\
                      'sv_pruning_mag_unstruct*ecapa_tdnn*sr90*' \\
        --output results/struct_vs_unstruct/ecapa_tdnn

    # With RTF benchmark on CPU
    python scripts/visualize_structured_vs_unstructured.py \\
        --base_dirs /data/aloradad/results/cnceleb \\
        --experiments 'sv_pruning_mag_struct*ecapa_tdnn*sr90*' \\
        --rtf --rtf_device cpu --rtf_iters 20 \\
        --output /tmp/struct_vs_unstruct
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Reuse styling and discovery from the sibling visualize.py + shared constants
# from visualize_common.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize import (  # noqa: E402
    discover_experiments,
    experiment_sort_key,
    export_standalone_legend,
    get_style,
    make_label,
    setup_matplotlib,
)
from visualize_common import (  # noqa: E402
    PARAM_BAR_ALPHA,
    PARAM_BAR_COLOR,
    PERFECT_LINE_KW,
    RATE_LINESTYLES,
    layerwise_figsize,
    panel_models,
    ylim_for_rate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("struct_vs_unstruct")

ZERO_TOL = 1e-12  # treat |w| < tol as pruned for already-baked checkpoints

_TEST_CKPT_LOG_RE = re.compile(r"Test ckpt path:\s*(\S+\.ckpt)")

# ---------------------------------------------------------------------------
# Checkpoint introspection
# ---------------------------------------------------------------------------

def find_best_ckpt(exp_dir: str, info: Optional[Dict] = None) -> Optional[str]:
    """Return the checkpoint that was actually used for testing in this run.

    For Bregman methods, reads ``train.log`` and extracts the path on the
    ``Test ckpt path:`` line — the exact monitor-best file that was evaluated.
    Pruning methods always use ``last.ckpt`` because the final checkpoint
    carries the fully-converged mask/hook state.

    Falls back to ``last.ckpt`` (or ``last-vN.ckpt``), then the highest-epoch
    file, when the log is absent, has no such line, or the file cannot be found.
    """
    is_pruning = "pruning" in (info or {}).get("method_class", "")

    if not is_pruning:
        # Primary: exact test ckpt recorded in the training log.
        # The logged path may point to the original training cluster (not this
        # machine), so resolve by basename inside exp_dir/checkpoints/.
        log_path = os.path.join(exp_dir, "train.log")
        if os.path.exists(log_path):
            with open(log_path) as f:
                for line in f:
                    if "Test ckpt path:" not in line:
                        continue
                    m = _TEST_CKPT_LOG_RE.search(line)
                    if m:
                        local = os.path.join(exp_dir, "checkpoints", os.path.basename(m.group(1)))
                        if os.path.exists(local):
                            return local
        log.warning("Best ckpt not found in train.log for %s — falling back to last", exp_dir)
    # Fallback: last.ckpt or highest-epoch file in checkpoints/
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None
    files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if not files:
        return None
    last = sorted(f for f in files if f.startswith("last"))
    if last:
        return os.path.join(ckpt_dir, last[-1])  # last-v1 > last
    epoch_files = sorted(f for f in files if f.startswith("epoch"))
    if epoch_files:
        return os.path.join(ckpt_dir, epoch_files[-1])
    return os.path.join(ckpt_dir, files[0])


def extract_pruned_layers(ckpt_path: str) -> List[Dict]:
    """Walk a Lightning checkpoint and return one record per pruned layer.

    Handles both representations:
      - pruning hook attached: ``{name}_orig`` + ``{name}_mask`` pairs
      - hook removed (``prune.remove`` baked in): plain ``{name}`` with
        explicit zeros (see ``scripts/make_pruning_permanent.py``)

    Each record:
      {
        "name": str,           # parameter path, e.g. "audio_encoder.....conv.weight"
        "kind": str,           # "conv1d" | "conv2d" | "linear"
        "shape": tuple,
        "weight": Tensor,      # float — effective (orig * mask if applicable)
        "mask":   Tensor,      # bool — True where retained, shape == weight.shape
      }

    Tensors with ndim < 2 (BN gains, biases) are skipped — sparsity stats
    only make sense for matrix / kernel tensors.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)

    # Pair up _orig / _mask first so plain weight keys don't double-count.
    orig_keys = {k[: -len("_orig")] for k in sd if k.endswith("_orig")}
    mask_keys = {k[: -len("_mask")] for k in sd if k.endswith("_mask")}
    paired = orig_keys & mask_keys

    layers: List[Dict] = []

    for base in sorted(paired):
        w = sd[base + "_orig"].detach().to(torch.float32)
        m = sd[base + "_mask"].detach().to(torch.float32)
        if w.ndim < 2:
            continue
        layers.append(
            dict(
                name=base,
                kind=_kind_of(w),
                shape=tuple(w.shape),
                weight=(w * m),
                mask=(m != 0),
            )
        )

    # Also accept already-baked weights as pruned if a substantial fraction
    # is exactly zero. This covers checkpoints that ran make_pruning_permanent.
    handled = paired | {b + "_orig" for b in paired} | {b + "_mask" for b in paired}
    for k, v in sd.items():
        if k in handled or not isinstance(v, torch.Tensor) or v.ndim < 2:
            continue
        if not k.endswith(".weight"):
            continue
        sparsity = (v.abs() < ZERO_TOL).float().mean().item()
        if sparsity < 0.01:  # treat as dense
            continue
        layers.append(
            dict(
                name=k,
                kind=_kind_of(v),
                shape=tuple(v.shape),
                weight=v.to(torch.float32),
                mask=(v.abs() >= ZERO_TOL),
            )
        )

    return layers


def _kind_of(t: torch.Tensor) -> str:
    return {2: "linear", 3: "conv1d", 4: "conv2d", 5: "conv3d"}.get(t.ndim, "other")


# ---------------------------------------------------------------------------
# Layer-level sparsity statistics
# ---------------------------------------------------------------------------

def layer_stats(weight: torch.Tensor, mask: torch.Tensor) -> Dict:
    """Compute the per-layer numbers used by every plot.

    Conventions
        - dim 0 of the weight tensor is the *row* axis: it indexes output
          filters (Conv) / output features (Linear).
        - dim 1 is the *column* axis: input channels / input features.
        - density = fraction of *retained* (non-zero) entries.

    The structuredness score is ``max(var_per_row_density, var_per_col_density)``
    — an informal diagnostic. A structured pruner zeroes whole rows or whole
    columns, making one distribution bimodal and driving its variance toward
    ``p(1-p)``; an unstructured pruner spreads zeros evenly, leaving both
    variances near zero.
    """
    z = (~mask).float()  # 1 where zero, 0 where retained
    row_dim = z.shape[0]
    col_dim = z.shape[1]

    flat_row = z.reshape(row_dim, -1).mean(dim=1)  # sparsity per row
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
    # "Structural density": fraction of compute blocks that survive after
    # dropping fully-zero rows AND fully-zero columns. Dense kernels can
    # realize this speedup directly; sparse-kernel-only savings are captured
    # by weight_density = 1 - nominal_sparsity instead.
    structural_density = (kept_row * kept_col * spatial) / max(total_elements, 1)
    weight_density = 1.0 - z.mean().item()

    return dict(
        nominal_sparsity=z.mean().item(),
        per_row_density=per_row_density.cpu().numpy(),
        per_col_density=per_col_density.cpu().numpy(),
        structuredness=float(max(per_row_density.var().item(),
                                 per_col_density.var().item())),
        fully_zero_row_frac=fully_zero_row / max(row_dim, 1),
        fully_zero_col_frac=fully_zero_col / max(col_dim, 1),
        structural_density=structural_density,
        weight_density=weight_density,
        n_params=total_elements,
        row_dim=row_dim,
        col_dim=col_dim,
    )


def aggregate_stats(layers: List[Dict]) -> Dict:
    """Aggregate layer stats into experiment-level summary numbers."""
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
        mean_structuredness=float(np.mean([s["structuredness"] for s in stats])),
        mean_fully_zero_row_frac=float(np.mean([s["fully_zero_row_frac"] for s in stats])),
        mean_fully_zero_col_frac=float(np.mean([s["fully_zero_col_frac"] for s in stats])),
        total_params=total,
        kept_params_structural=int(round(kept_struct)),
    )


# ---------------------------------------------------------------------------
# Per-experiment plots
# ---------------------------------------------------------------------------

def _short_layer_name(name: str, max_len: int = 36) -> str:
    """Render a deep parameter path as a compact, human-readable label.

    Maps the nested ECAPA-TDNN / WeSpeaker / ResNet naming scheme to a short
    form that surfaces the meaningful structural unit. Examples::

        layer1.conv                       → Conv1
        layer4.se_res2block.0.conv        → SERes2block4.0.conv
        layer4.se_res2block.1.convs.3     → SERes2block4.1.convs.3
        layer4.se_res2block.2.linear1     → SERes2block4.2.linear1
        mfa.conv                          → MFA.conv
        asp.linear1                       → ASP.linear1
        fc.linear                         → FC
        classifier                        → Classifier
        layerN.X.Y                        → LN.X.Y   (other "layerN" forms)

    The sub-index of an SE-Res2 block (the ``.0/.1/.2`` position inside it)
    is preserved because it disambiguates pre-conv / multi-scale / post-conv
    sub-modules that would otherwise share the same label.
    """
    s = re.sub(r"^audio_encoder\.encoder\.\d+\.", "", name)
    s = re.sub(r"\.weight$", "", s)

    # ECAPA-TDNN / WeSpeaker structural blocks
    s = re.sub(r"^layer(\d+)\.se_res2block\.", r"SERes2block\1.", s)
    s = re.sub(r"^layer(\d+)\.conv$", r"Conv\1", s)
    s = re.sub(r"^layer(\d+)\.", r"L\1.", s)
    s = re.sub(r"^mfa\.", "MFA.", s)
    s = re.sub(r"^asp\.", "ASP.", s)
    s = re.sub(r"^fc\.linear$", "FC", s)
    s = re.sub(r"^fc$", "FC", s)
    s = re.sub(r"^classifier$", "Classifier", s)

    # ResNet34 / WeSpeaker structural block
    s = re.sub(r"seg_1$", "Linear", s)
    # L4.0.shortcut.0 -> L4.residual
    s = re.sub(r"L(\d+)\.\d+\.shortcut\.\d+", r"L\1.residual", s)

    if len(s) > max_len:
        s = "…" + s[-(max_len - 1):]
    return s


def plot_layerwise_sparsity(
    layers: List[Dict], out_path: str, title: str, *,
    annotate: bool = False,
) -> None:
    setup_matplotlib(font_size=9)

    n = len(layers)
    if n == 0:
        return
    fig, ax = plt.subplots(figsize=(6.4, max(2.4, 0.18 * n + 1.0)))
    names = [_short_layer_name(s["name"]) for s in layers]
    sparsities = [s["stats"]["nominal_sparsity"] * 100 for s in layers]
    struct_densities = [s["stats"]["structural_density"] for s in layers]
    fz_row = [s["stats"]["fully_zero_row_frac"] * 100 for s in layers]
    fz_col = [s["stats"]["fully_zero_col_frac"] * 100 for s in layers]

    # Color bars by structural_density: fraction of compute that survives after
    # collapsing fully-zero rows and columns. 0 = all compute eliminated
    # (maximally structured), 1 = no structural gain (unstructured). viridis_r
    # maps low density (structured) to bright yellow and high density to purple.
    cmap = plt.get_cmap("viridis_r")
    norm = plt.Normalize(0.0, 1.0)
    colors = [cmap(norm(d)) for d in struct_densities]

    y = np.arange(n)
    ax.barh(y, sparsities, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    s_metric = r"$\mathsf{{s}}^*={rate}\%$" if plt.rcParams.get("text.usetex") else "Sparsity (%)"
    ax.set_xlabel(f"{s_metric}")
    ax.set_title(title, fontsize=9)

    if annotate:
        # Annotation = fraction of *entirely zero* rows / columns. This is
        # different from nominal sparsity (which counts individual zero
        # elements): a layer can be 90% sparse with 0% zero rows (unstructured)
        # or 90% sparse with 90% zero columns (structured along input axis).
        for yi, sp, fz_r, fz_c in zip(y, sparsities, fz_row, fz_col):
            annot = (
                f"row=0:{fz_r:.0f}\\%  col=0:{fz_c:.0f}\\%"
                if plt.rcParams.get("text.usetex")
                else f"row=0:{fz_r:.0f}%  col=0:{fz_c:.0f}%"
            ) if (fz_r + fz_c) > 1 else ""
            if annot:
                ax.text(min(sp + 1, 99), yi, annot, va="center", fontsize=6, color="#333333")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label("Structural density (row×col surviving after pruning)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.savefig(out_path)
    plt.close(fig)


def _fmt_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def plot_layerwise_sparsity_nparams(layers: List[Dict], out_path: str, title: str) -> None:
    """Same horizontal-bar layout as plot_layerwise_sparsity but bars are
    colored by the number of parameters in each layer (min → max)."""
    setup_matplotlib(font_size=9)

    n = len(layers)
    if n == 0:
        return
    fig, ax = plt.subplots(figsize=(6.4, max(2.4, 0.18 * n + 1.0)))
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
    s_metric = r"$\mathsf{{s}}^*={rate}\%$" if plt.rcParams.get("text.usetex") else "Sparsity (%)"
    ax.set_xlabel(f"{s_metric}")
    ax.set_title(title, fontsize=9)

    for yi, sp, p in zip(y, sparsities, n_params):
        ax.text(min(sp + 1, 98), yi, _fmt_params(p),
                va="center", ha="left", fontsize=4, color="#454444")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label(
        f"\# parameters ({_fmt_params(min_p)} → {_fmt_params(max_p)})", fontsize=8
    )
    cb.ax.tick_params(labelsize=7)
    cb.formatter = plt.FuncFormatter(lambda x, _: _fmt_params(int(x)))
    cb.update_ticks()

    fig.savefig(out_path)
    plt.close(fig)


def plot_per_filter_histograms(layers: List[Dict], out_path: str, title: str,
                               max_panels: int = 12) -> None:
    """Per-layer density histograms.

    Two overlaid histograms per panel — per-row density (dim 0; output
    filters) and per-column density (dim 1; input channels). Structured
    pruning along one axis collapses that histogram to a Dirac at 0 (or 1);
    unstructured runs leave both as narrow spikes around the global density.
    """
    setup_matplotlib(font_size=9)
    pick = _representative_layers(layers, max_panels)
    if not pick:
        return
    n = len(pick)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 1.9), sharex=True)
    axes = np.atleast_2d(axes).flatten()

    for ax, layer in zip(axes, pick):
        s = layer["stats"]
        ax.hist(s["per_row_density"], bins=20, range=(0, 1),
                color="#1f77b4", alpha=0.65, label="row", edgecolor="white", linewidth=0.3)
        ax.hist(s["per_col_density"], bins=20, range=(0, 1),
                color="#d62728", alpha=0.55, label="col", edgecolor="white", linewidth=0.3)
        density_overall = 1 - s["nominal_sparsity"]
        ax.axvline(density_overall, color="#000", linestyle=":", linewidth=0.8)
        ax.set_title(_short_layer_name(layer["name"]), fontsize=7)
        ax.set_xlim(0, 1)
        ax.tick_params(axis="both", labelsize=6)

    for ax in axes[len(pick):]:
        ax.set_visible(False)

    axes[0].legend(loc="upper center", fontsize=7, frameon=False, ncol=2)
    fig.suptitle(title + r"  —  per-row (blue) / per-column (red) density",
                 fontsize=9)
    fig.supxlabel("Per-row / per-column density (1 = retained)", fontsize=8)
    fig.supylabel("Count", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)


def plot_mask_heatmaps(layers: List[Dict], out_path: str, title: str,
                       max_panels: int = 12) -> None:
    """Show the binary mask collapsed to a 2-D (row × column) presence map.

    For Conv layers the mask is 3-D / 4-D (row, col, *kernel*). We reduce the
    kernel dim by **mean** so the value at ``(i, j)`` is the *fraction of
    retained kernel elements* between output filter ``i`` and input
    channel ``j``. This keeps the colormap range in [0, 1] and makes the
    structural distinction visible at a glance:

      - Structured along the column axis ⇒ entire input columns are zero
        at every kernel offset, so mean = 0 ⇒ solid black columns.
      - Structured along the row axis    ⇒ solid black rows.
      - Unstructured                     ⇒ fine grey speckle (each
        (row, col) has only some kernel elements pruned).

    Max would erase partial pruning (any retained element ⇒ white); sum is
    just unscaled mean. Mean is the standard choice (cf. Han et al., Deep
    Compression).
    """
    setup_matplotlib(font_size=9)
    pick = _representative_layers(layers, max_panels)
    if not pick:
        return
    n = len(pick)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.2))
    axes = np.atleast_2d(axes).flatten()

    for ax, layer in zip(axes, pick):
        m = layer["mask"].float()
        if m.ndim > 2:
            m2d = m.reshape(m.shape[0], m.shape[1], -1).mean(dim=2)
        else:
            m2d = m
        ax.imshow(m2d.cpu().numpy(), cmap="gray", aspect="auto",
                  vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(_short_layer_name(layer["name"]) + f"  {tuple(layer['shape'])}",
                     fontsize=7)
        ax.set_xlabel("column axis (input channels)", fontsize=7)
        ax.set_ylabel("row axis (output filters)", fontsize=7)
        ax.tick_params(labelsize=6)

    for ax in axes[len(pick):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)


def _representative_layers(layers: List[Dict], k: int) -> List[Dict]:
    """Pick the first/last conv, first/last linear, plus the largest few."""
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
    # Fill remainder with the largest tensors not yet chosen, preserving
    # original order so plots are stable run-to-run.
    by_size = sorted(layers, key=lambda l: -l["stats"]["n_params"])
    for l in by_size:
        if len(chosen) >= k:
            break
        add(l)
    chosen.sort(key=lambda l: layers.index(l))
    return chosen


# ---------------------------------------------------------------------------
# RTF benchmark (optional — requires the SV LightningModule to instantiate)
# ---------------------------------------------------------------------------

def measure_rtf(exp_dir: str, ckpt_path: str, layers: List[Dict],
                device: str, audio_seconds: float, sample_rate: int,
                warmup: int, iters: int) -> Optional[Dict]:
    """Time the forward pass of the audio encoder on synthetic input.

    Bakes the masks in (so the multiply hook does not pollute timing) and
    instantiates only the audio_encoder + audio_processor from the saved
    Hydra hyper_parameters — avoids the full SpeakerVerification module
    pulling in datasets / scoring config.
    """
    try:
        from hydra.utils import instantiate
        from src.modules.encoder_wrappers import EncoderWrapper
    except Exception as e:  # pragma: no cover — import-time failure
        log.warning("RTF skipped: cannot import encoder_wrappers (%s)", e)
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters")
    if not hp or "model" not in hp:
        log.warning("RTF skipped for %s: no hyper_parameters in checkpoint", exp_dir)
        return None
    model_cfg = hp["model"]

    try:
        audio_processor = instantiate(model_cfg["audio_processor"])
        audio_processor_normalizer = instantiate(model_cfg["audio_processor_normalizer"])
        raw_encoder = instantiate(model_cfg["audio_encoder"])
        encoder = EncoderWrapper(
            encoder=raw_encoder,
            audio_processor=audio_processor,
            audio_processor_normalizer=audio_processor_normalizer,
        )
    except Exception as e:
        log.warning("RTF skipped for %s: instantiation failed (%s)", exp_dir, e)
        return None

    # Bake effective weights into the encoder. Iterate the layer records and
    # write the masked tensor to the matching parameter, dropping the _orig /
    # _mask hook entirely (so timing reflects a regular dense forward).
    enc_state = OrderedDict()
    sd = ckpt["state_dict"]
    for k, v in sd.items():
        if k.endswith("_mask") or k.endswith("_orig"):
            continue
        if k.startswith("audio_encoder."):
            enc_state[k[len("audio_encoder."):]] = v
    # Add baked-in pruned weights from the layer records.
    for layer in layers:
        full = layer["name"]
        if not full.startswith("audio_encoder."):
            continue
        enc_state[full[len("audio_encoder."):]] = layer["weight"]

    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    if missing:
        log.debug("RTF: %d missing keys when loading encoder for %s",
                  len(missing), exp_dir)

    encoder.eval()
    try:
        encoder.to(device)
    except Exception as e:
        log.warning("RTF skipped for %s: cannot move to %s (%s)", exp_dir, device, e)
        return None

    audio_len = int(sample_rate * audio_seconds)
    audio = torch.randn(1, audio_len, device=device)

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()

    timings: List[float] = []
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                _ = encoder(audio, wav_lens=torch.tensor([audio_len], device=device))
            if use_cuda:
                torch.cuda.synchronize()
            for _ in range(iters):
                t0 = time.perf_counter()
                _ = encoder(audio, wav_lens=torch.tensor([audio_len], device=device))
                if use_cuda:
                    torch.cuda.synchronize()
                timings.append(time.perf_counter() - t0)
    except Exception as e:
        log.warning("RTF skipped for %s: forward failed (%s)", exp_dir, e)
        return None

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
# EER readout from the existing aggregator output
# ---------------------------------------------------------------------------

def load_eer_lookup(model_name: Optional[str], test_set: Optional[str],
                    repo_root: str) -> Dict[str, float]:
    """Return a dict mapping experiment name -> EER (raw, latest run).

    Reads ``results/test_eval/metrics/{model}/eer_leaderboard.csv`` produced
    by ``scripts/aggregate_json_scores.py``. Filters to the chosen
    ``test_set`` (e.g. ``cnceleb_concatenated``) and to ``is_latest``.
    """
    if not model_name:
        return {}
    csv_path = os.path.join(repo_root, "results", "test_eval", "metrics",
                            model_name, "eer_leaderboard.csv")
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
    return {row["exp"]: float(row[eer_col])
            for _, row in df.iterrows() if pd.notna(row[eer_col])}


def _infer_model_name(experiments: List[Tuple[str, Dict]]) -> Optional[str]:
    """ECAPA-TDNN vs ResNet34 — pick whichever appears in the dirnames."""
    names = [info["dirname"] for _, info in experiments]
    if any("ecapa_tdnn" in n for n in names):
        return "ecapa_tdnn"
    if any("resnet34" in n for n in names):
        return "resnet34"
    return None


# ---------------------------------------------------------------------------
# Cross-experiment plots
# ---------------------------------------------------------------------------

def _scatter_with_styles(ax, summaries: List[Dict], xfn, yfn) -> None:
    seen_labels = set()
    for s in summaries:
        info = s["info"]
        x, y = xfn(s), yfn(s)
        if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
            continue
        color, marker, _ = get_style(info)
        label = make_label(info)
        kwargs = dict(color=color, marker=marker, s=60, linewidth=0.8, zorder=3)
        # 'x' / '+' are unfilled glyphs — matplotlib ignores edgecolor on
        # them and warns. Apply edgecolor only to filled markers.
        if marker not in ("x", "+", "1", "2", "3", "4"):
            kwargs.update(edgecolor="black", linewidth=0.5)
        if label not in seen_labels:
            kwargs["label"] = label
            seen_labels.add(label)
        ax.scatter([x], [y], **kwargs)


def plot_flops_vs_sparsity(summaries: List[Dict], out_path: str) -> None:
    setup_matplotlib(font_size=10)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot([0, 100], [0, 100], **PERFECT_LINE_KW, label="perfect speedup ($y=x$)")

    _scatter_with_styles(
        ax, summaries,
        xfn=lambda s: s["agg"]["nominal_sparsity"] * 100,
        yfn=lambda s: (1 - s["agg"]["structural_density"]) * 100,
    )

    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_xlabel("Nominal sparsity (\%)" if plt.rcParams.get("text.usetex") else "Nominal sparsity (%)")
    ax.set_ylabel("Realizable compute reduction (\%)" if plt.rcParams.get("text.usetex")
                  else "Realizable compute reduction (%)")
    ax.set_title("Structured sparsity hugs the diagonal; unstructured does not", fontsize=9)
    ax.legend(fontsize=7, loc="upper left", frameon=True, framealpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_structural_density_vs_sparsity(summaries: List[Dict], out_path: str) -> None:
    setup_matplotlib(font_size=10)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot([0, 100], [100, 0], **PERFECT_LINE_KW, label="perfect structural reduction ($y=100-x$)")
    _scatter_with_styles(
        ax, summaries,
        xfn=lambda s: s["agg"]["nominal_sparsity"] * 100,
        yfn=lambda s: s["agg"]["structural_density"] * 100,
    )
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_xlabel("Nominal sparsity (\%)" if plt.rcParams.get("text.usetex") else "Nominal sparsity (%)")
    ax.set_ylabel("Structural density (\%)" if plt.rcParams.get("text.usetex") else "Structural density (%)")
    ax.set_title("Realizable compute density after collapsing zero rows/cols", fontsize=9)
    ax.legend(fontsize=7, loc="best", frameon=True, framealpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_eer_vs_effective_flops(summaries: List[Dict], out_path: str) -> None:
    setup_matplotlib(font_size=10)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))

    def _x(s):
        d = s["agg"]["structural_density"]
        # Using density on a log axis avoids cramping the [0.01, 1.0] range.
        return d if d > 0 else None

    def _y(s):
        return s.get("eer", None)

    _scatter_with_styles(ax, summaries, xfn=_x, yfn=lambda s: _y(s) * 100 if _y(s) is not None else None)

    # Dense baseline reference if present
    dense = [s for s in summaries if s["agg"]["nominal_sparsity"] < 0.01 and s.get("eer") is not None]
    if dense:
        eer_dense = min(s["eer"] for s in dense) * 100
        ax.axhline(eer_dense, color="#888", linestyle=":", linewidth=0.8,
                   label=f"dense baseline ({eer_dense:.2f}%)")

    ax.set_xscale("log")
    ax.set_xlim(0.005, 1.05)
    ax.set_xlabel("Realizable compute density (log scale)")
    ax.set_ylabel("EER (\%)" if plt.rcParams.get("text.usetex") else "EER (%)")
    ax.set_title("Pareto: EER vs realizable compute", fontsize=9)
    ax.legend(fontsize=7, loc="best", frameon=True, framealpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Shared layerwise helper
# ---------------------------------------------------------------------------
# All cross-experiment per-layer sparsity plots (overlay-rates, per-rate
# cross-model, no-bucket cross-model) call ``_plot_layerwise_panel`` for the
# inner panel work, then assemble a single legend via
# ``_assemble_layerwise_legend`` and emit it with ``_emit_layerwise_legend``.
# This guarantees consistent figure size, panel size, axis labels, twin-axis
# parameter bars, and legend ordering across every layerwise PDF.


def _plot_layerwise_panel(
    ax: "plt.Axes",
    members: List[Dict],
    *,
    rate_linestyles: Optional[Dict[int, Any]] = None,
    target_rates: Optional[List[int]] = None,
    show_param_bars: bool = True,
    show_target_lines: bool = True,
) -> Dict[str, Any]:
    """Render one layerwise-sparsity panel.

    ``members``   summaries to draw on this axis.
    ``rate_linestyles`` if set, override each curve's linestyle by its
                  integer sparsity rate (overlay-rates mode); else use the
                  linestyle returned by ``get_style``.
    ``target_rates`` rates to draw as horizontal reference lines; if None,
                  derive from ``members``.
    Returns dict with: legend_handles, bar_handle, target_handle, twin_ax,
    canon_names.
    """
    canon = max(members, key=lambda s: len(s["layers"]))
    canon_names = [l["name"] for l in canon["layers"]]
    canon_nparams = [max(int(l.get("n_params", 0)), 1) for l in canon["layers"]]
    x = np.arange(len(canon_names))

    twin_ax = None
    bar_handle = None
    if show_param_bars:
        twin_ax = ax.twinx()
        twin_ax.set_yscale("log")
        bars = twin_ax.bar(
            x, canon_nparams,
            color=PARAM_BAR_COLOR, alpha=PARAM_BAR_ALPHA,
            width=0.85, zorder=0, edgecolor="none",
        )
        if len(bars) > 0:
            bar_handle = bars[0]
        ax.set_zorder(twin_ax.get_zorder() + 1)
        ax.patch.set_visible(False)

    legend_handles: "OrderedDict[str, Tuple[Any, Dict]]" = OrderedDict()
    for s in members:
        info_local = {k: v for k, v in s["info"].items() if k != "_gradient_color"}
        color, marker, ls = get_style(info_local)
        if rate_linestyles is not None:
            ls = rate_linestyles.get(info_local.get("sparsity"), ls)
        label = make_label(info_local)
        lookup = {l["name"]: l["nominal_sparsity"] * 100 for l in s["layers"]}
        ys = [lookup.get(name, np.nan) for name in canon_names]
        line, = ax.plot(
            x, ys, color=color, marker=marker, linestyle=ls,
            markersize=4, linewidth=1.0, alpha=0.9, zorder=3,
        )
        if label not in legend_handles:
            legend_handles[label] = (line, info_local)

    target_handle = None
    if show_target_lines:
        rates = (
            target_rates
            if target_rates is not None
            else sorted({s["info"].get("sparsity") for s in members
                         if s["info"].get("sparsity") is not None})
        )
        for r in rates:
            ax.axhline(r, **PERFECT_LINE_KW)
        if rates:
            target_handle = plt.Line2D(
                [0], [0],
                color=PERFECT_LINE_KW["color"],
                linestyle=PERFECT_LINE_KW["linestyle"],
                linewidth=PERFECT_LINE_KW["linewidth"],
                alpha=PERFECT_LINE_KW["alpha"],
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [_short_layer_name(n, max_len=24) for n in canon_names],
        rotation=60, ha="right", fontsize=6,
    )
    ax.set_xlabel("Layer (input → output)", fontsize=9)

    return dict(
        legend_handles=legend_handles,
        bar_handle=bar_handle,
        target_handle=target_handle,
        twin_ax=twin_ax,
        canon_names=canon_names,
    )


def _assemble_layerwise_legend(
    legend_handles: "OrderedDict[str, Tuple[Any, Dict]]",
    *,
    target_handle: Optional[Any] = None,
    target_label: Optional[str] = None,
    bar_handle: Optional[Any] = None,
    bar_label: Optional[str] = None,
) -> Tuple[List[Any], List[str]]:
    """Sort method handles by ``experiment_sort_key``, then append target/bar entries."""
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


def _emit_layerwise_legend(
    fig: "plt.Figure",
    out_path: str,
    handles: List[Any],
    labels: List[str],
    *,
    legend_mode: str,
) -> None:
    """Either embed the legend at the bottom of fig (inline) or write a sidecar PDF (split)."""
    ncol = min(6, max(2, len(labels)))
    if legend_mode == "inline":
        fig.legend(
            handles, labels,
            loc="lower center", ncol=ncol, fontsize=7,
            frameon=True, framealpha=0.8, bbox_to_anchor=(0.5, -0.02),
        )
        fig.tight_layout(rect=(0, 0.08, 1, 1))
    elif legend_mode == "split":
        legend_path = os.path.splitext(out_path)[0] + "_legend.pdf"
        export_standalone_legend(
            handles, labels, legend_path, ncol, font_size=8,
        )
        fig.tight_layout()
    else:
        raise ValueError(
            f"legend_mode must be 'inline' or 'split', got {legend_mode!r}"
        )


def plot_layerwise_overlay_rates(
    summaries: List[Dict], out_path: str, *,
    legend_mode: str = "inline",
    show_param_bars: bool = True,
    show_target_lines: bool = True,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """Single-panel overlay: every sparsity-rate group on one axes.

    Curves at different rates are distinguished by ``RATE_LINESTYLES``;
    color/marker still encode method/variant. Comparing one backbone's
    behaviour across multiple rates without separate per-rate figures.
    """
    setup_matplotlib(font_size=10)
    members = [s for s in summaries
               if s.get("layers") and s["info"].get("sparsity") is not None]
    if not members:
        return

    fig, ax = plt.subplots(figsize=layerwise_figsize(1))
    panel = _plot_layerwise_panel(
        ax, members,
        rate_linestyles=RATE_LINESTYLES,
        target_rates=sorted({s["info"]["sparsity"] for s in members}),
        show_param_bars=show_param_bars,
        show_target_lines=show_target_lines,
    )

    usetex = plt.rcParams.get("text.usetex")
    ax.set_ylabel(
        r"$\mathsf{s}(\theta)$ [\%]" if usetex else "Per-layer sparsity (%)"
    )
    ax.set_ylim(*(ylim or (0, 101)))
    if panel["twin_ax"] is not None:
        panel["twin_ax"].set_ylabel(
            r"\# Parameters" if usetex else "# Parameters", fontsize=8,
        )
        panel["twin_ax"].tick_params(axis="y", labelsize=7)

    target_label = r"$\mathsf{s}^*$" if usetex else "target s*"
    bar_label = (r"\# parameters" if usetex else "# parameters") if panel["bar_handle"] else None
    handles, labels = _assemble_layerwise_legend(
        panel["legend_handles"],
        target_handle=panel["target_handle"], target_label=target_label,
        bar_handle=panel["bar_handle"], bar_label=bar_label,
    )
    _emit_layerwise_legend(fig, out_path, handles, labels, legend_mode=legend_mode)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_layerwise_per_rate_cross_model(
    summaries: List[Dict], out_dir: str, *,
    legend_mode: str = "inline",
    show_param_bars: bool = True,
    show_target_lines: bool = True,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """One PDF per sparsity target, with one panel per backbone.

    For each integer rate present in ``summaries`` build an N-panel figure
    (one per known model in ``MODEL_REGISTRY``, ordered by ``panel_order``).
    Per-rate y-limits come from ``YLIM_PER_RATE`` unless ``ylim`` overrides.
    """
    setup_matplotlib(font_size=9)
    from collections import defaultdict

    groups: Dict[int, List[Dict]] = defaultdict(list)
    for s in summaries:
        rate = s["info"].get("sparsity")
        if rate is None or not s.get("layers"):
            continue
        groups[rate].append(s)

    for rate, members in sorted(groups.items()):
        by_model: Dict[str, List[Dict]] = defaultdict(list)
        for s in members:
            model = s["info"].get("model")
            if model:
                by_model[model].append(s)
        models = panel_models(by_model)
        if not models:
            continue

        n_models = len(models)
        fig, axes = plt.subplots(
            1, n_models, figsize=layerwise_figsize(n_models), sharey=True,
        )
        if n_models == 1:
            axes = [axes]

        usetex = plt.rcParams.get("text.usetex", False)
        legend_handles: "OrderedDict[str, Tuple[Any, Dict]]" = OrderedDict()
        bar_handle = None
        target_handle = None
        twins: List[Any] = []

        for ax, model in zip(axes, models):
            panel = _plot_layerwise_panel(
                ax, by_model[model],
                target_rates=[rate],
                show_param_bars=show_param_bars,
                show_target_lines=show_target_lines,
            )
            for label, entry in panel["legend_handles"].items():
                if label not in legend_handles:
                    legend_handles[label] = entry
            if bar_handle is None:
                bar_handle = panel["bar_handle"]
            if target_handle is None:
                target_handle = panel["target_handle"]
            if panel["twin_ax"] is not None:
                twins.append(panel["twin_ax"])

        ylim_lo, ylim_hi = ylim if ylim is not None else ylim_for_rate(rate, scale="percent")
        axes[0].set_ylim(ylim_lo, ylim_hi)
        axes[0].set_ylabel(
            r"$\mathsf{s}(\theta)$ [\%]" if usetex else "Per-layer sparsity (%)"
        )

        for i, ax2 in enumerate(twins):
            if i < len(twins) - 1:
                ax2.set_yticklabels([])
                ax2.tick_params(right=False)
            else:
                ax2.set_ylabel(
                    r"\# Parameters" if usetex else "# Parameters", fontsize=8,
                )

        target_label = (rf"$\mathsf{{s}}^*$={rate}\%" if usetex else f"s={rate}%")
        bar_label = (r"\# parameters" if usetex else "# parameters") if bar_handle else None
        handles, labels = _assemble_layerwise_legend(
            legend_handles,
            target_handle=target_handle, target_label=target_label,
            bar_handle=bar_handle, bar_label=bar_label,
        )
        out_path = os.path.join(out_dir, f"cross_model_layerwise_sr{rate}.pdf")
        _emit_layerwise_legend(fig, out_path, handles, labels, legend_mode=legend_mode)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def plot_layerwise_no_bucket(
    summaries: List[Dict], out_path: str, *,
    legend_mode: str = "split",
    show_param_bars: bool = True,
    show_target_lines: bool = True,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """One panel per backbone, all rates mixed (no rate bucketing).

    Useful for paper figures comparing e.g. fixed-λ + sr75 + sr90 runs
    side-by-side per backbone. Curve linestyle still comes from ``get_style``
    (rate-driven), so different rates remain distinguishable within a panel.
    """
    setup_matplotlib(font_size=9)
    from collections import defaultdict

    by_model: Dict[str, List[Dict]] = defaultdict(list)
    for s in summaries:
        m = s["info"].get("model")
        if not m or not s.get("layers"):
            continue
        by_model[m].append(s)

    models = panel_models(by_model)
    if not models:
        return

    n_models = len(models)
    fig, axes = plt.subplots(
        1, n_models, figsize=layerwise_figsize(n_models), sharey=True,
    )
    if n_models == 1:
        axes = [axes]

    usetex = plt.rcParams.get("text.usetex", False)
    legend_handles: "OrderedDict[str, Tuple[Any, Dict]]" = OrderedDict()
    bar_handle = None
    target_handle = None
    twins: List[Any] = []

    for ax, model in zip(axes, models):
        panel = _plot_layerwise_panel(
            ax, by_model[model],
            show_param_bars=show_param_bars,
            show_target_lines=show_target_lines,
        )
        for label, entry in panel["legend_handles"].items():
            if label not in legend_handles:
                legend_handles[label] = entry
        if bar_handle is None:
            bar_handle = panel["bar_handle"]
        if target_handle is None:
            target_handle = panel["target_handle"]
        if panel["twin_ax"] is not None:
            twins.append(panel["twin_ax"])

    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_ylabel(
        r"Per-layer sparsity (\%)" if usetex else "Per-layer sparsity (%)"
    )

    for i, ax2 in enumerate(twins):
        if i < len(twins) - 1:
            ax2.set_yticklabels([])
            ax2.tick_params(right=False)
        else:
            ax2.set_ylabel(
                r"\# Parameters" if usetex else "# Parameters", fontsize=8,
            )

    target_label = "target sparsity"
    bar_label = (r"\# parameters" if usetex else "# parameters") if bar_handle else None
    handles, labels = _assemble_layerwise_legend(
        legend_handles,
        target_handle=target_handle, target_label=target_label,
        bar_handle=bar_handle, bar_label=bar_label,
    )
    _emit_layerwise_legend(fig, out_path, handles, labels, legend_mode=legend_mode)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_rtf_vs_sparsity(summaries: List[Dict], out_path: str) -> None:
    setup_matplotlib(font_size=10)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    has_rtf = [s for s in summaries if s.get("rtf") is not None]
    if not has_rtf:
        plt.close(fig)
        return
    _scatter_with_styles(
        ax, has_rtf,
        xfn=lambda s: s["agg"]["nominal_sparsity"] * 100,
        yfn=lambda s: s["rtf"]["rtf_median"],
    )
    dense = [s for s in has_rtf if s["agg"]["nominal_sparsity"] < 0.01]
    if dense:
        ax.axhline(dense[0]["rtf"]["rtf_median"], color="#888",
                   linestyle=":", linewidth=0.8, label="dense baseline")
    device = has_rtf[0]["rtf"]["device"]
    ax.set_xlabel("Nominal sparsity (\%)" if plt.rcParams.get("text.usetex") else "Nominal sparsity (%)")
    ax.set_ylabel("RTF (median)")
    ax.set_title(f"Measured RTF on {device}", fontsize=9)
    ax.legend(fontsize=7, loc="best", frameon=True, framealpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def process_experiment(exp_dir: str, info: Dict, out_root: str,
                       rtf_args: Optional[Dict],
                       *, annotate_layerwise: bool = False) -> Optional[Dict]:
    ckpt = find_best_ckpt(exp_dir, info=info)
    if ckpt is None:
        log.warning("No checkpoint found in %s — skipping", exp_dir)
        return None
    log.info("→ %s", info["dirname"])
    log.info("   ckpt: %s", os.path.relpath(ckpt, exp_dir))

    layers = extract_pruned_layers(ckpt)
    if not layers:
        log.warning("No prunable layers found in %s — recording as dense baseline",
                    info["dirname"])

    enriched = []
    for l in layers:
        l = dict(l)
        l["stats"] = layer_stats(l["weight"], l["mask"])
        enriched.append(l)

    agg = aggregate_stats(enriched)
    log.info("   nominal=%.1f%% structural_density=%.1f%% structuredness=%.4f "
             "fully-zero row/col=%.0f%%/%.0f%%",
             agg["nominal_sparsity"] * 100,
             agg["structural_density"] * 100,
             agg["mean_structuredness"],
             agg["mean_fully_zero_row_frac"] * 100,
             agg["mean_fully_zero_col_frac"] * 100)

    out_dir = os.path.join(out_root, info["dirname"])
    os.makedirs(out_dir, exist_ok=True)

    title = f"{make_label(info)}"
    if enriched:
        plot_layerwise_sparsity(enriched, os.path.join(out_dir, "layerwise_sparsity.pdf"), title,
                                annotate=annotate_layerwise)
        plot_layerwise_sparsity_nparams(enriched, os.path.join(out_dir, "layerwise_sparsity_nparams.pdf"), title)
        plot_per_filter_histograms(enriched, os.path.join(out_dir, "per_filter_sparsity_hist.pdf"), title)
        plot_mask_heatmaps(enriched, os.path.join(out_dir, "mask_heatmap.pdf"), title)

    rtf = None
    if rtf_args is not None:
        rtf = measure_rtf(
            exp_dir, ckpt, enriched,
            device=rtf_args["device"],
            audio_seconds=rtf_args["audio_seconds"],
            sample_rate=rtf_args["sample_rate"],
            warmup=rtf_args["warmup"],
            iters=rtf_args["iters"],
        )
        if rtf:
            log.info("   RTF (median, %s) = %.4f", rtf["device"], rtf["rtf_median"])

    summary = dict(
        exp_dir=exp_dir,
        info={k: v for k, v in info.items() if not k.startswith("_")},
        ckpt=ckpt,
        agg=agg,
        layers=[
            dict(
                name=l["name"], kind=l["kind"], shape=list(l["shape"]),
                # Drop large arrays from the JSON; keep scalars only.
                **{k: v for k, v in l["stats"].items()
                   if k not in ("per_row_density", "per_col_density")},
            )
            for l in enriched
        ],
        rtf=rtf,
    )
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=_json_safe)
    return dict(
        info=info, agg=agg, rtf=rtf, exp_dir=exp_dir,
        # Lean per-layer slice for cross-experiment plots (no big arrays).
        layers=[
            dict(name=l["name"], kind=l["kind"],
                 nominal_sparsity=l["stats"]["nominal_sparsity"],
                 n_params=l["stats"]["n_params"])
            for l in enriched
        ],
    )


def _json_safe(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray, torch.Tensor)):
        return None  # already filtered
    raise TypeError(f"Not JSON-serializable: {type(o)}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_dirs", nargs="+", required=True,
                    help="One or more dataset roots, e.g. /data/aloradad/results/cnceleb")
    ap.add_argument("--experiments", nargs="+", required=True,
                    help="Glob patterns matching experiment directory names")
    ap.add_argument("--output", required=True,
                    help="Output directory (per-experiment subfolders + cross_exp/)")
    ap.add_argument("--test_set", default=None,
                    help="Test set name to read EER from (e.g. cnceleb_concatenated)."
                         " Defaults to the first one available in the leaderboard CSV.")
    ap.add_argument("--max_heatmap_layers", type=int, default=10)
    ap.add_argument(
        "--legend-mode", dest="legend_mode",
        choices=["inline", "split"], default="split",
        help=(
            "Legend handling for the cross-model layerwise plot. "
            "inline: embed legend at the bottom (default). "
            "split: omit inline legend and save a shared "
            "<plot>_legend.pdf for LaTeX side-by-side use."
        ),
    )
    ap.add_argument(
        "--layerwise-mode", dest="layerwise_mode",
        choices=["overlay-rates", "per-rate-cross-model", "no-bucket", "all"],
        default="all",
        help=(
            "Which layerwise PDF(s) to emit under cross_exp/. "
            "overlay-rates: one figure with every rate overlaid on a single panel "
            "(layerwise_sparsity_all_rates.pdf). "
            "per-rate-cross-model: one figure per rate with one panel per backbone "
            "(cross_model_layerwise_sr{rate}.pdf). "
            "no-bucket: one figure with one panel per backbone, all rates mixed "
            "(overlay_layerwise.pdf). "
            "all: emit all three (default; reproduces the previous behaviour)."
        ),
    )
    ap.add_argument(
        "--annotate-layerwise", dest="annotate_layerwise",
        action="store_true",
        help="Annotate per-experiment layerwise sparsity bars with fully-zero "
             "row/column fractions.",
    )
    ap.add_argument(
        "--layerwise-ylim", dest="layerwise_ylim",
        nargs=2, type=float, default=None, metavar=("LO", "HI"),
        help="Override y-axis limits for layerwise plots in percent. "
             "Default: per-rate from visualize_common.YLIM_PER_RATE.",
    )
    ap.add_argument(
        "--no-param-bars", dest="show_param_bars",
        action="store_false",
        help="Disable the n_params twin-axis bars on layerwise plots.",
    )
    ap.add_argument(
        "--no-target-lines", dest="show_target_lines",
        action="store_false",
        help="Disable target-rate horizontal reference lines.",
    )
    # RTF
    ap.add_argument("--rtf", action="store_true",
                    help="Measure forward-pass RTF on the trained encoder")
    ap.add_argument("--rtf_device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--rtf_audio_seconds", type=float, default=5.0)
    ap.add_argument("--rtf_sample_rate", type=int, default=16000)
    ap.add_argument("--rtf_warmup", type=int, default=10)
    ap.add_argument("--rtf_iters", type=int, default=50)
    args = ap.parse_args()

    setup_matplotlib(font_size=10)
    os.makedirs(args.output, exist_ok=True)
    log.info("Discovering experiments under %s matching %s",
             args.base_dirs, args.experiments)
    experiments = discover_experiments(args.base_dirs, args.experiments)
    if not experiments:
        log.error("No experiments matched the patterns.")
        return 1
    log.info("Matched %d experiments", len(experiments))

    rtf_args = None
    if args.rtf:
        rtf_args = dict(
            device=args.rtf_device,
            audio_seconds=args.rtf_audio_seconds,
            sample_rate=args.rtf_sample_rate,
            warmup=args.rtf_warmup,
            iters=args.rtf_iters,
        )

    summaries: List[Dict] = []
    for exp_dir, info in experiments:
        try:
            s = process_experiment(
                exp_dir, info, args.output, rtf_args,
                annotate_layerwise=args.annotate_layerwise,
            )
        except Exception as e:
            log.exception("Failed on %s: %s", info["dirname"], e)
            continue
        if s is not None:
            summaries.append(s)

    if not summaries:
        log.error("No experiments produced summaries — nothing to aggregate.")
        return 1

    # Optionally pull EER for each experiment from the aggregator output
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_name = _infer_model_name(experiments)
    eer_lookup = load_eer_lookup(model_name, args.test_set, repo_root)
    if not eer_lookup:
        # Fall back to whatever's in the leaderboard, picking the first dataset.
        eer_lookup = load_eer_lookup(model_name, None, repo_root)
    log.info("EER readout: %d/%d experiments matched in leaderboard",
             sum(1 for s in summaries if s["info"]["dirname"] in eer_lookup),
             len(summaries))
    for s in summaries:
        s["eer"] = eer_lookup.get(s["info"]["dirname"])

    cross_dir = os.path.join(args.output, "cross_exp")
    os.makedirs(cross_dir, exist_ok=True)
    plot_flops_vs_sparsity(summaries, os.path.join(cross_dir, "flops_vs_sparsity.pdf"))
    plot_structural_density_vs_sparsity(summaries, os.path.join(cross_dir, "structural_density_vs_sparsity.pdf"))
    plot_eer_vs_effective_flops(summaries, os.path.join(cross_dir, "eer_vs_effective_flops.pdf"))

    layerwise_ylim = tuple(args.layerwise_ylim) if args.layerwise_ylim else None
    common_kw = dict(
        legend_mode=args.legend_mode,
        show_param_bars=args.show_param_bars,
        show_target_lines=args.show_target_lines,
        ylim=layerwise_ylim,
    )
    modes = (
        ["overlay-rates", "per-rate-cross-model", "no-bucket"]
        if args.layerwise_mode == "all" else [args.layerwise_mode]
    )
    if "overlay-rates" in modes:
        plot_layerwise_overlay_rates(
            summaries,
            os.path.join(cross_dir, "layerwise_sparsity_all_rates.pdf"),
            **common_kw,
        )
    if "per-rate-cross-model" in modes:
        plot_layerwise_per_rate_cross_model(summaries, cross_dir, **common_kw)
    if "no-bucket" in modes:
        plot_layerwise_no_bucket(
            summaries,
            os.path.join(cross_dir, "overlay_layerwise.pdf"),
            **common_kw,
        )
    if rtf_args is not None:
        plot_rtf_vs_sparsity(summaries, os.path.join(cross_dir, "rtf_vs_sparsity.pdf"))

    log.info("Done. Output → %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
