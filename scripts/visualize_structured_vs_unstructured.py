#!/usr/bin/env python3
"""Diagnose structured vs unstructured sparsity in trained SV checkpoints.

Per-experiment plots (one folder per discovered run):
    layerwise_sparsity.pdf       — per-layer sparsity bars, colored by
                                   structuredness score
    layerwise_sparsity_nparams.pdf — same bars colored by parameter count
    per_filter_sparsity_hist.pdf — distribution of per-row / per-column
                                   density: bimodal ⇒ structured
    mask_heatmap.pdf             — binary mask of representative layers
    summary.json                 — per-experiment metrics dump

Head-anatomy plots (``--head_anatomy``, in ``{exp}/head_anatomy/``) explain the
bright column the heatmap shows on an over-sized head — see the docstring of
``src/vis/head_anatomy.py``. Requires a head with more rows than the dataset
has classes.

Cross-experiment plots (in {output}/cross_exp/):
    flops_vs_sparsity.pdf
    structural_density_vs_sparsity.pdf
    eer_vs_effective_flops.pdf
    rtf_vs_sparsity.pdf          (--rtf only)

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

    # Explain the head stripe, including a forward pass for pooled features
    python scripts/visualize_structured_vs_unstructured.py \\
        --base_dirs /data/aloradad/results/cifar10/resnet18 \\
        --experiments 'bregman_*-resnet18-cifar10-*sr99-classifier_10k' \\
        --head_anatomy --activations --data_dir data \\
        --output results/img/cifar10/resnet18/head_anatomy
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from visualize import discover_experiments  # noqa: E402

from src.vis.common import (  # noqa: E402
    RATE_LINESTYLES,
    layerwise_figsize,
    make_label,
    panel_models,
    setup_matplotlib,
    ylim_for_rate,
)
from src.vis.head_anatomy import (  # noqa: E402
    find_checkpoints,
    read_head_anatomy,
    read_pooled_features,
    real_class_count,
    render_ckpt_evolution,
    render_head_anatomy,
    render_norm_inventory,
    render_norm_shift_vs_stripe,
    render_pooled_features,
    summarize,
)
from src.vis.pruning_compare import (  # noqa: E402
    _infer_model_name,
    _json_safe,
    assemble_layerwise_legend,
    emit_layerwise_legend,
    read_eer_lookup,
    read_pruning_experiment,
    read_rtf,
    render_eer_vs_effective_flops,
    render_flops_vs_sparsity,
    render_layerwise_panel,
    render_layerwise_sparsity,
    render_layerwise_sparsity_nparams,
    render_mask_heatmaps,
    render_per_filter_histograms,
    render_rtf_vs_sparsity,
    render_structural_density_vs_sparsity,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("struct_vs_unstruct")


# ---------------------------------------------------------------------------
# Head anatomy
# ---------------------------------------------------------------------------


def _emit_head_anatomy(
    exp_dir: str,
    out_dir: str,
    title: str,
    opts: Dict,
) -> None:
    """Write the head-anatomy figures for one run into
    ``{out_dir}/head_anatomy``.

    Args:
        exp_dir: representative seed directory (holds ``checkpoints/``, ``.hydra/``).
        out_dir: the run's output folder.
        title: figure title.
        opts: parsed ``--head_anatomy*`` / ``--activations*`` options.
    """
    anat_dir = os.path.join(out_dir, "head_anatomy")
    os.makedirs(anat_dir, exist_ok=True)

    ckpts = find_checkpoints(exp_dir, limit=opts["n_ckpts"])
    n_real = real_class_count(exp_dir, _PROJECT_ROOT)
    log.info(f"   head anatomy: {len(ckpts)} ckpt(s), {n_real} real classes")
    series = [read_head_anatomy(c, n_real) for c in ckpts]
    latest = series[-1]
    log.info(
        f"   stripe col={latest['stripe_col']}"
        f" share={latest['stripe_energy_share']:.3f}"
        f" | {latest['norm_name']} beta"
        f" argmax={int(latest['norm_beta'].argmax())}"
        f" max={latest['norm_beta'].max():.1f}"
    )

    fig, axes = plt.subplots(3, 1, figsize=(6.4, 5.6), sharex=True)
    render_norm_shift_vs_stripe(latest, axes)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(anat_dir, "norm_shift_vs_stripe.pdf"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 2.9))
    render_head_anatomy(latest, axes)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(anat_dir, "head_anatomy.pdf"))
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(7.6, 7.0))
    render_norm_inventory(latest, axes)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=3.0)
    fig.savefig(os.path.join(anat_dir, "norm_inventory.pdf"))
    plt.close(fig)

    if len(series) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0))
        render_ckpt_evolution(series, axes)
        fig.suptitle(title, fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(os.path.join(anat_dir, "ckpt_evolution.pdf"))
        plt.close(fig)

    feat = None
    if opts["activations"]:
        feat = read_pooled_features(
            exp_dir,
            ckpts[-1],
            opts["data_dir"],
            n_batches=opts["activation_batches"],
            device=opts["activation_device"],
        )
        fig, axes = plt.subplots(1, 3, figsize=(10.0, 2.9))
        render_pooled_features(feat, latest, axes)
        fig.suptitle(title, fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(os.path.join(anat_dir, "pooled_features.pdf"))
        plt.close(fig)

    with open(os.path.join(anat_dir, "head_anatomy.json"), "w") as f:
        json.dump(
            dict(
                exp_dir=exp_dir,
                per_checkpoint=[summarize(a) for a in series],
                latest=summarize(latest, feat),
            ),
            f,
            indent=2,
            default=_json_safe,
        )


# ---------------------------------------------------------------------------
# Per-experiment orchestration
# ---------------------------------------------------------------------------


def _process_experiment(
    exp_dir: str,
    info: Dict,
    out_root: str,
    rtf_args: Optional[Dict],
    *,
    annotate_layerwise: bool = False,
    include_all_layers: bool = False,
    head_anatomy_opts: Optional[Dict] = None,
) -> Optional[Dict]:
    data = read_pruning_experiment(
        exp_dir,
        info,
        include_all_layers=include_all_layers,
    )
    if data is None:
        return None

    layers_enriched = data["layers_enriched"]
    agg = data["agg"]
    out_dir = os.path.join(out_root, info["dirname"])
    os.makedirs(out_dir, exist_ok=True)
    title = make_label(info)

    if layers_enriched:
        n = len(layers_enriched)
        fig_h = max(2.4, 0.18 * n + 1.0)

        fig, ax = plt.subplots(figsize=(6.4, fig_h))
        render_layerwise_sparsity(
            layers_enriched,
            ax,
            annotate=annotate_layerwise,
        )
        ax.set_title(title, fontsize=9)
        fig.savefig(
            os.path.join(out_dir, "layerwise_sparsity.pdf"),
        )
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.4, fig_h))
        render_layerwise_sparsity_nparams(layers_enriched, ax)
        ax.set_title(title, fontsize=9)
        fig.savefig(
            os.path.join(out_dir, "layerwise_sparsity_nparams.pdf"),
        )
        plt.close(fig)

        max_hist_panels = 12
        n_pick = min(max_hist_panels, len(layers_enriched))
        cols = min(4, n_pick)
        rows = int(np.ceil(n_pick / cols))
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(cols * 2.5, rows * 1.9),
            sharex=True,
        )
        render_per_filter_histograms(layers_enriched, axes, max_hist_panels)
        fig.suptitle(
            title + (r"  —  per-row (blue) / per-column (red) density"),
            fontsize=9,
        )
        fig.supxlabel(
            "Per-row / per-column density (1 = retained)",
            fontsize=8,
        )
        fig.supylabel("Count", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(
            os.path.join(out_dir, "per_filter_sparsity_hist.pdf"),
        )
        plt.close(fig)

        max_heatmap_panels = 24
        n_pick_hm = min(max_heatmap_panels, len(layers_enriched))
        cols_hm = min(3, n_pick_hm)
        rows_hm = int(np.ceil(n_pick_hm / cols_hm))
        fig, axes = plt.subplots(
            rows_hm,
            cols_hm,
            figsize=(cols_hm * 2.6, rows_hm * 2.2),
        )
        render_mask_heatmaps(
            layers_enriched,
            axes,
            max_heatmap_panels,
        )
        fig.suptitle(title, fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(os.path.join(out_dir, "mask_heatmap.pdf"))
        plt.close(fig)

    if head_anatomy_opts is not None:
        _emit_head_anatomy(exp_dir, out_dir, title, head_anatomy_opts)

    rtf = None
    if rtf_args is not None:
        rtf = read_rtf(
            exp_dir,
            data["ckpt"],
            layers_enriched,
            device=rtf_args["device"],
            audio_seconds=rtf_args["audio_seconds"],
            sample_rate=rtf_args["sample_rate"],
            warmup=rtf_args["warmup"],
            iters=rtf_args["iters"],
        )
        if rtf:
            log.info(
                f"   RTF (median, {rtf['device']}) = {rtf['rtf_median']:.4f}"
            )

    summary = dict(
        exp_dir=exp_dir,
        info=data["info"],
        ckpt=data["ckpt"],
        agg=agg,
        layers=[
            dict(
                name=l["name"],
                kind=l["kind"],
                shape=list(l["shape"]),
                **{
                    k: v
                    for k, v in l["stats"].items()
                    if k not in ("per_row_density", "per_col_density")
                },
            )
            for l in layers_enriched
        ],
        rtf=rtf,
    )
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=_json_safe)

    return dict(
        info=info,
        agg=agg,
        rtf=rtf,
        exp_dir=exp_dir,
        layers=data["layers_lean"],
    )


# ---------------------------------------------------------------------------
# Cross-experiment layerwise figures
# ---------------------------------------------------------------------------


def _layerwise_axis_labels(axes: List, twins: List) -> None:
    usetex = plt.rcParams.get("text.usetex", False)
    axes[0].set_ylabel(
        r"$\mathsf{s}(\theta)$ [\%]" if usetex else "Per-layer sparsity (%)"
    )
    for i, ax2 in enumerate(twins):
        if i < len(twins) - 1:
            ax2.set_yticklabels([])
            ax2.tick_params(right=False)
        else:
            ax2.set_ylabel(
                r"\# Parameters" if usetex else "# Parameters",
                fontsize=8,
            )
            ax2.tick_params(axis="y", labelsize=7)


def _emit_overlay_rates(
    summaries: List[Dict],
    out_path: str,
    *,
    legend_mode: str,
    show_param_bars: bool,
    show_target_lines: bool,
    ylim: Optional[Tuple],
) -> None:
    members = [
        s
        for s in summaries
        if s.get("layers") and s["info"].get("sparsity") is not None
    ]
    if not members:
        return

    usetex = plt.rcParams.get("text.usetex", False)
    fig, ax = plt.subplots(figsize=layerwise_figsize(1))
    panel = render_layerwise_panel(
        ax,
        members,
        rate_linestyles=RATE_LINESTYLES,
        target_rates=sorted({s["info"]["sparsity"] for s in members}),
        show_param_bars=show_param_bars,
        show_target_lines=show_target_lines,
    )

    ax.set_ylabel(
        r"$\mathsf{s}(\theta)$ [\%]" if usetex else "Per-layer sparsity (%)"
    )
    ax.set_ylim(*(ylim or (0, 101)))
    if panel["twin_ax"] is not None:
        panel["twin_ax"].set_ylabel(
            r"\# Parameters" if usetex else "# Parameters",
            fontsize=8,
        )
        panel["twin_ax"].tick_params(axis="y", labelsize=7)

    target_label = r"$\mathsf{s}^*$" if usetex else "target s*"
    bar_label = (
        (r"\# parameters" if usetex else "# parameters")
        if panel["bar_handle"]
        else None
    )
    handles, labels = assemble_layerwise_legend(
        panel["legend_handles"],
        target_handle=panel["target_handle"],
        target_label=target_label,
        bar_handle=panel["bar_handle"],
        bar_label=bar_label,
    )
    emit_layerwise_legend(
        fig,
        out_path,
        handles,
        labels,
        legend_mode=legend_mode,
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _emit_per_rate_cross_model(
    summaries: List[Dict],
    out_dir: str,
    *,
    legend_mode: str,
    show_param_bars: bool,
    show_target_lines: bool,
    ylim: Optional[Tuple],
) -> None:
    usetex = plt.rcParams.get("text.usetex", False)
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
            1,
            n_models,
            figsize=layerwise_figsize(n_models),
            sharey=True,
        )
        if n_models == 1:
            axes = [axes]

        all_handles = {}
        bar_handle = None
        target_handle = None
        twins: List = []

        for ax, model in zip(axes, models):
            panel = render_layerwise_panel(
                ax,
                by_model[model],
                target_rates=[rate],
                show_param_bars=show_param_bars,
                show_target_lines=show_target_lines,
            )
            for label, entry in panel["legend_handles"].items():
                if label not in all_handles:
                    all_handles[label] = entry
            if bar_handle is None:
                bar_handle = panel["bar_handle"]
            if target_handle is None:
                target_handle = panel["target_handle"]
            if panel["twin_ax"] is not None:
                twins.append(panel["twin_ax"])

        ylim_lo, ylim_hi = (
            ylim if ylim is not None else ylim_for_rate(rate, scale="percent")
        )
        axes[0].set_ylim(ylim_lo, ylim_hi)
        _layerwise_axis_labels(axes, twins)

        target_label = (
            rf"$\mathsf{{s}}^*$={rate}\%" if usetex else f"s={rate}%"
        )
        bar_label = (
            (r"\# parameters" if usetex else "# parameters")
            if bar_handle
            else None
        )
        handles, labels = assemble_layerwise_legend(
            all_handles,
            target_handle=target_handle,
            target_label=target_label,
            bar_handle=bar_handle,
            bar_label=bar_label,
        )
        out_path = os.path.join(out_dir, f"cross_model_layerwise_sr{rate}.pdf")
        emit_layerwise_legend(
            fig,
            out_path,
            handles,
            labels,
            legend_mode=legend_mode,
        )
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def _emit_no_bucket(
    summaries: List[Dict],
    out_path: str,
    *,
    legend_mode: str,
    show_param_bars: bool,
    show_target_lines: bool,
    ylim: Optional[Tuple],
) -> None:
    usetex = plt.rcParams.get("text.usetex", False)
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
        1,
        n_models,
        figsize=layerwise_figsize(n_models),
        sharey=True,
    )
    if n_models == 1:
        axes = [axes]

    all_handles = {}
    bar_handle = None
    target_handle = None
    twins: List = []

    for ax, model in zip(axes, models):
        panel = render_layerwise_panel(
            ax,
            by_model[model],
            show_param_bars=show_param_bars,
            show_target_lines=show_target_lines,
        )
        for label, entry in panel["legend_handles"].items():
            if label not in all_handles:
                all_handles[label] = entry
        if bar_handle is None:
            bar_handle = panel["bar_handle"]
        if target_handle is None:
            target_handle = panel["target_handle"]
        if panel["twin_ax"] is not None:
            twins.append(panel["twin_ax"])

    if ylim is not None:
        axes[0].set_ylim(*ylim)
    _layerwise_axis_labels(axes, twins)

    bar_label = (
        (r"\# parameters" if usetex else "# parameters")
        if bar_handle
        else None
    )
    handles, labels = assemble_layerwise_legend(
        all_handles,
        target_handle=target_handle,
        target_label="target sparsity",
        bar_handle=bar_handle,
        bar_label=bar_label,
    )
    emit_layerwise_legend(
        fig,
        out_path,
        handles,
        labels,
        legend_mode=legend_mode,
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--base_dirs",
        nargs="+",
        required=True,
        help="One or more dataset roots",
    )
    ap.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="Glob patterns matching experiment directory names",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output directory (per-experiment subfolders + cross_exp/)",
    )
    ap.add_argument(
        "--test_set",
        default=None,
        help="Test set name for EER readout (e.g. cnceleb_concatenated)",
    )
    ap.add_argument(
        "--legend-mode",
        dest="legend_mode",
        choices=["inline", "split"],
        default="split",
    )
    ap.add_argument(
        "--layerwise-mode",
        dest="layerwise_mode",
        choices=["overlay-rates", "per-rate-cross-model", "no-bucket", "all"],
        default="all",
    )
    ap.add_argument(
        "--annotate-layerwise",
        dest="annotate_layerwise",
        action="store_true",
    )
    ap.add_argument(
        "--all-layers",
        dest="all_layers",
        action="store_true",
        default=False,
        help="also plot 1-D parameters (norm scales), not just conv/linear weights",
    )
    ap.add_argument(
        "--layerwise-ylim",
        dest="layerwise_ylim",
        nargs=2,
        type=float,
        default=None,
        metavar=("LO", "HI"),
    )
    ap.add_argument(
        "--no-param-bars",
        dest="show_param_bars",
        action="store_false",
    )
    ap.add_argument(
        "--no-target-lines",
        dest="show_target_lines",
        action="store_false",
    )
    ap.add_argument(
        "--head_anatomy",
        action="store_true",
        help="Explain the head stripe; needs a head wider than the class count",
    )
    ap.add_argument(
        "--head_anatomy_ckpts",
        type=int,
        default=6,
        help="Read at most this many of the newest checkpoints",
    )
    ap.add_argument(
        "--activations",
        action="store_true",
        help="Also run a forward pass for pooled-feature statistics",
    )
    ap.add_argument(
        "--data_dir",
        default=os.path.join(_PROJECT_ROOT, "data"),
        help="Local dataset root, replacing the training-time path",
    )
    ap.add_argument("--activation_batches", type=int, default=16)
    ap.add_argument(
        "--activation_device",
        default="cpu",
        choices=["cpu", "cuda"],
    )
    ap.add_argument("--rtf", action="store_true")
    ap.add_argument("--rtf_device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--rtf_audio_seconds", type=float, default=5.0)
    ap.add_argument("--rtf_sample_rate", type=int, default=16000)
    ap.add_argument("--rtf_warmup", type=int, default=10)
    ap.add_argument("--rtf_iters", type=int, default=50)
    args = ap.parse_args()

    setup_matplotlib(font_size=10)
    os.makedirs(args.output, exist_ok=True)
    log.info(
        f"Discovering experiments under {args.base_dirs} matching {args.experiments}"
    )
    experiments = discover_experiments(args.base_dirs, args.experiments)
    if not experiments:
        log.error("No experiments matched the patterns.")
        return 1
    log.info(f"Matched {len(experiments)} experiments")

    rtf_args = None
    if args.rtf:
        rtf_args = dict(
            device=args.rtf_device,
            audio_seconds=args.rtf_audio_seconds,
            sample_rate=args.rtf_sample_rate,
            warmup=args.rtf_warmup,
            iters=args.rtf_iters,
        )

    head_anatomy_opts = None
    if args.head_anatomy:
        head_anatomy_opts = dict(
            n_ckpts=args.head_anatomy_ckpts,
            activations=args.activations,
            data_dir=args.data_dir,
            activation_batches=args.activation_batches,
            activation_device=args.activation_device,
        )

    summaries: List[Dict] = []
    for exp_dir, info in experiments:
        s = _process_experiment(
            exp_dir,
            info,
            args.output,
            rtf_args,
            annotate_layerwise=args.annotate_layerwise,
            include_all_layers=args.all_layers,
            head_anatomy_opts=head_anatomy_opts,
        )
        if s is not None:
            summaries.append(s)

    if not summaries:
        log.error("No experiments produced summaries — nothing to aggregate.")
        return 1

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_name = _infer_model_name(experiments)
    eer_lookup = read_eer_lookup(model_name, args.test_set, repo_root)
    if not eer_lookup:
        eer_lookup = read_eer_lookup(model_name, None, repo_root)
    n_eer_matched = sum(
        1 for s in summaries if s["info"]["dirname"] in eer_lookup
    )
    log.info(
        f"EER readout: {n_eer_matched}/{len(summaries)} experiments matched in leaderboard"
    )
    for s in summaries:
        s["eer"] = eer_lookup.get(s["info"]["dirname"])

    cross_dir = os.path.join(args.output, "cross_exp")
    os.makedirs(cross_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    render_flops_vs_sparsity(summaries, ax)
    fig.tight_layout()
    fig.savefig(
        os.path.join(cross_dir, "flops_vs_sparsity.pdf"),
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    render_structural_density_vs_sparsity(summaries, ax)
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            cross_dir,
            "structural_density_vs_sparsity.pdf",
        ),
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    render_eer_vs_effective_flops(summaries, ax)
    fig.tight_layout()
    fig.savefig(
        os.path.join(cross_dir, "eer_vs_effective_flops.pdf"),
    )
    plt.close(fig)

    layerwise_ylim = (
        tuple(args.layerwise_ylim) if args.layerwise_ylim else None
    )
    common_kw = dict(
        legend_mode=args.legend_mode,
        show_param_bars=args.show_param_bars,
        show_target_lines=args.show_target_lines,
        ylim=layerwise_ylim,
    )
    modes = (
        ["overlay-rates", "per-rate-cross-model", "no-bucket"]
        if args.layerwise_mode == "all"
        else [args.layerwise_mode]
    )
    if "overlay-rates" in modes:
        _emit_overlay_rates(
            summaries,
            os.path.join(
                cross_dir,
                "layerwise_sparsity_all_rates.pdf",
            ),
            **common_kw,
        )
    if "per-rate-cross-model" in modes:
        _emit_per_rate_cross_model(summaries, cross_dir, **common_kw)
    if "no-bucket" in modes:
        _emit_no_bucket(
            summaries,
            os.path.join(cross_dir, "overlay_layerwise.pdf"),
            **common_kw,
        )
    if rtf_args is not None:
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        drew = render_rtf_vs_sparsity(summaries, ax)
        if drew:
            fig.tight_layout()
            fig.savefig(
                os.path.join(cross_dir, "rtf_vs_sparsity.pdf"),
            )
        plt.close(fig)

    log.info(f"Done. Output → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
