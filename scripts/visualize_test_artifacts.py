#!/usr/bin/env python3
"""Visualization of test-time artifacts: embedding projections, score distributions,
and enrollment similarity heatmaps.

Separate from visualize_training.py because data sources are fundamentally different
(.pt embeddings + multi-million row CSVs vs epoch-level training logs).

Usage:
    python scripts/visualize_test_artifacts.py \\
        --base_dir /dataHDD/ahmad/comfort26_sem/cnceleb \\
        --experiments "sv_vanilla_*" "sv_bregman_*-sr90" \\
        --test_sets cnceleb_concatenated \\
        --plots all \\
        --embed_method umap \\
        --score_col both \\
        --output results/figures/test_artifacts/
"""

import argparse
import hashlib
import json
import os
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Reuse shared utilities from visualize.py
sys.path.insert(0, os.path.dirname(__file__))
from visualize import discover_experiments, make_label, setup_matplotlib

matplotlib.use("pdf")


# ---------------------------------------------------------------------------
# Data discovery and loading
# ---------------------------------------------------------------------------


def resolve_latest_run(test_set_dir):
    """Return path to the latest timestamped run directory."""
    last_run_file = os.path.join(test_set_dir, "LAST_RUN")
    if os.path.isfile(last_run_file):
        ts = open(last_run_file).read().strip()
        candidate = os.path.join(test_set_dir, ts)
        if os.path.isdir(candidate):
            return candidate

    # Fallback: lexicographic sort of YYYYMMDD_HHMMSS dirs
    subdirs = sorted(
        d
        for d in os.listdir(test_set_dir)
        if os.path.isdir(os.path.join(test_set_dir, d))
        and re.match(r"\d{8}_\d{6}$", d)
    )
    if subdirs:
        return os.path.join(test_set_dir, subdirs[-1])
    return None


def discover_test_sets(exp_dir):
    """List test sets with their artifact directories under an experiment."""
    artifacts_root = os.path.join(exp_dir, "test_artifacts")
    if not os.path.isdir(artifacts_root):
        return []

    result = []
    for name in sorted(os.listdir(artifacts_root)):
        if name.startswith("_"):
            continue
        test_set_dir = os.path.join(artifacts_root, name)
        if not os.path.isdir(test_set_dir):
            continue
        run_dir = resolve_latest_run(test_set_dir)
        if run_dir:
            result.append((name, run_dir))
    return result


def _safe_filename(name):
    """Convert test set name to safe filename component (/ -> _)."""
    return name.replace("/", "_")


def load_embeddings(artifacts_dir, test_set_name):
    """Load enrollment and test embeddings from .pt files."""
    safe = _safe_filename(test_set_name)
    enrol_path = os.path.join(artifacts_dir, f"{safe}_enrol_embeds.pt")
    test_path = os.path.join(artifacts_dir, f"{safe}_embeds.pt")

    enrol = (
        torch.load(enrol_path, map_location="cpu", weights_only=True)
        if os.path.isfile(enrol_path)
        else None
    )
    test = (
        torch.load(test_path, map_location="cpu", weights_only=True)
        if os.path.isfile(test_path)
        else None
    )
    return enrol, test


def load_scores(artifacts_dir, test_set_name):
    """Load scores CSV with only needed columns for memory efficiency."""
    safe = _safe_filename(test_set_name)
    path = os.path.join(artifacts_dir, f"{safe}_scores.csv")
    if not os.path.isfile(path):
        return None

    # First read just the header to check available columns
    header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [
        c
        for c in [
            "enroll_path",
            "test_path",
            "trial_label",
            "score",
            "norm_score",
        ]
        if c in header
    ]
    return pd.read_csv(path, usecols=usecols)


def extract_speaker_id(key):
    """Extract speaker ID from an embedding key.

    Handles:
        cnceleb_id00800-enroll -> id00800
        CN-Celeb_wav/eval/test/id00800-singing-01-001.wav -> id00800
        id10270/x6uYqmx31kE/00001.wav -> id10270
    """
    m = re.search(r"(id\d+)", key)
    if m:
        return m.group(1)
    # Fallback: first path component
    parts = key.replace("\\", "/").split("/")
    return parts[0].split("-")[0]


# ---------------------------------------------------------------------------
# Plot 1: Embedding projections (UMAP / t-SNE)
# ---------------------------------------------------------------------------


def _compute_projection(embeddings, method, cache_path=None):
    """Run dimensionality reduction, with optional caching."""
    if cache_path and os.path.isfile(cache_path):
        data = np.load(cache_path)
        return data["coords"]

    if method == "umap":
        try:
            import umap
        except ImportError:
            print(
                "Error: umap-learn not installed. Install with: pip install umap-learn"
            )
            return None
        reducer = umap.UMAP(
            n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42
        )
        coords = reducer.fit_transform(embeddings)
    elif method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            print(
                "Error: scikit-learn not installed. Install with: pip install scikit-learn"
            )
            return None
        reducer = TSNE(
            n_components=2,
            perplexity=30,
            metric="cosine",
            learning_rate="auto",
            init="pca",
            random_state=42,
        )
        coords = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown projection method: {method}")

    if cache_path:
        np.savez_compressed(cache_path, coords=coords)
    return coords


def plot_embedding_projection(
    enrol_embeds,
    test_embeds,
    output_path,
    method="umap",
    n_highlight=15,
    max_utt_per_speaker=50,
    title=None,
    no_cache=False,
):
    """Plot 2D embedding projection with highlighted speaker clusters."""
    if test_embeds is None:
        print("  [skip] no test embeddings for projection")
        return

    # Group test embeddings by speaker
    spk_to_keys = {}
    for key in test_embeds:
        spk = extract_speaker_id(key)
        spk_to_keys.setdefault(spk, []).append(key)

    # Select top speakers by utterance count
    spk_counts = sorted(
        spk_to_keys.items(), key=lambda x: len(x[1]), reverse=True
    )
    highlight_spks = {spk for spk, _ in spk_counts[:n_highlight]}

    # Build arrays: subsample highlighted, keep all others (for context)
    keys_ordered = []
    is_highlighted = []
    spk_labels = []

    for spk, keys in spk_to_keys.items():
        if spk in highlight_spks:
            sampled = keys[:max_utt_per_speaker]
        else:
            sampled = keys[:max_utt_per_speaker]
        for k in sampled:
            keys_ordered.append(k)
            is_highlighted.append(spk in highlight_spks)
            spk_labels.append(spk)

    # Add enrollment embeddings
    enrol_keys = list(enrol_embeds.keys()) if enrol_embeds else []
    enrol_spks = [extract_speaker_id(k) for k in enrol_keys]
    n_test = len(keys_ordered)

    all_embeds = []
    for k in keys_ordered:
        all_embeds.append(test_embeds[k].numpy())
    for k in enrol_keys:
        all_embeds.append(enrol_embeds[k].numpy())
    all_embeds = np.stack(all_embeds)

    # Projection with caching
    params_hash = hashlib.md5(  # nosec B303 — non-security cache key
        f"{method}_{n_highlight}_{max_utt_per_speaker}".encode()
    ).hexdigest()[:8]
    cache_path = None
    if not no_cache:
        cache_dir = os.path.dirname(output_path)
        cache_path = os.path.join(
            cache_dir,
            f".projection_cache_{method}_n{len(all_embeds)}_{params_hash}.npz",
        )

    print(
        f"  Computing {method.upper()} projection for {len(all_embeds)} points..."
    )
    coords = _compute_projection(all_embeds, method, cache_path)
    if coords is None:
        return

    test_coords = coords[:n_test]
    enrol_coords = coords[n_test:]

    # Assign colors to highlighted speakers
    cmap = plt.cm.tab20
    highlight_list = sorted(highlight_spks)
    spk_color_map = {
        spk: cmap(i / max(len(highlight_list), 1))
        for i, spk in enumerate(highlight_list)
    }

    fig, ax = plt.subplots(figsize=(7, 6))

    # Plot non-highlighted (background)
    bg_mask = np.array([not h for h in is_highlighted])
    if bg_mask.any():
        ax.scatter(
            test_coords[bg_mask, 0],
            test_coords[bg_mask, 1],
            c="#cccccc",
            s=3,
            alpha=0.15,
            rasterized=True,
        )

    # Plot highlighted speakers
    for spk in highlight_list:
        mask = np.array(
            [s == spk and h for s, h in zip(spk_labels, is_highlighted)]
        )
        if not mask.any():
            continue
        color = spk_color_map[spk]
        ax.scatter(
            test_coords[mask, 0],
            test_coords[mask, 1],
            c=[color],
            s=8,
            alpha=0.6,
            label=spk,
            rasterized=True,
        )

    # Plot enrollment embeddings
    for i, (key, spk) in enumerate(zip(enrol_keys, enrol_spks)):
        color = spk_color_map.get(spk, "#333333")
        ax.scatter(
            enrol_coords[i, 0],
            enrol_coords[i, 1],
            c=[color],
            s=80,
            marker="*",
            edgecolors="black",
            linewidths=0.5,
            zorder=10,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        fontsize=6,
        ncol=3,
        loc="upper right",
        framealpha=0.7,
        markerscale=1.5,
        handletextpad=0.3,
        columnspacing=0.5,
    )
    if title:
        ax.set_title(title)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 2: Score distributions
# ---------------------------------------------------------------------------


def load_metrics_json(artifacts_dir, test_set_name):
    """Load metrics JSON for a test set.

    Returns parsed dict or None.
    """
    safe = _safe_filename(test_set_name)
    path = os.path.join(artifacts_dir, f"{safe}_metrics.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _get_eer_for_score_col(metrics, score_col):
    """Extract EER and threshold from metrics dict for the given score column.

    Returns (eer, eer_threshold) or (None, None) if unavailable.
    """
    if metrics is None:
        return None, None

    # Newer format: nested under "norm" and "raw" keys
    if "norm" in metrics and "raw" in metrics:
        key = "raw" if score_col == "score" else "norm"
        section = metrics[key]
        eer = section.get("eer")
        eer_threshold = section.get("eer_threshold")
        # Only return if both values are present and numeric
        if eer is not None and eer_threshold is not None:
            return eer, eer_threshold
        return None, None

    # Older flat format — ambiguous, skip
    print(
        "  [warn] metrics JSON has flat format (no norm/raw keys), skipping EER line"
    )
    return None, None


def plot_score_distribution(
    scores_df,
    output_path,
    score_col="score",
    title=None,
    n_bins=200,
    metrics=None,
):
    """Plot target vs impostor score distributions with KDE overlay."""
    if scores_df is None or score_col not in scores_df.columns:
        print(f"  [skip] column '{score_col}' not in scores")
        return

    targets = scores_df.loc[scores_df["trial_label"] == 1, score_col].dropna()
    impostors = scores_df.loc[
        scores_df["trial_label"] == 0, score_col
    ].dropna()

    if len(targets) == 0 or len(impostors) == 0:
        print("  [skip] empty target or impostor set")
        return

    fig, ax = plt.subplots(figsize=(6, 3.5))

    # Histograms
    all_scores = pd.concat([targets, impostors])
    bins = np.linspace(all_scores.min(), all_scores.max(), n_bins + 1)

    ax.hist(
        impostors,
        bins=bins,
        density=True,
        alpha=0.55,
        color="#1f77b4",
        label="Impostor",
        edgecolor="none",
    )
    ax.hist(
        targets,
        bins=bins,
        density=True,
        alpha=0.55,
        color="#d62728",
        label="Target",
        edgecolor="none",
    )

    # KDE overlay
    try:
        from scipy.stats import gaussian_kde

        x_range = np.linspace(all_scores.min(), all_scores.max(), 500)
        kde_imp = gaussian_kde(impostors)
        kde_tgt = gaussian_kde(targets)
        ax.plot(
            x_range,
            kde_imp(x_range),
            color="#1f77b4",
            linewidth=1.0,
            alpha=0.8,
        )
        ax.plot(
            x_range,
            kde_tgt(x_range),
            color="#d62728",
            linewidth=1.0,
            alpha=0.8,
        )
    except ImportError:
        pass  # scipy optional for KDE

    # EER threshold line from metrics JSON
    eer, eer_thresh = _get_eer_for_score_col(metrics, score_col)
    if eer_thresh is not None:
        ax.axvline(
            eer_thresh, color="black", linestyle="--", linewidth=0.8, alpha=0.7
        )
        eer_pct = eer * 100 if eer < 1 else eer
        ax.text(
            eer_thresh,
            ax.get_ylim()[1] * 0.92,
            f" EER={eer_pct:.2f}%\n thr={eer_thresh:.3f}",
            fontsize=7,
            va="top",
            ha="left",
        )

    x_label = "Cosine Score" if score_col == "score" else "Normalized Score"
    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    if title:
        ax.set_title(title)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_score_distribution_dual(
    scores_df, output_path, title=None, n_bins=200, metrics=None
):
    """Plot raw and normalized score distributions side by side."""
    if scores_df is None:
        return
    has_raw = "score" in scores_df.columns
    has_norm = "norm_score" in scores_df.columns
    if not has_raw and not has_norm:
        print("  [skip] no score columns found")
        return

    cols = []
    if has_raw:
        cols.append(("score", "Cosine Score"))
    if has_norm:
        cols.append(("norm_score", "Normalized Score"))
    if len(cols) < 2:
        # Fall back to single plot
        plot_score_distribution(
            scores_df,
            output_path,
            score_col=cols[0][0],
            title=title,
            n_bins=n_bins,
            metrics=metrics,
        )
        return

    fig, axes = plt.subplots(2, 1, figsize=(6, 6))

    for ax, (col, xlabel) in zip(axes, cols):
        targets = scores_df.loc[scores_df["trial_label"] == 1, col].dropna()
        impostors = scores_df.loc[scores_df["trial_label"] == 0, col].dropna()
        if len(targets) == 0 or len(impostors) == 0:
            continue

        all_scores = pd.concat([targets, impostors])
        bins = np.linspace(all_scores.min(), all_scores.max(), n_bins + 1)

        ax.hist(
            impostors,
            bins=bins,
            density=True,
            alpha=0.55,
            color="#1f77b4",
            label="Impostor",
            edgecolor="none",
        )
        ax.hist(
            targets,
            bins=bins,
            density=True,
            alpha=0.55,
            color="#d62728",
            label="Target",
            edgecolor="none",
        )

        try:
            from scipy.stats import gaussian_kde

            x_range = np.linspace(all_scores.min(), all_scores.max(), 500)
            ax.plot(
                x_range,
                gaussian_kde(impostors)(x_range),
                color="#1f77b4",
                linewidth=1.0,
                alpha=0.8,
            )
            ax.plot(
                x_range,
                gaussian_kde(targets)(x_range),
                color="#d62728",
                linewidth=1.0,
                alpha=0.8,
            )
        except ImportError:
            pass

        # EER threshold line from metrics JSON
        eer, eer_thresh = _get_eer_for_score_col(metrics, col)
        if eer_thresh is not None:
            ax.axvline(
                eer_thresh,
                color="black",
                linestyle="--",
                linewidth=0.8,
                alpha=0.7,
            )
            eer_pct = eer * 100 if eer < 1 else eer
            ax.text(
                eer_thresh,
                ax.get_ylim()[1] * 0.92,
                f" EER={eer_pct:.2f}%\n thr={eer_thresh:.3f}",
                fontsize=7,
                va="top",
                ha="left",
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend(loc="upper right")

    if title:
        axes[0].set_title(title)

    fig.subplots_adjust(hspace=0.35)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 3: Enrollment similarity heatmap
# ---------------------------------------------------------------------------


def plot_enrollment_heatmap(enrol_embeds, output_path, title=None):
    """Plot cosine similarity heatmap of enrollment embeddings with
    hierarchical clustering."""
    if enrol_embeds is None or len(enrol_embeds) == 0:
        print("  [skip] no enrollment embeddings")
        return

    keys = list(enrol_embeds.keys())
    spk_ids = [extract_speaker_id(k) for k in keys]
    n = len(keys)

    # Stack and L2-normalize
    E = torch.stack([enrol_embeds[k] for k in keys]).numpy()
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    E = E / norms

    # Cosine similarity
    S = E @ E.T

    # Hierarchical clustering for ordering
    try:
        from scipy.cluster.hierarchy import (
            dendrogram,
            linkage,
            optimal_leaf_ordering,
        )
        from scipy.spatial.distance import squareform

        dist = 1.0 - S
        np.fill_diagonal(dist, 0.0)
        dist = np.maximum(dist, 0.0)  # numerical safety
        condensed = squareform(dist)
        Z = linkage(condensed, method="average")
        Z = optimal_leaf_ordering(Z, condensed)
        leaf_order = dendrogram(Z, no_plot=True)["leaves"]

        S = S[np.ix_(leaf_order, leaf_order)]
        spk_ids = [spk_ids[i] for i in leaf_order]
    except ImportError:
        pass

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(S, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

    # Tick labels
    if n <= 50:
        ax.set_xticks(range(n))
        ax.set_xticklabels(spk_ids, rotation=90, fontsize=5)
        ax.set_yticks(range(n))
        ax.set_yticklabels(spk_ids, fontsize=5)
    else:
        step = max(1, n // 30)
        ticks = list(range(0, n, step))
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [spk_ids[i] for i in ticks], rotation=90, fontsize=5
        )
        ax.set_yticks(ticks)
        ax.set_yticklabels([spk_ids[i] for i in ticks], fontsize=5)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Cosine Similarity")

    if title:
        ax.set_title(title)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI and main loop
# ---------------------------------------------------------------------------

VALID_PLOTS = {"embeddings", "scores", "heatmap", "all"}


def main():
    parser = argparse.ArgumentParser(
        description="Visualize test-time artifacts: embeddings, scores, enrollment heatmaps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--base_dir",
        required=True,
        help="Root dir containing experiment folders.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="Glob patterns for experiment directory names.",
    )
    parser.add_argument(
        "--test_sets",
        nargs="*",
        default=None,
        help="Filter to specific test sets (default: all).",
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        default=["all"],
        choices=sorted(VALID_PLOTS),
        help="Which plots to generate.",
    )
    parser.add_argument(
        "--embed_method",
        default="umap",
        choices=["umap", "tsne"],
        help="Dimensionality reduction method for embeddings.",
    )
    parser.add_argument(
        "--n_highlight",
        type=int,
        default=15,
        help="Number of speakers to highlight in embedding plot.",
    )
    parser.add_argument(
        "--score_col",
        default="both",
        choices=["score", "norm_score", "both"],
        help="Which score column(s) to plot.",
    )
    parser.add_argument(
        "--output",
        default="results/figures/test_artifacts/",
        help="Output directory.",
    )
    parser.add_argument("--font_size", type=int, default=10)
    parser.add_argument(
        "--no_cache", action="store_true", help="Disable projection caching."
    )
    args = parser.parse_args()

    plots = set(args.plots)
    if "all" in plots:
        plots = {"embeddings", "scores", "heatmap"}

    setup_matplotlib(args.font_size)

    experiments = discover_experiments(args.base_dir, args.experiments)
    if not experiments:
        print("No experiments matched the given patterns.")
        return

    print(f"Found {len(experiments)} experiments:")
    for _, info in experiments:
        print(f"  {info['dirname']}  ->  {make_label(info)}")

    for exp_dir, info in experiments:
        exp_label = make_label(info)
        test_sets = discover_test_sets(exp_dir)
        if not test_sets:
            print(f"\n[skip] {info['dirname']}: no test artifacts")
            continue

        for test_set_name, artifacts_dir in test_sets:
            if args.test_sets and test_set_name not in args.test_sets:
                continue

            print(f"\n--- {info['dirname']} / {test_set_name} ---")
            out_dir = os.path.join(args.output, info["dirname"], test_set_name)
            os.makedirs(out_dir, exist_ok=True)

            title_prefix = f"{exp_label} — {test_set_name}"

            # Embedding projection
            if "embeddings" in plots:
                enrol, test = load_embeddings(artifacts_dir, test_set_name)
                plot_embedding_projection(
                    enrol,
                    test,
                    os.path.join(
                        out_dir, f"embeddings_{args.embed_method}.pdf"
                    ),
                    method=args.embed_method,
                    n_highlight=args.n_highlight,
                    title=title_prefix,
                    no_cache=args.no_cache,
                )

            # Score distributions
            if "scores" in plots:
                scores = load_scores(artifacts_dir, test_set_name)
                metrics = load_metrics_json(artifacts_dir, test_set_name)
                if args.score_col == "both":
                    plot_score_distribution_dual(
                        scores,
                        os.path.join(out_dir, "score_dist_dual.pdf"),
                        title=title_prefix,
                        metrics=metrics,
                    )
                if args.score_col in ("score", "both"):
                    plot_score_distribution(
                        scores,
                        os.path.join(out_dir, "score_dist_raw.pdf"),
                        score_col="score",
                        title=title_prefix,
                        metrics=metrics,
                    )
                if args.score_col in ("norm_score", "both"):
                    plot_score_distribution(
                        scores,
                        os.path.join(out_dir, "score_dist_norm.pdf"),
                        score_col="norm_score",
                        title=title_prefix,
                        metrics=metrics,
                    )

            # Enrollment heatmap
            if "heatmap" in plots:
                enrol = load_embeddings(artifacts_dir, test_set_name)[0]
                plot_enrollment_heatmap(
                    enrol,
                    os.path.join(out_dir, "enrollment_heatmap.pdf"),
                    title=title_prefix,
                )

    print("\nDone.")


if __name__ == "__main__":
    main()
