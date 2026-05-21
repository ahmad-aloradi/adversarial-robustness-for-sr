#!/usr/bin/env python3
"""Visualize EER / minDCF vs bit-width across quantization specs.

Reads per-test-set metric JSONs produced by ``src/eval.py`` runs that were
executed under different ``quantization=<spec>`` configurations, and renders
one PDF per (dataset, metric) pair showing the bit-width sweep.

Inputs are discovered by walking ``--input_dir`` for files matching::

    <input_dir>/<run_label>/.../<safe_test_name>_metrics.json

where ``<run_label>`` is expected to encode the spec (e.g. via the run
directory name containing ``int4_w``, ``fp16``, etc.). When the spec cannot
be parsed from the path, the file is logged and skipped.

Usage::

    python scripts/visualize_quantization.py \\
        --input_dir results/quantization_sweep \\
        --output_dir results/quantization_sweep/plots
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

# Bit-width axis order (smaller = more aggressive quant; FP32 on the left for
# reference). The 32-bit anchor is included so the baseline appears at the
# top of every plot when present.
SPEC_ORDER = [
    "fp32",
    "fp16",
    "int8_w",
    "int8_wa",
    "int4_w",
    "int4_wa",
    "int2_w",
    "int2_wa",
    "int1_w",
    "int1_wa",
]

SPEC_LABELS = {
    "fp32": "FP32",
    "fp16": "FP16",
    "int8_w": "INT8 (w)",
    "int8_wa": "INT8 (w+a)",
    "int4_w": "INT4 (w)",
    "int4_wa": "INT4 (w+a)",
    "int2_w": "INT2 (w)",
    "int2_wa": "INT2 (w+a)",
    "int1_w": "INT1 (w)",
    "int1_wa": "INT1 (w+a)",
}

SPEC_RE = re.compile(
    r"(?P<spec>fp32|fp16|int[1-8]_(?:w|wa))", re.IGNORECASE
)


def _parse_spec_from_path(path: Path) -> str | None:
    """Extract the spec name from any component of ``path``."""
    for part in path.parts:
        m = SPEC_RE.search(part)
        if m:
            return m.group("spec").lower()
    return None


def _collect_metrics(input_dir: Path) -> dict[tuple[str, str], dict[str, dict]]:
    """Walk ``input_dir`` and group metric files by (test_set, metric_kind).

    Returns
    -------
    dict
        ``{(test_set, "EER" | "minDCF"): {spec: {"raw": float, "norm": float}}}``
    """
    bucket: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)

    for metrics_path in input_dir.rglob("*_metrics.json"):
        spec = _parse_spec_from_path(metrics_path)
        if spec is None:
            print(
                f"[skip] no quantization spec parsed from {metrics_path}"
            )
            continue

        try:
            with metrics_path.open() as f:
                payload = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed JSON in {metrics_path}: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError(
                f"{metrics_path}: expected top-level dict, got "
                f"{type(payload).__name__}"
            )

        norm_block = payload.get("norm", {})
        raw_block = payload.get("raw", {})
        if not isinstance(norm_block, dict):
            raise ValueError(
                f"{metrics_path}: 'norm' must be a dict, got "
                f"{type(norm_block).__name__}"
            )
        if not isinstance(raw_block, dict):
            raise ValueError(
                f"{metrics_path}: 'raw' must be a dict, got "
                f"{type(raw_block).__name__}"
            )

        test_set = payload.get("test_set") or metrics_path.parent.name
        for metric_kind in ("EER", "minDCF"):
            norm = norm_block.get(metric_kind)
            raw = raw_block.get(metric_kind)
            if norm is None and raw is None:
                continue
            bucket[(test_set, metric_kind)][spec] = {
                "norm": float(norm) if norm is not None else None,
                "raw": float(raw) if raw is not None else None,
            }

    return bucket


def _ordered_specs(present: Iterable[str]) -> list[str]:
    present_set = set(present)
    return [s for s in SPEC_ORDER if s in present_set]


def _plot_one(
    test_set: str,
    metric_kind: str,
    spec_to_values: dict[str, dict],
    output_path: Path,
) -> None:
    specs = _ordered_specs(spec_to_values)
    if not specs:
        return

    x = np.arange(len(specs))
    norm_vals = [spec_to_values[s]["norm"] for s in specs]
    raw_vals = [spec_to_values[s]["raw"] for s in specs]

    fig, ax = plt.subplots(figsize=(max(6.0, 0.8 * len(specs) + 2.5), 4.0))
    width = 0.4

    has_norm = any(v is not None for v in norm_vals)
    has_raw = any(v is not None for v in raw_vals)

    if has_norm:
        ax.bar(
            x - width / 2,
            [v if v is not None else 0.0 for v in norm_vals],
            width=width,
            label="norm",
            color="#2c7fb8",
            edgecolor="black",
            linewidth=0.5,
        )
    if has_raw:
        ax.bar(
            x + width / 2,
            [v if v is not None else 0.0 for v in raw_vals],
            width=width,
            label="raw",
            color="#a6bddb",
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([SPEC_LABELS.get(s, s) for s in specs], rotation=30, ha="right")
    ax.set_ylabel(metric_kind)
    ax.set_title(f"{test_set} — {metric_kind} vs bit width")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    if has_norm or has_raw:
        ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[ok] {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing test_artifacts/ subtrees from quantized eval runs.",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Where to write PDFs. Defaults to <input_dir>/plots/.",
    )
    args = p.parse_args()

    output_dir = args.output_dir or (args.input_dir / "plots")
    bucket = _collect_metrics(args.input_dir)
    if not bucket:
        print(f"No *_metrics.json files found under {args.input_dir}")
        return

    for (test_set, metric_kind), spec_values in bucket.items():
        safe_name = (
            test_set.replace("/", "_").replace("\\", "_") + f"_{metric_kind}.pdf"
        )
        _plot_one(test_set, metric_kind, spec_values, output_dir / safe_name)


if __name__ == "__main__":
    main()
