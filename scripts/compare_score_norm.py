#!/usr/bin/env python3
"""Compare verification metrics before vs after score normalization.

Reads a CSV containing trial labels and both raw and normalized scores, then
computes EER and minDCF using `VerificationMetrics`.

Example:
    python scripts/compare_score_norm.py \
        --csv /path/to/test_scores.csv
"""

from __future__ import annotations

import argparse
import math
from typing import Dict

import pandas as pd
import torch

from src.modules.metrics.metrics import VerificationMetrics


def _as_float(x: torch.Tensor) -> float:
    return x.detach().cpu().item() if torch.is_tensor(x) else float(x)


def compute_metrics(
    scores: torch.Tensor,
    labels: torch.Tensor,
    positive_label: int,
    beta: float,
    threshold: float | None,
    cfa: float,
    cfr: float,
    p_target: float,
) -> Dict[str, float]:
    metric = VerificationMetrics(
        positive_label=positive_label,
        beta=beta,
        threshold=threshold,
        Cfa=cfa,
        Cfr=cfr,
        P_target=p_target,
        compute_on_step=False,
    )
    metric.update(scores=scores, labels=labels)
    out = metric.compute()
    return {
        "eer": _as_float(out["eer"]),
        "eer_threshold": _as_float(out["eer_threshold"]),
        "minDCF": _as_float(out["minDCF"]),
        "minDCF_threshold": _as_float(out["minDCF_threshold"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare EER/minDCF before vs after score normalization using a CSV "
            "with raw and normalized scores."
        )
    )
    parser.add_argument("--csv", required=True, help="Path to scores CSV")
    parser.add_argument(
        "--score-col", default="score", help="Column name for raw scores"
    )
    parser.add_argument(
        "--norm-score-col",
        default="norm_score",
        help="Column name for normalized scores",
    )
    parser.add_argument(
        "--label-col",
        default="trial_label",
        help="Column name for labels (0/1)",
    )
    parser.add_argument(
        "--positive-label",
        type=int,
        default=1,
        help="Value representing the positive class (default: 1)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Beta for F-score computation (default: 1.0)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed threshold to use for final stats (default: EER threshold)",
    )
    parser.add_argument(
        "--cfa", type=float, default=1.0, help="Cost of false acceptance"
    )
    parser.add_argument(
        "--cfr", type=float, default=1.0, help="Cost of false rejection"
    )
    parser.add_argument(
        "--p-target",
        type=float,
        default=0.01,
        help="Prior probability of target trials",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    required = [args.score_col, args.norm_score_col, args.label_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in CSV: {missing}. Found columns: {list(df.columns)}"
        )

    # Drop rows with NaN or non-finite values in required columns
    df = df.dropna(subset=required)
    for col in required:
        df = df[pd.to_numeric(df[col], errors="coerce").notna()]

    if len(df) == 0:
        raise ValueError("No valid rows found after filtering NaNs/non-numeric values.")

    scores_raw = torch.tensor(df[args.score_col].astype(float).values, dtype=torch.float32)
    scores_norm = torch.tensor(
        df[args.norm_score_col].astype(float).values, dtype=torch.float32
    )
    labels = torch.tensor(df[args.label_col].astype(int).values, dtype=torch.long)

    raw_metrics = compute_metrics(
        scores=scores_raw,
        labels=labels,
        positive_label=args.positive_label,
        beta=args.beta,
        threshold=args.threshold,
        cfa=args.cfa,
        cfr=args.cfr,
        p_target=args.p_target,
    )
    norm_metrics = compute_metrics(
        scores=scores_norm,
        labels=labels,
        positive_label=args.positive_label,
        beta=args.beta,
        threshold=args.threshold,
        cfa=args.cfa,
        cfr=args.cfr,
        p_target=args.p_target,
    )

    def _delta(a: float, b: float) -> str:
        if any(math.isnan(x) for x in (a, b)):
            return "nan"
        return f"{(b - a):+.6f}"

    print("=== Verification Metrics (raw vs normalized) ===")
    print(f"Rows used: {len(df)}")
    print("\nEER:")
    print(f"  raw : {raw_metrics['eer']:.6f} (thr={raw_metrics['eer_threshold']:.6f})")
    print(
        f"  norm: {norm_metrics['eer']:.6f} (thr={norm_metrics['eer_threshold']:.6f}) "
        f"delta={_delta(raw_metrics['eer'], norm_metrics['eer'])}"
    )

    print("\nminDCF:")
    print(
        f"  raw : {raw_metrics['minDCF']:.6f} (thr={raw_metrics['minDCF_threshold']:.6f})"
    )
    print(
        f"  norm: {norm_metrics['minDCF']:.6f} (thr={norm_metrics['minDCF_threshold']:.6f}) "
        f"delta={_delta(raw_metrics['minDCF'], norm_metrics['minDCF'])}"
    )


if __name__ == "__main__":
    main()
