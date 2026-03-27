import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd

TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")  # e.g. 20260223_094614
HYDRA_RUN_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$"
)  # e.g. 2026-01-27_16-00-10
METRICSS_KEYVALS = {
    "EER": "eer",
    "minDCF": "mindcf",
    "DCF": "detection_cost",
}  # EER Must be present!
METRICSS_KEYS = list(METRICSS_KEYVALS.keys())


def _find_metric_value(obj, target_keys: set) -> Union[float, str]:
    """Recursively search for a metric key (case-insensitive) and return its
    numeric value."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() in target_keys:
                try:
                    return float(v)
                except Exception:
                    return "N/A"
            found = _find_metric_value(v, target_keys)
            if found != "N/A":
                return found
    elif isinstance(obj, list):
        for e in obj:
            found = _find_metric_value(e, target_keys)
            if found != "N/A":
                return found
    return "N/A"


QUALIFIERS = ("norm", "raw")


def extract_metrics(
    metrics: Any, target_keys: set
) -> Dict[str, Union[float, str]]:
    """Extract a metric from the JSON, returning {qualifier: value}.

    If the JSON has top-level "norm" / "raw" dict keys, extract the metric from
    each sub-dict separately → {"norm": 0.30, "raw": 0.31}. Otherwise (flat
    structure) → {"": 0.12}.
    """
    if isinstance(metrics, dict) and any(q in metrics for q in QUALIFIERS):
        result = {}
        for q in QUALIFIERS:
            if q in metrics and isinstance(metrics[q], dict):
                result[q] = _find_metric_value(metrics[q], target_keys)
        return result
    # Flat structure
    return {"": _find_metric_value(metrics, target_keys)}


def collect_metrics_from_bases(
    base_dirs: List[Union[str, Path]],
    save_csv: Union[str, None] = None,
    ignore_prefixes: List[str] = None,
) -> List[Dict[str, Any]]:
    """Scan the given base directories for experiment folders and
    `test_artifacts/*/<timestamp>/*_metrics.json`, returning a list of result
    dicts.

    If save_csv is provided and pandas is available, save results there.
    """
    if ignore_prefixes is None:
        ignore_prefixes = ["_"]  # default: skip dirs starting with underscore

    results = []

    for base_dir in base_dirs:
        base = Path(base_dir)
        if not base.exists():
            print(f"Warning: base dir does not exist: {base}")
            continue
        if not base.is_dir():
            print(f"Warning: base path is not a directory: {base}")
            continue

        # All immediate subdirectories are experiment folders
        for exp_entry in base.iterdir():
            if not exp_entry.is_dir():
                continue
            exp_folder = exp_entry.name
            # Skip unnamed Hydra run dirs (timestamp-only names)
            if HYDRA_RUN_RE.match(exp_folder):
                continue
            exp_path = exp_entry

            test_artifacts_path = exp_path / "test_artifacts"
            if (
                not test_artifacts_path.exists()
                or not test_artifacts_path.is_dir()
            ):
                # nothing to do for this experiment
                continue

            # iterate datasets inside test_artifacts, ignoring names that start with any ignore_prefix
            for dataset_entry in test_artifacts_path.iterdir():
                if not dataset_entry.is_dir():
                    continue
                raw_dataset = dataset_entry.name
                if any(
                    raw_dataset.startswith(pref) for pref in ignore_prefixes
                ):
                    # skip cohort cache / hidden dirs etc.
                    continue

                # Strip duplicate dataset prefix (e.g. cnceleb_cnceleb_multi → cnceleb_multi)
                dataset = raw_dataset
                first_word = dataset.split("_", 1)[0]
                dup_prefix = f"{first_word}_{first_word}_"
                if dataset.startswith(dup_prefix):
                    dataset = dataset[len(first_word) + 1 :]

                # find timestamp-style subdirs
                for ts_entry in dataset_entry.iterdir():
                    if not ts_entry.is_dir():
                        continue
                    ts_name = ts_entry.name
                    if not TIMESTAMP_RE.match(ts_name):
                        # skip non-timestamp dirs
                        continue

                    # expected metrics filename uses original dir name
                    metrics_file = ts_entry / f"{raw_dataset}_metrics.json"
                    if not metrics_file.exists():
                        # Skip incomplete tests
                        continue

                    # read JSON
                    with metrics_file.open("r", encoding="utf-8") as fh:
                        metrics_obj = json.load(fh)

                    # extract metrics; flatten qualified variants into columns
                    extracted_values = {}
                    for key in METRICSS_KEYVALS:
                        qualified = extract_metrics(
                            metrics_obj, target_keys={METRICSS_KEYVALS[key]}
                        )
                        for qualifier, value in qualified.items():
                            col = f"{key}_{qualifier}" if qualifier else key
                            extracted_values[col] = value

                    # build result record
                    record = {
                        "exp": exp_folder,
                        "dataset": dataset,
                        "run_ts": ts_name,
                        "metrics_file": f"{ts_name}/{dataset}_metrics.json",
                        **extracted_values,
                        # 'raw_metrics': metrics_obj,  # store the whole parsed JSON in case you need it later
                    }

                    results.append(record)

    # Optionally convert to pandas DataFrame and save
    if save_csv:
        df = pd.DataFrame(results)
        # convert timestamp-like column to actual datetime if possible
        if "timestamp" in df.columns:
            df["timestamp_dt"] = pd.to_datetime(
                df["timestamp"], format="%Y%m%d_%H%M%S"
            )
        df.to_csv(save_csv, index=False)
        print(f"Saved results to {save_csv}")

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Publication-ready test metrics (json) parsing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--base_dirs",
        type=str,
        nargs="+",
        required=True,
        help="Root dir containing experiment folders.",
    )

    parser.add_argument(
        "--output_dir",
        default=f"{os.environ.get('HOME')}/adversarial-robustness-for-sr/results/test_metrics",
        type=str,
        help="Root dir containing experiment folders.",
    )

    args = parser.parse_args()
    base_dirs = args.base_dirs
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Collect results; if you want a CSV output, pass a path like "collected_metrics.csv"
    results = collect_metrics_from_bases(base_dirs, save_csv=None)

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(results)

    # Mark the latest run per (exp, dataset) based on run_ts
    df["is_latest"] = df.groupby(["exp", "dataset"])["run_ts"].transform(
        lambda x: x == x.max()
    )

    # Detect all metric columns present (EER, EER_raw, EER_norm, minDCF, etc.)
    all_metric_cols = [
        c
        for c in df.columns
        if any(c == k or c.startswith(f"{k}_") for k in METRICSS_KEYS)
    ]
    for col in all_metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Unified EER for ranking: prefer _raw when available, fall back to unqualified
    if "EER_raw" in df.columns and "EER" in df.columns:
        df["EER_rank"] = df["EER_raw"].fillna(df["EER"])
    elif "EER_raw" in df.columns:
        df["EER_rank"] = df["EER_raw"]
    else:
        df["EER_rank"] = df["EER"]
    eer_col = "EER_rank"

    # Convert timestamp to datetime for proper sorting
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], format="%Y%m%d_%H%M%S"
        )

    # ---------------------------------------------------------
    # 1️⃣ Global ranking (across all datasets)
    # ---------------------------------------------------------

    df_sorted = df.sort_values(eer_col, ascending=True).reset_index(drop=True)
    df_sorted["rank"] = df_sorted[eer_col].rank(method="min")

    display_cols = ["rank", "exp", "dataset"] + all_metric_cols
    print("\n=== Global Ranking (Lower EER = Better) ===")
    print(df_sorted[display_cols])

    df_sorted.to_csv(f"{output_dir}/eer_global_ranking.csv", index=False)

    # ---------------------------------------------------------
    # 2️⃣ Ranking per dataset (recommended in research)
    # ---------------------------------------------------------

    df_per_dataset = (
        df.sort_values(["dataset", eer_col])
        .groupby("dataset", group_keys=False)
        .apply(lambda x: x.assign(rank=x[eer_col].rank(method="min")))
        .reset_index(drop=True)
    )

    display_cols_ds = ["dataset", "rank", "exp"] + all_metric_cols
    print("\n=== Per-Dataset Ranking ===")
    print(df_per_dataset[display_cols_ds])

    df_per_dataset.to_csv(
        f"{output_dir}/eer_per_dataset_ranking.csv", index=False
    )

    # ---------------------------------------------------------
    # 3️⃣ Optional: Compact leaderboard view
    # ---------------------------------------------------------

    leaderboard = df_per_dataset.sort_values(["dataset", "rank"])[
        display_cols_ds
    ]

    print("\n=== Leaderboard ===")
    print(leaderboard)
    leaderboard.to_csv(f"{output_dir}/eer_leaderboard.csv", index=False)
