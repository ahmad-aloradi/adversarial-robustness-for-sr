import os
import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Union
import pandas as pd


TIMESTAMP_RE = re.compile(r'^\d{8}_\d{6}$')  # e.g. 20260223_094614
METRICSS_KEYVALS =  {'EER': 'eer', 'minDCF': 'mindcf', 'DCF': 'detection_cost'} # EER Must be present!
METRICSS_KEYS = list(METRICSS_KEYVALS.keys())

def extract_metrics(metrics: Any, target_keys: set) -> Union[float, str]:
    """
    Recursively search the metrics structure for a value whose key looks like 'eer' (case-insensitive).
    Returns the first numeric value found, or 'N/A' if none found.
    """    
    def _is_number(x):
        return isinstance(x, (int, float))
    
    def _rec(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(k, str) and k.lower() in target_keys:
                    if _is_number(v):
                        return float(v)
                    # sometimes EER is stored as string like "3.12"
                    try:
                        return float(v)
                    except Exception:
                        return 'N/A'
                # keep searching deeper
                found = _rec(v)
                if found != 'N/A':
                    return found
        elif isinstance(obj, list):
            for e in obj:
                found = _rec(e)
                if found != 'N/A':
                    return found
        # not found at this branch
        return 'N/A'
    
    return _rec(metrics)


def collect_metrics_from_bases(base_dirs: List[Union[str, Path]],
                               save_csv: Union[str, None] = None,
                               ignore_prefixes: List[str] = None) -> List[Dict[str, Any]]:
    """
    Scan the given base directories for experiment folders and `test_artifacts/*/<timestamp>/*_metrics.json`,
    returning a list of result dicts. If save_csv is provided and pandas is available, save results there.
    """
    if ignore_prefixes is None:
        ignore_prefixes = ['_']  # default: skip dirs starting with underscore

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
            exp_path = exp_entry

            test_artifacts_path = exp_path / 'test_artifacts'
            if not test_artifacts_path.exists() or not test_artifacts_path.is_dir():
                # nothing to do for this experiment
                continue

            # iterate datasets inside test_artifacts, ignoring names that start with any ignore_prefix
            for dataset_entry in test_artifacts_path.iterdir():
                if not dataset_entry.is_dir():
                    continue
                dataset = dataset_entry.name
                if any(dataset.startswith(pref) for pref in ignore_prefixes):
                    # skip cohort cache / hidden dirs etc.
                    continue

                # find timestamp-style subdirs
                for ts_entry in dataset_entry.iterdir():
                    if not ts_entry.is_dir():
                        continue
                    ts_name = ts_entry.name
                    if not TIMESTAMP_RE.match(ts_name):
                        # skip non-timestamp dirs
                        continue

                    # expected metrics filename: {dataset}_metrics.json
                    metrics_file = ts_entry / f"{dataset}_metrics.json"
                    if not metrics_file.exists():
                        # Skip incomplete tests
                        continue

                    # read JSON
                    with metrics_file.open('r', encoding='utf-8') as fh:
                        metrics_obj = json.load(fh)

                    # extract EER (first match) and other common top-level fields if present
                    extracted_values = {}
                    for key in METRICSS_KEYVALS:
                        extracted_values[key] = extract_metrics(metrics_obj, target_keys=set([METRICSS_KEYVALS[key]]))

                    # build result record
                    record = {
                        'exp': exp_folder,
                        'dataset': dataset,
                        # 'timestamp': ts_name,
                        'metrics_file': f"{ts_name}/{dataset}_metrics.json",
                        # 'metrics_file': str(metrics_file.resolve()),
                        **extracted_values,
                        # 'raw_metrics': metrics_obj,  # store the whole parsed JSON in case you need it later
                    }

                    results.append(record)

    # Optionally convert to pandas DataFrame and save
    if save_csv:
        df = pd.DataFrame(results)
        # convert timestamp-like column to actual datetime if possible
        if 'timestamp' in df.columns:
            df['timestamp_dt'] = pd.to_datetime(df['timestamp'], format='%Y%m%d_%H%M%S')
        df.to_csv(save_csv, index=False)
        print(f"Saved results to {save_csv}")

    return results



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Publication-ready test metrics (json) parsing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--base_dirs", 
                        type=str,
                        nargs="+",
                        required=True,
                        help="Root dir containing experiment folders.")

    parser.add_argument("--output_dir", 
                        default=f"{os.environ.get('HOME')}/adversarial-robustness-for-sr/results/test_metrics",
                        type=str, 
                        help="Root dir containing experiment folders.")


    args = parser.parse_args()
    base_dirs = args.base_dirs
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Collect results; if you want a CSV output, pass a path like "collected_metrics.csv"
    results = collect_metrics_from_bases(base_dirs, save_csv=None)

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(results)

    # Ensure EER is numeric (important if some entries are strings)
    df["EER"] = pd.to_numeric(df["EER"], errors="coerce")

    # Convert timestamp to datetime for proper sorting
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d_%H%M%S")

    # ---------------------------------------------------------
    # 1️⃣ Global ranking (across all datasets)
    # ---------------------------------------------------------

    df_sorted = df.sort_values("EER", ascending=True).reset_index(drop=True)

    # Add rank column (1 = best EER)
    df_sorted["rank"] = df_sorted["EER"].rank(method="min")

    print("\n=== Global Ranking (Lower EER = Better) ===")
    print(df_sorted[["rank", "exp", "dataset"] + METRICSS_KEYS])

    df_sorted.to_csv(f"{output_dir}/eer_global_ranking.csv", index=False)


    # ---------------------------------------------------------
    # 2️⃣ Ranking per dataset (recommended in research)
    # ---------------------------------------------------------

    df_per_dataset = (
        df.sort_values(["dataset", "EER"])
        .groupby("dataset", group_keys=False)
        .apply(lambda x: x.assign(rank=x["EER"].rank(method="min")))
        .reset_index(drop=True)
    )

    print("\n=== Per-Dataset Ranking ===")
    print(df_per_dataset[["dataset", "rank", "exp"] + METRICSS_KEYS])

    df_per_dataset.to_csv(f"{output_dir}/eer_per_dataset_ranking.csv", index=False)


    # ---------------------------------------------------------
    # 3️⃣ Optional: Compact leaderboard view
    # ---------------------------------------------------------

    leaderboard = (
        df_per_dataset
        .sort_values(["dataset", "rank"])
        [["dataset", "rank", "exp"] + METRICSS_KEYS]
    )

    print("\n=== Leaderboard ===")
    print(leaderboard)
    leaderboard.to_csv(f"{output_dir}/eer_leaderboard.csv", index=False)