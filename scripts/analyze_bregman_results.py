#!/usr/bin/env python3
"""
Bregman Experiment Results Analysis Tool

Extracts metrics from TensorBoard logs, verifies experiment correctness,
and generates comparison tables and plots for Bregman pruning experiments.
"""

import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

warnings.filterwarnings("ignore", category=DeprecationWarning)


def extract_metrics(run_dir: Path) -> Dict[str, Any]:
    """Extract metrics from a Bregman experiment run directory.

    Args:
        run_dir: Path to Hydra output directory (e.g., logs/train/runs/2026-02-09_10-00-00)

    Returns:
        Dictionary containing:
        - config: Hydra config dict
        - sparsity: DataFrame with columns [step, value]
        - global_lambda: DataFrame with columns [step, value]
        - validation_metrics: Dict of {metric_name: DataFrame}
        - available_scalars: List of all scalar tags found
    """
    run_dir = Path(run_dir)

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Load Hydra config
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = OmegaConf.load(config_path)

    # Find TensorBoard event files
    tb_dir = run_dir / "tensorboard"
    if not tb_dir.exists():
        raise FileNotFoundError(f"TensorBoard directory not found: {tb_dir}")

    event_files = list(tb_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files in: {tb_dir}")

    # Load TensorBoard data (use latest event file)
    event_file = max(event_files, key=lambda p: p.stat().st_mtime)
    ea = EventAccumulator(str(event_file))
    ea.Reload()

    # Get available scalars
    available_scalars = ea.Tags()["scalars"]

    # Extract Bregman-specific metrics
    result = {
        "config": config,
        "available_scalars": available_scalars,
        "sparsity": None,
        "global_lambda": None,
        "validation_metrics": {},
    }

    # Extract sparsity
    if "bregman/sparsity" in available_scalars:
        events = ea.Scalars("bregman/sparsity")
        result["sparsity"] = pd.DataFrame(
            {"step": [e.step for e in events], "value": [e.value for e in events]}
        )

    # Extract global lambda
    if "bregman/global_lambda" in available_scalars:
        events = ea.Scalars("bregman/global_lambda")
        result["global_lambda"] = pd.DataFrame(
            {"step": [e.step for e in events], "value": [e.value for e in events]}
        )

    # Extract validation metrics (any scalar starting with "valid/")
    for scalar_tag in available_scalars:
        if scalar_tag.startswith("valid/"):
            events = ea.Scalars(scalar_tag)
            result["validation_metrics"][scalar_tag] = pd.DataFrame(
                {"step": [e.step for e in events], "value": [e.value for e in events]}
            )

    return result


def verify_experiment(
    metrics: Dict[str, Any], baseline_eer: float = 8.86
) -> Dict[str, Tuple[bool, str]]:
    """Verify experiment correctness and stability.

    Args:
        metrics: Output from extract_metrics
        baseline_eer: Baseline EER for degradation computation (default: 8.86%)

    Returns:
        Dict of {check_name: (passed: bool, message: str)}
    """
    checks = {}

    config = metrics["config"]
    sparsity_df = metrics["sparsity"]
    lambda_df = metrics["global_lambda"]

    # Extract target sparsity from config
    try:
        target_sparsity = config.callbacks.model_pruning.lambda_scheduler.target_sparsity
    except (AttributeError, KeyError):
        target_sparsity = None

    # Check 1: Target achievement
    if sparsity_df is not None and target_sparsity is not None:
        final_sparsity = sparsity_df["value"].iloc[-1]
        diff = abs(final_sparsity - target_sparsity)
        passed = diff <= 0.02
        checks["target_achievement"] = (
            passed,
            f"Final sparsity: {final_sparsity:.3f}, Target: {target_sparsity:.3f}, Diff: {diff:.3f}",
        )
    else:
        checks["target_achievement"] = (
            None,
            "Cannot verify - sparsity or target not available",
        )

    # Check 2: Training stability (no NaN in loss)
    train_loss_metrics = [
        k for k in metrics["validation_metrics"].keys() if "loss" in k.lower()
    ]
    if train_loss_metrics:
        for loss_key in train_loss_metrics:
            loss_df = metrics["validation_metrics"][loss_key]
            has_nan = loss_df["value"].isna().any()
            checks[f"stability_{loss_key}"] = (
                not has_nan,
                "No NaN values" if not has_nan else "Contains NaN values",
            )
    else:
        checks["stability_loss"] = (None, "No loss metrics found")

    # Check 3: Lambda bounds
    if lambda_df is not None:
        try:
            max_lambda = config.callbacks.model_pruning.lambda_scheduler.max_lambda
            hits_max_count = (lambda_df["value"] >= max_lambda * 0.99).sum()
            total_steps = len(lambda_df)
            ratio = hits_max_count / total_steps
            passed = ratio < 0.1
            checks["lambda_bounds"] = (
                passed,
                f"Lambda at max for {ratio*100:.1f}% of steps (threshold: 10%)",
            )
        except (AttributeError, KeyError):
            checks["lambda_bounds"] = (None, "max_lambda not in config")
    else:
        checks["lambda_bounds"] = (None, "Lambda data not available")

    # Check 4: Sparsity convergence
    if sparsity_df is not None:
        # Calculate deltas
        deltas = sparsity_df["value"].diff().abs()
        major_reversals = (deltas > 0.05).sum()
        total_steps = len(sparsity_df) - 1
        ratio = major_reversals / total_steps if total_steps > 0 else 0
        passed = ratio < 0.1
        checks["sparsity_convergence"] = (
            passed,
            f"Major reversals (|Δ| > 0.05): {major_reversals}/{total_steps} ({ratio*100:.1f}%)",
        )
    else:
        checks["sparsity_convergence"] = (None, "Sparsity data not available")

    return checks


def generate_comparison_table(
    experiment_dirs: List[Path], baseline_eer: float = 8.86
) -> pd.DataFrame:
    """Generate comparison table for multiple experiments.

    Args:
        experiment_dirs: List of experiment output directories
        baseline_eer: Baseline EER for degradation computation

    Returns:
        DataFrame with columns: [Config, Target, Achieved Sparsity, Best EER,
                                  EER Degradation, Lambda Final, Stable]
    """
    rows = []

    for exp_dir in experiment_dirs:
        try:
            metrics = extract_metrics(exp_dir)
            config = metrics["config"]

            # Extract config name from tags
            tags = config.get("tags", [])
            config_type = "unknown"
            for tag in tags:
                if tag in ["inverse_scale"]:
                    config_type = tag
                    break

            # Get target sparsity
            try:
                target = config.callbacks.model_pruning.lambda_scheduler.target_sparsity
            except (AttributeError, KeyError):
                target = None

            # Get achieved sparsity
            sparsity_df = metrics["sparsity"]
            achieved_sparsity = (
                sparsity_df["value"].iloc[-1] if sparsity_df is not None else None
            )

            # Get best validation metric (look for EER, Accuracy, or any valid/ metric)
            best_metric = None
            metric_name = None
            for metric_key, metric_df in metrics["validation_metrics"].items():
                if metric_df is not None and not metric_df.empty:
                    # Assume lower is better for EER, higher for Accuracy
                    if "eer" in metric_key.lower():
                        best_metric = metric_df["value"].min()
                        metric_name = "EER"
                    elif "accuracy" in metric_key.lower():
                        best_metric = metric_df["value"].max()
                        metric_name = "Accuracy"
                    elif best_metric is None:
                        # Use first available metric
                        best_metric = metric_df["value"].iloc[-1]
                        metric_name = metric_key.split("/")[-1]
                    break

            # Calculate degradation (only for EER)
            degradation = None
            if best_metric is not None and metric_name == "EER":
                degradation = best_metric - baseline_eer

            # Get final lambda
            lambda_df = metrics["global_lambda"]
            final_lambda = (
                lambda_df["value"].iloc[-1] if lambda_df is not None else None
            )

            # Check stability
            checks = verify_experiment(metrics, baseline_eer)
            stable = all(
                result[0] for result in checks.values() if result[0] is not None
            )

            rows.append(
                {
                    "Config": config_type,
                    "Target": f"{target:.2f}" if target else "N/A",
                    "Achieved Sparsity": (
                        f"{achieved_sparsity:.3f}" if achieved_sparsity else "N/A"
                    ),
                    "Best Metric": f"{best_metric:.3f}" if best_metric else "N/A",
                    "Metric Type": metric_name or "N/A",
                    "EER Degradation": (
                        f"{degradation:+.2f}" if degradation is not None else "N/A"
                    ),
                    "Lambda Final": f"{final_lambda:.2e}" if final_lambda else "N/A",
                    "Stable": "Yes" if stable else "No",
                    "Run Directory": exp_dir.name,
                }
            )

        except Exception as e:
            rows.append(
                {
                    "Config": "ERROR",
                    "Target": "N/A",
                    "Achieved Sparsity": "N/A",
                    "Best Metric": "N/A",
                    "Metric Type": "N/A",
                    "EER Degradation": "N/A",
                    "Lambda Final": "N/A",
                    "Stable": "No",
                    "Run Directory": f"{exp_dir.name} - Error: {str(e)}",
                }
            )

    return pd.DataFrame(rows)


def plot_sparsity_evolution(
    experiment_dirs: List[Path], output_path: Path, use_epochs: bool = False
) -> None:
    """Plot sparsity evolution over training for multiple experiments.

    Args:
        experiment_dirs: List of experiment output directories
        output_path: Where to save the plot
        use_epochs: If True, use epochs on x-axis; otherwise use steps
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for exp_dir in experiment_dirs:
        try:
            metrics = extract_metrics(exp_dir)
            sparsity_df = metrics["sparsity"]
            config = metrics["config"]

            if sparsity_df is None or sparsity_df.empty:
                continue

            # Extract label from tags
            tags = config.get("tags", [])
            config_type = "unknown"
            target = None
            for tag in tags:
                if tag in ["inverse_scale", "scheduled", "ema"]:
                    config_type = tag
                if isinstance(tag, str) and tag.startswith("sparsity_"):
                    target = float(tag.split("_")[1])

            # Get target from config if not in tags
            if target is None:
                try:
                    target = (
                        config.callbacks.model_pruning.lambda_scheduler.target_sparsity
                    )
                except (AttributeError, KeyError):
                    target = None

            label = f"{config_type}"
            if target is not None:
                label += f" (target={target:.1f})"

            # Plot sparsity
            x_data = sparsity_df["step"]
            if use_epochs:
                # Estimate epochs from steps (rough approximation)
                steps_per_epoch = config.get("trainer", {}).get("max_epochs", 25)
                x_data = x_data / (len(sparsity_df) / steps_per_epoch)

            ax.plot(x_data, sparsity_df["value"], label=label, linewidth=2, alpha=0.8)

            # Plot target line
            if target is not None:
                ax.axhline(
                    y=target,
                    linestyle="--",
                    alpha=0.5,
                    linewidth=1,
                    color=ax.get_lines()[-1].get_color(),
                )

        except Exception as e:
            print(f"Warning: Could not plot {exp_dir.name}: {e}")
            continue

    ax.set_xlabel("Epoch" if use_epochs else "Step", fontsize=12)
    ax.set_ylabel("Sparsity", fontsize=12)
    ax.set_title("Bregman Sparsity Evolution", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Sparsity evolution plot saved to: {output_path}")
    plt.close()


def find_recent_runs(base_dir: Path, tag: str, n: int) -> List[Path]:
    """Find N most recent runs with a specific tag.

    Args:
        base_dir: Base directory to search (e.g., logs/train/runs)
        tag: Tag to search for (e.g., "bregman_verify")
        n: Number of recent runs to return

    Returns:
        List of run directories, sorted by modification time (newest first)
    """
    runs = []

    if not base_dir.exists():
        return runs

    for run_dir in base_dir.iterdir():
        if not run_dir.is_dir():
            continue

        config_path = run_dir / ".hydra" / "config.yaml"
        if not config_path.exists():
            continue

        try:
            config = OmegaConf.load(config_path)
            tags = config.get("tags", [])
            if tag in tags:
                runs.append((run_dir, run_dir.stat().st_mtime))
        except Exception:
            continue

    # Sort by modification time (newest first)
    runs.sort(key=lambda x: x[1], reverse=True)

    return [run_dir for run_dir, _ in runs[:n]]


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Bregman pruning experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dirs",
        nargs="+",
        type=Path,
        help="Experiment directories to analyze",
    )

    parser.add_argument(
        "--latest",
        type=int,
        help="Analyze N most recent runs with 'bregman_verify' tag",
    )

    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("logs/train/runs"),
        help="Base directory for finding recent runs (default: logs/train/runs)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for saving plots and tables (default: results/)",
    )

    parser.add_argument(
        "--baseline-eer",
        type=float,
        default=8.86,
        help="Baseline EER for degradation computation (default: 8.86)",
    )

    parser.add_argument(
        "--plot-epochs",
        action="store_true",
        help="Use epochs on x-axis instead of steps",
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting (only generate table)",
    )

    args = parser.parse_args()

    # Determine which runs to analyze
    if args.dirs:
        experiment_dirs = args.dirs
    elif args.latest:
        print(
            f"Finding {args.latest} most recent runs with 'bregman_verify' tag in {args.base_dir}..."
        )
        experiment_dirs = find_recent_runs(args.base_dir, "bregman_verify", args.latest)
        if not experiment_dirs:
            print(f"No runs found with 'bregman_verify' tag in {args.base_dir}")
            return
        print(f"Found {len(experiment_dirs)} runs:")
        for run_dir in experiment_dirs:
            print(f"  - {run_dir}")
    else:
        parser.error("Must specify either --dirs or --latest")

    # Generate comparison table
    print("\nGenerating comparison table...")
    comparison_df = generate_comparison_table(experiment_dirs, args.baseline_eer)

    # Save table
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / "bregman_comparison.csv"
    comparison_df.to_csv(table_path, index=False)
    print(f"Comparison table saved to: {table_path}")

    # Print table
    print("\n" + "=" * 80)
    print("BREGMAN EXPERIMENT COMPARISON")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80)

    # Generate plot
    if not args.no_plot:
        print("\nGenerating sparsity evolution plot...")
        plot_path = args.output_dir / "bregman_sparsity_evolution.png"
        plot_sparsity_evolution(experiment_dirs, plot_path, args.plot_epochs)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
