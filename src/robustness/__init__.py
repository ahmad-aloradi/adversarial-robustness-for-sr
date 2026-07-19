"""Post-hoc robustness benchmarks for finished image-classification runs.

Two benchmarks, both evaluated on the exact checkpoint whose clean
``test_accuracy`` is recorded in the run's ``results.json``:

- Corruptions: the published {dataset}-C sets, read as shipped
  (``corruption_data``); every corruption type on disk, severities 1-5.
- Adversarial attacks: any ``torchattacks`` class via a generic adapter
  (``attacks``); AutoAttack is the first configured one. Attacks need no
  training — they perturb test inputs against the frozen checkpoint.

Entry point: ``scripts/eval_robustness.py``.
"""

from src.robustness.accuracy import evaluate_accuracy
from src.robustness.attacks import evaluate_attack
from src.robustness.corruption_data import (
    CorruptionDataset,
    build_corruption_loader,
    list_corruptions,
)
from src.robustness.normalization import NormalizedModel, extract_normalization
from src.robustness.results_io import (
    merge_robustness_results,
    read_reference_test_accuracy,
    resolve_best_checkpoint,
)
from src.robustness.run_loggers import (
    append_metrics_to_run,
    build_run_loggers,
    find_wandb_run_id,
)
from src.robustness.sample_io import (
    attack_sample_dir,
    save_adversarial_samples,
)

__all__ = [
    "CorruptionDataset",
    "NormalizedModel",
    "append_metrics_to_run",
    "attack_sample_dir",
    "build_corruption_loader",
    "build_run_loggers",
    "evaluate_accuracy",
    "evaluate_attack",
    "extract_normalization",
    "find_wandb_run_id",
    "list_corruptions",
    "merge_robustness_results",
    "read_reference_test_accuracy",
    "resolve_best_checkpoint",
    "save_adversarial_samples",
]
