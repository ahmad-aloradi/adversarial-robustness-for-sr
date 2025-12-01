import csv
import json
from collections import OrderedDict
import shutil
from pathlib import Path
from typing import Any, List, Optional, Union

import torch
from pytorch_lightning import LightningModule, Trainer

from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


def _finalize_pruned_state_dict(state_dict: OrderedDict) -> OrderedDict:
    """Convert torch pruning reparameterization keys to dense parameters."""

    converted = OrderedDict()
    modified = False

    for key, value in state_dict.items():
        if key.endswith("_mask"):
            # Mask tensors are only an implementation detail; drop them entirely
            modified = True
            continue

        if key.endswith("_orig"):
            base_key = key[:-5]
            mask_key = f"{base_key}_mask"
            mask = state_dict.get(mask_key)
            tensor = value
            if mask is not None and isinstance(mask, torch.Tensor):
                # Reapply the mask once so the checkpoint stores the sparse tensor directly
                tensor = value * mask
            converted[base_key] = tensor
            modified = True
            continue

        # Skip placeholder entry if corresponding _orig exists
        if f"{key}_orig" in state_dict:
            # torch.nn.utils.prune leaves a thin wrapper parameter that just references *_orig
            # Avoid duplicating it in the rewritten state dict
            continue

        converted[key] = value

    return converted if modified else state_dict


def make_checkpoint_pruning_permanent(ckpt_path: str, backup: bool = True) -> bool:
    """Rewrite a Lightning checkpoint so pruned params are stored without masks."""

    if not Path(ckpt_path).is_file():
        log.warning(f"Checkpoint {ckpt_path} not found; skipping pruning finalization")
        return False

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        log.warning(f"Checkpoint {ckpt_path} missing state_dict; skipping")
        return False

    new_state_dict = _finalize_pruned_state_dict(state_dict)
    if new_state_dict is state_dict:
        # Nothing to do
        return False

    checkpoint["state_dict"] = new_state_dict
    if backup:
        backup_path = f"{ckpt_path}.pre_pruning_backup"
        # Keep a copy of the original sparse checkpoint for reproducibility/debugging
        shutil.copy2(ckpt_path, backup_path)
        log.info(f"Backup of original checkpoint saved to: {backup_path}")

    # Overwrite original checkpoint with mask-free weights so downstream loads succeed
    torch.save(checkpoint, ckpt_path)
    log.info(f"Pruning made permanent in checkpoint: {ckpt_path}")
    return True


def process_state_dict(
    state_dict: Union[OrderedDict, dict],
    symbols: int = 0,
    exceptions: Optional[Union[str, List[str]]] = None,
) -> OrderedDict:
    """Filter and map model state dict keys.

    Args:
        state_dict (Union[OrderedDict, dict]): State dict.
        symbols (int): Determines how many symbols should be cut in the
            beginning of state dict keys. Default to 0.
        exceptions (Union[str, List[str]], optional): Determines exceptions,
            i.e. substrings, which keys should not contain.

    Returns:
        OrderedDict: Filtered state dict.
    """

    new_state_dict = OrderedDict()
    if exceptions:
        if isinstance(exceptions, str):
            exceptions = [exceptions]
    for key, value in state_dict.items():
        is_exception = False
        if exceptions:
            for exception in exceptions:
                if key.startswith(exception):
                    is_exception = True
        if not is_exception:
            new_state_dict[key[symbols:]] = value

    return new_state_dict


def save_state_dicts(
    trainer: Trainer,
    model: LightningModule,
    dirname: str,
    symbols: int = 6,
    exceptions: Optional[Union[str, List[str]]] = None,
) -> None:
    """Save model state dicts for last and best checkpoints.

    Args:
        trainer (Trainer): Lightning trainer.
        model (LightningModule): Lightning model.
        dirname (str): Saving directory.
        symbols (int): Determines how many symbols should be cut in the
            beginning of state dict keys. Default to 6 for cutting
            Lightning name prefix.
        exceptions (Union[str, List[str]], optional): Determines exceptions,
            i.e. substrings, which keys should not contain.  Default to [loss].
    """

    # save state dict for last checkpoint
    mapped_state_dict = process_state_dict(
        model.state_dict(), symbols=symbols, exceptions=exceptions
    )
    path = f"{dirname}/last_ckpt.pth"
    torch.save(mapped_state_dict, path)
    log.info(f"Last ckpt state dict saved to: {path}")

    # save state dict for best checkpoint
    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    if best_ckpt_path == "":
        log.warning("Best ckpt not found! Skipping...")
        return

    best_ckpt_score = trainer.checkpoint_callback.best_model_score
    if best_ckpt_score is not None:
        prefix = str(best_ckpt_score.detach().cpu().item())
        prefix = prefix.replace(".", "_")
    else:
        log.warning("Best ckpt score not found! Use prefix <unknown>!")
        prefix = "unknown"

    # Ensure checkpoint does not contain pruning reparameterization tensors
    made_permanent = make_checkpoint_pruning_permanent(best_ckpt_path, backup=False)
    if made_permanent:
        log.info("Best checkpoint converted to dense weights before export")

    # load model from best checkpoint (note that .load_from_checkpoint is a classmethod!)
    model = type(model).load_from_checkpoint(best_ckpt_path)
    mapped_state_dict = process_state_dict(
        model.state_dict(), symbols=symbols, exceptions=exceptions
    )
    path = f"{dirname}/best_ckpt_{prefix}.pth"
    torch.save(mapped_state_dict, path)
    log.info(f"Best ckpt state dict saved to: {path}")


def save_predictions_from_dataloader(
    predictions: List[Any], path: Path
) -> None:
    """Save predictions returned by `Trainer.predict` method for single
    dataloader.

    Args:
        predictions (List[Any]): Predictions returned by `Trainer.predict` method.
        path (Path): Path to predictions.
    """

    if path.suffix == ".csv":
        with open(path, "w") as csv_file:
            writer = csv.writer(csv_file)
            for batch in predictions:
                keys = list(batch.keys())
                batch_size = len(batch[keys[0]])
                for i in range(batch_size):
                    row = {key: batch[key][i].tolist() for key in keys}
                    writer.writerow(row)

    elif path.suffix == ".json":
        processed_predictions = {}
        for batch in predictions:
            keys = [key for key in batch.keys() if key != "names"]
            batch_size = len(batch[keys[0]])
            for i in range(batch_size):
                item = {key: batch[key][i].tolist() for key in keys}
                if "names" in batch.keys():
                    processed_predictions[batch["names"][i]] = item
                else:
                    processed_predictions[len(processed_predictions)] = item
        with open(path, "w") as json_file:
            json.dump(processed_predictions, json_file, ensure_ascii=False)

    else:
        raise NotImplementedError(f"{path.suffix} is not implemented!")


def save_predictions(
    predictions: List[Any], dirname: str, output_format: str = "json"
) -> None:
    """Save predictions returned by `Trainer.predict` method.

    Due to `LightningDataModule.predict_dataloader` return type is
    Union[DataLoader, List[DataLoader]], so `Trainer.predict` method can return
    a list of dictionaries, one for each provided batch containing their
    respective predictions, or a list of lists, one for each provided dataloader
    containing their respective predictions, where each list contains dictionaries.

    Args:
        predictions (List[Any]): Predictions returned by `Trainer.predict` method.
        dirname (str): Dirname for predictions.
        output_format (str): Output file format. It could be `json` or `csv`.
            Default to `json`.
    """

    if not predictions:
        log.warning("Predictions is empty! Saving was cancelled ...")
        return

    if output_format not in ("json", "csv"):
        raise NotImplementedError(
            f"{output_format} is not implemented! Use `json` or `csv`."
            "Or change `src.utils.saving.save_predictions` func logic."
        )

    path = Path(dirname) / "predictions"
    path.mkdir(parents=True, exist_ok=True)

    if isinstance(predictions[0], dict):
        target_path = path / f"predictions.{output_format}"
        save_predictions_from_dataloader(predictions, target_path)
        log.info(f"Saved predictions to: {str(target_path)}")
        return

    elif isinstance(predictions[0], list):
        for idx, predictions_idx in enumerate(predictions):
            if not predictions_idx:
                log.warning(
                    f"Predictions for DataLoader #{idx} is empty! Skipping..."
                )
                continue
            target_path = path / f"predictions_{idx}.{output_format}"
            save_predictions_from_dataloader(predictions_idx, target_path)
            log.info(
                f"Saved predictions for DataLoader #{idx} to: "
                f"{str(target_path)}"
            )
        return

    raise Exception(
        "Passed predictions format is not supported by default!\n"
        "Make sure that it is formed correctly! It requires as List[Dict[str, Any]] type"
        "in case of predict_dataloader returns DataLoader or List[List[Dict[str, Any]]]"
        "type in case of predict_dataloader returns List[DataLoader]!\n"
        "Or change `src.utils.saving.save_predictions` function logic."
    )
