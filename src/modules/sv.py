import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src import utils
from src.callbacks.pruning.utils.pruning_manager import PruningManager
from src.datamodules.components.utils import (
    BaseDataset,
    FullUtteranceCohortDataset,
)
from src.datamodules.components.voxceleb.voxceleb_dataset import (
    TrainCollate,
    VoxcelebItem,
    VoxCelebVerificationItem,
)
from src.modules.encoder_wrappers import EncoderWrapper
from src.modules.scoring import ScoringPipeline, build_scoring_pipeline

log = utils.get_pylogger(__name__)

# Constants for test checkpointing
TEST_CHECKPOINT_INTERVAL = 50_000  # Save trial results every N batches


class SpeakerVerification(pl.LightningModule):
    """SV model for speaker verification with audio embeddings."""

    def __init__(
        self,
        model: DictConfig,
        criterion: DictConfig,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        logging_params: DictConfig,
        metrics: DictConfig,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the model with configurations for all components."""
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.logging_params = logging_params
        self.data_augmentation = kwargs.get("data_augmentation", None)

        # Initialize metrics
        self._setup_metrics(metrics)

        # Initialize model components
        self._setup_model_components(model)

        # Setup training components
        self._setup_training_components(criterion, optimizer, lr_scheduler)

        # Initialize text embedding cache with appropriate limits
        self._embeds_cache_config = model.get("embedding_cache", {})
        self._max_cache_size = self._embeds_cache_config.get(
            "max_size", 500000
        )
        self._bypass_warmup = self._embeds_cache_config.get(
            "bypass_warmup", False
        )

        # Scoring pipeline (handles centering, normalization, enrollment aggregation)
        scoring_config = kwargs.get("scoring", {})
        self.scoring_pipeline = build_scoring_pipeline(config=scoring_config)

    # Setup init
    def _setup_metrics(self, metrics: DictConfig) -> None:
        """Initialize all metrics for training, validation and testing."""
        self.train_metric = instantiate(metrics.train)
        self.valid_metric = instantiate(metrics.valid)
        self.test_metric = instantiate(metrics.test)
        self.valid_metric_best = instantiate(metrics.valid_best)

    def _setup_model_components(self, model: DictConfig) -> None:
        """Initialize encoders and classifiers, wrapping the encoder for a
        unified interface."""
        # Audio processing
        self.audio_processor = instantiate(model.audio_processor)
        self.audio_processor_normalizer = instantiate(
            model.audio_processor_normalizer
        )

        # Instantiate the raw encoder
        raw_audio_encoder = instantiate(model.audio_encoder)

        # Wrap the encoder and its pre-processing steps
        self.audio_encoder = EncoderWrapper(
            encoder=raw_audio_encoder,
            audio_processor=self.audio_processor,
            audio_processor_normalizer=self.audio_processor_normalizer,
        )

        self.classifier = instantiate(model.classifier)

        # Setup wav augmentation if configured
        if self.data_augmentation is not None:
            assert (
                "wav_augmenter" in self.data_augmentation.augmentations
            ), "Expected augmentations.wav_augmenter when passing data_augmentation"
            self.wav_augmenter = instantiate(
                self.data_augmentation.augmentations.wav_augmenter
            )

    def _setup_training_components(
        self,
        criterion: DictConfig,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
    ) -> None:
        """Initialize loss functions, optimizer and learning rate scheduler."""
        self.train_criterion = instantiate(criterion.train_criterion)
        self.valid_criterion = instantiate(criterion.valid_criterion)
        self.optimizer = optimizer
        self.slr_params = lr_scheduler

    def _log_step_metrics(
        self, results: Dict[str, Any], batch: VoxcelebItem, stage: str
    ) -> None:
        criterion = getattr(self, f"{stage}_criterion")

        # Log losses
        logged_dict = {
            f"{stage}/{criterion.__class__.__name__}": results["loss"].item()
        }

        self.log_dict(
            logged_dict, batch_size=batch.audio.shape[0], **self.logging_params
        )

        # Log metrics
        metric = getattr(self, f"{stage}_metric")
        computed_metric = metric(results["outputs"]["logits"], batch.class_id)

        self.log(
            f"{stage}/{metric.__class__.__name__}",
            computed_metric,
            batch_size=batch.audio.shape[0],
            **self.logging_params,
        )

    # Lightning hooks
    def _get_audio_embeddings(
        self, batch_audio: torch.Tensor, batch_audio_lens: torch.Tensor
    ) -> torch.Tensor:
        """Computes audio embeddings using the wrapped encoder, which handles
        the full pipeline."""
        # Move tensors to the correct device.
        if self.device != batch_audio.device:
            batch_audio = batch_audio.to(self.device)
        if self.device != batch_audio_lens.device:
            batch_audio_lens = batch_audio_lens.to(self.device)

        return self.audio_encoder(wavs=batch_audio, wav_lens=batch_audio_lens)

    def forward(self, batch: VoxcelebItem) -> Dict[str, torch.Tensor]:
        """Process audio inputs with optimized embedding caching."""
        # Add waveform augmentation if specified.
        if self.training and hasattr(self, "wav_augmenter"):
            max_audio_length = max(batch.audio_length)
            batch.audio, audio_length_norm = self.wav_augmenter(
                batch.audio, batch.audio_length / max_audio_length
            )
            batch.audio_length = audio_length_norm * max_audio_length
            batch.class_id = self.wav_augmenter.replicate_labels(
                batch.class_id
            )

        audio_emb = self._get_audio_embeddings(batch.audio, batch.audio_length)
        logits = self.classifier(audio_emb)

        return {"embeds": audio_emb, "logits": logits}

    def model_step(
        self, batch: VoxcelebItem, criterion: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Perform a single model step."""
        outputs = self(batch)

        # Compute loss
        if isinstance(criterion, torch.nn.CrossEntropyLoss):
            main_loss = criterion(outputs["logits"], batch.class_id)
        elif criterion.__class__.__name__ == "LogSoftmaxWrapper":
            if outputs["logits"].ndim == 3 and outputs["logits"].shape[1] == 1:
                outputs["logits"] = outputs["logits"].squeeze(1)
            main_loss = criterion(
                outputs["logits"].unsqueeze(1), batch.class_id.unsqueeze(1)
            )
        else:
            raise ValueError("Invalid criterion")

        return {"loss": main_loss, "outputs": outputs}

    def on_train_start(self) -> None:
        # by default lightning executes validation step sanity checks before
        # training starts, so we need to make sure valid_metric_best doesn't store
        # accuracy from these checks
        self.valid_metric_best.reset()
        self.audio_encoder.train()

    def training_step(
        self, batch: VoxcelebItem, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        results = self.model_step(batch, self.train_criterion)

        if batch_idx % 3000 == 0:
            torch.cuda.empty_cache()

        self._log_step_metrics(results, batch, "train")

        return results

    def on_train_epoch_end(self) -> None:
        self.train_metric.reset()

    def on_validation_start(self) -> None:
        pass

    def on_validation_epoch_end(self) -> None:
        self.valid_metric.reset()

    @torch.inference_mode()
    def validation_step(
        self, batch: VoxcelebItem, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        results = self.model_step(batch, self.valid_criterion)
        self._log_step_metrics(results, batch, "valid")
        return results

    @torch.inference_mode()
    def on_test_start(self) -> None:
        """Pre-computes embeddings and prepares for per-dataloader metric
        logging."""
        # Get all configured test dataloaders from the datamodule
        test_dataloaders = self.trainer.datamodule.test_dataloader()

        # Normalize to dictionary format for consistent handling
        if not isinstance(test_dataloaders, dict):
            # Infer a descriptive base name from the datamodule class
            datamodule_class = self.trainer.datamodule.__class__.__name__
            base_name = (
                datamodule_class.replace("DataModule", "")
                .replace("Module", "")
                .lower()
            )

            if isinstance(test_dataloaders, (list, tuple)):
                # Multiple loaders returned as list/tuple - use base_name with index
                test_dataloaders = {
                    f"{base_name}_{i}": loader
                    for i, loader in enumerate(test_dataloaders)
                }
                log.info(
                    f"Normalized {len(test_dataloaders)} test loaders to dictionary format with base name '{base_name}'"
                )
            else:
                # Single loader returned - use just the base_name
                test_dataloaders = {base_name: test_dataloaders}
                log.info(
                    f"Normalized single test loader to dictionary format with key '{base_name}'"
                )

        # Store normalized dict for use in test_step
        self.test_dataloaders_dict = test_dataloaders
        test_filenames = list(test_dataloaders.keys())

        log.info(
            f"Found {len(test_filenames)} test set(s): {', '.join(test_filenames)}"
        )

        # Compute cohort embeddings per dataset (if needed for scoring pipeline)
        cohort_data_by_dataset: Dict[str, Dict[str, torch.Tensor]] = {}
        train_dm = self.trainer.datamodule

        self.test_sets_data = {}
        self.last_batch_indices = {}

        # Check if scoring pipeline needs cohort data
        needs_cohort = (
            self.scoring_pipeline.config.norm_method != "none"
            or self.scoring_pipeline.config.mean_source == "cohort"
        )

        for test_filename, dataloader in test_dataloaders.items():
            # Skip already completed test sets
            if self._is_test_set_complete(test_filename):
                log.info(f"Skipping '{test_filename}' - already complete")
                self.last_batch_indices[test_filename] = -2  # Mark as skip
                continue

            log.info(f"Processing '{test_filename}'...")
            base_dataset = test_filename.split("/")[0]

            # Get cohort data if needed for scoring
            cohort_data = None
            if needs_cohort:
                cohort_data = self._get_cohort_embeddings_for_dataset(
                    base_dataset=base_dataset,
                    train_dm=train_dm,
                    cache=cohort_data_by_dataset,
                )

            # Determine the index of the last batch for this dataloader
            num_batches = len(dataloader)
            if num_batches > 0:
                self.last_batch_indices[test_filename] = num_batches - 1
                log.info(
                    f"Registered '{test_filename}' with {num_batches} batches. Last batch index: {num_batches - 1}."
                )
            else:
                raise ValueError("Dataloader has zero batches")

            # Try to load cached embeddings, otherwise compute them
            cached = self._load_cached_embeddings(test_filename)
            if cached:
                enrol_embeds, test_embeds = (
                    cached["enrol_embeds"],
                    cached["test_embeds"],
                )
            else:
                (
                    enroll_dataloader,
                    trial_unique_dataloader,
                ) = self.trainer.datamodule.get_enroll_and_trial_dataloaders(
                    test_filename
                )
                enrol_embeds = self._compute_embeddings(
                    enroll_dataloader, mode="enrollment"
                )
                test_embeds = self._compute_embeddings(
                    trial_unique_dataloader, mode="test"
                )
                self._save_embeddings_cache(
                    test_filename, enrol_embeds, test_embeds
                )

            # Load partial trial results if resuming
            (
                trial_results,
                resume_batch_idx,
                resume_batch_paths,
            ) = self._load_partial_trial_results(test_filename)

            # Configure scoring pipeline with cohort data for this dataset
            scoring_pipeline = build_scoring_pipeline(
                config=self.hparams.get("scoring", {})
            )
            if cohort_data is not None:
                scoring_pipeline.set_cohort(
                    embeddings=cohort_data["embeddings"],
                    speaker_ids=cohort_data.get("speaker_ids"),
                )
                log.info(
                    f"Configured scoring pipeline for '{test_filename}': "
                    f"enrollment_aggregation={scoring_pipeline.config.enrollment_aggregation}, "
                    f"norm={scoring_pipeline.config.norm_method}, "
                    f"mean_source={scoring_pipeline.config.mean_source}"
                )

            # Store all data for this test set
            self.test_sets_data[test_filename] = {
                "enrol_embeds": enrol_embeds,
                "test_embeds": test_embeds,
                "scoring_pipeline": scoring_pipeline,
                "trial_results": trial_results,
                "resume_batch_idx": resume_batch_idx,
                "resume_batch_paths": resume_batch_paths,
            }

        log.info("Finished computing embeddings for all test sets")

    @torch.inference_mode()
    def test_step(
        self,
        batch: VoxCelebVerificationItem,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Handle test step and trigger logging on the last batch of each
        dataloader."""
        # Get test set name from dataloader index
        test_filenames = list(self.test_dataloaders_dict.keys())
        test_filename = test_filenames[dataloader_idx]

        # Skip if this test set was already completed or marked to skip
        if self.last_batch_indices.get(test_filename, -1) == -2:
            return

        # Skip batches already processed (resumption) with path check on the boundary batch
        test_data = self.test_sets_data[test_filename]
        resume_idx = test_data.get("resume_batch_idx", -1)
        resume_paths = test_data.get("resume_batch_paths")
        if batch_idx < resume_idx:
            return
        if batch_idx == resume_idx and resume_paths is not None:
            if list(batch.enroll_path) == resume_paths.get(
                "enroll_path"
            ) and list(batch.test_path) == resume_paths.get("test_path"):
                return
            log.warning(
                f"Resume boundary mismatch for '{test_filename}' at batch {batch_idx}; processing batch to realign."
            )

        # Move batch to device
        batch = self._move_batch_to_device(batch)

        # Run trial evaluation for this test set
        self._trials_eval_step_multi_test(batch, test_data)

        # Periodic checkpointing
        if (batch_idx + 1) % TEST_CHECKPOINT_INTERVAL == 0:
            self._save_trial_results_checkpoint(
                test_filename,
                test_data["trial_results"],
                batch_idx,
                {
                    "enroll_path": list(batch.enroll_path),
                    "test_path": list(batch.test_path),
                },
            )
            log.info(
                f"Checkpoint saved for '{test_filename}' at batch {batch_idx}"
            )

        # Check if this is the last batch for the current dataloader
        is_last_batch = batch_idx == self.last_batch_indices.get(
            test_filename, -1
        )

        if is_last_batch:
            log.info(
                f"Last batch for '{test_filename}' (idx: {batch_idx}) reached. Finalizing and logging metrics."
            )
            run_timestamp = self._epoch_end_common_multi_test(test_filename)
            self._mark_test_complete(
                test_filename, run_timestamp=run_timestamp
            )

    def on_test_epoch_end(self) -> None:
        """Callback at the end of the test epoch.

        This is now primarily a cleanup step. It also handles any test sets
        where the number of batches could not be determined.
        """
        # Process any test sets that were not handled in test_step (fallback)
        if hasattr(self, "last_batch_indices"):
            for test_filename, last_idx in self.last_batch_indices.items():
                if last_idx == -2:  # Skipped (already complete)
                    continue
                if last_idx == -1:
                    log.warning(
                        f"Running fallback metric computation for '{test_filename}' at epoch end."
                    )
                    run_timestamp = self._epoch_end_common_multi_test(
                        test_filename
                    )
                    self._mark_test_complete(
                        test_filename, run_timestamp=run_timestamp
                    )

        log.info("Test epoch finished. All test sets have been processed.")
        # Clean up stored data to free memory
        if hasattr(self, "test_sets_data"):
            del self.test_sets_data
        if hasattr(self, "last_batch_indices"):
            del self.last_batch_indices
        if hasattr(self, "test_dataloaders_dict"):
            del self.test_dataloaders_dict

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and learning rate schedulers.

        This method dynamically adapts its behavior based on the configured optimizer:

        1.  If a Bregman-type optimizer (AdaBreg, LinBreg, ProxSGD) is used,
            it initializes the PruningManager to handle parameter groups with
            specific regularization settings defined in the config.

        2.  For any standard optimizer (e.g., Adam, SGD), it applies the
            optimizer to all model parameters uniformly.
        """
        BREGMAN_OPTIMIZERS = {"AdaBreg", "LinBreg", "ProxSGD"}
        optimizer_class_name = self.hparams.optimizer._target_.split(".")[-1]

        # Use the two-step partial instantiation pattern for the optimizer
        optimizer_partial = instantiate(self.hparams.optimizer)

        if optimizer_class_name in BREGMAN_OPTIMIZERS:
            # --- Pruning-Aware Optimizer Logic ---
            self.pruning_manager = PruningManager(
                pl_module=self, group_configs=self.hparams.model.pruning_groups
            )
            optimizer_param_groups = (
                self.pruning_manager.get_optimizer_param_groups()
            )

            # Manually instantiate the regularization object for each group
            for group in optimizer_param_groups:
                if "reg" in group and isinstance(
                    group.get("reg"), (dict, DictConfig)
                ):
                    group["reg"] = instantiate(group["reg"])

            optimizer = optimizer_partial(params=optimizer_param_groups)

        else:
            # --- Standard Optimizer Logic ---
            optimizer = optimizer_partial(params=self.parameters())

        # --- Common Scheduler Logic ---
        if self.hparams.get("lr_scheduler"):
            # Instantiate the scheduler, which now receives a fully formed optimizer
            scheduler = instantiate(
                self.hparams.lr_scheduler.scheduler, optimizer=optimizer
            )

            lr_scheduler_dict = {"scheduler": scheduler}
            if self.hparams.lr_scheduler.get("extras"):
                for key, value in self.hparams.lr_scheduler.get(
                    "extras"
                ).items():
                    lr_scheduler_dict[key] = value
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}

        return {"optimizer": optimizer}

    # Dev and eval utils
    def _get_test_artifacts_dir(self, test_filename: str) -> Path:
        """Get the base artifacts directory for a test set."""
        safe_name = test_filename.replace("/", "_").replace("\\", "_")
        return (
            Path(self.trainer.default_root_dir) / "test_artifacts" / safe_name
        )

    def _get_test_cache_dir(self, test_filename: str) -> Path:
        """Get the cache directory for a test set (embeds/checkpoints)."""
        return self._get_test_artifacts_dir(test_filename) / "cache"

    def _get_cohort_cache_path(self, base_dataset: str) -> Path:
        """Get the per-dataset cohort embeddings cache path.

        Stored once per experiment to avoid mixing datasets.
        Uses 'full_utt' suffix to distinguish from legacy segment-based caches.
        """
        cache_root = (
            Path(self.trainer.default_root_dir)
            / "test_artifacts"
            / "_cohort_cache"
        )
        return cache_root / f"{base_dataset}_cohort_full_utt.pt"

    def _save_embeddings_cache(
        self, test_filename: str, enrol_embeds: dict, test_embeds: dict
    ) -> None:
        """Save embeddings to cache for potential resumption."""
        cache_dir = self._get_test_cache_dir(test_filename)
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(enrol_embeds, cache_dir / "enrol_embeds_cache.pt")
        torch.save(test_embeds, cache_dir / "test_embeds_cache.pt")

    def _load_cached_embeddings(
        self, test_filename: str
    ) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """Load cached embeddings for a test set if they exist."""
        cache_dir = self._get_test_cache_dir(test_filename)

        enrol_path = cache_dir / "enrol_embeds_cache.pt"
        test_path = cache_dir / "test_embeds_cache.pt"

        if enrol_path.exists() and test_path.exists():
            log.info(
                f"Loading cached enrollment and test embeddings for '{test_filename}'"
            )
            return {
                "enrol_embeds": torch.load(enrol_path),
                "test_embeds": torch.load(test_path),
            }
        return None

    def _save_trial_results_checkpoint(
        self,
        test_filename: str,
        trial_results: list,
        batch_idx: int,
        batch_paths: dict,
    ) -> None:
        """Save trial results checkpoint for resumption."""
        cache_dir = self._get_test_cache_dir(test_filename)
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "trial_results": trial_results,
                "last_batch_idx": batch_idx,
                "last_batch_paths": batch_paths,
            },
            cache_dir / "trial_results_checkpoint.pt",
        )

    def _load_partial_trial_results(
        self, test_filename: str
    ) -> tuple[list, int, Optional[dict]]:
        """Load partial trial results if they exist.

        Returns (results, last_batch_idx, last_batch_paths).
        """
        cache_dir = self._get_test_cache_dir(test_filename)
        checkpoint_path = cache_dir / "trial_results_checkpoint.pt"

        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            log.info(
                f"Resuming '{test_filename}' from batch {checkpoint['last_batch_idx'] + 1}"
            )
            return (
                checkpoint["trial_results"],
                checkpoint["last_batch_idx"],
                checkpoint.get("last_batch_paths"),
            )
        return [], -1, None

    def _mark_test_complete(
        self, test_filename: str, run_timestamp: Optional[str] = None
    ) -> None:
        """Mark a test set as complete.

        Writes a COMPLETE marker and also records the last successful run
        timestamp (so "COMPLETE" implies there is a corresponding results
        directory).
        """
        base_dir = self._get_test_artifacts_dir(test_filename)
        base_dir.mkdir(parents=True, exist_ok=True)
        (base_dir / "COMPLETE").touch()
        if run_timestamp is not None:
            (base_dir / "LAST_RUN").write_text(str(run_timestamp))
        checkpoint_path = (
            self._get_test_cache_dir(test_filename)
            / "trial_results_checkpoint.pt"
        )
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    def _is_test_set_complete(self, test_filename: str) -> bool:
        """Check if a test set has already been fully evaluated.

        "COMPLETE" alone is treated as insufficient if there is no
        corresponding timestamped results directory.
        """
        base_dir = self._get_test_artifacts_dir(test_filename)
        if not (base_dir / "COMPLETE").exists():
            return False

        last_run_path = base_dir / "LAST_RUN"
        if last_run_path.exists():
            ts = last_run_path.read_text().strip()
            if ts:
                run_dir = base_dir / ts
                if run_dir.is_dir():
                    return True

        # Fallback: accept COMPLETE if at least one timestamped run dir exists.
        has_run_dir = any(
            p.is_dir() and p.name[:9].isdigit() and "_" in p.name
            for p in base_dir.iterdir()
        )
        if not has_run_dir:
            log.warning(
                f"Found COMPLETE for '{test_filename}' but no timestamped results directory under {base_dir}. "
                "Treating as incomplete to regenerate results."
            )
            return False
        return True

    def _compute_embeddings(self, dataloader, mode: str) -> dict:
        """Compute embeddings for test or enrollment data.

        For enrollment mode with CNCeleb multi-utterance enrollments, this method
        aggregates embeddings per enroll_id using the scoring pipeline's configured
        aggregation method (mean or length_weighted).

        Args:
            dataloader: DataLoader yielding batches
            mode: 'test' or 'enrollment'

        Returns:
            Dictionary mapping path/enroll_id to embedding tensor
        """
        embeddings_dict = {}
        desc = f"Computing {mode} embeddings"

        with tqdm(dataloader, desc=desc, leave=False) as pbar:
            for batch in pbar:
                outputs = self(batch)
                embed = outputs["embeds"]
                # Handle frame-level embeddings (if applicable)
                if len(embed.shape) == 3:  # [B, num_frames, embedding_dim]
                    raise NotImplementedError(
                        "Frame-level embeddings not supported in _compute_embeddings"
                    )
                    # embed = embed.mean(dim=1)  # [B, embedding_dim]

                # Check if this is a multi-utterance enrollment batch (CNCeleb multi-mode)
                if (
                    mode == "enrollment"
                    and hasattr(batch, "utt_counts")
                    and batch.utt_counts is not None
                ):
                    # Aggregate embeddings per enroll_id using scoring pipeline
                    # This respects the enrollment_aggregation config (mean or length_weighted)
                    idx = 0
                    for enroll_id, count in zip(batch.enroll_id, batch.utt_counts):
                        utt_embeds = embed[idx : idx + count]  # [count, embed_dim]
                        utt_lengths = batch.audio_length[idx : idx + count]  # [count]
                        aggregated_embed = self.scoring_pipeline.aggregate_enrollment(
                            utt_embeds, lengths=utt_lengths
                        )
                        embeddings_dict[enroll_id] = aggregated_embed
                        idx += count
                elif (
                    mode == "enrollment"
                    and hasattr(batch, "enroll_id")
                    and batch.enroll_id is not None
                ):
                    # Concatenated-enrollment mode with enroll_id (CNCeleb concatenated-mode)
                    embeddings_dict.update(
                        {eid: emb for eid, emb in zip(batch.enroll_id, embed)}
                    )
                else:
                    # Standard single-utterance handling (test mode or VoxCeleb)
                    embeddings_dict.update(
                        {
                            path: emb
                            for path, emb in zip(batch.audio_path, embed)
                        }
                    )

        return embeddings_dict

    def _compute_cohort_embeddings(
        self, dataloader
    ) -> Dict[str, torch.Tensor]:
        """Compute cohort embeddings with speaker IDs for proper AS-Norm.

        Returns:
            Dictionary with:
                - 'embeddings': [N, embed_dim] tensor
                - 'speaker_ids': [N] tensor of integer speaker IDs
        """
        embeddings_list = []
        speaker_ids_list = []
        speaker_to_id = {}
        next_id = 0

        with tqdm(
            dataloader, desc="Computing cohort embeddings", leave=False
        ) as pbar:
            for batch in pbar:
                outputs = self(batch)
                cohort = outputs["embeds"]

                # Handle frame-level embeddings (if applicable)
                if len(cohort.shape) == 3:  # [B, num_frames, embedding_dim]
                    cohort = cohort.mean(dim=1)  # [B, embedding_dim]

                embeddings_list.append(cohort)

                # Extract speaker IDs from batch
                if hasattr(batch, "speaker_id"):
                    batch_speakers = batch.speaker_id
                elif hasattr(batch, "class_id"):
                    batch_speakers = batch.class_id.tolist()
                else:
                    # Fallback: assign unique speaker per utterance (no speaker grouping)
                    batch_speakers = list(
                        range(next_id, next_id + cohort.shape[0])
                    )

                # Map speaker labels to integer IDs
                batch_ids = []
                for spk in batch_speakers:
                    if spk not in speaker_to_id:
                        speaker_to_id[spk] = next_id
                        next_id += 1
                    batch_ids.append(speaker_to_id[spk])
                speaker_ids_list.extend(batch_ids)

        cohort_embeds = torch.cat(embeddings_list, dim=0)
        speaker_ids = torch.tensor(speaker_ids_list, dtype=torch.long)

        log.info(
            f"Cohort: {cohort_embeds.shape[0]} utterances from "
            f"{len(speaker_to_id)} speakers"
        )

        return {
            "embeddings": cohort_embeds,
            "speaker_ids": speaker_ids,
        }

    def _get_cohort_embeddings_for_dataset(
        self,
        base_dataset: str,
        train_dm,
        cache: Dict[str, Dict[str, torch.Tensor]],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Return (and cache) cohort embeddings with speaker IDs for the given
        dataset.

        Returns:
            Dictionary with 'embeddings' and 'speaker_ids' tensors, or None if
            score normalization is disabled.
        """
        if base_dataset in cache:
            return cache[base_dataset]

        cohort_cache_path = self._get_cohort_cache_path(base_dataset)
        if cohort_cache_path.exists():
            log.info(
                f"Loading cohort embeddings for '{base_dataset}' from: {cohort_cache_path}"
            )
            cached_data = torch.load(cohort_cache_path)
            # Handle legacy cache format (tensor only, no speaker IDs)
            if isinstance(cached_data, torch.Tensor):
                log.warning(
                    f"Legacy cohort cache detected for '{base_dataset}'. "
                    "Speaker-level averaging disabled. Delete cache to recompute with speaker IDs."
                )
                cached_data = {"embeddings": cached_data, "speaker_ids": None}
            cache[base_dataset] = cached_data
            return cache[base_dataset]

        # Ensure train_data is populated
        if getattr(train_dm, "train_data", None) is None:
            train_dm.setup(stage="fit")

        train_dataset = train_dm.train_data
        if train_dataset is None:
            raise ValueError(
                f"Score normalization requires training data for '{base_dataset}', "
                "but none was found. Set scoring.norm_method='none' or enable training for this dataset."
            )

        log.info(f"Computing cohort embeddings for '{base_dataset}'...")

        # Always wrap with FullUtteranceCohortDataset to ensure full utterances
        # This handles both pre-segmented data (multiple segments per file) and
        # random cropping mode (max_duration would otherwise crop during loading)
        if isinstance(train_dataset, BaseDataset):
            cohort_dataset = FullUtteranceCohortDataset(train_dataset)
        else:
            log.warning(
                f"Training dataset for '{base_dataset}' is not a BaseDataset; "
                "cohort embeddings may use chunked audio"
            )
            cohort_dataset = train_dataset

        loader_cfg = train_dm.hparams.loaders.train
        cohort_loader = DataLoader(
            cohort_dataset,
            batch_size=getattr(loader_cfg, "batch_size", 64),
            num_workers=getattr(loader_cfg, "num_workers", 0),
            shuffle=False,
            pin_memory=getattr(loader_cfg, "pin_memory", False),
            collate_fn=TrainCollate(),
        )
        cohort_data = self._compute_cohort_embeddings(cohort_loader)
        cache[base_dataset] = cohort_data
        cohort_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cohort_data, cohort_cache_path)
        log.info(
            f"Saved cohort embeddings for '{base_dataset}' to: {cohort_cache_path}"
        )
        return cache[base_dataset]

    def _move_batch_to_device(
        self, batch: VoxCelebVerificationItem
    ) -> VoxCelebVerificationItem:
        """Move batch tensors to the model's device."""
        batch.enroll_audio = batch.enroll_audio.to(self.device)
        batch.test_audio = batch.test_audio.to(self.device)
        batch.enroll_length = batch.enroll_length.to(self.device)
        batch.test_length = batch.test_length.to(self.device)
        return batch

    def _trials_eval_step_multi_test(
        self, batch: VoxCelebVerificationItem, test_data: Dict
    ):
        """Evaluation step for a specific test set using the unified scoring
        pipeline.

        The scoring pipeline handles:
        - Mean centering (if configured)
        - Raw cosine similarity computation
        - Score normalization (AS-Norm, S-Norm, etc. if configured)

        For CNCeleb, enrollment embeddings are keyed by enroll_id (aggregated from
        multiple utterances). For VoxCeleb, they are keyed by enroll_path.
        """
        embeds = test_data["test_embeds"]
        enrol_embeds = test_data["enrol_embeds"]
        scoring_pipeline: ScoringPipeline = test_data["scoring_pipeline"]

        trial_embeddings = torch.stack(
            [embeds[path] for path in batch.test_path]
        )

        # Determine whether to look up by enroll_id or enroll_path
        # CNCeleb uses enroll_id (aggregated multi-utterance), VoxCeleb uses enroll_path
        if hasattr(batch, "enroll_id") and batch.enroll_id[0] is not None:
            sample_key = batch.enroll_id[0]
            if sample_key in enrol_embeds:
                enroll_embeddings = torch.stack(
                    [enrol_embeds[eid] for eid in batch.enroll_id]
                )
            else:
                # Fallback to enroll_path for backward compatibility
                enroll_embeddings = torch.stack(
                    [enrol_embeds[path] for path in batch.enroll_path]
                )
        else:
            enroll_embeddings = torch.stack(
                [enrol_embeds[path] for path in batch.enroll_path]
            )

        # Use scoring pipeline for complete scoring (centering + raw score + normalization)
        normalized_scores, raw_scores = scoring_pipeline.score_batch(
            enroll_embeddings, trial_embeddings
        )

        batch_dict = {
            "enroll_path": batch.enroll_path,
            "test_path": batch.test_path,
            "trial_label": batch.trial_label,
            "same_country_label": batch.same_country_label,
            "same_gender_label": batch.same_gender_label,
            "score": raw_scores.detach().cpu().tolist(),
            "norm_score": normalized_scores.detach().cpu().tolist(),
        }
        test_data["trial_results"].append(batch_dict)

    def _epoch_end_common_multi_test(self, test_filename: str) -> str:
        """Handle epoch end for a specific test set."""
        t0 = time.perf_counter()
        # Sanitize test_filename for use in file paths (replace all path separators)
        safe_test_filename = test_filename.replace("/", "_").replace("\\", "_")

        test_data = self.test_sets_data[test_filename]
        enrol_embeds = test_data["enrol_embeds"]
        trials_embeds = test_data["test_embeds"]
        trial_results = test_data["trial_results"]

        log.info(
            f"Finalizing '{test_filename}': {len(trial_results)} batches accumulated"
        )
        t_build0 = time.perf_counter()

        # Build scores DataFrame
        scores = pd.DataFrame(
            [
                {
                    "enroll_path": enroll_path,
                    "test_path": test_path,
                    "trial_label": trial_label,
                    "same_country_label": same_country_label,
                    "same_gender_label": same_gender_label,
                    "score": score,
                    "norm_score": norm_score,
                }
                for batch in trial_results
                for enroll_path, test_path, trial_label, same_country_label, same_gender_label, score, norm_score in zip(
                    batch["enroll_path"],
                    batch["test_path"],
                    batch["trial_label"],
                    batch["same_country_label"],
                    batch["same_gender_label"],
                    batch["score"],
                    batch["norm_score"],
                )
            ]
        )
        log.info(
            f"Finalizing '{test_filename}': built scores DataFrame in {time.perf_counter() - t_build0:.1f}s (rows={len(scores)})"
        )

        # Create a temporary metric instance for this specific test set
        temp_metric = instantiate(self.hparams.metrics.test)

        # Update the temporary metric with this test set's data
        all_norm_scores = []
        all_labels = []
        for batch in trial_results:
            all_norm_scores.extend(batch["norm_score"])
            all_labels.extend(batch["trial_label"])
        t_metric0 = time.perf_counter()
        temp_metric.update(
            scores=torch.tensor(all_norm_scores, dtype=torch.float32),
            labels=torch.tensor(all_labels, dtype=torch.long),
        )
        log.info(
            f"Finalizing '{test_filename}': metric.update in {time.perf_counter() - t_metric0:.1f}s"
        )

        # Compute metrics for this specific test set
        t_compute0 = time.perf_counter()
        metrics = temp_metric.compute()
        log.info(
            f"Finalizing '{test_filename}': metric.compute in {time.perf_counter() - t_compute0:.1f}s"
        )

        # Log metrics with a clear prefix for each test set
        temp_metric_class_name = temp_metric.__class__.__name__
        prefixed_metrics = {
            f"test/{test_filename}/{temp_metric_class_name}/{key}": value
            for key, value in metrics.items()
        }
        # Add batch_size to avoid PyTorch Lightning warning about ambiguous batch size inference
        batch_size = (
            len(trial_results[0]["trial_label"]) if trial_results else 1
        )
        self.log_dict(
            prefixed_metrics, batch_size=batch_size, **self.logging_params
        )

        # Update scores DataFrame with computed metrics (optional; metrics are always saved in JSON)
        scores.loc[:, metrics.keys()] = [
            v.item() if torch.is_tensor(v) else v for v in metrics.values()
        ]

        # Set up directory for saving test artifacts
        # Layout: test_artifacts/<test_set>/<timestamp>/...
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = self._get_test_artifacts_dir(test_filename)
        artifacts_dir = base_dir / run_timestamp
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Save scores as CSV
        t_csv0 = time.perf_counter()
        scores.to_csv(
            artifacts_dir / f"{safe_test_filename}_scores.csv", index=False
        )
        log.info(
            f"Finalizing '{test_filename}': wrote CSV in {time.perf_counter() - t_csv0:.1f}s"
        )

        # Save embeddings
        t_save0 = time.perf_counter()
        torch.save(
            enrol_embeds,
            artifacts_dir / f"{safe_test_filename}_enrol_embeds.pt",
        )
        torch.save(
            trials_embeds, artifacts_dir / f"{safe_test_filename}_embeds.pt"
        )
        log.info(
            f"Finalizing '{test_filename}': saved embeddings in {time.perf_counter() - t_save0:.1f}s"
        )

        # Save test metrics as a JSON file
        metrics_for_save = {
            k: v.item() if torch.is_tensor(v) else v
            for k, v in metrics.items()
        }
        metrics_for_save["test_set"] = test_filename  # Add test set identifier
        with open(
            artifacts_dir / f"{safe_test_filename}_metrics.json", "w"
        ) as f:
            json.dump(metrics_for_save, f, indent=4)

        log.info(
            f"Finalizing '{test_filename}': total finalize time {time.perf_counter() - t0:.1f}s"
        )
        return run_timestamp
