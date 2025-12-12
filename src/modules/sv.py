import json
import time
from typing import Any, Dict, Optional
from pathlib import Path
from datetime import datetime

import random
import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm
import pandas as pd
from omegaconf import OmegaConf

from src.datamodules.components.voxceleb.voxceleb_dataset import (
    VoxcelebItem,
    VoxCelebVerificationItem,
    TrainCollate)
from src import utils
from src.modules.encoder_wrappers import EncoderWrapper
from src.callbacks.pruning.utils.pruning_manager import PruningManager
from src.modules.metrics.metrics import AS_norm 


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
        self.data_augemntation = kwargs.get("data_augemntation", None)
        
        # Initialize metrics
        self._setup_metrics(metrics)
        
        # Initialize model components
        self._setup_model_components(model)
        
        # Setup training components
        self._setup_training_components(criterion, optimizer, lr_scheduler)
        
        # Freeze pretrained components (Ignore for now)

        # Initialize text embedding cache with appropriate limits
        self._embeds_cache_config = model.get("embedding_cache", {})
        self._max_cache_size = self._embeds_cache_config.get("max_size", 500000)
        self._bypass_warmup = self._embeds_cache_config.get("bypass_warmup", False)
        
        # Embeddings norm configs
        self.normalize_test_scores = kwargs.get("normalize_test_scores", False)
        self.scores_norm = kwargs.get("scores_norm",
                                      OmegaConf.create({"embeds_metric_params": {}, "scores_norm_params": {}}))

        # Test-time artifact controls
        # - Repeating dataset-level metrics in every CSV row is expensive/large; metrics are always saved to JSON.
        self.scores_csv_include_metrics: bool = bool(kwargs.get("scores_csv_include_metrics", True))

    ############ Setup init ############
    def _setup_metrics(self, metrics: DictConfig) -> None:
        """Initialize all metrics for training, validation and testing."""
        self.train_metric = instantiate(metrics.train)
        self.valid_metric = instantiate(metrics.valid)
        self.test_metric = instantiate(metrics.test)
        self.valid_metric_best = instantiate(metrics.valid_best)

    def _setup_model_components(self, model: DictConfig) -> None:
        """Initialize encoders and classifiers, wrapping the encoder for a unified interface."""
        # Audio processing
        self.audio_processor = instantiate(model.audio_processor)
        self.audio_processor_normalizer = instantiate(model.audio_processor_normalizer)
        
        # Instantiate the raw encoder
        raw_audio_encoder = instantiate(model.audio_encoder)
        
        # Wrap the encoder and its pre-processing steps
        self.audio_encoder = EncoderWrapper(
            encoder=raw_audio_encoder,
            audio_processor=self.audio_processor,
            audio_processor_normalizer=self.audio_processor_normalizer
        )
        
        self.classifier = instantiate(model.classifier)

        # Setup wav augmentation if configured
        if self.data_augemntation is not None:
            assert "wav_augmenter" in self.data_augemntation.augmentations, 'Expected augmentations.wav_augmenter when passing data_augemntation'
            self.wav_augmenter = instantiate(self.data_augemntation.augmentations.wav_augmenter)

    def _setup_training_components(self, criterion: DictConfig, optimizer: DictConfig, lr_scheduler: DictConfig) -> None:
        """Initialize loss functions, optimizer and learning rate scheduler."""
        self.train_criterion = instantiate(criterion.train_criterion)
        self.valid_criterion = instantiate(criterion.valid_criterion)     
        self.optimizer = optimizer
        self.slr_params = lr_scheduler

    def _freeze_pretrained_components(self) -> None:
        """Freeze pretrained components and enable training for others."""
        # Access the original encoder through the wrapper
        original_encoder = self.audio_encoder.encoder

        # if hasattr(self.audio_encoder, "encode_batch"):  Define what to freeze -- IGNORE ---  
        for param in original_encoder.parameters():
            param.requires_grad = False

    def _log_step_metrics(self, results: Dict[str, Any], batch: VoxcelebItem, stage: str) -> None:
        criterion = getattr(self, f"{stage}_criterion")
        
        # Log losses
        logged_dict = {f"{stage}/{criterion.__class__.__name__}": results['loss'].item()}

        self.log_dict(
            logged_dict,
            batch_size=batch.audio.shape[0],
            **self.logging_params
        )

        # Log metrics
        metric = getattr(self, f"{stage}_metric")
        computed_metric = metric(results["outputs"][f"logits"], batch.class_id)
        
        self.log(
            f"{stage}/{metric.__class__.__name__}",
            computed_metric,
            batch_size=batch.audio.shape[0],
            **self.logging_params
        )

    ############ Caching ############
    def _warmup_cache(self, fraction: float = 0.2):
        """Pre-computes and caches embeddings for unique training samples.
        Uses batched processing for memory efficiency and shows a progress bar.
        """
        # Get unique audios from training data
        unique_audios = list(set(self.trainer.datamodule.train_data.dataset.audio))
        unique_audios_lens = list(set(self.trainer.datamodule.train_data.dataset.audio_lens))
        
        # Define batch size for processing
        batch_size = 384
        
        # Optional: Limit to subset of texts for faster startup
        max_samples = min(len(unique_audios), int(self._max_cache_size * fraction))
        if max_samples < len(unique_audios):
            log.info(f"Limiting cache warmup to {max_samples} samples out of {len(unique_audios)} total")
            # Randomly sample to ensure diversity
            indices = random.sample(range(len(unique_audios)), max_samples)
            unique_audios = [unique_audios[i] for i in indices]
            unique_audios_lens = [unique_audios_lens[i] for i in indices]
        
        # Process texts in batches with progress bar
        with torch.no_grad():
            with tqdm(total=len(unique_audios), desc="Warming up cache", leave=False) as pbar:
                for i in range(0, len(unique_audios), batch_size):
                    batch_audios = unique_audios[i: i + batch_size]
                    batch_audios_lens = unique_audios_lens[i: i + batch_size] 
                    # Get embeddings and immediately delete the tensor since we only need the cached values
                    _ = self.get_audio_embeddings(batch_audios, batch_audios_lens)
                    pbar.update(len(batch_audios))

    ############ Lightning ############
    def _get_audio_embeddings(self, batch_audio: torch.Tensor, batch_audio_lens: torch.Tensor) -> torch.Tensor:
        """
        Computes audio embeddings using the wrapped encoder, which handles the full pipeline.
        """
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
            batch.audio, audio_length_norm = self.wav_augmenter(batch.audio, batch.audio_length / max_audio_length)
            batch.audio_length = audio_length_norm * max_audio_length
            batch.class_id = self.wav_augmenter.replicate_labels(batch.class_id)

        audio_emb = self._get_audio_embeddings(batch.audio, batch.audio_length)
        logits = self.classifier(audio_emb)
            
        return {"embeds": audio_emb, 'logits': logits}

    def model_step(self, batch: VoxcelebItem, criterion: Optional[Any] = None) -> Dict[str, Any]:
        """Perform a single model step."""
        outputs = self(batch)

        # Compute loss
        if isinstance(criterion, torch.nn.CrossEntropyLoss):
            main_loss = criterion(outputs[f"logits"], batch.class_id)
        elif criterion.__class__.__name__ == 'LogSoftmaxWrapper':
            if outputs[f"logits"].ndim == 3 and outputs[f"logits"].shape[1] == 1:
                outputs[f"logits"] = outputs[f"logits"].squeeze(1)
            main_loss = criterion(outputs[f"logits"].unsqueeze(1), batch.class_id.unsqueeze(1))
        else:
            raise ValueError("Invalid criterion")
        
        return {"loss": main_loss, "outputs": outputs}

    def on_train_start(self) -> None:
        # by default lightning executes validation step sanity checks before
        # training starts, so we need to make sure valid_metric_best doesn't store
        # accuracy from these checks
        self.valid_metric_best.reset()
        self.audio_encoder.train()
        if bool(len(self._embeds_cache_config)) and (self.global_step == 0) and (not self._bypass_warmup):
            self._warmup_cache()

    def training_step(self, batch: VoxcelebItem, batch_idx: int) -> Dict[str, torch.Tensor]:
        results = self.model_step(batch, self.train_criterion)
        
        if batch_idx % 3000 == 0:
            torch.cuda.empty_cache()

        self._log_step_metrics(results, batch, 'train')

        return results

    def on_train_epoch_end(self) -> None:
        self.train_metric.reset()

        # Log sparsity information if pruning is being used
        self._log_sparsity_info_if_pruning()

    def _log_sparsity_info_if_pruning(self) -> None:
        """Log sparsity information if model is being pruned."""
        # Check if any parameters have pruning masks (indicating pruning is active)
        has_pruning_masks = any(
            hasattr(module, f"{param_name}_mask")
            for module in self.modules()
            for param_name, _ in module.named_parameters(recurse=False)
        )
        
        if has_pruning_masks:
            sparsity_info = self.get_model_sparsity_info()
            
            # Log basic sparsity metrics
            self.log("pruning/overall_sparsity", sparsity_info['overall_sparsity'])
            self.log("pruning/total_parameters", sparsity_info['total_parameters'])
            self.log("pruning/pruned_parameters", sparsity_info['pruned_parameters'])
            
            # Log detailed info periodically
            if self.current_epoch % 10 == 0:  # Every 10 epochs
                log.info(f"Epoch {self.current_epoch} - Model Sparsity: {sparsity_info['overall_sparsity']:.4f} "
                        f"({sparsity_info['pruned_parameters']}/{sparsity_info['total_parameters']} parameters)")

    def get_model_sparsity_info(self) -> Dict[str, Any]:
        """Get detailed sparsity information for debugging pruning callbacks.
        
        Returns:
            Dictionary with sparsity statistics
        """
        total_params = 0
        pruned_params = 0
        masked_modules = 0
        
        sparsity_info = {
            'modules_with_masks': [],
            'modules_without_masks': [],
            'total_parameters': 0,
            'pruned_parameters': 0,
            'overall_sparsity': 0.0,
            'epoch': getattr(self, 'current_epoch', -1)  # Include current epoch for context
        }
        
        for name, module in self.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if not isinstance(param, torch.Tensor):
                    continue
                    
                mask_name = f"{param_name}_mask"
                param_count = param.numel()
                total_params += param_count
                
                if hasattr(module, mask_name):
                    mask = getattr(module, mask_name)
                    pruned_count = param_count - mask.sum().item()
                    pruned_params += pruned_count
                    masked_modules += 1
                    
                    sparsity_info['modules_with_masks'].append({
                        'module': f"{name}.{param_name}",
                        'total': param_count,
                        'pruned': pruned_count,
                        'sparsity': float(pruned_count) / param_count
                    })
                else:
                    sparsity_info['modules_without_masks'].append({
                        'module': f"{name}.{param_name}",
                        'total': param_count
                    })
        
        sparsity_info['total_parameters'] = total_params
        sparsity_info['pruned_parameters'] = pruned_params
        sparsity_info['overall_sparsity'] = float(pruned_params) / max(1, total_params)
        
        return sparsity_info

    def inspect_pruning_state(self, detailed: bool = False) -> None:
        """Manual method to inspect current pruning state - useful for debugging.
        
        Args:
            detailed: If True, shows detailed per-module information
        """
        sparsity_info = self.get_model_sparsity_info()
        
        print(f"\n=== PRUNING STATE INSPECTION (Epoch {sparsity_info['epoch']}) ===")
        print(f"Overall Sparsity: {sparsity_info['overall_sparsity']:.4f}")
        print(f"Total Parameters: {sparsity_info['total_parameters']:,}")
        print(f"Pruned Parameters: {sparsity_info['pruned_parameters']:,}")
        print(f"Modules with Masks: {len(sparsity_info['modules_with_masks'])}")
        print(f"Modules without Masks: {len(sparsity_info['modules_without_masks'])}")
        
        if detailed and sparsity_info['modules_with_masks']:
            print("\nDetailed Module Sparsity:")
            sorted_modules = sorted(sparsity_info['modules_with_masks'], 
                                  key=lambda x: x['sparsity'], reverse=True)
            for module_info in sorted_modules:
                print(f"  {module_info['module']}: {module_info['sparsity']:.4f} "
                      f"({module_info['pruned']}/{module_info['total']})")
        
        print("=== END INSPECTION ===\n")

    def on_validation_start(self) -> None:
        pass

    def on_validation_epoch_end(self) -> None:
        self.valid_metric.reset()

    @torch.inference_mode()
    def validation_step(self, batch: VoxcelebItem, batch_idx: int) -> Dict[str, torch.Tensor]:
        results = self.model_step(batch, self.valid_criterion)
        self._log_step_metrics(results, batch, 'valid')
        return results

    @torch.inference_mode()
    def on_test_start(self) -> None:
        """Pre-computes embeddings and prepares for per-dataloader metric logging."""
        # Get all configured test dataloaders from the datamodule
        test_dataloaders = self.trainer.datamodule.test_dataloader()
        
        # Normalize to dictionary format for consistent handling
        if not isinstance(test_dataloaders, dict):
            # Infer a descriptive base name from the datamodule class
            datamodule_class = self.trainer.datamodule.__class__.__name__
            base_name = datamodule_class.replace('DataModule', '').replace('Module', '').lower()
            
            if isinstance(test_dataloaders, (list, tuple)):
                # Multiple loaders returned as list/tuple - use base_name with index
                test_dataloaders = {f'{base_name}_{i}': loader for i, loader in enumerate(test_dataloaders)}
                log.info(f"Normalized {len(test_dataloaders)} test loaders to dictionary format with base name '{base_name}'")
            else:
                # Single loader returned - use just the base_name
                test_dataloaders = {base_name: test_dataloaders}
                log.info(f"Normalized single test loader to dictionary format with key '{base_name}'")
        
        # Store normalized dict for use in test_step
        self.test_dataloaders_dict = test_dataloaders
        test_filenames = list(test_dataloaders.keys())
        
        log.info(f"Found {len(test_filenames)} test set(s): {', '.join(test_filenames)}")
        
        # Compute cohort embeddings per dataset (if needed for AS-Norm)
        cohort_embeddings_by_dataset: Dict[str, torch.Tensor] = {}
        train_dm = self.trainer.datamodule

        self.test_sets_data = {}
        self.last_batch_indices = {}

        for test_filename, dataloader in test_dataloaders.items():
            # Skip already completed test sets
            if self._is_test_set_complete(test_filename):
                log.info(f"Skipping '{test_filename}' - already complete")
                self.last_batch_indices[test_filename] = -2  # Mark as skip
                continue

            log.info(f"Processing '{test_filename}'...")
            base_dataset = test_filename.split('/')[0]

            cohort_embeddings = self._get_cohort_embeddings_for_dataset(
                base_dataset=base_dataset,
                train_dm=train_dm,
                cache=cohort_embeddings_by_dataset,
            ) if self.normalize_test_scores else None
            
            # Determine the index of the last batch for this dataloader
            num_batches = len(dataloader)
            if num_batches > 0:
                self.last_batch_indices[test_filename] = num_batches - 1
                log.info(f"Registered '{test_filename}' with {num_batches} batches. Last batch index: {num_batches - 1}.")
            else:
                raise ValueError("Dataloader has zero batches")

            # Try to load cached embeddings, otherwise compute them
            cached = self._load_cached_embeddings(test_filename)
            if cached:
                enrol_embeds, test_embeds = cached['enrol_embeds'], cached['test_embeds']
            else:
                enroll_dataloader, trial_unique_dataloader = self.trainer.datamodule.get_enroll_and_trial_dataloaders(test_filename)
                enrol_embeds = self._compute_embeddings(enroll_dataloader, mode='enrollment')
                test_embeds = self._compute_embeddings(trial_unique_dataloader, mode='test')
                self._save_embeddings_cache(test_filename, enrol_embeds, test_embeds)

            # Load partial trial results if resuming
            trial_results, resume_batch_idx, resume_batch_paths = self._load_partial_trial_results(test_filename)
            
            # Store all data for this test set
            self.test_sets_data[test_filename] = {
                'enrol_embeds': enrol_embeds,
                'test_embeds': test_embeds,
                'cohort_embeddings': cohort_embeddings,
                'trial_results': trial_results,
                'resume_batch_idx': resume_batch_idx,
                'resume_batch_paths': resume_batch_paths
            }
            
        log.info(f"Finished computing embeddings for all test sets")

    @torch.inference_mode()
    def test_step(self, batch: VoxCelebVerificationItem, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Handle test step and trigger logging on the last batch of each dataloader."""
        # Get test set name from dataloader index
        test_filenames = list(self.test_dataloaders_dict.keys())
        test_filename = test_filenames[dataloader_idx]
        
        # Skip if this test set was already completed or marked to skip
        if self.last_batch_indices.get(test_filename, -1) == -2:
            return
        
        # Skip batches already processed (resumption) with path check on the boundary batch
        test_data = self.test_sets_data[test_filename]
        resume_idx = test_data.get('resume_batch_idx', -1)
        resume_paths = test_data.get('resume_batch_paths')
        if batch_idx < resume_idx:
            return
        if batch_idx == resume_idx and resume_paths is not None:
            if list(batch.enroll_path) == resume_paths.get('enroll_path') and list(batch.test_path) == resume_paths.get('test_path'):
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
                test_data['trial_results'],
                batch_idx,
                {
                    'enroll_path': list(batch.enroll_path),
                    'test_path': list(batch.test_path)
                }
            )
            log.info(f"Checkpoint saved for '{test_filename}' at batch {batch_idx}")

        # Check if this is the last batch for the current dataloader
        is_last_batch = (batch_idx == self.last_batch_indices.get(test_filename, -1))

        if is_last_batch:
            log.info(f"Last batch for '{test_filename}' (idx: {batch_idx}) reached. Finalizing and logging metrics.")
            run_timestamp = self._epoch_end_common_multi_test(test_filename)
            self._mark_test_complete(test_filename, run_timestamp=run_timestamp)

    def on_test_epoch_end(self) -> None:
        """Callback at the end of the test epoch.
        
        This is now primarily a cleanup step. It also handles any test sets
        where the number of batches could not be determined.
        """
        # Process any test sets that were not handled in test_step (fallback)
        if hasattr(self, 'last_batch_indices'):
            for test_filename, last_idx in self.last_batch_indices.items():
                if last_idx == -2:  # Skipped (already complete)
                    continue
                if last_idx == -1:
                    log.warning(f"Running fallback metric computation for '{test_filename}' at epoch end.")
                    run_timestamp = self._epoch_end_common_multi_test(test_filename)
                    self._mark_test_complete(test_filename, run_timestamp=run_timestamp)

        log.info("Test epoch finished. All test sets have been processed.")
        # Clean up stored data to free memory
        if hasattr(self, 'test_sets_data'):
            del self.test_sets_data
        if hasattr(self, 'last_batch_indices'):
            del self.last_batch_indices
        if hasattr(self, 'test_dataloaders_dict'):
            del self.test_dataloaders_dict

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configures optimizers and learning rate schedulers.

        This method dynamically adapts its behavior based on the configured optimizer:
        
        1.  If a Bregman-type optimizer (AdaBreg, LinBreg, ProxSGD) is used,
            it initializes the PruningManager to handle parameter groups with
            specific regularization settings defined in the config.

        2.  For any standard optimizer (e.g., Adam, SGD), it applies the
            optimizer to all model parameters uniformly.
        """
        BREGMAN_OPTIMIZERS = {"AdaBreg", "LinBreg", "ProxSGD"}
        optimizer_class_name = self.hparams.optimizer._target_.split('.')[-1]

        # Use the two-step partial instantiation pattern for the optimizer
        optimizer_partial = instantiate(self.hparams.optimizer)

        if optimizer_class_name in BREGMAN_OPTIMIZERS:
            # --- Pruning-Aware Optimizer Logic ---
            self.pruning_manager = PruningManager(
                pl_module=self,
                group_configs=self.hparams.model.pruning_groups
            )
            optimizer_param_groups = self.pruning_manager.get_optimizer_param_groups()
            
            # Manually instantiate the regularization object for each group
            for group in optimizer_param_groups:
                if 'reg' in group and isinstance(group.get('reg'), (dict, DictConfig)):
                    group['reg'] = instantiate(group['reg'])
            
            optimizer = optimizer_partial(params=optimizer_param_groups)

        else:
            # --- Standard Optimizer Logic ---
            optimizer = optimizer_partial(params=self.parameters())

        # --- Common Scheduler Logic ---
        if self.hparams.get("lr_scheduler"):
            # Instantiate the scheduler, which now receives a fully formed optimizer
            scheduler = instantiate(
                self.hparams.lr_scheduler.scheduler,
                optimizer=optimizer
            )
            
            lr_scheduler_dict = {"scheduler": scheduler}
            if self.hparams.lr_scheduler.get("extras"):
                for key, value in self.hparams.lr_scheduler.get("extras").items():
                    lr_scheduler_dict[key] = value
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
        
        return {"optimizer": optimizer}

    ############ Dev and eval utils ############
    def _get_test_artifacts_dir(self, test_filename: str) -> Path:
        """Get the base artifacts directory for a test set."""
        safe_name = test_filename.replace('/', '_').replace('\\', '_')
        return Path(self.trainer.default_root_dir) / "test_artifacts" / safe_name

    def _get_test_cache_dir(self, test_filename: str) -> Path:
        """Get the cache directory for a test set (embeds/checkpoints)."""
        return self._get_test_artifacts_dir(test_filename) / "cache"

    def _get_cohort_cache_path(self, base_dataset: str) -> Path:
        """Get the per-dataset cohort embeddings cache path.

        Stored once per experiment to avoid mixing datasets.
        """
        cache_root = Path(self.trainer.default_root_dir) / "test_artifacts" / "_cohort_cache"
        return cache_root / f"{base_dataset}_test_cohort_embeds_cache.pt"

    def _save_embeddings_cache(self, test_filename: str, enrol_embeds: dict, test_embeds: dict) -> None:
        """Save embeddings to cache for potential resumption."""
        cache_dir = self._get_test_cache_dir(test_filename)
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(enrol_embeds, cache_dir / "enrol_embeds_cache.pt")
        torch.save(test_embeds, cache_dir / "test_embeds_cache.pt")

    def _load_cached_embeddings(self, test_filename: str) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """Load cached embeddings for a test set if they exist."""
        cache_dir = self._get_test_cache_dir(test_filename)

        enrol_path = cache_dir / "enrol_embeds_cache.pt"
        test_path = cache_dir / "test_embeds_cache.pt"
        
        if enrol_path.exists() and test_path.exists():
            log.info(f"Loading cached enrollment and test embeddings for '{test_filename}'")
            return {
                'enrol_embeds': torch.load(enrol_path),
                'test_embeds': torch.load(test_path)
            }
        return None

    def _save_trial_results_checkpoint(self, test_filename: str, trial_results: list, batch_idx: int, batch_paths: dict) -> None:
        """Save trial results checkpoint for resumption."""
        cache_dir = self._get_test_cache_dir(test_filename)
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {'trial_results': trial_results, 'last_batch_idx': batch_idx, 'last_batch_paths': batch_paths},
            cache_dir / "trial_results_checkpoint.pt"
        )

    def _load_partial_trial_results(self, test_filename: str) -> tuple[list, int, Optional[dict]]:
        """Load partial trial results if they exist. Returns (results, last_batch_idx, last_batch_paths)."""
        cache_dir = self._get_test_cache_dir(test_filename)
        checkpoint_path = cache_dir / "trial_results_checkpoint.pt"
        
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            log.info(f"Resuming '{test_filename}' from batch {checkpoint['last_batch_idx'] + 1}")
            return checkpoint['trial_results'], checkpoint['last_batch_idx'], checkpoint.get('last_batch_paths')
        return [], -1, None

    def _mark_test_complete(self, test_filename: str, run_timestamp: Optional[str] = None) -> None:
        """Mark a test set as complete.

        Writes a COMPLETE marker and also records the last successful run timestamp
        (so "COMPLETE" implies there is a corresponding results directory).
        """
        base_dir = self._get_test_artifacts_dir(test_filename)
        base_dir.mkdir(parents=True, exist_ok=True)
        (base_dir / "COMPLETE").touch()
        if run_timestamp is not None:
            (base_dir / "LAST_RUN").write_text(str(run_timestamp))
        checkpoint_path = self._get_test_cache_dir(test_filename) / "trial_results_checkpoint.pt"
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    def _is_test_set_complete(self, test_filename: str) -> bool:
        """Check if a test set has already been fully evaluated.

        "COMPLETE" alone is treated as insufficient if there is no corresponding
        timestamped results directory.
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
            p.is_dir() and p.name[:9].isdigit() and '_' in p.name
            for p in base_dir.iterdir()
        )
        if not has_run_dir:
            log.warning(
                f"Found COMPLETE for '{test_filename}' but no timestamped results directory under {base_dir}. "
                "Treating as incomplete to regenerate results."
            )
            return False
        return True

    def _compute_embeddings(self, dataloader, mode: str = 'test') -> dict:
        embeddings_dict = {}
        desc = f"Computing {mode} embeddings"

        with tqdm(dataloader, desc=desc, leave=False) as pbar:
            for batch in pbar:
                outputs = self(batch)
                embed = outputs['embeds']
                # Handle frame-level embeddings (if applicable)
                if len(embed.shape) == 3:  # [B, num_frames, embedding_dim]
                    embed = embed.mean(dim=1)  # [B, embedding_dim]

                embeddings_dict.update({path: emb for path, emb in zip(batch.audio_path, embed)})

        return embeddings_dict

    def _compute_cohort_embeddings(self, dataloader) -> torch.Tensor:
        # Compute only. Loading/saving is handled by _get_cohort_embeddings_for_dataset.
        embeddings_list = []
        with tqdm(dataloader, desc="Computing cohort embeddings", leave=False) as pbar:
            for batch in pbar:
                outputs = self(batch)
                cohort = outputs['embeds']

                # Handle frame-level embeddings (if applicable)
                if len(cohort.shape) == 3:  # [B, num_frames, embedding_dim]
                    cohort = cohort.mean(dim=1)  # [B, embedding_dim]

                embeddings_list.append(cohort)
        
        cohort_embeds = torch.cat(embeddings_list, dim=0)
        return cohort_embeds

    def _get_cohort_embeddings_for_dataset(
        self,
        base_dataset: str,
        train_dm,
        cache: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Return (and cache) cohort embeddings for the given dataset name."""
        if base_dataset in cache:
            return cache[base_dataset]

        cohort_cache_path = self._get_cohort_cache_path(base_dataset)
        if cohort_cache_path.exists():
            log.info(f"Loading cohort embeddings for '{base_dataset}' from: {cohort_cache_path}")
            cache[base_dataset] = torch.load(cohort_cache_path)
            return cache[base_dataset]
        
        collate_fn = None

        # Ensure the datamodule exposes cohort datasets even if training is disabled
        if hasattr(train_dm, "ensure_cohort_datasets_for_normalization"):
            train_dm.ensure_cohort_datasets_for_normalization()

        # If the datamodule can set up fit stage lazily, do it to populate train_data
        if (not hasattr(train_dm, "train_data")) or (getattr(train_dm, "train_data", None) is None):
            setup_fn = getattr(train_dm, "setup", None)
            if callable(setup_fn):
                setup_fn(stage="fit")

        train_dataset = None
        if hasattr(train_dm, "get_train_dataset"):
            train_dataset = train_dm.get_train_dataset(base_dataset)
        elif hasattr(train_dm, "_train_sets"):
            train_dataset = next((ds for name, ds in getattr(train_dm, "_train_sets") if name == base_dataset), None)
        elif hasattr(train_dm, "train_data"):
            train_dataset = getattr(train_dm, "train_data")
        # Fall back to pulling dataset from the train dataloader if available
        if train_dataset is None and hasattr(train_dm, "train_dataloader"):
            try:
                loader = train_dm.train_dataloader()
                train_dataset = getattr(loader, "dataset", None)
                if collate_fn is None and hasattr(loader, "collate_fn"):
                    collate_fn = loader.collate_fn
            except Exception:
                pass

        assert train_dataset is not None, (
            f"Score normalization requires training data for '{base_dataset}', "
            "but none was found. Disable normalize_test_scores or enable training for this dataset."
        )

        log.info(f"Computing cohort embeddings for '{base_dataset}'...")
        if hasattr(train_dm, "get_train_collate"):
            collate_fn = train_dm.get_train_collate(base_dataset)
        cohort_loader = DataLoader(
            train_dataset,
            batch_size=getattr(train_dm.hparams.loaders.train, 'batch_size', 64),
            num_workers=getattr(train_dm.hparams.loaders.train, 'num_workers', 0),
            shuffle=False,
            pin_memory=getattr(train_dm.hparams.loaders.train, 'pin_memory', False),
            collate_fn=collate_fn or getattr(train_dm, '_train_collate', None) or TrainCollate(),
        )
        cache[base_dataset] = self._compute_cohort_embeddings(cohort_loader)
        cohort_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache[base_dataset], cohort_cache_path)
        log.info(f"Saved cohort embeddings for '{base_dataset}' to: {cohort_cache_path}")
        log.info(
            f"Computed cohort embeddings for '{base_dataset}' of shape: "
            f"{cache[base_dataset].shape}"
        )
        return cache[base_dataset]

    def _move_batch_to_device(self, batch: VoxCelebVerificationItem) -> VoxCelebVerificationItem:
        """Move batch tensors to the model's device."""
        batch.enroll_audio = batch.enroll_audio.to(self.device)
        batch.test_audio = batch.test_audio.to(self.device)
        batch.enroll_length = batch.enroll_length.to(self.device)
        batch.test_length = batch.test_length.to(self.device)
        return batch
        
    def _trials_eval_step_multi_test(self, batch: VoxCelebVerificationItem, test_data: Dict):
        """Evaluation step for a specific test set."""
        embeds = test_data['test_embeds']
        enrol_embeds = test_data['enrol_embeds']
        cohort_embeddings = test_data['cohort_embeddings']
        
        trial_embeddings = torch.stack([embeds[path] for path in batch.test_path])
        enroll_embeddings = torch.stack([enrol_embeds[path] for path in batch.enroll_path])

        # Compute raw cosine similarity scores
        raw_scores = torch.nn.functional.cosine_similarity(enroll_embeddings, trial_embeddings)
        
        # Apply AS-Norm if cohort embeddings are available (same cohort for all test sets)
        if cohort_embeddings is not None:
            normalized_scores = []
            for i in range(trial_embeddings.shape[0]):
                norm_score = AS_norm(score=raw_scores[i],
                                     enroll_embedding=enroll_embeddings[i, ...],
                                     test_embedding=trial_embeddings[i, ...], 
                                     cohort_embeddings=cohort_embeddings,
                                     **self.scores_norm.scores_norm_params)
                normalized_scores.append(norm_score)
            normalized_scores = torch.tensor(normalized_scores, device=raw_scores.device)
        else:
            normalized_scores = raw_scores.clone()
        
        batch_dict = {
            "enroll_path": batch.enroll_path,
            "test_path": batch.test_path,
            "trial_label": batch.trial_label,
            "same_country_label": batch.same_country_label,
            "same_gender_label": batch.same_gender_label,
            "score": raw_scores.detach().cpu().tolist(),
            "norm_score": normalized_scores.detach().cpu().tolist(),
        }
        test_data['trial_results'].append(batch_dict)
        
    def _epoch_end_common_multi_test(self, test_filename: str) -> str:
        """Handle epoch end for a specific test set."""
        t0 = time.perf_counter()
        # Sanitize test_filename for use in file paths (replace all path separators)
        safe_test_filename = test_filename.replace('/', '_').replace('\\', '_')
        
        test_data = self.test_sets_data[test_filename]
        enrol_embeds = test_data['enrol_embeds']
        trials_embeds = test_data['test_embeds']
        trial_results = test_data['trial_results']

        log.info(f"Finalizing '{test_filename}': {len(trial_results)} batches accumulated")
        t_build0 = time.perf_counter()
        
        # Build scores DataFrame
        scores = pd.DataFrame([
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
        ])
        log.info(f"Finalizing '{test_filename}': built scores DataFrame in {time.perf_counter() - t_build0:.1f}s (rows={len(scores)})")

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
        log.info(f"Finalizing '{test_filename}': metric.update in {time.perf_counter() - t_metric0:.1f}s")
        
        # Compute metrics for this specific test set
        t_compute0 = time.perf_counter()
        metrics = temp_metric.compute()
        log.info(f"Finalizing '{test_filename}': metric.compute in {time.perf_counter() - t_compute0:.1f}s")
        
        # Log metrics with a clear prefix for each test set
        temp_metric_class_name = temp_metric.__class__.__name__
        prefixed_metrics = {f"test/{test_filename}/{temp_metric_class_name}/{key}": value for key, value in metrics.items()}
        # Add batch_size to avoid PyTorch Lightning warning about ambiguous batch size inference
        batch_size = len(trial_results[0]["trial_label"]) if trial_results else 1
        self.log_dict(prefixed_metrics, batch_size=batch_size, **self.logging_params)

        # Update scores DataFrame with computed metrics (optional; metrics are always saved in JSON)
        if self.scores_csv_include_metrics:
            scores.loc[:, metrics.keys()] = [v.item() if torch.is_tensor(v) else v for v in metrics.values()]
        
        # Set up directory for saving test artifacts
        # Layout: test_artifacts/<test_set>/<timestamp>/...
        run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_dir = self._get_test_artifacts_dir(test_filename)
        artifacts_dir = base_dir / run_timestamp
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Save scores as CSV
        t_csv0 = time.perf_counter()
        scores.to_csv(artifacts_dir / f"{safe_test_filename}_scores.csv", index=False)
        log.info(f"Finalizing '{test_filename}': wrote CSV in {time.perf_counter() - t_csv0:.1f}s")
        
        # Save embeddings
        t_save0 = time.perf_counter()
        torch.save(enrol_embeds, artifacts_dir / f"{safe_test_filename}_enrol_embeds.pt")
        torch.save(trials_embeds, artifacts_dir / f"{safe_test_filename}_embeds.pt")
        log.info(f"Finalizing '{test_filename}': saved embeddings in {time.perf_counter() - t_save0:.1f}s")

        # Save test metrics as a JSON file 
        metrics_for_save = {k: v.item() if torch.is_tensor(v) else v for k, v in metrics.items()}
        metrics_for_save['test_set'] = test_filename  # Add test set identifier
        with open(artifacts_dir / f"{safe_test_filename}_metrics.json", "w") as f:
            json.dump(metrics_for_save, f, indent=4)

        log.info(f"Finalizing '{test_filename}': total finalize time {time.perf_counter() - t0:.1f}s")
        return run_timestamp
