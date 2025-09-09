import os
import io
import json
from typing import Any, Dict, Optional, Tuple, Union
from collections import defaultdict
from pathlib import Path
import inspect

import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from src.datamodules.components.voxceleb.voxceleb_dataset import (
    VoxcelebItem,
    VoxCelebVerificationItem,
    TrainCollate)
from src import utils
from src.callbacks.pruning.utils.pruning_manager import PruningManager
from src.modules.metrics.metrics import AS_norm 
from datetime import datetime


log = utils.get_pylogger(__name__)


###################################

def load_pretrained_model(
    filename: str,
    repo_id: str,
    cache_dir: Optional[str] = None,
    map_location: str = 'cuda',
    *args,
    **kwargs
) -> nn.Module:
    """Hydra factory that returns a loaded TorchScript model directly.

    This mirrors the behavior of ``PretrainedModelLoader.__call__`` but allows
    Hydra's ``instantiate`` to yield the final ``nn.Module`` in a single step.

    Args:
        filename: Name of the file in the Hugging Face repo (e.g. 'ecapa2.pt').
        repo_id: Hugging Face repository ID (e.g. 'user/repo').
        cache_dir: Optional local cache directory.
        map_location: Device mapping for ``torch.jit.load``.
        *args, **kwargs: Ignored extra arguments for forward compatibility.

    Returns:
        Loaded ``nn.Module`` (TorchScript) ready for inference.
    """
    from huggingface_hub import hf_hub_download
    model_file = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
    model = torch.jit.load(model_file, map_location=map_location)
    return model


###################################
class EmbeddingMetrics:
    def __init__(
        self,
        trainer: pl.Trainer,
        stage: str,
        num_speakers_in_cohort: int = 6000,
        min_utts_per_speaker: int = 6,
        speaker_col: str = 'speaker_id'
    ):
        """Initialize the EmbeddingMetrics class.

        Args:
            trainer: PyTorch Lightning Trainer instance.
            stage: Stage of evaluation ('valid' or 'test').
            num_speakers_in_cohort: Number of speakers to include in the cohort (default: 6000).
            min_utts_per_speaker: Target number of utterances to sample per speaker (default: 6).
            speaker_col: Column name for speaker IDs in the dataset (default: 'speaker_id').
        """
        self.trainer = trainer
        self.stage = stage
        self.num_speakers_in_cohort = num_speakers_in_cohort
        self.min_utts_per_speaker = min_utts_per_speaker
        self.speaker_col = speaker_col

    @property
    def _trial_results(self):
        """Cached list of trial results."""
        if not hasattr(self, '_results'):
            self._results = []
        return self._results

    @property
    def _cohort_indices(self):
        """
        Compute balanced cohort indices ensuring good coverage across speakers.

        Returns:
            List of selected indices for the cohort.
        """
        if not hasattr(self, '_indices'):
            df = self.trainer.datamodule.train_data.dataset
            
            # Validate required column
            if self.speaker_col not in df.columns:
                raise ValueError(f"Missing required column: {self.speaker_col}")
            
            # Get unique speakers
            unique_speakers = df[self.speaker_col].unique()
            num_speakers_to_select = min(len(unique_speakers), self.num_speakers_in_cohort)
            
            # Randomly sample speakers
            selected_speakers = random.sample(list(unique_speakers), num_speakers_to_select)
            
            # Sample utterances from each selected speaker
            all_indices = []
            for speaker in selected_speakers:
                indices = df[df[self.speaker_col] == speaker].index.tolist()
                all_indices.extend(
                    random.sample(indices, min(self.min_utts_per_speaker, len(indices)))
                )
            
            self._indices = all_indices
        
        return self._indices

    def get_cohort_loader(self) -> DataLoader:
        """Get DataLoader for the cohort subset.

        Returns:
            DataLoader for the cohort data.
        """
        return DataLoader(
            dataset=Subset(self.trainer.datamodule.train_data, self._cohort_indices),
            batch_size=self.trainer.datamodule.loaders.train.batch_size,
            num_workers=self.trainer.datamodule.loaders.train.num_workers,
            pin_memory=self.trainer.datamodule.loaders.train.pin_memory,
            collate_fn=TrainCollate(),
            shuffle=False
        )

    @property
    def cohort_embeddings(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get the computed cohort embeddings.

        Returns:
            Dictionary of cohort embeddings or None if not set.
        """
        return self._embeddings

    def set_cohort_embeddings(self, embeddings_dict: Dict[str, torch.Tensor]) -> None:
        """Set the cohort embeddings and update metrics.

        Args:
            embeddings_dict: Dictionary mapping model IDs to their cohort embeddings.
        """
        self._embeddings = embeddings_dict
        
        # Set cohort embeddings in the metrics
        metrics = [
            self.trainer.model.valid_metric if self.stage == 'valid'
            else self.trainer.model.test_metric
        ]
        for metric in metrics:
            metric.cohort_embeddings = embeddings_dict

    def cleanup(self):
        """Delete cached attributes to free memory."""
        if hasattr(self, '_results'):
            delattr(self, '_results')
        if hasattr(self, '_indices'):
            delattr(self, '_indices')
        if hasattr(self, '_embeddings'):
            delattr(self, '_embeddings')

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
        
        # Freeze pretrained components
        self._freeze_pretrained_components()

        # Initialize text embedding cache with appropriate limits
        self._embeds_cache_config = model.get("embedding_cache", {})
        self._max_cache_size = self._embeds_cache_config.get("max_size", 500000)
        self._bypass_warmup = self._embeds_cache_config.get("bypass_warmup", False)
        
        # Embeddings norm configs
        self.normalize_test_scores = kwargs.get("normalize_test_scores", False)
        self.scores_norm = kwargs.get("scores_norm",
                                      OmegaConf.create({"embeds_metric_params": {}, "scores_norm_params": {}}))

    ############ Setup init ############
    def _setup_metrics(self, metrics: DictConfig) -> None:
        """Initialize all metrics for training, validation and testing."""
        self.train_metric = instantiate(metrics.train)
        self.valid_metric = instantiate(metrics.valid)
        self.test_metric = instantiate(metrics.test)
        self.valid_metric_best = instantiate(metrics.valid_best)

    def _setup_model_components(self, model: DictConfig) -> None:
        """Initialize encoders and classifiers."""
        # Audio processing
        self.audio_processor = instantiate(model.audio_processor)
        self.audio_processor_normalizer = instantiate(model.audio_processor_normalizer)
        self.audio_encoder = instantiate(model.audio_encoder)
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
        if hasattr(self.audio_encoder, "encode_batch"):
            for param in self.audio_encoder.parameters():
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
        if hasattr(self.audio_encoder, "encode_batch"):
            # For speechbrain encoders (e.g., x-vector)
            audio_emb = self.audio_encoder.encode_batch(
                wavs=batch_audio,
                wav_lens=batch_audio_lens/max(batch_audio_lens)
                ).squeeze(1)
        else:
            # TODO: strange mismatch during testing 
            if self.device != batch_audio.device:
                batch_audio = batch_audio.to(self.device)
            if self.device != batch_audio_lens.device:
                batch_audio_lens = batch_audio_lens.to(self.device)

            input_values = self.audio_processor(batch_audio)
            if not isinstance(self.audio_processor_normalizer, nn.Identity) and self.audio_processor_normalizer is not None:
                input_values = self.audio_processor_normalizer(input_values, lengths=batch_audio_lens/max(batch_audio_lens))
            
            # Check if the encoder's forward method accepts a 'lengths' argument
            if hasattr(self.audio_encoder, 'code'):
                # 1. Check for torch.jit models
                with torch.jit.optimized_execution(False):
                    audio_emb = self.audio_encoder(input_values).squeeze(1)
            else:
                # 2. For regular PyTorch models, inspect the signature
                sig = inspect.signature(self.audio_encoder.forward)
                if 'lengths' in sig.parameters:
                    audio_emb = self.audio_encoder(input_values, lengths=batch_audio_lens/max(batch_audio_lens)).squeeze(1)
                else:
                    audio_emb = self.audio_encoder(input_values).squeeze(1)

        return audio_emb

    def forward(self, batch: VoxcelebItem) -> Dict[str, torch.Tensor]:
        """Process audio inputs with optimized embedding caching."""
        # Add waveform augmentation if specified.
        if self.training and hasattr(self, "wav_augmenter"):
            batch.audio, batch.audio_length = self.wav_augmenter(batch.audio, batch.audio_length / max(batch.audio_length))
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
        test_filenames = list(test_dataloaders.keys())
        
        log.info(f"Found {len(test_filenames)} test set(s): {', '.join(test_filenames)}")
        
        # Compute cohort embeddings once from training data (if needed for AS-Norm)
        cohort_embeddings = None
        if self.normalize_test_scores:
            log.info("Computing shared cohort embeddings for AS-Norm...")
            # Use training loader params for cohort loader.
            train_dm = self.trainer.datamodule
            cohort_loader = DataLoader(
                train_dm.train_data,
                batch_size=getattr(train_dm.loaders.train, 'batch_size', 256),
                num_workers=getattr(train_dm.loaders.train, 'num_workers', 0),
                shuffle=False,
                pin_memory=getattr(train_dm.loaders.train, 'pin_memory', False),
                collate_fn=TrainCollate()
            )
            cohort_embeddings = self._compute_cohort_embeddings(cohort_loader)
            log.info(f"Computed cohort embeddings of shape: {cohort_embeddings.shape}")

        self.test_sets_data = {}
        self.last_batch_indices = {}

        for test_filename, dataloader in test_dataloaders.items():
            log.info(f"Processing '{test_filename}'...")
            
            # Determine the index of the last batch for this dataloader
            # This is needed for triggering metric computation at the right time
            try:
                num_batches = len(dataloader)
                if num_batches > 0:
                    self.last_batch_indices[test_filename] = num_batches - 1
                    log.info(f"Registered '{test_filename}' with {num_batches} batches. Last batch index: {num_batches - 1}.")
            except TypeError:
                log.warning(f"Could not determine the number of batches for '{test_filename}'. "
                            "Metric computation will fall back to on_test_epoch_end.")
                self.last_batch_indices[test_filename] = -1 # Fallback

            # Get unique utterances for enrollment and trials
            enroll_dataloader, trial_unique_dataloader = self.trainer.datamodule.get_enroll_and_trial_dataloaders(test_filename)
            
            # Compute embeddings for enrollment and test utterances
            enrol_embeds = self._compute_embeddings(enroll_dataloader, mode='enrollment')
            test_embeds = self._compute_embeddings(trial_unique_dataloader, mode='test')
            
            # Store all data for this test set (sharing the same cohort embeddings)
            self.test_sets_data[test_filename] = {
                'enrol_embeds': enrol_embeds,
                'test_embeds': test_embeds,
                'cohort_embeddings': cohort_embeddings,
                'trial_results': []
            }
            
        log.info(f"Finished computing embeddings for all test sets")

    @torch.inference_mode()
    def test_step(self, batch: VoxCelebVerificationItem, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Handle test step and trigger logging on the last batch of each dataloader."""
        # Move batch to device (needed for some weird device error)
        batch = self._move_batch_to_device(batch)
        
        # Get test set name from dataloader index
        test_filenames = list(self.trainer.datamodule.test_dataloader().keys())
        test_filename = test_filenames[dataloader_idx]
        test_data = self.test_sets_data[test_filename]
        
        # Run trial evaluation for this test set
        self._trials_eval_step_multi_test(batch, test_data)

        # Check if this is the last batch for the current dataloader
        is_last_batch = (batch_idx == self.last_batch_indices.get(test_filename, -1))

        if is_last_batch:
            log.info(f"Last batch for '{test_filename}' (idx: {batch_idx}) reached. Finalizing and logging metrics.")
            self._epoch_end_common_multi_test(test_filename)

    def on_test_epoch_end(self) -> None:
        """Callback at the end of the test epoch.
        
        This is now primarily a cleanup step. It also handles any test sets
        where the number of batches could not be determined.
        """
        # Process any test sets that were not handled in test_step (fallback)
        if hasattr(self, 'last_batch_indices'):
            for test_filename, last_idx in self.last_batch_indices.items():
                if last_idx == -1:
                    log.warning(f"Running fallback metric computation for '{test_filename}' at epoch end.")
                    self._epoch_end_common_multi_test(test_filename)

        log.info("Test epoch finished. All test sets have been processed.")
        # Clean up stored data to free memory
        if hasattr(self, 'test_sets_data'):
            del self.test_sets_data
        if hasattr(self, 'last_batch_indices'):
            del self.last_batch_indices

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

    def _compute_cohort_embeddings(self, dataloader) -> dict:
        exp_root_path = Path(self.trainer.default_root_dir)
        cohort_path = next(exp_root_path.rglob('test*/test_cohort_embeds.pt'), None)
        assert cohort_path is None or cohort_path.is_file(), f'Unexpected cohort_path file: {cohort_path}'        
        
        if cohort_path is not None:
            log.info('Loading Cohort Embeddings')
            return torch.load(cohort_path)

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

        # Don't update the shared metric - we'll compute metrics separately per test set
        # self.test_metric.update(scores=normalized_scores, labels=torch.tensor(batch.trial_label))
        
        batch_dict = {
            "enroll_path": batch.enroll_path,
            "test_path": batch.test_path,
            "enroll_length": batch.enroll_length,
            "test_length": batch.test_length,
            "trial_label": batch.trial_label,
            "same_country_label": batch.same_country_label,
            "same_gender_label": batch.same_gender_label,
            "score": raw_scores.detach().cpu().tolist(),
            "norm_score": normalized_scores.detach().cpu().tolist(),
        }
        test_data['trial_results'].append(batch_dict)
        
    def _epoch_end_common_multi_test(self, test_filename: str) -> None:
        """Handle epoch end for a specific test set."""
        test_data = self.test_sets_data[test_filename]
        enrol_embeds = test_data['enrol_embeds']
        trials_embeds = test_data['test_embeds']
        trial_results = test_data['trial_results']
        
        # Build scores DataFrame
        scores = pd.DataFrame([
            {
                "enroll_path": enroll_path,
                "test_path": test_path,
                "enroll_length": enroll_length,
                "test_length": test_length,
                "trial_label": trial_label,
                "same_country_label": same_country_label,
                "same_gender_label": same_gender_label,
                "score": score,
                "norm_score": norm_score,
            }
            for batch in trial_results
            for enroll_path, test_path, enroll_length, test_length, 
            trial_label, same_country_label, same_gender_label, score, norm_score in zip(
                batch["enroll_path"],
                batch["test_path"],
                batch["enroll_length"],
                batch["test_length"],
                batch["trial_label"],
                batch["same_country_label"],
                batch["same_gender_label"],
                batch["score"],
                batch["norm_score"],
            )
        ])

        # Create a temporary metric instance for this specific test set
        temp_metric = instantiate(self.hparams.metrics.test)
        
        # Update the temporary metric with this test set's data
        all_norm_scores = []
        all_labels = []
        for batch in trial_results:
            all_norm_scores.extend(batch["norm_score"])
            all_labels.extend(batch["trial_label"])
        
        temp_metric.update(scores=torch.tensor(all_norm_scores), labels=torch.tensor(all_labels))
        
        # Compute metrics for this specific test set
        metrics = temp_metric.compute()
        
        # Log metrics with a clear prefix for each test set
        temp_metric_class_name = temp_metric.__class__.__name__
        prefixed_metrics = {f"test/{test_filename}/{temp_metric_class_name}/{key}": value for key, value in metrics.items()}
        self.log_dict(prefixed_metrics, on_step=False, on_epoch=True, sync_dist=True)

        # Minimal console log to confirm completion
        main_metric_key = next(iter(metrics))
        main_metric_val = metrics[main_metric_key].item() if torch.is_tensor(metrics[main_metric_key]) else metrics[main_metric_key]
        log.info(f"Logged {test_filename} results. Main metric ({main_metric_key}): {main_metric_val:.4f}")

        # Update scores DataFrame with computed metrics
        scores.loc[:, metrics.keys()] = [v.item() if torch.is_tensor(v) else v for v in metrics.values()]
        
        # Set up directory for saving test artifacts (with test set name)
        dir_suffix = f"_{test_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        artifacts_dir = os.path.join(self.trainer.default_root_dir, f"test_artifacts{dir_suffix}")
        os.makedirs(artifacts_dir, exist_ok=True)

        # Save scores as CSV
        scores.to_csv(os.path.join(artifacts_dir, f"{test_filename}_scores.csv"), index=False)
        
        # Save embeddings
        torch.save(enrol_embeds, os.path.join(artifacts_dir, f"{test_filename}_enrol_embeds.pt"))
        torch.save(trials_embeds, os.path.join(artifacts_dir, f"{test_filename}_embeds.pt"))
        if test_data['cohort_embeddings'] is not None:
            torch.save(test_data['cohort_embeddings'], os.path.join(artifacts_dir, f"test_cohort_embeds.pt"))  # Same cohort for all test sets

        # Plot and log figures for this test set
        figures = temp_metric.plot_curves() or {}
        for name, fig in figures.items():
            self.log_figure_with_fallback(f"{test_filename}/binary_metrics_plots/{name}_scores", fig)

        # Save test metrics as a JSON file 
        metrics_for_save = {k: v.item() if torch.is_tensor(v) else v for k, v in metrics.items()}
        metrics_for_save['test_set'] = test_filename  # Add test set identifier
        with open(os.path.join(artifacts_dir, f"{test_filename}_metrics.json"), "w") as f:
            json.dump(metrics_for_save, f, indent=4)

    def log_figure_with_fallback(self, name: str, fig: plt.Figure) -> None:
        """Log figure with fallback for loggers that don't support figure logging.
        
        Args:
            name: Name of the figure
            fig: Matplotlib figure to log
        """
        # Save figure to disk as fallback
        artifacts_dir = os.path.join(self.trainer.default_root_dir, f"{os.path.dirname(name)}")
        os.makedirs(artifacts_dir, exist_ok=True)
        fig_path = os.path.join(artifacts_dir, f"{os.path.basename(name)}.png")
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        
        # Convert figure to image buffer for loggers that need it
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        if self.logger is None:
            log.warning("No logger available for logging figures")
            return
        
        # Handle either a single logger or a collection of loggers
        loggers = self.logger.loggers if hasattr(self.logger, 'loggers') else [self.logger]
        logged_successfully = False
        
        for logger in loggers:
            try:
                # TensorBoard logger
                if hasattr(logger, 'experiment') and hasattr(logger.experiment, 'add_figure'):
                    logger.experiment.add_figure(f'{name}', fig, global_step=self.global_step)
                    logged_successfully = True
                    continue
                    
                # Weights & Biases logger
                if hasattr(logger, 'experiment') and 'wandb' in str(type(logger.experiment).__module__):
                    import wandb
                    logger.experiment.log({f'{name}': wandb.Image(fig)}, step=self.global_step)
                    logged_successfully = True
                    continue
                    
                # Neptune logger - simplified approach using file path
                if hasattr(logger, 'experiment') and 'neptune' in str(type(logger.experiment).__module__):
                    logger.experiment[f'{name}'].upload(fig)
                    logged_successfully = True
                    continue
                    
                # MLflow logger
                if hasattr(logger, 'experiment') and hasattr(logger.experiment, 'log_figure'):
                    logger.experiment.log_figure(logger.run_id, fig, f'{name}.png')
                    logged_successfully = True
                    continue
                    
                # Logger type doesn't support figure logging
                logger_type = type(logger).__name__
                log.debug(f"Logger type {logger_type} doesn't support figure logging")
                
            except Exception as e:
                log.warning(f"Error logging figure {name} to logger {type(logger).__name__}: {str(e)}")
        
        if not logged_successfully:
            log.info(f"Figure '{name}' was saved to disk but couldn't be logged to any logger")
        
        # Always close the figure to prevent memory leaks
        plt.close(fig)
