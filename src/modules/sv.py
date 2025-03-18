import os
import json
from typing import Any, Dict, Optional, Tuple
from collections import defaultdict
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

from src.datamodules.components.voxceleb.voxceleb_dataset import (
    VoxcelebItem,
    VoxCelebVerificationItem,
    TrainCollate)
from src import utils
from src.modules.components.utils import EmbeddingCache
from src.modules.metrics.metrics import AS_norm 
from datetime import datetime


log = utils.get_pylogger(__name__)



###################################
class EmbeddingMetrics:
    def __init__(self, trainer: 'pl.Trainer', stage: str, cohort_per_model: int = 1000):
        self.trainer = trainer
        self.stage = stage
        self.cohort_per_model = cohort_per_model

    @property
    def _trial_results(self):
        if not hasattr(self, '_results'):
            self._results = []
        return self._results

    @property
    def _cohort_indices(self, speaker_col: str = 'speaker_id'):
        """
        Compute balanced cohort indices ensuring good coverage across speakers.
        
        Args:
            speaker_col: Name of the column containing speaker IDs
            
        Returns:
            List of selected indices for cohort
        """
        if not hasattr(self, '_indices'):
            df = self.trainer.datamodule.train_data.dataset
            
            # Validate required columns
            if speaker_col not in df.columns:
                raise ValueError(f"Missing required column: {speaker_col}")
            
            # Get unique speakers
            unique_speakers = df[speaker_col].unique()
            min_speakers_required = int(0.8 * len(unique_speakers))
            
            # Sample from each speaker (max 10 samples per speaker)
            all_indices = []
            for speaker in list(unique_speakers):
                indices = df[df[speaker_col] == speaker].index.tolist()
                all_indices.extend(random.sample(indices, min(10, len(indices))))
                
                # Stop once we have enough speaker coverage
                if len(all_indices) >= min_speakers_required:
                    break
            
            self._indices = all_indices
        
        return self._indices

    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        assert self.stage == 'test'
        enroll_loader, unique_loader = self.trainer.datamodule.enrolling_dataloader()
        return enroll_loader, unique_loader

    def get_cohort_loader(self) -> DataLoader:        
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
        """Get the computed cohort embeddings."""
        return self._embeddings

    def set_cohort_embeddings(self, embeddings_dict: Dict[str, torch.Tensor]) -> None:
        """Set the cohort embeddings.
        
        Args:
            embeddings_dict: Dictionary mapping model IDs to their cohort embeddings
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
        self.batch_sizes = kwargs.get("batch_sizes")
        
        # Initialize metrics
        self._setup_metrics(metrics)
        
        # Initialize model components
        self._setup_model_components(model)
        
        # Setup training components
        self._setup_training_components(criterion, optimizer, lr_scheduler)
        
        # Freeze pretrained components
        self._freeze_pretrained_components(finetune=False)

        # Initialize text embedding cache with appropriate limits
        self._embeds_cache_config = model.get("embedding_cache", {})
        self._max_cache_size = self._embeds_cache_config.get("max_size", 500000)
        self._bypass_warmup = self._embeds_cache_config.get("bypass_warmup", False)
        self._embedding_cache = EmbeddingCache(max_size=self._max_cache_size)
        
        # Initialize cohort embeddings for score normalization
        self.normalize_test_scores = model.get("normalize_test_scores", False)

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
        self.audio_encoder = instantiate(model.audio_encoder)
        self.audio_processor_kwargs = model.audio_processor_kwargs
        self.classifier = instantiate(model.classifiers.classifier)

    def _setup_training_components(self, criterion: DictConfig, optimizer: DictConfig, lr_scheduler: DictConfig) -> None:
        """Initialize loss functions, optimizer and learning rate scheduler."""
        self.train_criterion = instantiate(criterion.train_criterion)
        self.valid_criterion = instantiate(criterion.valid_criterion)     
        self.optimizer = optimizer
        self.slr_params = lr_scheduler

    def _freeze_pretrained_components(self, finetune=False) -> None:
        """Freeze pretrained components and enable training for others."""
        for param in self.audio_encoder.parameters():
            param.requires_grad = finetune

    def _log_step_metrics(self, results: Dict[str, Any], batch: VoxcelebItem, stage: str) -> None:
        criterion = getattr(self, f"{stage}_criterion")
        
        # Log losses
        logged_dict = {f"{stage}/{criterion.__class__.__name__}": results['loss'].item()}

        self.log_dict(
            logged_dict,
            batch_size=getattr(self.batch_sizes, stage),
            **self.logging_params
        )

        # Log metrics
        metric = getattr(self, f"{stage}_metric")
        computed_metric = metric(results["outputs"][f"logits"], batch.class_id)
        
        self.log(
            f"{stage}/{metric.__class__.__name__}",
            computed_metric,
            batch_size=getattr(self.batch_sizes, stage),
            **self.logging_params
        )

    ############ Caching ############
    def _warmup_cache(self):
        """Pre-computes and caches text embeddings for unique training texts.
        
        Uses batched processing for memory efficiency and shows a progress bar.
        """
        # Get unique audios from training data
        unique_audios = list(set(self.trainer.datamodule.train_data.dataset.audio))
        unique_audios_lens = list(set(self.trainer.datamodule.train_data.dataset.audio_lens))
        
        # Define batch size for processing
        batch_size = 384
        
        # Optional: Limit to subset of texts for faster startup
        max_texts = batch_size * 10  # Uncomment to process only 10 batches
        unique_audios = unique_audios[:max_texts]
        unique_audios_lens = unique_audios_lens[:max_texts]
        
        # Process texts in batches with progress bar
        with torch.no_grad():
            with tqdm(total=len(unique_audios), desc="Warming up cache") as pbar:
                for i in range(0, len(unique_audios), batch_size):
                    batch_audios = unique_audios[i: i + batch_size]
                    batch_audios_lens = unique_audios_lens[i: i + batch_size] 
                    # Get embeddings and immediately delete the tensor since we only need the cached values
                    _ = self.get_audio_embeddings(batch_audios, batch_audios_lens)
                    pbar.update(len(batch_audios))

    ############ Lightning ############
    def _get_uncached_audio_embeddings(self, batch_audio: torch.Tensor, batch_audio_lens: torch.Tensor) -> torch.Tensor:
        if hasattr(self.audio_encoder, "encode_batch"):
            # For speechbrain encoders (e.g., x-vector)
            audio_emb = self.audio_encoder.encode_batch(
                wavs=batch_audio,
                wav_lens=batch_audio_lens/max(batch_audio_lens)
                ).squeeze(1)
        
        # For transformers-based encoders (e.g., wav2vec) 
        elif False:              
            input_values = self.audio_processor(
                batch_audio, 
                **self.audio_processor_kwargs).input_values.squeeze(0).to(self.device)
            audio_outputs = self.audio_encoder(input_values)
            audio_emb = audio_outputs.last_hidden_state.mean(dim=1)
            
        else:
            # TODO: strange mismatch during testing 
            if self.device != batch_audio.device:
                batch_audio = batch_audio.to(self.device)
            if self.device != batch_audio_lens.device:
                batch_audio_lens = batch_audio_lens.to(self.device)
            input_values = self.audio_processor(batch_audio)
            audio_emb = self.audio_encoder(input_values, lengths=batch_audio_lens/max(batch_audio_lens)).squeeze(1)
        
        return audio_emb

    def _get_audio_embeddings(self, batch_audio: torch.Tensor, batch_audio_lens: torch.Tensor) -> torch.Tensor:
        """Get audio embeddings with caching optimization.
        
        Args:
            batch_audio: Tensor of audio inputs to embed
            batch_audio_lens: Tensor of audio lengths
            
        Returns:
            torch.Tensor: Stacked tensor of embeddings for all audio inputs on the model's device
        """
        # Pre-allocate list for embeddings
        audio_embeddings = [None] * len(batch_audio)
        uncached_audio = []
        uncached_lens = []
        uncached_indices = []
        
        # Check cache for each audio input
        for idx, (audio, length) in enumerate(zip(batch_audio, batch_audio_lens)):
            audio_key = str(audio.cpu().numpy().tobytes())
            cached_embedding = self._embedding_cache.get(audio_key)
            if cached_embedding is not None:
                # Move cached embedding to current device
                audio_embeddings[idx] = cached_embedding.to(self.device)
            else:
                uncached_audio.append(audio)
                uncached_lens.append(length)
                uncached_indices.append(idx)
        
        # Process uncached audio in a single batch if any exist
        if uncached_audio:
            # Stack uncached audio into a batch
            uncached_audio = torch.stack(uncached_audio)
            uncached_lens = torch.stack(uncached_lens)
            
            # Get embeddings for uncached audio
            with torch.no_grad():
                new_embeddings = self._get_uncached_audio_embeddings(uncached_audio, uncached_lens)
            
            # Update cache and embeddings list
            for idx, (audio, embedding) in enumerate(zip(uncached_audio, new_embeddings)):
                # Create key from audio tensor
                audio_key = str(audio.cpu().numpy().tobytes())
                # Store embedding in cache (detached and on CPU)
                self._embedding_cache.update(audio_key, embedding.detach().cpu())
                # Use the embedding directly from GPU for current forward pass
                audio_embeddings[uncached_indices[idx]] = embedding
        
        # Stack all embeddings and ensure they're on the correct device
        return torch.stack(audio_embeddings)

    def forward(self, batch: VoxcelebItem) -> Dict[str, torch.Tensor]:
        """Process text/audio inputs with optimized embedding caching."""
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
        if self.current_epoch == 0 and not self._bypass_warmup:
            self._warmup_cache()

    def training_step(self, batch: VoxcelebItem, batch_idx: int) -> Dict[str, torch.Tensor]:
        results = self.model_step(batch, self.train_criterion)
        
        if batch_idx % 3000 == 0:
            torch.cuda.empty_cache()

        self._log_step_metrics(results, batch, 'train')
        return results

    def on_train_epoch_end(self) -> None:
        self.train_metric.reset()

        # Cache processing
        stats = self._embedding_cache.stats()
        self.log("train/cache/cache_hit_rate", stats["hit_rate"])
        self.log("train/cache/cache_size", len(self._embedding_cache))
        # resize the cache if it exceeds the max size
        if len(self._embedding_cache) > self._max_cache_size:
            self._embedding_cache.resize(self._max_cache_size)

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
        """Compute embeddings for test trials."""
        self.metric_tracker = EmbeddingMetrics(self.trainer, 'test')
        enroll_test_loader, unique_test_loader = self.metric_tracker.get_loaders()

        if self.normalize_test_scores:
            cohort_loader = self.metric_tracker.get_cohort_loader()
            cohort_embeddings = self._compute_cohort_embeddings(cohort_loader)
            self.metric_tracker.set_cohort_embeddings(cohort_embeddings)
        else:
            self.metric_tracker.set_cohort_embeddings(None)

        self.enrol_embeds = self._compute_test_embeddings(enroll_test_loader, mode='enrollment')
        self.test_embeds = self._compute_test_embeddings(unique_test_loader, mode='test')

    @torch.inference_mode()
    def test_step(self, batch: VoxCelebVerificationItem, batch_idx: int) -> None:
        """Compute EER and minDCF on these test trials"""        
        self._trials_eval_step(batch, is_test=True)

    def on_test_epoch_end(self) -> None:
        self._epoch_end_common(is_test=True)

    def configure_optimizers(self) -> Dict:
        """Configure optimizers and learning rate schedulers."""
        optimizer: torch.optim = instantiate(self.optimizer)(params=self.parameters())
        
        if self.slr_params.get("scheduler"):
            scheduler: torch.optim.lr_scheduler = instantiate(
                self.slr_params.scheduler,
                optimizer=optimizer,
                _convert_="partial",
            )
            
            lr_scheduler_dict = {"scheduler": scheduler}
            if self.slr_params.get("extras"):
                for key, value in self.slr_params.get("extras").items():
                    lr_scheduler_dict[key] = value
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
        
        return {"optimizer": optimizer}

    ############ Dev and eval utils ############
    def _compute_test_embeddings(self, dataloader, mode: str = 'test') -> dict:
        embeddings_dict = {}
        desc = f"Computing {mode} embeddings"

        with tqdm(dataloader, desc=desc) as pbar: 
            for batch in pbar:
                outputs = self(batch)
                embed = outputs['embeds']
                # Handle frame-level embeddings (if applicable)
                if len(embed.shape) == 3:  # [1, num_frames, embedding_dim]
                    embed = embed.mean(dim=1)  # [num_frames, embedding_dim]

                embeddings_dict.update({path: emb for path, emb in zip(batch.audio_path, embed)})

        return embeddings_dict

    def _compute_cohort_embeddings(self, dataloader) -> dict:
        embeddings_dict = {}

        with tqdm(dataloader, desc="Computing cohort embeddings") as pbar:
            for batch in pbar:
                outputs = self(batch)
                cohort = outputs['embeds']

                # Handle frame-level embeddings (if applicable)
                if len(cohort.shape) == 3:  # [1, num_frames, embedding_dim]
                    cohort = cohort.mean(dim=1)  # [num_frames, embedding_dim]

                # Extract model keys from audio paths
                model_keys = [path.split(os.sep)[-5] for path in batch.audio_path]
                
                # Ensure model_keys is a list
                if not isinstance(model_keys, (list, tuple)):
                    model_keys = [model_keys]

                # Update embeddings dictionary
                for model_key in set(model_keys):
                    if model_key not in embeddings_dict:
                        embeddings_dict[model_key] = []
                    embeddings_dict[model_key].append(cohort)

        # Stack all embeddings for each model into a single tensor
        final_embeddings = {
            model_id: torch.cat(embeddings, dim=0)
            for model_id, embeddings in embeddings_dict.items()
        }

        return final_embeddings

    def _trials_eval_step(self, batch, is_test: bool):
        """Common logic for test and validation steps."""
        embeds = self.test_embeds if is_test else self.dev_embeds
        enrol = self.enrol_embeds if is_test else self.enrol_dev_embeds
        metric = self.test_metric if is_test else self.valid_metric

        trial_embeddings = torch.stack([embeds[path] for path in batch.test_path])
        enroll_embeddings = torch.stack([enrol[path] for path in batch.enroll_path])
        cohort_embeddings = metric.cohort_embeddings

        # Compute raw cosine similarity scores
        raw_scores = torch.nn.functional.cosine_similarity(enroll_embeddings, trial_embeddings)
        
        # Apply AS-Norm
        if cohort_embeddings is not None:
            normalized_scores = []
            for i, (enroll_emb, test_emb) in enumerate(zip(enroll_embeddings, trial_embeddings)):
                raw_score = raw_scores[i]
                norm_score = AS_norm(score=raw_score,
                                     enroll_embedding=enroll_emb,
                                     test_embedding=test_emb, 
                                     cohort_embeddings=cohort_embeddings,
                                     topk=500)
                normalized_scores.append(norm_score)
            
            # Convert back to tensor
            normalized_scores = torch.tensor(normalized_scores, device=raw_scores.device)
        
        else:
            normalized_scores = raw_scores.clone()

        # Update metric with normalized scores
        metric.update(scores=normalized_scores, labels=torch.tensor(batch.trial_label))
        
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
        self.metric_tracker._trial_results.append(batch_dict)

    def _epoch_end_common(self, is_test: bool) -> None:
        """Common logic for test and validation epoch end."""
        metric = self.test_metric if is_test else self.valid_metric
        enrol_embeds = self.enrol_embeds if is_test else self.enrol_dev_embeds
        trials_embeds = self.test_embeds if is_test else self.dev_embeds

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
            for batch in self.metric_tracker._trial_results
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

        self._end_of_epoch_metrics(
            enrol_embeds=enrol_embeds,
            trials_embeds=trials_embeds,
            scores=scores,
            metric=metric,
            is_test=is_test
        )

        self.metric_tracker.cleanup()

    def _end_of_epoch_metrics(self, enrol_embeds: Dict, trials_embeds: Dict, scores: pd.DataFrame, metric, is_test: bool) -> None:
        """Compute EER and minDCF, handle logging and saving of artifacts."""
        # Compute metrics (EER, minDCF, etc.)
        metrics = metric.compute()
        
        # Log metrics with appropriate prefix (test or valid)
        stage = 'test' if is_test else 'valid'
        prefixed_metrics = {f"{stage}/{metric.__class__.__name__}/{key}": value for key, value in metrics.items()}
        self.log_dict(prefixed_metrics, **self.logging_params)

        # Update scores DataFrame with computed metrics
        scores.loc[:, metrics.keys()] = [v.item() if torch.is_tensor(v) else v for v in metrics.values()]
        
        # Set up directory for saving artifacts
        dir_suffix = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if is_test else ""
        artifacts_dir = os.path.join(self.trainer.default_root_dir, f"{stage}_artifacts{dir_suffix}")
        os.makedirs(artifacts_dir, exist_ok=True)

        # Save scores as CSV
        scores.to_csv(os.path.join(artifacts_dir, f"{stage}_scores.csv"), index=False)
        
        # Save embeddings
        torch.save(enrol_embeds, os.path.join(artifacts_dir, f"{stage}_enrol_embeds.pt"))
        torch.save(trials_embeds, os.path.join(artifacts_dir, f"{stage}_embeds.pt"))
        if metric.cohort_embeddings is not None:
            torch.save(metric.cohort_embeddings, os.path.join(artifacts_dir, f"{stage}_cohort_embeds.pt"))

        # Plot and log figures for the current epoch
        figures = metric.plot_curves() or {}
        for name, fig in figures.items():
            self.log_figure_with_fallback(f"{stage}_{name}_scores", fig, stage=stage, step=self.current_epoch)

        # Save test metrics as a JSON file during the test phase
        if is_test:
            metrics_for_save = {k: v.item() if torch.is_tensor(v) else v for k, v in metrics.items()}
            with open(os.path.join(artifacts_dir, "test_metrics.json"), "w") as f:
                json.dump(metrics_for_save, f, indent=4)
 
    def log_figure_with_fallback(self, name: str, fig: plt.Figure, stage: str, step: int) -> None:
        """Log figure with fallback for loggers that don't support figure logging."""
        try:
            if hasattr(self.logger, 'experiment'):
                logger_type = type(self.logger.experiment).__name__
                if logger_type == 'SummaryWriter':  # TensorBoard
                    self.logger.experiment.add_figure(f'binary_metrics_plots/{name}', fig, global_step=step)
                else:  # Other loggers like WandB or MLFlow
                    self.logger.experiment[f'binary_metrics_plots/{name}'].upload(fig)
        finally:
            # Always close the figure to prevent memory leaks
            plt.close(fig)

    ############ Load and save ############
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Saves the text embedding cache state in the model checkpoint.
        
        Args:
            checkpoint: Dictionary containing model checkpoint data
        """
        # Call parent class's save checkpoint method if it exists
        super().on_save_checkpoint(checkpoint)
        
        # Save cache contents and metadata
        cache_state = {
            'max_size': self._embedding_cache.max_size,
            'hits': self._embedding_cache.hits,
            'misses': self._embedding_cache.misses,
            'contents': {
                key: tensor.cpu() 
                for key, tensor in self._embedding_cache._cache.items()
            }
        }
        checkpoint['text_embedding_cache'] = cache_state

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Restores the text embedding cache state from the model checkpoint.
        
        Args:
            checkpoint: Dictionary containing model checkpoint data
        """
        # Call parent class's load checkpoint method if it exists
        super().on_load_checkpoint(checkpoint)
        
        # Restore cache if it exists in checkpoint
        if 'text_embedding_cache' in checkpoint:
            cache_state = checkpoint['text_embedding_cache']
            
            # Recreate cache with saved size
            self._embedding_cache = EmbeddingCache(max_size=cache_state['max_size'])
            
            # Restore performance counters
            self._embedding_cache.hits = cache_state['hits']
            self._embedding_cache.misses = cache_state['misses']
            
            # Restore cached embeddings
            for key, tensor in cache_state['contents'].items():
                self._embedding_cache.update(key, tensor)