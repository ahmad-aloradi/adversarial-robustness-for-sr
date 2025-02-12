import os
from typing import Any, Dict, Optional, List, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from src.datamodules.components.vpc25.vpc_dataset import VPC25Item, VPC25VerificationItem
from src import utils
from src.modules.components.utils import EmbeddingCache
from src.modules.losses.components.focal_loss import FocalLoss


log = utils.get_pylogger(__name__)


# Override the following constants if needed
MAIN_FEATS = "fusion"
AUX_FEATS = "audio"
EMBED_FEATS = "fusion"  # which embedding to use for speaker ID

# Constants
METRIC_NAMES = {
    "TRAIN": "train",
    "VALID": "valid",
    "TEST": "test",
    "BEST": "valid_best"
}

LOSS_TYPES = {
    "TEXT": "text",
    "AUDIO": "audio",
    "GENDER": "gender",
    "FUSION": "fusion",
    "MAIN": MAIN_FEATS,
    "AUX": AUX_FEATS
}

EMBEDS = {
    "TEXT": "text_embed",
    "AUDIO": "audio_embed",
    "FUSION": "fusion_embed",
    "ID": f"{EMBED_FEATS}_embed",
    "CLASS": "class_preds"
}


###################################
""""Custom Fusion Models"""
class NormalizedWeightedSum(nn.Module):
    def __init__(self, audio_embedding_size, text_embedding_size, bottleneck_size, *args, **kwargs):
        super().__init__()
        # Projection layers
        self.audio_proj = nn.Sequential(
            nn.LayerNorm(audio_embedding_size),
            nn.Linear(audio_embedding_size, bottleneck_size)
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_embedding_size),
            nn.Linear(text_embedding_size, bottleneck_size)
        )
        # Learnable weights for weighted sum
        self.audio_weight = nn.Parameter(torch.tensor(1.0))
        self.text_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, emebds):
        assert len(emebds) == 2, f"Expected 2 embeddings, but found: {len(emebds)}"
        assert type(emebds) == tuple, f"Expected tuple of embeddings, but found: {type(emebds)}"
        audio_emb = emebds[0]
        text_emb = emebds[1]

        audio_emb = self.audio_proj(audio_emb)
        text_emb = self.text_proj(text_emb)
    
        return {"fusion_emb": self.audio_weight * audio_emb + self.text_weight * text_emb, 
                "audio_emb": audio_emb,
                "text_emb": text_emb}  

class ConcatFusion(torch.nn.Module):
    def __init__(self, dim=1, *args, **kwargs):
        super(ConcatFusion, self).__init__()
        self.dim = dim

    def forward(self, emebds):
        assert len(emebds) == 2, f"Expected 2 embeddings, but found: {len(emebds)}"
        assert type(emebds) == tuple, f"Expected tuple of embeddings, but found: {type(emebds)}"
        return {"fusion_emb": torch.cat(emebds, dim=self.dim),
                "audio_emb": emebds[0],
                "text_emb": emebds[1]}

###################################
class EnhancedCriterion(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.focal_loss = FocalLoss(gamma=2.0)
        self.eps = 1e-8
        
    def get_contrastive_pairs(self, embeddings: torch.Tensor, targets: torch.Tensor):
        """Create positive and negative pairs from embeddings within the batch."""
        batch_size = embeddings.size(0)
        # Create a mask for positive pairs (same class)
        labels_matrix = targets.expand(batch_size, batch_size)
        positive_mask = labels_matrix == labels_matrix.T
        # Remove self-pairs from positive mask
        positive_mask.fill_diagonal_(False)
        
        # Mask for negative pairs (different class)
        negative_mask = ~positive_mask
        negative_mask.fill_diagonal_(False)
        
        return positive_mask, negative_mask
    
    def contrastive_loss(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss using cosine similarity with improved numerical stability."""
        # Check for valid input
        if embeddings.size(0) < 2:
            return torch.tensor(0.0, device=embeddings.device)
            
        # Normalize embeddings and clamp for stability
        embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        embeddings_norm = torch.clamp(embeddings_norm, min=-1.0 + self.eps, max=1.0 - self.eps)
        
        # Compute similarity matrix with temperature scaling
        similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.T)
        similarity_matrix = similarity_matrix / self.temperature
        
        # Get positive and negative masks
        positive_mask, negative_mask = self.get_contrastive_pairs(embeddings, targets)
        
        # Count valid pairs per sample
        n_positives = positive_mask.sum(dim=1)
        n_negatives = negative_mask.sum(dim=1)
        
        # Skip samples with no positive or negative pairs
        valid_samples = (n_positives > 0) & (n_negatives > 0)
        if not valid_samples.any():
            return torch.tensor(0.0, device=embeddings.device)
            
        # Apply log-sum-exp trick for numerical stability
        max_sim = torch.max(similarity_matrix, dim=1, keepdim=True)[0].detach()
        exp_sim = torch.exp(similarity_matrix - max_sim)
        
        # Compute log probability ratio for valid samples
        pos_exp_sum = torch.sum(exp_sim * positive_mask, dim=1)
        neg_exp_sum = torch.sum(exp_sim * negative_mask, dim=1)
        
        # Compute per-sample loss with careful handling of denominators
        denominator = pos_exp_sum + neg_exp_sum + self.eps
        per_sample_loss = -torch.log(pos_exp_sum / denominator + self.eps)
        
        # Only consider loss for valid samples
        per_sample_loss = per_sample_loss * valid_samples.float()
        
        # Average loss over valid samples
        n_valid_samples = valid_samples.sum()
        loss = per_sample_loss.sum() / (n_valid_samples + self.eps)
        
        # Check for NaN and return zero if found
        if torch.isnan(loss):
            log.warning("NaN detected in contrastive loss! Returning zero.")
            return torch.tensor(0.0, device=embeddings.device)
            
        return loss

    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce_loss(outputs[f"{LOSS_TYPES['MAIN']}_logits"], targets)
        contr_loss = self.contrastive_loss(outputs[EMBEDS["FUSION"]], targets)
        focal_loss = self.focal_loss(outputs[f"{LOSS_TYPES['MAIN']}_logits"], targets)
        
        # Log component losses for debugging
        if torch.isnan(ce_loss) or torch.isnan(contr_loss) or torch.isnan(focal_loss):
            log.warning(f"NaN detected in losses - CE: {ce_loss.item()}, Contrastive: {contr_loss.item()}, Focal: {focal_loss.item()}")
        
        # Increase CE loss weight and reduce contrastive loss weight
        return 1.0 * ce_loss + 0.05 * contr_loss + 0.1 * focal_loss

class ResidualBlock(nn.Module):
    """A residual block with batch normalization and dropout."""
    def __init__(self, in_features: int, hidden_features: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_features, in_features),
            nn.BatchNorm1d(in_features)
        )
        
        # Initialize weights
        for m in self.block.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu(x + self.block(x))

class FusionClassifier(nn.Module):
    """Fusion classifier with residual connections."""
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_size)
        self.residual1 = ResidualBlock(input_size, hidden_size, dropout)
        self.residual2 = ResidualBlock(input_size, hidden_size, dropout)
        self.classifier = nn.Linear(input_size, num_classes)
        
        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.residual1(x)
        x = self.residual2(x)
        return self.classifier(x)

###################################
class EmbeddingMetrics:
    def __init__(self, trainer, stage: str):
        self.trainer = trainer
        self.stage = stage
        
    @property
    def _trial_results(self):
        if not hasattr(self, '_results'):
            self._results = []
        return self._results

    def get_loaders(self):
        if self.stage == 'test':
            return self.trainer.datamodule.test_enrollment_dataloader()
        return self.trainer.datamodule.dev_enrollment_dataloader()

    def cleanup(self):
        if hasattr(self, '_results'):
            delattr(self, '_results')

class MultiModalVPCModel(pl.LightningModule):
    """Multi-modal Voice Print Classification Model.
    
    This model combines text and audio processing for speaker identification and gender classification.
    """
    
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
        self._freeze_pretrained_components(finetune_audioenc=model.get("finetune_audioenc", True)) 

        # Initialize text embedding cache with appropriate limits
        self._text_embeds_cache_config = model.get("embedding_cache", {})
        self._max_cache_size = self._text_embeds_cache_config.get("max_size", 500000)
        self._bypass_text_warmup = self._text_embeds_cache_config.get("bypass_warmup", False)
        self._text_embedding_cache = EmbeddingCache(max_size=self._max_cache_size)

    ############ Setup init ############
    def _setup_metrics(self, metrics: DictConfig) -> None:
        """Initialize all metrics for training, validation and testing."""
        self.train_metric = instantiate(metrics.train)
        self.valid_metric = instantiate(metrics.valid)
        self.test_metric = instantiate(metrics.test)
        self.valid_metric_best = instantiate(metrics.valid_best)

    def _setup_model_components(self, model: DictConfig) -> None:
        """Initialize encoders and classifiers."""
        # Text processing
        self.text_processor = instantiate(model.text_processor)
        self.text_encoder = instantiate(model.text_encoder)
        self.text_processor_kwargs = model.text_processor_kwargs

        # Audio processing
        self.audio_processor = instantiate(model.audio_processor)
        self.audio_encoder = instantiate(model.audio_encoder)
        self.audio_processor_kwargs = model.audio_processor_kwargs
        
        # Fusion and classification
        self.audio_classifier = instantiate(model.classifiers.audio_classifier)
        self.fuser = instantiate(model.classifiers.fuser)
        self.fusion_classifier = instantiate(model.classifiers.fusion_classifier)

    def _setup_training_components(self, criterion: DictConfig, optimizer: DictConfig, lr_scheduler: DictConfig) -> None:
        """Initialize loss functions, optimizer and learning rate scheduler."""
        self.train_criterion = instantiate(criterion.train_criterion)        
        self._audio_embed_train = len(self.train_criterion) > 1
        self.optimizer = optimizer
        self.slr_params = lr_scheduler

    def _freeze_pretrained_components(self, finetune_audioenc: bool = False) -> None:
        """Freeze pretrained components and enable training for others."""
        if hasattr(self.audio_encoder, "encode_batch"):
                self._finetune_audioenc = finetune_audioenc    # Finetune for speechbrain encoders (e.g., x-vector)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.audio_encoder.parameters():
            param.requires_grad = self._finetune_audioenc

    def _log_step_metrics(self, results: Dict[str, Any], batch: VPC25Item, stage: str) -> None:
        criterion = getattr(self, f"{stage}_criterion")
        main_loss = criterion.main_criterion.__class__.__name__
        
        # Log losses
        logged_dict = {
            f"{LOSS_TYPES['MAIN']}_{main_loss}/{stage}": results[LOSS_TYPES['MAIN']].item()
        }
        
        if self._audio_embed_train:
            aux_loss = criterion.aux_criterion.__class__.__name__
            logged_dict.update({
                f"multitask_loss/{stage}": results["loss"].item(),
                f"{LOSS_TYPES['AUX']}_{aux_loss}/{stage}": results[LOSS_TYPES['AUX']].item()
            })
        
        self.log_dict(
            logged_dict,
            batch_size=getattr(self.batch_sizes, stage),
            **self.logging_params
        )

        # Log metrics
        metric = getattr(self, f"{stage}_metric")
        computed_metric = metric(
            results["outputs"][f"{LOSS_TYPES['MAIN']}_logits"],
            batch.class_id
        )
        
        self.log(
            f"{metric.__class__.__name__}/{stage}",
            computed_metric,
            batch_size=getattr(self.batch_sizes, stage),
            **self.logging_params
        )

    ############ Caching ############
    def _warmup_cache(self):
        """Pre-computes and caches text embeddings for unique training texts.
        
        Uses batched processing for memory efficiency and shows a progress bar.
        Cache warmup runs on a subset of unique texts (2 batches worth) to reduce startup time.
        """
        # Get unique texts from training data
        unique_texts = list(set(self.trainer.datamodule.train_data.dataset.text))
        
        # Limit to first two batches worth of texts for faster startup
        batch_size = 384
        
        # Process texts in batches with progress bar
        with torch.no_grad():
            with tqdm(total=len(unique_texts), desc="Warming up cache") as pbar:
                for i in range(0, len(unique_texts), batch_size):
                    batch_texts = unique_texts[i:i+batch_size]
                    # Get embeddings and immediately delete the tensor since we only need the cached values
                    embeddings = self.get_text_embeddings(batch_texts)
                    del embeddings  # Explicitly free memory
                    torch.cuda.empty_cache()  # Clear CUDA cache if using GPU
                    pbar.update(len(batch_texts))

    def get_text_embeddings(self, batch_texts: List[str]) -> torch.Tensor:
        """Get text embeddings with caching optimization.
        
        Args:
            batch_texts: List of text strings to embed
            
        Returns:
            torch.Tensor: Stacked tensor of embeddings for all texts on the model's device
        """
        # Pre-allocate list for embeddings
        text_embeddings = [None] * len(batch_texts)
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for each text
        for idx, text in enumerate(batch_texts):
            cached_embedding = self._text_embedding_cache.get(text)
            if cached_embedding is not None:
                # Move cached embedding to current device
                text_embeddings[idx] = cached_embedding.to(self.device)
            else:
                uncached_texts.append(text)
                uncached_indices.append(idx)
        
        # Process uncached texts in a single batch
        if uncached_texts:
            # Process all uncached texts in a single batch
            inputs_text = self.text_processor(uncached_texts, **self.text_processor_kwargs)
            inputs_text.input_ids = inputs_text.input_ids.to(self.device)
            inputs_text.attention_mask = inputs_text.attention_mask.to(self.device)
            
            with torch.no_grad():
                text_outputs = self.text_encoder(inputs_text.input_ids, attention_mask=inputs_text.attention_mask)
                new_embeddings = text_outputs.pooler_output
            
            # Update cache and embeddings list
            for idx, (text, embedding) in enumerate(zip(uncached_texts, new_embeddings)):
                # Store embedding in cache (detached and on CPU)
                self._text_embedding_cache.update(text, embedding.detach().cpu())
                # Use the embedding directly from GPU for current forward pass
                text_embeddings[uncached_indices[idx]] = embedding
        
        # Stack all embeddings and ensure they're on the correct device
        return torch.stack(text_embeddings)

    ############ Lightning ############
    def _get_audio_embeddings(self, batch_audio: torch.Tensor) -> torch.Tensor:
        if hasattr(self.audio_encoder, "encode_batch"):
            # For speechbrain encoders (e.g., x-vector)
            audio_emb = self.audio_encoder.encode_batch(batch_audio).squeeze(1)
        else:
            # For transformers-based encoders (e.g., wav2vec) 
            input_values = self.audio_processor(
                batch_audio, 
                **self.audio_processor_kwargs).input_values.squeeze(0).to(self.device)
            audio_outputs = self.audio_encoder(input_values)
            audio_emb = audio_outputs.last_hidden_state.mean(dim=1)

        return audio_emb

    def forward(self, batch: VPC25Item) -> Dict[str, torch.Tensor]:
        """Process text/audio inputs with optimized embedding caching."""
        # Process text (with cache optimization)
        text_emb = self.get_text_embeddings(batch.text)

        # Process audio (no caching)
        audio_emb = self._get_audio_embeddings(batch.audio)

        # Fusion and classification
        fused_feats = self.fuser((audio_emb, text_emb))
        fusion_logits = self.fusion_classifier(fused_feats["fusion_emb"])
        class_prob = torch.nn.functional.softmax(fusion_logits, dim=1)
        class_preds = torch.argmax(class_prob, dim=-1)
        audio_logits = self.audio_classifier(fused_feats["audio_emb"])

        return {
            EMBEDS["TEXT"]: fused_feats["text_emb"],
            EMBEDS["AUDIO"]: fused_feats["audio_emb"],
            EMBEDS["FUSION"]: fused_feats["fusion_emb"],
            EMBEDS["CLASS"]: class_preds,
            f"{LOSS_TYPES['MAIN']}_logits": fusion_logits,
            f"{LOSS_TYPES['AUX']}_logits": audio_logits,
        }

    def model_step(self, batch: VPC25Item, criterion: Optional[Any] = None) -> Dict[str, Any]:
        """Perform a single model step."""
        outputs = self(batch)

        # Compute loss
        main_loss = (
                    criterion.main_criterion(outputs, batch.class_id)
                    if isinstance(criterion.main_criterion, EnhancedCriterion)
                    else criterion.main_criterion(outputs[f"{LOSS_TYPES['MAIN']}_logits"], batch.class_id)
                )

        aux_loss = torch.zeros_like(main_loss)
        if self._audio_embed_train:
            aux_loss = criterion.aux_criterion(outputs[f"{LOSS_TYPES['AUX']}_logits"], batch.class_id)
            total_loss = main_loss + aux_loss
        else:
            total_loss = main_loss

        return {
            LOSS_TYPES["MAIN"]: main_loss,
            LOSS_TYPES["AUX"]: aux_loss,
            "loss": total_loss, # must be called loss for lightning to track it
            "outputs": outputs}

    def on_train_start(self) -> None:
        # by default lightning executes validation step sanity checks before
        # training starts, so we need to make sure valid_metric_best doesn't store
        # accuracy from these checks
        self.valid_metric_best.reset()
        self.audio_encoder.train()
        if self.current_epoch == 0 and not self._bypass_text_warmup:
            self._warmup_cache()

    def training_step(self, batch: VPC25Item, batch_idx: int) -> Dict[str, torch.Tensor]:
        results = self.model_step(batch, self.train_criterion)
        self._log_step_metrics(results, batch, METRIC_NAMES["TRAIN"])
        return results

    def on_train_epoch_end(self) -> None:
        self.train_metric.reset()

        # Cache processing
        stats = self._text_embedding_cache.stats()
        self.log("train/cache_hit_rate", stats["hit_rate"])
        self.log("train/cache_size", len(self._text_embedding_cache))
        # resize the cache if it exceeds the max size
        if len(self._text_embedding_cache) > self._max_cache_size:
            self._text_embedding_cache.resize(self._max_cache_size)

    @torch.inference_mode()
    def on_validation_start(self) -> None:
        """Compute embeddings for eval trials."""
        self.metric_tracker = EmbeddingMetrics(self.trainer, 'valid')
        enroll_dev_loader, unique_dev_loader = self.metric_tracker.get_loaders()
        self.enrol_dev_embeds = self._compute_enrollment_embeddings(enroll_dev_loader)
        self.dev_embeds = self._compute_test_embeddings(unique_dev_loader)

    @torch.inference_mode()
    def on_test_start(self) -> None:
        """Compute embeddings for test trials."""
        self.metric_tracker = EmbeddingMetrics(self.trainer, 'test')
        enroll_test_loader, unique_test_loader = self.metric_tracker.get_loaders()
        self.enrol_embeds = self._compute_enrollment_embeddings(enroll_test_loader)
        self.test_embeds = self._compute_test_embeddings(unique_test_loader)

    @torch.inference_mode()
    def validation_step(self, batch: VPC25VerificationItem, batch_idx: int) -> None:
        """Compute EER and minDCF on validation trials"""
        self._trials_eval_step(batch, is_test=False)

    @torch.inference_mode()
    def test_step(self, batch: VPC25VerificationItem, batch_idx: int) -> None:
        """Compute EER and minDCF on these test trials"""        
        self._trials_eval_step(batch, is_test=True)

    def on_validation_epoch_end(self) -> None:
        self._epoch_end_common(is_test=False)

        # log the valid_metric_best
        valid_metric = self.valid_metric.compute()
        self.valid_metric_best.update(valid_metric)
        best_metrics_dict = self.valid_metric_best.compute()

        prefixed_metrics_best = {
            f"{self.valid_metric_best.__class__.__name__}/{METRIC_NAMES['BEST']}/{key}": value 
            for key, value in best_metrics_dict.items()
        }
        self.log_dict(prefixed_metrics_best, **self.logging_params)
        self.valid_metric.reset()

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
    def _compute_enrollment_embeddings(self, dataloader) -> dict:
        embeddings_dict = {}
        enrol_embeds = defaultdict(dict)

        with tqdm(dataloader, desc="Computing enrollment embeddings") as pbar:
            for batch in pbar:
                outputs = self(batch)
                enroll = outputs[EMBEDS["ID"]]
                # Handle frame-level embeddings (if applicable)
                if len(enroll.shape) == 3:  # [1, num_frames, embedding_dim]
                    enroll = enroll.mean(dim=1)  # [1, embedding_dim]

                # Explicitly check/create nested structure
                model_key = batch.model
                speaker_key = batch.speaker_id

                if model_key not in embeddings_dict:
                    embeddings_dict[model_key] = {}  # Create model entry
                model_dict = embeddings_dict[model_key]

                if speaker_key not in model_dict:
                    model_dict[speaker_key] = []  # Create speaker entry
                model_dict[speaker_key].append(enroll)  # Append to list

        # Aggregate embeddings by taking the mean
        for model_id, class_embeddings in embeddings_dict.items():
            for speaker_id, embeddings in class_embeddings.items():
                stacked_embeddings = torch.cat(embeddings, dim=0)
                enrol_embeds[model_id][speaker_id] = stacked_embeddings.mean(dim=0)

        return enrol_embeds

    def _compute_test_embeddings(self, dataloader) -> dict:
        embeddings_dict = {}

        with tqdm(dataloader, desc="Computing test/dev embeddings") as pbar:        
            for batch in pbar:
                outputs = self(batch)
                test = outputs[EMBEDS["ID"]]
                # Handle frame-level embeddings (if applicable)
                if len(test.shape) == 3:  # [1, num_frames, embedding_dim]
                    test = test.mean(dim=1)  # [num_frames, embedding_dim]

                embeddings_dict.update({path: emb for path, emb in zip(batch.audio_path, test)})

        return embeddings_dict

    def _trials_eval_step(self, batch, is_test: bool):
        """Common logic for test and validation steps."""
        embeds = self.test_embeds if is_test else self.dev_embeds
        enrol = self.enrol_embeds if is_test else self.enrol_dev_embeds
        metric = self.test_metric if is_test else self.valid_metric

        trial_embeddings = torch.stack([embeds[path] for path in batch.audio_path])
        enroll_embeddings = torch.stack([enrol[model][enroll_id]
                                        for model, enroll_id in zip(batch.model, batch.enroll_id)])
        scores = torch.nn.functional.cosine_similarity(enroll_embeddings, trial_embeddings)
        
        batch_dict = {
            "enrollment_id": batch.enroll_id,
            "audio_path": batch.audio_path,
            "label": batch.trial_label,
            "score": scores.tolist(),
            "model": batch.model,            
        }
        self.metric_tracker._trial_results.append(batch_dict)
        metric.update(scores, torch.tensor(batch.trial_label))

    def _epoch_end_common(self, is_test: bool) -> None:
        """Common logic for test and validation epoch end."""
        metric = self.test_metric if is_test else self.valid_metric
        enrol_embeds = self.enrol_embeds if is_test else self.enrol_dev_embeds
        trials_embeds = self.test_embeds if is_test else self.dev_embeds

        scores = pd.DataFrame([
            {
                "enrollment_id": enroll_id,
                "audio_path": audio_path,
                "label": label,
                "score": score,
                "model": model,                
            }
            for batch in self.metric_tracker._trial_results
            for enroll_id, audio_path, label, score, model in zip(
                batch["enrollment_id"],
                batch["audio_path"],
                batch["label"],
                batch["score"],
                batch["model"],
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
        """compute EER and minDCF"""
        metrics = metric.compute()
        
        postfix = METRIC_NAMES['TEST'] if is_test else METRIC_NAMES['VALID']
        prefixed_metrics = {
            f"{metric.__class__.__name__}/{postfix}/{key}": value 
            for key, value in metrics.items()
        }
        self.log_dict(prefixed_metrics, **self.logging_params)

        # log self.fuser.audio_weight and self.fuser.text_weight if they exist (NormalizedWeightedSum)
        if hasattr(self.fuser, "audio_weight") and hasattr(self.fuser, "text_weight"):
            self.log("audio_weight", self.fuser.audio_weight, **self.logging_params)
            self.log("text_weight", self.fuser.text_weight, **self.logging_params)

        # Update scores dataframe with metrics
        scores.loc[:, metrics.keys()] = [v.item() if torch.is_tensor(v) else v for v in metrics.values()]
        
        # Save scores to CSV
        stage = 'test' if is_test else 'valid'
        save_path = os.path.join(self.trainer.log_dir, f"{stage}_scores.csv")
        scores.to_csv(save_path, index=False)
        torch.save(enrol_embeds, os.path.join(self.trainer.log_dir, "enrol_embeds.pt"))
        torch.save(trials_embeds, os.path.join(self.trainer.log_dir, f"{stage}_embeds.pt"))

        figures = metric.plot_curves() or {}
        for name, fig in figures.items():
            self.log_figure_with_fallback(f"{stage}_{name}_scores", fig, step=self.current_epoch)

    def log_figure_with_fallback(self, name: str, fig: plt.Figure, step: int) -> None:
        """Log figure with fallback for loggers that don't support figure logging."""
        if hasattr(self.logger, 'experiment'):
            self.logger.experiment[f'metrics/{name}'].upload(fig)

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
            'max_size': self._text_embedding_cache.max_size,
            'hits': self._text_embedding_cache.hits,
            'misses': self._text_embedding_cache.misses,
            'contents': {
                key: tensor.cpu() 
                for key, tensor in self._text_embedding_cache._cache.items()
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
            self._text_embedding_cache = EmbeddingCache(max_size=cache_state['max_size'])
            
            # Restore performance counters
            self._text_embedding_cache.hits = cache_state['hits']
            self._text_embedding_cache.misses = cache_state['misses']
            
            # Restore cached embeddings
            for key, tensor in cache_state['contents'].items():
                self._text_embedding_cache.update(key, tensor)