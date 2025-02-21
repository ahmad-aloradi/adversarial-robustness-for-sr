import os
from typing import Any, Dict, Optional, List, Tuple, Literal, Callable
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto

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

from src.datamodules.components.vpc25.vpc_dataset import VPC25Item, VPC25VerificationItem, VPC25ClassCollate
from src import utils
from src.modules.components.utils import EmbeddingCache
from src.modules.metrics.metrics import AS_norm 
from src.modules.losses.components.focal_loss import FocalLoss
from datetime import datetime


log = utils.get_pylogger(__name__)


# which embedding to use for speaker ID; override when necessary
EMBED_FEATS = "fusion"

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
}

EMBEDS = {
    "TEXT": "text_embed",
    "AUDIO": "audio_embed",
    "FUSION": "fusion_embed",
    "ID": f"{EMBED_FEATS}_embed",
    "CLASS": "class_preds"
}

class EmbeddingType(Enum):
    TEXT = auto()
    AUDIO = auto()
    FUSION = auto()
    LAST_HIDDEN = auto()  # The layer before classifier

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
    def __init__(
            self, 
            classification_loss: Callable,
            temperature: float = 0.07
            ):
        super().__init__()
        self.classification_loss = classification_loss
        self.temperature = temperature
        self.focal_loss = FocalLoss(gamma=2.0)
        self.eps = 1e-8
        self.unsqueeze = True if self.classification_loss.__class__.__name__ == 'LogSoftmaxWrapper' else False
        
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
        targets_unsqeezed = targets.unsqueeze(1) if self.unsqueeze else targets
        outputs_unsqeezed = {
            k: v.unsqueeze(1) if self.unsqueeze and 'logits' in k else v
            for k, v in outputs.items()
        }

        classification_loss = self.classification_loss(outputs_unsqeezed[f"{LOSS_TYPES['FUSION']}_logits"], targets_unsqeezed)
        contr_loss = self.contrastive_loss(outputs[EMBEDS["FUSION"]], targets)
        focal_loss = self.focal_loss(outputs[f"{LOSS_TYPES['FUSION']}_logits"], targets)
        
        # Log component losses for debugging
        if torch.isnan(classification_loss) or torch.isnan(contr_loss) or torch.isnan(focal_loss):
            log.warning(f"NaN detected in losses - CE: {classification_loss.item()}, Contrastive: {contr_loss.item()}, Focal: {focal_loss.item()}")
        
        return classification_loss + 0.05 * contr_loss + 0.1 * focal_loss


class FusionClassifierWithResiduals(nn.Module):
    def __init__(
        self,
        fuse_model: nn.Module,
        input_size: int, 
        hidden_size: int, 
        num_classes: int, 
        dropout: float = 0.3,
        num_residuals: int = 2,
        norm_type: Literal['batch', 'layer'] = 'batch',
        embedding_type: EmbeddingType = EmbeddingType.FUSION
    ):
        super().__init__()
        self.embedding_type = embedding_type

        # define fuse_model
        self.fuse_model = fuse_model
        
        # Factory for normalization layers
        norm_factory: Callable = {
            'batch': nn.BatchNorm1d,
            'layer': nn.LayerNorm
        }[norm_type]
        
        self.input_norm = norm_factory(input_size)
        self.classifier = nn.Linear(input_size, num_classes)
        
        # Create residual blocks
        self.residuals = nn.ModuleList([
            self._create_residual_block(input_size, hidden_size, dropout, norm_factory)
            for _ in range(num_residuals)
        ])

    @staticmethod
    def get_embedding(fused_feats: Dict[str, torch.Tensor], 
                    last_hidden: torch.Tensor,
                    embedding_type: EmbeddingType) -> torch.Tensor:
        """Get the selected embedding type."""
        embedding_map = {
            EmbeddingType.TEXT: fused_feats["text_emb"],
            EmbeddingType.AUDIO: fused_feats["audio_emb"],
            EmbeddingType.FUSION: fused_feats["fusion_emb"],
            EmbeddingType.LAST_HIDDEN: last_hidden
        }
        assert embedding_type in embedding_map, f"Invalid embedding type: {embedding_type}"
        assert embedding_map[embedding_type].dim() == 2, f"Invalid embedding shape: {embedding_map[embedding_type].shape}"
        normalized_embeds = torch.nn.functional.normalize(embedding_map[embedding_type], p=2, dim=-1)
        return normalized_embeds

    @staticmethod
    def _create_residual_block(input_size: int, hidden_size: int, dropout: float, norm_factory: Callable
                               ) -> nn.Sequential:
        """Create a residual block with given specifications."""
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            norm_factory(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, input_size),
            norm_factory(input_size)
        )
    
    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the entire network."""
        # fuse features
        fused_feats = self.fuse_model(inputs)
        
        x = self.input_norm(fused_feats['fusion_emb'])
        for block in self.residuals:
            x = torch.nn.functional.relu(x + block(x))

        logits = self.classifier(x)
        class_prob = torch.nn.functional.softmax(logits, dim=1)
        class_preds = torch.argmax(class_prob, dim=-1)

        # Get selected embedding type
        features = FusionClassifierWithResiduals.get_embedding(fused_feats, x, self.embedding_type)

        return {
            EMBEDS["TEXT"]: fused_feats["text_emb"],
            EMBEDS["AUDIO"]: fused_feats["audio_emb"],
            EMBEDS["FUSION"]: fused_feats["audio_emb"],
            EMBEDS["ID"]: features,
            EMBEDS["CLASS"]: class_preds,
            f"fusion_logits": logits,
        }

###################################
@dataclass
class LossWeights:
    """Configurable weights for different loss components."""
    main: float = 1.0
    audio: float = 1.0
    text: float = 1.0
    fusion: float = 1.0
    consistency: float = 0.1
    confidence: float = 0.05
    smoothness: float = 0.01


class MultiModalFusionLoss(nn.Module):
    """
    Loss function for multi-modal fusion classification with multiple regularization terms.
    
    Features:
    - Individual classification losses for each modality
    - Consistency regularization between modalities
    - Confidence regularization to prevent overconfidence
    - Temporal smoothness regularization (optional)
    - Modality-specific loss weighting
    """
    
    def __init__(
        self,
        classification_loss: Callable,
        confidence_target: float = 0.9,
    ):
        super().__init__()
        self.classification_loss = classification_loss
        self.weights = LossWeights()
        self.confidence_target = confidence_target
        self.unsqueeze = True if self.classification_loss.__class__.__name__ == 'LogSoftmaxWrapper' else False
    
    def consistency_loss(
        self,
        predictions: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute consistency loss between different predictions using KL divergence.
        Uses average prediction as the target distribution.
        """
        stacked_preds = torch.stack([
            F.softmax(pred, dim=-1) for pred in predictions
        ], dim=1)
        
        mean_pred = stacked_preds.mean(dim=1, keepdim=True)
        
        kl_divs = F.kl_div(
            stacked_preds.log(),
            mean_pred.expand_as(stacked_preds),
            reduction='none'
        ).mean(dim=-1).sum(dim=-1)
        
        return kl_divs.mean()
    
    def confidence_loss(
        self,
        confidences: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Regularize prediction confidences to prevent overconfidence.
        Uses BCE loss against a target confidence value.
        """
        stacked_conf = torch.cat(confidences, dim=1)
        target_conf = torch.full_like(stacked_conf, self.confidence_target)
        
        return F.binary_cross_entropy(
            stacked_conf,
            target_conf,
            reduction='mean'
        )
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the complete loss with all components.
        
        Args:
            outputs: Dictionary containing model outputs
            targets: Ground truth labels
            
        Returns:
            Dictionary containing total loss and individual loss components
        """
        
        targets_unsqeezed = targets.unsqueeze(1) if self.unsqueeze else targets
        outputs_unsqeezed = {
            k: v.unsqueeze(1) if self.unsqueeze and 'logits' in k else v
            for k, v in outputs.items()
        }

        # Individual classification losses
        main_loss = self.classification_loss(outputs_unsqeezed["logits"], targets_unsqeezed)
        audio_loss = self.classification_loss(outputs_unsqeezed["audio_logits"], targets_unsqeezed)
        text_loss = self.classification_loss(outputs_unsqeezed["text_logits"], targets_unsqeezed)
        fusion_loss = self.classification_loss(outputs_unsqeezed["fusion_logits"], targets_unsqeezed)
        
        # Consistency between predictions
        consistency_loss = self.consistency_loss([
            outputs["audio_logits"],
            outputs["text_logits"],
            outputs["fusion_logits"]
        ])
        
        # Confidence regularization
        confidence_loss = self.confidence_loss([
            outputs["audio_confidence"],
            outputs["text_confidence"]
        ])
                
        # Combine all losses
        total_loss = (
            self.weights.main * main_loss +
            self.weights.audio * audio_loss +
            self.weights.text * text_loss +
            self.weights.fusion * fusion_loss +
            self.weights.consistency * consistency_loss +
            self.weights.confidence * confidence_loss
        )
        
        return {
            "loss": total_loss,
            "main_loss": main_loss,
            "audio_loss": audio_loss,
            "text_loss": text_loss,
            "fusion_loss": fusion_loss,
            "consistency_loss": consistency_loss,
            "confidence_loss": confidence_loss,
        }

    def update_weights(self, new_weights: LossWeights):
        """Update loss component weights during training."""
        self.weights = new_weights


class RobustFusionClassifier(nn.Module):
    """
    Classifier with:
    - Adaptive denoising with learned residual mixing
    - Modality-specific confidence estimation
    - Dynamic fusion gating
    - Simplified loss with stability enhancements
    """
    
    def __init__(self, audio_size: int, text_size: int, 
                 hidden_size: int, num_classes: int,
                 dropout_audio: float = 0.3, dropout_text: float = 0.1):
        super().__init__()
        
        # Audio processing path
        self.audio_branch = nn.Sequential(
            nn.LayerNorm(audio_size),
            nn.Linear(audio_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_audio)
        )
        
        # Denoising components
        self.denoise_transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.mix_controller = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1),
            nn.Sigmoid()
        )

        # Text processing path
        self.text_branch = nn.Sequential(
            nn.LayerNorm(text_size),
            nn.Linear(text_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_text)
        )

        # Confidence networks
        self.audio_confidence = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1),
            nn.Sigmoid()
        )
        self.text_confidence = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1),
            nn.Sigmoid()
        )

        # Classification heads
        self.audio_classifier = nn.Linear(hidden_size, num_classes)
        self.text_classifier = nn.Linear(hidden_size, num_classes)
        self.fusion_classifier = nn.Linear(hidden_size*2, num_classes)

        # Adaptive fusion gating
        self.adaptive_gate = nn.Sequential(
            nn.Linear(hidden_size * 2 + 2, hidden_size),  # *2 for text/audio,  +2 for confidences
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
            nn.Softmax(dim=-1)
            )
        
    def adaptive_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Train-time noise injection with random intensity"""
        if self.training:
            noise_scale = torch.rand(x.size(0), 1, device=x.device) * 0.2
            return x + torch.randn_like(x) * noise_scale
        return x

    def denoise_audio(self, audio_features: torch.Tensor) -> torch.Tensor:
        """Learnable residual denoising with adaptive mixing"""
        # noisy_features = self.adaptive_noise(audio_features)
        # denoised = self.denoise_transform(noisy_features)
        denoised = self.denoise_transform(audio_features)

        # Concatenate original and denoised features for mixing decision
        concat_input = torch.cat([audio_features, denoised], dim=-1)
        mix_ratio = self.mix_controller(concat_input)

        return mix_ratio * denoised + (1 - mix_ratio) * audio_features, mix_ratio

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        audio_emb, text_emb = inputs
        
        # Process modalities
        audio_features = self.audio_branch(audio_emb)
        audio_features, audio_noise_mix_ratio = self.denoise_audio(audio_features)
        text_features = self.text_branch(text_emb)
        
        # Calculate confidences
        audio_conf = self.audio_confidence(audio_features)
        text_conf = self.text_confidence(text_features)
                
        # Confidence-weighted fusion
        fusion_features = torch.cat([
            audio_features * audio_conf,
            text_features * text_conf
        ], dim=-1)

        # Predictions
        audio_logits = self.audio_classifier(audio_features)
        text_logits = self.text_classifier(text_features)
        fusion_logits = self.fusion_classifier(fusion_features)
        
        # Adaptive ensemble weighting
        gate_input = torch.cat([audio_features, text_features, audio_conf, text_conf], dim=-1)
        ensemble_weights = self.adaptive_gate(gate_input)
        
        # Final prediction with residual audio connection
        final_logits = (ensemble_weights[:, 0:1] * audio_logits +
                        ensemble_weights[:, 1:2] * text_logits +
                        ensemble_weights[:, 2:3] * fusion_logits)

        predicted_class = torch.argmax(final_logits, dim=-1)
        
        return {
            "logits": final_logits,
            "audio_logits": audio_logits,
            "text_logits": text_logits,
            "fusion_logits": fusion_logits,
            EMBEDS["CLASS"]: predicted_class,
            EMBEDS["AUDIO"]: audio_features,
            EMBEDS["TEXT"]: text_features,
            EMBEDS["FUSION"]: fusion_features,
            "audio_confidence": audio_conf,
            "text_confidence": text_conf,
            "ensemble_weights": ensemble_weights,
            "audio_noise_mix_ratio": audio_noise_mix_ratio
        }


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
    def _cohort_indices(self, 
                        speaker_col: str = 'speaker_id',
                        model_col: str = 'model',
                        model_extract_fn: callable = lambda x: x.split(os.sep)[0],
                        filepath_col: str = 'rel_filepath'):
        """
        Compute balanced cohort indices ensuring good coverage across models and speakers.
        
        Args:
            speaker_col: Name of the column containing speaker IDs
            model_col: Name of the column containing model identifiers
            model_extract_fn: Function to extract model ID from filepath
            filepath_col: Name of the column containing file paths
            
        Returns:
            List of selected indices for cohort
        """
        if not hasattr(self, '_indices'):
            df = self.trainer.datamodule.train_data.dataset
            
            # Extract model and speaker information if needed
            if model_col not in df.columns and filepath_col in df.columns:
                df[model_col] = df[filepath_col].apply(model_extract_fn)
            
            # Validate required columns
            required_cols = [speaker_col, model_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Get unique models and speakers
            unique_models = df[model_col].unique()
            unique_speakers = df[speaker_col].unique()
            min_speakers_per_model = 0.8 * len(unique_speakers)  # Require at least 80% speaker coverage
            
            # Validate speaker coverage across models
            model_speaker_coverage = {}
            for model in unique_models:
                model_speakers = df[df[model_col] == model][speaker_col].unique()
                coverage = len(model_speakers)
                model_speaker_coverage[model] = coverage
                if coverage < min_speakers_per_model:
                    print(f"WARNING: Model {model} only has {coverage} speakers out of {len(unique_speakers)} total speakers")
            
            # Calculate samples per model-speaker combination
            target_samples_per_speaker = max(1, self.cohort_per_model // len(unique_speakers))
            
            all_indices = []
            for model in unique_models:
                model_df = df[df[model_col] == model]
                model_indices = []
                
                # Sample from each speaker for this model
                for speaker in unique_speakers:
                    speaker_df = model_df[model_df[speaker_col] == speaker]
                    if len(speaker_df) > 0:
                        # Determine number of samples for this speaker
                        n_samples = min(target_samples_per_speaker, len(speaker_df))
                        
                        # Sample without replacement if possible
                        replace = n_samples > len(speaker_df)
                        sampled = speaker_df.sample(
                            n=n_samples,
                            replace=replace,
                            random_state=torch.initial_seed()
                        )
                        model_indices.extend(sampled.index.tolist())
                
                # If we haven't met our per-model quota, sample additional utterances
                remaining_samples = self.cohort_per_model - len(model_indices)
                if remaining_samples > 0:
                    # Exclude already sampled indices
                    available_indices = model_df.index.difference(model_indices)
                    if len(available_indices) > 0:
                        additional_df = df.loc[available_indices]
                        n_additional = min(remaining_samples, len(additional_df))
                        if n_additional > 0:
                            additional_samples = additional_df.sample(
                                n=n_additional,
                                replace=False,
                                random_state=torch.initial_seed()
                            )
                            model_indices.extend(additional_samples.index.tolist())
                
                all_indices.extend(model_indices)
            
            # Final validation
            final_df = df.loc[all_indices]
            final_models = final_df[model_col].unique()
            final_speakers = final_df[speaker_col].unique()
            
            print(f"Cohort statistics:")
            print(f"- Total samples: {len(all_indices)}")
            print(f"- Models represented: {len(final_models)}/{len(unique_models)}")
            print(f"- Speakers represented: {len(final_speakers)}/{len(unique_speakers)}")
            for model in final_models:
                model_samples = final_df[final_df[model_col] == model]
                print(f"- Model {model}: {len(model_samples)} samples, "
                    f"{len(model_samples[speaker_col].unique())} speakers")
            
            self._indices = all_indices
        
        return self._indices

    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        if self.stage == 'test':
            enroll_loader, unique_loader = self.trainer.datamodule.test_enrollment_dataloader()
        else:
            enroll_loader, unique_loader = self.trainer.datamodule.dev_enrollment_dataloader()
        return enroll_loader, unique_loader

    def get_cohort_loader(self) -> DataLoader:        
        return DataLoader(
            dataset=Subset(self.trainer.datamodule.train_data, self._cohort_indices),
            batch_size=self.trainer.datamodule.loaders.train.batch_size,
            num_workers=self.trainer.datamodule.loaders.train.num_workers,
            pin_memory=self.trainer.datamodule.loaders.train.pin_memory,
            collate_fn=VPC25ClassCollate(),
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
        
        # Initialize cohort embeddings for score normalization
        self.normalize_test_scores = model.get("normalize_test_scores", True)

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
        self.fusion_classifier = instantiate(model.classifiers.fusion_classifier)

    def _setup_training_components(self, criterion: DictConfig, optimizer: DictConfig, lr_scheduler: DictConfig) -> None:
        """Initialize loss functions, optimizer and learning rate scheduler."""
        self.train_criterion = instantiate(criterion.train_criterion)        
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
        
        # Log losses
        logged_dict = {
            f"{LOSS_TYPES['FUSION']}_{criterion.__class__.__name__}/{stage}": results[LOSS_TYPES['FUSION']].item()
        }
                
        self.log_dict(
            logged_dict,
            batch_size=getattr(self.batch_sizes, stage),
            **self.logging_params
        )

        # Log metrics
        metric = getattr(self, f"{stage}_metric")
        computed_metric = metric(
            results["outputs"][f"{LOSS_TYPES['FUSION']}_logits"],
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
    def _get_audio_embeddings(self, batch_audio: torch.Tensor, batch_audio_lens: torch.Tensor) -> torch.Tensor:
        if hasattr(self.audio_encoder, "encode_batch"):
            # For speechbrain encoders (e.g., x-vector)
            audio_emb = self.audio_encoder.encode_batch(wavs=batch_audio,
                                                        wav_lens=batch_audio_lens/max(batch_audio_lens)
                                                        ).squeeze(1)
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
        audio_emb = self._get_audio_embeddings(batch.audio, batch.audio_length)

        # Fuse embeddings and classify
        outputs = self.fusion_classifier((audio_emb, text_emb))

        return outputs

    def model_step(self, batch: VPC25Item, criterion: Optional[Any] = None) -> Dict[str, Any]:
        """Perform a single model step."""
        outputs = self(batch)

        # Compute loss
        if isinstance(criterion, EnhancedCriterion):
            main_loss = criterion(outputs, batch.class_id)
        elif isinstance(criterion, MultiModalFusionLoss):
            main_loss = criterion(outputs, batch.class_id)
            main_loss = main_loss['loss']
        elif criterion is not None:
            main_loss = criterion(outputs[f"fusion_logits"], batch.class_id)
        else:
            raise ValueError("Invalid criterion")
        
        return {LOSS_TYPES["FUSION"]: main_loss, "loss": main_loss, "outputs": outputs}

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

        self.metric_tracker.set_cohort_embeddings(None)

        self.enrol_dev_embeds = self._compute_enrollment_embeddings(enroll_dev_loader)
        self.dev_embeds = self._compute_test_embeddings(unique_dev_loader, mode='dev')

    @torch.inference_mode()
    def on_test_start(self) -> None:
        """Compute embeddings for test trials."""
        self.metric_tracker = EmbeddingMetrics(self.trainer, 'test')
        enroll_test_loader, unique_test_loader = self.metric_tracker.get_loaders()

        cohort_loader = self.metric_tracker.get_cohort_loader()
        if self.normalize_test_scores:
            cohort_embeddings = self._compute_cohort_embeddings(cohort_loader)
            self.metric_tracker.set_cohort_embeddings(cohort_embeddings)
        else:
            self.metric_tracker.set_cohort_embeddings(None)

        self.enrol_embeds = self._compute_enrollment_embeddings(enroll_test_loader)
        self.test_embeds = self._compute_test_embeddings(unique_test_loader, mode='test')

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

    def _compute_test_embeddings(self, dataloader, mode: str = 'test') -> dict:
        embeddings_dict = {}
        desc = f"Computing {mode} embeddings"

        with tqdm(dataloader, desc=desc) as pbar:        
            for batch in pbar:
                outputs = self(batch)
                test = outputs[EMBEDS["ID"]]
                # Handle frame-level embeddings (if applicable)
                if len(test.shape) == 3:  # [1, num_frames, embedding_dim]
                    test = test.mean(dim=1)  # [num_frames, embedding_dim]

                embeddings_dict.update({path: emb for path, emb in zip(batch.audio_path, test)})

        return embeddings_dict

    def _compute_cohort_embeddings(self, dataloader) -> dict:
        embeddings_dict = {}

        with tqdm(dataloader, desc="Computing cohort embeddings") as pbar:
            for batch in pbar:
                outputs = self(batch)
                cohort = outputs[EMBEDS["ID"]]

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

        trial_embeddings = torch.stack([embeds[path] for path in batch.audio_path])
        enroll_embeddings = torch.stack([enrol[model][enroll_id]
                                    for model, enroll_id in zip(batch.model, batch.enroll_id)])
        cohort_embeddings = metric.cohort_embeddings

        # Compute raw cosine similarity scores
        raw_scores = torch.nn.functional.cosine_similarity(enroll_embeddings, trial_embeddings)
        
        if cohort_embeddings is not None:
            normalized_scores = []
            for i, (enroll_emb, test_emb, model) in enumerate(zip(enroll_embeddings, trial_embeddings, batch.model)):
                raw_score = raw_scores[i]
                
                # Get model-specific cohort embeddings
                model_cohort = cohort_embeddings.get(model)
                assert model_cohort is not None, f"No cohort embeddings found for model {model}"
                if isinstance(model_cohort, dict):
                    model_cohort = torch.stack(list(model_cohort.values()))
                if model_cohort.ndim != 2:
                    raise ValueError(f"Invalid cohort embeddings shape for model {model}: {model_cohort.shape}")
                
                # Apply AS-Norm
                norm_score = AS_norm(score=raw_score,
                                     enroll_embedding=enroll_emb,
                                     test_embedding=test_emb, 
                                     cohort_embeddings=model_cohort, topk=300)
                normalized_scores.append(norm_score)
            
            # Convert back to tensor
            normalized_scores = torch.tensor(normalized_scores, device=raw_scores.device)
        
        else:
            normalized_scores = raw_scores.clone()

        # Update metric with normalized scores
        metric.update(scores=normalized_scores, labels=torch.tensor(batch.trial_label))
        
        batch_dict = {
            "enrollment_id": batch.enroll_id,
            "audio_path": batch.audio_path,
            "label": batch.trial_label,
            "score": raw_scores.detach().cpu().tolist(),
            "norm_score": normalized_scores.detach().cpu().tolist(),
            "model": batch.model,
        }
        self.metric_tracker._trial_results.append(batch_dict)

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
                "norm_score": norm_score,
                "model": model,                
            }
            for batch in self.metric_tracker._trial_results
            for enroll_id, audio_path, label, score, norm_score, model in zip(
                batch["enrollment_id"],
                batch["audio_path"],
                batch["label"],
                batch["score"],
                batch["norm_score"],
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
            f"{metric.__class__.__name__}/{postfix}/{key}": value for key, value in metrics.items()
        }
        self.log_dict(prefixed_metrics, **self.logging_params)

        # log self.fuser.audio_weight and self.fuser.text_weight if they exist (NormalizedWeightedSum)
        fuse_model = getattr(self.fusion_classifier, "fuse_model", None)
        if fuse_model is not None and hasattr(fuse_model, "audio_weight") and hasattr(fuse_model, "text_weight"):
            self.log("audio_weight", self.fusion_classifier.fuse_model.audio_weight, **self.logging_params)
            self.log("text_weight", self.fusion_classifier.fuse_model.text_weight, **self.logging_params)

        # Update scores dataframe with metrics
        scores.loc[:, metrics.keys()] = [v.item() if torch.is_tensor(v) else v for v in metrics.values()]
        
        # Save scores to CSV
        stage = 'test' if is_test else 'valid'
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifacts_dir = os.path.join(self.trainer.default_root_dir, f'test_artifacts_{timestamp}')
        os.makedirs(artifacts_dir, exist_ok=True)

        scores.to_csv(os.path.join(artifacts_dir, f"{stage}_scores.csv"), index=False)
        torch.save(enrol_embeds, os.path.join(artifacts_dir, f"{stage}_enrol_embeds.pt"))
        torch.save(trials_embeds, os.path.join(artifacts_dir, f"{stage}_embeds.pt"))
        if metric.cohort_embeddings is not None:
            torch.save(metric.cohort_embeddings, os.path.join(artifacts_dir, f"{stage}_cohort_embeds.pt"))

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