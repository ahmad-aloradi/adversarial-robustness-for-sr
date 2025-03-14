import os
import json
from typing import Any, Dict, Optional, List, Tuple, Literal, Callable, Union
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
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
class NormalizedWeightedSum(nn.Module):
    def __init__(self, audio_embedding_size, text_embedding_size, hidden_size):
        super().__init__()
        # Projection layers
        self.audio_proj = nn.Sequential(
            nn.LayerNorm(audio_embedding_size),
            nn.Linear(audio_embedding_size, hidden_size),
            nn.ReLU(inplace=True)
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_embedding_size),
            nn.Linear(text_embedding_size, hidden_size),
            nn.ReLU(inplace=True)
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


class FusionClassifierWithResiduals(nn.Module):
    def __init__(
        self,
        fuse_model: nn.Module,
        input_size: int, 
        hidden_size: int, 
        num_classes: int,
        dropout_residual: float = 0.1,
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
        
        # Audio & Text classifiers
        self.audio_classifier = nn.Linear(input_size, num_classes)
        self.text_classifier = nn.Linear(input_size, num_classes)
        
        # fusion classifier
        self.fusion_norm = norm_factory(input_size)
        self.residuals = nn.ModuleList([
            self._create_residual_block(input_size, hidden_size, dropout_residual, norm_factory)
            for _ in range(num_residuals)
        ])
        self.fusion_classifier = nn.Linear(input_size, num_classes)

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

        # classify audio and text features
        audio_logits = self.audio_classifier(fused_feats["audio_emb"])
        text_logits = self.text_classifier(fused_feats["text_emb"])

        # fusion classifier        
        x = self.fusion_norm(fused_feats['fusion_emb'])
        for block in self.residuals:
            x = torch.nn.functional.relu(x + block(x))
        fusion_logits = self.fusion_classifier(x)
        class_prob = torch.nn.functional.softmax(fusion_logits, dim=1)
        class_preds = torch.argmax(class_prob, dim=-1)

        # Get selected embedding type
        features = FusionClassifierWithResiduals.get_embedding(fused_feats, x, self.embedding_type)

        return {
            EMBEDS["TEXT"]: fused_feats["text_emb"],
            EMBEDS["AUDIO"]: fused_feats["audio_emb"],
            EMBEDS["FUSION"]: fused_feats["fusion_emb"],
            EMBEDS["ID"]: features,
            EMBEDS["CLASS"]: class_preds,
            f"fusion_logits": fusion_logits,
            f"audio_logits": audio_logits,
            f"text_logits": text_logits
        }


class RobustFusionClassifier(nn.Module):
    """
    Classifier with:
    - Adaptive denoising with learned residual mixing
    - Modality-specific confidence estimation
    - Dynamic fusion gating
    - Simplified loss with stability enhancements
    """
    
    def __init__(self,
                 audio_embedding_size: int, 
                 text_embedding_size: int, 
                 hidden_size: int,
                 num_classes: int,
                 dropout_audio: float = 0.3,
                 dropout_text: float = 0.1,
                 accum_method: Literal['sum', 'concat'] = 'sum',
                 norm_type: Literal['batch', 'layer'] = 'batch',
                 embedding_type: EmbeddingType = EmbeddingType.FUSION
                 ):
        super().__init__()
        
        # Setup hidden_size based off accumulation method
        assert accum_method in ['sum', 'concat'], f"Invalid accumulation method: {accum_method}"
        fusion_dependent_hidden = 2 * hidden_size if accum_method == 'concat' else hidden_size
        self.accum_method = accum_method
        self.embedding_type = embedding_type
        
        # Factory for normalization layers
        norm_factory: Callable = {
            'batch': nn.BatchNorm1d,
            'layer': nn.LayerNorm
        }[norm_type]

        # Distortion estimator
        self.distortion_estimator = nn.Sequential(
            nn.Linear(audio_embedding_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4)  # Distortion vector
        )
        
        # Audio branch conditioned on distortion
        self.audio_branch = nn.Sequential(
            norm_factory(audio_embedding_size + hidden_size // 4),  # Input size includes distortion vector
            nn.Linear(audio_embedding_size + hidden_size // 4, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_audio)
        )        

        # Text processing path
        self.text_branch = nn.Sequential(
            norm_factory(text_embedding_size),
            nn.Linear(text_embedding_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_text)
        )

        # Confidence networks
        self.audio_confidence = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        self.text_confidence = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

        # Classification heads
        self.audio_classifier = nn.Linear(hidden_size, num_classes)
        self.text_classifier = nn.Linear(hidden_size, num_classes)
        self.fusion_classifier = nn.Linear(fusion_dependent_hidden, num_classes)

        # Adaptive fusion gating
        self.adaptive_gate = nn.Sequential(
            nn.Linear(2 * hidden_size + 2, hidden_size),  # *2 for text/audio,  +2 for confidences
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
            nn.Softmax(dim=-1)
            )

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

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        audio_emb, text_emb = inputs
        
        # Process modalities
        # 1. process audio
        distortion_estimate = self.distortion_estimator(audio_emb)
        audio_input = torch.cat([audio_emb, distortion_estimate], dim=-1)
        audio_features = self.audio_branch(audio_input)        
        # 2. process text
        text_features = self.text_branch(text_emb)
        
        # Calculate confidences
        audio_conf = self.audio_confidence(audio_features)
        text_conf = self.text_confidence(text_features)
                
        # Confidence-weighted fusion
        if self.accum_method == 'sum':
            fusion_features = audio_features * audio_conf + text_features * text_conf
        elif self.accum_method == 'concat':
            fusion_features = torch.cat((audio_features * audio_conf, text_features * text_conf), dim=-1)

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

        class_prob = torch.nn.functional.softmax(final_logits, dim=1)
        predicted_class = torch.argmax(class_prob, dim=-1)

        # get representation
        fused_feats = {'text_emb': text_features, 'audio_emb': audio_features, 'fusion_emb': fusion_features}
        features = RobustFusionClassifier.get_embedding(fused_feats, last_hidden=fusion_features, embedding_type=self.embedding_type)
        
        return {
            "ensemble_logits": final_logits,
            "audio_logits": audio_logits,
            "text_logits": text_logits,
            "fusion_logits": fusion_logits,
            EMBEDS["CLASS"]: predicted_class,
            EMBEDS["AUDIO"]: audio_features,
            EMBEDS["TEXT"]: text_features,
            EMBEDS["FUSION"]: fusion_features,
            EMBEDS["ID"]: features,
            "audio_confidence": audio_conf,
            "text_confidence": text_conf,
            "ensemble_weights": ensemble_weights,
        }

###################################
@dataclass
class LossWeights:
    """Configurable weights for different loss components"""
    ensemble: float = 1.0
    fusion: float = 1.0
    audio: float = 0.2
    text: float = 0.2
    contrastive: float = 0.1
    consistency: float = 0.1
    confidence: float = 0.1

    @classmethod
    def from_dict(cls, weights_dict: Dict[str, float]) -> 'LossWeights':
        """Create LossWeights from a dictionary"""
        return cls(**{k: v for k, v in weights_dict.items() if hasattr(cls, k)})


class MultiModalLoss(nn.Module):
    """
    Loss function for multi-modal fusion with multiple regularization terms.
    Supports different classifier architectures with adaptive loss components.
    """
    def __init__(
        self,
        classification_loss: Callable,
        classifier_name: Literal['normalized', 'robust'],
        weights: Optional[LossWeights] = None,
        confidence_target: float = 0.9,
        weight_scheduler: Optional[Callable] = None,
        return_dict: bool = True
    ):
        super().__init__()
        # Core configuration
        self.classification_loss = classification_loss
        self.classifier_name = classifier_name
        self.weights = weights or LossWeights()
        self.confidence_target = confidence_target
        self.weight_scheduler = weight_scheduler
        self.return_dict = return_dict
        
        # Validation
        if classifier_name not in ['normalized', 'robust']:
            raise ValueError(f"Invalid classifier name: {classifier_name}. Expected 'normalized' or 'robust'")
            
        # Performance optimization
        self.eps = 1e-8
        self._current_epoch = 0
        self.unsqueeze = classification_loss.__class__.__name__ == 'LogSoftmaxWrapper'
        
        # Pre-compute constants
        self.embed_keys = {EMBEDS[modality].lower(): modality.lower() for modality in ["TEXT", "AUDIO", "FUSION"]}
                          
        # Define classifier-specific configurations
        self.key_configs = {
            'normalized': {
                'logits_patterns': [(r'(audio|text|fusion)_logits', lambda m: m.group(1))],
                'embedding_prefixes': list(self.embed_keys.keys()),
                'active_weights': {'audio', 'text', 'fusion', 'contrastive', 'consistency'}
            },
            'robust': {
                'logits_patterns': [ (r'(ensemble|audio|text|fusion)_logits', lambda m: m.group(1))],
                'embedding_prefixes': list(self.embed_keys.keys()) + [r'(audio|text|fusion)_features'],
                'confidence_pattern': r'(audio|text)_confidence',
                'active_weights': {'ensemble', 'audio', 'text', 'fusion', 'contrastive', 'consistency', 'confidence'}
            }
        }

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for weight scheduling."""
        self._current_epoch = epoch
        if self.weight_scheduler is not None:
            new_weights = self.weight_scheduler(epoch)
            if isinstance(new_weights, dict):
                self.weights = LossWeights.from_dict(new_weights)
            elif isinstance(new_weights, LossWeights):
                self.weights = new_weights

    def cosine_similarity_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity matrix with gradient clipping for stability."""
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.T)
        similarity_matrix = torch.clamp(similarity_matrix, min=-1.0 + self.eps, max=1.0 - self.eps)
        # return similarity_matrix / self.temperature
        return similarity_matrix

    def contrastive_loss(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute supervised contrastive loss with improved stability."""
        # Early return for small batches
        if embeddings.size(0) <= 1:
            return torch.tensor(0.0, device=embeddings.device)
            
        similarity_matrix = self.cosine_similarity_matrix(embeddings)
        
        # Create masks for positive and negative pairs
        batch_size = targets.size(0)
        labels_matrix = targets.unsqueeze(1).expand(batch_size, batch_size)
        positive_mask = labels_matrix == labels_matrix.T
        positive_mask.fill_diagonal_(False)
        negative_mask = ~positive_mask
        negative_mask.fill_diagonal_(False)
        
        # Check for valid samples (with both positive and negative pairs)
        valid_samples = (positive_mask.sum(dim=1) > 0) & (negative_mask.sum(dim=1) > 0)
        if not valid_samples.any():
            return torch.tensor(0.0, device=embeddings.device)
            
        # Calculate loss using log-sum-exp for numerical stability
        pos_similarities = similarity_matrix.masked_fill(~positive_mask, -1e9)
        neg_similarities = similarity_matrix.masked_fill(~negative_mask, -1e9)
        
        pos_logits = torch.logsumexp(pos_similarities, dim=1)
        neg_logits = torch.logsumexp(neg_similarities, dim=1)
        
        # Apply loss only to valid samples
        per_sample_loss = (-pos_logits + neg_logits) * valid_samples.float()
        n_valid = valid_samples.sum().item()
        
        return per_sample_loss.sum() / max(n_valid, 1)

    def consistency_loss(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        if len(predictions) < 2:
            return torch.tensor(0.0, device=predictions[0].device)
        
        # 1. Compute log-probabilities safely with log_softmax
        log_probs = [F.log_softmax(pred, dim=-1) for pred in predictions]  # (batch, num_classes)
        
        # 2. Convert to probabilities (clamped to avoid underflow)
        probs = [log_prob.exp().clamp(min=1e-8) for log_prob in log_probs]  # (batch, num_classes)
        stacked_probs = torch.stack(probs, dim=1)  # (batch, num_modalities, num_classes)
        
        # 3. Compute mean probability distribution
        mean_probs = stacked_probs.mean(dim=1, keepdim=True)  # (batch, 1, num_classes)
        mean_probs = mean_probs.clamp(min=1e-8)  # Avoid log(0)
        
        # 4. Compute KL divergence safely
        log_mean_probs = torch.log(mean_probs)  # (batch, 1, num_classes)
        
        # Expand to match stacked_probs shape
        log_mean_probs_expanded = log_mean_probs.expand_as(stacked_probs)
        
        # KL(p_i || mean_p) = sum(p_i * (log(p_i) - log(mean_p)))
        kl_divs = F.kl_div(
            input=log_mean_probs_expanded,  # log(mean_p)
            target=stacked_probs,           # p_i
            reduction='none'
        ).sum(dim=-1)  # Sum over classes
        
        return kl_divs.mean()  # Average over batch and modalities

    def confidence_loss(self, confidences: List[torch.Tensor]) -> torch.Tensor:
        """Regularize prediction confidences to prevent overconfidence."""
        if not confidences:
            return torch.tensor(0.0, device=confidences[0].device)
            
        stacked_conf = torch.cat(confidences, dim=1)
        target_conf = torch.full_like(stacked_conf, self.confidence_target)
        
        return F.binary_cross_entropy(stacked_conf, target_conf, reduction='mean')

    def _prepare_inputs(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor):
        """Prepare inputs by handling unsqueezing."""
        # Handle target unsqueezing
        targets_unsqueezed = targets.unsqueeze(1) if self.unsqueeze and targets.dim() == 1 else targets
        
        # Handle outputs unsqueezing
        outputs_unsqueezed = {}
        for k, v in outputs.items():
            needs_unsqueeze = (
                self.unsqueeze and 
                ('logits' in k.lower() or k == EMBEDS["CLASS"]) and 
                v.dim() == 2
            )
            outputs_unsqueezed[k] = v.unsqueeze(1) if needs_unsqueeze else v
            
        return outputs_unsqueezed, targets_unsqueezed

    def _extract_keys(self, outputs: Dict[str, torch.Tensor]) -> Tuple[Dict, Dict, Dict]:
        """Extract relevant keys based on classifier type."""
        config = self.key_configs[self.classifier_name]
        
        logits_keys = {}
        embedding_keys = {}
        confidence_keys = {}
        
        # Process keys
        for key in outputs:
            key_lower = key.lower()
            
            # Extract logits keys
            for pattern, name_fn in config.get('logits_patterns', []):
                if callable(name_fn):
                    import re
                    match = re.match(pattern, key_lower)
                    if match:
                        logits_keys[name_fn(match)] = key
                elif key_lower == pattern:
                    logits_keys[name_fn] = key
            
            # Extract embedding keys
            if key_lower in self.embed_keys:
                embedding_keys[self.embed_keys[key_lower]] = key
            
            # Extract confidence keys (robust classifier only)
            if self.classifier_name == 'robust' and key_lower.endswith('_confidence'):
                modality = key_lower[:-11]  # Remove '_confidence'
                if modality in ['audio', 'text']:
                    confidence_keys[modality] = key
                    
        # Determine active weights
        active_weights = set(config['active_weights'])
        if 'consistency' in active_weights and len(logits_keys) < 2:
            active_weights.remove('consistency')
        if 'confidence' in active_weights and not confidence_keys:
            active_weights.remove('confidence')
            
        return logits_keys, embedding_keys, confidence_keys, active_weights

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute loss based on classifier type and outputs."""
        # Prepare inputs
        outputs_unsqueezed, targets_unsqueezed = self._prepare_inputs(outputs, targets)
        
        # Extract keys and determine active weights
        logits_keys, embedding_keys, confidence_keys, active_weights = self._extract_keys(outputs)
        
        # Initialize losses container
        losses = {}
        
        # Compute classification losses for all modalities
        for modality, key in logits_keys.items():
            weight = getattr(self.weights, modality, 0)
            if modality in active_weights and weight > 0:
                losses[f"{modality}_loss"] = self.classification_loss(
                    outputs_unsqueezed[key], 
                    targets_unsqueezed
                )
        
        # Compute contrastive loss if applicable
        if (embedding_keys and "contrastive" in active_weights and 
                getattr(self.weights, "contrastive", 0) > 0):
            embed_key = embedding_keys.get("fusion")
            if embed_key:
                losses["contrastive_loss"] = self.contrastive_loss(
                    outputs[embed_key], 
                    targets
                )
        
        # Compute consistency loss if applicable
        if (len(logits_keys) >= 2 and "consistency" in active_weights and 
                getattr(self.weights, "consistency", 0) > 0):
            # Get all logits except for ensemble_logits
            logits_list = [outputs[key] for key in logits_keys.values() if 'ensemble_logits' not in key]
            losses["consistency_loss"] = self.consistency_loss(logits_list)
        
        # Compute confidence loss if applicable
        if (confidence_keys and "confidence" in active_weights and 
                getattr(self.weights, "confidence", 0) > 0):
            confidence_list = [outputs[key] for key in confidence_keys.values()]
            losses["confidence_loss"] = self.confidence_loss(confidence_list)
        
        # Combine all losses with weights
        total_loss = torch.tensor(0.0, device=targets.device, requires_grad=True)
        for loss_name, loss_value in losses.items():
            weight_attr = loss_name.split("_")[0]
            weight = getattr(self.weights, weight_attr, 0.0)
            if weight > 0:
                total_loss = total_loss + weight * loss_value
        
        # Handle potential NaN/Inf values
        if not torch.isfinite(total_loss):
            # Try to use reliable fallback losses
            for fallback in ["main_loss", "fusion_loss"]:
                if fallback in losses and torch.isfinite(losses[fallback]):
                    total_loss = losses[fallback]
                    break
            else:
                # Last resort fallback
                total_loss = torch.tensor(1.0, device=targets.device, requires_grad=True)
        
        # Add total loss to dictionary
        losses["loss"] = total_loss
        
        return losses if self.return_dict else total_loss
    
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
            f"{stage}/{LOSS_TYPES['FUSION']}_{criterion.__class__.__name__}": results[LOSS_TYPES['FUSION']].item()
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
        # Get unique texts from training data
        unique_texts = list(set(self.trainer.datamodule.train_data.dataset.text))
        
        # Define batch size for processing
        batch_size = 384
        
        # Optional: Limit to subset of texts for faster startup
        max_texts = batch_size * 10  # Uncomment to process only 2 batches
        unique_texts = unique_texts[:max_texts]
        
        # Process texts in batches with progress bar
        with torch.no_grad():
            with tqdm(total=len(unique_texts), desc="Warming up cache") as pbar:
                for i in range(0, len(unique_texts), batch_size):
                    batch_texts = unique_texts[i: i + batch_size]
                    # Get embeddings and immediately delete the tensor since we only need the cached values
                    _ = self.get_text_embeddings(batch_texts)
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
        if isinstance(criterion, MultiModalLoss):
            loss = criterion(outputs, batch.class_id)
            main_loss = loss["loss"] if isinstance(loss, dict) else loss
        elif isinstance(criterion, torch.nn.CrossEntropyLoss):
            main_loss = criterion(outputs[f"fusion_logits"], batch.class_id)
        elif criterion.__class__.__name__ == 'LogSoftmaxWrapper':
            main_loss = criterion(outputs[f"fusion_logits"].unsqueeze(1), batch.class_id.unsqueeze(1))
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
        
        if batch_idx % 3000 == 0:
            torch.cuda.empty_cache()

        self._log_step_metrics(results, batch, METRIC_NAMES["TRAIN"])
        return results

    def on_train_epoch_end(self) -> None:
        self.train_metric.reset()

        # Cache processing
        stats = self._text_embedding_cache.stats()
        self.log("train/cache/cache_hit_rate", stats["hit_rate"])
        self.log("train/cache/cache_size", len(self._text_embedding_cache))
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
        valid_metric = self.valid_metric.compute()
        self.valid_metric_best.update(valid_metric)
        best_metrics_dict = self.valid_metric_best.compute()

        self._epoch_end_common(is_test=False)
        torch.cuda.empty_cache()  # Clear CUDA cache if using GPU

        # Log the best metrics
        prefixed_metrics_best = {
            f"{METRIC_NAMES['BEST']}/{self.valid_metric_best.__class__.__name__}/{key}": value 
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
        """Compute EER and minDCF, handle logging and saving of artifacts."""
        # Compute metrics (EER, minDCF, etc.)
        metrics = metric.compute()
        
        # Log metrics with appropriate prefix (test or valid)
        stage = METRIC_NAMES['TEST'] if is_test else METRIC_NAMES['VALID']
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

        # Validation-specific logic: Check if this is the best validation epoch
        if not is_test:
            best_metrics = self.valid_metric_best.compute()
            current_eer = metrics.get(self.valid_metric_best.target_key, float('inf'))
            best_eer = best_metrics.get(self.valid_metric_best.target_key, float('inf'))
            
            # Save best validation scores, embeddings, and binary metrics plots
            if current_eer == best_eer:
                for name, fig in figures.items():
                    self.log_figure_with_fallback(f"best_valid_{name}_scores", fig, stage=stage, step=self.current_epoch)
                
                scores.to_csv(os.path.join(artifacts_dir, "best_valid_scores.csv"), index=False)
                torch.save(enrol_embeds, os.path.join(artifacts_dir, "best_valid_enrol_embeds.pt"))
                torch.save(trials_embeds, os.path.join(artifacts_dir, "best_valid_embeds.pt"))
                if metric.cohort_embeddings is not None:
                    torch.save(metric.cohort_embeddings, os.path.join(artifacts_dir, "best_valid_cohort_embeds.pt"))
 
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