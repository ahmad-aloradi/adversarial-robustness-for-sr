from typing import Any, Dict, Callable, Union, List, Tuple, Optional, Literal
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.losses.components.focal_loss import FocalLoss
from src.modules.constants import LOSS_TYPES, EMBEDS
from src import utils

log = utils.get_pylogger(__name__)


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


class EnhancedCriterion(nn.Module):
    def __init__(
            self, 
            classification_loss: Callable,
            temperature: float = 0.07,
            contrastive_weight: float = 0.05,
            focal_weight: float = 0.1
            ):
        super().__init__()
        self.classification_loss = classification_loss
        self.temperature = temperature
        self.focal_loss = FocalLoss(gamma=2.0)
        self.eps = 1e-8
        self.unsqueeze = True if self.classification_loss.__class__.__name__ == 'LogSoftmaxWrapper' else False
        self.contrastive_weight = contrastive_weight
        self.focal_weight = focal_weight
        
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
            
        # Check for NaN or Inf values and handle them
        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            log.warning("NaN or Inf detected in embeddings for contrastive loss")
            embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
            
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
        # Input validation
        for key, tensor in outputs.items():
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                log.warning(f"NaN or Inf detected in output tensor '{key}'")
                outputs[key] = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Prepare inputs for classification loss
        targets_unsqueezed = targets.unsqueeze(1) if self.unsqueeze else targets
        
        # Handle output formatting based on loss requirements
        fusion_logits_key = f"{LOSS_TYPES['FUSION']}_logits"
        fusion_embeds_key = EMBEDS["FUSION"]

        # Ensure keys exist in outputs
        if fusion_logits_key not in outputs:
            log.error(f"Missing key {fusion_logits_key} in outputs")
            return torch.tensor(float('nan'), device=targets.device)
        if fusion_embeds_key not in outputs:
            log.error(f"Missing key {fusion_embeds_key} in outputs")
            return torch.tensor(float('nan'), device=targets.device)
            
        # Prepare outputs with appropriate shapes
        outputs_unsqueezed = {}
        for k, v in outputs.items():
            if self.unsqueeze and 'logits' in k:
                outputs_unsqueezed[k] = v.unsqueeze(1)
            else:
                outputs_unsqueezed[k] = v
        
        # Compute classification loss
        try:
            if self.classification_loss.__class__.__name__ == 'BCEWithLogitsLoss' or \
               self.classification_loss.__class__.__name__ == 'BCELoss':
                # If using BCE loss, ensure inputs are in valid range
                fusion_logits = outputs_unsqueezed[fusion_logits_key]
                
                # If BCE (not BCE with logits), apply sigmoid and clamp
                if self.classification_loss.__class__.__name__ == 'BCELoss':
                    fusion_logits = torch.sigmoid(fusion_logits)
                    fusion_logits = torch.clamp(fusion_logits, min=self.eps, max=1.0-self.eps)
                    outputs_unsqueezed[fusion_logits_key] = fusion_logits
                    
            classification_loss = self.classification_loss(outputs_unsqueezed[fusion_logits_key], targets_unsqueezed)
        except Exception as e:
            log.error(f"Error in classification loss: {e}")
            classification_loss = torch.tensor(0.0, device=targets.device)
            
        # Compute contrastive loss
        try:
            contr_loss = self.contrastive_loss(outputs[fusion_embeds_key], targets)
        except Exception as e:
            log.error(f"Error in contrastive loss: {e}")
            contr_loss = torch.tensor(0.0, device=targets.device)
        
        # Compute focal loss
        try:
            # For focal loss, ensure logits are properly formatted
            focal_inputs = outputs[fusion_logits_key]
            if isinstance(self.focal_loss, FocalLoss) and hasattr(self.focal_loss, 'binary') and self.focal_loss.binary:
                # If binary focal loss, ensure proper sigmoid and clamping
                focal_inputs = torch.sigmoid(focal_inputs)
                focal_inputs = torch.clamp(focal_inputs, min=self.eps, max=1.0-self.eps)
                
            focal_loss = self.focal_loss(focal_inputs, targets)
        except Exception as e:
            log.error(f"Error in focal loss: {e}")
            focal_loss = torch.tensor(0.0, device=targets.device)
        
        # Log component losses for debugging
        if torch.isnan(classification_loss) or torch.isnan(contr_loss) or torch.isnan(focal_loss):
            log.warning(f"NaN detected in losses - CE: {classification_loss.item() if not torch.isnan(classification_loss) else 'NaN'}, "
                        f"Contrastive: {contr_loss.item() if not torch.isnan(contr_loss) else 'NaN'}, "
                        f"Focal: {focal_loss.item() if not torch.isnan(focal_loss) else 'NaN'}")
            
            # Return only the losses that aren't NaN
            total_loss = torch.tensor(0.0, device=targets.device)
            if not torch.isnan(classification_loss):
                total_loss = total_loss + classification_loss
            if not torch.isnan(contr_loss) and self.contrastive_weight > 0:
                total_loss = total_loss + self.contrastive_weight * contr_loss
            if not torch.isnan(focal_loss) and self.focal_weight > 0:
                total_loss = total_loss + self.focal_weight * focal_loss
                
            # If all losses are NaN, return a small constant to avoid breaking the backward pass
            if torch.isnan(total_loss):
                log.error("All losses are NaN, returning small constant")
                return torch.tensor(1e-5, device=targets.device, requires_grad=True)
                
            return total_loss
        
        # Combine losses with weights
        total_loss = classification_loss + self.contrastive_weight * contr_loss + self.focal_weight * focal_loss
        
        # Final sanity check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            log.error("Final loss is NaN or Inf, returning small constant")
            return torch.tensor(1e-5, device=targets.device, requires_grad=True)
            
        return total_loss


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
        contrastive_temprature: float = 1.0,
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
        self.contrastive_temprature = contrastive_temprature
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
        return similarity_matrix / self.contrastive_temprature

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
