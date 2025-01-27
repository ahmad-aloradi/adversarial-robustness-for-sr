import os
from typing import Any, Dict, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm
import pandas as pd

from src.datamodules.components.vpc25.vpc_dataset import VPC25Item, VPC25VerificationItem
from src import utils

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


class CrossAttentionFusion(torch.nn.Module):
    def __init__(self, audio_embedding_size, text_embedding_size, bottleneck_size, *args, **kwargs):
        super().__init__()
        self.audio_proj = torch.nn.Linear(audio_embedding_size, bottleneck_size)
        self.text_proj = torch.nn.Linear(text_embedding_size, bottleneck_size)
        self.transformer = torch.nn.TransformerEncoderLayer(d_model=bottleneck_size, nhead=8)

    def forward(self, emebds):
        assert len(emebds) == 2, f"Expected 2 embeddings, but found: {len(emebds)}"
        assert type(emebds) == tuple, f"Expected tuple of embeddings, but found: {type(emebds)}"
        audio_emb = emebds[0]
        text_emb = emebds[1]

        audio_emb = self.audio_proj(audio_emb)
        text_emb = self.text_proj(text_emb)
        combined = torch.cat((audio_emb.unsqueeze(1), text_emb.unsqueeze(1)), dim=1)
        fused = self.transformer(combined)

        return {"fusion_emb": fused.mean(dim=1),    # Mean pooling over the sequence dimension
                "audio_emb": audio_emb,
                "text_emb": text_emb}


class AttentionFusion(nn.Module):
    def __init__(self, audio_embedding_size, text_embedding_size, bottleneck_size, *args, **kwargs):
        super().__init__()
        num_heads = kwargs.get("num_heads", 4)
        self.attention = nn.MultiheadAttention(embed_dim=bottleneck_size, num_heads=num_heads)
        self.audio_proj = nn.Linear(audio_embedding_size, bottleneck_size)
        self.text_proj = nn.Linear(text_embedding_size, bottleneck_size)
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, emebds):
        assert len(emebds) == 2, f"Expected 2 embeddings, but found: {len(emebds)}"
        assert type(emebds) == tuple, f"Expected tuple of embeddings, but found: {type(emebds)}"
        audio_emb = emebds[0]
        text_emb = emebds[1]

        audio_proj = self.audio_proj(audio_emb)
        text_proj = self.text_proj(text_emb)
        
        # Scale embeddings by learnable temperature
        audio_proj = audio_proj * self.temperature
        text_proj = text_proj * self.temperature
        
        # Cross-attention between modalities
        fused_emb, _ = self.attention(audio_proj, text_proj, text_proj)
        return {"fusion_emb": fused_emb,
                "audio_emb": audio_proj,
                "text_emb": text_proj}


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
        self._freeze_pretrained_components()

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

    def _setup_training_components(self, criterion: DictConfig, optimizer: DictConfig, lr_scheduler: DictConfig
                                   ) -> None:
        """Initialize loss functions, optimizer and learning rate scheduler."""
        self.train_criterion = instantiate(criterion.train_criterion)
        self.valid_criterion = instantiate(criterion.valid_criterion)
        self.test_criterion = instantiate(criterion.test_criterion)
        if len(self.train_criterion) > 1:
            assert len(self.valid_criterion) == len(self.train_criterion), "Mismatch in number of losses"
            self._multitask_loss = True
        else:
            self._multitask_loss = False

        self.optimizer = optimizer
        self.slr_params = lr_scheduler

    def _freeze_pretrained_components(self, finetune_audioenc: bool = False) -> None:
        """Freeze pretrained components and enable training for others."""
        if hasattr(self.audio_encoder, "encode_batch"):
                finetune_audioenc = True    # Finetune for speechbrain encoders (e.g., x-vector)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.audio_encoder.parameters():
            param.requires_grad = finetune_audioenc

    def _log_step_metrics(self, results: Dict[str, Any], batch: VPC25Item, stage: str) -> None:
        """Log metrics for a single step."""
        # Log loss
        criterion = getattr(self, f"{stage}_criterion")
        main_loss = criterion.main_criterion.__class__.__name__
        
        if not self._multitask_loss:
            logged_dict = {
                f"{LOSS_TYPES['MAIN']}_{main_loss}/{stage}": results[LOSS_TYPES['MAIN']].item()
            }

        else:
            aux_loss = criterion.aux_criterion.__class__.__name__
            logged_dict = {
                f"multitask_loss/{stage}": results["loss"].item(),
                f"{LOSS_TYPES['MAIN']}_{main_loss}/{stage}": results[LOSS_TYPES['MAIN']].item(),
                f"{LOSS_TYPES['AUX']}_{aux_loss}/{stage}": results[LOSS_TYPES['AUX']].item()
                }
        
        self.log_dict(
            logged_dict,
            batch_size=getattr(self.batch_sizes, stage),
            **self.logging_params
        )

        # Log metrics
        metric = getattr(self, f"{stage}_metric")
        computed_metric = metric(results["outputs"][EMBEDS["CLASS"]], batch.class_id)

        self.log(
            f"{metric.__class__.__name__}/{stage}",
            computed_metric,
            batch_size=getattr(self.batch_sizes, stage),
            **self.logging_params,
        )

    def _compute_enrollment_embeddings(self, dataloader) -> dict:
        embeddings_dict = {}
        enrol_embeds = defaultdict(dict)

        for batch in tqdm(dataloader, desc="Computing enrollment embeddings"):
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

        for batch in tqdm(dataloader, desc="Computing test embeddings"):
            outputs = self(batch)
            test = outputs[EMBEDS["ID"]]
            # Handle frame-level embeddings (if applicable)
            if len(test.shape) == 3:  # [1, num_frames, embedding_dim]
                test = test.mean(dim=1)  # [num_frames, embedding_dim]

            embeddings_dict.update({path: emb for path, emb in zip(batch.audio_path, test)})

        return embeddings_dict

    def forward(self, batch: VPC25Item) -> Dict[str, torch.Tensor]:
        assert len(set(batch.sample_rate)) == 1, (
            "Wav2Vec2Processor expects sampling_rate of type: int, but found multiple sampling rates."
        )

        # Process audio for text input (assumming BERT-like model)
        inputs_text = self.text_processor(batch.text, **self.text_processor_kwargs).to(self.device)
        text_outputs = self.text_encoder(inputs_text.input_ids, attention_mask=inputs_text.attention_mask)
        text_emb = text_outputs.pooler_output

        # Process audio input
        if hasattr(self.audio_encoder, "encode_batch"):
            # For speechbrain encoders (e.g., x-vector)
            audio_emb = self.audio_encoder.encode_batch(batch.audio).squeeze(1)
        else:
            # For transformers-based encoders (e.g., wav2vec)
            input_values = self.audio_processor(batch.audio, **self.audio_processor_kwargs).input_values.squeeze(0).to(self.device)
            audio_outputs = self.audio_encoder(input_values)
            audio_emb = audio_outputs.last_hidden_state.mean(dim=1)

        # Combine features and get predictions
        fused_feats = self.fuser((audio_emb, text_emb))
        fusion_logits = self.fusion_classifier(fused_feats["fusion_emb"])
        class_prob = torch.nn.functional.softmax(fusion_logits, dim=1)
        class_preds = torch.argmax(class_prob, dim=-1)

        # auxillary audio classification
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
        main_loss = criterion.main_criterion(outputs[f"{LOSS_TYPES['MAIN']}_logits"], batch.class_id)
        if self._multitask_loss:
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
        self.text_encoder.train()

    def training_step(self, batch: VPC25Item, batch_idx: int) -> Dict[str, torch.Tensor]:
        results = self.model_step(batch, self.train_criterion)
        self._log_step_metrics(results, batch, METRIC_NAMES["TRAIN"])
        return results

    def validation_step(self, batch: VPC25Item, batch_idx: int) -> Dict[str, torch.Tensor]:
        results = self.model_step(batch, self.valid_criterion)
        self._log_step_metrics(results, batch, METRIC_NAMES["VALID"])
        return results

    def on_validation_epoch_end(self) -> None:
        valid_metric = self.valid_metric.compute()  # get current valid metric
        self.valid_metric_best.update(valid_metric)  # update best so far valid metric
        # log `valid_metric_best` as a value through `.compute()` method, instead
        # of as a metric object otherwise metric would be reset by lightning
        # after each epoch
        self.log(
            f"{self.valid_metric_best.__class__.__name__}/{METRIC_NAMES['BEST']}",
            self.valid_metric_best.compute(),
            **self.logging_params,
        )

    def on_test_start(self) -> None:
        """Compute embeddings for test trials."""
        enroll_loader, unique_test_loader = self.trainer.datamodule.enrollment_dataloader()
        self.enrol_embeds = self._compute_enrollment_embeddings(enroll_loader)
        self.test_embeds = self._compute_test_embeddings(unique_test_loader)

        self.scores = pd.DataFrame(columns=["enrollment_id", "audio_path", "label", "score", "model"])

    def test_step(self, batch: VPC25VerificationItem, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Compute EER and minDCF on these test trials"""        
        test_embeddings = torch.stack([self.test_embeds[path] for path in batch.audio_path])
        enroll_embeddings = torch.stack([self.enrol_embeds[model][enroll_id]
                                         for model, enroll_id in zip(batch.model, batch.enroll_id)])
        # compute similarity scores
        scores = torch.nn.functional.cosine_similarity(enroll_embeddings, test_embeddings)

        # Extend the dataframe with batch data then append it to scores dataframe
        batch_dict = {
            "score": scores.tolist(),
            "label": batch.trial_label,
            "model": batch.model,
            "enrollment_id": batch.enroll_id,
            "audio_path": batch.audio_path
        }
        self.scores = pd.concat([self.scores, pd.DataFrame(batch_dict)], ignore_index=True)

        # track test metrics
        self.test_metric.update(scores, torch.tensor(batch.trial_label))

    def on_test_epoch_end(self) -> None:
        """compute EER and minDCF"""
        metrics = self.test_metric.compute()
        self.log_dict(metrics, **self.logging_params)

        # Update scores dataframe with metrics
        self.scores.loc[:, metrics.keys()] = metrics.values()
        
        # Save scores to CSV
        save_path = os.path.join(self.trainer.log_dir, "test_scores.csv")
        self.scores.to_csv(save_path, index=False)
        torch.save(self.enrol_embeds, os.path.join(self.trainer.log_dir, "enrol_embeds.pt"))
        torch.save(self.test_embeds, os.path.join(self.trainer.log_dir, "test_embeds.pt"))

    def on_epoch_end(self) -> None:
        """Reset metrics at the end of each epoch."""
        self.train_metric.reset()
        self.valid_metric.reset()
        self.test_metric.reset()

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
    