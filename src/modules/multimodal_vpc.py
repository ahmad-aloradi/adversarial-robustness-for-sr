import os
from typing import Any, Dict, Optional
from collections import defaultdict

import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm
import pandas as pd
import speechbrain as sb

from src.datamodules.components.vpc25.vpc_dataset import VPC25Item, VPC25VerificationItem
from src import utils
from src.modules.metrics import load_metrics

log = utils.get_pylogger(__name__)

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
    "TOTAL": "total"
}

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
        main_metric, valid_metric_best, add_metrics = load_metrics(metrics)
        self.train_metric = main_metric.clone()
        self.valid_metric = main_metric.clone()
        self.valid_metric_best = valid_metric_best.clone()
        self.test_metric = main_metric.clone()
        self.test_add_metrics = add_metrics.clone(postfix=f"/{METRIC_NAMES['TEST']}")

    def _setup_model_components(self, model: DictConfig) -> None:
        """Initialize encoders and classifiers."""
        # Text processing
        self.text_processor = instantiate(model.text_processor)
        self.text_encoder = instantiate(model.text_encoder)
        self.text_projection = instantiate(model.classifiers.text_classifier)
        self.text_processor_kwargs = model.text_processor_kwargs
        
        # Audio processing
        self.audio_encoder = instantiate(model.audio_encoder)
        self.audio_projection = instantiate(model.classifiers.audio_classifier)
        self.audio_embeds_projection = instantiate(model.classifiers.audio_embeds_classifier)
        
        # Fusion and classification
        self.fusion_classifier = instantiate(model.classifiers.fusion_classifier)
        
    def _setup_training_components(
        self, 
        criterion: DictConfig, 
        optimizer: DictConfig,
        lr_scheduler: DictConfig
    ) -> None:
        """Initialize loss functions, optimizer and learning rate scheduler."""
        self.criterion_train = instantiate(criterion.criterion_train)
        self.criterion_val = instantiate(criterion.criterion_val)
        self.criterion_test = instantiate(criterion.criterion_test)
        self.optimizer = optimizer
        self.slr_params = lr_scheduler

    def _freeze_pretrained_components(self, finetune_audioenc: bool = True) -> None:
        """Freeze pretrained components and enable training for others."""
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.audio_encoder.parameters():
            param.requires_grad = finetune_audioenc

    def forward(self, batch: VPC25Item) -> Dict[str, torch.Tensor]:
        assert len(set(batch.sample_rate)) == 1, (
            "Wav2Vec2Processor expects sampling_rate of type: int, but found multiple sampling rates."
        )

        # Process audio for text input
        encoded_text = self.text_processor(batch.audio, **self.text_processor_kwargs).input_values
        encoded_text = encoded_text.squeeze(0).to(batch.audio.device)
        # Process text input
        text_embeddings = self.text_encoder(encoded_text).logits
        text_speaker_logits = self._agg_text_pred(text_embeddings, method='mean', keepdim=False)
        text_features = self.text_projection(text_speaker_logits)

        # Process audio input
        audio_embeddings = self.audio_encoder.encode_batch(batch.audio).squeeze(1)
        audio_features = self.audio_projection(audio_embeddings)
        audio_embeddings = self.audio_embeds_projection(audio_embeddings)

        # Combine features and get predictions
        combined_features = torch.cat((audio_features, text_features), dim=-1)
        fusion_logits = self.fusion_classifier(combined_features)

        return {
            "text_features": text_features,
            "audio_features": audio_embeddings,
            "fusion_logits": fusion_logits,
        }

    def model_step(self, batch: VPC25Item, criterion: Optional[Any] = None) -> Dict[str, Any]:
        """Perform a single model step."""
        outputs = self(batch)

        # Compute loss
        audio_criterion = criterion.audio_criterion(outputs["audio_features"], batch.class_id)
        fusion_loss = criterion.fusion_criterion(outputs["fusion_logits"], batch.class_id)
        loss = fusion_loss + audio_criterion
        
        return {"loss": loss, "outputs": outputs}

    def _agg_text_pred(self, text_preds: torch.Tensor, method = 'mean', keepdim=True) -> torch.Tensor:
        if method == 'mean':
            text_pred = text_preds.mean(1, keepdim=keepdim)
        elif method == 'max':
            text_pred = text_preds.max(1, keepdim=keepdim)
        else:
            raise NotImplementedError(f"aggregation method: {method} Not implemented")
        return text_pred

    def _log_step_metrics(self, results: Dict[str, Any], batch: VPC25Item, criterion: Any, stage: str) -> None:
        """Log metrics for a single step."""
        # Log loss
        self.log_dict({f"fusion_{criterion.fusion_criterion.__class__.__name__}/{stage}": results["loss"]},
                      batch_size=getattr(self.batch_sizes, stage), **self.logging_params)
        
        # Log metrics
        metric = getattr(self, f"{stage}_metric")
        metric(results["outputs"]["fusion_logits"], batch.class_id)
        self.log(f"{metric.__class__.__name__}/{stage}", metric, batch_size=getattr(self.batch_sizes, stage),
                 **self.logging_params)

    def training_step(self, batch: VPC25Item, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step."""
        results = self.model_step(batch, self.criterion_train)
        self._log_step_metrics(results, batch, self.criterion_train, METRIC_NAMES["TRAIN"])
        return {"loss": results["loss"]}

    def validation_step(self, batch: VPC25Item, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        results = self.model_step(batch, self.criterion_val)
        self._log_step_metrics(results, batch, self.criterion_val, METRIC_NAMES["VALID"])
        return {"loss": results["loss"]}

    def _compute_enrollment_embeddings(self, dataloader) -> dict:
        embeddings_dict = defaultdict(lambda: defaultdict(list))

        for batch in tqdm(dataloader, desc="Computing enrollment embeddings"):
            outputs = self(batch)
            enroll = outputs['audio_features']
            # Handle frame-level embeddings (if applicable)
            if len(enroll.shape) == 3:  # [1, num_frames, embedding_dim]
                enroll = enroll.mean(dim=1)  # [1, embedding_dim]

            embeddings_dict[batch.model][batch.speaker_id].append(enroll)
        return embeddings_dict

    def _compute_test_embeddings(self, dataloader) -> dict:
        embeddings_dict = defaultdict(dict)

        for batch in tqdm(dataloader, desc="Computing test embeddings"):
            outputs = self(batch)
            test = outputs['audio_features']
            # Handle frame-level embeddings (if applicable)
            if len(test.shape) == 3:  # [1, num_frames, embedding_dim]
                test = test.mean(dim=1)  # [num_frames, embedding_dim]

            embeddings_dict.update({path: emb for path, emb in zip(batch.audio_path, test)})
        
        return embeddings_dict

    def on_test_start(self) -> None:
        """Compute embeddings for test trials."""

        # TODO: correct(?)
        self.test_metric.reset()
        self.test_add_metrics.reset()

        _, enroll_loader, unique_test_loader = self.trainer.test_dataloaders
        self.enrol_embeds = defaultdict(dict)
        self.scores = pd.DataFrame(columns=['enrollment_id', 'audio_path', 'label', 'score', 'model'])

        # Compute and aggregate the enrollment embeddings
        enrollment_embeddings = self._compute_enrollment_embeddings(enroll_loader)
        for model_id, class_embeddings in enrollment_embeddings.items():
            for speaker_id, embeddings in class_embeddings.items():
                stacked_embeddings = torch.cat(embeddings, dim=0)
                self.enrol_embeds[model_id][speaker_id] = stacked_embeddings.mean(dim=0)

        self.test_embeddings = self._compute_test_embeddings(unique_test_loader)

    def test_step(self, batch: VPC25VerificationItem, batch_idx: int, dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:
        """Compute EER and minCDF on teh test trials"""
        test_embeddings = torch.stack([self.test_embeddings[path] for path in batch.audio_path])
        enroll_embeddings = torch.stack([self.enrol_embeds[model][enroll_id]
                                         for model, enroll_id in zip(batch.model, batch.enroll_id)])

        scores = torch.nn.functional.cosine_similarity(enroll_embeddings, test_embeddings)
        
        # Extend the dataframe with batch data
        batch_dict = {
            'score': scores.tolist(),
            'label': batch.trial_label,
            'model': batch.model,
            'enrollment_id': batch.enroll_id,
            'audio_path': batch.audio_path
        }

        # Append batch data to scores dataframe
        self.scores = pd.concat([self.scores, pd.DataFrame(batch_dict)], ignore_index=True)

    def on_test_end(self) -> None:
        """compute EER and minDCF"""
        min_dcf, dcf_thresh = sb.utils.metric_stats.minDCF(
                torch.Tensor(self.scores.score[self.scores.label == 1].tolist()), 
                torch.Tensor(self.scores.score[self.scores.label == 0].tolist())
                )
        eer, eer_thresh = sb.utils.metric_stats.EER(
                torch.Tensor(self.scores.score[self.scores.label == 1].tolist()), 
                torch.Tensor(self.scores.score[self.scores.label == 0].tolist())
                )

        # Log metrics
        metrics = {
            'EER': eer,
            'EER_threshold': eer_thresh,
            'minDCF': min_dcf,
            'minDCF_threshold': dcf_thresh
        }
        
        # Log all metrics at once
        self.log_dict(metrics, **self.logging_params)

        # Update scores dataframe with metrics
        self.scores.loc[:, metrics.keys()] = metrics.values()
        
        # Save scores to CSV
        save_path = os.path.join(self.trainer.log_dir, 'test_scores.csv')
        self.scores.to_csv(save_path, index=False)
        torch.save(self.enrol_embeds, os.path.join(self.trainer.log_dir, 'enrol_embeds.pt'))
        torch.save(self.test_embeddings, os.path.join(self.trainer.log_dir, 'test_embeds.pt'))

    def on_train_start(self) -> None:
        """Reset metrics at start of training."""
        # TODO: Is this correct here?
        self.valid_metric_best.reset()

    def on_train_epoch_end(self) -> None:
        """Reset training metrics at end of epoch."""
        self.train_metric.reset()

    def on_validation_epoch_end(self) -> None:
        """Update and log best validation metric."""
        valid_metric = self.valid_metric.compute()
        self.valid_metric_best(valid_metric)
        self.log(
            f"{self.valid_metric.__class__.__name__}/{METRIC_NAMES['BEST']}",
            self.valid_metric_best.compute(),
            **self.logging_params
        )

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