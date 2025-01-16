from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.datamodules.components.vpc25.vpc_dataset import VPCBatch
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
    "TOTAL": "total"
}

class MultiModalVPCModel(pl.LightningModule):
    """Multi-modal Voice Print Classification Model.
    
    This model combines text and audio processing for speaker identification and gender classification.
    It uses wav2vec2 for text encoding and speechbrain for audio encoding.
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
        self.batch_sizes = kwargs.get("batch_sizes")
        self.logging_params = logging_params
        
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
        # Text processing components
        self.text_encoder = instantiate(model.text_encoder)
        self.processor = instantiate(model.processor)
        self.text_classifier = instantiate(model.classifiers.text_classifier)
        
        # Audio processing components
        self.audio_encoder = instantiate(model.audio_encoder)
        self.audio_classifier = instantiate(model.classifiers.audio_classifier)
        
        # Gender classification
        self.gender_classifier = instantiate(model.classifiers.gender_classifier)

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
        self.weights = instantiate(criterion.loss_weights)
        self.optimizer = optimizer
        self.slr_params = lr_scheduler

    def _freeze_pretrained_components(self) -> None:
        """Freeze pretrained components and enable training for others."""
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.audio_encoder.parameters():
            param.requires_grad = True

    def forward(self, batch: VPCBatch) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        if len(set(batch.sample_rate)) != 1:
            raise ValueError("Wav2Vec2Processor expects sampling_rate of type: int, but found multiple sampling rates.")

        # Process audio input
        encoded = self.processor(
            batch.audio, 
            sampling_rate=batch.sample_rate[0], 
            return_tensors="pt"
        ).input_values.squeeze(0).to(batch.audio.device)

        # Get embeddings and logits
        text_embeddings = self.text_encoder(encoded).logits
        text_logits = self._agg_text_pred(text_embeddings, method='mean', keepdim=False)
        text_logits = self.text_classifier(text_logits)
        
        audio_embeddings = self.audio_encoder.encode_batch(batch.audio.squeeze(0)).squeeze(1)
        audio_logits = self.audio_classifier(audio_embeddings)
        gender_logits = self.gender_classifier(audio_embeddings)
        
        return {
            "text_logits": text_logits, 
            "audio_logits": audio_logits, 
            "gender_logits": gender_logits
        }

    def _agg_text_pred(self, text_preds: torch.Tensor, method: str ='mean', keepdim: bool = True
                       ) -> torch.Tensor:
        if method == 'mean':
            text_pred = text_preds.mean(1, keepdim=keepdim)
        elif method == 'max':
            text_pred = text_preds.max(1, keepdim=keepdim)
        else:
            raise NotImplementedError(f"aggregation method: {method} Not implemented")
        return text_pred

    def model_step(
        self, 
        batch: VPCBatch, 
        criterion: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Perform a single model step."""
        outputs = self(batch)
        
        # Compute losses
        text_loss = criterion.text_criterion(outputs["text_logits"], batch.speaker_id)
        audio_loss = criterion.audio_criterion(outputs["audio_logits"], batch.speaker_id)
        gender_loss = criterion.gender_criterion(
            outputs["gender_logits"].squeeze(-1), 
            batch.gender
        )
        
        total_loss = (
            self.weights["text"] * text_loss + 
            self.weights["audio"] * audio_loss + 
            self.weights["gender"] * gender_loss
        )
        
        losses = {
            LOSS_TYPES["TEXT"]: text_loss,
            LOSS_TYPES["AUDIO"]: audio_loss,
            LOSS_TYPES["GENDER"]: gender_loss,
            LOSS_TYPES["TOTAL"]: total_loss
        }
        
        return {"losses": losses, "outputs": outputs}

    def _log_step_metrics(
        self, 
        results: Dict[str, Any], 
        batch: VPCBatch, 
        criterion: Any,
        stage: str
    ) -> None:
        """Log metrics for a single step."""
        # Log losses
        self.log_dict({
            f"text_{criterion.text_criterion.__class__.__name__}/{stage}": results["losses"]["text"],
            f"audio_{criterion.audio_criterion.__class__.__name__}/{stage}": results["losses"]["audio"],
            f"gender_{criterion.gender_criterion.__class__.__name__}/{stage}": results["losses"]["gender"],
            f"total_loss/{stage}": results["losses"]["total"]
        }, **self.logging_params)
        
        # Log metrics
        metric = getattr(self, f"{stage}_metric")
        metric(results["outputs"]["audio_logits"], batch.speaker_id)
        self.log(
            f"{metric.__class__.__name__}/{stage}",
            metric,
            batch_size=getattr(self.batch_sizes, stage),
            **self.logging_params
        )

    def training_step(self, batch: VPCBatch, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step."""
        results = self.model_step(batch, self.criterion_train)
        self._log_step_metrics(results, batch, self.criterion_train, METRIC_NAMES["TRAIN"])
        return {"loss": results["losses"]["total"]}

    def validation_step(self, batch: VPCBatch, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        results = self.model_step(batch, self.criterion_val)
        self._log_step_metrics(results, batch, self.criterion_val, METRIC_NAMES["VALID"])
        return {"loss": results["losses"]["total"]}

    def test_step(self, batch: VPCBatch, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step."""
        results = self.model_step(batch, self.criterion_test)
        self._log_step_metrics(results, batch, self.criterion_test, METRIC_NAMES["TEST"])
        
        # Log additional test metrics
        self.test_add_metrics(results["outputs"]["audio_logits"], batch.speaker_id)
        self.log_dict(self.test_add_metrics, **self.logging_params)
        
        return {"loss": results["losses"]["total"]}

    def on_train_start(self) -> None:
        """Reset metrics at start of training."""
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
        optimizer: torch.optim  = instantiate(self.optimizer)(params=self.parameters())
        
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
