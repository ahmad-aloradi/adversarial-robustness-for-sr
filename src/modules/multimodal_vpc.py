from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.datamodules.components.vpc25.vpc_dataset import VPCBatch
from src import utils
from src.modules.metrics import load_metrics

log = utils.get_pylogger(__name__)

class MultiModalVPCModel(pl.LightningModule):
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
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.batch_sizes = kwargs.get("batch_sizes")

        # Get Metrics
        main_metric, valid_metric_best, add_metrics = load_metrics(metrics)
        self.train_metric = main_metric.clone()
        # self.train_add_metrics = add_metrics.clone(postfix="/train")
        self.valid_metric = main_metric.clone()
        self.valid_metric_best = valid_metric_best.clone()
        # self.valid_add_metrics = add_metrics.clone(postfix="/valid")
        self.test_metric = main_metric.clone()
        self.test_add_metrics = add_metrics.clone(postfix="/test")

        # Logging params
        self.logging_params = logging_params

        # Text classifier
        self.text_encoder = instantiate(model.text_encoder).to(self.device)
        self.processor = instantiate(model.processor)
        self.text_classifier = instantiate(model.classifiers.text_classifier)

        # Speaker classifier
        self.audio_encoder = instantiate(model.audio_encoder)
        self.audio_classifier = instantiate(model.classifiers.audio_classifier)

        # Gender classifier
        self.gender_classifier = instantiate(model.classifiers.gender_classifier)
        
        # Losses
        self.criterion_train = instantiate(criterion.criterion_train)
        self.criterion_val = instantiate(criterion.criterion_val)
        self.criterion_test = instantiate(criterion.criterion_test)
        self.weights = instantiate(criterion.loss_weights)
        
        # Training configs
        self.optimizer = optimizer
        self.slr_params = lr_scheduler
        
        # Freeze pretrained
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.audio_encoder.parameters():
            param.requires_grad = True

    def forward(self, batch: VPCBatch):
        assert len(set(batch.sample_rate)) == 1, "All audio samples should have the same sample rate"

        encoded = self.processor(batch.audio, sampling_rate=batch.sample_rate[0], return_tensors="pt"
                                 ).input_values
        encoded = encoded.squeeze(0).to(batch.audio.device)

        text_embeddings = self.text_encoder(encoded).logits
        # TODO: Anything to do with decoded_preds?
        # decoded_preds = self.processor.batch_decode(text_embeddings.argmax(dim=-1))

        # Text classifier
        text_logits = self._agg_text_pred(text_embeddings, method='mean', keepdim=False)
        text_logits = self.text_classifier(text_logits)
        
        audio_embeddings = self.audio_encoder.encode_batch(batch.audio.squeeze(0)).squeeze(1)
        audio_logits = self.audio_classifier(audio_embeddings)
        gender_logits = self.gender_classifier(audio_embeddings)
        
        return {"text_logits": text_logits, "audio_logits": audio_logits, "gender_logits": gender_logits}

    def _agg_text_pred(self, text_preds: torch.Tensor, method: str ='mean', keepdim: bool = True
                       ) -> torch.Tensor:
        if method == 'mean':
            text_pred = text_preds.mean(1, keepdim=keepdim)
        elif method == 'max':
            text_pred = text_preds.max(1, keepdim=keepdim)
        else:
            raise NotImplementedError(f"aggregation method: {method} Not implemented")
        return text_pred

    def model_step(self, batch: VPCBatch, criterion: Optional[Any] = None) -> Dict[str, torch.Tensor]:
        outputs = self(batch)
                
        # Cpompute losses
        text_loss = criterion.text_criterion(outputs["text_logits"], batch.speaker_id)
        audio_loss = criterion.audio_criterion(outputs["audio_logits"], batch.speaker_id)
        gender_loss = criterion.gender_criterion(outputs["gender_logits"].squeeze(-1), batch.gender)
        total_loss = self.weights["text"] * text_loss + self.weights["audio"] * audio_loss + self.weights["gender"] * gender_loss
        
        losses = {"text": text_loss, "audio": audio_loss, "gender": gender_loss, "total": total_loss}
        
        return {"losses": losses, "outputs": outputs}

    def on_train_start(self) -> None:
        self.valid_metric_best.reset()

    def training_step(self, batch: VPCBatch, batch_idx: int) -> torch.Tensor:
        results = self.model_step(batch, self.criterion_train)
        
        self.log_dict({
            f"text_{self.criterion_train.text_criterion.__class__.__name__}/train": results["losses"]["text"],
            f"audio_{self.criterion_train.audio_criterion.__class__.__name__}/train": results["losses"]["audio"],
            f"gender_{self.criterion_train.gender_criterion.__class__.__name__}/train": results["losses"]["gender"],
            f"total_loss/train": results["losses"]["total"]
            }, **self.logging_params)

        self.train_metric(results["outputs"]["audio_logits"],  batch.speaker_id)
        self.log(f"{self.train_metric.__class__.__name__}/train",
                 self.train_metric,
                 batch_size=self.batch_sizes.train,
                 **self.logging_params)

        return {"loss": results["losses"]["total"]}

    def on_train_epoch_end(self):
        self.train_metric.reset()

    def validation_step(self, batch: VPCBatch, batch_idx: int) -> None:
        results = self.model_step(batch, self.criterion_val)

        self.log_dict({
            f"text_{self.criterion_val.text_criterion.__class__.__name__}/valid": results["losses"]["text"],
            f"audio_{self.criterion_val.audio_criterion.__class__.__name__}/valid": results["losses"]["audio"],
            f"gender_{self.criterion_val.gender_criterion.__class__.__name__}/valid": results["losses"]["gender"],
            f"total_loss/valid": results["losses"]["total"]
            }, **self.logging_params)

        self.valid_metric(results["outputs"]["audio_logits"],  batch.speaker_id)
        self.log(
            f"{self.valid_metric.__class__.__name__}/valid",
            self.valid_metric,
            batch_size=self.batch_sizes.valid,
            **self.logging_params,
        )
        # self.valid_add_metrics(results["outputs"]["audio_logits"],  batch.speaker_id)
        # self.log_dict(self.valid_add_metrics, **self.logging_params)

        return {"loss": results["losses"]["total"]}
    
    def on_validation_epoch_end(self) -> None:
        valid_metric = self.valid_metric.compute()  # get current valid metric
        self.valid_metric_best(valid_metric)  # update best so far valid metric
        self.log(
            f"{self.valid_metric.__class__.__name__}/valid_best",
            self.valid_metric_best.compute(),
            **self.logging_params
            )
        
    def test_step(self, batch: VPCBatch, batch_idx: int) -> None:
        results = self.model_step(batch, self.criterion_test)
        self.log_dict({
            f"text_{self.criterion_test.text_criterion.__class__.__name__}/test": results["losses"]["text"],
            f"audio_{self.criterion_test.audio_criterion.__class__.__name__}/test": results["losses"]["audio"],
            f"gender_{self.criterion_test.gender_criterion.__class__.__name__}/test": results["losses"]["gender"],
            f"total_loss/test": results["losses"]["total"]
            }, **self.logging_params)
        
        self.test_metric(results["outputs"]["audio_logits"],  batch.speaker_id)
        self.log(
            f"{self.test_metric.__class__.__name__}/test",
            self.test_metric,
            batch_size=self.batch_sizes.test,
            **self.logging_params,
        )
        self.test_add_metrics(results["outputs"]["audio_logits"],  batch.speaker_id)
        self.log_dict(self.test_add_metrics, **self.logging_params)
        
        return {"loss": results["losses"]["total"]}

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