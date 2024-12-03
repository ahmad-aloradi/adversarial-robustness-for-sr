from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import pytorch_lightning as pl
from hydra.utils import instantiate
from src.datamodules.components.vpc25.vpc_dataset import VPCBatch
from src import utils

log = utils.get_pylogger(__name__)

class MultiModalVPCModel(pl.LightningModule):
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        lr_scheduler,
        network=None,  # From defaults/network classification.yaml
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Models
        self.num_classes = 7204
        self.text_encoder_output_dim = 768
        self.audio_encoder_output_dim = 512
        self.sample_rate = 16000

        self.text_encoder = instantiate(model.text_encoder).to(self.device)
        self.processor = instantiate(model.processor)
        self.text_classifier = nn.Linear(self.text_encoder_output_dim, self.num_classes)
        
        self.audio_encoder = instantiate(model.audio_encoder)
        self.audio_classifier = nn.Linear(self.audio_encoder_output_dim, self.num_classes)
        
        self.gender_classifier = nn.Sequential(
            nn.Linear(self.audio_encoder_output_dim, model.gender_classifier.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(model.gender_classifier.hidden_size, model.gender_classifier.num_classes)
        )
        
        # Losses
        self.criterion_train = instantiate(criterion.criterion_train)
        self.criterion_val = instantiate(criterion.criterion_val)
        self.criterion_test = instantiate(criterion.criterion_test)
        
        # Training configs
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        # Freeze pretrained
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        # for param in self.audio_encoder.parameters():
        #     param.requires_grad = False

    def forward(self, batch: VPCBatch):
        encoded = self.processor(
            batch.audio,
            sampling_rate=self.sample_rate, 
            return_tensors="pt"
            ).input_values
        encoded = encoded.squeeze(0).to(batch.audio.device)

        text_embeddings = self.text_encoder(encoded).logits
        pred_ids = text_embeddings.argmax(dim=-1)
        decoded_preds = self.processor.batch_decode(pred_ids)
        # TODO: Implement text classifier
        text_speaker_logits = text_embeddings
        # text_speaker_logits = self.text_classifier(text_embeddings)
        
        audio_embeddings = self.audio_encoder.encode_batch(batch.audio.squeeze(0)).squeeze(1)

        audio_speaker_logits = self.audio_classifier(audio_embeddings)
        gender_logits = self.gender_classifier(audio_embeddings)
        
        return {
            "text_speaker_logits": text_speaker_logits,
            "audio_speaker_logits": audio_speaker_logits,
            "gender_logits": gender_logits
        }

    def _step(self, batch: VPCBatch, criterion: Optional[Any] = None) -> Dict[str, torch.Tensor]:
        outputs = self(batch)
        
        # Re-define labels
        speaker_labels = torch.tensor([int(id) for id in batch.speaker_id], device=batch.audio.device)
        gender_labels = torch.tensor([0 if g == "M" else 1 for g in batch.gender], device=batch.audio.device)
        
        # Cpompute losses
        try:
            audio_speaker_loss = criterion(outputs["audio_speaker_logits"].unsqueeze(1), speaker_labels.unsqueeze(1))
            gender_loss = criterion(outputs["gender_logits"].unsqueeze(1), gender_labels.unsqueeze(-1))
        except:
            import ipdb; ipdb.set_trace()

        losses = {
            # "text_speaker": criterion(outputs["text_speaker_logits"], speaker_labels),
            "audio_speaker": audio_speaker_loss,
            "gender": gender_loss
        }
        
        accuracies = {
            # "text_speaker": (outputs["text_speaker_logits"].argmax(dim=1) == speaker_labels).float().mean(),
            "audio_speaker": (outputs["audio_speaker_logits"].argmax(dim=1) == speaker_labels).float().mean(),
            "gender": (outputs["gender_logits"].argmax(dim=1) == gender_labels).float().mean()
        }
        
        return {"losses": losses, "accuracies": accuracies, "outputs": outputs}

    def training_step(self, batch: VPCBatch, batch_idx: int) -> torch.Tensor:
        results = self._step(batch, self.criterion_train)
        total_loss = sum(results["losses"].values())
        
        for name, value in {**results["losses"], **results["accuracies"]}.items():
            self.log(f"train_{name}_{'loss' if name in results['losses'] else 'acc'}", value)
            
        return total_loss

    def validation_step(self, batch: VPCBatch, batch_idx: int) -> None:
        results = self._step(batch, self.criterion_val)
        for name, value in {**results["losses"], **results["accuracies"]}.items():
            self.log(f"val_{name}_{'loss' if name in results['losses'] else 'acc'}", value)

    def test_step(self, batch: VPCBatch, batch_idx: int) -> None:
        results = self._step(batch, self.criterion_test)
        for name, value in {**results["losses"], **results["accuracies"]}.items():
            self.log(f"test_{name}_{'loss' if name in results['losses'] else 'acc'}", value)

    def configure_optimizers(self) -> Dict:
        optimizer = instantiate(self.optimizer)(params=self.parameters())
        scheduler = instantiate(self.lr_scheduler)(optimizer=optimizer)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_total_loss"
        }