import json
from typing import Any, Dict, Optional
from pathlib import Path
from datetime import datetime

import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd

from src.modules.encoder_wrappers import EncoderWrapper
from src.callbacks.pruning.utils.pruning_manager import PruningManager
from src import utils

log = utils.get_pylogger(__name__)

# CM label mapping: bonafide=1 (positive), spoof=0 (negative)
CM_LABEL_MAP = {"bonafide": 1, "spoof": 0}


class CountermeasureModule(pl.LightningModule):
    """Countermeasure (anti-spoofing) module for ASVSpoof5 Track 1.

    Unlike speaker verification (pairwise trial scoring), CM is a per-utterance
    binary classification task: each audio sample is scored independently as
    bonafide (1) or spoof (0).

    Training:
        Binary classification with weighted CrossEntropyLoss

    Evaluation (validation / test):
        Forward each utterance → take the bonafide logit as the CM score →
        compute official ASVSpoof5 metrics (minDCF, EER, actDCF, CLLR)
        via ASVSpoof5Metrics (vendored from the challenge evaluation package).
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
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.logging_params = logging_params
        self.data_augmentation = kwargs.get("data_augmentation", None)

        self._setup_metrics(metrics)
        self._setup_model_components(model)
        self._setup_training_components(criterion, optimizer, lr_scheduler)

    # ------------------------------------------------------------------ #
    #  Setup helpers                                                       #
    # ------------------------------------------------------------------ #

    def _setup_metrics(self, metrics: DictConfig) -> None:
        self.train_metric = instantiate(metrics.train)
        self.test_metric = instantiate(metrics.test)
        self.valid_metric_best = instantiate(metrics.valid_best)

    def _setup_model_components(self, model: DictConfig) -> None:
        self.audio_processor = instantiate(model.audio_processor)
        self.audio_processor_normalizer = instantiate(
            model.audio_processor_normalizer
        )

        raw_audio_encoder = instantiate(model.audio_encoder)
        self.audio_encoder = EncoderWrapper(
            encoder=raw_audio_encoder,
            audio_processor=self.audio_processor,
            audio_processor_normalizer=self.audio_processor_normalizer,
        )
            
        self.classifier = instantiate(model.classifier)

        if self.data_augmentation is not None:
            assert (
                "wav_augmenter" in self.data_augmentation.augmentations
            ), "Expected augmentations.wav_augmenter when passing data_augmentation"
            self.wav_augmenter = instantiate(
                self.data_augmentation.augmentations.wav_augmenter
            )

    def _setup_training_components(
        self,
        criterion: DictConfig,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
    ) -> None:
        criterion_fn = instantiate(criterion.train_criterion)
        class_weights = list(criterion.get("class_weights", [1.0, 6.5]))
        self.train_criterion = criterion_fn(weight=torch.tensor(class_weights))
        self.optimizer = optimizer
        self.slr_params = lr_scheduler

    # ------------------------------------------------------------------ #
    #  Forward / model step                                                #
    # ------------------------------------------------------------------ #

    def _get_audio_embeddings(
        self, batch_audio: torch.Tensor, batch_audio_lens: torch.Tensor
    ) -> torch.Tensor:
        if self.device != batch_audio.device:
            batch_audio = batch_audio.to(self.device)
        if self.device != batch_audio_lens.device:
            batch_audio_lens = batch_audio_lens.to(self.device)
        return self.audio_encoder(wavs=batch_audio, wav_lens=batch_audio_lens)

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        """Forward pass through encoder + classifier.

        Accepts both training batches (DatasetItem via BaseCollate) and
        Track 1 evaluation batches (ASVSpoofTrack1Batch).
        """
        if self.training and hasattr(self, "wav_augmenter"):
            max_audio_length = max(batch.audio_length)
            batch.audio, audio_length_norm = self.wav_augmenter(
                batch.audio, batch.audio_length / max_audio_length
            )
            batch.audio_length = audio_length_norm * max_audio_length
            batch.class_id = self.wav_augmenter.replicate_labels(
                batch.class_id
            )

        audio_emb = self._get_audio_embeddings(batch.audio, batch.audio_length)
        logits = self.classifier(audio_emb)

        return {"embeds": audio_emb, "logits": logits}

    def model_step(
        self, batch, criterion: Optional[Any] = None
    ) -> Dict[str, Any]:
        outputs = self(batch)
        logits = outputs["logits"]
        main_loss = criterion(logits, batch.class_id)
        return {"loss": main_loss, "outputs": outputs}

    def _log_step_metrics(
        self, results: Dict[str, Any], batch, stage: str
    ) -> None:
        logged_dict = {
            f"{stage}/{self.train_criterion.__class__.__name__}": results["loss"].item()
        }
        self.log_dict(
            logged_dict,
            batch_size=batch.audio.shape[0],
            **self.logging_params,
        )

        metric = getattr(self, f"{stage}_metric")
        computed_metric = metric(results["outputs"]["logits"], batch.class_id)
        self.log(
            f"{stage}/{metric.__class__.__name__}",
            computed_metric,
            batch_size=batch.audio.shape[0],
            **self.logging_params,
        )

    # ------------------------------------------------------------------ #
    #  Lightning hooks — Training                                          #
    # ------------------------------------------------------------------ #

    def on_train_start(self) -> None:
        self.valid_metric_best.reset()
        self.audio_encoder.train()

    def training_step(self, batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        results = self.model_step(batch, self.train_criterion)
        self._log_step_metrics(results, batch, "train")
        return results

    def on_train_epoch_end(self) -> None:
        self.train_metric.reset()

    # ------------------------------------------------------------------ #
    #  Lightning hooks — Validation (Track 1 dev)                          #
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def on_validation_start(self) -> None:
        """Initialize accumulators for CM scores and labels."""
        self._val_cm_scores: list = []
        self._val_cm_labels: list = []

    @torch.inference_mode()
    def validation_step(
        self, batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Score each Track 1 utterance and accumulate for EER/minDCF."""
        self._cm_eval_step(batch, self._val_cm_scores, self._val_cm_labels)

    def on_validation_epoch_end(self) -> None:
        """Compute and log EER/minDCF on accumulated validation scores."""
        self._cm_epoch_end(
            self._val_cm_scores,
            self._val_cm_labels,
            stage="valid",
            is_test=False,
        )
        # Clean up
        del self._val_cm_scores, self._val_cm_labels

    # ------------------------------------------------------------------ #
    #  Lightning hooks — Test (Track 1 eval)                               #
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def on_test_start(self) -> None:
        """Initialize accumulators for CM scores and labels."""
        self._test_cm_scores: list = []
        self._test_cm_labels: list = []

    @torch.inference_mode()
    def test_step(
        self, batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Score each Track 1 utterance and accumulate for EER/minDCF."""
        self._cm_eval_step(batch, self._test_cm_scores, self._test_cm_labels)

    def on_test_epoch_end(self) -> None:
        """Compute, log, and save EER/minDCF on accumulated test scores."""
        self._cm_epoch_end(
            self._test_cm_scores,
            self._test_cm_labels,
            stage="test",
            is_test=True,
        )
        del self._test_cm_scores, self._test_cm_labels

    # ------------------------------------------------------------------ #
    #  CM evaluation logic                                                 #
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def _cm_eval_step(
        self,
        batch,
        scores_accum: list,
        labels_accum: list,
    ) -> None:
        """Score a batch of Track 1 utterances.

        The CM score is the log-softmax probability of the bonafide class
        (index 1). Higher score = more likely bonafide.

        The batch is an ``ASVSpoofTrack1Batch`` with ``cm_key`` tuples
        containing 'bonafide' or 'spoof' strings.
        """
        # Move audio to device
        audio = batch.audio.to(self.device)
        audio_length = batch.audio_length.to(self.device)

        # Get embeddings and logits
        emb = self._get_audio_embeddings(audio, audio_length)
        logits = self.classifier(emb)

        # CM score = log-softmax for bonafide class (index 1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        cm_scores = log_probs[:, 1]  # bonafide log-probability

        # Map cm_key strings to binary labels
        cm_labels = torch.tensor(
            [CM_LABEL_MAP[k] for k in batch.cm_key],
            dtype=torch.long,
            device=cm_scores.device,
        )

        scores_accum.append(cm_scores.detach().cpu())
        labels_accum.append(cm_labels.detach().cpu())

    def _cm_epoch_end(
        self,
        scores_accum: list,
        labels_accum: list,
        stage: str,
        is_test: bool,
    ) -> None:
        """Compute EER/minDCF from accumulated CM scores and labels."""
        if not scores_accum:
            log.warning(f"No CM scores accumulated for {stage}. Skipping.")
            return

        all_scores = torch.cat(scores_accum)
        all_labels = torch.cat(labels_accum)

        log.info(
            f"CM {stage}: {len(all_scores)} utterances "
            f"(bonafide={int((all_labels == 1).sum())}, "
            f"spoof={int((all_labels == 0).sum())})"
        )

        # Compute official ASVSpoof5 Track 1 metrics (EER, minDCF, actDCF, CLLR)
        self.test_metric.reset()
        self.test_metric.update(scores=all_scores, labels=all_labels)
        metrics = self.test_metric.compute()

        # Log all metrics under {stage}/{key} for monitoring and callbacks
        logged = {f"{stage}/{k}": v for k, v in metrics.items()}
        self.log_dict(
            logged,
            batch_size=len(all_scores),
            **self.logging_params,
        )

        # Update best metric tracker (valid only)
        if stage == "valid" and "minDCF" in metrics:
            self.valid_metric_best.update(metrics)

        # Save artifacts for test
        if is_test:
            self._save_test_artifacts(all_scores, all_labels, metrics, stage)

    def _save_test_artifacts(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        metrics: Dict,
        stage: str,
    ) -> None:
        """Save test scores, labels, and metrics to disk."""
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifacts_dir = (
            Path(self.trainer.default_root_dir)
            / "test_artifacts"
            / f"cm_{stage}"
            / run_timestamp
        )
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Save scores CSV
        df = pd.DataFrame(
            {
                "score": scores.numpy(),
                "label": labels.numpy(),
            }
        )
        df.to_csv(artifacts_dir / "cm_scores.csv", index=False)

        # Save metrics JSON
        metrics_for_save = {
            k: v.item() if torch.is_tensor(v) else v
            for k, v in metrics.items()
        }
        with open(artifacts_dir / "cm_metrics.json", "w") as f:
            json.dump(metrics_for_save, f, indent=4)

        log.info(f"Saved CM {stage} artifacts to: {artifacts_dir}")

    # ------------------------------------------------------------------ #
    #  Optimizer                                                           #
    # ------------------------------------------------------------------ #

    def configure_optimizers(self) -> Dict[str, Any]:
        BREGMAN_OPTIMIZERS = {"AdaBreg", "AdaBregW", "LinBreg", "ProxSGD"}
        optimizer_class_name = self.hparams.optimizer._target_.split(".")[-1]

        # Filter out None values from optimizer config to allow swapping
        # optimizer types (e.g., SGD→Adam) without leftover params
        from omegaconf import OmegaConf
        opt_cfg = OmegaConf.to_container(self.hparams.optimizer, resolve=True)
        opt_cfg = {k: v for k, v in opt_cfg.items() if v is not None}
        opt_cfg = OmegaConf.create(opt_cfg)
        optimizer_partial = instantiate(opt_cfg)

        if optimizer_class_name in BREGMAN_OPTIMIZERS:
            self.pruning_manager = PruningManager(
                pl_module=self,
                group_configs=self.hparams.model.pruning_groups,
            )
            optimizer_param_groups = (
                self.pruning_manager.get_optimizer_param_groups()
            )
            for group in optimizer_param_groups:
                if "reg" in group and isinstance(
                    group.get("reg"), (dict, DictConfig)
                ):
                    group["reg"] = instantiate(group["reg"])
            optimizer = optimizer_partial(params=optimizer_param_groups)
        else:
            optimizer = optimizer_partial(params=self.parameters())

        if self.hparams.get("lr_scheduler"):
            # Filter out None values to allow swapping scheduler types
            sched_cfg = OmegaConf.to_container(
                self.hparams.lr_scheduler.scheduler, resolve=True
            )
            sched_cfg = {k: v for k, v in sched_cfg.items() if v is not None}
            sched_cfg = OmegaConf.create(sched_cfg)
            scheduler = instantiate(sched_cfg, optimizer=optimizer)
            lr_scheduler_dict = {"scheduler": scheduler}
            if self.hparams.lr_scheduler.get("extras"):
                for key, value in self.hparams.lr_scheduler.get(
                    "extras"
                ).items():
                    lr_scheduler_dict[key] = value
            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler_dict,
            }

        return {"optimizer": optimizer}