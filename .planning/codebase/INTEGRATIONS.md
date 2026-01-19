# External Integrations

**Analysis Date:** 2026-01-19

## APIs & External Services

**Model Hubs:**
- Hugging Face Hub - Pretrained model downloads
  - SDK/Client: `huggingface_hub`
  - Auth: `HF_TOKEN` (optional, for gated models)
  - Usage: `hf_hub_download()` in `src/utils/hf_utils.py`
  - Models: WeSpeaker, ECAPA2, transformers models

- SpeechBrain - Pretrained speaker recognition models
  - SDK/Client: `speechbrain.inference.classifiers.EncoderClassifier`
  - Auth: None (public models)
  - Usage: Language identification in `src/modules/components/utils.py`
  - Models: `speechbrain/lang-id-commonlanguage_ecapa`

**Experiment Tracking:**
- Neptune.ai - Primary experiment logger
  - SDK/Client: `pytorch_lightning.loggers.NeptuneLogger`
  - Auth: `NEPTUNE_API_TOKEN` env var
  - Config: `configs/logger/neptune.yaml`
  - Project: `ahmad.aloradi94/vpc25`

- Weights & Biases - Alternative experiment logger
  - SDK/Client: `pytorch_lightning.loggers.wandb.WandbLogger`
  - Auth: `WANDB_API_TOKEN` env var
  - Config: `configs/logger/wandb.yaml`
  - Project: `comfort`, Entity: `al_sap`

- MLflow - Local experiment tracking
  - SDK/Client: `pytorch_lightning.loggers.mlflow.MLFlowLogger`
  - Auth: None (local)
  - Config: `configs/logger/mlflow.yaml`
  - Tracking URI: `${paths.log_dir}/mlflow/mlruns`

- TensorBoard - Local visualization
  - SDK/Client: `pytorch_lightning.loggers.TensorBoardLogger`
  - Auth: None (local)
  - Config: `configs/logger/tensorboard.yaml`

## Data Storage

**Datasets:**
- Local filesystem - Primary storage
  - Location: `${paths.data_dir}` (configurable, default: `data/`)
  - Formats: WAV, FLAC audio files
  - Metadata: CSV files with pandas

**File Storage:**
- Local filesystem only
  - Checkpoints: `${paths.output_dir}/checkpoints/`
  - Logs: `${paths.log_dir}/`
  - Artifacts: Hydra output directories

**Caching:**
- Embedding cache in `src/modules/components/utils.py`
  - `EmbeddingCache` class with LRU eviction
  - Thread-safe with RLock
  - Configurable max_size (default: 10000)
- Test embedding caching to disk in `src/modules/sv.py`
  - Cohort embeddings: `test_artifacts/_cohort_cache/`
  - Per-test-set embeddings: `test_artifacts/<test_set>/cache/`

## Authentication & Identity

**Auth Provider:**
- None - No user authentication
- API tokens for external services loaded from `.env`

## Monitoring & Observability

**Error Tracking:**
- None - Relies on experiment loggers

**Logs:**
- Python logging via `src/utils/pylogger.py`
- Hydra logging configuration in `configs/hydra/`
- Rich progress bars for training

## CI/CD & Deployment

**Hosting:**
- Local/HPC execution (no cloud deployment)
- Fabric for remote execution (`scripts/fabfile.py`)

**CI Pipeline:**
- Pre-commit hooks for code quality
- No automated CI/CD detected

## Environment Configuration

**Required env vars:**
- `PROJECT_ROOT` - Auto-set by pyrootutils
- `NEPTUNE_API_TOKEN` - Neptune.ai authentication
- `WANDB_API_TOKEN` - W&B authentication (optional)

**Optional env vars:**
- `NEPTUNE_RUN_ID` - Resume specific Neptune run
- `HF_TOKEN` - Hugging Face authentication for gated models

**Secrets location:**
- `.env` file at project root (git-ignored)

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- Neptune.ai metrics logging (async mode)
- W&B metrics logging
- MLflow metrics logging

## External Data Sources

**Augmentation Data (downloaded at runtime):**
- Noise dataset: Dropbox URL in `configs/experiment/sv/sv_aug.yaml`
  - URL: `https://www.dropbox.com/scl/fi/a09pj97s5ifan81dqhi4n/noises.zip?...`
  - Destination: `${paths.data_dir}/RIRS_NOISES/noise`

- RIR dataset: Dropbox URL in `configs/experiment/sv/sv_aug.yaml`
  - URL: `https://www.dropbox.com/scl/fi/linhy77c36mu10965a836/RIRs.zip?...`
  - Destination: `${paths.data_dir}/RIRS_NOISES/rir`

**Pretrained Models (downloaded on first use):**
- WeSpeaker models from Hugging Face
- SpeechBrain models (auto-cached to `local/pretrained_models/`)
- NeMo models from NGC/Hugging Face (optional)
- ECAPA2 TorchScript models from Hugging Face

## Hyperparameter Optimization

**Optuna Integration:**
- Config: `configs/hparams_search/optuna.yaml`
- Sweeper: `hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper`
- Sampler: TPE (Tree-structured Parzen Estimator)
- Storage: SQLite supported (configurable)

---

*Integration audit: 2026-01-19*
