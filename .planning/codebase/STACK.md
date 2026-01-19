# Technology Stack

**Analysis Date:** 2026-01-19

## Languages

**Primary:**
- Python 3.11 - All source code, training scripts, data processing

**Secondary:**
- YAML - Configuration files (Hydra configs in `configs/`)
- Bash - Dataset preparation scripts in `scripts/`

## Runtime

**Environment:**
- Python 3.11 (compatible version specified in `requirements.txt`)
- CUDA 12.4 (NVIDIA GPU support via PyTorch)

**Package Manager:**
- pip with `requirements.txt`
- Lockfile: Present (`requirements.txt` contains pinned versions)

## Frameworks

**Core:**
- PyTorch 2.5.1 - Deep learning framework
- PyTorch Lightning 2.5.0 - Training orchestration
- Hydra 1.3.1 - Configuration management
- OmegaConf 2.3.0 - Config parsing/manipulation

**Audio Processing:**
- torchaudio 2.5.1 - Audio loading and transforms
- soundfile 0.12.1 - Audio file I/O
- torch-audiomentations 0.11.1 - GPU-native audio augmentations
- SpeechBrain 1.0.2 - Feature extraction (Fbank), losses (AAM-Softmax), pretrained models

**ML/Metrics:**
- torchmetrics 0.11.0 - Metrics (Accuracy)
- scikit-learn 1.6.1 - EER/minDCF computation utilities

**Testing:**
- pytest 8.3.3 - Test framework
- pytest-mock 3.15.0 - Mocking support
- Config: `pyproject.toml` [tool.pytest.ini_options]

**Build/Dev:**
- pre-commit 3.8.0 - Git hooks
- black 24.10.0 - Code formatting (79 char line length)
- isort 5.13.2 - Import sorting (black profile)
- flake8 4.0.1 - Linting
- bandit 1.7.1 - Security linting
- docformatter 1.7.5 - Docstring formatting

## Key Dependencies

**Critical:**
- speechbrain 1.0.2 - AAM-Softmax loss, ECAPA-TDNN classifier, Fbank features, pretrained models
- wespeaker (git install) - Speaker verification model architectures (ECAPA-TDNN, CAMPPlus, ResNet293, ReDimNet)
- nemo_toolkit (optional) - TitaNet pretrained models
- transformers 4.51.3 - Hugging Face model loading
- huggingface_hub 0.34.4 - Model downloads

**Infrastructure:**
- pyrootutils 1.0.4 - Project root detection, .env loading
- python-dotenv 1.0.1 - Environment variable loading
- pandas 2.2.3 - Data manipulation (CSV metadata)
- tqdm 4.67.1 - Progress bars

**Experiment Tracking:**
- wandb 0.18.7 - Weights & Biases logging
- neptune 1.13.0 - Neptune.ai logging
- mlflow 2.17.2 - MLflow logging
- tensorboard 2.18.0 - TensorBoard logging

**Model Compression:**
- Custom Bregman optimizers in `src/callbacks/pruning/bregman/`
- PyTorch pruning utilities

## Configuration

**Environment:**
- `.env` file at project root (loaded by pyrootutils)
- Required vars:
  - `PROJECT_ROOT` - Set automatically by pyrootutils
  - `NEPTUNE_API_TOKEN` - For Neptune logging
  - `WANDB_API_TOKEN` - For W&B logging
  - `NEPTUNE_RUN_ID` (optional) - For resuming Neptune runs

**Build:**
- `setup.py` - Package installation (editable mode)
- `pyproject.toml` - pytest configuration
- `.pre-commit-config.yaml` - Pre-commit hooks
- `Makefile` - Common commands (test, format, train)

**Hydra Config Structure:**
```
configs/
├── train.yaml              # Main training config
├── eval.yaml               # Evaluation config
├── datamodule/datasets/    # Dataset configs (vpc, voxceleb, cnceleb, librispeech)
├── module/                 # Module configs
│   ├── sv.yaml            # Speaker verification module
│   ├── vpc.yaml           # VoicePrivacy module
│   └── sv_model/          # Model architecture configs
├── callbacks/              # Callback configs
├── trainer/                # Trainer configs (gpu, ddp, cpu)
├── logger/                 # Logger configs (wandb, neptune, mlflow, tensorboard)
├── experiment/sv/          # Complete experiment recipes
└── hparams_search/         # Optuna hyperparameter search
```

## Platform Requirements

**Development:**
- Linux (tested on Ubuntu with kernel 5.15.0)
- Python 3.11
- NVIDIA GPU with CUDA 12.4 (optional but recommended)
- ~32GB RAM recommended for large datasets

**Production:**
- GPU training: NVIDIA A100/V100/RTX series
- Distributed training: DDP strategy supported (`configs/trainer/ddp.yaml`)
- HPC cluster support (scripts in `scripts/`)

---

*Stack analysis: 2026-01-19*
