# Codebase Structure

**Analysis Date:** 2026-01-19

## Directory Layout

```
adversarial-robustness-for-sr/
├── configs/                    # Hydra configuration hierarchy
│   ├── train.yaml              # Main training config entry
│   ├── eval.yaml               # Evaluation config entry
│   ├── module/                 # LightningModule configs
│   │   ├── sv.yaml             # Speaker verification module
│   │   ├── vpc.yaml            # VoicePrivacy module
│   │   ├── sv_model/           # Model architecture configs
│   │   └── fusion_classifier/  # Multi-modal classifier configs
│   ├── datamodule/             # DataModule configs
│   │   ├── datasets/           # Dataset-specific configs
│   │   ├── loaders/            # DataLoader configs
│   │   └── transforms/         # Augmentation configs
│   ├── callbacks/              # Callback configs
│   ├── trainer/                # Trainer configs (gpu, cpu, ddp)
│   ├── logger/                 # Logger configs (tensorboard, wandb, neptune)
│   ├── experiment/             # Complete experiment recipes
│   │   ├── sv/                 # Speaker verification experiments
│   │   └── vpc/                # VoicePrivacy experiments
│   ├── paths/                  # Path resolution configs
│   ├── hydra/                  # Hydra runtime configs
│   ├── debug/                  # Debug mode configs
│   ├── hparams_search/         # Hyperparameter search configs
│   └── extras/                 # Extra utilities configs
├── src/                        # Python source code
│   ├── train.py                # Training entry point
│   ├── eval.py                 # Evaluation entry point
│   ├── modules/                # PyTorch Lightning modules
│   │   ├── sv.py               # SpeakerVerification LightningModule
│   │   ├── encoder_wrappers.py # Unified encoder interface
│   │   ├── augmentation.py     # GPU-native augmentations
│   │   ├── losses/             # Loss functions
│   │   ├── metrics/            # Metrics (EER, minDCF)
│   │   ├── components/         # Shared module components
│   │   └── models/             # Model architectures
│   ├── datamodules/            # PyTorch Lightning DataModules
│   │   ├── voxceleb_datamodule.py
│   │   ├── cnceleb_datamodule.py
│   │   ├── librispeech_datamodule.py
│   │   ├── vpc_datamodule.py
│   │   ├── multi_sv_datamodule.py
│   │   ├── components/         # Dataset implementations
│   │   │   ├── common.py       # Shared dataclasses (BaseDatasetCols)
│   │   │   ├── utils.py        # Audio processing utilities
│   │   │   ├── voxceleb/       # VoxCeleb dataset
│   │   │   ├── cnceleb/        # CN-Celeb dataset
│   │   │   ├── librispeech/    # LibriSpeech dataset
│   │   │   └── vpc25/          # VoicePrivacy 2025 dataset
│   │   └── preparation/        # Data preparation scripts
│   │       ├── base.py         # Base preparer class
│   │       ├── voxceleb.py     # VoxCeleb metadata preparation
│   │       ├── cnceleb.py      # CN-Celeb metadata preparation
│   │       └── vad.py          # Voice Activity Detection
│   ├── callbacks/              # PyTorch Lightning callbacks
│   │   ├── pruning/            # Model pruning callbacks
│   │   │   ├── prune.py        # MagnitudePruner
│   │   │   ├── bregman/        # Bregman Learning pruning
│   │   │   ├── scheduler.py    # Pruning schedule
│   │   │   └── utils/          # Pruning utilities
│   │   ├── epoch_summary_logger.py
│   │   ├── light_progress_bar.py
│   │   ├── margin_scheduler.py
│   │   └── wandb_callbacks.py
│   └── utils/                  # Shared utilities
│       ├── utils.py            # Core utilities (task_wrapper, instantiate_*)
│       ├── pylogger.py         # Logging setup
│       ├── hf_utils.py         # HuggingFace/WeSpeaker model loading
│       ├── torch_utils.py      # PyTorch utilities
│       ├── augmentation_utils.py
│       └── saving_utils.py
├── tests/                      # Test suite
│   ├── conftest.py             # Pytest fixtures
│   └── test_*.py               # Test files
├── scripts/                    # Utility scripts
│   ├── datasets/               # Dataset preparation scripts
│   └── fabfile.py              # Fabric deployment tasks
├── notebooks/                  # Jupyter notebooks
├── docs/                       # Documentation
├── data/                       # Data directory (gitignored)
├── logs/                       # Training logs and checkpoints
├── requirements.txt            # Python dependencies
├── env.yaml                    # Conda environment
├── Makefile                    # Common commands
├── CLAUDE.md                   # Development guidelines
└── README.md                   # Project documentation
```

## Directory Purposes

**configs/:**
- Purpose: Hydra configuration files for all instantiatable components
- Contains: YAML files organized by component type
- Key files: `train.yaml`, `eval.yaml`, `configs/experiment/sv/*.yaml`

**src/modules/:**
- Purpose: LightningModule implementations and neural network components
- Contains: Main module (`sv.py`), encoder wrappers, losses, metrics
- Key files: `sv.py` (main module), `encoder_wrappers.py`, `metrics/metrics.py`

**src/datamodules/:**
- Purpose: Data loading and preprocessing
- Contains: DataModules, Dataset classes, preparation scripts
- Key files: `voxceleb_datamodule.py`, `components/common.py`

**src/callbacks/:**
- Purpose: Training callbacks for pruning, logging, scheduling
- Contains: Pruning implementations, progress bars, metric loggers
- Key files: `pruning/prune.py`, `pruning/bregman/bregman_pruner.py`

**src/utils/:**
- Purpose: Shared utilities used across the codebase
- Contains: Logging, Hydra helpers, model loading, saving utilities
- Key files: `utils.py`, `pylogger.py`, `hf_utils.py`

**configs/experiment/:**
- Purpose: Complete experiment configurations combining all components
- Contains: Ready-to-run experiment recipes
- Key files: `sv/sv_vanilla.yaml`, `sv/sv_pruning_mag_struct.yaml`, `sv/sv_pruning_bregman.yaml`

**configs/module/sv_model/:**
- Purpose: Speaker encoder architecture configurations
- Contains: Model configs for different backends (WeSpeaker, SpeechBrain, NeMo)
- Key files: `wespeaker_ecapa_tdnn.yaml`, `speechbrain_ecapa_tdnn.yaml`

## Key File Locations

**Entry Points:**
- `src/train.py`: Main training entry, composes config and runs `trainer.fit()`
- `src/eval.py`: Evaluation entry, loads checkpoint and runs `trainer.test()`

**Configuration:**
- `configs/train.yaml`: Default training config composition
- `configs/eval.yaml`: Default evaluation config composition
- `configs/paths/default.yaml`: Path resolution using PROJECT_ROOT

**Core Logic:**
- `src/modules/sv.py`: SpeakerVerification LightningModule (962 lines)
- `src/modules/encoder_wrappers.py`: EncoderWrapper for unified encoder interface
- `src/modules/metrics/metrics.py`: VerificationMetrics (EER, minDCF, curves)

**Data Loading:**
- `src/datamodules/voxceleb_datamodule.py`: VoxCeleb data loading
- `src/datamodules/components/common.py`: BaseDatasetCols and shared dataclasses
- `src/datamodules/components/voxceleb/voxceleb_dataset.py`: VoxCeleb dataset and collate functions

**Pruning:**
- `src/callbacks/pruning/prune.py`: MagnitudePruner callback
- `src/callbacks/pruning/bregman/bregman_pruner.py`: BregmanPruner callback
- `src/callbacks/pruning/scheduler.py`: Pruning sparsity scheduling

**Testing:**
- `tests/conftest.py`: Pytest fixtures for configs and mock data
- `tests/test_*.py`: Individual test files

## Naming Conventions

**Files:**
- Python modules: `snake_case.py` (e.g., `voxceleb_datamodule.py`)
- Config files: `snake_case.yaml` (e.g., `wespeaker_ecapa_tdnn.yaml`)
- Experiment configs: `task_variant.yaml` (e.g., `sv_pruning_mag_struct.yaml`)

**Directories:**
- Python packages: `snake_case` (e.g., `datamodules`, `callbacks`)
- Config groups: `snake_case` (e.g., `sv_model`, `datasets`)

**Classes:**
- LightningModules: `PascalCase` (e.g., `SpeakerVerification`)
- DataModules: `DatasetNameDataModule` (e.g., `VoxCelebDataModule`)
- Callbacks: `PascalCase` (e.g., `MagnitudePruner`)
- Dataclasses: `PascalCase` (e.g., `VoxcelebItem`, `VoxCelebVerificationItem`)

**Config Keys:**
- Use `snake_case` for all config keys
- Use `_target_` for Hydra instantiation targets
- Use `???` for required parameters

## Where to Add New Code

**New Feature (Speaker Verification):**
- Primary code: `src/modules/sv.py` or new file in `src/modules/`
- Config: `configs/module/sv.yaml` or new file in `configs/module/`
- Tests: `tests/test_modules.py` or new file `tests/test_<feature>.py`

**New Dataset:**
- DataModule: `src/datamodules/<dataset>_datamodule.py`
- Dataset/Collate: `src/datamodules/components/<dataset>/<dataset>_dataset.py`
- Preparation: `src/datamodules/components/<dataset>/<dataset>_prep.py`
- Config: `configs/datamodule/datasets/<dataset>.yaml`

**New Model Architecture:**
- Implementation: Load via `src/utils/hf_utils.py` or add to `src/modules/models/`
- Config: `configs/module/sv_model/<model_name>.yaml`

**New Loss Function:**
- Implementation: `src/modules/losses/components/<loss>.py`
- Register in: `src/modules/losses/__init__.py`

**New Callback:**
- Implementation: `src/callbacks/<callback>.py`
- Config: `configs/callbacks/<callback>.yaml`

**New Experiment Recipe:**
- Config: `configs/experiment/sv/<experiment_name>.yaml` (compose existing configs)

**Utilities:**
- Shared helpers: `src/utils/<utility>.py`

## Special Directories

**logs/:**
- Purpose: Training outputs, checkpoints, TensorBoard logs
- Generated: Yes (by Hydra and Lightning)
- Committed: No (gitignored)

**data/:**
- Purpose: Dataset storage (audio files, metadata CSVs)
- Generated: Partially (metadata CSVs generated by preparation scripts)
- Committed: No (gitignored, too large)

**.planning/:**
- Purpose: GSD planning documents for Claude Code
- Generated: Yes (by `/gsd:*` commands)
- Committed: Depends on project policy

**local/:**
- Purpose: Machine-specific configurations and scripts
- Generated: No
- Committed: Partially (may contain local overrides)

---

*Structure analysis: 2026-01-19*
