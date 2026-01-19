# Architecture

**Analysis Date:** 2026-01-19

## Pattern Overview

**Overall:** PyTorch Lightning + Hydra ML Research Framework

**Key Characteristics:**
- Hydra-driven configuration with composable YAML configs
- PyTorch Lightning for training/validation/test orchestration
- Dependency injection via `hydra.utils.instantiate()` for all components
- Entry points use `pyrootutils` for path management and `@utils.task_wrapper` decorator
- Separation of concerns: configs define WHAT, code defines HOW

## Layers

**Entry Layer:**
- Purpose: Application entry points and Hydra integration
- Location: `src/train.py`, `src/eval.py`
- Contains: Main functions decorated with `@hydra.main()`, component instantiation
- Depends on: utils, modules, datamodules, callbacks
- Used by: CLI invocation (`python src/train.py`)

**Module Layer (LightningModule):**
- Purpose: Model logic, training/validation/test steps, loss computation
- Location: `src/modules/`
- Contains: `SpeakerVerification` LightningModule, encoder wrappers, losses, metrics
- Depends on: datamodules (batch types), callbacks (pruning), PyTorch
- Used by: Entry layer (train.py, eval.py)

**DataModule Layer:**
- Purpose: Data loading, preprocessing, dataset management
- Location: `src/datamodules/`
- Contains: `VoxCelebDataModule`, `CNCelebDataModule`, `LibriSpeechDataModule`, datasets, collate functions
- Depends on: preparation utilities, audio processing utilities
- Used by: Module layer (via trainer.fit())

**Callbacks Layer:**
- Purpose: Training hooks for pruning, logging, scheduling
- Location: `src/callbacks/`
- Contains: `MagnitudePruner`, `BregmanPruner`, progress bars, gradient monitors
- Depends on: Module layer (LightningModule)
- Used by: Trainer (injected via Hydra)

**Utils Layer:**
- Purpose: Shared utilities, helper functions
- Location: `src/utils/`
- Contains: Logging, Hydra helpers, saving utilities, PyTorch utilities
- Depends on: External libraries (hydra, pytorch_lightning)
- Used by: All other layers

**Config Layer:**
- Purpose: Declarative component configuration
- Location: `configs/`
- Contains: YAML files for all instantiatable components
- Depends on: Nothing (data only)
- Used by: Entry layer (Hydra composition)

## Data Flow

**Training Flow:**

1. `train.py` composes config via Hydra defaults + CLI overrides
2. `hydra.utils.instantiate()` creates DataModule, Module, Callbacks, Trainer
3. `trainer.fit(model, datamodule)` starts training
4. DataModule provides batches as dataclass objects (e.g., `VoxcelebItem`)
5. LightningModule.training_step() receives batch, computes embeddings via `EncoderWrapper`, computes loss
6. Callbacks execute hooks (pruning, logging) at epoch boundaries
7. Checkpoints saved based on `ModelCheckpoint` callback config

**Inference/Test Flow:**

1. `eval.py` loads checkpoint, composes config
2. `on_test_start()` precomputes enrollment/test embeddings
3. `test_step()` computes cosine similarity scores per trial
4. Optionally applies AS-Norm score normalization using cohort embeddings
5. Metrics (EER, minDCF) computed in `_epoch_end_common_multi_test()`
6. Artifacts (scores CSV, embeddings, metrics JSON) saved to `test_artifacts/`

**State Management:**
- Config state: OmegaConf DictConfig, immutable during run
- Model state: PyTorch state_dict, managed by Lightning checkpointing
- Metric state: TorchMetrics accumulation, reset per epoch
- Pruning state: PyTorch pruning masks, saved in checkpoint

## Key Abstractions

**SpeakerVerification (LightningModule):**
- Purpose: Main training/evaluation logic for speaker verification
- Location: `src/modules/sv.py`
- Pattern: Hydra-instantiated components (encoder, classifier, losses, metrics)
- Key methods: `forward()`, `training_step()`, `test_step()`, `configure_optimizers()`

**EncoderWrapper:**
- Purpose: Unified interface for different encoder backends (SpeechBrain, WeSpeaker, NeMo)
- Location: `src/modules/encoder_wrappers.py`
- Pattern: Strategy pattern - dispatches to backend-specific forward methods
- Key methods: `forward()`, `_forward_speechbrain()`, `_forward_wespeaker()`, `_forward_generic()`

**BaseDataset / DataModule:**
- Purpose: Standardized data loading across datasets
- Location: `src/datamodules/components/common.py`, `src/datamodules/*_datamodule.py`
- Pattern: Template method - common interface, dataset-specific implementations
- Key types: `VoxcelebItem`, `VoxCelebVerificationItem`, `BaseDatasetCols`

**MagnitudePruner / BregmanPruner:**
- Purpose: Model compression via weight pruning
- Location: `src/callbacks/pruning/prune.py`, `src/callbacks/pruning/bregman/`
- Pattern: Observer pattern via Lightning callbacks
- Key methods: `on_train_epoch_start()`, `on_train_epoch_end()`, `_apply_pruning()`

**VerificationMetrics:**
- Purpose: Compute EER, minDCF, and verification curves
- Location: `src/modules/metrics/metrics.py`
- Pattern: TorchMetrics Metric subclass with stateful accumulation
- Key methods: `update()`, `compute()`, `plot_curves()`

## Entry Points

**Training Entry (`src/train.py`):**
- Location: `src/train.py`
- Triggers: `python src/train.py [overrides]`
- Responsibilities:
  - Compose configuration via Hydra
  - Instantiate all components (model, datamodule, callbacks, trainer)
  - Execute `trainer.fit()` and optionally `trainer.test()`
  - Log hyperparameters and metadata
  - Return metric for hyperparameter optimization

**Evaluation Entry (`src/eval.py`):**
- Location: `src/eval.py`
- Triggers: `python src/eval.py ckpt_path=/path/to/ckpt.ckpt [predict=true]`
- Responsibilities:
  - Load checkpoint and compose config
  - Execute `trainer.test()` or `trainer.predict()`
  - Save predictions if predict=true

**Config Entry (`configs/train.yaml`):**
- Location: `configs/train.yaml`
- Triggers: Referenced by `@hydra.main()` in train.py
- Responsibilities:
  - Define default config composition via `defaults:` list
  - Set top-level parameters (seed, tags, train/test flags)

## Error Handling

**Strategy:** Fail-fast, minimal defensive code

**Patterns:**
- Assertions for tensor shape invariants (`assert embeddings.ndim == 2`)
- Let exceptions propagate naturally from PyTorch/Lightning
- Use `@utils.task_wrapper` to log exceptions before re-raising
- Hydra validates required config parameters at composition time

## Cross-Cutting Concerns

**Logging:**
- `src/utils/pylogger.py` provides `get_pylogger(__name__)`
- Lightning loggers (TensorBoard, WandB, Neptune) configured via `configs/logger/`
- `@rank_zero_only` decorator for distributed training

**Validation:**
- Config validation handled by Hydra (required fields marked with `???`)
- Shape assertions in forward passes
- DataModule checks for required artifacts in `_artifacts_ready()`

**Authentication:**
- Not applicable (research codebase, no user authentication)
- External service credentials via environment variables (`.env` file)

---

*Architecture analysis: 2026-01-19*
