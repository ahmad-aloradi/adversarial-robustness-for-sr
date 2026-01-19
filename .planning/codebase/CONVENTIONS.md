# Coding Conventions

**Analysis Date:** 2026-01-19

## Naming Patterns

**Files:**
- Python modules: `snake_case.py` (e.g., `voxceleb_datamodule.py`, `margin_loss.py`)
- Test files: `test_<module_name>.py` (e.g., `test_train.py`, `test_models.py`)
- Config files: `snake_case.yaml` (e.g., `default.yaml`, `sv_aug.yaml`)

**Functions:**
- Functions and methods: `snake_case` (e.g., `get_audio_embeddings`, `compute_eer`)
- Private methods: Single underscore prefix `_method_name` (e.g., `_setup_metrics`, `_log_step_metrics`)
- Test functions: `test_<description>` (e.g., `test_train_fast_dev_run`)

**Variables:**
- Local variables: `snake_case` (e.g., `batch_size`, `audio_emb`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `DATASET_DEFAULTS`, `TEST_CHECKPOINT_INTERVAL`)
- Module-level logger: `log` (instantiated via `utils.get_pylogger(__name__)`)

**Classes:**
- Classes: `PascalCase` (e.g., `SpeakerVerification`, `VoxCelebDataModule`)
- Dataclasses: `PascalCase` (e.g., `BaseDatasetCols`, `VoxcelebItem`)
- Callbacks: `PascalCase` ending with descriptive name (e.g., `MagnitudePruner`, `LightProgressBar`)

**Types:**
- Type aliases: `PascalCase` (e.g., `PLLogger`)
- Hydra config targets: Full module path `src.modules.sv.SpeakerVerification`

## Code Style

**Formatting:**
- Tool: Black
- Line length: 79 characters
- Config: `.pre-commit-config.yaml` line 33

**Linting:**
- Tool: Flake8
- Ignored rules: `E203, E402, E501, F401, F841`
- Config: `.pre-commit-config.yaml` lines 68-73

**Import Sorting:**
- Tool: isort
- Profile: `black`
- Multi-line style: Mode 3 (vertical hanging indent)

**Additional Pre-commit Hooks:**
- `pyupgrade`: Auto-upgrade to Python 3.8+ syntax
- `docformatter`: Wrap docstrings at 79 chars
- `bandit`: Security linting (ignores `B101` assert warnings)
- `shellcheck`: Shell script linting

## Import Organization

**Order:**
1. Standard library (`import os`, `from typing import ...`)
2. Third-party packages (`import torch`, `import hydra`)
3. Local modules (`from src.modules import ...`, `from src import utils`)

**Example from `src/train.py`:**
```python
from typing import Any, List, Optional, Tuple
import sys
import os

import hydra
import pyrootutils
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers.logger import Logger as PLLogger

from src import utils
from src.utils.augmentation_utils import prepare_speechbrain_augmentation
```

**Path Aliases:**
- None configured; use absolute imports from `src/`

## Error Handling

**Patterns:**
- Fail-fast philosophy: Let exceptions propagate naturally
- Use assertions for invariants in model code
- Raise errors for missing files instead of logging warnings

**Assertion Pattern:**
```python
# Tensor shape validation
assert scores.shape == labels.shape, "Scores and labels must have the same shape"
assert len(scores.shape) == 1, "Scores and labels must be 1D tensors"
```

**Configuration Validation:**
```python
assert self.hparams.get('loaders') is not None, "VoxCelebDataModule requires 'loaders' config"
```

**Avoid Excessive Try/Except:**
```python
# Preferred: Let it fail if wrong
result = torch.load(path)

# Discouraged: Over-defensive handling
try:
    result = torch.load(path)
except Exception as e:
    logger.error(f"Error: {e}")
    return None
```

## Logging

**Framework:** Python `logging` via `src/utils/pylogger.py`

**Initialization Pattern:**
```python
from src import utils
log = utils.get_pylogger(__name__)
```

**Multi-GPU Safe:** Logger methods are wrapped with `@rank_zero_only` to prevent duplicate logs in distributed training.

**When to Log:**
- Hydra component instantiation: `log.info(f"Instantiating <{cfg.module._target_}>")`
- Important milestones: Training start, testing start, checkpoint saves
- Configuration states: `log.info(f"Seed everything with <{cfg.seed}>")`

**Log Levels:**
- `info`: Normal operation milestones
- `warning`: Missing optional components, fallback behavior
- `error`: Reserved for actual errors

## Comments

**When to Comment:**
- Complex algorithms requiring explanation
- Non-obvious business logic
- Module-level constants explaining magic numbers

**JSDoc/Docstrings:**
- Public APIs: Google-style docstrings with Args/Returns
- Internal methods: Optional, keep brief
- LightningModule hooks: Document when behavior is specific

**Docstring Example:**
```python
def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute Equal Error Rate from verification scores.

    Finds threshold where False Accept Rate equals False Reject Rate
    by scanning thresholds between min and max scores.
    """
```

## Function Design

**Size:** Keep functions focused on single responsibility. Long methods like `SpeakerVerification._epoch_end_common_multi_test` exist but are broken into logical sections with timing logs.

**Parameters:**
- Use `DictConfig` for Hydra configuration objects
- Use dataclasses for structured data (`VoxcelebItem`, `DatasetItem`)
- Use `**kwargs` sparingly, prefer explicit parameters

**Return Values:**
- Dictionaries for complex returns: `{"loss": main_loss, "outputs": outputs}`
- Tuples for multiple simple returns: `Tuple[dict, dict]`
- Type hints on public APIs

## Module Design

**Exports:**
- Use `__init__.py` to expose public interfaces
- Example `src/modules/__init__.py` exports key classes

**Barrel Files:**
- `src/modules/losses/__init__.py` exports `load_loss`
- `src/modules/metrics/__init__.py` exports `load_metrics`

## Dataclasses

**Pattern:** Use frozen dataclasses for column definitions:
```python
@dataclass(frozen=True)
class BaseDatasetCols:
    DATASET: Literal['dataset_name'] = 'dataset_name'
    SPEAKER_ID: Literal['speaker_id'] = 'speaker_id'
    # ...
```

**Mutable Dataclasses:** For runtime data items:
```python
@dataclass
class DatasetItem:
    audio: torch.Tensor
    audio_length: int
    # ...
```

## Hydra/OmegaConf Patterns

**Dependency Injection:** Always use `hydra.utils.instantiate` for components:
```python
# Module instantiation
model: LightningModule = hydra.utils.instantiate(cfg.module, _recursive_=False)

# In LightningModule
self.audio_encoder = EncoderWrapper(
    encoder=instantiate(model.audio_encoder),
    # ...
)
```

**Config Access:**
```python
# Direct access
value = cfg.module.learning_rate

# Safe access with default
seed = cfg.get("seed")
```

**Open Dict Pattern:** For modifying frozen configs:
```python
from omegaconf import open_dict

with open_dict(cfg):
    cfg.trainer.max_epochs = 1
```

## Lightning Module Patterns

**Hyperparameters:**
```python
def __init__(self, model: DictConfig, criterion: DictConfig, ...):
    super().__init__()
    self.save_hyperparameters(logger=False)
```

**Setup Pattern:** Break initialization into private methods:
```python
def __init__(self, ...):
    self._setup_metrics(metrics)
    self._setup_model_components(model)
    self._setup_training_components(criterion, optimizer, lr_scheduler)
```

**Inference Mode:** Use decorator for validation/test steps:
```python
@torch.inference_mode()
def validation_step(self, batch, batch_idx):
    ...
```

## Constants

**Define explicitly at module level:**
```python
# src/modules/sv.py
TEST_CHECKPOINT_INTERVAL = 50_000  # Save trial results every N batches
```

**Use NamedTuples for defaults:**
```python
class VoxcelebDefaults(NamedTuple):
    dataset_name: str = 'voxceleb'
    language: str = None
    sample_rate: float = 16000
```

---

*Convention analysis: 2026-01-19*
