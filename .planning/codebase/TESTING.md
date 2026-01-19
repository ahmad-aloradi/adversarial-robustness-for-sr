# Testing Patterns

**Analysis Date:** 2026-01-19

## Test Framework

**Runner:**
- pytest (6.0+)
- Config: `pyproject.toml` lines 1-17

**Assertion Library:**
- Built-in pytest assertions
- `torch.testing.assert_close` for tensor comparisons

**Run Commands:**
```bash
make test              # Run fast tests (excludes @pytest.mark.slow)
make test-full         # Run all tests including slow ones
pytest tests/test_models.py -v  # Single test file
pytest -k "not slow"   # Explicit fast tests
```

## Test File Organization

**Location:**
- Separate `tests/` directory at project root
- Helpers in `tests/helpers/`

**Naming:**
- Test files: `test_<module>.py`
- Test functions: `test_<description>`

**Structure:**
```
tests/
├── conftest.py                       # Shared fixtures
├── helpers/
│   ├── __init__.py
│   ├── package_available.py          # Package availability checks
│   ├── run_if.py                     # Conditional skip decorator
│   └── run_sh_command.py             # Shell command runner
├── test_configs.py                   # Config instantiation tests
├── test_eval.py                      # Evaluation pipeline tests
├── test_light_progress_bar.py        # Progress bar tests
├── test_load_losses.py               # Loss loading tests
├── test_load_metrics.py              # Metric loading tests
├── test_models.py                    # Model architecture tests
├── test_multi_datamodule.py          # Multi-dataset datamodule tests
├── test_sweeps.py                    # Hydra sweep tests
├── test_train.py                     # Training pipeline tests
├── test_utils.py                     # Utility function tests
└── test_verification_metrics_fast.py # Verification metrics tests
```

## Test Structure

**Suite Organization:**
```python
# Standard test with fixture
def test_train_config(cfg_train: DictConfig):
    assert cfg_train
    assert cfg_train.datamodule
    HydraConfig().set_config(cfg_train)
    hydra.utils.instantiate(cfg_train.module, _recursive_=False)

# Parametrized test
@pytest.mark.parametrize("model_source_params", _MODULE_SOURCE)
def test_base_module(model_source_params: Dict[str, Any]):
    cfg = {"_target_": "src.modules.models.module.BaseModule", **model_source_params}
    cfg = omegaconf.OmegaConf.create(cfg)
    model = hydra.utils.instantiate(cfg)
    model.eval()
    tensor = torch.randn((2, 3, 224, 224))
    _ = model.forward(tensor)
```

**Slow Test Marking:**
```python
@pytest.mark.slow
def test_train_epoch_double_val_loop(cfg_train):
    """Train 1 epoch with validation loop twice per epoch."""
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.val_check_interval = 0.5
    train(cfg_train)
```

**Conditional Skip Pattern:**
```python
from tests.helpers.run_if import RunIf

@RunIf(min_gpus=1)
def test_train_fast_dev_run_gpu(cfg_train):
    """Run for 1 train, val and test step on GPU."""
    ...

@RunIf(sh=True)
@pytest.mark.slow
def test_experiments(tmp_path):
    """Test running all available experiment configs."""
    ...
```

## Fixtures

**Global Config Fixtures:** (`tests/conftest.py`)

```python
@pytest.fixture(scope="package")
@register_custom_resolvers(
    config_name="train.yaml",
    overrides=_HYDRA_OVERRIDES,
    **_HYDRA_PARAMS,
)
def cfg_train_global() -> DictConfig:
    """Package-scoped training config with test-friendly defaults."""
    with initialize_config_dir(...):
        cfg = compose(config_name="train.yaml", ...)
        with open_dict(cfg):
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 0.01
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.logger = None
    return cfg

@pytest.fixture(scope="function")
def cfg_train(cfg_train_global, tmp_path) -> DictConfig:
    """Function-scoped config with unique temp paths."""
    cfg = cfg_train_global.copy()
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
    yield cfg
    GlobalHydra.instance().clear()
```

**Test Data Fixtures:**
```python
@pytest.fixture(scope="package")
def progress_bar_status_message() -> Dict[str, Any]:
    return {
        "Info": "5/10 [######9     ] 59%",
        "Loss/L1": 112398746192834619827364,
        "Metrics/One": 0.888888,
        # ...
    }
```

**Local Fixtures:**
```python
# In test file
@pytest.fixture()
def multi_sv_cfg() -> Dict:
    return {
        "datasets": {...},
        "loaders": {...},
    }
```

## Mocking

**Framework:** No dedicated mocking framework; uses direct instantiation patterns

**Patterns:**
- Create lightweight test configs with OmegaConf
- Use `tmp_path` fixture for filesystem isolation
- Configure trainer with minimal resources (`fast_dev_run=True`, `cpu` accelerator)

**Config-Based Mocking:**
```python
# Create test config inline
cfg = {
    "_target_": "src.modules.models.module.BaseModule",
    "model_name": "torchvision.models/resnet18",
    "weights": None,
}
cfg = omegaconf.OmegaConf.create(cfg)
model = hydra.utils.instantiate(cfg)
```

**What to Mock:**
- External API calls (via config: `weights=None`)
- File system operations (via `tmp_path`)
- Heavy computations (via `limit_train_batches`, `fast_dev_run`)

**What NOT to Mock:**
- Core model forward passes
- Config loading and composition
- Hydra instantiation (test the real thing)

## Fixtures and Factories

**Test Data:**
```python
# Module-level test data tuples
_MODULE_SOURCE = (
    {"model_name": "torchvision.models/resnet18", "weights": None},
    {"model_name": "timm/tf_efficientnetv2_s", "pretrained": False},
    # ...
)

_TEST_LOSS_CFG = (
    {"_target_": "src.modules.losses.AngularPenaltySMLoss", "embedding_size": 16, "num_classes": 10},
    {"_target_": "torch.nn.CrossEntropyLoss"},
    # ...
)
```

**Location:**
- Inline in test files for test-specific data
- `conftest.py` for shared fixtures

## Coverage

**Requirements:** None enforced (no coverage thresholds configured)

**View Coverage:**
```bash
pytest --cov=src tests/
pytest --cov=src --cov-report=html tests/
```

## Test Types

**Unit Tests:**
- Model forward passes (`test_models.py`)
- Loss/metric instantiation (`test_load_losses.py`, `test_load_metrics.py`)
- Utility functions (`test_utils.py`)
- Progress bar formatting (`test_light_progress_bar.py`)

**Integration Tests:**
- Config composition and instantiation (`test_configs.py`)
- Training pipeline (`test_train.py`)
- Evaluation pipeline (`test_eval.py`)
- Multi-datamodule composition (`test_multi_datamodule.py`)

**E2E Tests:**
- Hydra sweeps (`test_sweeps.py`) - marked `@pytest.mark.slow`
- Full train-eval cycle (`test_train_eval_predict`) - marked `@pytest.mark.slow`

## Common Patterns

**Async Testing:**
- Not used (PyTorch Lightning handles async internally)

**Error Testing:**
```python
def test_multi_sv_no_train_dataset_raises(multi_sv_cfg):
    cfg = OmegaConf.create(multi_sv_cfg)
    cfg.datasets.vox.stages.train = False
    dm = MultiSVDataModule(datasets=cfg.datasets, loaders=cfg.loaders)
    dm.setup(stage="fit")

    with pytest.raises(RuntimeError):
        _ = dm.train_dataloader()
```

**Tensor Comparison:**
```python
# Use torch.testing for numerical comparisons
torch.testing.assert_close(out_fast["eer"], out_ref["eer"], rtol=0, atol=5e-3)
torch.testing.assert_close(out_fast["minDCF"], out_ref["minDCF"], rtol=0, atol=5e-3)
```

**File System Assertions:**
```python
def test_train_resume(tmp_path, cfg_train):
    # ... train first epoch ...
    files = os.listdir(tmp_path / "checkpoints")
    assert "last.ckpt" in files
    assert any(["epoch000" in str(file) for file in files])
```

## RunIf Decorator

**Location:** `tests/helpers/run_if.py`

**Available Conditions:**
- `min_gpus`: Minimum GPU count
- `min_torch` / `max_torch`: PyTorch version range
- `min_python`: Minimum Python version
- `skip_windows`: Skip on Windows
- `sh`: Requires `sh` module (Unix only)
- `tpu`: Requires TPU
- `fairscale`, `deepspeed`: Distributed training libraries
- `wandb`, `neptune`, `comet`, `mlflow`: Logging backends

**Usage:**
```python
@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_epoch_gpu_amp(cfg_train):
    """Train 1 epoch on GPU with mixed-precision."""
    ...

@RunIf(sh=True)
@pytest.mark.slow
def test_hydra_sweep(tmp_path):
    """Test default hydra sweep."""
    ...
```

## Environment Overrides

**Via Environment Variables:**
```python
# In conftest.py
def _get_env_overrides() -> List[str]:
    raw = os.environ.get("PYTEST_HYDRA_OVERRIDES", "")
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]
```

**Usage:**
```bash
PYTEST_HYDRA_OVERRIDES="trainer.max_epochs=2,seed=42" pytest tests/
```

## Test Configuration

**pytest.ini_options (pyproject.toml):**
```toml
[tool.pytest.ini_options]
addopts = [
  "--color=yes",
  "--durations=0",      # Show all test durations
  "--strict-markers",   # Fail on unknown markers
  "--doctest-modules",  # Run doctests
]
filterwarnings = [
  "ignore::DeprecationWarning",
  "ignore::UserWarning",
]
markers = [
  "slow: slow tests",
]
testpaths = "tests/"
```

---

*Testing analysis: 2026-01-19*
