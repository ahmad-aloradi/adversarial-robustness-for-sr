from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Tuple, Any

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src import utils

log = utils.get_pylogger(__name__)



class MultiSVDataModule(LightningDataModule):
    """Composite speaker verification datamodule that combines multiple dataset-specific datamodules.

    Each configured dataset can independently participate in the training, validation and test stages,
    enabling cross-dataset evaluation scenarios (e.g., train on VoxCeleb and evaluate on CNCeleb).
    """

    def __init__(
        self,
        datasets: Dict[str, DictConfig],
        loaders: Dict[str, DictConfig],
    ) -> None:
        super().__init__()
        self.datasets_cfg: Dict[str, DictConfig] = datasets
        self.loaders_cfg: Dict[str, DictConfig] = loaders

        self._datamodules: Dict[str, LightningDataModule] = {}
        self._train_sets: List[Tuple[str, Dataset]] = []
        self._val_sets: List[Tuple[str, Dataset]] = []
        self._train_collate = None
        self._val_collate = None
        self._test_loaders: "OrderedDict[str, DataLoader]" = OrderedDict()

        self._prepared: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _loader_kwargs(self, key: str) -> Dict[str, Any]:
        if key not in self.loaders_cfg:
            raise KeyError(f"Loader configuration '{key}' is missing in MultiSVDataModule.")
        cfg = self.loaders_cfg[key]
        if isinstance(cfg, DictConfig):
            return deepcopy(OmegaConf.to_container(cfg, resolve=True))
        return dict(cfg)

    def _dataset_cfg(self, name: str) -> DictConfig:
        if name not in self.datasets_cfg:
            raise KeyError(f"Unknown dataset '{name}' in MultiSVDataModule configuration.")
        return self.datasets_cfg[name]

    def _is_enabled(self, name: str, cfg: DictConfig) -> bool:
        assert "enabled" in cfg, f"Dataset '{name}' is missing required 'enabled' configuration field."
        return cfg["enabled"]

    def _stage_enabled(self, name: str, cfg: DictConfig, stage: str) -> bool:
        assert "stages" in cfg, f"Dataset '{name}' is missing required 'stages' configuration block."
        stages_cfg = cfg["stages"]
        if isinstance(stages_cfg, DictConfig):
            stages_cfg = OmegaConf.to_container(stages_cfg, resolve=True)    
        assert isinstance(stages_cfg, dict), f"Expected 'stages' to be a mapping, got {type(stages_cfg).__name__}."
        assert stage in stages_cfg, f"Expected 'stages' to contain '{stage}', got {list(stages_cfg.keys())}."
        return stages_cfg[stage]

    def _get_or_create_datamodule(self, name: str) -> LightningDataModule:
        if name not in self._datamodules:
            cfg = self._dataset_cfg(name)
            datamodule_cfg = cfg.get("datamodule")
            if datamodule_cfg is None:
                raise ValueError(f"Dataset '{name}' is missing the 'datamodule' configuration block.")
            dm = instantiate(datamodule_cfg, _recursive_=False)
            self._datamodules[name] = dm
        return self._datamodules[name]

    # ------------------------------------------------------------------
    # LightningDataModule API
    # ------------------------------------------------------------------
    def prepare_data(self) -> None:
        for name, cfg in self.datasets_cfg.items():
            if not self._is_enabled(name, cfg):
                continue
            dm = self._get_or_create_datamodule(name)
            dm.prepare_data()
            self._prepared[name] = True

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            self._setup_fit_stage()
        if stage in ("test", None):
            self._setup_test_stage()

    def _setup_fit_stage(self) -> None:
        self._train_sets.clear()
        self._val_sets.clear()
        self._train_collate = None
        self._val_collate = None

        for name, cfg in self.datasets_cfg.items():
            if not self._is_enabled(name, cfg):
                continue

            include_train = self._stage_enabled(name, cfg, "train")
            include_val = self._stage_enabled(name, cfg, "val")
            if not (include_train or include_val):
                continue

            dm = self._get_or_create_datamodule(name)
            dm.setup(stage="fit")

            if include_train:
                train_dataset = getattr(dm, "train_data", None)
                if train_dataset is None:
                    loader = dm.train_dataloader()
                    train_dataset = loader.dataset
                    self._train_collate = self._train_collate or loader.collate_fn
                else:
                    loader = dm.train_dataloader()
                    self._train_collate = self._train_collate or loader.collate_fn
                self._train_sets.append((name, train_dataset))
                log.info(f"Added '{name}' training dataset with {len(train_dataset)} samples to MultiSV training set.")

            if include_val:
                val_dataset = getattr(dm, "val_data", None)
                if val_dataset is None:
                    loader = dm.val_dataloader()
                    val_dataset = loader.dataset
                    self._val_collate = self._val_collate or loader.collate_fn
                else:
                    loader = dm.val_dataloader()
                    self._val_collate = self._val_collate or loader.collate_fn
                self._val_sets.append((name, val_dataset))
                log.info(f"Added '{name}' validation dataset with {len(val_dataset)} samples to MultiSV validation set.")

    def _setup_test_stage(self) -> None:
        self._test_loaders = OrderedDict()

        for name, cfg in self.datasets_cfg.items():
            if not self._is_enabled(name, cfg) or not self._stage_enabled(name, cfg, "test"):
                continue

            dm = self._get_or_create_datamodule(name)
            dm.setup(stage="test")
            dataset_loaders = dm.test_dataloader()

            if isinstance(dataset_loaders, dict):
                for key, loader in dataset_loaders.items():
                    alias = f"{name}/{key}" if key != name else key
                    self._test_loaders[alias] = loader
            else:
                self._test_loaders[name] = dataset_loaders

            log.info(f"Registered test loaders for dataset '{name}' in MultiSV datamodule.")

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------
    def train_dataloader(self) -> DataLoader:
        if not self._train_sets:
            raise RuntimeError("No datasets were configured for training in MultiSVDataModule.")

        datasets = [dataset for _, dataset in self._train_sets]
        train_dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
        loader_kwargs = self._loader_kwargs("train")
        collate_fn = self._train_collate
        return DataLoader(train_dataset, collate_fn=collate_fn, **loader_kwargs)

    def val_dataloader(self) -> Optional[DataLoader]:
        if not self._val_sets:
            return None
        datasets = [dataset for _, dataset in self._val_sets]
        val_dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
        loader_kwargs = self._loader_kwargs("valid")
        collate_fn = self._val_collate
        return DataLoader(val_dataset, collate_fn=collate_fn, **loader_kwargs)

    def test_dataloader(self):
        if not self._test_loaders:
            log.warning("No datasets were configured for testing in MultiSVDataModule.")
        return self._test_loaders

    def get_enroll_and_trial_dataloaders(self, dataset_name: str, *args, **kwargs):
        cfg = self._dataset_cfg(dataset_name)
        if not self._is_enabled(dataset_name, cfg) or not self._stage_enabled(dataset_name, cfg, "test"):
            raise ValueError(f"Dataset '{dataset_name}' is not configured for testing in MultiSVDataModule.")

        dm = self._get_or_create_datamodule(dataset_name)
        if not hasattr(dm, "get_enroll_and_trial_dataloaders"):
            raise AttributeError(
                f"Underlying datamodule for '{dataset_name}' does not implement 'get_enroll_and_trial_dataloaders'."
            )
        return dm.get_enroll_and_trial_dataloaders(*args, **kwargs)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def active_datasets(self) -> List[str]:
        return [name for name, cfg in self.datasets_cfg.items() if self._is_enabled(name, cfg)]



# ------------------------------------------------------------------
# Testing
# ------------------------------------------------------------------


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(search_from=__file__, indicator=[".env"], pythonpath=True, dotenv=True,)
    _HYDRA_PARAMS = {"version_base": "1.3",
                     "config_path": str(root / "configs"),
                     "config_name": "train.yaml"}

    @hydra.main(**_HYDRA_PARAMS)
    def test_datamodule(cfg):

        print("Running CNCelMultiSVDataModuleebDataModule test with the following config:")
        print(omegaconf.OmegaConf.to_yaml(cfg))

        # Instantiate the DataModule using the loaded config
        dm: "MultiSVDataModule" = MultiSVDataModule(datasets=cfg.datamodule.datasets, loaders=cfg.datamodule.loaders)

        dm.prepare_data()
        dm.setup(stage='fit')
        
        train_loader = dm.train_dataloader()
        print(f"Train loader has {len(train_loader)} batches.")
        
        val_loader = dm.val_dataloader()
        print(f"Validation loader has {len(val_loader)} batches.")
        
        dm.setup(stage='test')
        test_loaders = dm.test_dataloader()
        print(f"Test returns {len(test_loaders)} loaders.")

        test_loaders['cnceleb'].dataset.__getitem__(0)

        print("Getting enrollment and unique test dataloaders...")
        enroll_loader, test_unique_loader = dm.get_enroll_and_trial_dataloaders('voxceleb', test_filename='veri_test2')
        print(f"Enrollment loader has {len(enroll_loader)} batches.")
        print(f"Unique test loader has {len(test_unique_loader)} batches.")

    test_datamodule()