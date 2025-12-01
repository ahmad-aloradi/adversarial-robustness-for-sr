from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self._datamodules: Dict[str, LightningDataModule] = {}
        self._train_sets: List[Tuple[str, Dataset]] = []
        self._val_sets: List[Tuple[str, Dataset]] = []
        self._train_collate = None
        self._val_collate = None
        self._test_loaders: "OrderedDict[str, DataLoader]" = OrderedDict()
        self.train_data: Optional[Dataset] = None
        self.val_data: Optional[Dataset] = None

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
            # Merge parent's loaders (train/valid) with child's loaders (test/enrollment)
            if 'loaders' in datamodule_cfg:
                merged_loaders = OmegaConf.merge(self.hparams.loaders, datamodule_cfg['loaders'])
            else:
                merged_loaders = self.hparams.loaders
            
            dm = instantiate(datamodule_cfg, loaders=merged_loaders, _recursive_=False)
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
        self.train_data = None
        self.val_data = None

        for name, cfg in self.datasets_cfg.items():
            if not self._is_enabled(name, cfg):
                continue

            include_train = self._stage_enabled(name, cfg, "train")
            include_val = self._stage_enabled(name, cfg, "val")
            if not (include_train or include_val):
                continue

            dm = self._get_or_create_datamodule(name)
            dm.setup(stage="fit")

            stage_configs = (
                ("train", include_train, self._train_sets, "_train_collate", "train_data", "training", "training set"),
                ("val", include_val, self._val_sets, "_val_collate", "val_data", "validation", "validation set"),
            )

            for stage_name, enabled, collection, collate_attr, dataset_attr, log_label, collection_label in stage_configs:
                if not enabled:
                    continue
                self._register_stage_dataset(
                    stage=stage_name,
                    name=name,
                    datamodule=dm,
                    dataset_attr=dataset_attr,
                    collection=collection,
                    collate_attr=collate_attr,
                    log_label=log_label,
                    collection_label=collection_label,
                )

        self.train_data = self._merge_datasets([dataset for _, dataset in self._train_sets])
        self.val_data = self._merge_datasets([dataset for _, dataset in self._val_sets])

    def _register_stage_dataset(
        self,
        *,
        stage: str,
        name: str,
        datamodule: LightningDataModule,
        dataset_attr: str,
        collection: List[Tuple[str, Dataset]],
        collate_attr: str,
        log_label: str,
        collection_label: str,
    ) -> None:
        dataloader_fn = getattr(datamodule, f"{stage}_dataloader")
        loader: DataLoader = dataloader_fn()
        dataset: Optional[Dataset] = getattr(datamodule, dataset_attr, None) or getattr(loader, "dataset", None)
        if dataset is None:
            raise AttributeError(f"Underlying datamodule '{name}' must expose a dataset for stage '{stage}'.")

        if getattr(self, collate_attr) is None:
            setattr(self, collate_attr, loader.collate_fn)

        collection.append((name, dataset))
        log.info(
            f"Added {name} {log_label} dataset with {len(dataset)} samples to MultiSV {collection_label}."
        )

    @staticmethod
    def _merge_datasets(datasets: List[Dataset]) -> Optional[Dataset]:
        if not datasets:
            return None
        return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    def _setup_test_stage(self) -> None:
        self._test_loaders = OrderedDict()

        for name, cfg in self.datasets_cfg.items():
            if not self._is_enabled(name, cfg) or not self._stage_enabled(name, cfg, "test"):
                continue

            dm = self._get_or_create_datamodule(name)
            dm.setup(stage="test")
            loaders = dm.test_dataloader()

            size_before = len(self._test_loaders)
            for alias, loader in self._normalize_test_loaders(name, loaders):
                self._test_loaders[alias] = loader
            registered = len(self._test_loaders) - size_before
            log.info(f"Registered {registered} test loader(s) for dataset '{name}' in MultiSV datamodule.")

    def _normalize_test_loaders(self, name: str, loaders: Any) -> List[Tuple[str, DataLoader]]:
        if isinstance(loaders, Mapping):
            return [
                (key if key == name else f"{name}/{key}", loader)
                for key, loader in loaders.items()
            ]

        if isinstance(loaders, Sequence) and not isinstance(loaders, (str, bytes)):
            return [(f"{name}/{index}", loader) for index, loader in enumerate(loaders)]

        return [(name, loaders)]

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------
    def train_dataloader(self) -> DataLoader:
        if not self._train_sets:
            raise RuntimeError("No datasets were configured for training in MultiSVDataModule.")
        if self.train_data is None:
            raise RuntimeError("MultiSVDataModule.setup('fit') must be called before requesting the train dataloader.")
        loader_kwargs = self._loader_kwargs("train")
        collate_fn = self._train_collate
        return DataLoader(self.train_data, collate_fn=collate_fn, **loader_kwargs)

    def val_dataloader(self) -> Optional[DataLoader]:
        if not self._val_sets or self.val_data is None:
            return None
        loader_kwargs = self._loader_kwargs("valid")
        collate_fn = self._val_collate
        return DataLoader(self.val_data, collate_fn=collate_fn, **loader_kwargs)

    def test_dataloader(self):
        if not self._test_loaders:
            log.warning("No datasets were configured for testing in MultiSVDataModule.")
        return self._test_loaders

    def get_enroll_and_trial_dataloaders(self, dataset_name: str, *args, **kwargs):
        base_name = dataset_name
        alias_test_name: Optional[str] = None
        if "/" in dataset_name:
            base_name, alias_test_name = dataset_name.split("/", 1)
            kwargs.setdefault("test_filename", alias_test_name)

        cfg = self._dataset_cfg(base_name)
        if not self._is_enabled(base_name, cfg) or not self._stage_enabled(base_name, cfg, "test"):
            raise ValueError(f"Dataset '{dataset_name}' is not configured for testing in MultiSVDataModule.")

        dm = self._get_or_create_datamodule(base_name)
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

    @property
    def datasets_cfg(self) -> Dict[str, DictConfig]:
        return self.hparams.datasets

    @property
    def loaders_cfg(self) -> Dict[str, DictConfig]:
        return self.hparams.loaders


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
        assert "datasets" in cfg.datamodule, 'You are probably not using multi_sv.yaml config for datamodule'
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