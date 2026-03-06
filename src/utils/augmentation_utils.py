import os

import hydra

from src import utils  # noqa: E501

log = utils.get_pylogger(__name__)


def _remove_csv_if_exists(csv_file: str) -> None:
    if os.path.exists(csv_file):
        log.info(f"Removing existing augmentation CSV: {csv_file}")
        os.remove(csv_file)


def prepare_speechbrain_augmentation(cfg):
    """Prepare noise and RIR data for augmentation if configured."""
    if (
        "data_augmentation" in cfg.module
        and cfg.module.data_augmentation is not None
    ):
        if "prepare_noise_data" in cfg.module.data_augmentation:
            csv_file = cfg.module.data_augmentation.prepare_noise_data.csv_file
            _remove_csv_if_exists(csv_file)
            log.info(f"Preparing noise data for augmentation: {csv_file}")
            hydra.utils.instantiate(
                cfg.module.data_augmentation.prepare_noise_data
            )

        if "prepare_rir_data" in cfg.module.data_augmentation:
            csv_file = cfg.module.data_augmentation.prepare_rir_data.csv_file
            _remove_csv_if_exists(csv_file)
            log.info(f"Preparing RIR data for augmentation: {csv_file}")
            hydra.utils.instantiate(
                cfg.module.data_augmentation.prepare_rir_data
            )
