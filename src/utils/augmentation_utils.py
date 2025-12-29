import hydra
import os
from src import utils   # noqa: E501
log = utils.get_pylogger(__name__)


def prepare_speechbrain_augmentation(cfg):
    """Prepare noise and RIR data for augmentation if configured
    """
    if "data_augemntation" in cfg.module and "prepare_noise_data" in cfg.module.data_augemntation:
        log.info(f"{cfg.module.data_augemntation.prepare_noise_data.csv_file} Does not exist. Preparing noise data for augmentation")
        hydra.utils.instantiate(cfg.module.data_augemntation.prepare_noise_data)
        
    if "data_augemntation" in cfg.module and "prepare_rir_data" in cfg.module.data_augemntation:
        log.info(f"{cfg.module.data_augemntation.prepare_rir_data.csv_file} Does not exist. Preparing noise data for augmentation")
        hydra.utils.instantiate(cfg.module.data_augemntation.prepare_rir_data)