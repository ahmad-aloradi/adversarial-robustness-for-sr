import hydra
import torch
from omegaconf import DictConfig, OmegaConf, ListConfig

def load_loss(loss_cfg: DictConfig) -> torch.nn.Module:
    """Load loss module.

    Args:
        loss_cfg (DictConfig): Loss config.

    Returns:
        torch.nn.Module: Loss module.
    """

    # Convert only list-like weight parameters to tensors
    for param_name, param_value in loss_cfg.items():
        if "weight" in param_name and isinstance(param_value, ListConfig):
            loss_cfg[param_name] = torch.tensor(param_value).float()

    # Instantiate the loss using Hydra
    loss = hydra.utils.instantiate(loss_cfg)

    return loss
