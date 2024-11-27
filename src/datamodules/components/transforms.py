from typing import Any

import hydra
from omegaconf import DictConfig
from torch import Tensor

class TransformsWrapper:
    def __init__(self, transforms_cfg: DictConfig) -> None:
        """TransformsWrapper module for applying GPU-based audio augmentations.
        Handles batched audio data of fixed length.

        Args:
            transforms_cfg (DictConfig): Transforms config containing order and parameters
                for audio augmentations.
        """
        if not transforms_cfg.get("order"):
            raise RuntimeError(
                "TransformsWrapper requires param <order>, i.e. "
                "order of augmentations as List[augmentation name]"
            )

        self.transforms = []
        for transform_name in transforms_cfg.get("order"):
            transform = hydra.utils.instantiate(
                transforms_cfg.get(transform_name), _convert_="object"
            )
            self.transforms.append(transform)

    def __call__(
        self, 
        audio: Tensor,  # Shape: (batch_size, num_channels, audio_length)
        sample_rate: int,
        **kwargs: Any
    ) -> Tensor:
        """Apply GPU-based audio transformations in sequence to batched data.

        Args:
            audio: Input audio tensor of shape (batch_size, num_channels, audio_length)
                Already on GPU
            sample_rate: Sample rate of the audio
            kwargs: Additional arguments passed to transforms

        Returns:
            Tensor: Transformed audio
        """
        transformed_audio = audio.clone()
        for transform in self.transforms:
            transformed_audio = transform(transformed_audio, sample_rate, **kwargs)
        return transformed_audio