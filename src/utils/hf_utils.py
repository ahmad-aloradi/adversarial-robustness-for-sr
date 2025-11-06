from typing import Any, Dict, Optional
import torch
import torch.nn as nn

from src.utils import pylogger
log = pylogger.get_pylogger(__name__)


def load_nemo_model(model_name: str, **kwargs: Any) -> nn.Module:
    """
    Factory function to load a pretrained NeMo speaker model.

    This provides a stable target for Hydra to instantiate NeMo models
    that are loaded via the .from_pretrained() class method.

    Args:
        model_name: The name of the pretrained model on NGC or Hugging Face,
                    e.g., "nvidia/speakerverification_en_titanet_large".
        **kwargs: Additional keyword arguments to pass to from_pretrained.

    Returns:
        The instantiated NeMo model as an nn.Module.
    """
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        raise ImportError(
            "NeMo toolkit not found. Please install it with: "
            "pip install nemo_toolkit[asr]"
        )

    model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name, **kwargs)
    log.info(f"Successfully loaded NeMo model '{model_name}'")
    return model


def load_wespeaker_model(
    model_name: str,
    repo_id: str,
    checkpoint_filename: str,
    model_args: Dict[str, Any],
    **kwargs: Any
) -> nn.Module:
    """
    Generic factory function to load a pretrained WeSpeaker model.

    Args:
        model_name: The architecture name (e.g., "campplus", "resnet").
        repo_id: The Hugging Face repository ID.
        checkpoint_filename: The name of the PyTorch checkpoint file.
        model_args: Dictionary of arguments to pass to the model constructor.
        **kwargs: Additional keyword arguments (e.g., map_location).

    Returns:
        The instantiated PyTorch model with pretrained weights.
    """
    # Registry of available WeSpeaker models
    model_registry = {
        "campplus": "wespeaker.models.campplus.CAMPPlus",
        "resnet293": "wespeaker.models.resnet.ResNet293",
        "redimnetb4": "wespeaker.models.redimnet.ReDimNetB4",
        "redimnetb5": "wespeaker.models.redimnet.ReDimNetB5",
        "redimnetb6": "wespeaker.models.redimnet.ReDimNetB6",
        "dfresnet237": "wespeaker.models.gemini_dfresnet.Gemini_DF_ResNet237",
        "eres2net34_aug": "wespeaker.models.eres2net.ERes2Net34_aug",
        "ecapa_tdnn_glob_c1024": "wespeaker.models.ecapa_tdnn.ECAPA_TDNN_GLOB_c1024",
    }
    projection_inc_state_dict = ["resnet293", "ecapa_tdnn_glob_c1024"]

    if model_name not in model_registry:
        raise ValueError(f"Unknown WeSpeaker model name: {model_name}. Available: {list(model_registry.keys())}")

    try:
        from huggingface_hub import hf_hub_download
        import importlib
    except ImportError:
        raise ImportError("Please install huggingface_hub: pip install huggingface_hub")

    # Dynamically import the model class
    module_path, class_name = model_registry[model_name].rsplit('.', 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)

    # Instantiate the model with its specific arguments
    model = model_class(**model_args)

    if not (repo_id is None or checkpoint_filename is None):
        # Download the checkpoint and load the weights
        model_path = hf_hub_download(repo_id=repo_id, filename=checkpoint_filename)
        state_dict = torch.load(model_path, 
                                map_location=kwargs.get('map_location', 'cuda' if torch.cuda.is_available() else 'cpu'))
    
        # special handling for resnet293 to remove incompatible keys
        if model_name in projection_inc_state_dict:
            del state_dict['projection.weight'] 

        model.load_state_dict(state_dict)

    log.info(f"Successfully loaded WeSpeaker model '{model_name}' from {repo_id}")
    return model


def load_huggingface_model(model_name: str, **kwargs: Any) -> nn.Module:
    """
    Factory function to load a pretrained model directly from Hugging Face using
    the transformers library.

    Args:
        model_name: The name of the pretrained model on the Hugging Face Hub,
                    e.g., "microsoft/wavlm-base-plus".
        **kwargs: Additional keyword arguments to pass to from_pretrained.

    Returns:
        The instantiated Hugging Face model as an nn.Module.
    """
    try:
        from transformers import AutoModel
    except ImportError:
        raise ImportError(
            "Transformers library not found. Please install it with: "
            "pip install transformers"
        )

    model = AutoModel.from_pretrained(model_name, **kwargs)
    log.info(f"Successfully loaded Hugging Face model '{model_name}'")
    return model


def load_pretrained_model(
    filename: str,
    repo_id: str,
    cache_dir: Optional[str] = None,
    map_location: str = 'cuda',
    *args,
    **kwargs
) -> nn.Module:
    """Hydra factory that returns a loaded TorchScript model directly.

    This mirrors the behavior of ``PretrainedModelLoader.__call__`` but allows
    Hydra's ``instantiate`` to yield the final ``nn.Module`` in a single step.

    Args:
        filename: Name of the file in the Hugging Face repo (e.g. 'ecapa2.pt').
        repo_id: Hugging Face repository ID (e.g. 'user/repo').
        cache_dir: Optional local cache directory.
        map_location: Device mapping for ``torch.jit.load``.
        *args, **kwargs: Ignored extra arguments for forward compatibility.

    Returns:
        Loaded ``nn.Module`` (TorchScript) ready for inference.
    """
    from huggingface_hub import hf_hub_download
    model_file = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
    model = torch.jit.load(model_file, map_location=map_location)
    return model