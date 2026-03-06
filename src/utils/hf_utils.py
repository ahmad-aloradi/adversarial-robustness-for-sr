import importlib
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.utils import pylogger

log = pylogger.get_pylogger(__name__)

_WESPEAKER_REGISTRY = {
    "campplus": "wespeaker.models.campplus.CAMPPlus",
    "resnet34": "wespeaker.models.resnet.ResNet34",
    "resnet152": "wespeaker.models.resnet.ResNet152",
    "resnet221": "wespeaker.models.resnet.ResNet221",
    "resnet293": "wespeaker.models.resnet.ResNet293",
    "redimnetb4": "wespeaker.models.redimnet.ReDimNetB4",
    "redimnetb5": "wespeaker.models.redimnet.ReDimNetB5",
    "redimnetb6": "wespeaker.models.redimnet.ReDimNetB6",
    "dfresnet237": "wespeaker.models.gemini_dfresnet.Gemini_DF_ResNet237",
    "eres2net34_aug": "wespeaker.models.eres2net.ERes2Net34_aug",
    "ecapa_tdnn_glob_c512": "wespeaker.models.ecapa_tdnn.ECAPA_TDNN_GLOB_c512",
    "ecapa_tdnn_glob_c1024": "wespeaker.models.ecapa_tdnn.ECAPA_TDNN_GLOB_c1024",
}


def load_nemo_model(model_name: str, **kwargs: Any) -> nn.Module:
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
        model_name, **kwargs
    )
    log.info(f"Loaded NeMo model '{model_name}'")
    return model


def _instantiate_from_registry(
    model_name: str, model_args: Dict[str, Any]
) -> nn.Module:
    assert (
        model_name in _WESPEAKER_REGISTRY
    ), f"Unknown model '{model_name}'. Available: {list(_WESPEAKER_REGISTRY.keys())}"
    module_path, class_name = _WESPEAKER_REGISTRY[model_name].rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)(
        **model_args
    )


def _load_lightning_encoder_weights(model: nn.Module, ckpt_path: str) -> None:
    """Extract audio_encoder.encoder.0.* from a Lightning checkpoint and load
    into model."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)[
        "state_dict"
    ]
    prefix = "audio_encoder.encoder.0."
    encoder_sd = {
        k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)
    }
    assert encoder_sd, f"No keys with prefix '{prefix}' in {ckpt_path}"
    model.load_state_dict(encoder_sd, strict=True)


def _extract_classifier_state_dict(
    ckpt_path: str, prefix: str = "classifier."
) -> dict[str, torch.Tensor]:
    """Extract classifier keys from a Lightning checkpoint."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)[
        "state_dict"
    ]
    return {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}


def load_pretrained_classifier(
    ckpt_path: str,
    input_size: int,
    out_neurons: int,
) -> nn.Module:
    """Load a classifier from a Lightning checkpoint if shapes match.

    Instantiates a SpeechBrain Classifier, then attempts to load weights from
    the checkpoint. If the checkpoint has no classifier keys or the shapes
    don't match (e.g. different number of classes), returns the freshly-
    initialised classifier instead.
    """
    from speechbrain.lobes.models.ECAPA_TDNN import Classifier

    classifier = Classifier(input_size=input_size, out_neurons=out_neurons)

    cls_sd = _extract_classifier_state_dict(ckpt_path)
    if not cls_sd:
        log.warning("No classifier keys in checkpoint, using random init")
        return classifier

    # Check shape compatibility before loading
    model_sd = classifier.state_dict()
    shapes_match = all(
        k in model_sd and model_sd[k].shape == v.shape
        for k, v in cls_sd.items()
    )

    if shapes_match:
        classifier.load_state_dict(cls_sd, strict=True)
        log.info(f"Loaded pretrained classifier from {ckpt_path}")
    else:
        log.critical(
            f"Classifier shape mismatch (checkpoint vs config), "
            f"using random init for {out_neurons} classes"
        )

    return classifier


def _load_wespeaker_pretrained(
    model_dir: str, checkpoint_path: str
) -> nn.Module:
    import yaml
    from wespeaker.models.speaker_model import get_speaker_model

    # Config filename varies across repos (e.g. "config.yaml" vs "voxceleb_resnet34.yaml").
    yaml_files = [f for f in os.listdir(model_dir) if f.endswith(".yaml")]
    assert (
        len(yaml_files) == 1
    ), f"Expected 1 YAML in {model_dir}, found: {yaml_files}"
    with open(os.path.join(model_dir, yaml_files[0])) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)  # nosec B506

    model = get_speaker_model(config["model"])(**config["model_args"])
    # Drop the projection head — it's absent from the backbone architecture.
    checkpoint = {
        k: v
        for k, v in torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        ).items()
        if not k.startswith("projection")
    }
    model.load_state_dict(checkpoint, strict=True)
    return model


def load_wespeaker_model(
    model_name: str,
    repo_id: str = None,
    checkpoint_filename: str = None,
    local_ckpt_path: str = None,
    model_args: Dict[str, Any] = None,
    **kwargs: Any,
) -> nn.Module:
    """Load a WeSpeaker backbone.

    Provide exactly one of local_ckpt_path or repo_id (or neither for random-
    weight initialisation, which requires model_args).
    """
    assert not (
        local_ckpt_path is not None and repo_id is not None
    ), "Provide either local_ckpt_path or repo_id, not both"

    if local_ckpt_path is not None:
        assert (
            model_args is not None
        ), "model_args required with local_ckpt_path"
        model = _instantiate_from_registry(model_name, model_args)
        _load_lightning_encoder_weights(model, local_ckpt_path)
        log.info(
            f"Loaded '{model_name}' from local checkpoint: {local_ckpt_path}"
        )

    elif repo_id is not None:
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(repo_id=repo_id)
        checkpoint_path = os.path.join(
            model_dir, checkpoint_filename or "avg_model.pt"
        )
        model = _load_wespeaker_pretrained(model_dir, checkpoint_path)
        log.info(f"Loaded pretrained WeSpeaker model from {repo_id}")

    else:
        assert (
            model_args is not None
        ), "model_args required when no checkpoint source is given"
        model = _instantiate_from_registry(model_name, model_args)
        log.info(f"Instantiated '{model_name}' without pretrained weights")

    return model


def load_huggingface_model(model_name: str, **kwargs: Any) -> nn.Module:
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_name, **kwargs)
    log.info(f"Loaded Hugging Face model '{model_name}'")
    return model


def load_pretrained_model(
    filename: str,
    repo_id: str,
    cache_dir: Optional[str] = None,
    map_location: str = "cuda",
    *args,
    **kwargs,
) -> nn.Module:
    """Load a TorchScript model from Hugging Face Hub."""
    from huggingface_hub import hf_hub_download

    return torch.jit.load(
        hf_hub_download(
            repo_id=repo_id, filename=filename, cache_dir=cache_dir
        ),
        map_location=map_location,
    )
