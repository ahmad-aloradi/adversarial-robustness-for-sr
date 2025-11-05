import inspect
import torch
from torch import nn


class EncoderWrapper(nn.Module):
    """
    A comprehensive wrapper for audio encoder models to provide a unified interface.

    This wrapper abstracts away the differences between various encoder
    architectures by first identifying the model type (e.g., SpeechBrain, NeMo)
    and then handling the specific forward pass logic for "pretrained"
    (high-level) versus "from-scratch" (standard) versions.
    """
    def __init__(
        self,
        encoder: nn.Module,
        audio_processor: nn.Module,
        audio_processor_normalizer: nn.Module
    ):
        super().__init__()
        self.encoder = encoder
        self.audio_processor = audio_processor
        self.audio_processor_normalizer = audio_processor_normalizer

        # A high-level pretrained model is one that does its own feature extraction.
        # We infer this if the audio_processor is just an identity function.
        self._is_high_level_pretrained = isinstance(self.audio_processor, nn.Identity)
        if self._is_high_level_pretrained:
            assert isinstance(self.audio_processor_normalizer, nn.Identity), 'Pretrained models expect identity normalizer.'
        self._model_type = self._get_model_type()

    def _get_model_type(self) -> str:
        """Determines the model's library/type based on its module path or attributes."""
        module_name = self.encoder.__class__.__module__
        if "speechbrain" in module_name:
            return "speechbrain"
        if "nemo" in module_name:
            return "nemo"
        if hasattr(self.encoder, 'code'):
            return "torchscript"
        return "generic"

    def forward(self, wavs: torch.Tensor, wav_lens: torch.Tensor) -> torch.Tensor:
        """
        Processes raw audio waveforms and returns speaker embeddings by dispatching
        to the correct forward method based on the model type.
        """
        # Dispatch to the appropriate forward method based on model type
        if self._model_type == "speechbrain":
            embeddings = self._forward_speechbrain(wavs, wav_lens)
        elif self._model_type == "nemo":
            embeddings = self._forward_nemo(wavs, wav_lens)
        elif self._model_type == "torchscript":
            embeddings = self._forward_torchscript(wavs, wav_lens)
        else:  # Generic nn.Module
            embeddings = self._forward_generic(wavs, wav_lens)

        # Ensure the final output is a 2D tensor [batch, embedding_dim]
        assert embeddings.ndim == 2, f"Expected 2D embeddings, got {embeddings.shape}"
        return embeddings

    def _forward_speechbrain(self, wavs: torch.Tensor, wav_lens: torch.Tensor) -> torch.Tensor:
        """Handles both high-level and standard SpeechBrain models."""
        if self._is_high_level_pretrained:
            # Pretrained models (e.g., EncoderClassifier) handle their own processing
            normalized_lens = wav_lens / max(wav_lens) if max(wav_lens) > 1 else wav_lens
            return self.encoder.encode_batch(wavs=wavs, wav_lens=normalized_lens).squeeze(1)
        else:
            # Standard models require external feature processing
            return self._forward_generic(wavs, wav_lens).squeeze(1)

    def _forward_nemo(self, wavs: torch.Tensor, wav_lens: torch.Tensor) -> torch.Tensor:
        """Handles both high-level and standard NeMo models."""
        # Standard models require external feature processing
        features = self.audio_processor(wavs)
        if not isinstance(self.audio_processor_normalizer, nn.Identity):
            features = self.audio_processor_normalizer(features, lengths=wav_lens)
        
        # NeMo's standard forward returns a tuple (logits, embeddings)
        _, embeddings = self.encoder(input_signal=features, input_signal_length=wav_lens)
        return embeddings

    def _forward_torchscript(self, wavs: torch.Tensor, wav_lens: torch.Tensor) -> torch.Tensor:
        """Handles TorchScript models, assumed to be pretrained."""
        # Per user assumption, TorchScript models are treated as high-level.        
        with torch.jit.optimized_execution(False):
            return self.encoder(wavs)

    def _forward_generic(self, wavs: torch.Tensor, wav_lens: torch.Tensor) -> torch.Tensor:
        """A generic forward pass for standard nn.Module encoders."""
        # 1. Feature Extraction
        features = self.audio_processor(wavs)
        
        # 2. Feature Normalization
        if not isinstance(self.audio_processor_normalizer, nn.Identity):
            norm_sig = inspect.signature(self.audio_processor_normalizer.forward)
            if 'lengths' in norm_sig.parameters:
                # SpeechBrain normalizers expect relative lengths.
                wav_lens = wav_lens / wavs.shape[1] if wavs.shape[1] > 0 else wav_lens
                features = self.audio_processor_normalizer(features, lengths=wav_lens)
            else:
                features = self.audio_processor_normalizer(features)

        # 3. Encoding
        sig = inspect.signature(self.encoder.forward)
        possible_len_args = ('length', 'lengths', 'wav_lens', 'input_signal_length', 'length', 'x_len', 'lens')
        len_arg_name = next((arg for arg in possible_len_args if arg in sig.parameters), None)
        # Pass lengths-like if the model supports it.
        if len_arg_name:
            return self.encoder(features, **{len_arg_name: wav_lens})
        return self.encoder(features)


class SequentialEncoder(nn.Module):
    """
    A wrapper to chain a feature extractor (like WavLM) and a downstream
    speaker encoder (like ECAPA-TDNN).

    The feature extractor is frozen by default.
    """

    def __init__(self, 
                 feature_extractor: nn.Module,
                 speaker_encoder: nn.Module,
                 freeze_feature_extractor: bool = True):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.speaker_encoder = speaker_encoder

        if freeze_feature_extractor:
            self.feature_extractor.eval()
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def forward(self, wavs: torch.Tensor, wav_lens: torch.Tensor = None) -> torch.Tensor:
        """
        Extract features and then encode them to get speaker embeddings.

        Args:
            wavs: Raw audio waveforms.
            wav_lens: Length of each waveform.

        Returns:
            Speaker embeddings.
        """
        # Freeze the feature extractor during the forward pass as well
        if self.feature_extractor.training:
            self.feature_extractor.eval()

        with torch.no_grad():
            # The raw output from a transformers model is a dictionary-like object.
            # The features are in the 'last_hidden_state' attribute.
            output = self.feature_extractor(wavs)

            if hasattr(output, 'last_hidden_state'):
                feats = output.last_hidden_state
            elif isinstance(output, tuple):
                # Fallback for speechbrain-style tuple outputs
                feats = output[-1]
            else:
                # Assuming the output is the tensor itself
                feats = output

        # Now, pass the features to the speaker encoder
        # The speaker encoder might expect features and their lengths
        # We assume the feature extractor doesn't change the temporal dimension significantly
        # or the speaker encoder can handle variable length sequences.
        embeddings = self.speaker_encoder(feats)
        if embeddings.ndim == 3:
            assert embeddings.size(1) == 1, "Expected single embedding per utterance"
            embeddings = embeddings.squeeze(1)

        return embeddings
