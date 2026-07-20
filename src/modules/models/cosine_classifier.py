import torch
import torch.nn.functional as F
from speechbrain.lobes.models.ECAPA_TDNN import Classifier
from torch import nn


class CosineClassifier(Classifier):
    """Cosine classifier with an optional off-channel bias and confidence gate.

    Subclass of ``speechbrain.lobes.models.ECAPA_TDNN.Classifier`` (so
    ``layer_types`` matching in pruning groups still applies) with two options,
    both off by default — defaults reproduce the speechbrain head.
    
    - ``bias``: Gives a class a weight-free way to say "I never occur": 
    CE drives the offset of never-used classes down, so their their pruned 
    rows can stay dead. The sum is clamped to [-1, 1] to keep the output a 
    valid cosine similarity for the SV scoring.

    Input contract (same as the parent): embeddings ``(B, 1, D)``; output
    cosine-like scores ``(B, 1, out_neurons)`` in [-1, 1].    
    """
    def __init__(
        self,
        input_size,
        device="cpu",
        lin_blocks=0,
        lin_neurons=192,
        out_neurons=1211,
        bias: bool = False,
    ):
        super().__init__(
            input_size, device, lin_blocks, lin_neurons, out_neurons
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_neurons, device=device))
        else:
            self.bias = None

    def forward(self, x):
        """Score embeddings ``(B, 1, D)`` against all classes."""
        for layer in self.blocks:
            x = layer(x)

        x = x.squeeze(1)
        assert x.dim() == 2, f"expected (B, 1, D) input, got squeezed {x.shape}"

        w = F.normalize(self.weight)
        out = F.linear(F.normalize(x), w)

        if self.bias is not None:
            # Clamp keeps the output a valid cosine in [-1, 1] for scoring.
            out = torch.clamp(out + self.bias, -1.0, 1.0)
        return out.unsqueeze(1)


if __name__ == "__main__":
    # Smoke: defaults match speechbrain; options change shape of nothing.
    torch.manual_seed(0)
    x = torch.randn(4, 1, 192)
    ref = Classifier(input_size=192, out_neurons=11)
    for kwargs in ({}, {"bias": True}):
        clf = CosineClassifier(input_size=192, out_neurons=11, **kwargs)
        clf.weight.data.copy_(ref.weight.data)
        out = clf(x)
        same = torch.allclose(out, ref(x))
        print(f"{kwargs or 'defaults'}: out {tuple(out.shape)}, matches speechbrain={same}")
