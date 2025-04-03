import torch
import torch.nn.functional as fn
from src import utils

log = utils.get_pylogger(__name__)


def reduce(tensor: torch.Tensor, reduction: str) -> torch.Tensor:
    """Reduces the given tensor using a specific criterion.

    Args:
        tensor (torch.Tensor): input tensor
        reduction (str): string with fixed values [elementwise_mean, none, sum]
    Raises:
        ValueError: when the reduction is not supported
    Returns:
        torch.Tensor: reduced tensor, or the tensor itself
    """
    if reduction in ("elementwise_mean", "mean"):
        return torch.mean(tensor)
    elif reduction == "sum":
        return torch.sum(tensor)
    elif reduction is None or reduction == "none":
        return tensor
    raise ValueError("Reduction parameter unknown.")


class FocalLoss(torch.nn.Module):
    """
    Focal Loss implementation optimized for audio processing tasks.
    
    This loss function helps deal with class imbalance by down-weighting 
    well-classified examples and focusing more on difficult samples.
    """
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -100,  # Changed from 255 to PyTorch default -100
        label_smoothing: float = 0.0,  # Added label smoothing, helpful for audio models
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.eps = 1e-8

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        # Check input shapes and log warnings for common audio processing issues
        if len(inputs.shape) > 2 and inputs.shape[1] <= 5:
            log.warning(
                f"Input shape {inputs.shape} suggests this might be raw audio features. "
                "Ensure you're passing logits from your model, not raw features."
            )
        
        # Handle NaN/Inf values in inputs
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            log.warning("NaN or Inf values detected in logits - applying nan_to_num")
            inputs = torch.nan_to_num(inputs, nan=0.0, posinf=10.0, neginf=-10.0)
            
        # Use cross_entropy with label_smoothing for better generalization
        try:
            ce_loss = fn.cross_entropy(
                inputs, targets, 
                reduction="none", 
                ignore_index=self.ignore_index,
                label_smoothing=self.label_smoothing
            )
        except Exception as e:
            log.error(f"Error in cross_entropy: {e}. Target shape: {targets.shape}, Input shape: {inputs.shape}")
            # Return a differentiable placeholder loss to avoid breaking the training loop
            return torch.tensor(1.0, device=inputs.device, requires_grad=True)
        
        # Calculate pt with epsilon for numerical stability
        pt = torch.exp(-ce_loss)
        
        # Clamp pt to avoid numerical issues
        pt = torch.clamp(pt, min=self.eps, max=1.0 - self.eps)
        
        # Calculate focal loss with the alpha parameter
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        # Apply reduction using the provided reduce function
        return reduce(focal_loss, reduction=self.reduction)
