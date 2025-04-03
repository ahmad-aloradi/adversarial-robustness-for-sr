import logging
import numpy as np
import torch
import re
from typing import List, Union, Dict, Any, Optional
from collections import defaultdict
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import grad_norm
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import io

logger = logging.getLogger(__name__)


class GradientMonitor(Callback):
    """
    Monitor for tracking gradient norms and statistics during training.
    
    Parameters:
        log_freq (int): Frequency of gradient tracking in steps
        track_norm_types (List[Union[int, float]]): Types of norms to track (1=L1, 2=L2, etc.)
        visualize_on_end (bool): Generate summary plots at end of training
        track_per_layer (bool): Whether to track per-layer gradient statistics
        track_stats (bool): Whether to track gradient statistics (mean, std, max, min)
        max_layers_to_track (Optional[int]): Maximum number of layers to track (None = all)
        include_patterns (Optional[List[str]]): Regex patterns to include layers (None = all)
        exclude_patterns (Optional[List[str]]): Regex patterns to exclude layers (None = none)
        verbose (bool): Enable verbose logging
        max_history_length (int): Maximum number of gradient history entries to store
    """
    
    def __init__(
        self,
        log_freq: int = 1,
        track_norm_types: List[Union[int, float]] = [2],
        visualize_on_end: bool = True,
        track_per_layer: bool = True,
        track_stats: bool = True,
        max_layers_to_track: Optional[int] = None,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        verbose: bool = True,
        max_history_length: int = 1000
    ):
        super().__init__()
        self.log_freq = log_freq
        self.track_norm_types = track_norm_types
        self.visualize_on_end = visualize_on_end
        self.track_per_layer = track_per_layer
        self.track_stats = track_stats
        self.max_layers_to_track = max_layers_to_track
        self.include_patterns = include_patterns
        self.exclude_patterns = exclude_patterns
        self.verbose = verbose
        self.max_history_length = max_history_length
        self.gradient_history = defaultdict(list)
    
    def _get_norm_value(self, norm_val) -> float:
        """Extract float value from various norm value formats."""
        if isinstance(norm_val, dict):
            if any(k.endswith('_norm_total') for k in norm_val.keys()):
                total_key = next((k for k in norm_val.keys() if k.endswith('_norm_total')), None)
                if total_key and isinstance(norm_val[total_key], torch.Tensor):
                    return norm_val[total_key].item()
            elif "value" in norm_val:
                return norm_val["value"]
            elif "mean" in norm_val:
                return norm_val["mean"]
        elif isinstance(norm_val, torch.Tensor):
            return norm_val.item()
        elif isinstance(norm_val, (int, float)):
            return float(norm_val)
        else:
            raise ValueError(f"Unknown norm value format: {type(norm_val)}")
            # logger.warning(f"Unknown norm value format: {type(norm_val)}. Using default 0.0")
            # return 0.0
    
    def _filter_parameters(self, named_parameters):
        """
        Filter model parameters based on include/exclude patterns and max count.
        By default, returns all parameters unless patterns are specified.
        """
        filtered_params = []
        
        # Check if filtering is actually needed
        if self.include_patterns is None and self.exclude_patterns is None and self.max_layers_to_track is None:
            # Fast path - just return all parameters with gradients
            return [(name, param) for name, param in named_parameters if param.grad is not None]
        
        # Compile regex patterns for faster matching
        include_re = [re.compile(pattern) for pattern in self.include_patterns] if self.include_patterns else None
        exclude_re = [re.compile(pattern) for pattern in self.exclude_patterns] if self.exclude_patterns else None
        
        for name, param in named_parameters:
            # Skip parameters without gradients
            if param.grad is None:
                continue
                
            # Apply include patterns if specified
            if include_re:
                if not any(pattern.search(name) for pattern in include_re):
                    continue
            
            # Apply exclude patterns if specified
            if exclude_re:
                if any(pattern.search(name) for pattern in exclude_re):
                    continue
            
            filtered_params.append((name, param))
        
        # Limit to max_layers_to_track if specified
        if self.max_layers_to_track is not None and len(filtered_params) > self.max_layers_to_track:
            filtered_params = filtered_params[:self.max_layers_to_track]
            
        return filtered_params

    def log_figure_with_fallback(self, trainer: pl.Trainer, fig: plt.Figure, name: str) -> None:
        """Log figure with fallback for loggers that don't support figure logging."""
        try:
            if hasattr(trainer.logger, 'experiment'):
                logger_type = type(trainer.logger.experiment).__name__
                if logger_type == 'SummaryWriter':  # TensorBoard
                    trainer.logger.experiment.add_figure(f'{name}', fig, global_step=trainer.global_step)
                elif logger_type == 'Run':  # WandB
                    import wandb
                    trainer.logger.experiment.log({
                        name: wandb.Image(fig)
                    }, step=trainer.global_step)
                else:  # Other loggers like MLFlow
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png')
                    buf.seek(0)
                    try:
                        trainer.logger.log_image(name, [buf])
                    except (AttributeError, TypeError):
                        logger.warning(f"Couldn't log figure with logger type: {logger_type}")
        finally:
            # Always close the figure to prevent memory leaks
            plt.close(fig)

    def _log_figure(self, trainer, fig, name):
        """Log figure to logger with proper handling for different logger types."""
        if not hasattr(trainer, 'logger') or trainer.logger is None:
            return
            
        logger_type = type(trainer.logger).__name__
        
        # Handle different logger types
        try:
            # For TensorBoard or other loggers with add_figure
            if hasattr(trainer.logger, 'experiment') and hasattr(trainer.logger.experiment, 'add_figure'):
                trainer.logger.experiment.add_figure(
                    name, fig, global_step=trainer.global_step
                )
            # For other loggers (e.g., WandbLogger), convert to image
            elif hasattr(trainer.logger, 'experiment') and hasattr(trainer.logger.experiment, 'log'):
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                
                # Different logger APIs
                if logger_type == 'WandbLogger':
                    import wandb
                    trainer.logger.experiment.log({
                        name: wandb.Image(buf)
                    }, step=trainer.global_step)
                else:
                    # Generic fallback
                    try:
                        trainer.logger.log_image(name, [buf])
                    except (AttributeError, TypeError):
                        logger.warning(f"Couldn't log figure with logger type: {logger_type}")
        except Exception as e:
            logger.warning(f"Error logging figure: {e}")
    
    def _manage_history_size(self):
        """Limit history size to prevent memory issues during long training runs."""
        if self.max_history_length <= 0:
            return
            
        for key in self.gradient_history:
            if len(self.gradient_history[key]) > self.max_history_length:
                # Keep the most recent entries
                history_length = len(self.gradient_history[key])
                # Remove oldest entries but keep some from the beginning for comparison
                keep_from_start = min(10, self.max_history_length // 10)
                keep_from_end = self.max_history_length - keep_from_start
                self.gradient_history[key] = (
                    self.gradient_history[key][:keep_from_start] + 
                    self.gradient_history[key][history_length-keep_from_end:]
                )
    
    def on_after_backward(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Compute and log gradient norms after backward pass (pre-clipping)."""
        if not hasattr(trainer, "global_step"):
            return
            
        global_step = trainer.global_step
        
        # Only log at specified frequency
        if global_step % self.log_freq != 0:
            return
        
        # Track norm values for history and visualization
        current_norms = {}
        
        # Compute and log gradient norms using Lightning's built-in utility
        for p in self.track_norm_types:
            norm_value = grad_norm(pl_module, p)
            extracted_norm = self._get_norm_value(norm_value)
            current_norms[f"norm_{p}"] = extracted_norm
            
            # Log the norm value
            if hasattr(trainer, "logger") and trainer.logger is not None:
                trainer.logger.log_metrics(
                    {f"gradients/global/norm_{p}": extracted_norm}, 
                    step=global_step
                )
        
        # Track per-layer gradient statistics if enabled
        if self.track_per_layer and hasattr(trainer, "logger") and trainer.logger is not None:
            named_parameters = self._filter_parameters(pl_module.named_parameters())
            layer_stats = {}
            layers_with_grad = 0
            
            for name, param in named_parameters:
                if param.grad is not None:
                    layers_with_grad += 1
                    # Track norms per layer
                    for p in self.track_norm_types:
                        if p in [1, 2]:  # Only L1 and L2 norms for per-layer tracking
                            layer_norm = torch.norm(param.grad.data, p=p).item()
                            layer_stats[f"gradients/layer/{name}/norm_{p}"] = layer_norm
                    
                    # Track additional statistics if enabled
                    if self.track_stats:
                        grad_data = param.grad.data
                        layer_stats[f"gradients/layer/{name}/grad_mean"] = grad_data.mean().item()
                        layer_stats[f"gradients/layer/{name}/grad_std"] = grad_data.std().item()
                        layer_stats[f"gradients/layer/{name}/grad_max"] = grad_data.max().item()
                        layer_stats[f"gradients/layer/{name}/grad_min"] = grad_data.min().item()
                        # Track percentage of zero gradients
                        zeros = (grad_data == 0).float().mean().item() * 100
                        layer_stats[f"gradients/layer/{name}/zeros_pct"] = zeros
            
            # Log all layer statistics at once
            trainer.logger.log_metrics(layer_stats, step=global_step)
            
            # Log diagnostic information if verbose
            if self.verbose and global_step == 0:
                logger.info(f"Found {layers_with_grad} layers with gradients out of {len(named_parameters)} tracked parameters")
            
            # Store per-layer history for visualization if needed
            if self.visualize_on_end:
                self.gradient_history["layers"].append(layer_stats)
        
        # Store global history for visualization
        if self.visualize_on_end:
            self.gradient_history["global"].append(current_norms)
            # Manage history size to prevent memory issues
            self._manage_history_size()
        
        # Log a warning if any norm is extremely large (potential exploding gradient)
        for p in self.track_norm_types:
            norm_key = f"norm_{p}"
            if norm_key in current_norms and current_norms[norm_key] > 100.0:
                logger.warning(
                    f"Potentially exploding gradient detected (step {global_step}): "
                    f"L{p} norm = {current_norms[norm_key]:.4f}"
                )
    
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Generate simple visualization at end of training."""
        if not self.visualize_on_end or not self.gradient_history["global"]:
            return
                
        # Skip visualization if there's no logger
        if not hasattr(trainer, 'logger') or trainer.logger is None:
            return
        
        # Plot pre and post-clipping gradient norms over time
        fig, ax = plt.subplots(figsize=(10, 6))
        steps = list(range(len(self.gradient_history["global"])))
        
        for norm_type in self.track_norm_types:
            norm_key = f"norm_{norm_type}"
            
            # Pre-clipping values
            pre_clip_values = [entry.get(norm_key, 0) for entry in self.gradient_history["global"]]
            if pre_clip_values and len(pre_clip_values) == len(steps):
                ax.plot(steps, pre_clip_values, label=f"L{norm_type} Norm", linestyle='-')
        
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Gradient Norm")
        ax.set_title("Gradient Norms During Training")
        ax.legend()
        
        # Log the figure using our robust method
        self.log_figure_with_fallback(trainer=trainer, fig=fig, name='gradients/gradient_norms_summary')
        
        # Rest of visualization code for layer heatmaps, etc.
        if self.track_per_layer and "layers" in self.gradient_history and self.gradient_history["layers"]:
            # Extract layer names and create heatmap for L2 norms
            layer_data = self.gradient_history["layers"]
            layer_names = sorted(set([
                key.split('/')[2] for key in layer_data[0].keys() 
                if key.startswith('gradients/layer/') and key.endswith('/norm_2')
            ]))
            
            if layer_names:
                # Create a matrix of layer norms over time
                norm_matrix = np.zeros((len(layer_names), len(layer_data)))
                for i, name in enumerate(layer_names):
                    for j, step_data in enumerate(layer_data):
                        key = f"gradients/layer/{name}/norm_2"
                        if key in step_data:
                            norm_matrix[i, j] = step_data[key]
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(12, max(8, len(layer_names) // 4)))
                im = ax.imshow(norm_matrix, aspect='auto', cmap='viridis')
                ax.set_yticks(range(len(layer_names)))
                ax.set_yticklabels([name[:30] + '...' if len(name) > 30 else name for name in layer_names])
                ax.set_xlabel('Training Steps')
                ax.set_title('Layer Gradient L2 Norms Over Time')
                fig.colorbar(im, ax=ax, label='Gradient L2 Norm')
                
                # Log the heatmap using our robust method
                self.log_figure_with_fallback(trainer=trainer, fig=fig, name='gradients/layer_gradient_heatmap')