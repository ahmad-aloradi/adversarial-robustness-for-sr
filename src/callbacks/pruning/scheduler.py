import math
from typing import Dict, Any, Optional
from src.utils import get_pylogger

logger = get_pylogger(__name__)


class PruningScheduler:
    """
    Manages the pruning schedule by pre-computing targets for each epoch.
    Supports state persistence to ensure consistency upon resumption.
    """
    def __init__(
        self,
        schedule_type: str,
        final_sparsity: float,
        epochs_to_ramp: int,
        initial_sparsity: float = 0.0,
        verbose: bool = True
    ):
        self.schedule_type = schedule_type
        self.final_sparsity = final_sparsity
        self.initial_sparsity = initial_sparsity
        self.epochs_to_ramp = max(1, epochs_to_ramp)
        self.verbose = verbose
        
        # Pre-compute the schedule map: {epoch_idx: target_sparsity}
        self.schedule_map: Dict[int, float] = {}
        self._generate_schedule()

    def _generate_schedule(self):
        """Generates the epoch-to-sparsity mapping based on configuration."""
        self.schedule_map.clear()
        
        for epoch in range(self.epochs_to_ramp):
            # Progress 0..1 logic: (epoch + 1) / N ensures we hit the final target exactly at the end
            progress = min(1.0, (epoch + 1) / self.epochs_to_ramp)
            
            target = 0.0
            if self.schedule_type == "linear":
                target = self.initial_sparsity + (self.final_sparsity - self.initial_sparsity) * progress
            
            elif self.schedule_type == "constant":
                # Constant Pruning Rate: S_t = 1 - (1 - S_final)^(t / N)
                remaining_final = 1.0 - self.final_sparsity
                remaining_initial = 1.0 - self.initial_sparsity
                
                if remaining_final <= 1e-9:
                    remaining_final = 1e-9
                if remaining_initial <= 1e-9:
                    target = 1.0
                else:
                    # Log-space interpolation
                    current_remaining = remaining_initial * (remaining_final / remaining_initial) ** progress
                    target = 1.0 - current_remaining
            else:
                raise ValueError(f"Unknown schedule_type: {self.schedule_type}")
            
            # Clamp and store
            self.schedule_map[epoch] = min(max(target, 0.0), self.final_sparsity)

    def get_target_sparsity(self, current_epoch: int) -> float:
        """
        Returns the target sparsity for the given epoch.
        If the schedule is complete (current_epoch >= ramp), returns the FINAL sparsity.
        This ensures the model is 'held' at the target sparsity indefinitely.
        """
        if current_epoch >= self.epochs_to_ramp:
            return self.final_sparsity
            
        return self.schedule_map.get(current_epoch, self.final_sparsity)

    def verify_schedule_feasibility(self, max_epochs: int) -> None:
        if max_epochs < self.epochs_to_ramp:
            raise ValueError(
                f"Pruning Schedule Error: epochs_to_ramp ({self.epochs_to_ramp}) > "
                f"trainer.max_epochs ({max_epochs}). The schedule cannot complete."
            )

    def state_dict(self) -> Dict[str, Any]:
        """Returns the scheduler state for checkpointing."""
        return {
            "schedule_type": self.schedule_type,
            "final_sparsity": self.final_sparsity,
            "epochs_to_ramp": self.epochs_to_ramp,
            "initial_sparsity": self.initial_sparsity,
            "schedule_map": self.schedule_map
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Restores scheduler state from checkpoint."""
        self.schedule_type = state["schedule_type"]
        self.final_sparsity = state["final_sparsity"]
        self.epochs_to_ramp = state["epochs_to_ramp"]
        self.initial_sparsity = state.get("initial_sparsity", 0.0)
        # Restore epochs to pruning map
        self.schedule_map = {int(k): float(v) for k, v in state["schedule_map"].items()}
        
        if self.verbose:
            logger.info(f"Restored Pruning Schedule from checkpoint: {len(self.schedule_map)} steps.")
