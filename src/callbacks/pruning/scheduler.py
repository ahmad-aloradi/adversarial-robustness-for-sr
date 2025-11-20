from typing import Optional

class PruningScheduler:
    """
    Handles the calculation of target sparsity over time for scheduled pruning.
    """
    def __init__(self, schedule_type: str, final_amount: float, epochs_to_ramp: int):
        self.schedule_type = schedule_type
        self.final_amount = final_amount
        self.epochs_to_ramp = max(1, epochs_to_ramp)
        self._constant_fraction: Optional[float] = None
        
        if schedule_type == "constant":
            self._constant_fraction = self._calculate_constant_fraction()

    def _get_schedule_span(self) -> int:
        """Return the integer epoch span used for scheduled pruning targets."""
        return max(1, self.epochs_to_ramp - 1)

    def _calculate_constant_fraction(self) -> float:
        """Compute constant pruning ratio required to hit final sparsity."""
        if self.final_amount is None:
            raise ValueError("constant schedule requires final_amount to be set")

        schedule_span = self._get_schedule_span()
        target_dense = max(0.0, 1.0 - self.final_amount)
        if target_dense == 0.0:
            fraction = 1.0
        else:
            fraction = 1.0 - target_dense ** (1.0 / float(schedule_span))

        return min(max(fraction, 0.0), 1.0)

    def get_target_sparsity(self, current_epoch: int) -> float:
        """Calculate target sparsity for current epoch."""
        if current_epoch < 0:
            return 0.0

        step_index = max(0, min(int(current_epoch), self._get_schedule_span()))

        if self.schedule_type == "linear":
            progress = step_index / float(self._get_schedule_span())
            return progress * self.final_amount
        
        if self.schedule_type == "constant":
            if step_index == 0:
                return 0.0
            # Use cached fraction
            fraction = self._constant_fraction
            return 1.0 - (1.0 - fraction) ** step_index

        raise ValueError(f"Unsupported schedule_type {self.schedule_type}")
    
    @property
    def constant_fraction(self) -> Optional[float]:
        return self._constant_fraction

    def is_target_reachable(self, total_epochs: int, tolerance: float = 1e-4) -> bool:
        """Check if the schedule reaches the final target within the given epochs."""
        if total_epochs < 1:
            return False
        
        final_sparsity = self.get_target_sparsity(total_epochs - 1)
        return final_sparsity + tolerance >= self.final_amount
