"""Tests for ParameterManager to ensure dead code removal."""

from src.callbacks.pruning.parameter_manager import ParameterManager


def test_parameter_manager_no_compute_sparsity_method():
    """Verify that ParameterManager doesn't have a compute_sparsity method.

    This method was removed in favor of the shared compute_sparsity utility in
    src.callbacks.pruning.shared_prune_utils.
    """
    manager = ParameterManager(min_param_elements=1)

    # Verify the method doesn't exist
    assert not hasattr(manager, "compute_sparsity"), (
        "ParameterManager should not have a compute_sparsity method. "
        "Use compute_sparsity from src.callbacks.pruning.shared_prune_utils instead."
    )
