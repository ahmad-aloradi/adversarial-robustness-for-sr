"""Unit tests for the shared trainer probes."""

from unittest.mock import Mock

import pytest

from src.utils.trainer_utils import total_training_steps


def test_returns_the_step_budget():
    """A finite budget comes back as an int."""
    assert (
        total_training_steps(Mock(estimated_stepping_batches=5000), "x")
        == 5000
    )


def test_rejects_infinite_budget():
    """max_epochs=-1 with no max_steps gives inf, which cannot size a
    schedule."""
    with pytest.raises(RuntimeError, match="finite"):
        total_training_steps(
            Mock(estimated_stepping_batches=float("inf")), "x"
        )


def test_rejects_max_steps_sentinel():
    """Lightning returns max_steps (-1 by default) for iterable datasets."""
    with pytest.raises(RuntimeError, match="-1"):
        total_training_steps(Mock(estimated_stepping_batches=-1), "x")


def test_error_names_the_caller():
    """The message says which feature needed the budget."""
    with pytest.raises(RuntimeError, match="The lambda trust region"):
        total_training_steps(
            Mock(estimated_stepping_batches=float("inf")),
            "The lambda trust region",
        )
