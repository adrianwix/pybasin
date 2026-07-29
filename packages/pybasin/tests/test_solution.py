import numpy as np
import pytest
import torch

from pybasin.solution import Solution


@pytest.fixture
def sample_solution() -> Solution:
    n_steps, n_batch, n_states = 100, 5, 2
    ic = torch.randn(n_batch, n_states)
    time = torch.linspace(0, 10, n_steps)
    y = torch.randn(n_batch, n_states, n_steps)
    return Solution(initial_condition=ic, time=time, y=y)


def test_solution_initialization(sample_solution: Solution) -> None:
    # 100 time steps
    assert sample_solution.time.shape == (100,)
    # 5 batches × 2 states × 100 steps
    assert sample_solution.y.shape == (5, 2, 100)
    # 5 batches × 2 states
    assert sample_solution.initial_condition.shape == (5, 2)


def test_solution_set_labels(sample_solution: Solution) -> None:
    labels = np.array([0, 1, 0, 1, 0])
    sample_solution.set_labels(labels)

    # Labels were set (not None)
    assert sample_solution.labels is not None
    # One label per batch (5 batches)
    assert len(sample_solution.labels) == 5


def test_solution_set_features(sample_solution: Solution) -> None:
    features = torch.randn(5, 3)
    sample_solution.set_features(features)

    # Features were set
    assert sample_solution.features is not None
    # 5 batches × 3 features
    assert sample_solution.features.shape == (5, 3)


def test_solution_shape_validation() -> None:
    # Solution rejects mismatched shapes (time has 100 steps but y has only 50)
    with pytest.raises(AssertionError):
        Solution(
            initial_condition=torch.randn(5, 2),
            time=torch.linspace(0, 10, 100),
            y=torch.randn(5, 2, 50),
        )


def test_solution_requires_contiguous_time_axis() -> None:
    # y's last (time) axis must be contiguous (stride(-1) == 1)
    y = torch.randn(100, 5, 2).permute(1, 2, 0)
    assert y.shape == (5, 2, 100)
    assert y.stride(-1) != 1
    with pytest.raises(AssertionError):
        Solution(
            initial_condition=torch.randn(5, 2),
            time=torch.linspace(0, 10, 100),
            y=y,
        )
