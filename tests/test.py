import numpy as np
import pytest
from monotonic_derivative import ensure_monotonic_derivative


def test_basic_usage():
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([100, 55, 53, 40, 35, 5])

    modified_y = ensure_monotonic_derivative(x, y)
    assert len(modified_y) == len(y)
    assert np.all(modified_y <= y)


def test_invalid_degree():
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([100, 55, 53, 40, 35, 5])

    with pytest.raises(ValueError):
        ensure_monotonic_derivative(x, y, degree=5)


def test_negative_derivative():
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([100, 55, 53, 40, 35, 5])

    modified_y = ensure_monotonic_derivative(x, y, force_negative_derivative=True)
    assert len(modified_y) == len(y)
    assert np.all(modified_y >= y)


def test_mismatched_lengths():
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([100, 55, 53, 40, 35])

    with pytest.raises(ValueError):
        ensure_monotonic_derivative(x, y)


def test_same_lengths():
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([100, 55, 53, 40, 35, 5])

    with pytest.raises(ValueError):
        ensure_monotonic_derivative(x, y, degree=6)
