import numpy as np
import pytest
from monotonic_derivative import ensure_monotonic_derivative


def test_basic_usage():
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([100, 55, 53, 40, 35, 5])
    d = 2
    modified_y = ensure_monotonic_derivative(
        x, y, degree=d, force_negative_derivative=False, verbose=False, save_plot=False
    )
    assert len(modified_y) == len(y)
    d2nd = np.diff(modified_y, n=d) / np.prod([np.diff(x) for _ in range(d)])
    assert all(d2nd[i] <= d2nd[i + 1] for i in range(len(d2nd) - 1))


def test_invalid_degree():
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([100, 55, 53, 40, 35, 5])

    with pytest.raises(ValueError):
        ensure_monotonic_derivative(x, y, degree=5)


def test_negative_derivative():
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([100, 55, 53, 40, 35, 5])
    d = 2
    modified_y = ensure_monotonic_derivative(
        x, y, degree=d, force_negative_derivative=True, verbose=False, save_plot=False
    )
    assert len(modified_y) == len(y)
    d2nd = np.diff(modified_y, n=d) / np.prod([np.diff(x) for _ in range(d)])
    assert all(d2nd[i] >= d2nd[i + 1] for i in range(len(d2nd) - 1))


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
