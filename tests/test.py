import numpy as np
import pytest

### Test for monotonic derivative smoothing
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


### Test for curve smoothing with genetic algo
from monotonic_derivative.curve_smoothing import (
    initialize_population,
    selection,
    crossover,
    mutation,
    calculate_fitness,
)


def test_initialize_population():
    population = initialize_population(5, 6)
    assert population.shape == (5, 6)


def test_fitness():
    points = np.array([10, 55, 53, 40, 35, 5])
    individual = np.array([10, 50, 45, 40, 30, 5])
    score = calculate_fitness(individual, points, alpha=0.5)
    lower_bound = -50  # Set the lower bound to an acceptable minimum value
    upper_bound = 50  # Set the upper bound to an acceptable maximum value
    assert lower_bound <= score <= upper_bound


def test_selection():
    points = np.array([10, 55, 53, 40, 35, 5])
    population = initialize_population(10, len(points))
    parents = selection(population, points, alpha=0.5)
    assert len(parents) == 2


def test_crossover():
    parent1 = np.array([10, 50, 45, 40, 30, 5])
    parent2 = np.array([20, 55, 40, 35, 25, 10])
    child1, child2 = crossover(parent1, parent2)
    assert len(child1) == len(parent1)
    assert len(child2) == len(parent2)


def test_mutation():
    individual = np.array([10, 50, 45, 40, 30, 5])
    mutated_individual = mutation(individual, mutation_rate=0.1)
    assert len(mutated_individual) == len(individual)
