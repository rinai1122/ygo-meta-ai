"""Tests for Nash equilibrium solver."""

import numpy as np
import pytest

from ygo_meta.simulation.nash import solve_nash


def test_rps_uniform():
    """Rock-Paper-Scissors: Nash equilibrium is uniform [1/3, 1/3, 1/3]."""
    # W[i][j] = prob that strategy i beats strategy j in RPS
    # Rock=0, Paper=1, Scissors=2
    A = np.array([
        [0.5, 0.0, 1.0],   # Rock vs Rock, Paper, Scissors
        [1.0, 0.5, 0.0],   # Paper vs Rock, Paper, Scissors
        [0.0, 1.0, 0.5],   # Scissors vs Rock, Paper, Scissors
    ], dtype=float)

    sol = solve_nash(A)
    assert sol.sigma.shape == (3,)
    assert abs(sol.sigma.sum() - 1.0) < 1e-6
    for p in sol.sigma:
        assert abs(p - 1 / 3) < 0.05, f"Expected ~1/3, got {p}"


def test_dominant_strategy():
    """If deck 0 dominates everything, Nash weight should be 1.0 on deck 0."""
    # Deck 0 always wins against deck 1 and 2
    A = np.array([
        [0.5, 0.9, 0.9],
        [0.1, 0.5, 0.5],
        [0.1, 0.5, 0.5],
    ], dtype=float)
    sol = solve_nash(A)
    assert sol.sigma[0] > 0.8, f"Expected deck 0 dominant, got sigma={sol.sigma}"


def test_probabilities_sum_to_one():
    rng = np.random.default_rng(0)
    A = rng.random((5, 5))
    # Make antisymmetric (zero-sum)
    A = (A + (1 - A.T)) / 2
    sol = solve_nash(A)
    assert abs(sol.sigma.sum() - 1.0) < 1e-6
    assert all(p >= -1e-9 for p in sol.sigma)


def test_nan_raises():
    A = np.array([[0.5, np.nan], [0.5, 0.5]])
    with pytest.raises(ValueError, match="NaN"):
        solve_nash(A)
