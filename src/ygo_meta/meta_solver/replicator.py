"""
Continuous replicator dynamics on a payoff matrix.

For deck shares x ∈ Δ^N and win-rate matrix W (W[i,j] = P(i beats j)):

    u_i = (W · x)_i               # deck i's expected WR vs current field
    ū   = xᵀ · W · x              # mean field WR
    ẋ_i = x_i (u_i − ū)

Integrate forward in time until ‖ẋ‖_∞ < epsilon.

Notes
-----
- Initial condition defaults to the uniform simplex 1/N. Callers can pass
  current ladder shares for a more realistic starting point.
- Pure replicator only prunes — it never resurrects extinct strategies.
  We don't add mutation: the gauntlet is already small (8–12 decks) and the
  user is asking "given these decks, what does the meta settle to?", not
  "discover unknown decks".
- Output probabilities sum to 1.0 ± 1e-6 (renormalised at each step to fight
  numerical drift).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReplicatorSolution:
    shares: np.ndarray         # length-N, sums to 1.0
    fitnesses: np.ndarray      # length-N, win rate vs final field
    mean_fitness: float
    iterations: int
    converged: bool


def solve_replicator(
    payoff: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    dt: float = 0.05,
    max_iters: int = 20_000,
    epsilon: float = 1e-6,
    extinction_threshold: float = 1e-5,
) -> ReplicatorSolution:
    """Run continuous replicator dynamics on `payoff`.

    Parameters
    ----------
    payoff : (N, N) array
        Empirical win-rate matrix; payoff[i, j] = P(deck i beats deck j).
        Diagonal is treated as 0.5 (ignored by the dynamics for symmetric games).
    x0 : (N,) array, optional
        Starting share. Defaults to uniform.
    dt : float
        Forward-Euler step size.
    max_iters : int
        Iteration cap before declaring non-convergence.
    epsilon : float
        Stop when ‖ẋ‖_∞ < epsilon.
    extinction_threshold : float
        Below this share, set x_i = 0 hard. Mass redistributes via renorm.
    """
    payoff = np.asarray(payoff, dtype=np.float64)
    n = payoff.shape[0]
    if payoff.shape != (n, n):
        raise ValueError(f"payoff must be square, got {payoff.shape}")

    if x0 is None:
        x = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        x = np.asarray(x0, dtype=np.float64).copy()
        if x.shape != (n,):
            raise ValueError(f"x0 has shape {x.shape}, expected ({n},)")
        s = x.sum()
        if s <= 0:
            raise ValueError("x0 must have positive sum")
        x /= s

    converged = False
    iters = 0
    for iters in range(1, max_iters + 1):
        u = payoff @ x
        u_bar = float(x @ u)
        xdot = x * (u - u_bar)
        if np.max(np.abs(xdot)) < epsilon:
            converged = True
            break
        x = x + dt * xdot
        x[x < extinction_threshold] = 0.0
        x = np.clip(x, 0.0, None)
        s = x.sum()
        if s <= 0:
            # All extinct — return uniform over original support as a fallback.
            x = np.full(n, 1.0 / n)
            break
        x /= s

    u = payoff @ x
    return ReplicatorSolution(
        shares=x,
        fitnesses=u,
        mean_fitness=float(x @ u),
        iterations=iters,
        converged=converged,
    )
