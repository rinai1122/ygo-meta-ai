"""
Nash equilibrium solver for the meta payoff matrix.

Primary: nashpy support_enumeration (exact, works well for D <= ~30).
Fallback: scipy linear programming for larger matrices or degenerate cases.

Usage:
    solution = solve_nash(payoff_matrix.matrix)
    print(solution.sigma)   # mixed strategy distribution over decks
    print(solution.top_decks(deck_ids, top_n=3))
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NashSolution:
    sigma: np.ndarray   # mixed strategy (probability per deck), sums to 1.0
    method: str         # "nashpy" or "lp"

    def support(self, threshold: float = 1e-4) -> list[int]:
        """Indices of decks with non-negligible Nash weight."""
        return [i for i, p in enumerate(self.sigma) if p > threshold]

    def top_decks(self, deck_ids: list[str], top_n: int = 3) -> list[dict]:
        ranked = sorted(enumerate(self.sigma), key=lambda x: -x[1])
        return [
            {"deck_id": deck_ids[i], "weight": float(w)}
            for i, w in ranked[:top_n]
            if w > 1e-6
        ]


def _solve_lp(A: np.ndarray) -> np.ndarray:
    """Solve the zero-sum game via linear programming (scipy)."""
    from scipy.optimize import linprog

    n = A.shape[0]
    # Row player: maximize v subject to A^T x >= v*1, sum(x)=1, x>=0
    # Equivalent to minimizing -v
    # Variables: [x_0, ..., x_{n-1}, v]
    c = np.zeros(n + 1)
    c[-1] = -1.0  # minimize -v

    # A^T x - v*1 >= 0  →  -A^T x + v*1 <= 0
    A_ub = np.hstack([-A.T, np.ones((n, 1))])
    b_ub = np.zeros(n)

    # sum(x) = 1
    A_eq = np.ones((1, n + 1))
    A_eq[0, -1] = 0.0
    b_eq = np.array([1.0])

    bounds = [(0, None)] * n + [(None, None)]
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    if result.status != 0:
        # Fallback: uniform
        return np.ones(n) / n
    x = result.x[:n]
    x = np.clip(x, 0, None)
    total = x.sum()
    return x / total if total > 1e-10 else np.ones(n) / n


def solve_nash(matrix: np.ndarray) -> NashSolution:
    """
    Solve for the Nash equilibrium mixed strategy of the row player.

    Args:
        matrix: D×D payoff matrix where matrix[i][j] = P(deck i beats deck j).
                Must be complete (no NaN).

    Returns:
        NashSolution with sigma (probability distribution over decks).
    """
    if np.any(np.isnan(matrix)):
        raise ValueError("Payoff matrix contains NaN — run all matchups first.")

    n = matrix.shape[0]

    # Small matrices: try nashpy first
    if n <= 30:
        try:
            import nashpy as nash

            A = matrix
            B = 1.0 - A
            game = nash.Game(A, B)
            equilibria = list(game.support_enumeration())
            if equilibria:
                sigma1, _ = equilibria[0]
                sigma1 = np.array(sigma1, dtype=float)
                sigma1 = np.clip(sigma1, 0, None)
                total = sigma1.sum()
                if total > 1e-10:
                    solution = NashSolution(sigma=sigma1 / total, method="nashpy")
                    assert abs(solution.sigma.sum() - 1.0) < 1e-6, (
                        f"Nash sigma sum = {solution.sigma.sum()}, expected 1.0"
                    )
                    return solution
        except Exception:
            pass

    # Fallback: LP (nashpy support_enumeration is exponential for large D)
    sigma = _solve_lp(matrix)
    solution = NashSolution(sigma=sigma, method="lp")
    assert abs(solution.sigma.sum() - 1.0) < 1e-6, (
        f"Nash sigma sum = {solution.sigma.sum()}, expected 1.0"
    )
    return solution
