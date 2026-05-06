"""
Phase 3 — meta solver.

Inputs an N×N empirical payoff matrix from the gauntlet and outputs the
predicted Master Duel BO1 meta share via continuous replicator dynamics.

`simulation/nash.py` (nashpy) is kept as an alternative solver for
sanity-checking the replicator fixed point.
"""

from ygo_meta.meta_solver.replicator import (  # noqa: F401
    ReplicatorSolution,
    solve_replicator,
)
