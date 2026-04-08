"""Human-evaluated flex-card meta solver.

Provides a `HumanBattleRunner` that implements the standard runner interface
by delegating matchup evaluation to a human through a web UI.
"""

from ygo_meta.evaluator.delta import (
    DEFAULT_N_BASELINE,
    DEFAULT_N_TECH,
    DeltaResult,
    TechVariant,
    compute_deltas,
    enqueue_delta_queries,
    wait_for_completion,
)
from ygo_meta.evaluator.human_runner import HumanBattleRunner
from ygo_meta.evaluator.judgment_store import (
    Judgment,
    JudgmentStore,
    PendingQuery,
    canonical_deck_hash,
)
from ygo_meta.evaluator.sampler import sample_queries_for_pair

__all__ = [
    "DEFAULT_N_BASELINE",
    "DEFAULT_N_TECH",
    "DeltaResult",
    "HumanBattleRunner",
    "Judgment",
    "JudgmentStore",
    "PendingQuery",
    "TechVariant",
    "canonical_deck_hash",
    "compute_deltas",
    "enqueue_delta_queries",
    "sample_queries_for_pair",
    "wait_for_completion",
]
