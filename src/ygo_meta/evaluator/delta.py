"""
Tech-card delta evaluator.

Given a baseline deck, an opponent deck, and a list of candidate tech cards,
this module produces and resolves judgment queries that estimate the *delta*
in win-rate from swapping each tech card into the baseline's flex slot.

Statistical model
-----------------
For each tech card C with true win-rate ``p_C`` against the opponent and
baseline true win-rate ``p_B``, the estimator is::

    Δ̂ = p̂_C - p̂_B
    Var(Δ̂) = p̂_C(1-p̂_C)/n_tech + p̂_B(1-p̂_B)/n_baseline

Because the baseline is amortised across K tech cards, the optimal allocation
satisfies ``n_baseline / n_tech ≈ √K``. Defaults: ``n_baseline=36, n_tech=12``
(roughly 3:1, well-suited for K≈10) — see CLAUDE.md.

Hand sampling
-------------
Every query gets a fresh hand (re-rolled flex slots). For tech-card queries the
candidate card is pinned into A's hand via ``force_a_in_hand=[code]`` so the
human is judging the *card in play*, not the chance of drawing it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

from ygo_meta.deck_builder.deck_model import Deck
from ygo_meta.evaluator.judgment_store import (
    Judgment,
    JudgmentStore,
    canonical_deck_hash,
)
from ygo_meta.evaluator.sampler import sample_queries_for_pair


# Default per-card budget. ~3:1 ratio is optimal for K≈10 candidates.
DEFAULT_N_BASELINE = 36
DEFAULT_N_TECH = 12


@dataclass
class TechVariant:
    """A baseline deck with one tech card swapped into a flex slot."""
    name: str            # human-readable tech name (e.g., "Maxx C")
    code: int            # YGOPro code of the tech card
    deck: Deck           # full deck variant containing the tech card


@dataclass
class DeltaResult:
    tech_name: str
    tech_code: int
    baseline_winrate: float
    tech_winrate: float
    delta: float
    n_baseline: int
    n_tech: int
    se_delta: float

    def to_dict(self) -> dict:
        return {
            "tech_name": self.tech_name,
            "tech_code": self.tech_code,
            "baseline_winrate": round(self.baseline_winrate, 4),
            "tech_winrate": round(self.tech_winrate, 4),
            "delta": round(self.delta, 4),
            "n_baseline": self.n_baseline,
            "n_tech": self.n_tech,
            "se_delta": round(self.se_delta, 4),
        }


def enqueue_delta_queries(
    store: JudgmentStore,
    baseline: Deck,
    opponent: Deck,
    tech_variants: list[TechVariant],
    n_baseline: int = DEFAULT_N_BASELINE,
    n_tech: int = DEFAULT_N_TECH,
    seed: int = 0,
) -> int:
    """Enqueue (n_baseline + n_tech * len(tech_variants)) pending queries.

    Each query is sampled with a fresh hand. Tech-card queries pin the
    candidate card into A's hand. Returns the number of NEW queries added
    (already-cached judgments are skipped automatically by the store).
    """
    added = 0

    # Baseline queries: balanced who_first, no forced cards.
    base_queries = sample_queries_for_pair(
        deck_a=baseline,
        deck_b=opponent,
        num_queries=n_baseline,
        seed=seed,
    )
    for q in base_queries:
        before = store.pending_count()
        store.append_pending(q)
        if store.pending_count() > before:
            added += 1

    # Tech-card queries.
    for k, tv in enumerate(tech_variants):
        tq = sample_queries_for_pair(
            deck_a=tv.deck,
            deck_b=opponent,
            num_queries=n_tech,
            seed=seed + 1000 * (k + 1),
            force_a_in_hand=[tv.code],
        )
        for q in tq:
            before = store.pending_count()
            store.append_pending(q)
            if store.pending_count() > before:
                added += 1

    return added


def compute_deltas(
    store: JudgmentStore,
    baseline: Deck,
    opponent: Deck,
    tech_variants: list[TechVariant],
) -> tuple[float, int, list[DeltaResult]]:
    """Compute baseline win-rate and per-tech delta from the judgment store.

    Returns ``(baseline_winrate, n_baseline_observed, [DeltaResult, ...])``.
    Tech variants without any judgments yet are skipped.
    """
    base_hash = canonical_deck_hash(baseline.main)
    opp_hash = canonical_deck_hash(opponent.main)

    base_judgments = store.judgments_for_pair(base_hash, opp_hash)
    n_b = len(base_judgments)
    p_b = (sum(j.bucket for j in base_judgments) / n_b) if n_b else 0.0
    var_b = (p_b * (1.0 - p_b) / n_b) if n_b else 0.0

    out: list[DeltaResult] = []
    for tv in tech_variants:
        tv_hash = canonical_deck_hash(tv.deck.main)
        tv_judgments = store.judgments_for_pair(tv_hash, opp_hash)
        n_t = len(tv_judgments)
        if n_t == 0:
            continue
        p_t = sum(j.bucket for j in tv_judgments) / n_t
        var_t = p_t * (1.0 - p_t) / n_t
        se = math.sqrt(var_t + var_b)
        out.append(DeltaResult(
            tech_name=tv.name,
            tech_code=tv.code,
            baseline_winrate=p_b,
            tech_winrate=p_t,
            delta=p_t - p_b,
            n_baseline=n_b,
            n_tech=n_t,
            se_delta=se,
        ))
    out.sort(key=lambda r: r.delta, reverse=True)
    return p_b, n_b, out


def wait_for_completion(
    store: JudgmentStore,
    baseline: Deck,
    opponent: Deck,
    tech_variants: list[TechVariant],
    n_baseline: int,
    n_tech: int,
    poll_interval: float = 2.0,
    timeout: float | None = None,
    on_progress=None,
) -> None:
    """Block until the store has at least n_baseline judgments for the
    baseline pair AND at least n_tech for every tech variant pair.
    """
    base_hash = canonical_deck_hash(baseline.main)
    opp_hash = canonical_deck_hash(opponent.main)
    deadline = None if timeout is None else time.time() + timeout

    while True:
        n_b = len(store.judgments_for_pair(base_hash, opp_hash))
        per_tech = [
            len(store.judgments_for_pair(canonical_deck_hash(tv.deck.main), opp_hash))
            for tv in tech_variants
        ]
        if on_progress is not None:
            on_progress(n_b, per_tech)
        if n_b >= n_baseline and all(n >= n_tech for n in per_tech):
            return
        if deadline is not None and time.time() > deadline:
            raise TimeoutError(
                f"Tech-delta evaluation timed out (baseline {n_b}/{n_baseline}, "
                f"tech {per_tech})"
            )
        time.sleep(poll_interval)
