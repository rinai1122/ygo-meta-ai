"""
Query sampler: given a deck pair, produces a balanced set of matchup queries
(5-card hands + who-first) for a human to judge.

v1 sampling policy (deliberately simple — bias can be added later):
- Sample ``num_queries`` queries per ordered deck pair.
- Alternate ``who_first`` so A-first and B-first are balanced.
- Each hand = 5 cards drawn without replacement from the deck's main list.
- Each draw is seeded by (deck_hash, seed, query_idx) for reproducibility.
"""

from __future__ import annotations

import random

from ygo_meta.deck_builder.deck_model import Deck
from ygo_meta.evaluator.judgment_store import (
    PendingQuery,
    canonical_deck_hash,
    make_pending_query,
)


def _sample_hand(
    main: list[int],
    rng: random.Random,
    hand_size: int = 5,
    force_in_hand: list[int] | None = None,
) -> list[int]:
    """Draw `hand_size` cards from `main`. If `force_in_hand` is given, those
    cards are pinned (must appear at least once each, in their requested
    multiplicity, capped at `hand_size`); the remaining slots are sampled
    without replacement from the deck minus the pinned cards.
    """
    forced = list(force_in_hand or [])[:hand_size]
    remaining_slots = hand_size - len(forced)

    # Build pool excluding forced copies (one removal per pinned card).
    pool = list(main)
    for c in forced:
        if c in pool:
            pool.remove(c)

    if remaining_slots <= 0:
        rng.shuffle(forced)
        return forced

    if len(pool) < remaining_slots:
        drawn = [rng.choice(pool) for _ in range(remaining_slots)] if pool else []
    else:
        drawn = rng.sample(pool, remaining_slots)

    hand = forced + drawn
    rng.shuffle(hand)
    return hand


def sample_queries_for_pair(
    deck_a: Deck,
    deck_b: Deck,
    num_queries: int,
    seed: int = 0,
    hand_size: int = 5,
    force_a_in_hand: list[int] | None = None,
    force_b_in_hand: list[int] | None = None,
) -> list[PendingQuery]:
    """Produce ``num_queries`` pending queries for the A-vs-B matchup."""
    if num_queries <= 0:
        return []

    out: list[PendingQuery] = []
    seen_ids: set[str] = set()
    # Oversample a bit so we can skip duplicates; bounded at 4×.
    for k in range(num_queries * 4):
        rng_a = random.Random(hash((canonical_deck_hash(deck_a.main), seed, k, "A")) & 0xFFFFFFFF)
        rng_b = random.Random(hash((canonical_deck_hash(deck_b.main), seed, k, "B")) & 0xFFFFFFFF)
        hand_a = _sample_hand(deck_a.main, rng_a, hand_size, force_a_in_hand)
        hand_b = _sample_hand(deck_b.main, rng_b, hand_size, force_b_in_hand)
        who_first = "A" if (k % 2 == 0) else "B"
        q = make_pending_query(
            deck_a_id=deck_a.variant_id,
            deck_a_archetype=deck_a.archetype,
            deck_a_main=deck_a.main,
            deck_b_id=deck_b.variant_id,
            deck_b_archetype=deck_b.archetype,
            deck_b_main=deck_b.main,
            hand_a=hand_a,
            hand_b=hand_b,
            who_first=who_first,
        )
        if q.query_id in seen_ids:
            continue
        seen_ids.add(q.query_id)
        out.append(q)
        if len(out) >= num_queries:
            break
    return out
