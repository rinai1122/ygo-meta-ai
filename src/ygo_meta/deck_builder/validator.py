"""
Validate a Deck against standard YGO deck size rules.

Returns a list of violation strings (empty list = valid).
"""

from __future__ import annotations

from collections import Counter

from ygo_meta.deck_builder.deck_model import Deck
from ygo_meta.deck_builder.lflist import LFList


def validate_deck(deck: Deck, lflist: LFList | None = None) -> list[str]:
    errors: list[str] = []

    n_main = len(deck.main)
    if n_main < 40:
        errors.append(f"Main deck too small: {n_main} cards (minimum 40)")
    elif n_main > 60:
        errors.append(f"Main deck too large: {n_main} cards (maximum 60)")

    n_extra = len(deck.extra)
    if n_extra > 15:
        errors.append(f"Extra deck too large: {n_extra} cards (maximum 15)")

    n_side = len(deck.side)
    if n_side > 15:
        errors.append(f"Side deck too large: {n_side} cards (maximum 15)")

    combined = Counter(deck.main + deck.side)
    for code, count in combined.items():
        if count > 3:
            errors.append(f"Card {code} appears {count} times (maximum 3)")

    if lflist is not None:
        all_zones = Counter(deck.main + deck.extra + deck.side)
        for code, count in all_zones.items():
            cap = lflist.max_copies(code)
            if count > cap:
                errors.append(
                    f"Card {code} appears {count} times "
                    f"(banlist '{lflist.name}' limit: {cap})"
                )

    return errors
