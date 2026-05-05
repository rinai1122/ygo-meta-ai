"""Smoke test: MagnetBlock (Master Duel-original combo) loads and validates.

This is the worked example for combining cards that never co-existed in TCG/OCG
banlist windows: Magnet Warriors + Block Dragon. The engine itself doesn't care
about TCG/OCG legality, so this should pass standard size/copy validation.
"""

from pathlib import Path

from ygo_meta.evaluator.archetype_loader import load_archetype_deck
from ygo_meta.deck_builder.validator import validate_deck


ENGINES_DIR = Path(__file__).resolve().parents[1] / "data" / "engines"


def test_magnetblock_deck_loads_and_validates() -> None:
    deck = load_archetype_deck("MagnetBlock", ENGINES_DIR)
    assert 40 <= len(deck.main) <= 60
    assert len(deck.extra) <= 15
    assert validate_deck(deck) == []


def test_magnetblock_contains_signature_cards() -> None:
    deck = load_archetype_deck("MagnetBlock", ENGINES_DIR)
    assert 94689206 in deck.main      # Block Dragon
    assert 42901635 in deck.extra     # Berserkion the Electromagna Warrior
