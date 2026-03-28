"""Tests for deck generation and validation."""

import tempfile
from pathlib import Path

import pytest
import yaml

from ygo_meta.deck_builder.generator import generate_decks
from ygo_meta.deck_builder.validator import validate_deck


STAPLES_YAML = {
    "handtraps": [
        {"code": 14558127, "name": "Ash Blossom", "copies_max": 3},
        {"code": 23434538, "name": "Effect Veiler", "copies_max": 3},
        {"code": 98095162, "name": "Ghost Ogre", "copies_max": 3},
        {"code": 40508768, "name": "Ghost Belle", "copies_max": 3},
        {"code": 27204311, "name": "Nibiru", "copies_max": 3},
        {"code": 55144522, "name": "Called by the Grave", "copies_max": 3},
        {"code": 15693423, "name": "Dark Ruler No More", "copies_max": 3},
    ]
}

# Engine with 30 distinct main deck cards (3 copies each of 10 unique codes)
# This leaves only 10 staple slots to fill, which the 7-entry pool can reliably cover.
_ENGINE_CARDS = (
    [10000001] * 3 + [10000002] * 3 + [10000003] * 3 + [10000004] * 3
    + [10000005] * 3 + [10000006] * 3 + [10000007] * 3 + [10000008] * 3
    + [10000009] * 3 + [10000010] * 3
)  # 30 cards
ENGINE_YDK = "\n".join(
    ["#created by test", "#main"]
    + [str(c) for c in _ENGINE_CARDS]
    + ["#extra", "!side", ""]
)


def make_fixtures(tmp_path: Path) -> tuple[Path, Path]:
    engine_dir = tmp_path / "engines" / "test_arch"
    engine_dir.mkdir(parents=True)
    ydk = engine_dir / "engine.ydk"
    ydk.write_text(ENGINE_YDK, encoding="utf-8")

    staples_dir = tmp_path / "staples"
    staples_dir.mkdir()
    (staples_dir / "handtraps.yaml").write_text(
        yaml.dump(STAPLES_YAML), encoding="utf-8"
    )
    return ydk, staples_dir


def test_generate_decks_size(tmp_path: Path) -> None:
    ydk, staples_dir = make_fixtures(tmp_path)
    decks = generate_decks(ydk, staples_dir, n_variants=4, seed=0)
    assert len(decks) == 4
    for deck in decks:
        assert 40 <= len(deck.main) <= 60, f"{deck.variant_id}: main={len(deck.main)}"
        assert len(deck.extra) <= 15
        assert len(deck.side) <= 15


def test_generate_decks_valid(tmp_path: Path) -> None:
    ydk, staples_dir = make_fixtures(tmp_path)
    decks = generate_decks(ydk, staples_dir, n_variants=4, seed=42)
    for deck in decks:
        errors = validate_deck(deck)
        assert errors == [], f"{deck.variant_id}: {errors}"


def test_validator_rejects_small_deck() -> None:
    from ygo_meta.deck_builder.deck_model import Deck
    deck = Deck("test", "test_v000", main=[1] * 30, extra=[], side=[])
    errors = validate_deck(deck)
    assert any("small" in e for e in errors)


def test_validator_rejects_large_extra() -> None:
    from ygo_meta.deck_builder.deck_model import Deck
    deck = Deck("test", "test_v000", main=[1] * 40, extra=[2] * 16, side=[])
    errors = validate_deck(deck)
    assert any("extra" in e.lower() for e in errors)
