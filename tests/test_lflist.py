"""Tests for lflist parser and banlist-aware deck validator."""

from pathlib import Path

from ygo_meta.deck_builder.deck_model import Deck
from ygo_meta.deck_builder.lflist import LFList, parse_lflist
from ygo_meta.deck_builder.validator import validate_deck


SAMPLE_LFLIST = """\
!Test Banlist
# a comment
$whitelist
11111111 0 -- Forbidden Card
22222222 1 -- Limited Card
33333333 2 -- Semi-Limited Card

!Other Banlist
44444444 0 -- Different forbidden
"""


def _padded_main(extra_codes: list[int]) -> list[int]:
    # Pad with distinct dummy codes to reach 40 cards without tripping 3-copy.
    pad = [90000000 + i for i in range(40 - len(extra_codes))]
    return extra_codes + pad


def test_parse_lflist_first_section(tmp_path: Path) -> None:
    p = tmp_path / "x.conf"
    p.write_text(SAMPLE_LFLIST, encoding="utf-8")
    lf = parse_lflist(p)
    assert lf.name == "Test Banlist"
    assert lf.limits == {11111111: 0, 22222222: 1, 33333333: 2}
    assert lf.max_copies(11111111) == 0
    assert lf.max_copies(22222222) == 1
    assert lf.max_copies(99999999) == 3  # unlisted = unlimited


def test_parse_lflist_named_section(tmp_path: Path) -> None:
    p = tmp_path / "x.conf"
    p.write_text(SAMPLE_LFLIST, encoding="utf-8")
    lf = parse_lflist(p, banlist_name="Other Banlist")
    assert lf.name == "Other Banlist"
    assert lf.limits == {44444444: 0}


def test_validate_flags_forbidden_card() -> None:
    lf = LFList(name="t", limits={11111111: 0})
    deck = Deck("a", "v0", main=_padded_main([11111111]), extra=[], side=[])
    errors = validate_deck(deck, lflist=lf)
    assert any("11111111" in e and "limit: 0" in e for e in errors)


def test_validate_flags_over_limited() -> None:
    lf = LFList(name="t", limits={22222222: 1})
    deck = Deck("a", "v0", main=_padded_main([22222222, 22222222]), extra=[], side=[])
    errors = validate_deck(deck, lflist=lf)
    assert any("22222222" in e and "limit: 1" in e for e in errors)


def test_validate_extra_deck_counted_for_banlist() -> None:
    lf = LFList(name="t", limits={55555555: 1})
    deck = Deck(
        "a",
        "v0",
        main=_padded_main([]),
        extra=[55555555, 55555555],
        side=[],
    )
    errors = validate_deck(deck, lflist=lf)
    assert any("55555555" in e and "limit: 1" in e for e in errors)


def test_validate_passes_when_under_limit() -> None:
    lf = LFList(name="t", limits={22222222: 1})
    deck = Deck("a", "v0", main=_padded_main([22222222]), extra=[], side=[])
    errors = validate_deck(deck, lflist=lf)
    assert errors == []


def test_validate_without_lflist_unchanged() -> None:
    deck = Deck("a", "v0", main=_padded_main([22222222]), extra=[], side=[])
    assert validate_deck(deck) == []
