"""Round-trip tests for YDK parser."""

import tempfile
from pathlib import Path

import pytest

from ygo_meta.deck_builder.ydk_parser import parse_ydk, write_ydk


YDK_CONTENT = """\
#created by test
#main
23434538
23434538
23434538
27204311
27204311
#extra
14558127
!side
"""


def test_parse_ydk(tmp_path: Path) -> None:
    ydk = tmp_path / "test.ydk"
    ydk.write_text(YDK_CONTENT, encoding="utf-8")
    deck = parse_ydk(ydk)
    assert deck.main == [23434538, 23434538, 23434538, 27204311, 27204311]
    assert deck.extra == [14558127]
    assert deck.side == []


def test_write_ydk_round_trip(tmp_path: Path) -> None:
    ydk = tmp_path / "test.ydk"
    ydk.write_text(YDK_CONTENT, encoding="utf-8")
    deck = parse_ydk(ydk, archetype="test", variant_id="test_v000")

    out = tmp_path / "out.ydk"
    write_ydk(deck, out)
    deck2 = parse_ydk(out, archetype="test", variant_id="test_v000")

    assert deck.main == deck2.main
    assert deck.extra == deck2.extra
    assert deck.side == deck2.side
