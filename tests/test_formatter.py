"""Tests for the game state formatter."""

import pytest

from ygo_meta.engine.types import (
    Global,
    Input,
    MsgSelectIdleCmd,
    IdleAction,
    MsgSelectYesNo,
    MsgSelectChain,
    ChainAction,
)
from ygo_meta.llm_agent.card_db import load_card_info
from ygo_meta.llm_agent.formatter import format_input, _build_action_block
from ygo_meta.engine.types import Card


def card_from_db(code: int, controller: int = 0, location: int = 0x02) -> Card:
    """Build a Card using real stats from card_info.json."""
    info = load_card_info().get(code, {})
    return Card(
        code=code,
        controller=controller,
        location=location,
        sequence=0,
        position=0x1,
        attribute=info.get("attribute", 0),
        race=0,
        level=info.get("level", 0),
        atk=info.get("atk") or 0,
        **{"def": info.get("def") or 0},
        types=0,
        counter=0,
        negated=0,
        overlay_count=0,
    )


def make_global() -> Global:
    return Global(lp=[8000, 8000], turn=1, phase=0x04, is_first=1, is_my_turn=1, num_cards=[5, 5])


# Use Ash Blossom (14558127) as the test card — a real, stable card in MD
_TEST_CODE = 14558127


def test_format_input_returns_three_blocks():
    card = card_from_db(_TEST_CODE)
    msg = MsgSelectIdleCmd(actions=[IdleAction(card=card, msg=5)])
    inp = Input(**{"global": make_global(), "cards": [card], "action_msg": msg})
    gs, board, actions = format_input(inp)
    assert "GAME STATE" in gs
    assert "BOARD" in board
    assert "[0]" in actions


def test_unknown_card_renders_unknown():
    card = card_from_db(0)
    msg = MsgSelectIdleCmd(actions=[IdleAction(card=card, msg=0)])
    inp = Input(**{"global": make_global(), "cards": [card], "action_msg": msg})
    gs, board, actions = format_input(inp)
    assert "[unknown]" in board


def test_yesno_has_two_options():
    card = card_from_db(_TEST_CODE)
    text = _build_action_block(MsgSelectYesNo(card=card, effect=0))
    assert "[0]" in text
    assert "[1]" in text


def test_chain_includes_do_not_chain():
    card = card_from_db(_TEST_CODE)
    text = _build_action_block(MsgSelectChain(actions=[ChainAction(card=card, effect=0)], forced=0))
    assert "Do not chain" in text


def test_chain_forced_no_pass():
    card = card_from_db(_TEST_CODE)
    text = _build_action_block(MsgSelectChain(actions=[ChainAction(card=card, effect=0)], forced=1))
    assert "Do not chain" not in text
