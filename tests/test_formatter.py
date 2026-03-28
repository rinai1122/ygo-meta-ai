"""Tests for the game state formatter."""

import pytest

from ygo_meta.engine.types import (
    Card,
    Global,
    Input,
    MsgSelectIdleCmd,
    IdleAction,
    MsgSelectYesNo,
    MsgSelectChain,
    ChainAction,
)
from ygo_meta.llm_agent.formatter import format_input, _build_action_block


def make_card(code: int = 23434538, controller: int = 0, location: int = 0x02) -> Card:
    return Card(
        code=code,
        controller=controller,
        location=location,
        sequence=0,
        position=0x1,  # faceup_attack
        attribute=0x10,  # LIGHT
        race=0x40,  # Dragon
        level=4,
        atk=1800,
        **{"def": 1000},
        types=0,
        counter=0,
        negated=0,
        overlay_count=0,
    )


def make_global() -> Global:
    return Global(lp=[8000, 8000], turn=1, phase=0x04, is_first=1, is_my_turn=1, num_cards=[5, 5])


def test_format_input_returns_three_blocks():
    card = make_card()
    idle_action = IdleAction(card=card, msg=5)  # Activate
    msg = MsgSelectIdleCmd(actions=[idle_action])
    inp = Input(**{"global": make_global(), "cards": [card], "action_msg": msg})
    gs, board, actions = format_input(inp)
    assert "GAME STATE" in gs
    assert "BOARD" in board
    assert "[0]" in actions


def test_unknown_card_renders_unknown():
    card = make_card(code=0)
    idle_action = IdleAction(card=card, msg=0)
    msg = MsgSelectIdleCmd(actions=[idle_action])
    inp = Input(**{"global": make_global(), "cards": [card], "action_msg": msg})
    gs, board, actions = format_input(inp)
    assert "[unknown]" in board


def test_yesno_has_two_options():
    card = make_card()
    msg = MsgSelectYesNo(card=card, effect=0)
    text = _build_action_block(msg)
    assert "[0]" in text
    assert "[1]" in text


def test_chain_includes_do_not_chain():
    card = make_card()
    msg = MsgSelectChain(actions=[ChainAction(card=card, effect=0)], forced=0)
    text = _build_action_block(msg)
    assert "Do not chain" in text


def test_chain_forced_no_pass():
    card = make_card()
    msg = MsgSelectChain(actions=[ChainAction(card=card, effect=0)], forced=1)
    text = _build_action_block(msg)
    assert "Do not chain" not in text
