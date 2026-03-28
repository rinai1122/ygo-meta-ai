"""Tests for the game state formatter."""

import pytest

from ygo_meta.engine.types import (
    ActionMsg,
    AnnounceAttrib,
    Attribute,
    BattleCmd,
    BattleCmdType,
    Card,
    CardInfo,
    CardLocation,
    Chain,
    Controller,
    Global,
    IdleCmd,
    IdleCmdData,
    IdleCmdType,
    Input,
    Location,
    MsgAnnounceAttrib,
    MsgSelectBattleCmd,
    MsgSelectChain,
    MsgSelectEffectYn,
    MsgSelectIdleCmd,
    MsgSelectYesNo,
    Phase,
    Position,
    Race,
    Type,
)
from ygo_meta.llm_agent.card_db import card_from_db
from ygo_meta.llm_agent.formatter import format_input, _build_action_block


def make_global() -> Global:
    return Global(my_lp=8000, op_lp=8000, turn=1, phase="main1", is_first=True, is_my_turn=True)


def make_card_info(code: int) -> CardInfo:
    return CardInfo(code=code, controller="me", location="hand", sequence=0)


def make_card_location(code: int = 0) -> CardLocation:
    return CardLocation(controller="me", location="hand", sequence=0, overlay_sequence=-1)


def _wrap_msg(variant) -> ActionMsg:
    return ActionMsg(data=variant)


# Use Ash Blossom (14558127) as the test card — a real, stable card in MD
_TEST_CODE = 14558127


def _dummy_opp_card() -> Card:
    """Minimal opponent card to satisfy Input's min 2 cards constraint."""
    return Card(
        code=0, controller="opponent", location="hand", sequence=0,
        position="faceup_attack", overlay_sequence=-1,
        attribute="none", race="none", level=0,
        counter=0, negated=False, attack=0, defense=0, types=[],
    )


def test_format_input_returns_three_blocks():
    card = card_from_db(_TEST_CODE)
    cmd = IdleCmd(
        cmd_type="activate",
        data=IdleCmdData(card_info=make_card_info(_TEST_CODE), effect_description=0, response=0),
    )
    msg = MsgSelectIdleCmd(msg_type="select_idlecmd", idle_cmds=[cmd])
    inp = Input(**{"global": make_global(), "cards": [card, _dummy_opp_card()], "action_msg": _wrap_msg(msg)})
    gs, board, actions = format_input(inp)
    assert "GAME STATE" in gs
    assert "BOARD" in board
    assert "[0]" in actions


def test_unknown_card_renders_unknown():
    card = card_from_db(0)
    cmd = IdleCmd(cmd_type="summon", data=None)
    msg = MsgSelectIdleCmd(msg_type="select_idlecmd", idle_cmds=[cmd])
    inp = Input(**{"global": make_global(), "cards": [card, _dummy_opp_card()], "action_msg": _wrap_msg(msg)})
    gs, board, actions = format_input(inp)
    assert "[unknown]" in board


def test_yesno_has_two_options():
    msg = MsgSelectYesNo(msg_type="select_yesno", effect_description=0)
    text = _build_action_block(msg)
    assert "[0]" in text
    assert "[1]" in text


def test_chain_includes_do_not_chain():
    chain = Chain(
        code=_TEST_CODE,
        location=make_card_location(),
        effect_description=0,
        response=0,
    )
    msg = MsgSelectChain(msg_type="select_chain", forced=False, chains=[chain])
    text = _build_action_block(msg)
    assert "Do not chain" in text


def test_chain_forced_no_pass():
    chain = Chain(
        code=_TEST_CODE,
        location=make_card_location(),
        effect_description=0,
        response=0,
    )
    msg = MsgSelectChain(msg_type="select_chain", forced=True, chains=[chain])
    text = _build_action_block(msg)
    assert "Do not chain" not in text
