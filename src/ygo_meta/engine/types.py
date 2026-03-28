"""
Pydantic models mirroring vendor/ygo-agent/ygoinf/ygoinf/models.py.

These are copied (not imported) to avoid depending on the ygo-agent Python package
directly. When ygo-agent updates its models, update this file to match.

Source reference: vendor/ygo-agent/ygoinf/ygoinf/models.py
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Global game state
# ---------------------------------------------------------------------------


class Global(BaseModel):
    lp: list[int]           # [my_lp, opp_lp]
    turn: int
    phase: int              # encoded phase index
    is_first: int           # 1 if this player goes first
    is_my_turn: int         # 1 if it's this player's turn
    num_cards: list[int]    # card counts per location


# ---------------------------------------------------------------------------
# Card representation
# ---------------------------------------------------------------------------


class Card(BaseModel):
    code: int               # card code (0 = unknown/face-down)
    controller: int         # 0 = me, 1 = opponent
    location: int           # encoded location
    sequence: int
    position: int           # face-up/down, attack/defense
    attribute: int
    race: int
    level: int
    atk: int
    def_: int = Field(alias="def")
    types: int              # bitfield
    counter: int
    negated: int
    overlay_count: int

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Action messages (discriminated union)
# ---------------------------------------------------------------------------


class IdleAction(BaseModel):
    card: Card
    msg: int    # action type code


class MsgSelectIdleCmd(BaseModel):
    msg: Literal["select_idlecmd"] = "select_idlecmd"
    actions: list[IdleAction]


class ChainAction(BaseModel):
    card: Card
    effect: int


class MsgSelectChain(BaseModel):
    msg: Literal["select_chain"] = "select_chain"
    actions: list[ChainAction]
    forced: int


class CardAction(BaseModel):
    card: Card


class MsgSelectCard(BaseModel):
    msg: Literal["select_card"] = "select_card"
    actions: list[CardAction]
    min_: int = Field(alias="min")
    max_: int = Field(alias="max")

    model_config = {"populate_by_name": True}


class MsgSelectTribute(BaseModel):
    msg: Literal["select_tribute"] = "select_tribute"
    actions: list[CardAction]
    min_: int = Field(alias="min")
    max_: int = Field(alias="max")

    model_config = {"populate_by_name": True}


class SumAction(BaseModel):
    card: Card
    level: int


class MsgSelectSum(BaseModel):
    msg: Literal["select_sum"] = "select_sum"
    actions: list[SumAction]
    overflow: int
    must_just: int
    level_sum: int


class PositionAction(BaseModel):
    position: int


class MsgSelectPosition(BaseModel):
    msg: Literal["select_position"] = "select_position"
    card: Card
    actions: list[PositionAction]


class BattleAction(BaseModel):
    card: Optional[Card] = None
    msg: int    # attack/activate/end


class MsgSelectBattleCmd(BaseModel):
    msg: Literal["select_battlecmd"] = "select_battlecmd"
    actions: list[BattleAction]


class MsgSelectYesNo(BaseModel):
    msg: Literal["select_yesno"] = "select_yesno"
    card: Card
    effect: int


class MsgSelectEffectYn(BaseModel):
    msg: Literal["select_effectyn"] = "select_effectyn"
    card: Card
    effect: int


class AttribAction(BaseModel):
    attribute: int


class MsgAnnounceAttrib(BaseModel):
    msg: Literal["announce_attrib"] = "announce_attrib"
    actions: list[AttribAction]
    count: int


class NumberAction(BaseModel):
    number: int


class MsgAnnounceNumber(BaseModel):
    msg: Literal["announce_number"] = "announce_number"
    actions: list[NumberAction]


class PlaceAction(BaseModel):
    controller: int
    location: int
    sequence: int


class MsgSelectPlace(BaseModel):
    msg: Literal["select_place"] = "select_place"
    actions: list[PlaceAction]


class MsgSelectDisField(BaseModel):
    msg: Literal["select_disfield"] = "select_disfield"
    actions: list[PlaceAction]


ActionMsg = Annotated[
    Union[
        MsgSelectIdleCmd,
        MsgSelectChain,
        MsgSelectCard,
        MsgSelectTribute,
        MsgSelectSum,
        MsgSelectPosition,
        MsgSelectBattleCmd,
        MsgSelectYesNo,
        MsgSelectEffectYn,
        MsgAnnounceAttrib,
        MsgAnnounceNumber,
        MsgSelectPlace,
        MsgSelectDisField,
    ],
    Field(discriminator="msg"),
]


# ---------------------------------------------------------------------------
# Top-level request / response
# ---------------------------------------------------------------------------


class Input(BaseModel):
    global_: Global = Field(alias="global")
    cards: list[Card]
    action_msg: ActionMsg

    model_config = {"populate_by_name": True}


class DuelPredictRequest(BaseModel):
    input: Input
    prev_action_idx: int
    index: int


class ActionPrediction(BaseModel):
    action_idx: int
    prob: float
    win_rate: Optional[float] = None


class DuelPredictResponse(BaseModel):
    predict_results: list[ActionPrediction]
    index: int


class DuelCreateResponse(BaseModel):
    duelId: str
    index: int
