"""
Engine types: direct re-exports from vendor ygoinf.models.

Never redefine these locally. vendor/ygo-agent/ygoinf/ygoinf/models.py is the
single source of truth. Importing from here gives all downstream code the exact
types the inference server expects and returns.
"""
from ygoinf.models import (
    # Enums
    Controller,
    Location,
    Position,
    Attribute,
    Race,
    Type,
    Phase,
    IdleCmdType,
    BattleCmdType,
    # Sub-models
    CardInfo,
    CardLocation,
    Place,
    Chain,
    IdleCmdData,
    IdleCmd,
    BattleCmdData,
    BattleCmd,
    SelectAbleCard,
    SelectTributeCard,
    SelectSumCard,
    AnnounceAttrib,
    AnnounceNumber,
    SelectUnselectCard,
    Option,
    # Top-level game state
    Global,
    Card,
    # Action message variants
    MsgSelectIdleCmd,
    MsgSelectBattleCmd,
    MsgSelectChain,
    MsgSelectCard,
    MsgSelectTribute,
    MsgSelectSum,
    MsgSelectPosition,
    MsgSelectYesNo,
    MsgSelectEffectYn,
    MsgAnnounceAttrib,
    MsgAnnounceNumber,
    MsgSelectPlace,
    MsgSelectDisfield,
    MsgSelectUnselectCard,
    MsgSelectOption,
    # Wrapper
    ActionMsg,
    Input,
    # API request/response
    ActionPredict,
    MsgResponse,
    DuelPredictRequest,
    DuelPredictResponse,
    DuelCreateResponse,
)

__all__ = [
    "Controller", "Location", "Position", "Attribute", "Race", "Type", "Phase",
    "IdleCmdType", "BattleCmdType",
    "CardInfo", "CardLocation", "Place", "Chain",
    "IdleCmdData", "IdleCmd", "BattleCmdData", "BattleCmd",
    "SelectAbleCard", "SelectTributeCard", "SelectSumCard",
    "AnnounceAttrib", "AnnounceNumber", "SelectUnselectCard", "Option",
    "Global", "Card",
    "MsgSelectIdleCmd", "MsgSelectBattleCmd", "MsgSelectChain",
    "MsgSelectCard", "MsgSelectTribute", "MsgSelectSum",
    "MsgSelectPosition", "MsgSelectYesNo", "MsgSelectEffectYn",
    "MsgAnnounceAttrib", "MsgAnnounceNumber",
    "MsgSelectPlace", "MsgSelectDisfield",
    "MsgSelectUnselectCard", "MsgSelectOption",
    "ActionMsg", "Input",
    "ActionPredict", "MsgResponse",
    "DuelPredictRequest", "DuelPredictResponse", "DuelCreateResponse",
]
