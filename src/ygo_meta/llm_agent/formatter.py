"""
Converts the ygoinf Input model into human-readable text for the LLM.

The output has three blocks:
  GAME STATE  — turn, phase, LP
  BOARD       — cards grouped by controller and location
  OPTIONS     — numbered list of valid actions
"""

from __future__ import annotations

from ygo_meta.engine.types import (
    ActionMsg,
    Card,
    Global,
    Input,
    MsgAnnounceAttrib,
    MsgAnnounceNumber,
    MsgSelectBattleCmd,
    MsgSelectCard,
    MsgSelectChain,
    MsgSelectDisField,
    MsgSelectEffectYn,
    MsgSelectIdleCmd,
    MsgSelectPlace,
    MsgSelectPosition,
    MsgSelectSum,
    MsgSelectTribute,
    MsgSelectYesNo,
)
from ygo_meta.llm_agent.card_db import get_card_name, get_card_type

# ---------------------------------------------------------------------------
# Location / phase / attribute / race constants (from ygopro-core)
# ---------------------------------------------------------------------------

LOCATION_NAMES = {
    0x01: "Deck",
    0x02: "Hand",
    0x04: "Monster Zone",
    0x08: "Spell/Trap Zone",
    0x10: "Graveyard",
    0x20: "Banished",
    0x40: "Extra Deck",
    0x80: "Extra Monster Zone",
}

PHASE_NAMES = {
    0x01: "DRAW",
    0x02: "STANDBY",
    0x04: "MAIN1",
    0x08: "BATTLE_START",
    0x10: "BATTLE_STEP",
    0x20: "DAMAGE",
    0x40: "DAMAGE_CALC",
    0x80: "BATTLE",
    0x100: "MAIN2",
    0x200: "END",
}

ATTRIBUTE_NAMES = {
    0x01: "EARTH", 0x02: "WATER", 0x04: "FIRE",
    0x08: "WIND", 0x10: "LIGHT", 0x20: "DARK", 0x40: "DIVINE",
}

POSITION_NAMES = {
    0x1: "faceup_attack",
    0x2: "facedown_attack",
    0x4: "faceup_defense",
    0x8: "facedown_defense",
}

IDLE_MSG_NAMES = {
    0: "Summon",
    1: "Set (monster)",
    2: "Special Summon",
    3: "Reposition",
    4: "Set (spell/trap)",
    5: "Activate",
    6: "Enter Battle Phase",
    7: "Enter Main Phase 2",
    8: "End Turn",
}

BATTLE_MSG_NAMES = {
    0: "Attack",
    1: "Activate effect",
    2: "Go to Main Phase 2",
    3: "End turn",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _card_str(card: Card, brief: bool = False) -> str:
    if card.code == 0:
        return "[unknown]"
    name = f"[{get_card_name(card.code)}]"
    if brief:
        return name
    ctype = get_card_type(card.code)
    if ctype == "spell":
        return f"{name} | Spell"
    if ctype == "trap":
        return f"{name} | Trap"
    pos  = POSITION_NAMES.get(card.position, f"pos={card.position}")
    attr = ATTRIBUTE_NAMES.get(card.attribute, "?")
    return f"{name} ATK {card.atk}/DEF {card.def_} | {pos} | Lv{card.level} | {attr}"


def _phase_str(phase: int) -> str:
    for mask, name in PHASE_NAMES.items():
        if phase & mask:
            return name
    return f"phase={phase}"


# ---------------------------------------------------------------------------
# Block builders
# ---------------------------------------------------------------------------

def _build_game_state_block(g: Global) -> str:
    my_lp, opp_lp = g.lp[0], g.lp[1]
    phase = _phase_str(g.phase)
    first = "Yes" if g.is_first else "No"
    my_turn = "Yes" if g.is_my_turn else "No"
    return (
        "=== GAME STATE ===\n"
        f"Turn: {g.turn} | Phase: {phase} | You go first: {first} | Your turn: {my_turn}\n"
        f"Your LP: {my_lp} | Opponent LP: {opp_lp}"
    )


def _build_board_block(cards: list[Card]) -> str:
    def group(controller: int) -> dict[int, list[Card]]:
        result: dict[int, list[Card]] = {}
        for c in cards:
            if c.controller == controller:
                result.setdefault(c.location, []).append(c)
        return result

    def render_location(loc: int, loc_cards: list[Card], hidden: bool) -> str:
        loc_name = LOCATION_NAMES.get(loc, f"zone={loc}")
        if hidden:
            return f"  {loc_name}: {len(loc_cards)} card(s)"
        parts = ", ".join(_card_str(c) for c in loc_cards)
        return f"  {loc_name}: {parts}"

    lines = ["=== BOARD ==="]
    for label, ctrl, hide_hand in [("YOUR FIELD", 0, False), ("OPPONENT FIELD", 1, True)]:
        lines.append(label + ":")
        groups = group(ctrl)
        for loc in sorted(groups):
            # Hide opponent's hand and deck
            hidden = hide_hand and loc in (0x01, 0x02)
            lines.append(render_location(loc, groups[loc], hidden))
        if not groups:
            lines.append("  (empty)")
    return "\n".join(lines)


def _build_action_block(msg: ActionMsg) -> str:
    lines: list[str] = []

    if isinstance(msg, MsgSelectIdleCmd):
        for i, a in enumerate(msg.actions):
            action_name = IDLE_MSG_NAMES.get(a.msg, f"action={a.msg}")
            card_name = get_card_name(a.card.code)
            lines.append(f"[{i}] {action_name} {card_name}" if card_name != "unknown" else f"[{i}] {action_name}")

    elif isinstance(msg, MsgSelectBattleCmd):
        for i, a in enumerate(msg.actions):
            action_name = BATTLE_MSG_NAMES.get(a.msg, f"action={a.msg}")
            if a.card:
                lines.append(f"[{i}] {action_name} with {get_card_name(a.card.code)}")
            else:
                lines.append(f"[{i}] {action_name}")

    elif isinstance(msg, MsgSelectChain):
        for i, a in enumerate(msg.actions):
            lines.append(f"[{i}] Chain {get_card_name(a.card.code)}")
        if not msg.forced:
            lines.append(f"[{len(msg.actions)}] Do not chain")

    elif isinstance(msg, MsgSelectCard):
        for i, a in enumerate(msg.actions):
            lines.append(f"[{i}] Select {_card_str(a.card, brief=True)}")
        lines.append(f"(select {msg.min_}–{msg.max_} card(s))")

    elif isinstance(msg, MsgSelectTribute):
        for i, a in enumerate(msg.actions):
            lines.append(f"[{i}] Tribute {_card_str(a.card, brief=True)}")
        lines.append(f"(tribute {msg.min_}–{msg.max_} monster(s))")

    elif isinstance(msg, MsgSelectSum):
        for i, a in enumerate(msg.actions):
            lines.append(f"[{i}] Select {_card_str(a.card, brief=True)} (Lv{a.level})")
        lines.append(f"(target level sum: {msg.level_sum})")

    elif isinstance(msg, MsgSelectPosition):
        for i, a in enumerate(msg.actions):
            pos_name = POSITION_NAMES.get(a.position, f"pos={a.position}")
            lines.append(f"[{i}] {pos_name}")

    elif isinstance(msg, MsgSelectYesNo):
        lines.append(f"[0] Yes — activate effect of {get_card_name(msg.card.code)}")
        lines.append("[1] No")

    elif isinstance(msg, MsgSelectEffectYn):
        lines.append(f"[0] Activate effect of {get_card_name(msg.card.code)}")
        lines.append("[1] Do not activate")

    elif isinstance(msg, MsgAnnounceAttrib):
        for i, a in enumerate(msg.actions):
            attr_name = ATTRIBUTE_NAMES.get(a.attribute, f"attr={a.attribute}")
            lines.append(f"[{i}] {attr_name}")
        lines.append(f"(announce {msg.count} attribute(s))")

    elif isinstance(msg, MsgAnnounceNumber):
        for i, a in enumerate(msg.actions):
            lines.append(f"[{i}] {a.number}")

    elif isinstance(msg, (MsgSelectPlace, MsgSelectDisField)):
        for i, a in enumerate(msg.actions):
            who = "my" if a.controller == 0 else "opponent's"
            loc_name = LOCATION_NAMES.get(a.location, f"zone={a.location}")
            lines.append(f"[{i}] Place on {who} {loc_name} slot {a.sequence}")

    else:
        lines.append("[0] (action)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def format_input(input_: Input) -> tuple[str, str, str]:
    """Return (game_state_block, board_block, action_block)."""
    game_state = _build_game_state_block(input_.global_)
    board = _build_board_block(input_.cards)
    actions = _build_action_block(input_.action_msg)
    return game_state, board, actions


def format_prompt(input_: Input, template: str) -> str:
    game_state, board, actions = format_input(input_)
    return template.format(
        game_state_block=game_state,
        board_block=board,
        action_block=actions,
    )
