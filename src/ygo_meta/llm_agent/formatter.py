"""
Converts the ygoinf Input model into human-readable text for the LLM.

Uses vendor types directly (ygoinf.models). ActionMsg.data holds the actual
message variant; never access ActionMsg directly as if it were the message.

Output blocks:
  GAME STATE  — turn, phase, LP
  BOARD       — cards grouped by controller and location
  OPTIONS     — numbered list of valid actions
"""

from __future__ import annotations

from typing import Optional

from ygo_meta.engine.types import (
    ActionMsg,
    AnnounceAttrib,
    BattleCmd,
    BattleCmdType,
    Card,
    CardLocation,
    Chain,
    Controller,
    Global,
    IdleCmd,
    IdleCmdType,
    Input,
    Location,
    MsgAnnounceAttrib,
    MsgAnnounceNumber,
    MsgSelectBattleCmd,
    MsgSelectCard,
    MsgSelectChain,
    MsgSelectDisfield,
    MsgSelectEffectYn,
    MsgSelectIdleCmd,
    MsgSelectOption,
    MsgSelectPlace,
    MsgSelectPosition,
    MsgSelectSum,
    MsgSelectTribute,
    MsgSelectUnselectCard,
    MsgSelectYesNo,
    Position,
    SelectSumCard,
)
from ygo_meta.llm_agent.card_db import get_card_name, get_card_type


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
    pos = card.position.value
    attr = card.attribute.value.upper()
    return f"{name} ATK {card.attack}/DEF {card.defense} | {pos} | Lv{card.level} | {attr}"


def _find_card_at(cards: list[Card], loc: CardLocation) -> Optional[Card]:
    """Look up a card by its location in the cards list."""
    for c in cards:
        if (c.controller.value == loc.controller.value
                and c.location.value == loc.location.value
                and c.sequence == loc.sequence):
            return c
    return None


def _card_name_at(cards: list[Card], loc: CardLocation) -> str:
    card = _find_card_at(cards, loc)
    if card and card.code:
        return f"[{get_card_name(card.code)}]"
    return "[unknown]"


# ---------------------------------------------------------------------------
# Block builders
# ---------------------------------------------------------------------------

def _build_game_state_block(g: Global) -> str:
    phase = g.phase.value.upper()
    first = "Yes" if g.is_first else "No"
    my_turn = "Yes" if g.is_my_turn else "No"
    return (
        "=== GAME STATE ===\n"
        f"Turn: {g.turn} | Phase: {phase} | You go first: {first} | Your turn: {my_turn}\n"
        f"Your LP: {g.my_lp} | Opponent LP: {g.op_lp}"
    )


def _build_board_block(cards: list[Card]) -> str:
    def group(ctrl_value: str) -> dict[str, list[Card]]:
        result: dict[str, list[Card]] = {}
        for c in cards:
            if c.controller.value == ctrl_value:
                result.setdefault(c.location.value, []).append(c)
        return result

    def render_location(loc_name: str, loc_cards: list[Card], hidden: bool) -> str:
        if hidden:
            return f"  {loc_name}: {len(loc_cards)} card(s)"
        parts = ", ".join(_card_str(c) for c in loc_cards)
        return f"  {loc_name}: {parts}"

    lines = ["=== BOARD ==="]
    for label, ctrl, hide_hand in [("YOUR FIELD", "me", False), ("OPPONENT FIELD", "opponent", True)]:
        lines.append(label + ":")
        groups = group(ctrl)
        for loc_name in sorted(groups):
            hidden = hide_hand and loc_name in ("hand", "deck")
            lines.append(render_location(loc_name, groups[loc_name], hidden))
        if not groups:
            lines.append("  (empty)")
    return "\n".join(lines)


def _build_action_block(msg, cards: list[Card] = ()) -> str:
    """
    Build the OPTIONS block from a message variant (not ActionMsg wrapper).
    Pass `cards` (from Input.cards) so card names can be looked up by location.
    """
    lines: list[str] = []

    if isinstance(msg, MsgSelectIdleCmd):
        for i, cmd in enumerate(msg.idle_cmds):
            label = cmd.cmd_type.value.replace("_", " ")
            code = cmd.data.card_info.code if cmd.data else 0
            name = get_card_name(code) if code else ""
            lines.append(f"[{i}] {label}" + (f" {name}" if name else ""))

    elif isinstance(msg, MsgSelectBattleCmd):
        for i, cmd in enumerate(msg.battle_cmds):
            label = cmd.cmd_type.value.replace("_", " ")
            code = cmd.data.card_info.code if cmd.data else 0
            name = get_card_name(code) if code else ""
            lines.append(f"[{i}] {label}" + (f" with {name}" if name else ""))

    elif isinstance(msg, MsgSelectChain):
        for i, chain in enumerate(msg.chains):
            lines.append(f"[{i}] Chain [{get_card_name(chain.code)}]")
        if not msg.forced:
            lines.append(f"[{len(msg.chains)}] Do not chain")

    elif isinstance(msg, MsgSelectCard):
        for i, sel in enumerate(msg.cards):
            lines.append(f"[{i}] Select {_card_name_at(list(cards), sel.location)}")
        lines.append(f"(select {msg.min}–{msg.max} card(s))")

    elif isinstance(msg, MsgSelectTribute):
        for i, sel in enumerate(msg.cards):
            lines.append(f"[{i}] Tribute {_card_name_at(list(cards), sel.location)} (Lv{sel.level})")
        lines.append(f"(tribute {msg.min}–{msg.max} monster(s))")

    elif isinstance(msg, MsgSelectSum):
        all_cards = list(msg.must_cards) + list(msg.cards)
        for i, sel in enumerate(all_cards):
            name = _card_name_at(list(cards), sel.location)
            lv = sel.level1 or sel.level2
            lines.append(f"[{i}] Select {name} (Lv{lv})")
        lines.append(f"(target level sum: {msg.level_sum})")

    elif isinstance(msg, MsgSelectPosition):
        for i, pos in enumerate(msg.positions):
            lines.append(f"[{i}] {pos.value}")

    elif isinstance(msg, MsgSelectYesNo):
        lines.append("[0] Yes")
        lines.append("[1] No")

    elif isinstance(msg, MsgSelectEffectYn):
        lines.append(f"[0] Activate effect of [{get_card_name(msg.code)}]")
        lines.append("[1] Do not activate")

    elif isinstance(msg, MsgAnnounceAttrib):
        for i, a in enumerate(msg.attributes):
            lines.append(f"[{i}] {a.attribute.value.upper()}")
        lines.append(f"(announce {msg.count} attribute(s))")

    elif isinstance(msg, MsgAnnounceNumber):
        for i, a in enumerate(msg.numbers):
            lines.append(f"[{i}] {a.number}")

    elif isinstance(msg, (MsgSelectPlace, MsgSelectDisfield)):
        for i, place in enumerate(msg.places):
            who = "my" if place.controller.value == "me" else "opponent's"
            lines.append(f"[{i}] Place on {who} {place.location.value} slot {place.sequence}")

    elif isinstance(msg, MsgSelectUnselectCard):
        for i, sel in enumerate(msg.selectable_cards):
            lines.append(f"[{i}] Select {_card_name_at(list(cards), sel.location)}")

    elif isinstance(msg, MsgSelectOption):
        for i, opt in enumerate(msg.options):
            lines.append(f"[{i}] Option {opt.code}")

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
    actions = _build_action_block(input_.action_msg.data, input_.cards)
    return game_state, board, actions


def format_prompt(input_: Input, template: str) -> str:
    game_state, board, actions = format_input(input_)
    return template.format(
        game_state_block=game_state,
        board_block=board,
        action_block=actions,
    )
