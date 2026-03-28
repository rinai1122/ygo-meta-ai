"""
Demo: run the LLM agent on realistic synthetic game states.

Usage:
    python scripts/demo_agent.py

Requires ANTHROPIC_API_KEY in environment.
No ygoinf server or ygoenv needed — game states are constructed directly.
Card stats (ATK, DEF, level, attribute) are loaded from data/card_info.json
so they are always accurate and never hardcoded.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ygo_meta.engine.types import (
    BattleAction,
    Card,
    CardAction,
    ChainAction,
    Global,
    IdleAction,
    Input,
    MsgSelectBattleCmd,
    MsgSelectCard,
    MsgSelectChain,
    MsgSelectIdleCmd,
    MsgSelectYesNo,
)
from ygo_meta.llm_agent.agent import LLMAgent
from ygo_meta.llm_agent.card_db import load_card_info
from ygo_meta.llm_agent.formatter import format_input

DIVIDER = "=" * 72

# ---------------------------------------------------------------------------
# Card builder — pulls real stats from card_info.json
# ---------------------------------------------------------------------------

def card_from_db(
    code: int,
    controller: int = 0,
    location: int = 0x02,   # Hand
    position: int = 0x1,    # faceup_attack
    sequence: int = 0,
) -> Card:
    """Build a Card using real ATK/DEF/level/attribute from card_info.json."""
    info = load_card_info().get(code, {})
    return Card(
        code=code,
        controller=controller,
        location=location,
        sequence=sequence,
        position=position,
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


def unknown_card(controller: int = 1, location: int = 0x04, sequence: int = 0) -> Card:
    """Opponent's face-down / unknown card (code=0)."""
    return Card(
        code=0, controller=controller, location=location, sequence=sequence,
        position=0x8,  # facedown_defense
        attribute=0, race=0, level=0, atk=0, **{"def": 0},
        types=0, counter=0, negated=0, overlay_count=0,
    )


def dummy_card() -> Card:
    """Placeholder for phase-change actions that don't involve a specific card."""
    return Card(
        code=0, controller=0, location=0x02, sequence=0, position=0x1,
        attribute=0, race=0, level=0, atk=0, **{"def": 0},
        types=0, counter=0, negated=0, overlay_count=0,
    )


def make_global(
    my_lp: int = 8000, opp_lp: int = 8000, turn: int = 1,
    phase: int = 0x04, is_first: int = 1, is_my_turn: int = 1,
) -> Global:
    return Global(lp=[my_lp, opp_lp], turn=turn, phase=phase,
                  is_first=is_first, is_my_turn=is_my_turn, num_cards=[5, 5])


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_scenario(agent: LLMAgent, title: str, input_: Input) -> None:
    print(f"\n{DIVIDER}")
    print(f"  SCENARIO: {title}")
    print(DIVIDER)
    gs, board, actions = format_input(input_)
    print(gs)
    print()
    print(board)
    print()
    print("=== OPTIONS ===")
    print(actions)
    print()
    choice = agent.choose_action(input_)
    action_lines = actions.strip().splitlines()
    chosen_line = next((l for l in action_lines if l.startswith(f"[{choice}]")), f"[{choice}]")
    print(f">>> LLM chose: {chosen_line}")


# ---------------------------------------------------------------------------
# Scenarios — all card stats come from card_from_db(), never hardcoded
# ---------------------------------------------------------------------------

def scenario_branded_opening() -> tuple[str, Input]:
    """Turn 1 Main Phase 1 (going first). Branded Dracotail hand."""
    faimena = card_from_db(1498449,  sequence=0)   # Dracotail Faimena
    rahu1   = card_from_db(32548318, sequence=1)   # Rahu Dracotail (Quick-Play Spell)
    rahu2   = card_from_db(32548318, sequence=2)
    albaz   = card_from_db(68468459, sequence=3)   # Fallen of Albaz
    ash     = card_from_db(14558127, sequence=4)   # Ash Blossom

    _dummy = dummy_card()
    actions = MsgSelectIdleCmd(actions=[
        IdleAction(card=faimena, msg=0),   # Normal Summon Dracotail Faimena
        IdleAction(card=albaz,   msg=0),   # Normal Summon Fallen of Albaz
        IdleAction(card=rahu1,   msg=4),   # Set Rahu Dracotail
        IdleAction(card=rahu2,   msg=5),   # Activate Rahu Dracotail
        IdleAction(card=_dummy,  msg=6),   # Enter Battle Phase
        IdleAction(card=_dummy,  msg=8),   # End Turn
    ])
    return "Branded Dracotail — Turn 1 Main Phase (going first)", Input(**{
        "global": make_global(),
        "cards": [faimena, rahu1, rahu2, albaz, ash],
        "action_msg": actions,
    })


def scenario_vs_battle_phase() -> tuple[str, Input]:
    """Turn 3 Battle Phase. Vanquish Soul Caesar Valius vs unknown opponent monster."""
    caesar  = card_from_db(91073013, location=0x04, sequence=0)  # Caesar Valius on field
    dr_mad  = card_from_db(29280200, location=0x04, sequence=1)  # Dr. Mad Love on field
    hand_c  = card_from_db(80181649, location=0x02, sequence=0)  # "A Case for K9" in hand
    opp_mon = unknown_card(controller=1, location=0x04, sequence=0)

    actions = MsgSelectBattleCmd(actions=[
        BattleAction(card=caesar, msg=0),  # Attack with Caesar Valius
        BattleAction(card=caesar, msg=1),  # Activate Caesar effect
        BattleAction(card=None,   msg=2),  # Go to Main Phase 2
        BattleAction(card=None,   msg=3),  # End turn
    ])
    return "K9 Vanquishsoul — Turn 3 Battle Phase", Input(**{
        "global": make_global(my_lp=7200, opp_lp=5600, turn=3, phase=0x10),
        "cards": [caesar, dr_mad, hand_c, opp_mon],
        "action_msg": actions,
    })


def scenario_chain_opportunity() -> tuple[str, Input]:
    """Opponent's turn. Opponent activates a spell — chain Ash Blossom, Nibiru, or pass."""
    ash    = card_from_db(14558127, sequence=0)  # Ash Blossom
    nibiru = card_from_db(27204311, sequence=1)  # Nibiru
    opp_st = unknown_card(controller=1, location=0x08, sequence=0)

    actions = MsgSelectChain(
        actions=[ChainAction(card=ash, effect=0), ChainAction(card=nibiru, effect=0)],
        forced=0,
    )
    return "Opponent's Turn — Chain Window (Ash / Nibiru / Pass)", Input(**{
        "global": make_global(turn=2, phase=0x04, is_first=0, is_my_turn=0),
        "cards": [ash, nibiru, opp_st],
        "action_msg": actions,
    })


def scenario_card_selection() -> tuple[str, Input]:
    """Branded Fusion resolving — select 1 fusion material from hand/field."""
    albaz    = card_from_db(68468459, location=0x02, sequence=0)  # Fallen of Albaz (hand)
    cartesia = card_from_db(95515789, location=0x02, sequence=1)  # Blazing Cartesia (hand)
    faimena  = card_from_db(1498449,  location=0x04, sequence=0)  # Dracotail Faimena (field)

    actions = MsgSelectCard(
        actions=[CardAction(card=albaz), CardAction(card=cartesia), CardAction(card=faimena)],
        **{"min": 1, "max": 1},
    )
    return "Branded Fusion — Select 1 Fusion Material", Input(**{
        "global": make_global(),
        "cards": [albaz, cartesia, faimena],
        "action_msg": actions,
    })


def scenario_yesno_trigger() -> tuple[str, Input]:
    """Opponent searches — activate Ash Blossom?"""
    ash = card_from_db(14558127, sequence=0)

    return "Opponent searches — Activate Ash Blossom? (Yes/No)", Input(**{
        "global": make_global(my_lp=6000, opp_lp=8000, turn=4, phase=0x04,
                              is_first=0, is_my_turn=0),
        "cards": [ash],
        "action_msg": MsgSelectYesNo(card=ash, effect=0),
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    model = os.environ.get("YGO_MODEL", "claude-haiku-4-5-20251001")
    print(f"Model: {model}")
    agent = LLMAgent(model=model, max_tokens=16)

    for title, inp in [
        scenario_branded_opening(),
        scenario_vs_battle_phase(),
        scenario_chain_opportunity(),
        scenario_card_selection(),
        scenario_yesno_trigger(),
    ]:
        run_scenario(agent, title, inp)

    print(f"\n{DIVIDER}\n  Demo complete.\n{DIVIDER}")


if __name__ == "__main__":
    main()
