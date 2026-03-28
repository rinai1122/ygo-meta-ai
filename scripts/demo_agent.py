"""
Demo: run the LLM agent on realistic synthetic game states.

Usage:
    python scripts/demo_agent.py

Requires ANTHROPIC_API_KEY in environment.
No ygoinf server or ygoenv needed — game states are constructed directly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make src/ importable when running as a script
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
from ygo_meta.llm_agent.formatter import format_input

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIVIDER = "=" * 72

def make_card(
    code: int,
    controller: int = 0,
    location: int = 0x02,   # Hand by default
    position: int = 0x1,    # faceup_attack
    atk: int = 1500,
    def_: int = 1000,
    level: int = 4,
    attribute: int = 0x20,  # DARK
    sequence: int = 0,
) -> Card:
    return Card(
        code=code,
        controller=controller,
        location=location,
        sequence=sequence,
        position=position,
        attribute=attribute,
        race=0x40,
        level=level,
        atk=atk,
        **{"def": def_},
        types=0,
        counter=0,
        negated=0,
        overlay_count=0,
    )


def make_global(
    my_lp: int = 8000,
    opp_lp: int = 8000,
    turn: int = 1,
    phase: int = 0x04,      # MAIN1
    is_first: int = 1,
    is_my_turn: int = 1,
) -> Global:
    return Global(lp=[my_lp, opp_lp], turn=turn, phase=phase,
                  is_first=is_first, is_my_turn=is_my_turn, num_cards=[5, 5])


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
# Scenarios
# ---------------------------------------------------------------------------

def scenario_branded_opening() -> tuple[str, Input]:
    """
    Turn 1, Main Phase 1 (going first).
    Hand: Dracotail Faimena, Rahu Dracotail x2, Fallen of Albaz, Ash Blossom.
    Board: empty.
    """
    # Hand cards (controller=0, location=0x02)
    faimena     = make_card(1498449,  location=0x02, atk=1500, def_=2000, level=4, attribute=0x20)
    rahu1       = make_card(32548318, location=0x02, atk=1600, def_=1000, level=4, attribute=0x04, sequence=1)
    rahu2       = make_card(32548318, location=0x02, atk=1600, def_=1000, level=4, attribute=0x04, sequence=2)
    albaz       = make_card(68468459, location=0x02, atk=1500, def_=2000, level=4, attribute=0x20, sequence=3)
    ash         = make_card(14558127, location=0x02, atk=0,    def_=1800, level=3, attribute=0x04, sequence=4)

    dummy = make_card(0)  # code=0 → renders as unnamed (used for phase-change actions)
    actions = MsgSelectIdleCmd(actions=[
        IdleAction(card=faimena,  msg=0),  # Normal Summon Dracotail Faimena
        IdleAction(card=rahu1,    msg=4),  # Set Rahu Dracotail
        IdleAction(card=albaz,    msg=0),  # Normal Summon Fallen of Albaz
        IdleAction(card=albaz,    msg=5),  # Activate Fallen of Albaz
        IdleAction(card=rahu2,    msg=5),  # Activate Rahu Dracotail
        IdleAction(card=dummy,    msg=6),  # Enter Battle Phase
        IdleAction(card=dummy,    msg=8),  # End Turn
    ])

    inp = Input(**{
        "global": make_global(turn=1, phase=0x04, is_first=1, is_my_turn=1),
        "cards": [faimena, rahu1, rahu2, albaz, ash],
        "action_msg": actions,
    })
    return "Branded Dracotail — Turn 1 Main Phase (going first)", inp


def scenario_vs_battle_phase() -> tuple[str, Input]:
    """
    Turn 3, Battle Phase.
    My field: Vanquish Soul Caesar Valius (ATK 2800) + Dr. Mad Love on field.
    Opponent: 1 faceup ATK monster (code=0, unknown 2000 ATK).
    Options: attack with Caesar, activate Caesar effect, go to Main 2, end turn.
    """
    caesar = make_card(91073013, location=0x04, atk=2800, def_=2500, level=8,
                       attribute=0x20, sequence=0)  # Monster Zone
    dr_mad = make_card(29280200, location=0x04, atk=0, def_=2000, level=4,
                       attribute=0x20, sequence=1)

    opp_unknown = make_card(0, controller=1, location=0x04, atk=2000, def_=0,
                            level=4, attribute=0x01, sequence=0)

    # Hand: Dr. Mad Love effect card still in hand
    hand_card = make_card(80181649, location=0x02, atk=0, def_=0, level=4,
                          attribute=0x20, sequence=0)

    actions = MsgSelectBattleCmd(actions=[
        BattleAction(card=caesar, msg=0),   # Attack with Caesar Valius
        BattleAction(card=caesar, msg=1),   # Activate Caesar effect
        BattleAction(card=None,   msg=2),   # Go to Main Phase 2
        BattleAction(card=None,   msg=3),   # End turn
    ])

    inp = Input(**{
        "global": make_global(my_lp=7200, opp_lp=5600, turn=3,
                              phase=0x10, is_first=1, is_my_turn=1),
        "cards": [caesar, dr_mad, opp_unknown, hand_card],
        "action_msg": actions,
    })
    return "K9 Vanquishsoul — Turn 3 Battle Phase", inp


def scenario_chain_opportunity() -> tuple[str, Input]:
    """
    Opponent's turn. Opponent activates Branded Fusion.
    We can chain Ash Blossom or Nibiru (if conditions met), or pass.
    """
    ash    = make_card(14558127, location=0x02, atk=0, def_=1800, level=3,
                       attribute=0x04, sequence=0)
    nibiru = make_card(27204311, location=0x02, atk=1000, def_=1000, level=11,
                       attribute=0x10, sequence=1)

    opp_field = make_card(0, controller=1, location=0x08, atk=0, def_=0,
                          level=0, attribute=0x01, sequence=0)  # face-down s/t

    # Opponent just activated a spell — we get a chain window
    # In ygoinf, this comes as MsgSelectChain with forced=0
    actions = MsgSelectChain(
        actions=[
            ChainAction(card=ash,    effect=0),
            ChainAction(card=nibiru, effect=0),
        ],
        forced=0,
    )

    inp = Input(**{
        "global": make_global(my_lp=8000, opp_lp=8000, turn=2,
                              phase=0x04, is_first=0, is_my_turn=0),
        "cards": [ash, nibiru, opp_field],
        "action_msg": actions,
    })
    return "Opponent's Turn — Chain Window (Ash Blossom or Nibiru or Pass)", inp


def scenario_card_selection() -> tuple[str, Input]:
    """
    Activate Branded Fusion — need to select Fusion Materials from field/hand.
    Select 1 card from: Fallen of Albaz (hand), Blazing Cartesia (hand),
    Dracotail Faimena (monster zone).
    """
    albaz    = make_card(68468459, location=0x02, atk=1500, def_=2000, level=4,
                         attribute=0x20, sequence=0)
    cartesia = make_card(95515789, location=0x02, atk=1500, def_=2100, level=4,
                         attribute=0x10, sequence=1)
    faimena  = make_card(1498449,  location=0x04, atk=1500, def_=2000, level=4,
                         attribute=0x20, sequence=0)

    actions = MsgSelectCard(
        actions=[
            CardAction(card=albaz),
            CardAction(card=cartesia),
            CardAction(card=faimena),
        ],
        **{"min": 1, "max": 1},
    )

    inp = Input(**{
        "global": make_global(turn=1, phase=0x04, is_first=1, is_my_turn=1),
        "cards": [albaz, cartesia, faimena],
        "action_msg": actions,
    })
    return "Branded Fusion activated — Select 1 Fusion Material", inp


def scenario_yesno_trigger() -> tuple[str, Input]:
    """
    Our Ash Blossom can respond to opponent's search effect — yes/no prompt.
    """
    ash = make_card(14558127, location=0x02, atk=0, def_=1800, level=3,
                    attribute=0x04, sequence=0)

    actions = MsgSelectYesNo(card=ash, effect=0)

    inp = Input(**{
        "global": make_global(my_lp=6000, opp_lp=8000, turn=4,
                              phase=0x04, is_first=0, is_my_turn=0),
        "cards": [ash],
        "action_msg": actions,
    })
    return "Opponent searches — Activate Ash Blossom & Joyous Spring? (Yes/No)", inp


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

    scenarios = [
        scenario_branded_opening(),
        scenario_vs_battle_phase(),
        scenario_chain_opportunity(),
        scenario_card_selection(),
        scenario_yesno_trigger(),
    ]

    for title, inp in scenarios:
        run_scenario(agent, title, inp)

    print(f"\n{DIVIDER}")
    print("  Demo complete.")
    print(DIVIDER)


if __name__ == "__main__":
    main()
