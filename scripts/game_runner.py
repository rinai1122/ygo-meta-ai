"""
Game loop subprocess for LLM agent testing.

Runs under the YGOAGENT_VENV (ygoenv + ygoinf available).
Communicates with the parent process via JSON lines on stdin/stdout.

Parent sends:  {"action": K}
We send:       {"type": "step", "to_play": P, "num_options": N, "prompt": "..."}
               {"type": "done", "ep_idx": E, "reward": R, "turns": T}

Usage (called by play.py, not directly):
  python game_runner.py --deck DIR_OR_YDK --num-episodes N --seed S \
                        --code-list PATH --card-names PATH \
                        [--deck1 NAME] [--deck2 NAME] [--env-id YGOPro-v1]
"""

from __future__ import annotations

import argparse
import json
import os
import random as _random
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

# ---------------------------------------------------------------------------
# Reverse lookup tables derived from ygoinf — always in sync with the engine.
# ---------------------------------------------------------------------------

from ygoinf.features import (
    phase_to_id, location_to_id, controller_to_id, position_to_id,
    attribute_to_id, type_to_id, action_act_to_id, action_phase_to_id,
    msg_to_id, place_to_id,
)

_ID_TO_PHASE        = {v: k.value for k, v in phase_to_id.items()}
_ID_TO_LOCATION     = {v: k.value for k, v in location_to_id.items()}
_ID_TO_CONTROLLER   = {v: k.value for k, v in controller_to_id.items()}
_ID_TO_POSITION     = {v: k.value for k, v in position_to_id.items()}
_ID_TO_ATTRIBUTE    = {v: k.value for k, v in attribute_to_id.items()}
_TYPE_ID_TO_NAME    = {v: k.value for k, v in type_to_id.items()}
_ID_TO_ACT          = {v: k.value for k, v in action_act_to_id.items()}
_ID_TO_ACTION_PHASE = {v: k.value for k, v in action_phase_to_id.items()}
_ID_TO_MSG          = {v: k.value for k, v in msg_to_id.items()}
_ID_TO_PLACE        = {v: k.value for k, v in place_to_id.items()}


# ---------------------------------------------------------------------------
# Obs decoding helpers
# ---------------------------------------------------------------------------

def _decode_card(row: np.ndarray, id_to_code: dict, card_names: dict) -> dict | None:
    """Decode one row of cards_ array (41 uint8) into a card dict. Returns None if empty.

    NOTE: ygoenv's C++ does NOT encode card codes in bytes 0-1 of the cards array
    (they are always 0).  Card codes are only available in action rows (bytes 1-2).
    We use location_id as the "slot is occupied" check instead.
    """
    location_id = int(row[2])
    if location_id == 0:
        return None
    types = [_TYPE_ID_TO_NAME[i] for i in range(25) if row[16 + i]]
    return {
        "name": "",   # filled in by format_obs_prompt via action row bytes 1-2
        "location": _ID_TO_LOCATION.get(location_id, "unknown"),
        "sequence": int(row[3]),
        "controller": _ID_TO_CONTROLLER.get(int(row[4]), "me"),
        "position": _ID_TO_POSITION.get(int(row[5]), "none"),
        "is_overlay": bool(row[6]),
        "attribute": _ID_TO_ATTRIBUTE.get(int(row[7]), "none"),
        "level": int(row[9]),
        "atk": int(row[12]) * 256 + int(row[13]),
        "def_": int(row[14]) * 256 + int(row[15]),
        "types": types,
    }


def _card_display(card: dict) -> str:
    """One-line card description for the board block."""
    name = card["name"] or "unknown"
    types = card["types"]
    if "spell" in types:
        return f"[{name}] | Spell"
    if "trap" in types:
        return f"[{name}] | Trap"
    attr = card["attribute"].upper()
    # Hand cards have no meaningful position in YGO — skip it.
    if card.get("location") == "hand":
        return f"[{name}] ATK {card['atk']}/DEF {card['def_']} | Lv{card['level']} | {attr}"
    pos = card["position"]
    return f"[{name}] ATK {card['atk']}/DEF {card['def_']} | {pos} | Lv{card['level']} | {attr}"


def _describe_action(row: np.ndarray, cards: list, id_to_code: dict, card_names: dict) -> str:
    """Produce a human-readable description of one action from its encoded row (12 uint8).

    Action row layout (from features.py encode_action):
      row[0]   = card_index (1-based into cards list)
      row[1:3] = int_transform(card_id)  ← code lives here, NOT in cards array
      row[3]   = msg_id
      row[4]   = act_id
      row[5]   = finish flag
      row[6]   = effect
      row[7]   = phase_id
      row[8]   = position_id
      row[9]   = number
      row[10]  = place_id
      row[11]  = attribute_id
    """
    card_index = int(row[0])   # 1-based into cards list; 0 = N/A
    card_id = int(row[1]) * 256 + int(row[2])   # code_id encoded in action row
    msg = _ID_TO_MSG.get(int(row[3]), "unknown")
    act = _ID_TO_ACT.get(int(row[4]), "none")
    finish = bool(row[5])
    phase = _ID_TO_ACTION_PHASE.get(int(row[7]), "none")
    pos_id = int(row[8])
    number = int(row[9])
    place_id = int(row[10])
    attr_id = int(row[11])

    # Resolve card name: prefer the pre-filled name from summon/set actions
    # (set by format_obs_prompt's pre-fill loop using the card's own code),
    # fall back to direct card_id decode for actions without a prior name.
    card = cards[card_index - 1] if card_index > 0 and card_index - 1 < len(cards) else None
    if card_id:
        code = id_to_code.get(card_id, 0)
        card_name = card_names.get(str(code), f"Card#{code}") if code else ""
    else:
        card_name = ""
    # Activate actions may encode a linked-effect card ID instead of the
    # activating card's own ID (e.g. Branded Fusion encodes Albaz). Fall
    # back to the pre-filled name, which was set from a summon/set action
    # for the same card slot and therefore carries the correct card name.
    if not card_name and card and card.get("name"):
        card_name = card["name"]
    card_atk = card["atk"] if card else 0

    # Phase transitions take priority
    if phase == "battle":
        return "Proceed to Battle Phase"
    if phase == "main2":
        return "Proceed to Main Phase 2"
    if phase == "end":
        return "End Turn"

    if act == "summon_faceup_attack":
        return f"Normal Summon [{card_name}] in Attack Position (ATK {card_atk})"
    if act == "summon_facedown_defense":
        return f"Set [{card_name}] face-down (Defense)"
    if act == "special_summon":
        return f"Special Summon [{card_name}]"
    if act == "set":
        return f"Set [{card_name}] face-down (spell/trap)"
    if act == "reposition":
        return f"Reposition [{card_name}]"
    if act == "attack":
        return f"Attack with [{card_name}] (ATK {card_atk})"
    if act == "direct_attack":
        return f"Direct attack with [{card_name}] (ATK {card_atk})"
    if act == "activate":
        return f"Activate [{card_name}]" if card_name else "Activate effect"
    if act == "cancel":
        return "Do not activate / Pass"
    if finish:
        return "Confirm selection"

    # Fallback by message type
    if msg in ("select_card", "select_tribute", "select_sum", "select_unselect_card"):
        loc = card["location"] if card else "unknown"
        return f"Select [{card_name}] from {loc}" if card_name else "Select card"
    if msg == "select_position":
        return f"Set position: {_ID_TO_POSITION.get(pos_id, 'unknown')}"
    if msg in ("select_place", "select_disfield"):
        return f"Choose zone: {_ID_TO_PLACE.get(place_id, f'zone {place_id}')}"
    if msg in ("select_yesno", "select_effectyn"):
        return "Yes" if act != "cancel" else "No"
    if msg == "announce_attrib":
        return f"Declare attribute: {_ID_TO_ATTRIBUTE.get(attr_id, 'unknown').upper()}"
    if msg == "announce_number":
        return f"Declare number: {number}"
    if msg == "select_option":
        return "Choose option"
    return f"Action ({msg}/{act})"


def format_obs_prompt(
    global_arr: np.ndarray,
    cards_arr: np.ndarray,
    actions_arr: np.ndarray,
    num_options: int,
    id_to_code: dict,
    card_names: dict,
) -> str:
    """Build a text prompt from raw numpy obs, matching formatter.py's output structure."""
    # Decode global state
    my_lp = int(global_arr[0]) * 256 + int(global_arr[1])
    op_lp = int(global_arr[2]) * 256 + int(global_arr[3])
    turn = int(global_arr[4])
    phase = _ID_TO_PHASE.get(int(global_arr[5]), "unknown").upper()
    is_first = bool(global_arr[6])
    is_my_turn = bool(global_arr[7])

    game_state = (
        "=== GAME STATE ===\n"
        f"Turn: {turn} | Phase: {phase} | "
        f"You go first: {'Yes' if is_first else 'No'} | "
        f"Your turn: {'Yes' if is_my_turn else 'No'}\n"
        f"Your LP: {my_lp} | Opponent LP: {op_lp}"
    )

    # Decode cards and group by controller/location
    cards = [_decode_card(cards_arr[i], id_to_code, card_names) for i in range(cards_arr.shape[0])]

    # ygoenv omits card codes from the cards array; action rows carry them.
    # Pre-fill names here so the board block can show card names to the LLM.
    # Process all actions but never overwrite an already-set name: summon/set
    # actions (which appear first in the C++ action list and carry the card's
    # own sequential ID) take priority over activate actions (which may carry
    # a linked-effect card ID such as Albaz for a Branded Fusion effect).
    for i in range(num_options):
        row = actions_arr[i]
        ci = int(row[0])   # 1-based index into cards list
        cid = int(row[1]) * 256 + int(row[2])   # code_id from action row
        if ci > 0 and cid and ci - 1 < len(cards) and cards[ci - 1] is not None:
            code = id_to_code.get(cid, 0)
            name = card_names.get(str(code), "") if code else ""
            if name and not cards[ci - 1]["name"]:
                cards[ci - 1]["name"] = name

    my_by_loc: dict[str, list] = {}
    op_by_loc: dict[str, list] = {}
    for card in cards:
        if card is None:
            continue
        bucket = my_by_loc if card["controller"] == "me" else op_by_loc
        bucket.setdefault(card["location"], []).append(card)

    # Locations where we show count only (contents are hidden or not useful without names).
    _COUNT_ONLY = {"deck", "extra"}

    board_lines = ["=== BOARD ===", "YOUR FIELD:"]
    if my_by_loc:
        for loc in sorted(my_by_loc):
            loc_cards = my_by_loc[loc]
            if loc in _COUNT_ONLY:
                board_lines.append(f"  {loc}: {len(loc_cards)} card(s)")
            else:
                parts = ", ".join(_card_display(c) for c in loc_cards)
                board_lines.append(f"  {loc}: {parts}")
    else:
        board_lines.append("  (empty)")

    board_lines.append("OPPONENT FIELD:")
    if op_by_loc:
        for loc in sorted(op_by_loc):
            if loc in ("hand", "deck", "extra"):
                board_lines.append(f"  {loc}: {len(op_by_loc[loc])} card(s)")
            else:
                parts = ", ".join(_card_display(c) for c in op_by_loc[loc])
                board_lines.append(f"  {loc}: {parts}")
    else:
        board_lines.append("  (empty)")

    board = "\n".join(board_lines)

    # Decode actions
    option_lines = [
        f"[{i}] {_describe_action(actions_arr[i], cards, id_to_code, card_names)}"
        for i in range(num_options)
    ]
    options_block = "\n".join(option_lines)

    return (
        f"{game_state}\n\n"
        f"{board}\n\n"
        f"=== YOUR OPTIONS ===\n{options_block}\n\n"
        "Which action do you choose? Reply with the action number only."
    )


# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="YGO game runner for LLM agent testing")
    parser.add_argument("--deck", required=True, help="Deck directory or single YDK file path")
    parser.add_argument("--deck1", default=None, help="Deck1 name (default: deck stem or 'random')")
    parser.add_argument("--deck2", default=None, help="Deck2 name (default: same as deck1)")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--code-list", required=True, help="Path to code_list.txt")
    parser.add_argument("--card-names", required=True, help="Path to card_names.json")
    parser.add_argument("--env-id", default="YGOPro-v1")
    parser.add_argument("--checkpoint", default=None, help="Path to RL agent .flax_model checkpoint")
    parser.add_argument("--llm-player", type=int, default=0, help="Which player index is LLM (0 or 1)")
    parser.add_argument("--num-embeddings", type=int, default=None, help="RL model embedding size (must match checkpoint)")
    parser.add_argument("--num-layers", type=int, default=2, help="Model num_layers (must match checkpoint)")
    parser.add_argument("--num-channels", type=int, default=128, help="Model num_channels (must match checkpoint)")
    parser.add_argument("--rnn-channels", type=int, default=512, help="Model rnn_channels (must match checkpoint)")
    parser.add_argument("--critic-width", type=int, default=128, help="Model critic_width (must match checkpoint)")
    parser.add_argument("--critic-depth", type=int, default=3, help="Model critic_depth (must match checkpoint)")
    args = parser.parse_args()

    # code_list.txt has lines like "2511 1" (code + flag) but init_code_list
    # expects one integer per line.  The C++ engine uses ALL entries (flag=0 and
    # flag=1) for sequential ID assignment, so we must do the same.
    import tempfile
    with open(args.code_list, "r") as _f:
        _codes_only = "\n".join(
            parts[0]
            for line in _f
            for parts in (line.split(),)
            if parts
        )
    _tmp_code_list = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    _tmp_code_list.write(_codes_only)
    _tmp_code_list.close()

    from ygoinf.features import init_code_list, code_to_id
    init_code_list(_tmp_code_list.name)
    id_to_code = {v: k for k, v in code_to_id.items()}

    # Load card names
    with open(args.card_names, "r", encoding="utf-8") as f:
        card_names = json.load(f)

    # Copy deck files to a temp dir with LF line endings (Windows YDK files use CRLF
    # which the C++ game engine can't parse, returning "found: 0 cards").
    import tempfile
    _tmp_deck_dir = tempfile.mkdtemp()
    deck_src = Path(args.deck)
    if deck_src.is_dir():
        for ydk in deck_src.glob("*.ydk"):
            (Path(_tmp_deck_dir) / ydk.name).write_bytes(ydk.read_bytes().replace(b"\r\n", b"\n"))
        _deck_arg = _tmp_deck_dir
    else:
        _lf_copy = Path(_tmp_deck_dir) / deck_src.name
        _lf_copy.write_bytes(deck_src.read_bytes().replace(b"\r\n", b"\n"))
        _deck_arg = str(_lf_copy)

    # Write _tokens.ydk so the C++ engine can preload token card data (e.g. Primal
    # Being Token 27204312 summoned by Nibiru).  Without this the engine crashes with
    # "[card_reader_callback] Card not found" when a token is summoned mid-game.
    _vendor_root = Path(__file__).resolve().parent.parent / "vendor" / "ygo-agent"
    _cdb = _vendor_root / "assets" / "locale" / "en" / "cards.cdb"
    if _cdb.exists():
        import sqlite3 as _sqlite3
        TYPE_TOKEN = 0x4000
        _con = _sqlite3.connect(str(_cdb))
        _token_ids = [r[0] for r in _con.execute(
            "SELECT id FROM datas WHERE (type & ?) != 0", (TYPE_TOKEN,)
        ).fetchall()]
        _con.close()
        _tokens_ydk = Path(_tmp_deck_dir) / "_tokens.ydk"
        with open(_tokens_ydk, "w") as _f:
            _f.write("#main\n")
            for _tid in _token_ids:
                _f.write(f"{_tid}\n")
            _f.write("#extra\n#side\n")

    # Initialize ygoenv via ygoai helper (registers deck files in C++ module).
    # The ygopro-core binary reads card Lua scripts as "./script/c{code}.lua"
    # relative to the process CWD — both at init time AND at duel creation time.
    # The scripts live at vendor/ygo-agent/scripts/script/ (symlink → ygopro-scripts).
    # Set CWD to vendor scripts dir and keep it there for the lifetime of this process.
    import os as _os
    _vendor_scripts = _vendor_root / "scripts"
    if _vendor_scripts.is_dir():
        _os.chdir(str(_vendor_scripts))
    from ygoai.utils import init_ygopro
    _preload = _cdb.exists()
    deck_name = init_ygopro(args.env_id, "english", _deck_arg, args.code_list, preload_tokens=_preload)
    deck1 = args.deck1 or deck_name
    deck2 = args.deck2 or deck1

    import ygoenv
    from ygoai.rl.env import RecordEpisodeStatistics

    _random.seed(args.seed)
    np.random.seed(args.seed)
    seed = _random.randint(0, int(1e8))

    envs = ygoenv.make(
        task_id=args.env_id,
        env_type="gymnasium",
        num_envs=1,
        num_threads=1,
        seed=seed,
        deck1=deck1,
        deck2=deck2,
        player=-1,
        max_options=24,
        n_history_actions=32,
        play_mode="self",
        async_reset=False,
        verbose=False,
    )
    envs.num_envs = 1

    # Load RL agent before wrapping (obs_space needed for model init)
    rl_agent = None
    rl_params = None
    rl_rstate = None
    get_rl_probs = None
    if args.checkpoint:
        import jax
        import jax.numpy as jnp
        import flax
        from functools import partial
        from ygoai.rl.jax.agent import RNNAgent, ModelArgs

        obs_space = envs.observation_space
        _sample_obs = jax.tree.map(lambda x: jnp.array([x]), obs_space.sample())
        rl_agent = RNNAgent(
            embedding_shape=args.num_embeddings,
            num_layers=args.num_layers,
            num_channels=args.num_channels,
            rnn_channels=args.rnn_channels,
            critic_width=args.critic_width,
            critic_depth=args.critic_depth,
        )
        _key = jax.random.PRNGKey(args.seed)
        rl_rstate = rl_agent.init_rnn_state(1)
        rl_params = jax.jit(rl_agent.init)(_key, _sample_obs, rl_rstate)
        with open(args.checkpoint, "rb") as _f:
            rl_params = flax.serialization.from_bytes(rl_params, _f.read())
        rl_params = jax.device_put(rl_params)

        @partial(jax.jit)
        def get_rl_probs(params, rstate, obs):
            next_rstate, logits = rl_agent.apply(params, obs, rstate)[:2]
            probs = jax.nn.softmax(logits, axis=-1)
            return next_rstate, probs

    envs = RecordEpisodeStatistics(envs)

    ep_idx = 0
    obs, infos = envs.reset()

    while ep_idx < args.num_episodes:
        to_play = int(infos["to_play"][0])
        num_options = max(1, int(infos["num_options"][0]))

        if rl_agent is not None and to_play != args.llm_player:
            # RL agent's turn — handled in-process, no parent communication needed
            obs_rl = {k: (v if k != 'mask_' else None) for k, v in obs.items()}
            rl_rstate, probs = get_rl_probs(rl_params, rl_rstate, obs_rl)
            action = int(np.array(probs).argmax(axis=1)[0])
            action = min(action, num_options - 1)
        else:
            # LLM (or random) player's turn — ask parent process
            prompt = format_obs_prompt(
                obs["global_"][0],
                obs["cards_"][0],
                obs["actions_"][0],
                num_options,
                id_to_code,
                card_names,
            )
            sys.stdout.write(json.dumps({
                "type": "step",
                "to_play": to_play,
                "num_options": num_options,
                "prompt": prompt,
            }) + "\n")
            sys.stdout.flush()

            line = sys.stdin.readline()
            if not line:
                break
            action = max(0, min(int(json.loads(line.strip())["action"]), num_options - 1))

        prev_to_play = to_play
        obs, rewards, dones, infos = envs.step(np.array([action]))

        if dones[0]:
            reward = float(rewards[0])
            # reward > 0: the player who just acted (prev_to_play) won
            winner = prev_to_play if reward > 0 else (1 - prev_to_play)
            sys.stdout.write(json.dumps({
                "type": "done",
                "ep_idx": ep_idx,
                "reward": reward,
                "winner": winner,
                "turns": int(infos["l"][0]),
            }) + "\n")
            sys.stdout.flush()
            ep_idx += 1
            # envpool auto-resets; obs/infos already contain the new episode's first state
            if rl_agent is not None:
                rl_rstate = rl_agent.init_rnn_state(1)  # reset RNN state for next episode

    envs.close()


if __name__ == "__main__":
    main()
