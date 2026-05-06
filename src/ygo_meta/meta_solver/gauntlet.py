"""
Gauntlet runner — round-robin payoff matrix over a fixed deck list.

Reuses `simulation/payoff_matrix.py` and `simulation/battle_runner.py`.
The gauntlet differs from the deprecated evolutionary loop in
`simulation/evolution.py`: there is no mutation between generations, no
Nash-driven deck pruning, and no staple-combo search. The deck set is
fixed by the user (5–7 meta + 3–5 theorycraft) and we just measure pairwise
win rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np

from ygo_meta.deck_builder.deck_model import Deck
from ygo_meta.deck_builder.lflist import parse_lflist
from ygo_meta.deck_builder.validator import validate_deck
from ygo_meta.deck_builder.ydk_parser import parse_ydk
from ygo_meta.simulation.battle_runner import BattleRunner
from ygo_meta.simulation.payoff_matrix import PayoffMatrix, build_matrix

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_LFLIST = _PROJECT_ROOT / "data" / "lflist" / "master_duel.lflist.conf"
_CACHE_SCHEMA = "seat-balanced-payoff-v2"


@dataclass
class GauntletResult:
    deck_ids: list[str]
    payoff: np.ndarray  # (N, N), payoff[i,j] = P(i beats j)


def load_decks(decks_dir: Path) -> list[Deck]:
    """Load every `.ydk` directly under *decks_dir* as a Deck."""
    decks: list[Deck] = []
    for ydk in sorted(Path(decks_dir).glob("*.ydk")):
        decks.append(parse_ydk(ydk))
    if not decks:
        raise FileNotFoundError(f"no .ydk files under {decks_dir}")
    return decks


def _deck_fingerprints(
    decks: list[Deck],
    *,
    seat_balanced: bool,
    num_episodes: int,
) -> dict[str, str]:
    out: dict[str, str] = {
        "__schema__": _CACHE_SCHEMA,
        "__seat_balanced__": str(seat_balanced),
        "__num_episodes__": str(num_episodes),
    }
    for deck in decks:
        payload = {
            "main": deck.main,
            "extra": deck.extra,
            "side": deck.side,
        }
        raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        out[deck.variant_id] = hashlib.sha256(raw).hexdigest()
    return out


def validate_gauntlet_decks(
    decks: list[Deck],
    *,
    lflist_path: Path | None = _DEFAULT_LFLIST,
) -> None:
    """Raise ValueError if any gauntlet deck is invalid for this run."""
    lflist = parse_lflist(lflist_path) if lflist_path and lflist_path.exists() else None
    errors: list[str] = []
    seen: set[str] = set()
    for deck in decks:
        if deck.variant_id in seen:
            errors.append(f"{deck.variant_id}: duplicate deck id")
        seen.add(deck.variant_id)
        for err in validate_deck(deck, lflist=lflist):
            errors.append(f"{deck.variant_id}: {err}")
    if errors:
        raise ValueError("invalid gauntlet deck(s):\n" + "\n".join(errors))


def run_gauntlet(
    decks: list[Deck],
    *,
    runner: BattleRunner | None = None,
    num_episodes: int = 500,
    seed: int = 0,
    cache_path: Path | None = None,
    cache_ids_path: Path | None = None,
    max_workers: int | None = None,
    seat_balanced: bool = True,
    validate_decks: bool = True,
    lflist_path: Path | None = _DEFAULT_LFLIST,
) -> GauntletResult:
    """Run round-robin BO1 matchups; return the empirical payoff matrix.

    If *cache_path* / *cache_ids_path* exist, missing pairs are computed
    incrementally (resumable across crashes).
    """
    runner = runner or BattleRunner()
    if validate_decks:
        validate_gauntlet_decks(decks, lflist_path=lflist_path)

    existing: PayoffMatrix | None = None
    fp_path = cache_ids_path.with_suffix(".fingerprints.json") if cache_ids_path else None
    if cache_path and cache_ids_path and cache_path.exists() and cache_ids_path.exists():
        current_fps = _deck_fingerprints(
            decks,
            seat_balanced=seat_balanced,
            num_episodes=num_episodes,
        )
        cached_fps = None
        if fp_path and fp_path.exists():
            try:
                cached_fps = json.loads(fp_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                cached_fps = None
        if cached_fps == current_fps:
            try:
                candidate = PayoffMatrix.load(cache_path, cache_ids_path)
                if candidate.matrix.shape == (len(decks), len(decks)):
                    existing = candidate
                else:
                    print("  gauntlet cache shape changed; ignoring stale cache", flush=True)
            except (OSError, ValueError, json.JSONDecodeError):
                print("  gauntlet cache is unreadable; ignoring stale cache", flush=True)
        else:
            print("  deck contents changed; ignoring stale gauntlet cache", flush=True)

    pm = build_matrix(
        decks,
        runner,
        num_episodes=num_episodes,
        seed=seed,
        existing=existing,
        max_workers=max_workers,
        seat_balanced=seat_balanced,
    )

    if cache_path and cache_ids_path:
        pm.save(cache_path, cache_ids_path)
        if fp_path:
            fp_path.write_text(
                json.dumps(
                    _deck_fingerprints(
                        decks,
                        seat_balanced=seat_balanced,
                        num_episodes=num_episodes,
                    ),
                    indent=2,
                ),
                encoding="utf-8",
            )

    return GauntletResult(deck_ids=list(pm.deck_ids), payoff=pm.matrix.copy())
