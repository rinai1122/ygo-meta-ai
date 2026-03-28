"""
Generate M×N decks by combining main engine YDK files with staple combinations.

Usage:
    decks = generate_decks(
        engine_path=Path("data/engines/snake_eye/engine.ydk"),
        staples_dir=Path("data/staples/"),
        n_variants=4,
        target_size=40,
        seed=42,
    )
"""

from __future__ import annotations

import random
from pathlib import Path

import yaml

from ygo_meta.deck_builder.deck_model import Deck
from ygo_meta.deck_builder.validator import validate_deck
from ygo_meta.deck_builder.ydk_parser import parse_ydk


def _load_staple_pool(staples_dir: Path) -> tuple[list[dict], list[dict]]:
    """
    Load staple entries from all YAML files in staples_dir.

    Returns:
        main_pool:  entries destined for the main deck (all keys except 'extradeck')
        extra_pool: entries destined for the extra deck (key == 'extradeck')
    """
    main_pool: list[dict] = []
    extra_pool: list[dict] = []
    for yaml_file in sorted(staples_dir.glob("*.yaml")):
        with open(yaml_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not data:
            continue
        for key, entries in data.items():
            if isinstance(entries, list):
                if key == "extradeck":
                    extra_pool.extend(entries)
                else:
                    main_pool.extend(entries)
    return main_pool, extra_pool


def _sample_staple_combo(
    pool: list[dict],
    available_slots: int,
    rng: random.Random,
) -> tuple[list[int], dict[str, int]]:
    """
    Sample a combination of staple cards to fill `available_slots` main deck slots.

    Returns:
        codes: flat list of card codes (with repetition for 2-3 copies)
        combo_dict: {card_name: n_copies} for record-keeping
    """
    rng.shuffle(pool)
    codes: list[int] = []
    combo: dict[str, int] = {}
    remaining = available_slots

    entries_left = len(pool)
    for entry in pool:
        entries_left -= 1
        if remaining <= 0:
            break
        max_copies = min(entry.get("copies_max", 3), remaining)
        if max_copies <= 0:
            continue
        # If this is the last entry or we must fill remaining slots, use max_copies.
        # Otherwise pick a random count so each variant has a different composition.
        if entries_left == 0 or remaining <= max_copies:
            copies = max_copies
        else:
            copies = rng.randint(1, max_copies)
        code = entry["code"]
        name = entry.get("name", str(code))
        codes.extend([code] * copies)
        combo[name] = copies
        remaining -= copies

    return codes, combo


def generate_decks(
    engine_path: Path,
    staples_dir: Path,
    n_variants: int = 4,
    target_size: int = 40,
    seed: int = 0,
    main_only: bool = False,
) -> list[Deck]:
    """
    Generate n_variants decks for one archetype by combining the engine YDK
    with different random staple combinations.

    Each deck has exactly `target_size` main deck cards (engine + staples).
    If the engine already meets or exceeds target_size, no staples are added.

    main_only: if True, only main deck staples (handtraps/backrow) are added;
               the engine's extra deck is kept as-is with no additions.
    """
    rng = random.Random(seed)
    engine_deck = parse_ydk(engine_path)
    main_pool, extra_pool = _load_staple_pool(staples_dir)
    archetype = engine_deck.archetype
    available_main = max(0, target_size - len(engine_deck.main))

    decks: list[Deck] = []
    for i in range(n_variants):
        if available_main > 0 and main_pool:
            staple_codes, combo = _sample_staple_combo(main_pool, available_main, rng)
        else:
            staple_codes, combo = [], {}

        main = engine_deck.main + staple_codes

        # Extra deck: keep engine's extra deck as-is when main_only.
        if not main_only:
            available_extra = max(0, 15 - len(engine_deck.extra))
            extra_staples = [e["code"] for e in extra_pool[:available_extra]] if available_extra and extra_pool else []
        else:
            extra_staples = []
        extra = (engine_deck.extra + extra_staples)[:15]

        variant_id = f"{archetype}_v{i:03d}"
        deck = Deck(
            archetype=archetype,
            variant_id=variant_id,
            main=main[:60],
            extra=extra,
            side=[],
            staple_combo=combo,
        )
        errors = validate_deck(deck)
        if errors:
            import warnings
            warnings.warn(f"{variant_id}: {errors}", stacklevel=2)
        decks.append(deck)

    return decks


def generate_all_decks(
    engines_dir: Path,
    staples_dir: Path,
    archetypes: list[str] | None = None,
    n_variants: int = 4,
    seed: int = 0,
) -> list[Deck]:
    """
    Generate decks for multiple archetypes.

    archetypes: list of subdirectory names under engines_dir.
                If None, all subdirectories with an engine.ydk are used.
    """
    if archetypes is None:
        archetypes = [
            d.name for d in sorted(engines_dir.iterdir())
            if d.is_dir() and (d / "engine.ydk").exists()
        ]

    all_decks: list[Deck] = []
    for i, arch in enumerate(archetypes):
        engine_path = engines_dir / arch / "engine.ydk"
        if not engine_path.exists():
            raise FileNotFoundError(f"No engine.ydk for archetype '{arch}' at {engine_path}")
        decks = generate_decks(
            engine_path=engine_path,
            staples_dir=staples_dir,
            n_variants=n_variants,
            seed=seed + i * 1000,
        )
        all_decks.extend(decks)
    return all_decks
