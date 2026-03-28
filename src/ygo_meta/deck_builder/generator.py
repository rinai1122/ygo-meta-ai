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


def _load_staple_pool(staples_dir: Path) -> list[dict]:
    """Load all staple entries from all YAML files in staples_dir."""
    pool: list[dict] = []
    for yaml_file in sorted(staples_dir.glob("*.yaml")):
        with open(yaml_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not data:
            continue
        for key, entries in data.items():
            if isinstance(entries, list):
                pool.extend(entries)
    return pool


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
) -> list[Deck]:
    """
    Generate n_variants decks for one archetype by combining the engine YDK
    with different random staple combinations.

    Each deck has exactly `target_size` main deck cards (engine + staples).
    If the engine already meets or exceeds target_size, no staples are added.
    """
    rng = random.Random(seed)
    engine_deck = parse_ydk(engine_path)
    pool = _load_staple_pool(staples_dir)
    archetype = engine_deck.archetype
    available = max(0, target_size - len(engine_deck.main))

    decks: list[Deck] = []
    for i in range(n_variants):
        if available > 0 and pool:
            staple_codes, combo = _sample_staple_combo(pool, available, rng)
        else:
            staple_codes, combo = [], {}

        main = engine_deck.main + staple_codes
        # If still short (pool exhausted), leave as-is — validator will report it.

        variant_id = f"{archetype}_v{i:03d}"
        deck = Deck(
            archetype=archetype,
            variant_id=variant_id,
            main=main[:60],
            extra=engine_deck.extra[:15],
            side=[],
            staple_combo=combo,
        )
        errors = validate_deck(deck)
        if errors:
            # Log but still include — caller can filter
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
