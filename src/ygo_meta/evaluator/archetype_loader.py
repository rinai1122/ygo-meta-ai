"""
Archetype baseline loader.

A complete baseline deck = ``engine.ydk`` (the archetype's actual engine cards)
+ ``baseline.ydk`` (the staple/flex configuration under test). Both files live
in ``data/engines/<archetype>/``. We concatenate the ``main``, ``extra``, and
``side`` sections of both, in that order, so the LAST main-deck slot — which
``ygo-eval-tech-delta`` treats as the flex slot under test — comes from the
baseline file (the staple side, where flex slots actually live).
"""

from __future__ import annotations

from pathlib import Path

from ygo_meta.deck_builder.deck_model import Deck
from ygo_meta.deck_builder.ydk_parser import parse_ydk


def resolve_archetype(name: str, engines_dir: Path) -> Path:
    """Return the baseline.ydk path for an archetype name OR a direct YDK path."""
    p = Path(name)
    if p.suffix == ".ydk" and p.exists():
        return p
    candidate = engines_dir / name / "baseline.ydk"
    if not candidate.exists():
        raise FileNotFoundError(
            f"baseline for '{name}' not found at {candidate}"
        )
    return candidate


def load_archetype_deck(name: str, engines_dir: Path) -> Deck:
    """Load engine.ydk + baseline.ydk for an archetype and merge them.

    If ``name`` is a direct YDK path, that file is loaded as-is (no merge).
    """
    p = Path(name)
    if p.suffix == ".ydk" and p.exists():
        return parse_ydk(p)

    arch_dir = engines_dir / name
    engine_path = arch_dir / "engine.ydk"
    baseline_path = arch_dir / "baseline.ydk"
    if not engine_path.exists():
        raise FileNotFoundError(f"missing {engine_path}")
    if not baseline_path.exists():
        raise FileNotFoundError(f"missing {baseline_path}")

    engine = parse_ydk(engine_path, archetype=name, variant_id=name)
    baseline = parse_ydk(baseline_path, archetype=name, variant_id=name)

    return Deck(
        archetype=name,
        variant_id=name,
        main=list(engine.main) + list(baseline.main),
        extra=list(engine.extra) + list(baseline.extra),
        side=list(engine.side) + list(baseline.side),
    )
