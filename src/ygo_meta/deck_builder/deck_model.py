"""Deck dataclass and related types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Deck:
    archetype: str
    variant_id: str
    main: list[int]   # card codes, 40-60 entries
    extra: list[int]  # 0-15 entries
    side: list[int]   # 0-15 entries
    staple_combo: dict[str, int] = field(default_factory=dict)
    # key = card name, value = number of copies from staples

    def total_main(self) -> int:
        return len(self.main)

    def total_extra(self) -> int:
        return len(self.extra)
