"""
Parse EDOPro/YGOPro `lflist.conf` banlist files.

Format (one banlist per file, or many separated by `!<name>` headers):

    !2024.4 Master Duel
    # comment
    $whitelist
    12345678 0 -- Forbidden
    23456789 1 -- Limited
    34567890 2 -- Semi-Limited

A card not listed is unrestricted (3 copies). Lines starting with `#` are
comments. Lines starting with `$` are mode directives we ignore.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LFList:
    name: str
    limits: dict[int, int] = field(default_factory=dict)

    def max_copies(self, code: int) -> int:
        return self.limits.get(code, 3)


def parse_lflist(path: Path, banlist_name: str | None = None) -> LFList:
    """Load one banlist from a .conf file.

    If the file contains multiple `!<name>` sections, pass `banlist_name` to
    pick one; otherwise the first section (or all entries if no header) is used.
    """
    current: str | None = None
    sections: dict[str, dict[int, int]] = {}
    default_bucket: dict[int, int] = {}

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("!"):
                current = line[1:].strip()
                sections.setdefault(current, {})
                continue
            if line.startswith("$"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                code = int(parts[0])
                count = int(parts[1])
            except ValueError:
                continue
            bucket = sections[current] if current else default_bucket
            bucket[code] = count

    if banlist_name is not None:
        if banlist_name not in sections:
            raise KeyError(
                f"Banlist '{banlist_name}' not found in {path}. "
                f"Available: {sorted(sections)}"
            )
        return LFList(name=banlist_name, limits=sections[banlist_name])

    if sections:
        first = next(iter(sections))
        return LFList(name=first, limits=sections[first])
    return LFList(name=path.stem, limits=default_bucket)
