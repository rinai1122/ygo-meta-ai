"""
Parse and write YGOPro YDK deck files.

YDK format:
    #created by <author>
    #main
    <card_code>
    ...
    #extra
    <card_code>
    ...
    !side
    <card_code>
    ...
"""

from __future__ import annotations

from pathlib import Path

from ygo_meta.deck_builder.deck_model import Deck


def parse_ydk(path: Path, archetype: str = "", variant_id: str = "") -> Deck:
    main: list[int] = []
    extra: list[int] = []
    side: list[int] = []
    section = "main"

    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#created"):
                continue
            if line == "#main":
                section = "main"
            elif line == "#extra":
                section = "extra"
            elif line == "!side":
                section = "side"
            else:
                try:
                    code = int(line)
                except ValueError:
                    continue
                if section == "main":
                    main.append(code)
                elif section == "extra":
                    extra.append(code)
                elif section == "side":
                    side.append(code)

    return Deck(
        archetype=archetype or path.parent.name,
        variant_id=variant_id or path.stem,
        main=main,
        extra=extra,
        side=side,
    )


def write_ydk(deck: Deck, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"#created by ygo-meta-ai ({deck.variant_id})\n")
        f.write("#main\n")
        for code in deck.main:
            f.write(f"{code}\n")
        f.write("#extra\n")
        for code in deck.extra:
            f.write(f"{code}\n")
        f.write("!side\n")
        for code in deck.side:
            f.write(f"{code}\n")
