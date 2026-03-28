"""
One-time script: parse vendor/ygo-agent/scripts/code_list.txt and write
data/card_names.json as {code_int: card_name}.

Re-run after `git submodule update` to pick up newly added cards.

code_list.txt format (one entry per line):
    <card_code> <card_name>
or just:
    <card_code>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
CODE_LIST = ROOT / "vendor" / "ygo-agent" / "scripts" / "code_list.txt"
OUT_PATH = ROOT / "data" / "card_names.json"


def main() -> None:
    if not CODE_LIST.exists():
        print(f"ERROR: code_list.txt not found at {CODE_LIST}", file=sys.stderr)
        print("Run: git submodule update --init --recursive", file=sys.stderr)
        sys.exit(1)

    db: dict[str, str] = {}
    with open(CODE_LIST, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            code = parts[0]
            name = parts[1].strip() if len(parts) > 1 else code
            db[code] = name

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

    print(f"Written {len(db)} card entries to {OUT_PATH}")


if __name__ == "__main__":
    main()
