"""
Build data/card_names.json by fetching card names for all codes in
vendor/ygo-agent/scripts/code_list.txt from the YGOPRODeck API.

Re-run after `git submodule update` to pick up newly added card codes.

Usage:
    python scripts/build_card_db.py

The YGOPRODeck API is free and does not require authentication.
Rate limit: ~20 req/s. The full card list is fetched in one request.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
CODE_LIST = ROOT / "vendor" / "ygo-agent" / "scripts" / "code_list.txt"
OUT_PATH = ROOT / "data" / "card_names.json"

YGOPRODECK_URL = "https://db.ygoprodeck.com/api/v7/cardinfo.php?misc=yes"


def load_codes() -> list[int]:
    if not CODE_LIST.exists():
        print(f"ERROR: code_list.txt not found at {CODE_LIST}", file=sys.stderr)
        print("Run: git submodule update --init --recursive", file=sys.stderr)
        sys.exit(1)
    codes = []
    with open(CODE_LIST, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            codes.append(int(parts[0]))
    return codes


def fetch_all_cards() -> dict[int, str]:
    """Fetch the full card list from YGOPRODeck API and return {code: name}."""
    print(f"Fetching card list from YGOPRODeck API...", flush=True)
    req = urllib.request.Request(
        YGOPRODECK_URL,
        headers={"User-Agent": "ygo-meta-ai/1.0"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = json.loads(resp.read().decode("utf-8"))

    db: dict[int, str] = {}
    for card in raw.get("data", []):
        db[card["id"]] = card["name"]
        # Also index alternate art IDs from "card_images"
        for img in card.get("card_images", []):
            alt_id = img.get("id")
            if alt_id and alt_id != card["id"]:
                db[alt_id] = card["name"]
    return db


def main() -> None:
    codes = load_codes()
    print(f"Loaded {len(codes)} card codes from code_list.txt")

    api_db = fetch_all_cards()
    print(f"Fetched {len(api_db)} cards from YGOPRODeck API")

    # Build output: only codes present in code_list.txt
    out: dict[str, str] = {}
    missing = 0
    for code in codes:
        name = api_db.get(code)
        if name:
            out[str(code)] = name
        else:
            out[str(code)] = f"#{code}"  # fallback: show the raw code
            missing += 1

    if missing:
        print(f"Warning: {missing} codes had no match in YGOPRODeck (stored as #code)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Written {len(out)} entries to {OUT_PATH}")


if __name__ == "__main__":
    main()
