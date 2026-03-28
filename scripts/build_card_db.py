"""
Build data/card_names.json from the YGOPRODeck API.

Stores ALL cards from the API (not just those in code_list.txt), so the deck
editor always has the full, up-to-date card pool including newly released cards.

Usage:
    python scripts/build_card_db.py

Re-run (or let GitHub Actions run it weekly) to pick up new card releases.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT_PATH = ROOT / "data" / "card_names.json"

YGOPRODECK_URL = "https://db.ygoprodeck.com/api/v7/cardinfo.php?misc=yes"


def fetch_all_cards() -> dict[int, str]:
    """Fetch the full card list from YGOPRODeck and return {code: name}."""
    print("Fetching card list from YGOPRODeck API...", flush=True)
    req = urllib.request.Request(
        YGOPRODECK_URL,
        headers={"User-Agent": "ygo-meta-ai/1.0 (github.com/rinai1122/ygo-meta-ai)"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = json.loads(resp.read().decode("utf-8"))

    db: dict[int, str] = {}
    for card in raw.get("data", []):
        db[card["id"]] = card["name"]
        # Also index alternate art IDs so all prints are searchable
        for img in card.get("card_images", []):
            alt_id = img.get("id")
            if alt_id and alt_id != card["id"]:
                db[alt_id] = card["name"]
    return db


def main() -> None:
    db = fetch_all_cards()
    print(f"Fetched {len(db)} cards from YGOPRODeck API")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in db.items()}, f, ensure_ascii=False, indent=2)

    print(f"Written {len(db)} entries to {OUT_PATH}")


if __name__ == "__main__":
    main()
