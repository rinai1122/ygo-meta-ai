"""
Build data/card_names.json and data/card_info.json from the YGOPRODeck API.

Stores ALL cards from the API (not just those in code_list.txt), so the deck
editor always has the full, up-to-date card pool including newly released cards.

card_names.json  — {code_str: name}
card_info.json   — {code_str: {
    "in_md":    bool,            # available in Master Duel
    "ban_ocg":  str | null,      # "Forbidden" / "Limited" / "Semi-Limited" / null
                                 # OCG banlist is used as the closest public approximation
                                 # of the MD banlist (MD is OCG-rules based)
    "is_extra": bool,            # true for Fusion/Synchro/Xyz/Link monsters
}}

Note on MD banlist: YGOPRODeck's API does not expose Master Duel ban status directly.
OCG ban data is used as the best available public approximation.
The deck editor labels these as "OCG (≈MD)".

Usage:
    python scripts/build_card_db.py

Re-run (or let GitHub Actions run it weekly) to pick up new card releases / ban changes.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

ROOT      = Path(__file__).parent.parent
OUT_NAMES = ROOT / "data" / "card_names.json"
OUT_INFO  = ROOT / "data" / "card_info.json"

YGOPRODECK_URL = "https://db.ygoprodeck.com/api/v7/cardinfo.php?misc=yes"
_HEADERS       = {"User-Agent": "ygo-meta-ai/1.0 (github.com/rinai1122/ygo-meta-ai)"}

# Card type substrings that indicate an Extra Deck card — checked case-insensitively
_EXTRA_KEYWORDS = ("fusion", "synchro", "xyz", "link")


def fetch_all_cards() -> tuple[dict[int, str], dict[str, dict]]:
    """
    Fetch full card list from YGOPRODeck and return:
      names: {code: name}
      info:  {code_str: {"in_md": bool, "ban_ocg": str|None, "is_extra": bool}}

    ban_ocg is derived from banlist_info.ban_ocg in the API response and is the
    best available public approximation of the Master Duel banlist.
    """
    print("Fetching card data from YGOPRODeck API...", flush=True)
    req = urllib.request.Request(YGOPRODECK_URL, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = json.loads(resp.read().decode("utf-8"))

    names: dict[int, str] = {}
    info:  dict[str, dict] = {}

    for card in raw.get("data", []):
        cid:  int = card["id"]
        name: str = card["name"]
        names[cid] = name

        # Master Duel availability
        misc  = card.get("misc_info", [{}])[0] if card.get("misc_info") else {}
        in_md = "Master Duel" in misc.get("formats", [])

        # OCG ban status (closest public approximation of MD banlist)
        ban_raw = card.get("banlist_info", {}).get("ban_ocg")
        ban_ocg = "Forbidden" if ban_raw == "Banned" else ban_raw

        # Extra Deck detection — case-insensitive to handle "XYZ Monster" etc.
        ctype    = card.get("type", "").lower()
        is_extra = any(kw in ctype for kw in _EXTRA_KEYWORDS)

        entry: dict = {
            "in_md":    in_md,
            "ban_ocg":  ban_ocg,
            "is_extra": is_extra,
        }
        info[str(cid)] = entry

        # Also index alternate art IDs so all prints are searchable
        for img in card.get("card_images", []):
            alt_id = img.get("id")
            if alt_id and alt_id != cid:
                names[alt_id] = name
                info[str(alt_id)] = entry

    return names, info


def main() -> None:
    names, info = fetch_all_cards()
    print(f"Fetched {len(names)} card entries from YGOPRODeck API")

    OUT_NAMES.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_NAMES, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in names.items()}, f, ensure_ascii=False, indent=2)
    print(f"Written {len(names)} entries to {OUT_NAMES}")

    with open(OUT_INFO, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    extra_count   = sum(1 for v in info.values() if v["is_extra"])
    md_count      = sum(1 for v in info.values() if v["in_md"])
    banned_count  = sum(1 for v in info.values() if v["ban_ocg"] == "Forbidden")
    limited_count = sum(1 for v in info.values() if v["ban_ocg"] == "Limited")
    semi_count    = sum(1 for v in info.values() if v["ban_ocg"] == "Semi-Limited")
    print(f"Written {len(info)} entries to {OUT_INFO}")
    print(
        f"  In Master Duel: {md_count}  |  Extra Deck cards: {extra_count}\n"
        f"  OCG banlist  Forbidden: {banned_count}  Limited: {limited_count}  "
        f"Semi-Limited: {semi_count}"
    )


if __name__ == "__main__":
    main()
