# Banlist files (`lflist.conf`)

This directory holds EDOPro-format `lflist.conf` files used by
`ygo_meta.deck_builder.validator.validate_deck(..., lflist=...)`.

## Master Duel banlist

The official Master Duel banlist is **not** committed here — it changes every
few weeks, and committing a stale copy would silently mis-validate decks. Drop
the current file at:

    data/lflist/master_duel.lflist.conf

Sources (pick one and keep updated):

- ProjectIgnis EDOPro `LFLists` repo (search for the Master Duel section).
- Master Duel Meta's published "Forbidden & Limited" page.

## Format

```
!2024.4 Master Duel
# comments allowed
$whitelist
12345678 0   -- forbidden
23456789 1   -- limited
34567890 2   -- semi-limited
```

A card not listed is unrestricted (3 copies). Multiple banlists may live in
one file; pass `parse_lflist(path, banlist_name=...)` to pick a specific one.
