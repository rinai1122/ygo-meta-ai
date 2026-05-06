# Master Duel Meta Result - May 5, 2026

## External Meta Check

Checked on 2026-05-05. Master Duel Meta currently lists the latest release as
`The Frenzied Tuning`, released on 2026-04-09, and shows an upcoming
Forbidden/Limited List change for 2026-05-07.

Current Master Duel Meta power-ranking leaders from the tier list:

| Rank | Deck / Engine | Power |
| ---: | --- | ---: |
| 1 | Kewl Tune | 34.5 |
| 2 | K9 Engine | 14.0 |
| 3 | Vanquish Soul K9 | 13.0 |

Sources:

- https://www.masterduelmeta.com/tier-list
- https://www.masterduelmeta.com/
- https://www.wargamer.com/yu-gi-oh-master-duel/banlist

## Local Gauntlet Scope

The local May 2026 workspace currently has these curated decks:

- `KewlTune_Sweep_MDM_2026-04-12`
- `MagnetAdamancipator_YGOPRODeck_697472`
- `RadiantSkyStriker_YGOPRODeck_706980`
- `RyzealMitsurugi_Fuan_MDM_2026-03-07`

This means the local run includes one current top MDM leader (`Kewl Tune`), but
does not yet include local YDKs for `K9 Engine` or `Vanquish Soul K9`.

## Guarded Top-Tier Seat-Balanced Result

Input files:

- `results/meta_md_may_2026/gauntlet_top_tier_seat_balanced_200_guarded/payoff.npy`
- `results/meta_md_may_2026/gauntlet_top_tier_seat_balanced_200_guarded/deck_ids.json`
- `results/meta_md_may_2026/gauntlet_top_tier_seat_balanced_200_guarded/meta_prediction.json`

This run uses 200 games in each seat direction per pairing, so every cell is
based on 400 total games. It also uses the guarded RL runner that maps cards
outside the checkpoint embedding table to the model's unknown-card bucket and
avoids obvious self-turn Maxx "C" / Mulcharmy activations when pass is legal.

Replicator result:

| Rank | Deck | Predicted Share | WR vs Final Field |
| ---: | --- | ---: | ---: |
| 1 | `KewlTune_Sweep_MDM_2026-04-12` | 100.00% | 50.00% |
| 2 | `K9Vanquishsoul` | 0.00% | 37.00% |
| 3 | `RyzealMitsurugi_Fuan_MDM_2026-03-07` | 0.00% | 36.25% |

Solver status: converged in 1807 iterations, mean fitness 0.5000.

Rows are the deck being evaluated; columns are the opponent. Values are row deck
win rates.

| Deck | K9 Vanquish Soul | Kewl Tune | Ryzeal Mitsurugi |
| --- | ---: | ---: | ---: |
| K9 Vanquish Soul | 0.500 | 0.370 | 0.583 |
| Kewl Tune | 0.630 | 0.500 | 0.637 |
| Ryzeal Mitsurugi | 0.417 | 0.362 | 0.500 |

## Previous Full 500-Episode Result

Input files:

- `results/meta_md_may_2026/gauntlet/payoff.npy`
- `results/meta_md_may_2026/gauntlet/deck_ids.json`
- `results/meta_md_may_2026/meta_prediction.json`

Replicator result:

| Rank | Deck | Predicted Share | WR vs Final Field |
| ---: | --- | ---: | ---: |
| 1 | `MagnetAdamancipator_YGOPRODeck_697472` | 100.00% | 50.00% |
| 2 | `RyzealMitsurugi_Fuan_MDM_2026-03-07` | 0.00% | 13.40% |
| 3 | `KewlTune_Sweep_MDM_2026-04-12` | 0.00% | 12.40% |
| 4 | `RadiantSkyStriker_YGOPRODeck_706980` | 0.00% | 1.80% |

Solver status: converged in 644 iterations, mean fitness 0.5000.

## Payoff Matrix

Rows are the deck being evaluated; columns are the opponent. Values are row deck
win rates.

| Deck | Kewl Tune | Magnet Adamancipator | Radiant Sky Striker | Ryzeal Mitsurugi |
| --- | ---: | ---: | ---: | ---: |
| Kewl Tune | 0.500 | 0.124 | 0.674 | 0.324 |
| Magnet Adamancipator | 0.876 | 0.500 | 0.982 | 0.866 |
| Radiant Sky Striker | 0.326 | 0.018 | 0.500 | 0.224 |
| Ryzeal Mitsurugi | 0.676 | 0.134 | 0.776 | 0.500 |

## Notes

- This result uses 500 BO1 games per matchup for the four local decks.
- `results/meta_md_may_2026/meta_prediction_smoke.json` is retained as the
  earlier smoke run, but the table above uses the full refresh.
- The current external meta check indicates the next local refresh should add
  `K9 Engine` and `Vanquish Soul K9` YDKs before treating this as the newest
  Master Duel meta coverage.
