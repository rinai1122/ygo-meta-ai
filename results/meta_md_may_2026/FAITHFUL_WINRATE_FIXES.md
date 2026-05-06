# Faithful Winrate Matrix Fixes

Date: 2026-05-06

This pass intentionally removed the output-side 60/40 normalization idea. The
payoff matrices are empirical outputs from the runner, not capped post hoc.

## Code Changes

- `src/ygo_meta/simulation/payoff_matrix.py`
  - Runs both seat orders for each unordered matchup by default.
  - Averages `A(P0) vs B(P1)` with `B(P0) vs A(P1)` from each deck's perspective.
  - Keeps matrix antisymmetry without changing measured matchup direction.

- `src/ygo_meta/meta_solver/gauntlet.py`
  - Fingerprints now include cache schema, seat-balance mode, and episode count.
  - This prevents stale one-seat or smoke-run caches from being reused as full
    gauntlet results.

- `src/ygo_meta/simulation/obs_sanitizer.py`
  - Maps card IDs beyond the checkpoint embedding table to ID 0, the model's
    unknown-card bucket.
  - This avoids feeding newer Master Duel cards into invalid or unrelated
    trained embeddings.

- `scripts/game_runner.py`
  - Uses the observation sanitizer before RL inference.
  - Avoids obvious self-turn Maxx "C" / Mulcharmy activations when pass is
    legal.
  - Keeps action choice restricted to currently legal engine options.

## New Runs

- `results/meta_md_may_2026/gauntlet_seat_balanced_80/`
  - Six local decks.
  - 80 games per seat direction, 160 total games per unordered pair.
  - Uncapped empirical matrix.

- `results/meta_md_may_2026/gauntlet_top_tier_seat_balanced_200_guarded/`
  - K9 Vanquish Soul, Kewl Tune, Ryzeal Mitsurugi.
  - 200 games per seat direction, 400 total games per unordered pair.
  - Guarded runner, uncapped empirical matrix.

Top-tier guarded matrix:

| Deck | K9 Vanquish Soul | Kewl Tune | Ryzeal Mitsurugi |
| --- | ---: | ---: | ---: |
| K9 Vanquish Soul | 0.500 | 0.370 | 0.583 |
| Kewl Tune | 0.630 | 0.500 | 0.637 |
| Ryzeal Mitsurugi | 0.417 | 0.362 | 0.500 |

## Verification

- `pytest tests/test_game_runner_obs.py tests/test_gauntlet_safety.py`
  - Passed with the committed focused tests.
- `python -m py_compile scripts/game_runner.py src/ygo_meta/simulation/obs_sanitizer.py src/ygo_meta/simulation/payoff_matrix.py src/ygo_meta/meta_solver/gauntlet.py`
  - Passed.
