"""
Build and manage the D×D payoff matrix.

W[i][j] = win rate of deck i against deck j.
The matrix is antisymmetric in a zero-sum sense: W[i][j] + W[j][i] ≈ 1.0.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ygo_meta.deck_builder.deck_model import Deck
from ygo_meta.simulation.battle_runner import BattleResult, BattleRunner


class PayoffMatrix:
    def __init__(self, deck_ids: list[str]) -> None:
        self.deck_ids = deck_ids
        self._idx = {d: i for i, d in enumerate(deck_ids)}
        n = len(deck_ids)
        # NaN = not yet computed; diagonal = 0.5 (self-play)
        self.matrix = np.full((n, n), np.nan)
        np.fill_diagonal(self.matrix, 0.5)

    @property
    def n(self) -> int:
        return len(self.deck_ids)

    def set_result(self, result: BattleResult) -> None:
        i = self._idx[result.deck1_id]
        j = self._idx[result.deck2_id]
        self.matrix[i, j] = result.win_rate_d1
        self.matrix[j, i] = result.win_rate_d2

    def is_complete(self) -> bool:
        return not np.any(np.isnan(self.matrix))

    def missing_pairs(self) -> list[tuple[int, int]]:
        pairs = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if np.isnan(self.matrix[i, j]):
                    pairs.append((i, j))
        return pairs

    def save(self, npy_path: Path, ids_path: Path) -> None:
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(npy_path), self.matrix)
        ids_path.write_text(json.dumps(self.deck_ids, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, npy_path: Path, ids_path: Path) -> "PayoffMatrix":
        deck_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        pm = cls(deck_ids)
        pm.matrix = np.load(str(npy_path))
        return pm


def build_matrix(
    decks: list[Deck],
    runner: BattleRunner,
    num_episodes: int = 128,
    seed: int = 0,
    existing: PayoffMatrix | None = None,
    max_workers: int | None = None,
    seat_balanced: bool = True,
) -> PayoffMatrix:
    """
    Run all missing pairwise matchups and return a complete PayoffMatrix.

    If `existing` is provided, only missing matchups are run (cached results reused).

    By default each unordered pair is seat-balanced: A/B and B/A are both run
    and the two observations are averaged from each deck's perspective.  This
    avoids leaking player-slot / turn-order bias into archetype win rates.

    When the runner supports ``run_batch`` (e.g. :class:`BattleRunner` with an RL
    checkpoint), pairs are grouped into *max_workers* batches — each batch runs in
    a single subprocess that initialises JAX only once, giving a large speedup.

    Otherwise, pairs run in parallel via ``runner.run()`` (one subprocess each).
    """
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    pm = existing or PayoffMatrix([d.variant_id for d in decks])
    deck_map = {d.variant_id: d for d in decks}

    pairs = pm.missing_pairs()
    total = len(pairs)
    if not pairs:
        return pm

    use_batch = hasattr(runner, "run_batch")

    if max_workers is None:
        cpus = os.cpu_count() or 4
        # Batch workers are CPU-heavy (JAX inference) — cap to avoid contention.
        # Non-batch workers are subprocess-bound, so more threads are fine.
        max_workers = min(cpus // 2 or 1 if use_batch else cpus, total)
        max_workers = max(max_workers, 1)

    if use_batch:
        # Split pairs into chunks, one subprocess per chunk.
        chunks: list[list[tuple[int, int, int]]] = [[] for _ in range(max_workers)]
        for k, (i, j) in enumerate(pairs):
            chunks[k % max_workers].append((i, j, seed + k + 1))

        def _run_chunk(chunk: list[tuple[int, int, int]]) -> list[BattleResult]:
            batch = []
            for i, j, s in chunk:
                d1 = deck_map[pm.deck_ids[i]]
                d2 = deck_map[pm.deck_ids[j]]
                batch.append((d1, d2, num_episodes, s))
                if seat_balanced:
                    batch.append((d2, d1, num_episodes, s + 1_000_000))
            return runner.run_batch(batch)

        print(f"  Running {total} matchups in {max_workers} batch workers ...", flush=True)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_run_chunk, ch) for ch in chunks if ch]
            for future in as_completed(futures):
                results = future.result()
                if not seat_balanced:
                    for result in results:
                        pm.set_result(result)
                    continue
                if len(results) % 2 != 0:
                    raise RuntimeError("seat-balanced batch returned an odd number of results")
                for k in range(0, len(results), 2):
                    first = results[k]
                    swapped = results[k + 1]
                    if (
                        first.deck1_id != swapped.deck2_id
                        or first.deck2_id != swapped.deck1_id
                    ):
                        raise RuntimeError(
                            "seat-balanced batch result order mismatch: "
                            f"{first.deck1_id}/{first.deck2_id} then "
                            f"{swapped.deck1_id}/{swapped.deck2_id}"
                        )
                    total_eps = first.episodes + swapped.episodes
                    wr_d1 = (
                        first.win_rate_d1 * first.episodes
                        + swapped.win_rate_d2 * swapped.episodes
                    ) / total_eps
                    pm.set_result(BattleResult(
                        win_rate_d1=wr_d1,
                        win_rate_d2=1.0 - wr_d1,
                        episodes=total_eps,
                        deck1_id=first.deck1_id,
                        deck2_id=first.deck2_id,
                    ))
    else:
        # Fallback: one subprocess per matchup, parallelised with threads.
        def _run_pair(k: int, i: int, j: int) -> BattleResult:
            d1_id, d2_id = pm.deck_ids[i], pm.deck_ids[j]
            d1, d2 = deck_map[d1_id], deck_map[d2_id]
            print(f"  [{k}/{total}] {d1_id} vs {d2_id}", flush=True)
            first = runner.run(d1, d2, num_episodes=num_episodes, seed=seed + k * 2)
            if not seat_balanced:
                return first
            swapped = runner.run(d2, d1, num_episodes=num_episodes, seed=seed + k * 2 + 1)
            total_eps = first.episodes + swapped.episodes
            wr_d1 = (
                first.win_rate_d1 * first.episodes
                + swapped.win_rate_d2 * swapped.episodes
            ) / total_eps
            return BattleResult(
                win_rate_d1=wr_d1,
                win_rate_d2=1.0 - wr_d1,
                episodes=total_eps,
                deck1_id=d1_id,
                deck2_id=d2_id,
            )

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_run_pair, k, i, j): (i, j)
                for k, (i, j) in enumerate(pairs, 1)
            }
            for future in as_completed(futures):
                pm.set_result(future.result())

    return pm
