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
) -> PayoffMatrix:
    """
    Run all missing pairwise matchups and return a complete PayoffMatrix.

    If `existing` is provided, only missing matchups are run (cached results reused).
    """
    pm = existing or PayoffMatrix([d.variant_id for d in decks])
    deck_map = {d.variant_id: d for d in decks}

    pairs = pm.missing_pairs()
    total = len(pairs)
    for k, (i, j) in enumerate(pairs, 1):
        d1_id, d2_id = pm.deck_ids[i], pm.deck_ids[j]
        d1, d2 = deck_map[d1_id], deck_map[d2_id]
        print(f"  [{k}/{total}] {d1_id} vs {d2_id}", flush=True)
        result = runner.run(d1, d2, num_episodes=num_episodes, seed=seed + k)
        pm.set_result(result)

    return pm
