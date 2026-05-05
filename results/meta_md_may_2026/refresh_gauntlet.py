from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ygo_meta.meta_solver.gauntlet import load_decks
from ygo_meta.meta_solver.replicator import solve_replicator
from ygo_meta.simulation.battle_runner import BattleRunner
from ygo_meta.simulation.payoff_matrix import build_matrix


ROOT = Path("results/meta_md_may_2026")
DECKS_DIR = ROOT / "decks"
GAUNTLET_DIR = ROOT / "gauntlet"
PAYOFF = GAUNTLET_DIR / "payoff.npy"
DECK_IDS = GAUNTLET_DIR / "deck_ids.json"
META = ROOT / "meta_prediction.json"


def main() -> None:
    decks = load_decks(DECKS_DIR)
    GAUNTLET_DIR.mkdir(parents=True, exist_ok=True)
    pm = build_matrix(
        decks,
        BattleRunner(),
        num_episodes=500,
        seed=0,
        existing=None,
        max_workers=1,
    )
    pm.save(PAYOFF, DECK_IDS)

    sol = solve_replicator(pm.matrix, dt=0.05, max_iters=20_000, epsilon=1e-6)
    META.write_text(
        json.dumps(
            {
                "deck_ids": pm.deck_ids,
                "shares": sol.shares.tolist(),
                "fitnesses": sol.fitnesses.tolist(),
                "mean_fitness": sol.mean_fitness,
                "iterations": sol.iterations,
                "converged": sol.converged,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {PAYOFF}")
    print(f"saved {DECK_IDS}")
    print(f"saved {META}")


if __name__ == "__main__":
    main()
