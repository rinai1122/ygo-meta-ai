from __future__ import annotations

import pytest

from ygo_meta.deck_builder.deck_model import Deck
from ygo_meta.meta_solver.gauntlet import run_gauntlet, validate_gauntlet_decks
from ygo_meta.simulation.payoff_matrix import build_matrix
from ygo_meta.simulation.battle_runner import BattleResult, BattleRunIncompleteError


def test_validate_gauntlet_decks_rejects_duplicate_ids() -> None:
    deck = Deck(archetype="a", variant_id="same", main=[1] * 40, extra=[], side=[])

    with pytest.raises(ValueError, match="duplicate deck id"):
        validate_gauntlet_decks([deck, deck], lflist_path=None)


def test_incomplete_battle_error_is_runtime_error() -> None:
    err = BattleRunIncompleteError("short run")

    assert isinstance(err, RuntimeError)


def test_run_gauntlet_ignores_corrupt_fingerprint_cache(tmp_path) -> None:
    deck1 = Deck(archetype="a", variant_id="a", main=list(range(40)), extra=[], side=[])
    deck2 = Deck(archetype="b", variant_id="b", main=list(range(100, 140)), extra=[], side=[])
    cache = tmp_path / "payoff.npy"
    ids = tmp_path / "deck_ids.json"
    ids.with_suffix(".fingerprints.json").write_text("", encoding="utf-8")

    class Runner:
        def run_batch(self, matchups):
            return [
                BattleResult(
                    win_rate_d1=0.25 if d1.variant_id == "a" else 0.60,
                    win_rate_d2=0.75 if d1.variant_id == "a" else 0.40,
                    episodes=n_ep,
                    deck1_id=d1.variant_id,
                    deck2_id=d2.variant_id,
                )
                for d1, d2, n_ep, _seed in matchups
            ]

    result = run_gauntlet(
        [deck1, deck2],
        runner=Runner(),
        num_episodes=4,
        cache_path=cache,
        cache_ids_path=ids,
        validate_decks=False,
    )

    assert result.payoff[0, 1] == 0.325


def test_build_matrix_averages_both_seats() -> None:
    deck1 = Deck(archetype="a", variant_id="a", main=list(range(40)), extra=[], side=[])
    deck2 = Deck(archetype="b", variant_id="b", main=list(range(100, 140)), extra=[], side=[])

    class Runner:
        def run(self, d1, d2, num_episodes, seed):
            if d1.variant_id == "a":
                wr1 = 0.90
            else:
                wr1 = 0.70
            return BattleResult(
                win_rate_d1=wr1,
                win_rate_d2=1.0 - wr1,
                episodes=num_episodes,
                deck1_id=d1.variant_id,
                deck2_id=d2.variant_id,
            )

    pm = build_matrix([deck1, deck2], Runner(), num_episodes=10, max_workers=1)

    assert pm.matrix[0, 1] == 0.60
    assert pm.matrix[1, 0] == 0.40
