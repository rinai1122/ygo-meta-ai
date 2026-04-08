"""
HumanBattleRunner — implements the BattleRunner ``run()`` interface by
emitting pending queries to a :class:`JudgmentStore` and blocking until a
human answers them through the web UI.

Drop-in for ``build_matrix(decks, runner, ...)``.
"""

from __future__ import annotations

import time
from pathlib import Path

from ygo_meta.deck_builder.deck_model import Deck
from ygo_meta.evaluator.judgment_store import (
    Judgment,
    JudgmentStore,
    canonical_deck_hash,
)
from ygo_meta.evaluator.sampler import sample_queries_for_pair
from ygo_meta.simulation.battle_runner import BattleResult


class HumanBattleRunner:
    """Runner that produces BattleResults from cached or freshly-collected
    human judgments.

    Parameters
    ----------
    store_dir:
        Directory for the JudgmentStore (``results/judgments`` by default).
    banlist_version:
        Stamped into every judgment. Change when the banlist changes so
        stale cached answers don't contaminate a new meta.
    poll_interval:
        Seconds between polls when waiting for a human answer.
    timeout:
        Max seconds to wait for a single matchup's queries to be answered.
        None = wait forever.
    verbose:
        Print progress lines while waiting.
    """

    def __init__(
        self,
        store_dir: Path,
        banlist_version: str = "unknown",
        poll_interval: float = 1.0,
        timeout: float | None = None,
        verbose: bool = True,
    ):
        self.store = JudgmentStore(Path(store_dir))
        self.banlist_version = banlist_version
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.verbose = verbose

    # ------------------------------------------------------------------
    # BattleRunner-compatible API
    # ------------------------------------------------------------------
    def run(
        self,
        deck1: Deck,
        deck2: Deck,
        num_episodes: int = 1,
        seed: int = 0,
    ) -> BattleResult:
        hash_a = canonical_deck_hash(deck1.main)
        hash_b = canonical_deck_hash(deck2.main)

        # 1. See if we already have enough judgments for this pair.
        existing = self.store.judgments_for_pair(hash_a, hash_b)
        need = max(0, num_episodes - len(existing))

        # 2. Emit pending queries for anything we still need.
        if need > 0:
            queries = sample_queries_for_pair(
                deck1, deck2, num_queries=need, seed=seed
            )
            for q in queries:
                self.store.append_pending(q)
            if self.verbose:
                print(
                    f"  [human] {deck1.variant_id} vs {deck2.variant_id}: "
                    f"waiting for {need} judgment(s) "
                    f"(pending queue: {self.store.pending_count()})",
                    flush=True,
                )

        # 3. Poll the store until we have enough.
        deadline = None if self.timeout is None else time.time() + self.timeout
        while True:
            existing = self.store.judgments_for_pair(hash_a, hash_b)
            if len(existing) >= num_episodes:
                break
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(
                    f"Timed out waiting for human judgments on "
                    f"{deck1.variant_id} vs {deck2.variant_id}"
                )
            time.sleep(self.poll_interval)

        # 4. Aggregate the most recent `num_episodes` judgments.
        chosen = existing[-num_episodes:]
        wr_a = sum(j.bucket for j in chosen) / len(chosen)
        return BattleResult(
            win_rate_d1=wr_a,
            win_rate_d2=1.0 - wr_a,
            episodes=num_episodes,
            deck1_id=deck1.variant_id,
            deck2_id=deck2.variant_id,
        )
