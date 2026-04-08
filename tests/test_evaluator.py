"""Tests for the human-evaluator package."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from ygo_meta.deck_builder.deck_model import Deck
from ygo_meta.evaluator.delta import (
    DEFAULT_N_BASELINE,
    DEFAULT_N_TECH,
    TechVariant,
    compute_deltas,
    enqueue_delta_queries,
)
from ygo_meta.evaluator.human_runner import HumanBattleRunner
from ygo_meta.evaluator.judgment_store import (
    VALID_BUCKETS,
    Judgment,
    JudgmentStore,
    canonical_deck_hash,
    make_pending_query,
)
from ygo_meta.evaluator.sampler import sample_queries_for_pair


def _deck(arch: str, variant: str, start_code: int) -> Deck:
    main = list(range(start_code, start_code + 40))
    return Deck(archetype=arch, variant_id=variant, main=main, extra=[], side=[])


# --------------------------------------------------------------------------
# judgment_store
# --------------------------------------------------------------------------


def test_canonical_hash_order_independent() -> None:
    assert canonical_deck_hash([3, 1, 2]) == canonical_deck_hash([1, 2, 3])
    assert canonical_deck_hash([1, 2, 3]) != canonical_deck_hash([1, 2, 4])


def test_pending_append_and_dedup(tmp_path: Path) -> None:
    store = JudgmentStore(tmp_path)
    q = make_pending_query(
        "A_v0", "A", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "B_v0", "B", [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        hand_a=[1, 2, 3, 4, 5], hand_b=[11, 12, 13, 14, 15], who_first="A",
    )
    store.append_pending(q)
    store.append_pending(q)  # dedup
    assert store.pending_count() == 1


def test_judgment_roundtrip_and_pending_removed(tmp_path: Path) -> None:
    store = JudgmentStore(tmp_path)
    q = make_pending_query(
        "A_v0", "A", [1, 2, 3, 4, 5],
        "B_v0", "B", [11, 12, 13, 14, 15],
        hand_a=[1, 2, 3, 4, 5], hand_b=[11, 12, 13, 14, 15], who_first="A",
    )
    store.append_pending(q)
    j = Judgment(
        query_id=q.query_id,
        deck_a_id=q.deck_a_id, deck_b_id=q.deck_b_id,
        deck_a_hash=q.deck_a_hash, deck_b_hash=q.deck_b_hash,
        hand_a=q.hand_a, hand_b=q.hand_b, who_first=q.who_first,
        bucket=0.75,
    )
    store.append_judgment(j)
    assert store.pending_count() == 0
    loaded = store.load_judgments()
    assert len(loaded) == 1
    assert loaded[0].bucket == 0.75
    assert loaded[0].who_first == "A"


def test_invalid_bucket_rejected(tmp_path: Path) -> None:
    store = JudgmentStore(tmp_path)
    j = Judgment(
        query_id="x", deck_a_id="a", deck_b_id="b",
        deck_a_hash="h", deck_b_hash="h2",
        hand_a=[1], hand_b=[2], who_first="A", bucket=0.42,
    )
    with pytest.raises(ValueError):
        store.append_judgment(j)


def test_invalid_who_first_rejected(tmp_path: Path) -> None:
    store = JudgmentStore(tmp_path)
    j = Judgment(
        query_id="x", deck_a_id="a", deck_b_id="b",
        deck_a_hash="h", deck_b_hash="h2",
        hand_a=[1], hand_b=[2], who_first="neither", bucket=0.5,
    )
    with pytest.raises(ValueError):
        store.append_judgment(j)


def test_every_judgment_has_who_first(tmp_path: Path) -> None:
    """Invariant: no row may omit turn order."""
    store = JudgmentStore(tmp_path)
    for bucket, who in [(0.0, "A"), (1.0, "B")]:
        j = Judgment(
            query_id=f"q_{bucket}", deck_a_id="a", deck_b_id="b",
            deck_a_hash="h", deck_b_hash="h2",
            hand_a=[1], hand_b=[2], who_first=who, bucket=bucket,
        )
        store.append_judgment(j)
    for j in store.load_judgments():
        assert j.who_first in ("A", "B")


# --------------------------------------------------------------------------
# sampler
# --------------------------------------------------------------------------


def test_sampler_balances_who_first() -> None:
    d1 = _deck("Arch1", "A_v0", 1000)
    d2 = _deck("Arch2", "B_v0", 2000)
    queries = sample_queries_for_pair(d1, d2, num_queries=10, seed=7)
    assert len(queries) == 10
    first_a = sum(1 for q in queries if q.who_first == "A")
    first_b = sum(1 for q in queries if q.who_first == "B")
    assert abs(first_a - first_b) <= 1  # balanced


def test_sampler_hand_shape() -> None:
    d1 = _deck("Arch1", "A_v0", 1000)
    d2 = _deck("Arch2", "B_v0", 2000)
    queries = sample_queries_for_pair(d1, d2, num_queries=4, seed=0)
    for q in queries:
        assert len(q.hand_a) == 5
        assert len(q.hand_b) == 5
        assert all(c in d1.main for c in q.hand_a)
        assert all(c in d2.main for c in q.hand_b)


def test_sampler_force_in_hand_pins_card() -> None:
    d1 = _deck("Arch1", "A_v0", 1000)
    d2 = _deck("Arch2", "B_v0", 2000)
    pinned = 1005
    queries = sample_queries_for_pair(
        d1, d2, num_queries=8, seed=3, force_a_in_hand=[pinned]
    )
    for q in queries:
        assert pinned in q.hand_a
        assert len(q.hand_a) == 5
        # Pinned card may also appear naturally; it MUST appear at least once.
    # And the rest of hand_a should still vary across queries (fresh re-roll).
    distinct_hands = {tuple(sorted(q.hand_a)) for q in queries}
    assert len(distinct_hands) > 1


def test_sampler_zero_budget() -> None:
    d1 = _deck("A", "A_v0", 1000)
    d2 = _deck("B", "B_v0", 2000)
    assert sample_queries_for_pair(d1, d2, num_queries=0) == []


# --------------------------------------------------------------------------
# human_runner (cache hit path; no polling required)
# --------------------------------------------------------------------------


def test_human_runner_uses_cached_judgments(tmp_path: Path) -> None:
    store = JudgmentStore(tmp_path)
    d1 = _deck("Arch1", "A_v0", 1000)
    d2 = _deck("Arch2", "B_v0", 2000)
    hash_a = canonical_deck_hash(d1.main)
    hash_b = canonical_deck_hash(d2.main)
    for bucket in (1.0, 0.75):
        store.append_judgment(Judgment(
            query_id=f"q_{bucket}",
            deck_a_id=d1.variant_id, deck_b_id=d2.variant_id,
            deck_a_hash=hash_a, deck_b_hash=hash_b,
            hand_a=[1000], hand_b=[2000], who_first="A",
            bucket=bucket,
        ))
    runner = HumanBattleRunner(store_dir=tmp_path, poll_interval=0.01, verbose=False)
    result = runner.run(d1, d2, num_episodes=2, seed=0)
    assert result.win_rate_d1 == pytest.approx(0.875)
    assert result.win_rate_d2 == pytest.approx(0.125)
    assert result.deck1_id == "A_v0"


def test_human_runner_blocks_then_resumes(tmp_path: Path) -> None:
    """Runner emits pending queries; an external 'answerer' thread drains them."""
    d1 = _deck("Arch1", "A_v0", 1000)
    d2 = _deck("Arch2", "B_v0", 2000)
    store = JudgmentStore(tmp_path)

    def answer_worker():
        for _ in range(50):
            pending = store.load_pending()
            if pending:
                q = pending[0]
                store.append_judgment(Judgment(
                    query_id=q.query_id,
                    deck_a_id=q.deck_a_id, deck_b_id=q.deck_b_id,
                    deck_a_hash=q.deck_a_hash, deck_b_hash=q.deck_b_hash,
                    hand_a=q.hand_a, hand_b=q.hand_b, who_first=q.who_first,
                    bucket=0.5,
                ))
                return
            time.sleep(0.02)

    t = threading.Thread(target=answer_worker, daemon=True)
    t.start()
    runner = HumanBattleRunner(store_dir=tmp_path, poll_interval=0.02, timeout=5.0, verbose=False)
    result = runner.run(d1, d2, num_episodes=1, seed=0)
    t.join(timeout=2.0)
    assert result.win_rate_d1 == 0.5
    assert result.episodes == 1


def test_human_runner_timeout(tmp_path: Path) -> None:
    d1 = _deck("Arch1", "A_v0", 1000)
    d2 = _deck("Arch2", "B_v0", 2000)
    runner = HumanBattleRunner(store_dir=tmp_path, poll_interval=0.01, timeout=0.1, verbose=False)
    with pytest.raises(TimeoutError):
        runner.run(d1, d2, num_episodes=1, seed=0)


# --------------------------------------------------------------------------
# FastAPI server
# --------------------------------------------------------------------------


def test_eval_server_endpoints(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from ygo_meta.evaluator.web.server import create_app

    store = JudgmentStore(tmp_path)
    q = make_pending_query(
        "A_v0", "A", [1, 2, 3, 4, 5, 6, 7],
        "B_v0", "B", [11, 12, 13, 14, 15, 16, 17],
        hand_a=[1, 2, 3, 4, 5], hand_b=[11, 12, 13, 14, 15], who_first="A",
    )
    store.append_pending(q)

    app = create_app(tmp_path, banlist_version="test")
    client = TestClient(app)

    stats = client.get("/api/stats").json()
    assert stats["pending"] == 1 and stats["answered"] == 0

    queue = client.get("/api/queue").json()
    assert len(queue) == 1 and queue[0]["who_first"] == "A"

    full = client.get(f"/api/query/{q.query_id}").json()
    assert len(full["hand_a"]) == 5
    assert full["who_first"] == "A"
    assert "image" in full["hand_a"][0]

    # Invalid bucket rejected.
    bad = client.post(f"/api/query/{q.query_id}", json={"bucket": 0.3})
    assert bad.status_code == 400

    ok = client.post(f"/api/query/{q.query_id}", json={"bucket": 0.75, "note": "test"})
    assert ok.status_code == 200

    stats = client.get("/api/stats").json()
    assert stats["pending"] == 0 and stats["answered"] == 1

    judgments = store.load_judgments()
    assert judgments[0].bucket == 0.75
    assert judgments[0].banlist_version == "test"
    assert judgments[0].who_first == "A"


# --------------------------------------------------------------------------
# rename guard
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# tech-delta evaluator
# --------------------------------------------------------------------------


def _make_tech_variant(baseline: Deck, code: int, name: str) -> TechVariant:
    """Mirrors cli/tech_delta._make_tech_variant: replace the baseline's last
    main-deck slot with the candidate so the variant stays 40 cards.
    """
    new_main = list(baseline.main)
    new_main[-1] = code
    return TechVariant(
        name=name,
        code=code,
        deck=Deck(
            archetype=baseline.archetype,
            variant_id=f"{baseline.variant_id}+{name}",
            main=new_main, extra=[], side=[],
        ),
    )


def test_enqueue_delta_queries_total_count(tmp_path: Path) -> None:
    store = JudgmentStore(tmp_path)
    base = _deck("Arch1", "A_v0", 1000)
    opp = _deck("Arch2", "B_v0", 2000)
    techs = [
        _make_tech_variant(base, 9001, "MaxxC"),
        _make_tech_variant(base, 9002, "AshBlossom"),
    ]
    enqueue_delta_queries(store, base, opp, techs, n_baseline=12, n_tech=4, seed=0)
    # 12 baseline + 2*4 tech = 20 queries
    assert store.pending_count() == 12 + 2 * 4

    # Tech queries pin the tech code into hand A.
    pending = store.load_pending()
    base_hash = canonical_deck_hash(base.main)
    tech_pendings = [p for p in pending if p.deck_a_hash != base_hash]
    assert tech_pendings, "expected tech-card pendings"
    for p in tech_pendings:
        # Whichever tech variant this query belongs to, its code must appear in hand A.
        assert any(c in (9001, 9002) for c in p.hand_a)


def test_compute_deltas_math(tmp_path: Path) -> None:
    store = JudgmentStore(tmp_path)
    base = _deck("Arch1", "A_v0", 1000)
    opp = _deck("Arch2", "B_v0", 2000)
    tech = _make_tech_variant(base, 9001, "MaxxC")
    base_hash = canonical_deck_hash(base.main)
    tech_hash = canonical_deck_hash(tech.deck.main)
    opp_hash = canonical_deck_hash(opp.main)

    # Baseline: 4 judgments averaging 0.5
    for i, b in enumerate((1.0, 0.5, 0.5, 0.0)):
        store.append_judgment(Judgment(
            query_id=f"b{i}", deck_a_id="A_v0", deck_b_id="B_v0",
            deck_a_hash=base_hash, deck_b_hash=opp_hash,
            hand_a=[1000], hand_b=[2000], who_first="A", bucket=b,
        ))
    # Tech: 2 judgments averaging 0.875
    for i, b in enumerate((1.0, 0.75)):
        store.append_judgment(Judgment(
            query_id=f"t{i}", deck_a_id="A_v0+MaxxC", deck_b_id="B_v0",
            deck_a_hash=tech_hash, deck_b_hash=opp_hash,
            hand_a=[9001], hand_b=[2000], who_first="A", bucket=b,
        ))

    p_b, n_b, results = compute_deltas(store, base, opp, [tech])
    assert n_b == 4
    assert p_b == pytest.approx(0.5)
    assert len(results) == 1
    r = results[0]
    assert r.tech_winrate == pytest.approx(0.875)
    assert r.delta == pytest.approx(0.375)
    assert r.se_delta > 0


def test_compute_deltas_skips_unjudged_techs(tmp_path: Path) -> None:
    store = JudgmentStore(tmp_path)
    base = _deck("Arch1", "A_v0", 1000)
    opp = _deck("Arch2", "B_v0", 2000)
    tech_a = _make_tech_variant(base, 9001, "A")
    tech_b = _make_tech_variant(base, 9002, "B")
    _, _, results = compute_deltas(store, base, opp, [tech_a, tech_b])
    assert results == []


def test_default_allocation_ratio_is_three_to_one() -> None:
    """Sanity-check the documented 3:1 baseline:tech default."""
    assert DEFAULT_N_BASELINE == 3 * DEFAULT_N_TECH


# --------------------------------------------------------------------------
# server: reveal-next-card endpoint
# --------------------------------------------------------------------------


def test_eval_server_draw_endpoint_is_deterministic(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from ygo_meta.evaluator.web.server import create_app

    store = JudgmentStore(tmp_path)
    main_a = list(range(1000, 1040))
    main_b = list(range(2000, 2040))
    q = make_pending_query(
        "A_v0", "A", main_a,
        "B_v0", "B", main_b,
        hand_a=main_a[:5], hand_b=main_b[:5], who_first="A",
    )
    store.append_pending(q)

    app = create_app(tmp_path)
    client = TestClient(app)

    r1 = client.get(f"/api/query/{q.query_id}/draw?side=A&n=3").json()
    r2 = client.get(f"/api/query/{q.query_id}/draw?side=A&n=3").json()
    assert r1 == r2  # deterministic per (query, side)
    assert len(r1["cards"]) == 3
    drawn_codes = [c["code"] for c in r1["cards"]]
    # Drawn cards must come from the deck and not from the hand.
    for code in drawn_codes:
        assert code in main_a
        assert code not in q.hand_a

    # Calling with n=1 returns the prefix of the n=3 sequence.
    r_one = client.get(f"/api/query/{q.query_id}/draw?side=A&n=1").json()
    assert r_one["cards"][0]["code"] == drawn_codes[0]

    # Side B is independent.
    rb = client.get(f"/api/query/{q.query_id}/draw?side=B&n=2").json()
    assert all(c["code"] in main_b for c in rb["cards"])

    # Bad side rejected.
    bad = client.get(f"/api/query/{q.query_id}/draw?side=Z&n=1")
    assert bad.status_code == 400


def test_deck_uses_flex_combo() -> None:
    d = Deck(archetype="X", variant_id="X_v0", main=[1] * 40, extra=[], side=[])
    assert hasattr(d, "flex_combo")
    assert not hasattr(d, "staple_combo")
