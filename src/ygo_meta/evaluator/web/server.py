"""
FastAPI server for the human judgment UI.

Run:

    ygo-eval-server --store-dir results/judgments

Endpoints:

- ``GET  /api/queue``        → list of pending queries
- ``GET  /api/query/{id}``   → one pending query (full card lists)
- ``POST /api/query/{id}``   → submit an answer (bucket 0|0.25|0.5|0.75|1, optional note)
- ``GET  /api/stats``        → {pending: int, answered: int}
- ``GET  /``                 → static SPA (index.html)
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import typer
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ygo_meta.evaluator.judgment_store import (
    VALID_BUCKETS,
    Judgment,
    JudgmentStore,
)


class AnswerPayload(BaseModel):
    bucket: float
    note: str = ""
    evaluator_id: str = "human"


def _load_card_names() -> dict[int, str]:
    path = Path("data/card_names.json")
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}


def create_app(store_dir: Path, banlist_version: str = "unknown") -> FastAPI:
    store = JudgmentStore(store_dir)
    card_names = _load_card_names()
    static_dir = Path(__file__).parent / "static"

    app = FastAPI(title="YGO Human Evaluator")

    def _decorate_cards(codes: list[int]) -> list[dict]:
        return [
            {
                "code": c,
                "name": card_names.get(c, str(c)),
                "image": f"https://images.ygoprodeck.com/images/cards_small/{c}.jpg",
            }
            for c in codes
        ]

    @app.get("/api/stats")
    def stats() -> dict:
        return {
            "pending": store.pending_count(),
            "answered": len(store.load_judgments()),
            "banlist_version": banlist_version,
        }

    @app.get("/api/queue")
    def queue() -> list[dict]:
        return [
            {
                "query_id": q.query_id,
                "deck_a_id": q.deck_a_id,
                "deck_b_id": q.deck_b_id,
                "deck_a_archetype": q.deck_a_archetype,
                "deck_b_archetype": q.deck_b_archetype,
                "who_first": q.who_first,
            }
            for q in store.load_pending()
        ]

    @app.get("/api/query/{query_id}")
    def get_query(query_id: str) -> dict:
        q = store.get_pending(query_id)
        if q is None:
            raise HTTPException(404, f"query {query_id} not found or already answered")
        return {
            "query_id": q.query_id,
            "deck_a_id": q.deck_a_id,
            "deck_b_id": q.deck_b_id,
            "deck_a_archetype": q.deck_a_archetype,
            "deck_b_archetype": q.deck_b_archetype,
            "who_first": q.who_first,
            "hand_a": _decorate_cards(q.hand_a),
            "hand_b": _decorate_cards(q.hand_b),
            "deck_a_main": _decorate_cards(q.deck_a_main),
            "deck_b_main": _decorate_cards(q.deck_b_main),
        }

    @app.get("/api/query/{query_id}/draw")
    def draw_next(query_id: str, side: str = "A", n: int = 1) -> dict:
        """Reveal the next ``n`` cards from a side's deck (cards not in hand).

        Order is deterministic per (query_id, side) so repeated clicks always
        show the same sequence — useful for "what's the next draw?" reasoning
        with cards like Maxx C and Mulcharmy.
        """
        q = store.get_pending(query_id)
        if q is None:
            raise HTTPException(404, f"query {query_id} not found or already answered")
        if side not in ("A", "B"):
            raise HTTPException(400, "side must be 'A' or 'B'")
        if not (1 <= n <= 60):
            raise HTTPException(400, "n must be in [1, 60]")

        deck = q.deck_a_main if side == "A" else q.deck_b_main
        hand = q.hand_a if side == "A" else q.hand_b

        # Remove hand cards from the pool (one removal per copy in hand).
        pool = list(deck)
        for c in hand:
            if c in pool:
                pool.remove(c)

        seed_key = f"{query_id}:{side}".encode("utf-8")
        seed = int(hashlib.sha1(seed_key).hexdigest()[:8], 16)
        rng = random.Random(seed)
        rng.shuffle(pool)
        drawn = pool[:n]
        return {"side": side, "cards": _decorate_cards(drawn)}

    @app.post("/api/query/{query_id}")
    def post_answer(query_id: str, payload: AnswerPayload) -> dict:
        if payload.bucket not in VALID_BUCKETS:
            raise HTTPException(400, f"bucket must be one of {VALID_BUCKETS}")
        q = store.get_pending(query_id)
        if q is None:
            raise HTTPException(404, f"query {query_id} not found or already answered")
        j = Judgment(
            query_id=q.query_id,
            deck_a_id=q.deck_a_id,
            deck_b_id=q.deck_b_id,
            deck_a_hash=q.deck_a_hash,
            deck_b_hash=q.deck_b_hash,
            hand_a=q.hand_a,
            hand_b=q.hand_b,
            who_first=q.who_first,
            bucket=payload.bucket,
            evaluator_id=payload.evaluator_id,
            banlist_version=banlist_version,
            note=payload.note,
        )
        store.append_judgment(j)
        return {"ok": True, "remaining": store.pending_count()}

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app


cli = typer.Typer()


@cli.command()
def main(
    store_dir: Path = typer.Option(Path("results/judgments"), help="JudgmentStore directory"),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    banlist_version: str = typer.Option("unknown"),
) -> None:
    """Launch the human judgment web UI."""
    import uvicorn
    app = create_app(store_dir, banlist_version=banlist_version)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cli()
