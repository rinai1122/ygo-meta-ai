"""
JSONL-backed stores for human judgments and pending queries.

Two files under `results/judgments/`:

- ``judgments.jsonl`` — append-only; one completed judgment per line.
- ``pending.jsonl``   — mutable queue of queries awaiting a human answer.

Both files are safe to read concurrently (each line is an independent JSON object).
Writes use a coarse file lock so a runner and the web server can cooperate across
processes on the same machine.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


VALID_BUCKETS = (0.0, 0.25, 0.5, 0.75, 1.0)


def canonical_deck_hash(main: list[int]) -> str:
    """Stable identity for a deck's main list (order-independent)."""
    payload = ",".join(str(c) for c in sorted(main))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _query_id(
    hash_a: str,
    hash_b: str,
    hand_a: list[int],
    hand_b: list[int],
    who_first: str,
) -> str:
    key = "|".join(
        [
            hash_a,
            hash_b,
            ",".join(str(c) for c in hand_a),
            ",".join(str(c) for c in hand_b),
            who_first,
        ]
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


@dataclass
class PendingQuery:
    query_id: str
    deck_a_id: str
    deck_b_id: str
    deck_a_hash: str
    deck_b_hash: str
    deck_a_main: list[int]
    deck_b_main: list[int]
    deck_a_archetype: str
    deck_b_archetype: str
    hand_a: list[int]           # 5 card codes
    hand_b: list[int]           # 5 card codes
    who_first: str              # "A" or "B"
    created_at: float = field(default_factory=time.time)


@dataclass
class Judgment:
    query_id: str
    deck_a_id: str
    deck_b_id: str
    deck_a_hash: str
    deck_b_hash: str
    hand_a: list[int]
    hand_b: list[int]
    who_first: str
    bucket: float               # one of VALID_BUCKETS; fraction of wins for deck A
    evaluator_id: str = "human"
    banlist_version: str = "unknown"
    note: str = ""
    created_at: float = field(default_factory=time.time)


class JudgmentStore:
    """Append-only JSONL store with a sibling pending-query queue."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.judgments_path = self.root / "judgments.jsonl"
        self.pending_path = self.root / "pending.jsonl"
        self._lock_path = self.root / ".lock"

    # ---------- judgments ----------

    def append_judgment(self, j: Judgment) -> None:
        if j.bucket not in VALID_BUCKETS:
            raise ValueError(f"bucket {j.bucket} not in {VALID_BUCKETS}")
        if j.who_first not in ("A", "B"):
            raise ValueError(f"who_first must be 'A' or 'B', got {j.who_first!r}")
        with self._locked():
            with self.judgments_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(j)) + "\n")
            self._remove_pending_locked(j.query_id)

    def load_judgments(self) -> list[Judgment]:
        if not self.judgments_path.exists():
            return []
        out: list[Judgment] = []
        with self.judgments_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                out.append(Judgment(**data))
        return out

    def judgments_for_pair(self, hash_a: str, hash_b: str) -> list[Judgment]:
        """Return judgments matching this ordered deck pair (A vs B)."""
        return [
            j for j in self.load_judgments()
            if j.deck_a_hash == hash_a and j.deck_b_hash == hash_b
        ]

    # ---------- pending queries ----------

    def load_pending(self) -> list[PendingQuery]:
        if not self.pending_path.exists():
            return []
        out: list[PendingQuery] = []
        with self.pending_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                out.append(PendingQuery(**data))
        return out

    def append_pending(self, q: PendingQuery) -> None:
        with self._locked():
            existing = {p.query_id for p in self._load_pending_unlocked()}
            answered = {j.query_id for j in self._load_judgments_unlocked()}
            if q.query_id in existing or q.query_id in answered:
                return
            with self.pending_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(q)) + "\n")

    def pending_count(self) -> int:
        return len(self.load_pending())

    def get_pending(self, query_id: str) -> PendingQuery | None:
        for p in self.load_pending():
            if p.query_id == query_id:
                return p
        return None

    # ---------- internals ----------

    def _load_pending_unlocked(self) -> list[PendingQuery]:
        if not self.pending_path.exists():
            return []
        out: list[PendingQuery] = []
        for line in self.pending_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(PendingQuery(**json.loads(line)))
        return out

    def _load_judgments_unlocked(self) -> list[Judgment]:
        if not self.judgments_path.exists():
            return []
        out: list[Judgment] = []
        for line in self.judgments_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(Judgment(**json.loads(line)))
        return out

    def _remove_pending_locked(self, query_id: str) -> None:
        remaining = [
            p for p in self._load_pending_unlocked() if p.query_id != query_id
        ]
        tmp = self.pending_path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for p in remaining:
                f.write(json.dumps(asdict(p)) + "\n")
        os.replace(tmp, self.pending_path)

    def _locked(self):
        return _FileLock(self._lock_path)


class _FileLock:
    """Minimal cross-process lock via O_EXCL create/delete."""

    def __init__(self, path: Path, timeout: float = 10.0):
        self.path = path
        self.timeout = timeout

    def __enter__(self):
        deadline = time.time() + self.timeout
        while True:
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return self
            except FileExistsError:
                if time.time() > deadline:
                    # Stale lock — steal it.
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                time.sleep(0.02)

    def __exit__(self, exc_type, exc, tb):
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def make_pending_query(
    deck_a_id: str,
    deck_a_archetype: str,
    deck_a_main: list[int],
    deck_b_id: str,
    deck_b_archetype: str,
    deck_b_main: list[int],
    hand_a: list[int],
    hand_b: list[int],
    who_first: str,
) -> PendingQuery:
    hash_a = canonical_deck_hash(deck_a_main)
    hash_b = canonical_deck_hash(deck_b_main)
    qid = _query_id(hash_a, hash_b, hand_a, hand_b, who_first)
    return PendingQuery(
        query_id=qid,
        deck_a_id=deck_a_id,
        deck_b_id=deck_b_id,
        deck_a_hash=hash_a,
        deck_b_hash=hash_b,
        deck_a_main=list(deck_a_main),
        deck_b_main=list(deck_b_main),
        deck_a_archetype=deck_a_archetype,
        deck_b_archetype=deck_b_archetype,
        hand_a=list(hand_a),
        hand_b=list(hand_b),
        who_first=who_first,
    )
