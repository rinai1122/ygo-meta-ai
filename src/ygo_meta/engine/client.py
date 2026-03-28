"""
Async HTTP client for the ygoinf inference server.

Usage:
    async with AsyncYgoInfClient() as client:
        duel = await client.create_duel()
        response = await client.predict(duel.duelId, request, index=0)
        await client.delete_duel(duel.duelId)
"""

from __future__ import annotations

import os

import httpx

from ygo_meta.engine.types import (
    DuelCreateResponse,
    DuelPredictRequest,
    DuelPredictResponse,
    Input,
)


class AsyncYgoInfClient:
    def __init__(self, base_url: str | None = None) -> None:
        port = os.environ.get("YGOAGENT_PORT", "3000")
        self._base_url = base_url or f"http://127.0.0.1:{port}"
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "AsyncYgoInfClient":
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=30.0)
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Client not started. Use as async context manager.")
        return self._client

    async def health(self) -> bool:
        try:
            r = await self.client.get("/")
            return r.status_code == 200
        except httpx.ConnectError:
            return False

    async def create_duel(self) -> DuelCreateResponse:
        r = await self.client.post("/v0/duels")
        r.raise_for_status()
        return DuelCreateResponse.model_validate(r.json())

    async def predict(
        self,
        duel_id: str,
        input_: Input,
        prev_action_idx: int,
        index: int,
    ) -> DuelPredictResponse:
        body = DuelPredictRequest(
            input=input_,
            prev_action_idx=prev_action_idx,
            index=index,
        )
        r = await self.client.post(
            f"/v0/duels/{duel_id}/predict",
            content=body.model_dump_json(by_alias=True),
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()
        return DuelPredictResponse.model_validate(r.json())

    async def delete_duel(self, duel_id: str) -> None:
        r = await self.client.delete(f"/v0/duels/{duel_id}")
        r.raise_for_status()
