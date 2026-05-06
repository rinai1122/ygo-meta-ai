"""Observation helpers for checkpoint-compatible RL inference."""

from __future__ import annotations

import numpy as np


def zero_unknown_card_ids(arr: np.ndarray, hi_col: int, max_card_id: int | None) -> np.ndarray:
    """Zero two-byte card IDs greater than the checkpoint embedding range."""
    if max_card_id is None:
        return arr
    out = np.array(arr, copy=True)
    ids = out[..., hi_col].astype(np.int32) * 256 + out[..., hi_col + 1].astype(np.int32)
    unknown = ids > int(max_card_id)
    out[..., hi_col] = np.where(unknown, 0, out[..., hi_col])
    out[..., hi_col + 1] = np.where(unknown, 0, out[..., hi_col + 1])
    return out


def sanitize_rl_obs(obs: dict, max_card_id: int | None) -> dict:
    """Map card IDs outside the checkpoint embedding table to ID 0.

    The engine may know newer cards than an older checkpoint embedding table.
    Feeding those IDs directly into the policy aliases them to invalid or
    unrelated embeddings; ID 0 is the model's explicit unknown-card bucket.
    """
    obs_rl = {k: (v if k != "mask_" else None) for k, v in obs.items()}
    if max_card_id is None:
        return obs_rl
    if obs_rl.get("cards_") is not None:
        obs_rl["cards_"] = zero_unknown_card_ids(obs_rl["cards_"], 0, max_card_id)
    if obs_rl.get("actions_") is not None:
        obs_rl["actions_"] = zero_unknown_card_ids(obs_rl["actions_"], 1, max_card_id)
    if obs_rl.get("h_actions_") is not None:
        obs_rl["h_actions_"] = zero_unknown_card_ids(obs_rl["h_actions_"], 1, max_card_id)
    return obs_rl
