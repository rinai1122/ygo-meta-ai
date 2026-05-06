from __future__ import annotations

import numpy as np

from ygo_meta.simulation.obs_sanitizer import sanitize_rl_obs


def test_sanitize_rl_obs_maps_out_of_embedding_card_ids_to_unknown() -> None:
    obs = {
        "cards_": np.array([[[0, 7, 1], [52, 233, 2]]], dtype=np.uint8),
        "actions_": np.array([[[0, 52, 234, 1], [0, 0, 8, 2]]], dtype=np.uint8),
        "h_actions_": np.array([[[0, 52, 235, 1], [0, 0, 9, 2]]], dtype=np.uint8),
        "mask_": np.array([[True, True]], dtype=bool),
    }

    sanitized = sanitize_rl_obs(obs, max_card_id=13_537)

    assert sanitized["cards_"][0, 0, :2].tolist() == [0, 7]
    assert sanitized["cards_"][0, 1, :2].tolist() == [0, 0]
    assert sanitized["actions_"][0, 0, 1:3].tolist() == [0, 0]
    assert sanitized["actions_"][0, 1, 1:3].tolist() == [0, 8]
    assert sanitized["h_actions_"][0, 0, 1:3].tolist() == [0, 0]
    assert sanitized["h_actions_"][0, 1, 1:3].tolist() == [0, 9]
    assert sanitized["mask_"] is None
