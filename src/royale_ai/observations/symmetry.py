"""Seat-symmetry normalization of RoyaleGym observations (MASTER_PLAN §6.2 invariance rule).

KNOWN ENGINE ISSUE KB-001 (royalesim 0.1.55, royalegym 0.1.31). The king towers' 4x4 tile
footprints (own frame: x 7-10, y 1-4 for one's own king, y 27-30 for the enemy's) are not
handled symmetrically under the 180-degree seat rotation. Three symptoms, all confined to
those boxes and measured in tests/test_seat_symmetry.py + tests/test_action_contract.py:

  a. the engine accepts troops IN PLACE on 7 edge tiles of the ENEMY king box (masked out by
     RoyaleGym, so unreachable for any mask-respecting policy);
  b. the action mask for TROOP slots on one's OWN king box edge differs between seats (the
     engine relocates those taps off the tower, asymmetrically);
  c. spatial channel ``enemy_troop_zone`` differs between seats on 10 cells of the enemy box.

Left alone, (b) and (c) let a policy infer which seat it plays from static cells. This
adapter removes them: troop taps onto one's own king box are masked for BOTH seats (they are
relocated anyway, so no placement is lost: the adjacent tiles remain), and the enemy-box cells
of ``enemy_troop_zone`` are set to 0 (the box interior's value) for both seats.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from royale_ai.adapters.action_schema import TILES_18X32_V1 as V1

OWN_KING_BOX = (slice(1, 5), slice(7, 11))  # (rows y, cols x), own frame
ENEMY_KING_BOX = (slice(27, 31), slice(7, 11))
ENEMY_TROOP_ZONE_CHANNEL = "enemy_troop_zone"

_OWN_BOX_TILES = [(tx, ty) for ty in range(1, 5) for tx in range(7, 11)]


def own_king_box_actions(slots: Iterable[int]) -> np.ndarray:
    return np.array([V1.encode(s, tx, ty) for s in slots for tx, ty in _OWN_BOX_TILES], dtype=np.int64)


def symmetrize(obs: dict[str, np.ndarray], troop_slots: Iterable[int], zone_channel: int) -> dict[str, np.ndarray]:
    """A copy of one seat's observation with the KB-001 asymmetries removed.

    ``troop_slots``: hand slots holding a TROOP card (from the seat's own, public hand).
    ``zone_channel``: index of ``enemy_troop_zone`` in ``obs_builder.channel_names()``.
    """
    out = {k: v.copy() for k, v in obs.items()}
    slots = list(troop_slots)
    if slots:
        out["action_mask"][own_king_box_actions(slots)] = 0
        if "mask_planes" in out:
            for s in slots:
                out["mask_planes"][s][OWN_KING_BOX] = 0
    out["spatial"][zone_channel][ENEMY_KING_BOX] = 0
    assert out["action_mask"][0] == 1, "WAIT must stay legal"
    return out


class SymmetricObservations:
    """Binds ``symmetrize`` to a ClashParallelEnv: call ``seat_obs(obs, seat)`` after reset/step."""

    def __init__(self, env) -> None:
        self.env = env
        self.zone_channel = list(env.obs_builder.channel_names()).index(ENEMY_TROOP_ZONE_CHANNEL)
        self._kinds = {c.card_id: c.card_kind for c in env.engine.cards()}

    def troop_slots(self, team: int) -> list[int]:
        hand = self.env.engine.state().players[team].hand
        return [i for i, cid in enumerate(hand) if self._kinds.get(cid) == "TROOP"]

    def seat_obs(self, obs: dict[str, dict[str, np.ndarray]], seat: str) -> dict[str, np.ndarray]:
        team = 0 if seat == "blue" else 1
        return symmetrize(obs[seat], self.troop_slots(team), self.zone_channel)
